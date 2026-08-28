"""Shared-equity portfolio replay for independently configured assets."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.gui.config_logic import build_backtest_config, load_config_json
from crypto_strategy_lab.loader import load_backtest_data


def _component_run(label, config_path, initial_equity, risk_fraction, progress=None):
    values = load_config_json(config_path)
    config = build_backtest_config(values, require_paths=True)
    config = replace(
        config,
        initial_equity=float(initial_equity),
        risk_per_leg=float(risk_fraction),
        enable_trade_telemetry=True,
    )
    data, intrabar = load_backtest_data(config)
    callback = None
    if progress is not None:
        callback = lambda done, total, completed, opened: progress(
            label, done, total, completed, opened
        )
    engine = BacktestEngine(data, config, intrabar, progress_callback=callback, progress_interval=100)
    trades = engine.run()
    trades = trades.copy()
    trades["asset"] = label
    trades["portfolio_trade_key"] = label + "-" + trades["pair_id"].astype(str)
    telemetry = engine.telemetry_frame().copy()
    if not telemetry.empty:
        telemetry["asset"] = label
        telemetry["portfolio_trade_key"] = label + "-" + telemetry["pair_id"].astype(str)
    return config, trades, telemetry


def _shared_equity_replay(trades, initial_equity, risk_fraction, maximum_total_risk=0.05):
    events = []
    for row in trades.itertuples():
        events.append((row.entry_time, 0, "entry", row.portfolio_trade_key, row.asset, row.pair_net_r))
        events.append((row.exit_time, 1, "exit", row.portfolio_trade_key, row.asset, row.pair_net_r))
    events.sort(key=lambda item: (item[0], item[1]))

    equity = float(initial_equity)
    peak = equity
    maximum_drawdown = 0.0
    active = {}
    assignments = {}
    realized_rows = []
    maximum_open_trades = 0
    maximum_open_risk_fraction = 0.0
    blocked_entries = 0

    for timestamp, _, kind, key, asset, result_multiple in events:
        if kind == "entry":
            risk_amount = equity * risk_fraction
            open_risk = sum(active.values())
            risk_limit = equity * maximum_total_risk
            if open_risk + risk_amount > risk_limit + 1e-12:
                assignments[key] = {
                    "portfolio_accepted": False,
                    "portfolio_block_reason": "MAXIMUM_TOTAL_PORTFOLIO_RISK",
                }
                blocked_entries += 1
                continue
            active[key] = risk_amount
            assignments[key] = {
                "portfolio_accepted": True,
                "portfolio_block_reason": "",
                "portfolio_entry_equity": equity,
                "portfolio_risk_amount": risk_amount,
            }
            maximum_open_trades = max(maximum_open_trades, len(active))
            maximum_open_risk_fraction = max(maximum_open_risk_fraction, sum(active.values()) / equity)
            continue
        if key not in active:
            continue
        risk_amount = active.pop(key)
        pnl = risk_amount * float(result_multiple)
        equity += pnl
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        maximum_drawdown = min(maximum_drawdown, drawdown)
        realized_rows.append(
            {
                "timestamp": timestamp,
                "asset": asset,
                "portfolio_trade_key": key,
                "portfolio_result_multiple": result_multiple,
                "portfolio_pnl": pnl,
                "portfolio_equity": equity,
                "portfolio_drawdown": drawdown,
            }
        )
    return pd.DataFrame(realized_rows), assignments, maximum_open_trades, maximum_drawdown, maximum_open_risk_fraction, blocked_entries


def _mark_to_market_curve(trades, telemetries, realized, assignments, initial_equity):
    start = trades["entry_time"].min().floor("h")
    end = trades["exit_time"].max().ceil("h")
    grid = pd.date_range(start, end, freq="1h")
    closed = realized.set_index("timestamp")["portfolio_equity"]
    closed = closed[~closed.index.duplicated(keep="last")].sort_index()
    closed = closed.reindex(grid, method="ffill").fillna(float(initial_equity))
    unrealized = pd.Series(0.0, index=grid)

    telemetry = pd.concat(telemetries, ignore_index=True) if telemetries else pd.DataFrame()
    if not telemetry.empty:
        telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True).dt.floor("h")
        indexed_trades = trades.set_index("portfolio_trade_key")
        for key, journey in telemetry.groupby("portfolio_trade_key"):
            if key not in assignments or key not in indexed_trades.index:
                continue
            trade = indexed_trades.loc[key]
            standalone_risk = np.nanmax(
                [trade.get("long_risk_amount", np.nan), trade.get("short_risk_amount", np.nan)]
            )
            if not np.isfinite(standalone_risk) or standalone_risk <= 0:
                continue
            values = (
                journey.groupby("timestamp")["pair_unrealized_pnl"].last().sort_index()
                / standalone_risk
                * assignments[key]["portfolio_risk_amount"]
            )
            active_grid = grid[
                (grid >= trade["entry_time"].floor("h"))
                & (grid < trade["exit_time"].ceil("h"))
            ]
            if len(active_grid):
                unrealized.loc[active_grid] += values.reindex(active_grid, method="ffill").fillna(0).to_numpy()

    frame = pd.DataFrame(
        {
            "timestamp": grid,
            "closed_equity": closed.to_numpy(),
            "open_trade_unrealized_pnl": unrealized.to_numpy(),
        }
    )
    frame["mark_to_market_equity"] = frame["closed_equity"] + frame["open_trade_unrealized_pnl"]
    frame["mark_to_market_peak"] = frame["mark_to_market_equity"].cummax()
    frame["mark_to_market_drawdown"] = (
        frame["mark_to_market_equity"] / frame["mark_to_market_peak"] - 1.0
    )
    return frame


def _periodic(realized, frequency, initial_equity):
    frame = realized.copy().set_index("timestamp")
    # pandas < 2.2 uses M/Y, while newer releases require ME/YE.
    alternatives = {
        "ME": ("ME", "M"), "M": ("ME", "M"),
        "YE": ("YE", "Y"), "Y": ("YE", "Y"),
    }.get(frequency, (frequency,))
    last_error = None
    for compatible_frequency in alternatives:
        try:
            grouped = frame.resample(compatible_frequency).agg(
                trade_count=("portfolio_pnl", "size"),
                net_pnl=("portfolio_pnl", "sum"),
            )
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise last_error
    grouped["ending_equity"] = float(initial_equity) + grouped["net_pnl"].cumsum()
    grouped["starting_equity"] = grouped["ending_equity"].shift(fill_value=float(initial_equity))
    grouped["return_percentage"] = grouped["net_pnl"] / grouped["starting_equity"] * 100.0
    return grouped.reset_index()


def run_portfolio(
    components,
    output_root="output",
    initial_equity=1000.0,
    risk_per_asset=0.01,
    maximum_total_risk=0.05,
    progress=None,
):
    """Run component configs and replay their trades against one shared account."""
    if len(components) < 2:
        raise ValueError("Portfolio mode requires at least two component configurations.")
    if not 0 < float(risk_per_asset) <= float(maximum_total_risk) < 1:
        raise ValueError("Portfolio risk settings must satisfy: 0 < risk per asset <= maximum total risk < 100%.")
    configs = {}
    trade_frames = []
    telemetry_frames = []
    for label, path in components:
        if progress:
            progress(label, 0, 0, 0, 0)
        config, trades, telemetry = _component_run(
            str(label).upper(), path, initial_equity, risk_per_asset, progress
        )
        configs[str(label).upper()] = config
        trade_frames.append(trades)
        telemetry_frames.append(telemetry)

    trades = pd.concat(trade_frames, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades = trades.sort_values(["entry_time", "asset", "pair_id"]).reset_index(drop=True)
    realized, assignments, max_open, closed_dd, max_open_risk, blocked_entries = _shared_equity_replay(
        trades, initial_equity, risk_per_asset, maximum_total_risk
    )
    for key, values in assignments.items():
        mask = trades["portfolio_trade_key"].eq(key)
        for field, value in values.items():
            trades.loc[mask, field] = value
    pnl_map = realized.set_index("portfolio_trade_key")["portfolio_pnl"]
    equity_map = realized.set_index("portfolio_trade_key")["portfolio_equity"]
    trades["portfolio_pnl"] = trades["portfolio_trade_key"].map(pnl_map)
    trades["portfolio_equity_after_exit"] = trades["portfolio_trade_key"].map(equity_map)
    trades["portfolio_accepted"] = trades["portfolio_trade_key"].map(lambda key: bool(assignments.get(key,{}).get("portfolio_accepted",False)))
    trades["portfolio_block_reason"] = trades["portfolio_trade_key"].map(lambda key: assignments.get(key,{}).get("portfolio_block_reason",""))
    accepted_trades=trades.loc[trades["portfolio_accepted"]].copy()

    mtm = _mark_to_market_curve(
        accepted_trades, telemetry_frames, realized, assignments, initial_equity
    )
    ending_equity = float(realized.iloc[-1]["portfolio_equity"])
    mtm_dd = float(mtm["mark_to_market_drawdown"].min())
    monthly = _periodic(realized, "ME", initial_equity)
    yearly = _periodic(realized, "YE", initial_equity)
    negative_months = int((monthly["net_pnl"] < 0).sum())
    negative = monthly["net_pnl"].lt(0).astype(int)
    groups = negative.ne(negative.shift()).cumsum()
    maximum_losing_month_streak = int(negative.groupby(groups).sum().max())

    summary = {
        "portfolio_assets": [label for label, _ in components],
        "initial_equity": float(initial_equity),
        "risk_per_asset": float(risk_per_asset),
        "maximum_total_portfolio_risk": float(maximum_total_risk),
        "maximum_observed_open_risk": float(max_open_risk),
        "maximum_open_trades": int(max_open),
        "candidate_trades": int(len(trades)),
        "trades_blocked_by_portfolio_risk": int(blocked_entries),
        "total_trades": int(len(accepted_trades)),
        "ending_equity": ending_equity,
        "total_return_percentage": (ending_equity / initial_equity - 1.0) * 100.0,
        "closed_equity_maximum_drawdown_percentage": float(closed_dd * 100.0),
        "mark_to_market_maximum_drawdown_percentage": float(mtm_dd * 100.0),
        "positive_months": int((monthly["net_pnl"] > 0).sum()),
        "negative_months": negative_months,
        "maximum_losing_month_streak": maximum_losing_month_streak,
        "worst_month_return_percentage": float(monthly["return_percentage"].min()),
    }

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    labels = "_".join(str(label).upper() for label, _ in components)
    run_dir = Path(output_root) / f"PORTFOLIO_{labels}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    trades.to_csv(run_dir / "portfolio_trade_list.csv", index=False)
    realized.to_csv(run_dir / "portfolio_realized_equity.csv", index=False)
    mtm.to_csv(run_dir / "portfolio_mark_to_market_equity.csv", index=False)
    monthly.to_csv(run_dir / "portfolio_monthly_results.csv", index=False)
    yearly.to_csv(run_dir / "portfolio_yearly_results.csv", index=False)
    component_rows = []
    for label, path in components:
        asset = str(label).upper()
        asset_realized = realized[realized["asset"].eq(asset)]
        component_rows.append(
            {
                "asset": asset,
                "config_path": str(path),
                "trades": int(len(asset_realized)),
                "portfolio_pnl_contribution": float(asset_realized["portfolio_pnl"].sum()),
            }
        )
        (run_dir / f"{asset.lower()}_config.json").write_text(
            json.dumps(load_config_json(path), indent=2, default=str)
        )
    pd.DataFrame(component_rows).to_csv(run_dir / "portfolio_components.csv", index=False)
    (run_dir / "portfolio_summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "portfolio_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
    )
    return summary, trades, mtm, run_dir
