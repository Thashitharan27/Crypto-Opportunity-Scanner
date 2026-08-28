"""Command-line entry point for the current Strategy Profile backtester."""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import pandas as pd

from crypto_strategy_lab.config import (
    BacktestConfig,
    DailyEntryMissedPolicy,
    EntryMode,
    IntrabarMissingPolicy,
    RiskMode,
    TiePolicy,
)
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.gui.config_logic import build_backtest_config, load_config_json
from crypto_strategy_lab.lifecycle import export_lifecycle_reports
from crypto_strategy_lab.loader import load_backtest_data
from crypto_strategy_lab.output_manager import (
    create_run_dir,
    periodic_results,
    update_latest,
    write_config,
    write_run_info,
    write_trade_column_metadata,
)
from crypto_strategy_lab.plots import save_plots
from crypto_strategy_lab.report_workbooks import (
    build_backtest_workbook,
    build_indicator_workbook,
    build_performance_breakdowns,
)
from crypto_strategy_lab.statistics import (
    adx_analysis,
    bb_width_analysis,
    di_pressure_analysis,
    di_spread_analysis,
    mean_reversion_analysis,
    equity_curve,
    summarize,
)
from crypto_strategy_lab.support_resistance_analysis import generate_sr_analysis_reports
from crypto_strategy_lab.telemetry import (
    add_journey_columns,
    partial_take_profit_analysis,
    save_journey_charts,
    stop_loss_journey_analysis,
    trade_journey_analysis,
    winner_loser_journey_analysis,
)


def enum_value(enum_cls):
    def parse(value: str):
        normalized = value.upper()
        try:
            return enum_cls(value)
        except ValueError:
            try:
                return enum_cls(normalized)
            except ValueError:
                return enum_cls[normalized]
    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crypto Strategy Lab backtester using the current v2 Strategy Profile configuration contract"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Current v2 GUI configuration JSON. Legacy configuration files are rejected.",
    )
    parser.add_argument("--input", type=Path, help="Strategy CSV path")
    parser.add_argument("--intrabar-input", type=Path, dest="intrabar_csv")
    intrabar = parser.add_mutually_exclusive_group()
    intrabar.add_argument("--use-intrabar", action="store_true", dest="use_intrabar_data")
    intrabar.add_argument("--no-intrabar", action="store_false", dest="use_intrabar_data")
    parser.set_defaults(use_intrabar_data=None)
    parser.add_argument("--strategy-timeframe", type=int, dest="strategy_timeframe_minutes")
    parser.add_argument("--intrabar-timeframe", type=int, dest="intrabar_timeframe_minutes")
    parser.add_argument("--data-start", dest="data_start_date")
    parser.add_argument("--trading-start", dest="trading_start_date")
    parser.add_argument("--trading-end", dest="trading_end_date")
    parser.add_argument("--max-leverage-per-leg", type=float, dest="max_effective_leverage_per_leg")
    parser.add_argument("--max-combined-leverage", type=float, dest="max_combined_effective_leverage")
    parser.add_argument("--intrabar-missing-policy", type=enum_value(IntrabarMissingPolicy), choices=list(IntrabarMissingPolicy))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--risk-mode", type=enum_value(RiskMode), choices=list(RiskMode))
    parser.add_argument("--fixed-r", type=float)
    parser.add_argument("--percent-r", type=float)
    parser.add_argument("--atr-period", type=int)
    parser.add_argument("--atr-multiplier", type=float)
    parser.add_argument("--risk-per-leg", type=float)
    parser.add_argument("--initial-equity", type=float)
    parser.add_argument("--maker-fee", type=float)
    parser.add_argument("--taker-fee", type=float)
    parser.add_argument("--slippage", type=float)
    parser.add_argument("--tie-policy", type=enum_value(TiePolicy), choices=list(TiePolicy))
    parser.add_argument("--entry-mode", type=enum_value(EntryMode), choices=list(EntryMode))
    parser.add_argument("--entry-interval", type=int)
    parser.add_argument("--enable-daily-entry-schedule", action="store_true", default=None)
    parser.add_argument("--daily-entry-time")
    parser.add_argument("--daily-entry-timezone")
    parser.add_argument("--daily-entry-missed-policy", type=enum_value(DailyEntryMissedPolicy), choices=list(DailyEntryMissedPolicy))
    parser.add_argument("--max-active-pairs", type=int)
    parser.add_argument("--zero-cost-comparison", action="store_true", default=None)
    parser.add_argument("--disable-trade-telemetry", action="store_false", default=None, dest="enable_trade_telemetry")
    parser.add_argument("--no-full-telemetry-csv", action="store_false", default=None, dest="save_full_telemetry_csv")
    parser.add_argument("--no-trade-journey-summary", action="store_false", default=None, dest="save_trade_journey_summary")
    parser.add_argument("--no-trade-journey-charts", action="store_false", default=None, dest="save_trade_journey_charts")
    parser.add_argument("--telemetry-interval-minutes", type=int)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> BacktestConfig:
    if args.config is not None:
        config = build_backtest_config(load_config_json(args.config), require_paths=True)
    else:
        config = BacktestConfig()

    overrides = {}
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        overrides["input_csv" if key == "input" else key] = value
    return replace(config, **overrides) if overrides else config


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    data, intrabar = load_backtest_data(config)
    engine = BacktestEngine(data, config, intrabar)
    trades = engine.run()
    telemetry = engine.telemetry_frame()
    if config.enable_trade_telemetry:
        trades = add_journey_columns(trades, telemetry)

    output_root = config.output_dir
    run_dir = create_run_dir(config)
    write_config(config, run_dir)
    output_failures = []

    def run_output_step(label: str, action):
        print(f"{label}...")
        try:
            return action()
        except Exception as exc:  # output exports should fail independently.
            tb = traceback.format_exc()
            output_failures.append({"step": label, "error": str(exc), "traceback": tb})
            print(f"ERROR while {label}: {exc}")
            print(tb)
            return None

    run_output_step("Saving trade_list.csv", lambda: trades.to_csv(run_dir / "trade_list.csv", index=False))
    run_output_step(
        "Building partial_take_profit_analysis",
        lambda: partial_take_profit_analysis(trades).to_csv(run_dir / "partial_take_profit_analysis.csv", index=False),
    )
    if config.enable_trade_telemetry:
        if config.save_full_telemetry_csv:
            run_output_step("Saving telemetry", lambda: telemetry.to_csv(run_dir / "trade_telemetry.csv", index=False))
        if config.save_trade_journey_summary:
            run_output_step("Building trade_journey_analysis", lambda: trade_journey_analysis(trades).to_csv(run_dir / "trade_journey_analysis.csv", index=False))
            run_output_step("Building winner_loser_journey_analysis", lambda: winner_loser_journey_analysis(trades).to_csv(run_dir / "winner_loser_journey_analysis.csv", index=False))
            run_output_step("Building stop_loss_journey_analysis", lambda: stop_loss_journey_analysis(trades, telemetry).to_csv(run_dir / "stop_loss_journey_analysis.csv", index=False))
        if config.enable_indicator_lifecycle_analysis:
            run_output_step(
                "Building indicator lifecycle reports",
                lambda: export_lifecycle_reports(
                    trades,
                    telemetry,
                    run_dir,
                    phases=config.lifecycle_phases,
                    checkpoints=config.lifecycle_early_checkpoints,
                    minimum_sample=config.lifecycle_minimum_bucket_sample,
                    charts=config.create_lifecycle_charts,
                    flat_threshold_pct=config.lifecycle_flat_pattern_threshold_pct,
                ),
            )

    run_output_step("Saving skipped_signals.csv", lambda: pd.DataFrame(trades.attrs.get("skipped_signals", [])).to_csv(run_dir / "skipped_signals.csv", index=False))
    run_output_step("Saving skipped_daily_entries.csv", lambda: pd.DataFrame(trades.attrs.get("skipped_daily_entries", [])).to_csv(run_dir / "skipped_daily_entries.csv", index=False))
    indicator_tables = {
        "ADX": adx_analysis(trades),
        "BB Width": bb_width_analysis(trades),
        "DI Spread": di_spread_analysis(trades),
        "DI Pressure": di_pressure_analysis(trades),
        "Mean Reversion": mean_reversion_analysis(trades),
    }
    run_output_step("Saving indicator analysis workbook", lambda: build_indicator_workbook(indicator_tables, run_dir))
    run_output_step("Saving DI mean reversion analysis", lambda: mean_reversion_analysis(trades).to_csv(run_dir / "di_mean_reversion_analysis.csv", index=False))
    run_output_step("Building support/resistance analysis", lambda: generate_sr_analysis_reports(trades, run_dir))
    run_output_step("Saving trade column metadata", lambda: write_trade_column_metadata(run_dir))

    equity = equity_curve(trades, config.initial_equity)
    run_output_step("Saving equity_curve.csv", lambda: equity.to_csv(run_dir / "equity_curve.csv", index=False))
    monthly, yearly = periodic_results(trades, "ME"), periodic_results(trades, "YE")
    chart_warnings = run_output_step("Saving charts", lambda: save_plots(trades, equity, run_dir / "charts")) or []
    if config.enable_trade_telemetry and config.save_trade_journey_charts:
        chart_warnings.extend(run_output_step("Saving journey charts", lambda: save_journey_charts(trades, telemetry, run_dir / "charts")) or [])
    for warning in chart_warnings:
        print(f"WARNING: {warning}")

    summary = summarize(trades, config.initial_equity)
    summary.update(
        {
            "use_intrabar_data": config.use_intrabar_data,
            "intrabar_csv": str(config.intrabar_csv) if config.intrabar_csv else None,
            "strategy_timeframe": config.strategy_timeframe_minutes,
            "intrabar_timeframe": config.intrabar_timeframe_minutes,
            "atr_timeframe": config.strategy_timeframe_minutes,
            "atr_period": config.atr_period,
            "atr_multiplier": config.atr_multiplier,
            "indicator_data_start": str(data.timestamp.min()),
            "data_start": config.data_start_date,
            "trading_start": config.trading_start_date,
            "trading_end": config.trading_end_date,
            "warmup_candles": engine.warmup_candle_count,
            "first_valid_atr_timestamp": str(engine.first_valid_atr_timestamp),
            "strategy_profile_run_mode": config.strategy_profile_run_mode,
        }
    )

    if config.zero_cost_comparison:
        ideal_cfg = replace(config, maker_fee=0, taker_fee=0, slippage=0, zero_cost_comparison=False)
        ideal_trades = BacktestEngine(data, ideal_cfg, intrabar).run()
        ideal = summarize(ideal_trades, ideal_cfg.initial_equity)
        keys = ("win_rate", "total_net_r", "ending_equity", "total_return_percentage", "profit_factor", "maximum_drawdown")
        summary["zero_cost_comparison"] = {
            "actual": {key: summary.get(key) for key in keys},
            "zero_cost": {key: ideal.get(key) for key in keys},
        }

    if output_failures:
        summary["failed_output_reports"] = output_failures
        print("WARNING: Some output reports failed: " + ", ".join(f["step"] for f in output_failures))
    if config.use_intrabar_data and summary.get("intrabar_exit_count") == 0:
        print("WARNING: use_intrabar_data=True but 1M_INTRABAR exit count is 0. Check intrabar path, overlap, and timestamp alignment.")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    market_regime, direction_regime = build_performance_breakdowns(trades)
    run_output_step(
        "Saving backtest report workbook",
        lambda: build_backtest_workbook(summary, config, run_dir, monthly, yearly, market_regime, direction_regime),
    )
    write_run_info(config, summary, run_dir)
    (run_dir / "log.txt").write_text("Command-line backtest completed.\n")
    update_latest(output_root, run_dir)
    summary["output_dir"] = str(run_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
