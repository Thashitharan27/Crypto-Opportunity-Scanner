"""Run the ATR-checkpoint extension against a saved baseline configuration."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.gui.config_logic import build_backtest_config
from crypto_strategy_lab.loader import load_backtest_data
from crypto_strategy_lab.statistics import summarize


def main() -> None:
    run_dir = Path(sys.argv[1])
    raw = json.loads((run_dir / "config.json").read_text())
    config = build_backtest_config(raw, require_paths=True)
    config = replace(
        config,
        enable_atr_checkpoint_tp_extension=True,
        atr_checkpoint_di_spread_minimum=30.0,
        atr_checkpoint_bb_width_minimum=0.03,
        atr_checkpoint_profit_lock_start=3.0,
        atr_checkpoint_profit_lock_distance=1.0,
        enable_trade_telemetry=False,
        save_full_telemetry_csv=False,
        save_trade_journey_summary=False,
        save_trade_journey_charts=False,
        enable_indicator_lifecycle_analysis=False,
        create_lifecycle_charts=False,
        trading_start_date=sys.argv[2] if len(sys.argv) > 2 else config.trading_start_date,
    )
    data, intrabar = load_backtest_data(config)
    engine = BacktestEngine(data, config, intrabar)
    trades = engine.run()
    summary = summarize(trades)
    passes = failures = locks = 0
    for side in ("long", "short"):
        passes += int(trades[f"{side}_atr_checkpoint_pass_count"].fillna(0).sum())
        failures += int(trades[f"{side}_atr_checkpoint_fail_count"].fillna(0).sum())
        locks += int(trades[f"{side}_atr_checkpoint_profit_lock_r"].notna().sum())
    print(json.dumps({
        "asset": run_dir.name[:3],
        "trades": summary["total_trades"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "average_net_r": summary["average_net_r"],
        "total_net_r": summary["total_net_r"],
        "total_return_percentage": summary["total_return_percentage"],
        "maximum_drawdown_percentage": summary["maximum_drawdown_percentage"],
        "checkpoint_passes": passes,
        "checkpoint_failures": failures,
        "trades_with_profit_lock": locks,
        "outcomes": summary["outcomes"],
    }, allow_nan=True))


if __name__ == "__main__":
    main()
