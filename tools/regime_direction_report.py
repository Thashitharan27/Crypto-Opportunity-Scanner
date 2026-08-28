"""Generate regime/direction reports from an existing backtest trade_list.csv.

Usage:
    python tools/regime_direction_report.py path/to/run_folder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crypto_strategy_lab.regime_direction_report import (
    regime_direction_di_summary,
    regime_direction_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime/direction summary CSVs")
    parser.add_argument("run_dir", type=Path, help="Backtest output folder containing trade_list.csv")
    parser.add_argument("--di-threshold", type=float, default=30.0)
    args = parser.parse_args()

    trade_path = args.run_dir / "trade_list.csv"
    if not trade_path.is_file():
        raise FileNotFoundError(f"Trade list not found: {trade_path}")

    trades = pd.read_csv(trade_path)
    summary_path = args.run_dir / "regime_direction_summary.csv"
    di_path = args.run_dir / "regime_direction_di_summary.csv"

    regime_direction_summary(trades).to_csv(summary_path, index=False)
    regime_direction_di_summary(trades, args.di_threshold).to_csv(di_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {di_path}")


if __name__ == "__main__":
    main()
