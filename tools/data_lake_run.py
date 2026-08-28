"""Thin CLI adapter for the authoritative ResearchRunner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_strategy_lab.bayesian_sampling_reporting import BayesianSamplingCsvManifestReporter
from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data_lake_config import load_data_lake_config
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.prepared_cache import PreparedRunCache
from crypto_strategy_lab.research_adapters import NativeSimulator, NativeStrategyPolicy
from crypto_strategy_lab.research_runner import ResearchRunner


def _utc(value):
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    ).to_pydatetime()


def build_parser():
    parser = argparse.ArgumentParser(description="Run Crypto Strategy Lab through ResearchRunner")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional runtime override for ReportingConfig.output_dir",
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = load_data_lake_config(args.config)
    request = DataRequest(
        symbol=args.symbol,
        start=_utc(args.start),
        end=_utc(args.end),
        strategy_interval=f"{config.data.strategy_timeframe_minutes}m",
        intrabar_interval=(
            f"{config.data.intrabar_timeframe_minutes}m"
            if config.data.use_intrabar_data
            else None
        ),
    )
    store = MarketDataStore(args.raw_root, args.cache_root)
    output_root = args.output_dir or Path(config.reporting.output_dir)
    runner = ResearchRunner(
        store,
        production_feature_registry(),
        PreparedRunCache(args.cache_root),
        NativeStrategyPolicy(),
        NativeSimulator(),
        (BayesianSamplingCsvManifestReporter(output_root),),
    )
    result = runner.run(request, config)
    print(
        json.dumps(
            {
                "run_dir": str(result.output_dir.resolve()),
                "trade_rows": len(result.trades),
                "prepared_cache_hit": result.prepared_cache_hit,
                "prepared_cache_key": result.prepared_cache_key,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())