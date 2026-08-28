"""Profile the authoritative composed Data Lake research run."""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data_lake_config import load_data_lake_config
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.prepared_cache import PreparedRunCache
from crypto_strategy_lab.research_adapters import NativeSimulator, NativeStrategyPolicy
from crypto_strategy_lab.research_runner import ResearchRunner


_FINGERPRINT_COLUMNS = (
    "pair_id", "trade_id", "result_type", "side", "strategy_candle_open_time",
    "strategy_entry_time", "entry_time", "exit_time", "long_exit_reason",
    "long_exit_source", "short_exit_reason", "short_exit_source", "pair_net_pnl",
    "pair_net_r", "equity_after_trade",
)


def _utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _trade_fingerprint(trades: pd.DataFrame) -> str:
    columns = [column for column in _FINGERPRINT_COLUMNS if column in trades.columns]
    if not columns:
        payload = f"rows={len(trades)};columns={','.join(map(str, trades.columns))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    normalized = trades.loc[:, columns].copy()
    for column in columns:
        normalized[column] = normalized[column].astype("string").fillna("<NA>")
    return hashlib.sha256(
        normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _profile_rows(profile: cProfile.Profile, sort_key: str, limit: int):
    rows = []
    for (filename, line, function), values in pstats.Stats(profile).stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        rows.append(
            {
                "function": function,
                "file": str(Path(filename)),
                "line": int(line),
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "self_seconds": float(self_seconds),
                "cumulative_seconds": float(cumulative_seconds),
            }
        )
    key = "cumulative_seconds" if sort_key == "cumulative" else "self_seconds"
    rows.sort(key=lambda row: float(row[key]), reverse=True)
    return rows[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Crypto Strategy Lab authoritative ResearchRunner stages"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--intrabar-start")
    parser.add_argument("--include-trade-flow", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top < 1:
        raise SystemExit("--top must be at least 1")

    config = load_data_lake_config(args.config)
    if args.include_trade_flow:
        config = replace(
            config,
            features=replace(config.features, trade_flow_enabled=True),
        )
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
    intrabar_start = _utc(args.intrabar_start) if args.intrabar_start else None
    store = MarketDataStore(args.raw_root, args.cache_root)
    runner = ResearchRunner(
        store,
        production_feature_registry(),
        PreparedRunCache(args.cache_root),
        NativeStrategyPolicy(),
        NativeSimulator(),
        (),
    )

    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    run = runner.run(request, config, intrabar_start=intrabar_start)
    profile.disable()
    elapsed = time.perf_counter() - started

    result = {
        "profile_contract": "research_runner_profile_v2",
        "config_path": str(args.config.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "request": {
            "symbol": request.symbol,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "strategy_interval": request.strategy_interval,
            "intrabar_interval": request.intrabar_interval,
            "intrabar_start": intrabar_start.isoformat() if intrabar_start else None,
        },
        "stage_timings": dict(run.stage_timings),
        "profiled_total_seconds": elapsed,
        "strategy_rows": run.strategy_rows,
        "intrabar_rows": run.intrabar_rows,
        "trade_rows": len(run.trades),
        "prepared_cache_hit": run.prepared_cache_hit,
        "trade_fingerprint": _trade_fingerprint(run.trades),
        "top_by_cumulative": _profile_rows(profile, "cumulative", args.top),
        "top_by_self": _profile_rows(profile, "self", args.top),
    }
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
