"""Benchmark the authoritative composed Data Lake execution path on local data."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Mapping

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
    "pair_id",
    "trade_id",
    "result_type",
    "side",
    "strategy_candle_open_time",
    "strategy_entry_time",
    "entry_time",
    "exit_time",
    "long_exit_reason",
    "long_exit_source",
    "short_exit_reason",
    "short_exit_source",
    "pair_net_pnl",
    "pair_net_r",
    "equity_after_trade",
)


def _utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Crypto Strategy Lab Data Lake v2 preparation and simulation"
    )
    parser.add_argument("--config", required=True, type=Path, help="Current Data Lake strategy JSON")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="Inclusive UTC data start")
    parser.add_argument("--end", required=True, help="Exclusive UTC data end")
    parser.add_argument("--intrabar-start", help="Optional later inclusive UTC start for intrabar data")
    parser.add_argument("--iterations", type=int, default=2, help="Number of identical runs (default: 2)")
    parser.add_argument(
        "--skip-catalog-refresh",
        action="store_true",
        help="Use the existing Data Lake catalog instead of refreshing it once before timing runs",
    )
    parser.add_argument(
        "--include-trade-flow",
        action="store_true",
        help="Include aggregate-cache-backed trade-flow research in preparation timings",
    )
    parser.add_argument(
        "--include-order-book",
        action="store_true",
        help="Include compact-cache-backed order-book research in preparation timings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The result is always printed to stdout.",
    )
    return parser


def _trade_fingerprint(trades: pd.DataFrame) -> str:
    columns = [column for column in _FINGERPRINT_COLUMNS if column in trades.columns]
    if not columns:
        payload = f"rows={len(trades)};columns={','.join(map(str, trades.columns))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    normalized = trades.loc[:, columns].copy()
    for column in columns:
        normalized[column] = normalized[column].astype("string").fillna("<NA>")
    payload = normalized.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_hits(metadata: dict[str, dict]) -> dict[str, object]:
    core_names = {
        "core_directional",
        "production_market_context",
        "support_resistance",
        "state_transition_daily",
    }
    return {
        "core_directional": bool(metadata["core_directional"]["cache_hit"]),
        "production_market_context": bool(
            metadata["production_market_context"]["cache_hit"]
        ),
        "support_resistance": (
            bool(metadata["support_resistance"]["cache_hit"])
            if "support_resistance" in metadata
            else None
        ),
        "state_transition_daily": (
            bool(metadata["state_transition_daily"]["cache_hit"])
            if "state_transition_daily" in metadata
            else None
        ),
        "research": {
            name: bool(values["cache_hit"])
            for name, values in sorted(metadata.items())
            if name not in core_names
        },
    }


def _trade_aggregate_cache(metadata: Mapping[str, dict]) -> dict[str, object] | None:
    trade_flow = metadata.get("trade_flow_context")
    if not trade_flow:
        return None
    value = trade_flow.get("trade_aggregate_cache")
    return dict(value) if isinstance(value, dict) else None


def _book_snapshot_events(store) -> dict[str, dict[str, int]]:
    raw = getattr(store, "order_book_snapshot_cache_events", {})
    return {
        str(dataset): {
            "partitions_built": int(values.get("partitions_built", 0) or 0),
            "partitions_reused": int(values.get("partitions_reused", 0) or 0),
        }
        for dataset, values in raw.items()
    }


def _book_snapshot_delta(before, after) -> dict[str, dict[str, object]] | None:
    result = {}
    for dataset in sorted(set(before) | set(after)):
        left = before.get(dataset, {})
        right = after.get(dataset, {})
        built = int(right.get("partitions_built", 0)) - int(
            left.get("partitions_built", 0)
        )
        reused = int(right.get("partitions_reused", 0)) - int(
            left.get("partitions_reused", 0)
        )
        if built or reused:
            result[dataset] = {
                "hit": built == 0 and reused > 0,
                "partitions_built": built,
                "partitions_reused": reused,
            }
    return result or None


def _median(records: list[dict], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def _quality_summary(data_quality) -> tuple[dict[str, str], dict[str, int]]:
    if data_quality is None:
        return {}, {"hit": 0, "miss": 0}
    datasets = {
        item.display_key: item.status.value
        for item in data_quality.datasets
    }
    hits = sum(bool(item.cache_hit) for item in data_quality.datasets)
    return datasets, {"hit": hits, "miss": len(data_quality.datasets) - hits}


def main() -> int:
    args = build_parser().parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")

    config = load_data_lake_config(args.config)
    if args.include_trade_flow:
        config = replace(
            config,
            features=replace(config.features, trade_flow_enabled=True),
        )
    if args.include_order_book:
        config = replace(
            config,
            features=replace(config.features, order_book_enabled=True),
        )
    request = DataRequest(
        symbol=args.symbol,
        start=_utc(args.start),
        end=_utc(args.end),
        strategy_interval=f"{int(config.data.strategy_timeframe_minutes)}m",
        intrabar_interval=(
            f"{int(config.data.intrabar_timeframe_minutes)}m"
            if config.data.use_intrabar_data
            else None
        ),
    )
    intrabar_start = _utc(args.intrabar_start) if args.intrabar_start else None
    store = MarketDataStore(args.raw_root, args.cache_root)
    simulator = NativeSimulator()
    runner = ResearchRunner(
        store,
        production_feature_registry(),
        PreparedRunCache(args.cache_root),
        NativeStrategyPolicy(),
        simulator,
        (),
    )

    catalog_seconds = 0.0
    if not args.skip_catalog_refresh:
        started = time.perf_counter()
        store.refresh_catalog()
        catalog_seconds = time.perf_counter() - started

    records: list[dict[str, object]] = []
    fingerprints: list[str] = []
    for iteration in range(1, args.iterations + 1):
        total_started = time.perf_counter()
        canonical_before = dict(store.canonical_cache_events)
        book_before = _book_snapshot_events(store)
        result_run = runner.run(
            request,
            config,
            refresh_catalog=False,
            intrabar_start=intrabar_start,
        )
        canonical_after = dict(store.canonical_cache_events)
        book_after = _book_snapshot_events(store)
        canonical_delta = {
            name: int(canonical_after.get(name, 0) - canonical_before.get(name, 0))
            for name in sorted(set(canonical_before) | set(canonical_after))
        }

        preparation_seconds = float(result_run.stage_timings["data_features"])
        prepared_cache_seconds = float(result_run.stage_timings["prepared_cache"])
        engine_init_seconds = prepared_cache_seconds + float(
            result_run.stage_timings.get("engine_init", 0.0)
        )
        simulation_seconds = float(result_run.stage_timings["simulation"])
        trades = result_run.trades
        total_seconds = time.perf_counter() - total_started
        fingerprint = _trade_fingerprint(trades)
        fingerprints.append(fingerprint)
        quality_datasets, quality_cache = _quality_summary(result_run.data_quality)
        feature_metadata = dict(result_run.feature_cache_metadata)
        records.append(
            {
                "iteration": iteration,
                "preparation_seconds": preparation_seconds,
                "prepared_cache_seconds": prepared_cache_seconds,
                "engine_init_seconds": engine_init_seconds,
                "simulation_seconds": simulation_seconds,
                "total_seconds": total_seconds,
                "strategy_rows": result_run.strategy_rows,
                "intrabar_rows": result_run.intrabar_rows,
                "trade_rows": len(trades),
                "intrabar_index_mode": None,
                "intrabar_iteration_mode": None,
                "feature_cache_hits": _cache_hits(feature_metadata),
                "trade_aggregate_cache": _trade_aggregate_cache(feature_metadata),
                "book_snapshot_cache": _book_snapshot_delta(book_before, book_after),
                "prepared_cache_hit": result_run.prepared_cache_hit,
                "prepared_cache_key": result_run.prepared_cache_key,
                "canonical_cache": canonical_delta,
                "trade_fingerprint": fingerprint,
                "data_quality_status": (
                    result_run.data_quality.overall_status.value
                    if result_run.data_quality
                    else None
                ),
                "data_quality_datasets": quality_datasets,
                "data_quality_cache": quality_cache,
            }
        )

    warm_records = records[1:] if len(records) > 1 else records
    result = {
        "benchmark_contract": "data_lake_execution_v1",
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
        "iterations": args.iterations,
        "catalog_refresh_seconds": catalog_seconds,
        "catalog_refresh_skipped": bool(args.skip_catalog_refresh),
        "trade_flow_enabled": bool(args.include_trade_flow),
        "order_book_enabled": bool(args.include_order_book),
        "trade_fingerprints_identical": len(set(fingerprints)) <= 1,
        "warm_median_seconds": {
            "preparation": _median(warm_records, "preparation_seconds"),
            "prepared_cache": _median(warm_records, "prepared_cache_seconds"),
            "engine_init": _median(warm_records, "engine_init_seconds"),
            "simulation": _median(warm_records, "simulation_seconds"),
            "total": _median(warm_records, "total_seconds"),
        },
        "runs": records,
    }

    text = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["trade_fingerprints_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
