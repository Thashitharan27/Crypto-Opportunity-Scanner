"""Profile canonical data/feature preparation used by ResearchRunner."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import cProfile
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import pstats
import sys
import time
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data.backtest_service import load_backtest_bundle
from crypto_strategy_lab.data_lake_config import load_data_lake_config
from crypto_strategy_lab.features import production_feature_registry


Event = dict[str, object]


def _utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _rows(value) -> int | None:
    try:
        return int(len(value))
    except (TypeError, AttributeError):
        return None


def _record(events, *, category, name, started, rows=None, **metadata):
    event = {
        "category": category,
        "name": name,
        "seconds": float(time.perf_counter() - started),
    }
    if rows is not None:
        event["rows"] = int(rows)
    event.update({key: value for key, value in metadata.items() if value is not None})
    events.append(event)


def _summarize_events(events: list[Event], total_seconds: float) -> dict[str, object]:
    category_seconds: dict[str, float] = defaultdict(float)
    category_calls: dict[str, int] = defaultdict(int)
    for event in events:
        category = str(event["category"])
        category_seconds[category] += float(event["seconds"])
        category_calls[category] += 1
    measured = float(sum(category_seconds.values()))
    return {
        "category_seconds": {
            key: float(value)
            for key, value in sorted(
                category_seconds.items(), key=lambda item: item[1], reverse=True
            )
        },
        "category_calls": dict(sorted(category_calls.items())),
        "instrumented_seconds": measured,
        "unattributed_seconds": max(0.0, float(total_seconds) - measured),
    }


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


@contextmanager
def _instrument_preparation(store: MarketDataStore, events: list[Event]) -> Iterator[None]:
    """Instrument canonical dataset boundaries without replacing feature authority."""
    original_load_dataset = store.load_dataset
    original_load_execution_klines = store.load_execution_klines

    def timed_load_dataset(request, dataset, *, interval=None):
        started = time.perf_counter()
        frame = original_load_dataset(request, dataset, interval=interval)
        _record(
            events,
            category="dataset_load",
            name=f"{dataset.value}:{interval or '-'}",
            started=started,
            rows=_rows(frame),
            dataset=dataset.value,
            interval=interval,
        )
        return frame

    def timed_load_execution_klines(request, interval=None):
        started = time.perf_counter()
        frame = original_load_execution_klines(request, interval)
        _record(
            events,
            category="dataset_load",
            name=f"klines:{interval or request.strategy_interval}:execution",
            started=started,
            rows=_rows(frame),
            dataset="klines",
            interval=interval or request.strategy_interval,
            projection="canonical_execution_ohlv",
        )
        return frame

    store.load_dataset = timed_load_dataset
    store.load_execution_klines = timed_load_execution_klines
    try:
        yield
    finally:
        store.load_dataset = original_load_dataset
        store.load_execution_klines = original_load_execution_klines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ResearchRunner canonical data/feature preparation"
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
    registry = production_feature_registry()

    events: list[Event] = []
    profile = cProfile.Profile()
    started = time.perf_counter()
    with _instrument_preparation(store, events):
        profile.enable()
        bundle = load_backtest_bundle(
            store,
            request,
            refresh_catalog=False,
            intrabar_start=intrabar_start,
            feature_registry=registry,
            feature_config=config.features,
        )
        profile.disable()
    elapsed = time.perf_counter() - started

    result = {
        "profile_contract": "research_runner_preparation_profile_v2",
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
        "trade_flow_enabled": bool(config.features.trade_flow_enabled),
        "preparation_seconds_profiled": elapsed,
        "stage_summary": _summarize_events(events, elapsed),
        "events": sorted(events, key=lambda event: float(event["seconds"]), reverse=True),
        "strategy_rows": len(bundle.strategy),
        "intrabar_rows": len(bundle.intrabar) if bundle.intrabar is not None else 0,
        "feature_cache_hits": {
            "core_directional": bool(
                bundle.technical_features.attrs.get("feature_cache_hit", False)
            ),
            "production_market_context": bool(
                bundle.context_features.attrs.get("feature_cache_hit", False)
            ),
            "support_resistance": (
                bool(bundle.support_resistance_features.attrs.get("feature_cache_hit", False))
                if bundle.support_resistance_features is not None
                else None
            ),
            "state_transition_daily": (
                bool(bundle.state_transition_daily_features.attrs.get("feature_cache_hit", False))
                if bundle.state_transition_daily_features is not None
                else None
            ),
            "research": {
                name: bool(frame.attrs.get("feature_cache_hit", False))
                for name, frame in sorted(bundle.research_features.items())
            },
        },
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
