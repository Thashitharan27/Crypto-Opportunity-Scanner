"""Profile only the authoritative reporting/artifact publication stage.

This tool deliberately leaves preparation and simulation outside cProfile so the
result identifies what is expensive after the simulator has already completed.
It uses the real CsvManifestReporter and therefore preserves the exact production
artifact/validation path.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data_lake_config import load_data_lake_config
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.prepared_cache import PreparedRunCache
from crypto_strategy_lab.research_adapters import NativeSimulator, NativeStrategyPolicy
from crypto_strategy_lab.research_reporting import CsvManifestReporter
from crypto_strategy_lab.research_runner import ResearchRunner


def _utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _profile_rows(profile: cProfile.Profile, sort_key: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


class ProfiledReporter:
    """Wrap one production reporter and profile only its ``report`` call."""

    def __init__(self, reporter) -> None:
        self.reporter = reporter
        self.profile = cProfile.Profile()
        self.elapsed_seconds: float | None = None

    def begin(self, request, config) -> None:
        begin = getattr(self.reporter, "begin", None)
        if begin is not None:
            begin(request, config)

    def report(self, result, context) -> None:
        started = time.perf_counter()
        self.profile.enable()
        try:
            self.reporter.report(result, context)
        finally:
            self.profile.disable()
            self.elapsed_seconds = time.perf_counter() - started


def _artifact_summary(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(run_dir) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {
        name: {
            "path": values.get("path"),
            "format": values.get("format"),
            "rows": values.get("rows"),
            "bytes": values.get("bytes"),
        }
        for name, values in manifest.get("artifacts", {}).items()
    }
    research = manifest.get("research", {})
    reporting = {
        "manifest_reporting_seconds": manifest.get("execution_result", {})
        .get("stage_timings", {})
        .get("reporting"),
        "research_artifact_write_seconds": research.get("artifact_write_seconds"),
        "trade_rows": research.get("trade_row_count"),
        "feature_context_rows": research.get("feature_context_row_count"),
        "trade_columns": len(research.get("trade_columns", ())),
        "feature_context_columns": len(research.get("feature_context_columns", ())),
        "trade_context_parity_columns": len(
            research.get("trade_context_parity_columns", ())
        ),
    }
    return reporting, artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile only Crypto Strategy Lab's post-simulation reporting and "
            "artifact publication stage"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--intrabar-start")
    parser.add_argument("--include-trade-flow", action="store_true")
    parser.add_argument("--include-order-book", action="store_true")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        help=(
            "Optional directory in which to preserve the profiled run. When omitted, "
            "a temporary directory is used and removed after profiling."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def _run_profile(args, report_root: Path) -> dict[str, Any]:
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
    reporter = ProfiledReporter(CsvManifestReporter(report_root))
    runner = ResearchRunner(
        store,
        production_feature_registry(),
        PreparedRunCache(args.cache_root),
        NativeStrategyPolicy(),
        NativeSimulator(),
        (reporter,),
    )

    total_started = time.perf_counter()
    run = runner.run(request, config, intrabar_start=intrabar_start)
    total_seconds = time.perf_counter() - total_started
    if run.output_dir is None:
        raise RuntimeError("profiled production reporter did not publish a completed run")

    reporting_detail, artifacts = _artifact_summary(run.output_dir)
    stage_timings = dict(run.stage_timings)
    reporting_seconds = float(stage_timings.get("reporting", 0.0) or 0.0)
    result = {
        "profile_contract": "data_lake_reporting_profile_v1",
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
        "stage_timings": stage_timings,
        "profiled_reporting_seconds": reporter.elapsed_seconds,
        "reporting_share_of_total": (
            reporting_seconds / total_seconds if total_seconds > 0 else 0.0
        ),
        "total_run_seconds": total_seconds,
        "prepared_cache_hit": run.prepared_cache_hit,
        "strategy_rows": run.strategy_rows,
        "intrabar_rows": run.intrabar_rows,
        "trade_rows": len(run.trades),
        "reporting_detail": reporting_detail,
        "artifact_sizes": artifacts,
        "top_by_cumulative": _profile_rows(reporter.profile, "cumulative", args.top),
        "top_by_self": _profile_rows(reporter.profile, "self", args.top),
        "artifacts_preserved": args.artifacts_root is not None,
        "run_dir": str(run.output_dir.resolve()) if args.artifacts_root is not None else None,
    }
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.top < 1:
        raise SystemExit("--top must be at least 1")

    if args.artifacts_root is not None:
        args.artifacts_root.mkdir(parents=True, exist_ok=True)
        result = _run_profile(args, args.artifacts_root)
    else:
        with tempfile.TemporaryDirectory(prefix="crypto-strategy-report-profile-") as temp:
            result = _run_profile(args, Path(temp))

    text = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
