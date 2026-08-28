"""Validate canonical Data Lake coverage, schema, provenance, and causality.

Example:
    python tools/data_lake_validate.py \
      --raw-root "C:\\CryptoBots\\Binance Market Data" \
      --symbol BTCUSDT --strategy-interval 15m --intrabar-interval 1m \
      --start 2025-01-01 --end 2025-02-01
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from crypto_strategy_lab.data import (
    DataQualityReport,
    DataRequest,
    DatasetKind,
    MarketDataStore,
    validate_dataset,
)


def _utc_datetime(text: str) -> datetime:
    return pd.Timestamp(text, tz="UTC").to_pydatetime()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and validate the Binance Data Lake v2 path")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--strategy-interval", required=True)
    parser.add_argument("--intrabar-interval")
    parser.add_argument("--start", required=True, type=_utc_datetime)
    parser.add_argument("--end", required=True, type=_utc_datetime)
    return parser


def validate_canonical_klines(frame: pd.DataFrame, request: DataRequest, label: str) -> None:
    """Compatibility facade over the production validator for supplied test slices."""
    dataset_report = validate_dataset(
        frame,
        request,
        DatasetKind.KLINES,
        interval=request.strategy_interval,
        required=True,
    )
    # This helper validates a supplied slice, which need not represent the whole
    # request. The normal CLI and production path use MarketDataStore's complete
    # cached coverage/overlap-aware quality report below.
    integrity = tuple(
        issue
        for issue in dataset_report.issues
        if "COVERAGE_GAP" not in issue.code and issue.code != "MISSING_INTERNAL_INTERVAL"
    )
    if any(issue.severity.value == "ERROR" for issue in integrity):
        raise ValueError(
            f"{label} failed canonical validation: {[issue.code for issue in integrity]}"
        )


def print_report(report: DataQualityReport) -> None:
    print("\nDATA QUALITY")
    print(f"{'Dataset':38} {'Status':9} {'Rows':>10} Details")
    for item in report.datasets:
        details = ", ".join(
            f"{issue.code} ({issue.count})" for issue in item.issues
        )
        print(
            f"{item.display_key:38} {item.status.value:9} {item.row_count:10d} {details}"
        )


def main() -> int:
    args = build_parser().parse_args()
    store = MarketDataStore(args.raw_root, args.cache_root)
    count = store.refresh_catalog()
    print(f"Cataloged archives: {count}")

    request = DataRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        strategy_interval=args.strategy_interval,
        intrabar_interval=args.intrabar_interval,
    )
    reports = [
        store.data_quality_report(
            request,
            DatasetKind.KLINES,
            interval=request.strategy_interval,
            required=True,
        )
    ]
    if request.intrabar_interval:
        intrabar_request = DataRequest(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            strategy_interval=request.intrabar_interval,
            market=request.market,
            exchange=request.exchange,
        )
        reports.append(
            store.data_quality_report(
                intrabar_request,
                DatasetKind.KLINES,
                interval=request.intrabar_interval,
                required=True,
            )
        )

    report = DataQualityReport(tuple(reports))
    print_report(report)
    report.raise_for_errors()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
