"""Typed, dataset-aware validation for canonical market data.

This module deliberately has no dependency on features, strategies, simulators,
or reporting. A validated frame may therefore be handed to any downstream
consumer with the same operational contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .schemas import DatasetKind
from .timing import interval_to_timedelta


VALIDATION_CONTRACT_VERSION = "5"
QUALITY_CACHE_FORMAT_VERSION = 1


class DataQualityStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    code: str
    severity: DataQualityStatus
    message: str
    count: int = 1
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "count": int(self.count),
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "details": dict(sorted(self.details.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataQualityIssue":
        return cls(
            code=str(value["code"]),
            severity=DataQualityStatus(value["severity"]),
            message=str(value["message"]),
            count=int(value.get("count", 1)),
            first_timestamp=value.get("first_timestamp"),
            last_timestamp=value.get("last_timestamp"),
            details=value.get("details", {}),
        )


@dataclass(frozen=True, slots=True)
class DatasetQualityReport:
    dataset: str
    symbol: str
    interval: str | None
    required: bool
    requested_start: str
    requested_end: str
    observed_start: str | None
    observed_end: str | None
    complete_start: str | None
    complete_end: str | None
    row_count: int
    source_identity: str | None
    status: DataQualityStatus
    issues: tuple[DataQualityIssue, ...] = ()
    cache_hit: bool = False

    @property
    def display_key(self) -> str:
        interval = self.interval or "event"
        return f"{self.dataset}:{self.symbol}:{interval}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "symbol": self.symbol,
            "interval": self.interval,
            "required": self.required,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "complete_start": self.complete_start,
            "complete_end": self.complete_end,
            "row_count": int(self.row_count),
            "source_identity": self.source_identity,
            "status": self.status.value,
            "issues": [item.to_dict() for item in self.issues],
            "cache_hit": bool(self.cache_hit),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetQualityReport":
        return cls(
            dataset=str(value["dataset"]),
            symbol=str(value["symbol"]),
            interval=value.get("interval"),
            required=bool(value["required"]),
            requested_start=str(value["requested_start"]),
            requested_end=str(value["requested_end"]),
            observed_start=value.get("observed_start"),
            observed_end=value.get("observed_end"),
            complete_start=value.get("complete_start"),
            complete_end=value.get("complete_end"),
            row_count=int(value["row_count"]),
            source_identity=value.get("source_identity"),
            status=DataQualityStatus(value["status"]),
            issues=tuple(DataQualityIssue.from_dict(item) for item in value.get("issues", ())),
            cache_hit=bool(value.get("cache_hit", False)),
        )


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    datasets: tuple[DatasetQualityReport, ...]

    @property
    def overall_status(self) -> DataQualityStatus:
        statuses = {item.status for item in self.datasets}
        if DataQualityStatus.ERROR in statuses:
            return DataQualityStatus.ERROR
        if DataQualityStatus.WARN in statuses or DataQualityStatus.MISSING in statuses:
            return DataQualityStatus.WARN
        return DataQualityStatus.OK

    @property
    def issues(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for report in self.datasets for issue in report.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": VALIDATION_CONTRACT_VERSION,
            "overall_status": self.overall_status.value,
            "datasets": [item.to_dict() for item in self.datasets],
            "summary": {
                "dataset_count": len(self.datasets),
                "issue_count": len(self.issues),
                "cache_hits": sum(bool(item.cache_hit) for item in self.datasets),
            },
        }

    def raise_for_errors(self) -> None:
        errors = [item for item in self.datasets if item.status is DataQualityStatus.ERROR]
        if not errors:
            return
        detail = "; ".join(
            f"{report.display_key}: {', '.join(issue.code for issue in report.issues)}"
            for report in errors
        )
        raise DataQualityError(f"Required market data failed validation: {detail}", self)


class DataQualityError(ValueError):
    def __init__(self, message: str, report: DataQualityReport):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class DatasetValidationContract:
    dataset: DatasetKind
    timeline: str  # "fixed" or "event"
    timestamp_column: str
    logical_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    numeric_fields: tuple[str, ...] = ()
    positive_fields: tuple[str, ...] = ()
    non_negative_fields: tuple[str, ...] = ()
    nullable_numeric_fields: tuple[str, ...] = ()
    nullable_positive_fields: tuple[str, ...] = ()
    nullable_non_negative_fields: tuple[str, ...] = ()


_IDENTITY = ("symbol", "exchange", "market", "dataset", "available_at")
_CANDLE = (
    "period_start",
    "period_end",
    *_IDENTITY,
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_KLINE_KINDS = (
    DatasetKind.KLINES,
    DatasetKind.MARK_PRICE_KLINES,
    DatasetKind.INDEX_PRICE_KLINES,
)

CONTRACTS: dict[DatasetKind, DatasetValidationContract] = {
    kind: DatasetValidationContract(
        dataset=kind,
        timeline="fixed",
        timestamp_column="period_start",
        logical_key=("period_start",),
        required_columns=_CANDLE,
        numeric_fields=("open", "high", "low", "close", "volume"),
        positive_fields=("open", "high", "low", "close"),
        non_negative_fields=("volume",),
        nullable_non_negative_fields=(
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ),
    )
    for kind in _KLINE_KINDS
}
CONTRACTS.update(
    {
        # Premium-index klines are signed basis-like values rather than prices.
        # Their OHLC values may legitimately be negative or zero; they must only
        # be finite and obey ordinary OHLC bounds. Volume remains non-negative.
        DatasetKind.PREMIUM_INDEX_KLINES: DatasetValidationContract(
            dataset=DatasetKind.PREMIUM_INDEX_KLINES,
            timeline="fixed",
            timestamp_column="period_start",
            logical_key=("period_start",),
            required_columns=_CANDLE,
            numeric_fields=("open", "high", "low", "close"),
            non_negative_fields=("volume",),
            nullable_non_negative_fields=(
                "quote_volume",
                "trade_count",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
            ),
        ),
        # The current Binance metrics adapter does not preserve a declared row
        # interval, so metrics are validated as timestamped snapshots rather
        # than manufacturing a fixed grid. Compact metric fields are sparse in
        # some genuine Binance archives: missing means unknown, while supplied
        # values still have to be finite and non-negative.
        DatasetKind.FUTURES_METRICS: DatasetValidationContract(
            dataset=DatasetKind.FUTURES_METRICS,
            timeline="event",
            timestamp_column="event_time",
            logical_key=("event_time",),
            required_columns=("event_time", *_IDENTITY),
            nullable_non_negative_fields=(
                "open_interest",
                "open_interest_value",
                "top_trader_account_long_short_ratio",
                "top_trader_position_long_short_ratio",
                "global_long_short_account_ratio",
                "taker_long_short_volume_ratio",
            ),
        ),
        DatasetKind.FUNDING_RATE: DatasetValidationContract(
            dataset=DatasetKind.FUNDING_RATE,
            timeline="event",
            timestamp_column="event_time",
            logical_key=("event_time",),
            required_columns=("event_time", *_IDENTITY, "funding_rate"),
            numeric_fields=("funding_rate",),
            nullable_positive_fields=("funding_interval_hours",),
        ),
        DatasetKind.AGG_TRADES: DatasetValidationContract(
            dataset=DatasetKind.AGG_TRADES,
            timeline="event",
            timestamp_column="event_time",
            logical_key=("event_time", "agg_trade_id"),
            required_columns=(
                "event_time",
                *_IDENTITY,
                "agg_trade_id",
                "price",
                "quantity",
            ),
            positive_fields=("price",),
            non_negative_fields=("quantity", "agg_trade_id"),
        ),
        DatasetKind.TRADES: DatasetValidationContract(
            dataset=DatasetKind.TRADES,
            timeline="event",
            timestamp_column="event_time",
            logical_key=("event_time", "trade_id"),
            required_columns=(
                "event_time",
                *_IDENTITY,
                "trade_id",
                "price",
                "quantity",
            ),
            positive_fields=("price",),
            non_negative_fields=("quantity", "trade_id"),
        ),
        DatasetKind.BOOK_TICKER: DatasetValidationContract(
            dataset=DatasetKind.BOOK_TICKER, timeline="event", timestamp_column="event_time",
            logical_key=("update_id",),
            required_columns=("event_time", *_IDENTITY, "transaction_time", "update_id",
                              "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"),
            positive_fields=("best_bid_price", "best_ask_price"),
            non_negative_fields=("update_id", "best_bid_qty", "best_ask_qty"),
        ),
        DatasetKind.BOOK_DEPTH: DatasetValidationContract(
            dataset=DatasetKind.BOOK_DEPTH, timeline="event", timestamp_column="event_time",
            logical_key=("event_time", "percentage"),
            required_columns=("event_time", *_IDENTITY, "percentage", "depth", "notional"),
            numeric_fields=("percentage",), non_negative_fields=("depth", "notional"),
        ),
    }
)


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _issue(
    code: str,
    severity: DataQualityStatus,
    message: str,
    *,
    mask=None,
    timestamps=None,
    count: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> DataQualityIssue:
    if mask is not None:
        issue_count = int(mask.sum())
        selected = timestamps[mask] if timestamps is not None else None
    else:
        issue_count = int(count if count is not None else 1)
        selected = timestamps
    return DataQualityIssue(
        code=code,
        severity=severity,
        message=message,
        count=issue_count,
        first_timestamp=_iso(selected.min()) if selected is not None and len(selected) else None,
        last_timestamp=_iso(selected.max()) if selected is not None and len(selected) else None,
        details=dict(details or {}),
    )


def _status(
    required: bool,
    issues: Iterable[DataQualityIssue],
    *,
    missing: bool = False,
) -> DataQualityStatus:
    issues = tuple(issues)
    if missing:
        return DataQualityStatus.ERROR if required else DataQualityStatus.MISSING
    if any(item.severity is DataQualityStatus.ERROR for item in issues):
        return DataQualityStatus.ERROR if required else DataQualityStatus.WARN
    return DataQualityStatus.WARN if issues else DataQualityStatus.OK


def _gap_ranges(values: pd.DatetimeIndex, delta: pd.Timedelta, limit: int = 10) -> list[dict[str, Any]]:
    if not len(values):
        return []
    result: list[dict[str, Any]] = []
    start = previous = values[0]
    count = 1
    for current in values[1:]:
        if current - previous == delta:
            count += 1
        else:
            result.append({"start": _iso(start), "end": _iso(previous), "missing_count": count})
            if len(result) >= limit:
                return result
            start = current
            count = 1
        previous = current
    if len(result) < limit:
        result.append({"start": _iso(start), "end": _iso(previous), "missing_count": count})
    return result


def validate_dataset(
    frame: pd.DataFrame | None,
    request,
    dataset: DatasetKind,
    *,
    interval: str | None = None,
    required: bool = True,
    source_identity: str | None = None,
    coverage_start: Any | None = None,
    coverage_end: Any | None = None,
    extra_issues: Iterable[DataQualityIssue] = (),
) -> DatasetQualityReport:
    """Validate one canonical frame without manufacturing an event-stream grid."""
    try:
        contract = CONTRACTS[dataset]
    except KeyError as exc:
        raise ValueError(f"No data-quality contract for {dataset.value}") from exc

    start = pd.Timestamp(request.start)
    end = pd.Timestamp(request.end)
    source_identity = source_identity or (
        frame.attrs.get("canonical_source_identity") if frame is not None else None
    )
    extras = tuple(extra_issues)

    if frame is None or frame.empty:
        missing_issue = DataQualityIssue(
            "DATASET_MISSING",
            DataQualityStatus.ERROR if required else DataQualityStatus.MISSING,
            "No canonical rows are available for the requested range",
            0,
        )
        issues = (missing_issue, *extras)
        return DatasetQualityReport(
            dataset=dataset.value,
            symbol=request.symbol,
            interval=interval,
            required=required,
            requested_start=_iso(start),
            requested_end=_iso(end),
            observed_start=None,
            observed_end=None,
            complete_start=None,
            complete_end=None,
            row_count=0,
            source_identity=source_identity,
            status=_status(required, issues, missing=True),
            issues=issues,
        )

    issues: list[DataQualityIssue] = list(extras)
    missing_columns = sorted(set(contract.required_columns) - set(frame.columns))
    if missing_columns:
        issues.append(
            DataQualityIssue(
                "MISSING_REQUIRED_COLUMN",
                DataQualityStatus.ERROR,
                f"Missing canonical columns: {missing_columns}",
                len(missing_columns),
                details={"columns": missing_columns},
            )
        )
    if not source_identity:
        issues.append(
            DataQualityIssue(
                "MISSING_SOURCE_IDENTITY",
                DataQualityStatus.ERROR,
                "Canonical source identity is absent",
            )
        )

    if contract.timestamp_column in frame:
        timestamps = pd.to_datetime(
            frame[contract.timestamp_column], utc=True, errors="coerce"
        )
        malformed = timestamps.isna()
        if malformed.any():
            issues.append(
                _issue(
                    "MALFORMED_TIMESTAMP",
                    DataQualityStatus.ERROR,
                    "Timeline timestamp is not parseable",
                    mask=malformed,
                    timestamps=timestamps,
                )
            )
        if not timestamps.dropna().is_monotonic_increasing:
            issues.append(
                DataQualityIssue(
                    "NON_MONOTONIC_TIMELINE",
                    DataQualityStatus.ERROR,
                    "Canonical timestamps are not monotonically increasing",
                )
            )
    else:
        timestamps = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")

    present_keys = [column for column in contract.logical_key if column in frame]
    if len(present_keys) == len(contract.logical_key):
        duplicate = frame.duplicated(present_keys, keep=False)
        if duplicate.any():
            issues.append(
                _issue(
                    "DUPLICATE_LOGICAL_KEY",
                    DataQualityStatus.ERROR,
                    "Canonical logical keys are duplicated",
                    mask=duplicate,
                    timestamps=timestamps,
                )
            )

    for column, expected in (
        ("symbol", request.symbol),
        ("exchange", request.exchange),
        ("market", request.market.value),
        ("dataset", dataset.value),
    ):
        if column not in frame:
            continue
        observed = set(frame[column].dropna().astype(str))
        if observed != {str(expected)}:
            issues.append(
                DataQualityIssue(
                    "IDENTITY_MISMATCH",
                    DataQualityStatus.ERROR,
                    f"Canonical {column} does not match request",
                    details={"column": column, "observed": sorted(observed)},
                )
            )

    if "interval" in frame.columns and interval is not None:
        observed_intervals = set(frame["interval"].dropna().astype(str))
        if observed_intervals and observed_intervals != {str(interval)}:
            issues.append(
                DataQualityIssue(
                    "INTERVAL_MISMATCH",
                    DataQualityStatus.ERROR,
                    "Canonical interval does not match the requested interval",
                    details={"expected": interval, "observed": sorted(observed_intervals)},
                )
            )

    available = None
    if "available_at" in frame:
        available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        malformed_available = available.isna()
        if malformed_available.any():
            issues.append(
                _issue(
                    "MALFORMED_AVAILABLE_AT",
                    DataQualityStatus.ERROR,
                    "available_at is missing or malformed",
                    mask=malformed_available,
                    timestamps=timestamps,
                )
            )
        if contract.timeline == "fixed" and "period_end" in frame:
            comparison = pd.to_datetime(frame["period_end"], utc=True, errors="coerce")
        else:
            comparison = timestamps
        early = available < comparison
        if early.any():
            issues.append(
                _issue(
                    "NON_CAUSAL_AVAILABILITY",
                    DataQualityStatus.ERROR,
                    "available_at precedes the source observation",
                    mask=early,
                    timestamps=timestamps,
                )
            )

    period_end = None
    if contract.timeline == "fixed" and "period_end" in frame:
        period_end = pd.to_datetime(frame["period_end"], utc=True, errors="coerce")
        malformed_end = period_end.isna()
        if malformed_end.any():
            issues.append(
                _issue(
                    "MALFORMED_PERIOD_END",
                    DataQualityStatus.ERROR,
                    "period_end is missing or malformed",
                    mask=malformed_end,
                    timestamps=timestamps,
                )
            )
        bad_period = timestamps >= period_end
        if bad_period.any():
            issues.append(
                _issue(
                    "INVALID_PERIOD",
                    DataQualityStatus.ERROR,
                    "period_start must precede period_end",
                    mask=bad_period,
                    timestamps=timestamps,
                )
            )

    nullable_fields = {
        *contract.nullable_numeric_fields,
        *contract.nullable_positive_fields,
        *contract.nullable_non_negative_fields,
    }
    checked_numeric = set()
    for column in (
        *contract.numeric_fields,
        *contract.positive_fields,
        *contract.non_negative_fields,
        *contract.nullable_numeric_fields,
        *contract.nullable_positive_fields,
        *contract.nullable_non_negative_fields,
    ):
        if column not in frame or column in checked_numeric:
            continue
        checked_numeric.add(column)
        raw_values = frame[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        if column in nullable_fields:
            present = raw_values.notna()
            invalid_numeric = present & ~np.isfinite(values)
        else:
            present = pd.Series(True, index=values.index)
            invalid_numeric = ~np.isfinite(values)
        if invalid_numeric.any():
            issues.append(
                _issue(
                    "INVALID_NUMERIC",
                    DataQualityStatus.ERROR,
                    f"{column} must be finite when present",
                    mask=invalid_numeric,
                    timestamps=timestamps,
                    details={"column": column},
                )
            )
        if column in contract.positive_fields or column in contract.nullable_positive_fields:
            invalid_domain = present & (values <= 0)
        elif (
            column in contract.non_negative_fields
            or column in contract.nullable_non_negative_fields
        ):
            invalid_domain = present & (values < 0)
        else:
            invalid_domain = pd.Series(False, index=values.index)
        if invalid_domain.any():
            issues.append(
                _issue(
                    "INVALID_DOMAIN_VALUE",
                    DataQualityStatus.ERROR,
                    f"{column} is outside its valid domain",
                    mask=invalid_domain,
                    timestamps=timestamps,
                    details={"column": column},
                )
            )

    if all(column in frame for column in ("open", "high", "low", "close")):
        open_ = pd.to_numeric(frame["open"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        invalid_ohlc = (
            (low > high)
            | (open_ < low)
            | (open_ > high)
            | (close < low)
            | (close > high)
        )
        if invalid_ohlc.any():
            issues.append(
                _issue(
                    "INVALID_OHLC",
                    DataQualityStatus.ERROR,
                    "OHLC values violate candle bounds",
                    mask=invalid_ohlc,
                    timestamps=timestamps,
                )
            )

    if dataset is DatasetKind.BOOK_TICKER and {"best_bid_price", "best_ask_price"} <= set(frame):
        bid = pd.to_numeric(frame["best_bid_price"], errors="coerce")
        ask = pd.to_numeric(frame["best_ask_price"], errors="coerce")
        crossed = bid > ask
        locked = bid == ask
        if crossed.any():
            issues.append(_issue("CROSSED_BOOK", DataQualityStatus.ERROR,
                                 "best bid exceeds best ask", mask=crossed, timestamps=timestamps))
        if locked.any():
            issues.append(_issue("LOCKED_BOOK", DataQualityStatus.WARN,
                                 "best bid equals best ask", mask=locked, timestamps=timestamps))

    if dataset is DatasetKind.BOOK_DEPTH and "percentage" in frame:
        percentage = pd.to_numeric(frame["percentage"], errors="coerce")
        zero = percentage.eq(0)
        if zero.any():
            issues.append(_issue("ZERO_DEPTH_PERCENTAGE", DataQualityStatus.ERROR,
                                 "percentage-distance band cannot be zero", mask=zero, timestamps=timestamps))
        expected = {-5., -4., -3., -2., -1., 1., 2., 3., 4., 5.}
        partial_times = [time for time, group in frame.groupby("event_time", sort=False)
                         if not expected.issubset(set(pd.to_numeric(group["percentage"], errors="coerce")))]
        if partial_times:
            partial_index = pd.DatetimeIndex(partial_times)
            issues.append(_issue("PARTIAL_DEPTH_SNAPSHOT", DataQualityStatus.WARN,
                                 "percentage-band snapshot is incomplete",
                                 timestamps=partial_index, count=len(partial_index)))

    for taker, total in (
        ("taker_buy_base_volume", "volume"),
        ("taker_buy_quote_volume", "quote_volume"),
    ):
        if taker not in frame or total not in frame:
            continue
        taker_values = pd.to_numeric(frame[taker], errors="coerce")
        total_values = pd.to_numeric(frame[total], errors="coerce")
        tolerance = np.maximum(total_values.abs() * 1e-12, 1e-12)
        exceeds = taker_values > total_values + tolerance
        if exceeds.any():
            issues.append(
                _issue(
                    "TAKER_VOLUME_EXCEEDS_TOTAL",
                    DataQualityStatus.ERROR,
                    f"{taker} exceeds {total}",
                    mask=exceeds,
                    timestamps=timestamps,
                    details={"taker_column": taker, "total_column": total},
                )
            )

    complete_start = _iso(start)
    complete_end = _iso(end)
    valid_timestamps = pd.DatetimeIndex(timestamps.dropna())
    if contract.timeline == "fixed" and interval and len(valid_timestamps):
        delta = interval_to_timedelta(interval)
        expected = pd.date_range(start=start, end=end, freq=delta, inclusive="left")
        missing_grid = expected.difference(valid_timestamps)
        offset = valid_timestamps - start
        off_grid = valid_timestamps[(offset % delta) != pd.Timedelta(0)]
        if len(off_grid):
            issues.append(
                _issue(
                    "OFF_GRID_TIMESTAMP",
                    DataQualityStatus.ERROR,
                    "Fixed-cadence timestamps are off grid",
                    timestamps=off_grid,
                    count=len(off_grid),
                )
            )
        if len(missing_grid):
            leading = missing_grid[missing_grid < valid_timestamps.min()]
            trailing = missing_grid[missing_grid > valid_timestamps.max()]
            internal = missing_grid[
                (missing_grid > valid_timestamps.min())
                & (missing_grid < valid_timestamps.max())
            ]
            for code, values, message in (
                (
                    "LEADING_COVERAGE_GAP",
                    leading,
                    "Leading fixed-cadence coverage is missing",
                ),
                (
                    "TRAILING_COVERAGE_GAP",
                    trailing,
                    "Trailing fixed-cadence coverage is missing",
                ),
                (
                    "MISSING_INTERNAL_INTERVAL",
                    internal,
                    "Internal fixed-cadence intervals are missing",
                ),
            ):
                if not len(values):
                    continue
                issues.append(
                    _issue(
                        code,
                        DataQualityStatus.ERROR,
                        message,
                        timestamps=values,
                        count=len(values),
                        details={"ranges": _gap_ranges(values, delta)},
                    )
                )
            if len(leading):
                complete_start = _iso(valid_timestamps.min())
            if len(trailing):
                complete_end = _iso(valid_timestamps.max() + delta)

    # Catalog coverage is especially useful for irregular event streams where
    # a regular grid would be meaningless. If the catalog itself begins/ends
    # inside the requested slice, that is a real source-coverage gap.
    if coverage_start is not None and pd.Timestamp(coverage_start) > start:
        issues.append(
            DataQualityIssue(
                "LEADING_SOURCE_COVERAGE_GAP",
                DataQualityStatus.ERROR if required else DataQualityStatus.WARN,
                "Catalog source coverage begins after the requested start",
                details={"coverage_start": _iso(coverage_start)},
            )
        )
        complete_start = _iso(coverage_start)
    if coverage_end is not None and pd.Timestamp(coverage_end) < end:
        issues.append(
            DataQualityIssue(
                "TRAILING_SOURCE_COVERAGE_GAP",
                DataQualityStatus.ERROR if required else DataQualityStatus.WARN,
                "Catalog source coverage ends before the requested end",
                details={"coverage_end": _iso(coverage_end)},
            )
        )
        complete_end = _iso(coverage_end)

    observed_start = _iso(valid_timestamps.min()) if len(valid_timestamps) else None
    if period_end is not None and period_end.notna().any():
        observed_end = _iso(period_end.max())
    else:
        observed_end = _iso(valid_timestamps.max()) if len(valid_timestamps) else None

    return DatasetQualityReport(
        dataset=dataset.value,
        symbol=request.symbol,
        interval=interval,
        required=required,
        requested_start=_iso(start),
        requested_end=_iso(end),
        observed_start=observed_start,
        observed_end=observed_end,
        complete_start=complete_start,
        complete_end=complete_end,
        row_count=len(frame),
        source_identity=source_identity,
        status=_status(required, issues),
        issues=tuple(issues),
    )


_PROVENANCE_COLUMNS = {
    "source_archive",
    "source_fingerprint",
    "filename",
    "raw_root",
}


def _overlap_contract(frame: pd.DataFrame) -> DatasetValidationContract | None:
    if "dataset" not in frame.columns:
        return None
    values = tuple(frame["dataset"].dropna().astype(str).unique())
    if len(values) != 1:
        return None
    try:
        return CONTRACTS[DatasetKind(values[0])]
    except (KeyError, ValueError):
        return None


def _overlap_rows_equal(left: pd.Series, right: pd.Series, columns: Iterable[str]) -> bool:
    for column in columns:
        left_value = left[column]
        right_value = right[column]
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if left_value != right_value:
            return False
    return True


def _kline_overlap_row_is_valid(
    row: pd.Series,
    contract: DatasetValidationContract,
) -> bool:
    """Validate only row-value invariants needed to classify a repair override."""
    if contract.timeline != "fixed":
        return False
    required_values = {"open", "high", "low", "close", "volume"}
    if not required_values.issubset(row.index):
        return False

    nullable_fields = {
        *contract.nullable_numeric_fields,
        *contract.nullable_positive_fields,
        *contract.nullable_non_negative_fields,
    }
    checked = set()
    for column in (
        *contract.numeric_fields,
        *contract.positive_fields,
        *contract.non_negative_fields,
        *contract.nullable_numeric_fields,
        *contract.nullable_positive_fields,
        *contract.nullable_non_negative_fields,
    ):
        if column not in row.index or column in checked:
            continue
        checked.add(column)
        raw = row[column]
        if column in nullable_fields and pd.isna(raw):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(value):
            return False
        if column in contract.positive_fields or column in contract.nullable_positive_fields:
            if value <= 0:
                return False
        elif (
            column in contract.non_negative_fields
            or column in contract.nullable_non_negative_fields
        ) and value < 0:
            return False

    try:
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
    except (TypeError, ValueError):
        return False
    if low > high or open_ < low or open_ > high or close < low or close > high:
        return False

    for taker, total in (
        ("taker_buy_base_volume", "volume"),
        ("taker_buy_quote_volume", "quote_volume"),
    ):
        if taker not in row.index or total not in row.index:
            continue
        if pd.isna(row[taker]) or pd.isna(row[total]):
            continue
        try:
            taker_value = float(row[taker])
            total_value = float(row[total])
        except (TypeError, ValueError):
            return False
        if not np.isfinite(taker_value) or not np.isfinite(total_value):
            return False
        tolerance = max(abs(total_value) * 1e-12, 1e-12)
        if taker_value > total_value + tolerance:
            return False
    return True


def _is_valid_repair_override(
    group: pd.DataFrame,
    value_columns: Iterable[str],
    contract: DatasetValidationContract | None,
) -> bool:
    """Return true only when the winning row is valid and every differing loser is invalid."""
    if contract is None or len(group) < 2:
        return False
    winner = group.iloc[-1]
    if not _kline_overlap_row_is_valid(winner, contract):
        return False

    replaced_invalid = False
    for index in range(len(group) - 1):
        previous = group.iloc[index]
        if _overlap_rows_equal(previous, winner, value_columns):
            continue
        if _kline_overlap_row_is_valid(previous, contract):
            return False
        replaced_invalid = True
    return replaced_invalid


def classify_archive_overlap(
    frames: Iterable[pd.DataFrame],
    logical_key: Iterable[str],
) -> tuple[DataQualityIssue, ...]:
    """Classify raw/canonical partition overlap before last-source-wins resolution.

    Provenance columns are deliberately excluded from value comparison: two
    immutable archives containing the same market row necessarily have different
    archive names/fingerprints, but that does not make the market data conflict.

    A conflicting overlap remains an ERROR unless the last-precedence row is
    value-valid and every earlier row that differs from it is itself invalid under
    the dataset's fixed-kline contract. That narrow case represents an explicit
    repair overlay: a valid daily source can replace a bad monthly row without
    weakening conflict detection for two otherwise-valid but disagreeing sources.
    """
    frames = tuple(frames)
    if len(frames) < 2:
        return ()
    combined = pd.concat(frames, ignore_index=True)
    keys = list(logical_key)
    if not keys or any(key not in combined.columns for key in keys):
        raise ValueError("Archive overlap logical key is missing from source frames")
    duplicates = combined.duplicated(keys, keep=False)
    if not duplicates.any():
        return ()

    overlap = combined.loc[duplicates]
    value_columns = [
        column
        for column in combined.columns
        if column not in keys and column not in _PROVENANCE_COLUMNS
    ]
    contract = _overlap_contract(combined)
    conflicts = 0
    repaired_overrides = 0
    grouped = overlap.groupby(keys, dropna=False, sort=False)
    for _, group in grouped:
        differs = any(
            group[column].nunique(dropna=False) > 1 for column in value_columns
        )
        if not differs:
            continue
        if _is_valid_repair_override(group, value_columns, contract):
            repaired_overrides += 1
        else:
            conflicts += 1
    key_count = int(grouped.ngroups)
    result = [
        DataQualityIssue(
            "ARCHIVE_OVERLAP",
            DataQualityStatus.WARN,
            "Raw archives contain overlapping logical keys",
            key_count,
        )
    ]
    if repaired_overrides:
        result.append(
            DataQualityIssue(
                "INVALID_SOURCE_ROW_OVERRIDDEN",
                DataQualityStatus.WARN,
                "A later overlapping archive replaces an invalid source row with a valid row",
                repaired_overrides,
            )
        )
    if conflicts:
        result.append(
            DataQualityIssue(
                "CONFLICTING_ARCHIVE_OVERLAP",
                DataQualityStatus.ERROR,
                "Overlapping archives disagree for a logical key",
                conflicts,
            )
        )
    elif not repaired_overrides:
        result.append(
            DataQualityIssue(
                "IDENTICAL_ARCHIVE_OVERLAP",
                DataQualityStatus.WARN,
                "Overlapping source rows are identical",
                key_count,
            )
        )
    return tuple(result)


class DataQualityCache:
    """Disposable atomic JSON cache independent of L2/L3 identities."""

    def __init__(self, cache_root: Path):
        self.root = Path(cache_root) / "quality"

    def key(
        self,
        request,
        dataset: DatasetKind,
        interval: str | None,
        required: bool,
        source_identity: str,
    ) -> str:
        payload = {
            "cache_format_version": QUALITY_CACHE_FORMAT_VERSION,
            "validation_contract_version": VALIDATION_CONTRACT_VERSION,
            "dataset": dataset.value,
            "interval": interval,
            "required": bool(required),
            "source_identity": source_identity,
            "request": {
                "exchange": request.exchange,
                "market": request.market.value,
                "symbol": request.symbol,
                "start": _iso(request.start),
                "end": _iso(request.end),
            },
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def get_cached(
        self,
        request,
        dataset: DatasetKind,
        *,
        interval: str | None = None,
        required: bool = True,
        source_identity: str | None,
    ) -> DatasetQualityReport | None:
        # Missing data has no stable canonical identity. Do not persist a
        # DATASET_MISSING result under a permanent "None" key because newly
        # downloaded archives must be observed immediately.
        if not source_identity:
            return None
        key = self.key(request, dataset, interval, required, source_identity)
        path = self.root / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("cache_format_version") != QUALITY_CACHE_FORMAT_VERSION
                or payload.get("validation_contract_version") != VALIDATION_CONTRACT_VERSION
                or payload.get("key") != key
            ):
                return None
            report = DatasetQualityReport.from_dict(payload["report"])
            if report.source_identity != source_identity:
                return None
            return replace(report, cache_hit=True)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, AttributeError):
            return None

    def store(
        self,
        request,
        dataset: DatasetKind,
        report: DatasetQualityReport,
        *,
        interval: str | None = None,
        required: bool = True,
    ) -> None:
        source_identity = report.source_identity
        if not source_identity:
            return
        key = self.key(request, dataset, interval, required, source_identity)
        path = self.root / f"{key}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.{uuid4().hex}.tmp")
        payload = {
            "cache_format_version": QUALITY_CACHE_FORMAT_VERSION,
            "validation_contract_version": VALIDATION_CONTRACT_VERSION,
            "key": key,
            "report": replace(report, cache_hit=False).to_dict(),
        }
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def get_or_validate(
        self,
        frame: pd.DataFrame | None,
        request,
        dataset: DatasetKind,
        *,
        interval: str | None = None,
        required: bool = True,
        source_identity: str | None = None,
        coverage_start: Any | None = None,
        coverage_end: Any | None = None,
        extra_issues: Iterable[DataQualityIssue] = (),
    ) -> DatasetQualityReport:
        source_identity = source_identity or (
            frame.attrs.get("canonical_source_identity") if frame is not None else None
        )
        cached = self.get_cached(
            request,
            dataset,
            interval=interval,
            required=required,
            source_identity=source_identity,
        )
        if cached is not None:
            return cached
        report = validate_dataset(
            frame,
            request,
            dataset,
            interval=interval,
            required=required,
            source_identity=source_identity,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            extra_issues=extra_issues,
        )
        self.store(
            request,
            dataset,
            report,
            interval=interval,
            required=required,
        )
        return report


def validate_feature_timeline(definition, frame: pd.DataFrame, parameters=None) -> None:
    """Enforce generic output ordering and availability without parsing prose rules."""
    definition.validate_output(frame, parameters)
    time_col = "timestamp" if "timestamp" in frame else "date"
    times = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    if times.isna().any() or available.isna().any():
        raise ValueError("Feature timeline contains invalid timestamps")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Feature output timestamps must be ordered and unique")
    if (available < times).any():
        raise ValueError("Feature available_at precedes its logical timestamp")
