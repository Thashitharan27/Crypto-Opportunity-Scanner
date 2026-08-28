"""Bounded, content-addressed compact order-book research snapshots.

Only one immutable archive is materialized at a time. Public ``bookDepth`` is
kept as percentage-distance bands; this module never reconstructs price-level
L2 state and has no simulator dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

from ..progress import emit_progress
from .query import DataRequest
from .schemas import ArchiveRecord, DatasetKind
from .store import DataNotAvailableError, MarketDataStore

BOOK_SNAPSHOT_SCHEMA_VERSION = 2
BOOK_SNAPSHOT_CACHE_FORMAT_VERSION = 1
BOOK_SNAPSHOT_INTERVAL = "1m"
EXPECTED_DEPTH_BANDS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class OrderBookSnapshotResult:
    frame: pd.DataFrame
    source_identity: str
    cache_hit: bool
    partitions_built: int
    partitions_reused: int
    source_event_count: int


class OrderBookSnapshotStore:
    """Build compact completed-minute snapshots from public Binance archives."""

    def __init__(self, store: MarketDataStore) -> None:
        self.store = store
        self.root = Path(store.cache.root) / "order_book"
        if not hasattr(store, "order_book_snapshot_cache_events"):
            store.order_book_snapshot_cache_events = {}

    @staticmethod
    def _validate(dataset: DatasetKind, interval: str) -> None:
        if dataset not in {DatasetKind.BOOK_TICKER, DatasetKind.BOOK_DEPTH}:
            raise ValueError("order-book source must be BOOK_TICKER or BOOK_DEPTH")
        if interval != BOOK_SNAPSHOT_INTERVAL:
            raise ValueError("order-book compact base interval must be 1m")

    @staticmethod
    def _lookback_seconds(value: float | int | None) -> float:
        seconds = 60.0 if value is None else float(value)
        if seconds < 0:
            raise ValueError("order-book lookback must be non-negative")
        # One full minute is always included so an event in the immediately
        # preceding bucket can feed the first decision. Larger staleness limits
        # extend source selection without changing compact partition identity.
        return max(60.0, seconds)

    def _expanded_request(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        lookback_seconds: float | int | None = None,
    ) -> DataRequest:
        seconds = self._lookback_seconds(lookback_seconds)
        return DataRequest(
            request.symbol,
            request.start - pd.Timedelta(seconds=seconds),
            request.end,
            request.strategy_interval,
            market=request.market,
            exchange=request.exchange,
            datasets=(dataset,),
        )

    def _identity(self, record: ArchiveRecord, interval: str) -> str:
        payload = {
            "cache_format_version": BOOK_SNAPSHOT_CACHE_FORMAT_VERSION,
            "schema_version": BOOK_SNAPSHOT_SCHEMA_VERSION,
            "interval": interval,
            "exchange": record.exchange,
            "market": record.market.value,
            "dataset": record.dataset.value,
            "symbol": record.symbol,
            "source_fingerprint": record.fingerprint,
            "adapter_contract": self.store._adapter_for(
                record.dataset
            ).canonical_contract(),
        }
        return sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _paths(
        self, record: ArchiveRecord, interval: str, identity: str
    ) -> tuple[Path, Path]:
        path = (
            self.root
            / record.dataset.value
            / record.symbol
            / interval
            / f"{identity}.parquet"
        )
        return path, path.with_suffix(".json")

    @staticmethod
    def _utc_ns(values) -> pd.Series:
        return pd.to_datetime(values, utc=True, errors="coerce").astype(
            "datetime64[ns, UTC]"
        )

    @classmethod
    def _compact_ticker(
        cls, record: ArchiveRecord, events: pd.DataFrame
    ) -> pd.DataFrame:
        del record
        frame = events.copy()
        frame["event_time"] = cls._utc_ns(frame["event_time"])
        numeric = (
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
        )
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid = frame["event_time"].notna()
        valid &= frame["best_bid_price"].gt(0) & frame["best_ask_price"].gt(0)
        valid &= frame["best_bid_qty"].ge(0) & frame["best_ask_qty"].ge(0)
        valid &= frame["best_bid_price"].le(frame["best_ask_price"])
        frame = frame.loc[valid].sort_values(
            ["event_time", "update_id"], kind="stable"
        )
        frame["period_start"] = frame["event_time"].dt.floor("min")
        frame = frame.groupby("period_start", sort=True).tail(1)
        result = frame[["period_start", "event_time", *numeric]].rename(
            columns={"event_time": "source_event_at"}
        )
        result["period_end"] = result["period_start"] + pd.Timedelta(minutes=1)
        result["available_at"] = result["period_end"]
        result["book_ticker_observed"] = True
        result["book_ticker_covered"] = True
        result["book_ticker_locked"] = result["best_bid_price"].eq(
            result["best_ask_price"]
        )
        return result.reset_index(drop=True)

    @classmethod
    def _compact_depth(
        cls, record: ArchiveRecord, events: pd.DataFrame
    ) -> pd.DataFrame:
        del record
        frame = events.copy()
        frame["event_time"] = cls._utc_ns(frame["event_time"])
        for column in ("percentage", "depth", "notional"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid = (
            frame["event_time"].notna()
            & frame["percentage"].notna()
            & frame["percentage"].ne(0)
        )
        valid &= np.isfinite(frame["depth"]) & frame["depth"].ge(0)
        valid &= np.isfinite(frame["notional"]) & frame["notional"].ge(0)
        frame = frame.loc[valid].sort_values(
            ["event_time", "percentage"], kind="stable"
        )
        rows = []
        for timestamp, group in frame.groupby("event_time", sort=True):
            row = {
                "source_event_at": timestamp,
                "book_depth_snapshot_complete": set(EXPECTED_DEPTH_BANDS).issubset(
                    set(group["percentage"])
                ),
            }
            indexed = group.set_index("percentage")
            for band in range(1, 6):
                for side, percentage in (("bid", -band), ("ask", band)):
                    row[f"book_{side}_depth_{band}pct"] = (
                        indexed.at[percentage, "depth"]
                        if percentage in indexed.index
                        else np.nan
                    )
                    row[f"book_{side}_notional_{band}pct"] = (
                        indexed.at[percentage, "notional"]
                        if percentage in indexed.index
                        else np.nan
                    )
            rows.append(row)
        snapshots = pd.DataFrame(rows)
        if snapshots.empty:
            return snapshots
        snapshots["period_start"] = cls._utc_ns(
            snapshots["source_event_at"]
        ).dt.floor("min")
        snapshots = (
            snapshots.sort_values("source_event_at")
            .groupby("period_start", sort=True)
            .tail(1)
        )
        snapshots["period_end"] = snapshots["period_start"] + pd.Timedelta(
            minutes=1
        )
        snapshots["available_at"] = snapshots["period_end"]
        snapshots["book_depth_observed"] = True
        snapshots["book_depth_covered"] = True
        return snapshots.reset_index(drop=True)

    def _read(
        self, record: ArchiveRecord, interval: str, identity: str
    ) -> tuple[pd.DataFrame, int] | None:
        parquet, manifest = self._paths(record, interval, identity)
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                meta.get("identity") != identity
                or meta.get("schema_version") != BOOK_SNAPSHOT_SCHEMA_VERSION
                or meta.get("cache_format_version")
                != BOOK_SNAPSHOT_CACHE_FORMAT_VERSION
                or meta.get("source_fingerprint") != record.fingerprint
            ):
                return None
            with duckdb.connect() as con:
                frame = con.read_parquet(str(parquet)).df()
            if int(meta.get("row_count", -1)) != len(frame):
                return None
            for column in (
                "period_start",
                "period_end",
                "available_at",
                "source_event_at",
            ):
                if column in frame:
                    frame[column] = self._utc_ns(frame[column])
            return frame, int(meta.get("source_event_count", 0))
        except Exception:
            return None

    @staticmethod
    def _required_snapshot_columns(dataset: DatasetKind) -> set[str]:
        common = {
            "period_start",
            "period_end",
            "available_at",
            "source_event_at",
        }
        if dataset is DatasetKind.BOOK_TICKER:
            return common | {
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "book_ticker_observed",
                "book_ticker_covered",
            }
        return common | {
            "book_depth_snapshot_complete",
            "book_depth_observed",
            "book_depth_covered",
            "book_bid_depth_1pct",
            "book_ask_depth_1pct",
        }

    def _write(
        self,
        record: ArchiveRecord,
        interval: str,
        identity: str,
        frame: pd.DataFrame,
        source_event_count: int,
    ) -> None:
        parquet, manifest = self._paths(record, interval, identity)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}.{uuid4().hex}"
        tmp = parquet.with_suffix(f".{token}.tmp.parquet")
        tmp_manifest = manifest.with_suffix(f".{token}.tmp.json")
        try:
            with duckdb.connect() as con:
                con.register("snapshot", frame)
                escaped = str(tmp).replace("'", "''")
                con.execute(
                    f"COPY snapshot TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                check = con.read_parquet(str(tmp)).df()
            missing = self._required_snapshot_columns(record.dataset) - set(check.columns)
            if missing or len(check) != len(frame):
                raise ValueError(
                    "order-book snapshot cache validation failed: "
                    f"missing={sorted(missing)} rows={len(check)}/{len(frame)}"
                )
            metadata = {
                "identity": identity,
                "schema_version": BOOK_SNAPSHOT_SCHEMA_VERSION,
                "cache_format_version": BOOK_SNAPSHOT_CACHE_FORMAT_VERSION,
                "source_fingerprint": record.fingerprint,
                "row_count": len(frame),
                "source_event_count": int(source_event_count),
            }
            tmp_manifest.write_text(
                json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            if json.loads(tmp_manifest.read_text(encoding="utf-8")) != metadata:
                raise ValueError("order-book snapshot manifest validation failed")
            os.replace(tmp, parquet)
            os.replace(tmp_manifest, manifest)
        finally:
            tmp.unlink(missing_ok=True)
            tmp_manifest.unlink(missing_ok=True)

    @staticmethod
    def _combined(dataset: DatasetKind, identities) -> str:
        digest = sha256(json.dumps(sorted(identities)).encode()).hexdigest()
        return (
            f"order-book-snapshot-v{BOOK_SNAPSHOT_SCHEMA_VERSION}:"
            f"{dataset.value}:{len(identities)}:{digest}"
        )

    def source_identity(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        interval: str = "1m",
        lookback_seconds: float | int | None = None,
    ) -> str:
        self._validate(dataset, interval)
        expanded = self._expanded_request(
            request, dataset, lookback_seconds=lookback_seconds
        )
        records = self.store.catalog.records_for(
            self.store.raw_root, expanded, dataset, None
        )
        if not records:
            raise DataNotAvailableError(f"No {dataset.value} coverage")
        return self._combined(
            dataset, [self._identity(record, interval) for record in records]
        )

    def load(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        interval: str = "1m",
        lookback_seconds: float | int | None = None,
    ) -> OrderBookSnapshotResult:
        self._validate(dataset, interval)
        expanded = self._expanded_request(
            request, dataset, lookback_seconds=lookback_seconds
        )
        records = self.store.catalog.records_for(
            self.store.raw_root, expanded, dataset, None
        )
        if not records:
            raise DataNotAvailableError(f"No {dataset.value} coverage")

        progress = getattr(self.store, "progress_callback", None)
        progress_started = time.perf_counter()
        total = len(records)
        label = (
            "Order Book — Ticker"
            if dataset is DatasetKind.BOOK_TICKER
            else "Order Book — Depth"
        )
        emit_progress(
            progress,
            kind="cache",
            phase="order_book_cache",
            label=label,
            completed=0,
            total=total,
            built=0,
            reused=0,
            elapsed_seconds=0.0,
            current="Checking compact 1-minute partitions",
        )

        frames: list[pd.DataFrame] = []
        identities: list[str] = []
        built = reused = event_count = 0
        for index, record in enumerate(records, 1):
            identity = self._identity(record, interval)
            identities.append(identity)
            cached = self._read(record, interval, identity)
            if cached is None:
                events = self.store._adapter_for(dataset).read(record)
                source_events = len(events)
                event_count += source_events
                compact = (
                    self._compact_ticker(record, events)
                    if dataset is DatasetKind.BOOK_TICKER
                    else self._compact_depth(record, events)
                )
                if not compact.empty:
                    self._write(
                        record,
                        interval,
                        identity,
                        compact,
                        source_event_count=source_events,
                    )
                built += 1
                action = "Built missing partition"
            else:
                compact, source_events = cached
                event_count += source_events
                reused += 1
                action = "Reused cached partition"
            if not compact.empty:
                frames.append(compact)
            emit_progress(
                progress,
                kind="cache",
                phase="order_book_cache",
                label=label,
                completed=index,
                total=total,
                built=built,
                reused=reused,
                elapsed_seconds=time.perf_counter() - progress_started,
                current=action,
            )
        if not frames:
            raise DataNotAvailableError(f"No valid {dataset.value} observations")

        combined = pd.concat(frames, ignore_index=True).sort_values(
            "period_start", kind="stable"
        )
        combined = combined.drop_duplicates("period_start", keep="last")
        grid = pd.date_range(
            pd.Timestamp(expanded.start).floor("min"),
            pd.Timestamp(request.end).ceil("min"),
            freq="1min",
            inclusive="left",
        )
        combined = (
            combined.set_index("period_start")
            .reindex(grid)
            .reset_index(names="period_start")
        )
        prefix = (
            "book_ticker"
            if dataset is DatasetKind.BOOK_TICKER
            else "book_depth"
        )
        catalog_covered = pd.Series(False, index=combined.index)
        for record in records:
            if record.period_start is None or record.period_end is None:
                catalog_covered[:] = True
            else:
                catalog_covered |= (
                    combined["period_start"]
                    >= pd.Timestamp(record.period_start).floor("min")
                ) & (
                    combined["period_start"]
                    < pd.Timestamp(record.period_end).ceil("min")
                )
        combined[f"{prefix}_covered"] = combined[f"{prefix}_covered"].astype(
            "boolean"
        ).fillna(catalog_covered).astype(bool)
        combined[f"{prefix}_observed"] = combined[
            f"{prefix}_observed"
        ].astype("boolean").fillna(False).astype(bool)
        combined["period_end"] = combined["period_start"] + pd.Timedelta(minutes=1)
        combined["available_at"] = combined["period_end"]

        telemetry = self.store.order_book_snapshot_cache_events.setdefault(
            dataset.value,
            {"partitions_built": 0, "partitions_reused": 0},
        )
        telemetry["partitions_built"] += int(built)
        telemetry["partitions_reused"] += int(reused)

        return OrderBookSnapshotResult(
            combined,
            self._combined(dataset, identities),
            built == 0,
            built,
            reused,
            event_count,
        )

    def quality_report(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        required: bool = False,
    ):
        """Validate raw order-book sources one archive at a time.

        This reuses Task-12 contracts and DataQualityCache while avoiding the
        generic multi-partition ``MarketDataStore.load_dataset`` concatenation.
        Warm calls are metadata-only, and cold calls never retain more than one
        source partition plus rare overlap participants at a time.
        """
        from .quality import (
            DataQualityCache,
            DataQualityIssue,
            DataQualityStatus,
            DatasetQualityReport,
            validate_dataset,
        )

        self._validate(dataset, BOOK_SNAPSHOT_INTERVAL)
        try:
            source_identity = self.store.canonical_source_identity(
                request, dataset
            ).cache_identity()
        except DataNotAvailableError:
            return validate_dataset(None, request, dataset, required=required)

        cache = DataQualityCache(self.store.cache.root)
        cached = cache.get_cached(
            request,
            dataset,
            required=required,
            source_identity=source_identity,
        )
        if cached is not None:
            return cached

        records = self.store.catalog.records_for(
            self.store.raw_root, request, dataset, None
        )
        progress = getattr(self.store, "progress_callback", None)
        progress_started = time.perf_counter()
        progress_label = (
            "Order Book Quality — Ticker"
            if dataset is DatasetKind.BOOK_TICKER
            else "Order Book Quality — Depth"
        )
        emit_progress(
            progress,
            kind="cache",
            mode="validation",
            phase="order_book_quality",
            label=progress_label,
            completed=0,
            total=len(records),
            elapsed_seconds=0.0,
            current="Cold quality check; later identical runs reuse this result",
        )
        coverage = self.store.catalog.coverage(
            self.store.raw_root,
            market=request.market,
            dataset=dataset,
            symbol=request.symbol,
        )
        issues: list[DataQualityIssue] = []
        row_count = 0
        observed_start = None
        observed_end = None
        for index, record in enumerate(records, 1):
            emit_progress(
                progress,
                kind="cache",
                mode="validation",
                phase="order_book_quality",
                label=progress_label,
                completed=index - 1,
                total=len(records),
                elapsed_seconds=time.perf_counter() - progress_started,
                current=f"Validating source partition {index} of {len(records)}",
            )
            local_start = max(
                pd.Timestamp(request.start),
                pd.Timestamp(record.period_start)
                if record.period_start is not None
                else pd.Timestamp(request.start),
            )
            local_end = min(
                pd.Timestamp(request.end),
                pd.Timestamp(record.period_end)
                if record.period_end is not None
                else pd.Timestamp(request.end),
            )
            if local_start >= local_end:
                continue
            local_request = DataRequest(
                request.symbol,
                local_start.to_pydatetime(),
                local_end.to_pydatetime(),
                request.strategy_interval,
                market=request.market,
                exchange=request.exchange,
                datasets=(dataset,),
            )
            try:
                frame = self.store._adapter_for(dataset).read(record)
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(
                    DataQualityIssue(
                        "SOURCE_ADAPTER_ERROR",
                        DataQualityStatus.ERROR,
                        f"Order-book source partition failed canonical validation: {exc}",
                        details={"source_archive": str(record.path)},
                    )
                )
                continue
            if not frame.empty:
                timeline = pd.to_datetime(
                    frame["event_time"], utc=True, errors="coerce"
                )
                mask = (timeline >= local_start) & (timeline < local_end)
                frame = frame.loc[mask].copy()
                if not frame.empty:
                    frame["event_time"] = timeline.loc[mask]
                    row_count += len(frame)
                    first = frame["event_time"].min()
                    last = frame["event_time"].max()
                    observed_start = first if observed_start is None else min(
                        observed_start, first
                    )
                    observed_end = last if observed_end is None else max(
                        observed_end, last
                    )
            if frame.empty:
                continue
            partition_report = validate_dataset(
                frame,
                local_request,
                dataset,
                required=False,
                source_identity=f"partition:{self._identity(record, BOOK_SNAPSHOT_INTERVAL)}",
                coverage_start=local_start,
                coverage_end=local_end,
            )
            issues.extend(partition_report.issues)

        emit_progress(
            progress,
            kind="cache",
            mode="validation",
            phase="order_book_quality",
            label=progress_label,
            completed=len(records),
            total=len(records),
            elapsed_seconds=time.perf_counter() - progress_started,
            current="Quality validation cached",
        )
        issues.extend(self.store._archive_overlap_issues(request, dataset))

        complete_start = pd.Timestamp(request.start)
        complete_end = pd.Timestamp(request.end)
        if coverage.first_period is not None and pd.Timestamp(
            coverage.first_period
        ) > pd.Timestamp(request.start):
            severity = DataQualityStatus.ERROR if required else DataQualityStatus.WARN
            issues.append(
                DataQualityIssue(
                    "LEADING_SOURCE_COVERAGE_GAP",
                    severity,
                    "Catalog source coverage begins after the requested start",
                    details={
                        "coverage_start": pd.Timestamp(coverage.first_period).isoformat()
                    },
                )
            )
            complete_start = pd.Timestamp(coverage.first_period)
        if coverage.last_period is not None and pd.Timestamp(
            coverage.last_period
        ) < pd.Timestamp(request.end):
            severity = DataQualityStatus.ERROR if required else DataQualityStatus.WARN
            issues.append(
                DataQualityIssue(
                    "TRAILING_SOURCE_COVERAGE_GAP",
                    severity,
                    "Catalog source coverage ends before the requested end",
                    details={
                        "coverage_end": pd.Timestamp(coverage.last_period).isoformat()
                    },
                )
            )
            complete_end = pd.Timestamp(coverage.last_period)

        has_error = any(
            issue.severity is DataQualityStatus.ERROR for issue in issues
        )
        status = (
            DataQualityStatus.ERROR
            if required and has_error
            else DataQualityStatus.WARN
            if issues
            else DataQualityStatus.OK
        )
        report = DatasetQualityReport(
            dataset=dataset.value,
            symbol=request.symbol,
            interval=None,
            required=required,
            requested_start=pd.Timestamp(request.start).isoformat(),
            requested_end=pd.Timestamp(request.end).isoformat(),
            observed_start=(
                pd.Timestamp(observed_start).isoformat()
                if observed_start is not None
                else None
            ),
            observed_end=(
                pd.Timestamp(observed_end).isoformat()
                if observed_end is not None
                else None
            ),
            complete_start=complete_start.isoformat(),
            complete_end=complete_end.isoformat(),
            row_count=row_count,
            source_identity=source_identity,
            status=status,
            issues=tuple(issues),
        )
        cache.store(request, dataset, report, required=required)
        return report
