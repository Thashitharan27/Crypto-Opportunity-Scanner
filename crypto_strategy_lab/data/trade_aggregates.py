"""Partitioned, bounded aggregation of Binance trade-event archives.

Raw aggTrades/trades are transient inputs only.  This module processes one
catalog partition at a time and persists compact completed 1-minute aggregates
for FeatureRegistry consumers.  It deliberately has no dependency on strategy,
simulation, GUI, or reporting code.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

from ..progress import emit_progress
from .query import DataRequest
from .schemas import ArchiveRecord, DatasetKind
from .store import DataNotAvailableError, MarketDataStore


TRADE_AGGREGATE_SCHEMA_VERSION = 1
TRADE_AGGREGATE_CACHE_FORMAT_VERSION = 1
TRADE_AGGREGATE_INTERVAL = "1m"

_ADDITIVE_COLUMNS = (
    "source_event_count",
    "underlying_trade_count",
    "base_volume",
    "quote_volume",
    "aggressive_buy_base_volume",
    "aggressive_sell_base_volume",
    "aggressive_buy_quote_volume",
    "aggressive_sell_quote_volume",
    "trade_delta_base",
    "trade_delta_quote",
    "weighted_price_sum",
)
_LARGE_COLUMNS = (
    "large_source_event_count",
    "large_source_event_quote_volume",
    "large_buy_quote_volume",
    "large_sell_quote_volume",
)
_REQUIRED_COLUMNS = (
    "period_start",
    "period_end",
    "available_at",
    "trade_flow_source_covered",
    *_ADDITIVE_COLUMNS,
    *_LARGE_COLUMNS,
    "median_source_event_size",
    "last_event_at",
)


@dataclass(frozen=True, slots=True)
class TradeAggregateResult:
    """Compact aggregate frame plus cache/provenance observability."""

    frame: pd.DataFrame
    source_identity: str
    cache_hit: bool
    partitions_built: int
    partitions_reused: int
    source_event_count: int


class TradeAggregateStore:
    """Build/read content-addressed 1m aggregates one source partition at a time."""

    def __init__(self, store: MarketDataStore) -> None:
        self.store = store
        self.root = Path(store.cache.root) / "trade_aggregates"

    @staticmethod
    def _validate_source(dataset: DatasetKind) -> None:
        if dataset not in {DatasetKind.AGG_TRADES, DatasetKind.TRADES}:
            raise ValueError("trade aggregate source must be agg_trades or trades")

    @staticmethod
    def _expanded_request(request: DataRequest, dataset: DatasetKind) -> DataRequest:
        # A UTC-day prefix makes cvd_utc_day independent of the requested start.
        # One prior hour is also needed for 1h/intensity windows at a midnight start.
        start = pd.Timestamp(request.start)
        day_start = start.floor("D")
        window_start = start - pd.Timedelta(hours=1)
        expanded_start = min(day_start, window_start).to_pydatetime()
        return DataRequest(
            symbol=request.symbol,
            start=expanded_start,
            end=request.end,
            strategy_interval=request.strategy_interval,
            market=request.market,
            exchange=request.exchange,
            datasets=(dataset,),
        )

    def _partition_identity(
        self,
        record: ArchiveRecord,
        dataset: DatasetKind,
        large_trade_quote_threshold: float | None,
    ) -> str:
        adapter = self.store._adapter_for(dataset)
        payload = {
            "cache_format_version": TRADE_AGGREGATE_CACHE_FORMAT_VERSION,
            "aggregate_schema_version": TRADE_AGGREGATE_SCHEMA_VERSION,
            "aggregate_interval": TRADE_AGGREGATE_INTERVAL,
            "exchange": record.exchange,
            "market": record.market.value,
            "dataset": dataset.value,
            "symbol": record.symbol,
            "period_start": record.period_start,
            "period_end": record.period_end,
            "source_fingerprint": record.fingerprint,
            "adapter_contract": adapter.canonical_contract(),
            "large_trade_quote_threshold": large_trade_quote_threshold,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _partition_paths(
        self, dataset: DatasetKind, symbol: str, identity: str
    ) -> tuple[Path, Path]:
        directory = self.root / dataset.value / symbol / TRADE_AGGREGATE_INTERVAL
        parquet = directory / f"{identity}.parquet"
        return parquet, parquet.with_suffix(".json")

    @staticmethod
    def _normalize_loaded(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in ("period_start", "period_end", "available_at", "last_event_at"):
            if column in result:
                result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
        if "trade_flow_source_covered" in result:
            result["trade_flow_source_covered"] = result["trade_flow_source_covered"].astype(bool)
        return result

    @classmethod
    def _validate_aggregate_frame(cls, frame: pd.DataFrame) -> None:
        missing = sorted(set(_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Trade aggregate cache missing columns: {missing}")
        if frame.empty:
            return
        starts = pd.to_datetime(frame["period_start"], utc=True, errors="coerce")
        ends = pd.to_datetime(frame["period_end"], utc=True, errors="coerce")
        available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        if starts.isna().any() or ends.isna().any() or available.isna().any():
            raise ValueError("Trade aggregate cache contains malformed timestamps")
        if not starts.is_monotonic_increasing or starts.duplicated().any():
            raise ValueError("Trade aggregate cache timeline must be ordered and unique")
        one_minute = pd.Timedelta(minutes=1)
        if not ((ends - starts) == one_minute).all() or not (available == ends).all():
            raise ValueError("Trade aggregate buckets must be completed 1m intervals")
        covered = frame["trade_flow_source_covered"].astype(bool)
        for column in ("source_event_count", "underlying_trade_count", "base_volume", "quote_volume"):
            values = pd.to_numeric(frame.loc[covered, column], errors="coerce")
            if values.isna().any() or (values < 0).any():
                raise ValueError(f"Invalid covered trade aggregate values in {column}")

    def _read_cached_partition(
        self,
        record: ArchiveRecord,
        dataset: DatasetKind,
        identity: str,
        large_trade_quote_threshold: float | None,
    ) -> pd.DataFrame | None:
        parquet, manifest = self._partition_paths(dataset, record.symbol, identity)
        if not parquet.is_file() or not manifest.is_file():
            return None
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                metadata.get("cache_format_version") != TRADE_AGGREGATE_CACHE_FORMAT_VERSION
                or metadata.get("aggregate_schema_version") != TRADE_AGGREGATE_SCHEMA_VERSION
                or metadata.get("identity") != identity
                or metadata.get("source_fingerprint") != record.fingerprint
                or metadata.get("source_dataset") != dataset.value
                or metadata.get("large_trade_quote_threshold") != large_trade_quote_threshold
            ):
                return None
            with duckdb.connect() as con:
                frame = con.read_parquet(str(parquet)).df()
            frame = self._normalize_loaded(frame)
            self._validate_aggregate_frame(frame)
            if int(metadata.get("row_count", -1)) != len(frame):
                return None
            return frame
        except Exception:
            return None

    def _write_partition(
        self,
        record: ArchiveRecord,
        dataset: DatasetKind,
        identity: str,
        frame: pd.DataFrame,
        large_trade_quote_threshold: float | None,
    ) -> None:
        self._validate_aggregate_frame(frame)
        parquet, manifest = self._partition_paths(dataset, record.symbol, identity)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}.{uuid4().hex}"
        temporary = parquet.with_suffix(f".{token}.tmp.parquet")
        temporary_manifest = manifest.with_suffix(f".{token}.tmp.json")
        for path in (temporary, temporary_manifest):
            path.unlink(missing_ok=True)
        with duckdb.connect() as con:
            con.register("trade_aggregate", frame)
            escaped = str(temporary).replace("'", "''")
            con.execute(
                f"COPY trade_aggregate TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        # Read the temporary artifact before publishing it; corrupt/partial writes
        # are never allowed to become cache hits.
        with duckdb.connect() as con:
            verified = self._normalize_loaded(con.read_parquet(str(temporary)).df())
        self._validate_aggregate_frame(verified)
        metadata = {
            "cache_format_version": TRADE_AGGREGATE_CACHE_FORMAT_VERSION,
            "aggregate_schema_version": TRADE_AGGREGATE_SCHEMA_VERSION,
            "aggregate_interval": TRADE_AGGREGATE_INTERVAL,
            "identity": identity,
            "source_dataset": dataset.value,
            "source_fingerprint": record.fingerprint,
            "source_archive": str(record.path),
            "source_period_start": str(record.period_start) if record.period_start else None,
            "source_period_end": str(record.period_end) if record.period_end else None,
            "large_trade_quote_threshold": large_trade_quote_threshold,
            "row_count": len(frame),
            "source_event_count": int(pd.to_numeric(frame["source_event_count"], errors="coerce").sum()),
        }
        temporary_manifest.write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(parquet)
        temporary_manifest.replace(manifest)

    @staticmethod
    def _aggregate(
        record: ArchiveRecord,
        events: pd.DataFrame,
        large_trade_quote_threshold: float | None,
    ) -> pd.DataFrame:
        """Aggregate one immutable source archive into completed UTC minute buckets."""
        if large_trade_quote_threshold is not None and large_trade_quote_threshold <= 0:
            raise ValueError("large_trade_quote_threshold must be positive or null")
        dataset = record.dataset
        if dataset not in {DatasetKind.AGG_TRADES, DatasetKind.TRADES}:
            raise ValueError("Unsupported trade aggregate source")

        frame = events.copy()
        required = {"event_time", "price", "quantity", "is_buyer_maker"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Trade event frame missing columns: {missing}")

        if frame.empty:
            event_time = pd.Series([], dtype="datetime64[ns, UTC]")
        else:
            event_time = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
            frame["event_time"] = event_time
            frame = frame.sort_values("event_time", kind="stable").reset_index(drop=True)
            event_time = frame["event_time"]

        if record.period_start is not None and record.period_end is not None:
            grid_start = pd.Timestamp(record.period_start).floor("min")
            grid_end = pd.Timestamp(record.period_end).ceil("min")
        elif len(frame):
            grid_start = pd.Timestamp(event_time.min()).floor("min")
            grid_end = pd.Timestamp(event_time.max()).floor("min") + pd.Timedelta(minutes=1)
        else:
            return pd.DataFrame(columns=_REQUIRED_COLUMNS)

        grid = pd.date_range(grid_start, grid_end, freq="1min", inclusive="left")
        if not len(grid):
            return pd.DataFrame(columns=_REQUIRED_COLUMNS)

        if len(frame):
            in_span = (frame["event_time"] >= grid_start) & (frame["event_time"] < grid_end)
            frame = frame.loc[in_span].copy()

        if frame.empty:
            grouped = pd.DataFrame(index=grid)
        else:
            price = pd.to_numeric(frame["price"], errors="raise").astype(float)
            quantity = pd.to_numeric(frame["quantity"], errors="raise").astype(float)
            quote = (
                pd.to_numeric(frame["quote_quantity"], errors="coerce").astype(float)
                if "quote_quantity" in frame
                else price * quantity
            )
            if quote.isna().any():
                quote = quote.fillna(price * quantity)
            maker = frame["is_buyer_maker"].astype(bool)
            if dataset is DatasetKind.AGG_TRADES:
                if not {"first_trade_id", "last_trade_id"}.issubset(frame.columns):
                    raise ValueError("aggTrades aggregation requires first_trade_id and last_trade_id")
                first_id = pd.to_numeric(frame["first_trade_id"], errors="raise").astype("int64")
                last_id = pd.to_numeric(frame["last_trade_id"], errors="raise").astype("int64")
                underlying = last_id - first_id + 1
                if (underlying <= 0).any():
                    raise ValueError("aggTrades contains invalid underlying trade ID range")
            else:
                underlying = pd.Series(np.ones(len(frame), dtype=np.int64), index=frame.index)

            bucket = frame["event_time"].dt.floor("min")
            work = pd.DataFrame(
                {
                    "bucket": bucket,
                    "source_event_count": 1,
                    "underlying_trade_count": underlying.to_numpy(np.int64),
                    "base_volume": quantity,
                    "quote_volume": quote,
                    "aggressive_buy_base_volume": np.where(~maker, quantity, 0.0),
                    "aggressive_sell_base_volume": np.where(maker, quantity, 0.0),
                    "aggressive_buy_quote_volume": np.where(~maker, quote, 0.0),
                    "aggressive_sell_quote_volume": np.where(maker, quote, 0.0),
                    "weighted_price_sum": price * quantity,
                    "last_event_at": frame["event_time"],
                    "source_event_size": quantity,
                }
            )
            if large_trade_quote_threshold is None:
                work["large_source_event_count"] = np.nan
                work["large_source_event_quote_volume"] = np.nan
                work["large_buy_quote_volume"] = np.nan
                work["large_sell_quote_volume"] = np.nan
            else:
                large = quote >= float(large_trade_quote_threshold)
                work["large_source_event_count"] = large.astype(np.int64)
                work["large_source_event_quote_volume"] = np.where(large, quote, 0.0)
                work["large_buy_quote_volume"] = np.where(large & ~maker, quote, 0.0)
                work["large_sell_quote_volume"] = np.where(large & maker, quote, 0.0)

            aggregation: dict[str, Any] = {
                column: (column, "sum")
                for column in (
                    "source_event_count",
                    "underlying_trade_count",
                    "base_volume",
                    "quote_volume",
                    "aggressive_buy_base_volume",
                    "aggressive_sell_base_volume",
                    "aggressive_buy_quote_volume",
                    "aggressive_sell_quote_volume",
                    "weighted_price_sum",
                    *_LARGE_COLUMNS,
                )
            }
            aggregation["median_source_event_size"] = ("source_event_size", "median")
            aggregation["last_event_at"] = ("last_event_at", "max")
            grouped = work.groupby("bucket", sort=True).agg(**aggregation)

        grouped = grouped.reindex(grid)
        if large_trade_quote_threshold is None:
            fill_columns = tuple(column for column in _ADDITIVE_COLUMNS if not column.startswith("trade_delta"))
        else:
            fill_columns = (
                *(column for column in _ADDITIVE_COLUMNS if not column.startswith("trade_delta")),
                *_LARGE_COLUMNS,
            )
        for column in fill_columns:
            if column not in grouped:
                grouped[column] = 0.0
            grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0.0)
        if large_trade_quote_threshold is None:
            for column in _LARGE_COLUMNS:
                grouped[column] = np.nan
        if "median_source_event_size" not in grouped:
            grouped["median_source_event_size"] = np.nan
        if "last_event_at" not in grouped:
            grouped["last_event_at"] = pd.NaT

        grouped["trade_delta_base"] = (
            grouped["aggressive_buy_base_volume"] - grouped["aggressive_sell_base_volume"]
        )
        grouped["trade_delta_quote"] = (
            grouped["aggressive_buy_quote_volume"] - grouped["aggressive_sell_quote_volume"]
        )
        result = grouped.reset_index(names="period_start")
        result["period_end"] = result["period_start"] + pd.Timedelta(minutes=1)
        result["available_at"] = result["period_end"]
        result["trade_flow_source_covered"] = True
        result["last_event_at"] = pd.to_datetime(result["last_event_at"], utc=True, errors="coerce")
        return result.loc[:, list(_REQUIRED_COLUMNS)].reset_index(drop=True)

    @staticmethod
    def _combined_identity(dataset: DatasetKind, partition_identities: list[str]) -> str:
        encoded = json.dumps(
            {
                "schema_version": TRADE_AGGREGATE_SCHEMA_VERSION,
                "dataset": dataset.value,
                "partitions": sorted(partition_identities),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = sha256(encoded).hexdigest()
        return f"trade-aggregate-v{TRADE_AGGREGATE_SCHEMA_VERSION}:{dataset.value}:{len(partition_identities)}:{digest}"

    def source_identity(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        large_trade_quote_threshold: float | None = None,
    ) -> str:
        """Return aggregate-resource identity using catalog metadata only."""
        self._validate_source(dataset)
        expanded = self._expanded_request(request, dataset)
        records = self.store.catalog.records_for(self.store.raw_root, expanded, dataset, None)
        if not records:
            raise DataNotAvailableError(
                f"No catalog coverage for {request.symbol} {dataset.value} trade-flow research"
            )
        identities = [
            self._partition_identity(record, dataset, large_trade_quote_threshold)
            for record in records
        ]
        return self._combined_identity(dataset, identities)

    def load(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        large_trade_quote_threshold: float | None = None,
    ) -> TradeAggregateResult:
        """Return a compact contiguous minute grid, never a concatenated raw-event frame."""
        self._validate_source(dataset)
        if large_trade_quote_threshold is not None and large_trade_quote_threshold <= 0:
            raise ValueError("large_trade_quote_threshold must be positive or null")
        expanded = self._expanded_request(request, dataset)
        records = self.store.catalog.records_for(self.store.raw_root, expanded, dataset, None)
        if not records:
            raise DataNotAvailableError(
                f"No catalog coverage for {request.symbol} {dataset.value} trade-flow research"
            )

        progress = getattr(self.store, "progress_callback", None)
        progress_started = time.perf_counter()
        total = len(records)
        label = "Trade Flow — Aggregate Trades" if dataset is DatasetKind.AGG_TRADES else "Trade Flow — Trades"
        emit_progress(
            progress,
            kind="cache",
            phase="trade_flow_cache",
            label=label,
            completed=0,
            total=total,
            built=0,
            reused=0,
            elapsed_seconds=0.0,
            current="Checking compact 1-minute partitions",
        )

        partition_frames: list[pd.DataFrame] = []
        partition_identities: list[str] = []
        built = 0
        reused = 0
        adapter = self.store._adapter_for(dataset)
        for index, record in enumerate(records, 1):
            identity = self._partition_identity(record, dataset, large_trade_quote_threshold)
            partition_identities.append(identity)
            aggregate = self._read_cached_partition(
                record, dataset, identity, large_trade_quote_threshold
            )
            if aggregate is None:
                # Important memory boundary: one raw archive is normalized,
                # aggregated, released, and cached before the next is opened.
                events = adapter.read(record)
                aggregate = self._aggregate(record, events, large_trade_quote_threshold)
                del events
                self._write_partition(
                    record,
                    dataset,
                    identity,
                    aggregate,
                    large_trade_quote_threshold,
                )
                built += 1
                action = "Built missing partition"
            else:
                reused += 1
                action = "Reused cached partition"
            partition_frames.append(aggregate)
            emit_progress(
                progress,
                kind="cache",
                phase="trade_flow_cache",
                label=label,
                completed=index,
                total=total,
                built=built,
                reused=reused,
                elapsed_seconds=time.perf_counter() - progress_started,
                current=action,
            )

        combined = pd.concat(partition_frames, ignore_index=True) if partition_frames else pd.DataFrame()
        if combined.empty:
            raise DataNotAvailableError(
                f"No aggregate rows were produced for {request.symbol} {dataset.value}"
            )
        combined = self._normalize_loaded(combined)

        # Resolve normal daily/monthly archive overlap at the compact minute
        # boundary.  Conflicting covered minute aggregates are a source-integrity
        # error rather than something to sum twice or silently average.
        duplicate = combined["period_start"].duplicated(keep=False)
        if duplicate.any():
            overlap = combined.loc[duplicate]
            compare_columns = [
                column for column in (*_ADDITIVE_COLUMNS, *_LARGE_COLUMNS, "median_source_event_size", "last_event_at")
                if column in overlap
            ]
            for _, group in overlap.groupby("period_start", sort=False):
                if any(group[column].nunique(dropna=False) > 1 for column in compare_columns):
                    raise ValueError("Conflicting overlapping trade aggregate source partitions")
        combined = combined.sort_values("period_start", kind="stable").drop_duplicates(
            "period_start", keep="last"
        )

        start = pd.Timestamp(expanded.start).floor("min")
        end = pd.Timestamp(expanded.end).ceil("min")
        grid = pd.date_range(start, end, freq="1min", inclusive="left")
        combined = combined.set_index("period_start").reindex(grid)
        covered = combined["trade_flow_source_covered"].fillna(False).astype(bool)
        combined["trade_flow_source_covered"] = covered
        combined["period_end"] = grid + pd.Timedelta(minutes=1)
        combined["available_at"] = combined["period_end"]
        for column in (*_ADDITIVE_COLUMNS, *_LARGE_COLUMNS, "median_source_event_size"):
            if column not in combined:
                combined[column] = np.nan
            combined.loc[~covered, column] = np.nan
        if "last_event_at" not in combined:
            combined["last_event_at"] = pd.NaT
        combined.loc[~covered, "last_event_at"] = pd.NaT
        combined = combined.reset_index(names="period_start")
        combined["last_event_at"] = pd.to_datetime(combined["last_event_at"], utc=True, errors="coerce")
        combined = combined.loc[:, list(_REQUIRED_COLUMNS)].reset_index(drop=True)
        self._validate_aggregate_frame(combined)

        return TradeAggregateResult(
            frame=combined,
            source_identity=self._combined_identity(dataset, partition_identities),
            cache_hit=built == 0,
            partitions_built=built,
            partitions_reused=reused,
            source_event_count=int(pd.to_numeric(combined["source_event_count"], errors="coerce").sum()),
        )
