"""Single entry point for market data used by the future backtest pipeline."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
import json

import duckdb
import pandas as pd

from .binance.discovery import discover_archives
from .binance.events import (
    BookDepthArchiveAdapter,
    BookTickerArchiveAdapter,
    FundingRateArchiveAdapter,
    FuturesMetricsArchiveAdapter,
)
from .binance.klines import KlineArchiveAdapter, KlineLikeArchiveAdapter
from .binance.trades import AggTradesArchiveAdapter, TradesArchiveAdapter
from .cache import CANONICAL_CACHE_FORMAT_VERSION, CacheLayout
from .catalog import DataCatalog
from .query import DataRequest
from .schemas import ArchiveRecord, DatasetKind
from .source_identity import SourceSignature, canonical_partition_identity


class DataNotAvailableError(RuntimeError):
    """Raised when the catalog has no source coverage for a requested slice."""


class MarketDataStore:
    """Read-only view of raw Binance data plus disposable canonical caches."""

    def __init__(self, raw_root: Path, cache_root: Path = Path("cache")) -> None:
        self.raw_root = Path(raw_root)
        self.cache = CacheLayout(Path(cache_root))
        self.cache.ensure()
        self.catalog = DataCatalog(self.cache.catalog_db)
        self.canonical_cache_events = {"hit": 0, "miss": 0}
        self._adapters = {
            DatasetKind.KLINES: KlineArchiveAdapter(),
            DatasetKind.MARK_PRICE_KLINES: KlineLikeArchiveAdapter(
                DatasetKind.MARK_PRICE_KLINES
            ),
            DatasetKind.INDEX_PRICE_KLINES: KlineLikeArchiveAdapter(
                DatasetKind.INDEX_PRICE_KLINES
            ),
            DatasetKind.PREMIUM_INDEX_KLINES: KlineLikeArchiveAdapter(
                DatasetKind.PREMIUM_INDEX_KLINES
            ),
            DatasetKind.FUTURES_METRICS: FuturesMetricsArchiveAdapter(),
            DatasetKind.FUNDING_RATE: FundingRateArchiveAdapter(),
            DatasetKind.AGG_TRADES: AggTradesArchiveAdapter(),
            DatasetKind.TRADES: TradesArchiveAdapter(),
            DatasetKind.BOOK_TICKER: BookTickerArchiveAdapter(),
            DatasetKind.BOOK_DEPTH: BookDepthArchiveAdapter(),
        }

    def refresh_catalog(self) -> int:
        records = discover_archives(self.raw_root)
        self.catalog.sync_root(self.raw_root, records)
        return len(records)

    def _adapter_for(self, dataset: DatasetKind):
        try:
            return self._adapters[dataset]
        except KeyError as exc:
            raise NotImplementedError(
                f"Canonical adapter for {dataset.value!r} is registered in the architecture "
                "but has not been migrated yet"
            ) from exc

    def _ensure_canonical(self, record: ArchiveRecord) -> Path:
        adapter = self._adapter_for(record.dataset)
        contract = adapter.canonical_contract()
        identity = canonical_partition_identity(record, contract)
        target = self.cache.archive_parquet(record, identity)
        manifest = self.cache.archive_manifest(record, identity)
        if target.is_file() and manifest.is_file():
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                if (
                    metadata.get("cache_format_version") == CANONICAL_CACHE_FORMAT_VERSION
                    and metadata.get("canonical_identity") == identity
                    and metadata.get("raw_source_fingerprint") == record.fingerprint
                    and metadata.get("contract") == contract
                ):
                    with duckdb.connect() as con:
                        con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(target)])
                    self.canonical_cache_events["hit"] += 1
                    return target
            except Exception:
                pass
        self.canonical_cache_events["miss"] += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = adapter.read(record)
        if frame.empty:
            raise ValueError(f"Recognized Binance archive contains no rows: {record.path}")
        temporary = target.with_suffix(".tmp.parquet")
        temporary_manifest = manifest.with_suffix(".tmp.json")
        for path in (temporary, temporary_manifest):
            path.unlink(missing_ok=True)
        with duckdb.connect() as con:
            con.register("canonical_frame", frame)
            escaped = str(temporary).replace("'", "''")
            con.execute(
                f"COPY canonical_frame TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        temporary_manifest.write_text(
            json.dumps(
                {
                    "cache_format_version": CANONICAL_CACHE_FORMAT_VERSION,
                    "canonical_identity": identity,
                    "raw_source_fingerprint": record.fingerprint,
                    "contract": contract,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        temporary_manifest.replace(manifest)
        return target

    def canonical_source_identity(
        self, request, dataset, *, interval=None
    ) -> SourceSignature:
        records = self.catalog.records_for(self.raw_root, request, dataset, interval)
        if not records:
            raise DataNotAvailableError(
                f"No catalog coverage for {request.symbol} {dataset.value}"
            )
        adapter = self._adapter_for(dataset)
        return SourceSignature.from_canonical_identities(
            dataset,
            [
                canonical_partition_identity(record, adapter.canonical_contract())
                for record in records
            ],
        )

    def load_dataset(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        interval: str | None = None,
        available_at_cutoff=None,
    ) -> pd.DataFrame:
        """Load one canonical dataset for `[request.start, request.end)`.

        When ``available_at_cutoff`` is supplied, unavailable revisions are
        removed *before* logical-row de-duplication.  This ordering is essential
        for causal consumers: a later correction for the same logical candle
        must not hide the version that was observable at the cutoff.  Existing
        callers retain the prior behavior by leaving the argument unset.
        """

        if available_at_cutoff is not None:
            if (
                getattr(available_at_cutoff, "tzinfo", None) is None
                or available_at_cutoff.utcoffset() is None
            ):
                raise ValueError("available_at_cutoff must be timezone-aware")
            available_at_cutoff = available_at_cutoff.astimezone(timezone.utc)

        records = self.catalog.records_for(self.raw_root, request, dataset, interval)
        if not records:
            raise DataNotAvailableError(
                f"No catalog coverage for {request.symbol} {dataset.value} "
                f"interval={interval!r} from {request.start.isoformat()} to {request.end.isoformat()}"
            )
        parquet_paths = [self._ensure_canonical(record) for record in records]
        with duckdb.connect() as con:
            frame = con.read_parquet([str(path) for path in parquet_paths]).df()
        if frame.empty:
            return frame
        starts = pd.to_datetime(frame["period_start"], utc=True)
        mask = (starts >= request.start) & (starts < request.end)
        frame = frame.loc[mask].copy()
        frame["period_start"] = starts.loc[mask]
        for column in ("event_time", "period_end", "available_at"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        if available_at_cutoff is not None:
            if "available_at" not in frame.columns:
                raise ValueError(
                    f"{dataset.value} has no available_at column for causal loading"
                )
            available = frame["available_at"]
            frame = frame.loc[available.notna() & (available <= available_at_cutoff)].copy()
            if frame.empty:
                frame.attrs["canonical_source_identity"] = self.canonical_source_identity(
                    request, dataset, interval=interval
                ).cache_identity()
                return frame.reset_index(drop=True)
        sort_column = "event_time" if "event_time" in frame.columns else "period_start"

        if "agg_trade_id" in frame.columns:
            sort_columns = [sort_column, "agg_trade_id"]
            subset = [
                column
                for column in ("symbol", "agg_trade_id")
                if column in frame.columns
            ]
        elif "trade_id" in frame.columns:
            sort_columns = [sort_column, "trade_id"]
            subset = [
                column for column in ("symbol", "trade_id") if column in frame.columns
            ]
        elif "update_id" in frame.columns:
            sort_columns = [sort_column, "update_id"]
            subset = [
                column for column in ("symbol", "update_id") if column in frame.columns
            ]
        elif "percentage" in frame.columns:
            sort_columns = [sort_column, "percentage"]
            subset = [
                column
                for column in ("symbol", sort_column, "percentage")
                if column in frame.columns
            ]
        else:
            sort_columns = [sort_column]
            subset = [
                column
                for column in ("symbol", "interval", sort_column)
                if column in frame.columns
            ]

        frame = frame.sort_values(sort_columns, kind="stable")
        frame = frame.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
        frame.attrs["canonical_source_identity"] = self.canonical_source_identity(
            request, dataset, interval=interval
        ).cache_identity()
        return frame

    def source_signature(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        interval: str | None = None,
    ) -> SourceSignature:
        """Return catalog-based identity without reading canonical or raw rows."""

        records = self.catalog.records_for(self.raw_root, request, dataset, interval)
        if not records:
            raise DataNotAvailableError(
                f"No catalog coverage for {request.symbol} {dataset.value} "
                f"interval={interval!r} from {request.start.isoformat()} to {request.end.isoformat()}"
            )
        return self.canonical_source_identity(request, dataset, interval=interval)

    @staticmethod
    def _records_overlap(left: ArchiveRecord, right: ArchiveRecord) -> bool:
        if left.period_start is None or left.period_end is None:
            return True
        if right.period_start is None or right.period_end is None:
            return True
        return max(left.period_start, right.period_start) < min(
            left.period_end, right.period_end
        )

    def _archive_overlap_issues(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        interval: str | None = None,
    ):
        """Inspect overlapping immutable source partitions only on a quality-cache miss."""
        from .quality import CONTRACTS, classify_archive_overlap

        records = self.catalog.records_for(self.raw_root, request, dataset, interval)
        participants: set[int] = set()
        for left_index, left in enumerate(records):
            for right_index in range(left_index + 1, len(records)):
                if self._records_overlap(left, records[right_index]):
                    participants.add(left_index)
                    participants.add(right_index)
        if not participants:
            return ()

        frames: list[pd.DataFrame] = []
        for index in sorted(participants):
            path = self._ensure_canonical(records[index])
            with duckdb.connect() as con:
                frame = con.read_parquet(str(path)).df()
            if frame.empty or "period_start" not in frame:
                continue
            starts = pd.to_datetime(frame["period_start"], utc=True, errors="coerce")
            mask = (starts >= request.start) & (starts < request.end)
            frame = frame.loc[mask].copy()
            if frame.empty:
                continue
            frame["period_start"] = starts.loc[mask]
            for column in ("event_time", "period_end", "available_at"):
                if column in frame.columns:
                    frame[column] = pd.to_datetime(
                        frame[column], utc=True, errors="coerce"
                    )
            frames.append(frame.reset_index(drop=True))
        if len(frames) < 2:
            return ()
        return classify_archive_overlap(frames, CONTRACTS[dataset].logical_key)

    def data_quality_report(
        self,
        request: DataRequest,
        dataset: DatasetKind,
        *,
        interval: str | None = None,
        required: bool = True,
        frame: pd.DataFrame | None = None,
    ):
        """Return cached-or-computed quality using catalog identity before row loading.

        Warm validation is metadata-only until the caller independently needs the
        data. On a cache miss we validate canonical rows, coverage, and raw archive
        overlap once and persist a disposable JSON result independent of L2/L3.

        Order-book event streams use the same Task-12 contracts/cache but are
        validated one immutable partition at a time so a cold quality miss never
        requires a multi-year raw-event concatenation.
        """
        if (
            frame is None
            and interval is None
            and dataset in {DatasetKind.BOOK_TICKER, DatasetKind.BOOK_DEPTH}
        ):
            from .order_book import OrderBookSnapshotStore

            return OrderBookSnapshotStore(self).quality_report(
                request, dataset, required=required
            )

        from .quality import DataQualityCache, validate_dataset

        quality_cache = DataQualityCache(self.cache.root)
        try:
            source_identity = self.canonical_source_identity(
                request, dataset, interval=interval
            ).cache_identity()
        except DataNotAvailableError:
            return validate_dataset(
                None,
                request,
                dataset,
                interval=interval,
                required=required,
            )

        cached = quality_cache.get_cached(
            request,
            dataset,
            interval=interval,
            required=required,
            source_identity=source_identity,
        )
        if cached is not None:
            return cached

        if frame is None:
            frame = self.load_dataset(request, dataset, interval=interval)
        coverage = self.catalog.coverage(
            self.raw_root,
            market=request.market,
            dataset=dataset,
            symbol=request.symbol,
            interval=interval,
        )
        overlap_issues = self._archive_overlap_issues(
            request, dataset, interval=interval
        )
        report = validate_dataset(
            frame,
            request,
            dataset,
            interval=interval,
            required=required,
            source_identity=source_identity,
            coverage_start=coverage.first_period,
            coverage_end=coverage.last_period,
            extra_issues=overlap_issues,
        )
        quality_cache.store(
            request,
            dataset,
            report,
            interval=interval,
            required=required,
        )
        return report

    def event_archive_quality_report(
        self, request, dataset, *, interval=None, required=True
    ):
        """Validate large trade archives one partition at a time.

        This is the Task 6 readiness boundary for raw trade events.  It uses the
        canonical catalog identity and existing adapters/quality contracts, but
        deliberately never calls ``load_dataset`` for the complete request.
        """
        if dataset not in {DatasetKind.AGG_TRADES, DatasetKind.TRADES}:
            raise ValueError(f"partition-bounded event quality is unsupported for {dataset.value}")
        if interval is not None:
            raise ValueError(f"{dataset.value} does not accept an interval")

        from .quality import (
            DataQualityIssue, DataQualityStatus, DatasetQualityReport,
            validate_dataset,
        )

        records = self.catalog.records_for(self.raw_root, request, dataset, None)
        if not records:
            return validate_dataset(None, request, dataset, required=required)

        signature = self.canonical_source_identity(request, dataset).cache_identity()
        issues = []
        row_count = 0
        adapter = self._adapter_for(dataset)
        covered = []
        for record in records:
            start = max(request.start, record.period_start or request.start)
            end = min(request.end, record.period_end or request.end)
            if start >= end:
                continue
            frame = adapter.read(record)
            partition_request = DataRequest(
                request.symbol, start, end, request.strategy_interval,
                datasets=(dataset,), market=request.market, exchange=request.exchange,
            )
            partition = validate_dataset(
                frame, partition_request, dataset, required=required,
                source_identity=signature,
                coverage_start=record.period_start,
                coverage_end=record.period_end,
            )
            issues.extend(partition.issues)
            row_count += partition.row_count
            covered.append((start, end))
            del frame

        merged = []
        for start, end in sorted(covered):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        cursor = request.start
        gaps = []
        for start, end in merged:
            if cursor < start:
                gaps.append({"start": cursor.isoformat(), "end": start.isoformat(), "end_exclusive": True})
            cursor = max(cursor, end)
        if cursor < request.end:
            gaps.append({"start": cursor.isoformat(), "end": request.end.isoformat(), "end_exclusive": True})
        if gaps:
            issues.append(DataQualityIssue(
                "MISSING_INTERNAL_INTERVAL", DataQualityStatus.ERROR if required else DataQualityStatus.WARN,
                "Catalog archive coverage does not span the requested event range",
                count=len(gaps), details={"ranges": gaps},
            ))
        # Partition validation can legitimately repeat equivalent issues; stable
        # serialization and readiness only require their explicit presence.
        status = (DataQualityStatus.ERROR if any(i.severity is DataQualityStatus.ERROR for i in issues)
                  else DataQualityStatus.WARN if issues else DataQualityStatus.OK)
        return DatasetQualityReport(
            dataset.value, request.symbol, None, required,
            request.start.isoformat(), request.end.isoformat(),
            merged[0][0].isoformat() if merged else None,
            merged[-1][1].isoformat() if merged else None,
            merged[0][0].isoformat() if merged and not gaps else None,
            merged[-1][1].isoformat() if merged and not gaps else None,
            row_count, signature, status, tuple(issues),
        )

    def load_execution_klines(
        self,
        request: DataRequest,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """Load only the canonical columns needed by intrabar execution.

        The request predicate, OHLCV projection, overlap resolution, and ordering
        happen inside DuckDB so large execution frames do not materialize
        provenance metadata that the simulator cannot consume. Cross-archive
        duplicates use the same precedence as ``load_dataset``: catalog order is
        stable and the last matching archive wins.
        """

        effective_interval = (
            interval
            or request.intrabar_interval
            or request.strategy_interval
        )
        records = self.catalog.records_for(
            self.raw_root,
            request,
            DatasetKind.KLINES,
            effective_interval,
        )
        if not records:
            raise DataNotAvailableError(
                f"No catalog coverage for {request.symbol} {DatasetKind.KLINES.value} "
                f"interval={effective_interval!r} from {request.start.isoformat()} "
                f"to {request.end.isoformat()}"
            )

        parquet_paths = [
            str(self._ensure_canonical(record).resolve()) for record in records
        ]
        precedence_rows = [
            (path, rank) for rank, path in enumerate(parquet_paths)
        ]
        with duckdb.connect() as con:
            con.execute(
                "CREATE TEMP TABLE execution_source_precedence "
                "(path VARCHAR PRIMARY KEY, source_rank INTEGER NOT NULL)"
            )
            con.executemany(
                "INSERT INTO execution_source_precedence VALUES (?, ?)",
                precedence_rows,
            )
            frame = con.execute(
                """
                WITH ranked AS (
                    SELECT
                        source.period_start,
                        source.open,
                        source.high,
                        source.low,
                        source.close,
                        source.volume,
                        ROW_NUMBER() OVER (
                            PARTITION BY source.period_start
                            ORDER BY precedence.source_rank DESC
                        ) AS duplicate_rank
                    FROM read_parquet(?, filename = true) AS source
                    JOIN execution_source_precedence AS precedence
                      ON source.filename = precedence.path
                    WHERE source.period_start >= ? AND source.period_start < ?
                )
                SELECT period_start, open, high, low, close, volume
                FROM ranked
                WHERE duplicate_rank = 1
                ORDER BY period_start
                """,
                [parquet_paths, request.start, request.end],
            ).df()
        if frame.empty:
            return frame

        frame["period_start"] = pd.to_datetime(frame["period_start"], utc=True)
        return frame.reset_index(drop=True)

    def load_klines(
        self, request: DataRequest, interval: str | None = None
    ) -> pd.DataFrame:
        """Load canonical completed-candle records for a requested interval."""

        return self.load_dataset(
            request,
            DatasetKind.KLINES,
            interval=interval or request.strategy_interval,
        )
