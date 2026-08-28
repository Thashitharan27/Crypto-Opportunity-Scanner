"""Reuse-first candle acquisition for a bounded discovery shortlist.

No transport or raw cache lives here.  Missing immutable archives are delegated
to a backend and every claimed download is re-opened through MarketDataStore.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import importlib
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from ..quality import DataQualityStatus, DatasetQualityReport, MissingCoverageRange
from ..query import DataRequest
from ..schemas import DatasetKind, MarketKind
from ..source_identity import SourceSignature
from ..store import DataNotAvailableError, MarketDataStore
from .historical_discovery import HistoricalDiscoveryResult
from .universe import DiscoveryRow


class AcquisitionState(str, Enum):
    REUSED = "REUSED"
    ACQUIRED = "ACQUIRED"
    MISSING = "MISSING"
    QUALITY_FAILED = "QUALITY_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class SelectiveCandleAcquisitionConfig:
    shortlist_size: int = 25
    strategy_interval: str = "1h"
    max_workers: int = 4

    def __post_init__(self) -> None:
        if self.shortlist_size <= 0:
            raise ValueError("shortlist_size must be positive")
        if not 1 <= self.max_workers <= 32:
            raise ValueError("max_workers must be between 1 and 32")
        # DataRequest owns interval normalization and validation.
        normalized = DataRequest(
            "INTERVALCHECK", datetime(2000, 1, 1), datetime(2000, 1, 2),
            self.strategy_interval,
        ).strategy_interval
        object.__setattr__(self, "strategy_interval", normalized)


@dataclass(frozen=True, slots=True)
class RankedSymbol:
    symbol: str
    rank: int


@dataclass(frozen=True, slots=True)
class CandleAcquisitionRequest:
    """Transport instruction wrapping, rather than duplicating, DataRequest."""

    data_request: DataRequest
    missing_ranges: tuple[MissingCoverageRange, ...]


@dataclass(frozen=True, slots=True)
class BackendAcquisitionResult:
    state: AcquisitionState
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {
            AcquisitionState.ACQUIRED,
            AcquisitionState.MISSING,
            AcquisitionState.DOWNLOAD_FAILED,
            AcquisitionState.CANCELLED,
        }:
            raise ValueError(f"invalid backend acquisition state: {self.state.value}")


class CandleAcquisitionBackend(Protocol):
    def acquire(
        self,
        request: CandleAcquisitionRequest,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> BackendAcquisitionResult: ...


class AcquisitionBackendConfigurationError(RuntimeError):
    pass


DATA_HUB_DATASET_KEYS = {
    DatasetKind.KLINES: "klines",
    DatasetKind.FUTURES_METRICS: "metrics",
    DatasetKind.FUNDING_RATE: "fundingRate",
    DatasetKind.MARK_PRICE_KLINES: "markPriceKlines",
    DatasetKind.INDEX_PRICE_KLINES: "indexPriceKlines",
    DatasetKind.PREMIUM_INDEX_KLINES: "premiumIndexKlines",
    DatasetKind.AGG_TRADES: "aggTrades",
    DatasetKind.TRADES: "trades",
    DatasetKind.BOOK_DEPTH: "bookDepth",
    DatasetKind.BOOK_TICKER: "bookTicker",
}


def data_hub_dataset_key(dataset: DatasetKind) -> str:
    """Translate the canonical Data Lake family at the Data Hub boundary."""
    return DATA_HUB_DATASET_KEYS[dataset]


@dataclass(frozen=True, slots=True)
class ArchiveAcquisitionRequest:
    data_request: DataRequest
    dataset: DatasetKind
    interval: str | None
    missing_ranges: tuple[MissingCoverageRange, ...]


@dataclass(frozen=True, slots=True)
class SymbolAcquisitionResult:
    symbol: str
    rank: int
    state: AcquisitionState
    strategy_interval: str
    requested_start: datetime
    requested_end: datetime
    acquisition_ranges: tuple[MissingCoverageRange, ...] = ()
    row_count: int = 0
    quality_status: DataQualityStatus | None = None
    source_signature: SourceSignature | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CandleAcquisitionResult:
    symbols: tuple[SymbolAcquisitionResult, ...]


def shortlist_from_live(
    rows: Sequence[DiscoveryRow], limit: int
) -> tuple[RankedSymbol, ...]:
    return _bounded_ranked(
        (RankedSymbol(row.symbol, row.preliminary_rank) for row in rows
         if row.eligible and row.preliminary_rank is not None), limit
    )


def shortlist_from_historical(
    result: HistoricalDiscoveryResult, limit: int
) -> tuple[RankedSymbol, ...]:
    return _bounded_ranked(
        (RankedSymbol(row.symbol, row.rank) for row in result.eligible_candidates
         if row.eligible and row.rank is not None), limit
    )


def _bounded_ranked(rows: Iterable[RankedSymbol], limit: int) -> tuple[RankedSymbol, ...]:
    if limit <= 0:
        raise ValueError("shortlist limit must be positive")
    ordered = sorted(rows, key=lambda row: (row.rank, row.symbol))
    seen: set[str] = set()
    result: list[RankedSymbol] = []
    for row in ordered:
        symbol = row.symbol.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(RankedSymbol(symbol, row.rank))
        if len(result) == limit:
            break
    return tuple(result)


class BinanceDataHubBackend:
    """Configurable import bridge to Data Hub's existing archive downloader.

    Each scanner symbol occupies one worker, so Data Hub is deliberately called
    with ``workers=1`` to prevent multiplicative concurrency.
    """

    def __init__(self, root: Path, *, module: str = "binance_data_hub.archive_downloader", project_path: Path | None = None):
        self.root = Path(root)
        self.module = module
        self.project_path = Path(project_path) if project_path else None

    def _downloader(self):
        if self.project_path is not None:
            path = str(self.project_path.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            module = importlib.import_module(self.module)
            return module.download_archive_library
        except (ImportError, AttributeError) as exc:
            raise AcquisitionBackendConfigurationError(
                f"Binance Data Hub backend {self.module!r} is unavailable; configure "
                "its import module/project path explicitly"
            ) from exc

    def acquire(self, request: CandleAcquisitionRequest, *, cancelled=None) -> BackendAcquisitionResult:
        return self.acquire_archive(
            ArchiveAcquisitionRequest(
                request.data_request, DatasetKind.KLINES,
                request.data_request.strategy_interval, request.missing_ranges,
            ),
            cancelled=cancelled,
        )

    def acquire_archive(self, request: ArchiveAcquisitionRequest, *, cancelled=None) -> BackendAcquisitionResult:
        """Acquire canonical archive gaps without interpreting their contents."""
        downloader = self._downloader()
        for gap in request.missing_ranges:
            if cancelled and cancelled():
                return BackendAcquisitionResult(AcquisitionState.CANCELLED, "cancelled")
            try:
                outcome = downloader(
                    [request.data_request.symbol], [data_hub_dataset_key(request.dataset)],
                    [request.interval] if request.interval else [], self.root,
                    start_date=gap.start.date(),
                    # Data Hub's archive-date API is inclusive, whereas the
                    # DataRequest/quality boundary is half open.
                    end_date=(gap.end - timedelta(microseconds=1)).date(),
                    workers=1,
                    verify=False, cancelled=cancelled,
                )
            except Exception as exc:
                return BackendAcquisitionResult(AcquisitionState.DOWNLOAD_FAILED, str(exc))
            failed = _outcome_value(outcome, "failed")
            missing = _outcome_value(outcome, "missing")
            was_cancelled = _outcome_value(outcome, "cancelled")
            if was_cancelled:
                return BackendAcquisitionResult(
                    AcquisitionState.CANCELLED, "Data Hub cancelled acquisition"
                )
            if failed or outcome is False:
                return BackendAcquisitionResult(
                    AcquisitionState.DOWNLOAD_FAILED, "Data Hub reported failed archives"
                )
            if missing:
                return BackendAcquisitionResult(
                    AcquisitionState.MISSING, "Data Hub reported unpublished/missing archives"
                )
        return BackendAcquisitionResult(AcquisitionState.ACQUIRED)


def _outcome_value(outcome, name: str):
    """Read Data Hub's nested counts from mapping or equivalent result objects."""
    counts = (
        outcome.get("counts")
        if isinstance(outcome, Mapping)
        else getattr(outcome, "counts", None)
    )
    if counts is not None:
        value = counts.get(name) if isinstance(counts, Mapping) else getattr(counts, name, None)
    else:
        # Retain compatibility with small adapters exposing the counts directly.
        value = outcome.get(name) if isinstance(outcome, Mapping) else getattr(outcome, name, None)
    return bool(value)


class SelectiveCandleAcquirer:
    def __init__(self, store: MarketDataStore, backend: CandleAcquisitionBackend, config=SelectiveCandleAcquisitionConfig()):
        self.store, self.backend, self.config = store, backend, config

    def acquire(self, discovery, start: datetime, end: datetime, *, cancelled=None) -> CandleAcquisitionResult:
        shortlist = (
            shortlist_from_historical(discovery, self.config.shortlist_size)
            if isinstance(discovery, HistoricalDiscoveryResult)
            else shortlist_from_live(discovery, self.config.shortlist_size)
        )
        self.store.refresh_catalog()
        results: dict[str, SymbolAcquisitionResult] = {}
        pending: list[tuple[RankedSymbol, DataRequest, DatasetQualityReport, tuple[MissingCoverageRange, ...]]] = []
        for ranked in shortlist:
            request = DataRequest(ranked.symbol, start, end, self.config.strategy_interval,
                                  datasets=(DatasetKind.KLINES,), market=MarketKind.FUTURES_UM)
            if cancelled and cancelled():
                results[ranked.symbol] = self._result(ranked, request, AcquisitionState.CANCELLED, detail="not attempted")
                continue
            try:
                report = self.store.data_quality_report(
                    request, DatasetKind.KLINES, interval=request.strategy_interval
                )
            except ValueError as exc:
                results[ranked.symbol] = self._result(
                    ranked, request, AcquisitionState.QUALITY_FAILED,
                    detail=f"Data Lake validation failed: {exc}",
                )
                continue
            if report.status is DataQualityStatus.OK:
                results[ranked.symbol] = self._verified(ranked, request, report, AcquisitionState.REUSED)
            elif report.has_non_missing_errors():
                results[ranked.symbol] = self._result(ranked, request, AcquisitionState.QUALITY_FAILED,
                                                       quality=report, detail="existing source is structurally invalid")
            else:
                gaps = report.missing_coverage_ranges()
                if gaps:
                    pending.append((ranked, request, report, gaps))
                else:
                    results[ranked.symbol] = self._result(ranked, request, AcquisitionState.QUALITY_FAILED,
                                                           quality=report, detail="quality failure has no acquirable coverage gap")
        self._run_pending(pending, results, cancelled)
        return CandleAcquisitionResult(tuple(results[row.symbol] for row in shortlist))

    def _run_pending(self, pending, results, cancelled):
        iterator = iter(pending)
        futures: dict[Future, tuple] = {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            def schedule() -> None:
                while len(futures) < self.config.max_workers:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        return
                    ranked, request, report, gaps = item
                    if cancelled and cancelled():
                        results[ranked.symbol] = self._result(ranked, request, AcquisitionState.CANCELLED,
                                                               ranges=gaps, quality=report, detail="not attempted")
                        continue
                    acquisition = CandleAcquisitionRequest(request, gaps)
                    futures[pool.submit(self.backend.acquire, acquisition, cancelled=cancelled)] = item
            schedule()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    ranked, request, prior, gaps = futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = BackendAcquisitionResult(
                            AcquisitionState.DOWNLOAD_FAILED, str(exc)
                        )
                    if outcome.state is AcquisitionState.CANCELLED or (cancelled and cancelled()):
                        results[ranked.symbol] = self._result(ranked, request, AcquisitionState.CANCELLED,
                                                               ranges=gaps, quality=prior, detail=outcome.detail)
                    elif outcome.state is not AcquisitionState.ACQUIRED:
                        results[ranked.symbol] = self._result(
                            ranked, request, outcome.state, ranges=gaps,
                            quality=prior, detail=outcome.detail,
                        )
                    else:
                        results[ranked.symbol] = self._post_verify(ranked, request, gaps)
                schedule()
            for ranked, request, report, gaps in iterator:
                results[ranked.symbol] = self._result(ranked, request, AcquisitionState.CANCELLED,
                                                       ranges=gaps, quality=report, detail="not attempted")

    def _post_verify(self, ranked, request, gaps):
        self.store.refresh_catalog()
        try:
            report = self.store.data_quality_report(
                request, DatasetKind.KLINES, interval=request.strategy_interval
            )
        except ValueError as exc:
            return self._result(
                ranked, request, AcquisitionState.QUALITY_FAILED, ranges=gaps,
                detail=f"post-acquisition Data Lake validation failed: {exc}",
            )
        if report.status is DataQualityStatus.OK:
            return self._verified(ranked, request, report, AcquisitionState.ACQUIRED, gaps)
        state = AcquisitionState.QUALITY_FAILED if report.has_non_missing_errors() else AcquisitionState.MISSING
        return self._result(ranked, request, state, ranges=gaps, quality=report,
                            detail="post-acquisition Data Lake validation failed")

    def _verified(self, ranked, request, report, state, ranges=()):
        try:
            frame = self.store.load_dataset(request, DatasetKind.KLINES, interval=request.strategy_interval)
            signature = self.store.source_signature(request, DatasetKind.KLINES, interval=request.strategy_interval)
        except (DataNotAvailableError, ValueError) as exc:
            return self._result(ranked, request, AcquisitionState.QUALITY_FAILED, ranges=ranges,
                                quality=report, detail=str(exc))
        return self._result(ranked, request, state, ranges=ranges, quality=report,
                            rows=len(frame), signature=signature)

    @staticmethod
    def _result(ranked, request, state, *, ranges=(), quality=None, rows=0, signature=None, detail=None):
        return SymbolAcquisitionResult(ranked.symbol, ranked.rank, state, request.strategy_interval,
                                       request.start, request.end, tuple(ranges), rows,
                                       quality.status if quality else None, signature, detail)
