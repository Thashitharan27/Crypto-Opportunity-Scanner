"""Task 6: requirement-driven rich data acquisition for Task 5 candidates only."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import Enum
from typing import Callable, Mapping, Protocol

from .data.binance.selective_acquisition import (
    AcquisitionState, ArchiveAcquisitionRequest, BackendAcquisitionResult,
)
from .data.quality import DataQualityStatus, MissingCoverageRange
from .data.query import DataRequest
from .data.schemas import DatasetKind
from .data.source_identity import SourceSignature
from .data.store import DataNotAvailableError, MarketDataStore
from .data.timing import interval_to_timedelta, normalize_binance_interval
from .features import production_feature_registry
from .features.registry import FeatureRegistry
from .final_candidates import FinalCandidate, FinalCandidateSet


class RequirementRequiredness(str, Enum):
    REQUIRED_FOR_FEATURE = "REQUIRED_FOR_FEATURE"
    OPTIONAL_FOR_FEATURE = "OPTIONAL_FOR_FEATURE"


class FeatureReadiness(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RichDataAcquisitionConfig:
    enabled_features: tuple[str, ...] = ()
    feature_parameters: Mapping[str, Mapping[str, object]] | None = None
    intrabar_interval: str | None = None
    max_workers: int = 4

    def __post_init__(self):
        if not 1 <= self.max_workers <= 32:
            raise ValueError("max_workers must be between 1 and 32")
        object.__setattr__(self, "enabled_features", tuple(sorted(set(self.enabled_features))))
        if self.intrabar_interval is not None:
            object.__setattr__(self, "intrabar_interval", normalize_binance_interval(self.intrabar_interval))


@dataclass(frozen=True, slots=True)
class RichDataRequirement:
    symbol: str
    final_rank: int
    dataset: DatasetKind
    interval: str | None
    start: object
    end: object
    feature_names: tuple[str, ...]
    reasons: tuple[str, ...]
    requiredness: RequirementRequiredness
    data_request: DataRequest


@dataclass(frozen=True, slots=True)
class FeatureDataRequirementPlan:
    feature_name: str
    requirements: tuple[RichDataRequirement, ...]


@dataclass(frozen=True, slots=True)
class RichDataAcquisitionPlan:
    requirements: tuple[RichDataRequirement, ...]
    features: tuple[FeatureDataRequirementPlan, ...]


@dataclass(frozen=True, slots=True)
class RichDatasetResult:
    requirement: RichDataRequirement
    state: AcquisitionState
    missing_ranges: tuple[MissingCoverageRange, ...] = ()
    quality_status: DataQualityStatus | None = None
    source_signature: SourceSignature | None = None
    source_identity: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureDataReadiness:
    feature_name: str
    readiness: FeatureReadiness
    datasets: tuple[RichDatasetResult, ...]


@dataclass(frozen=True, slots=True)
class SymbolRichDataResult:
    symbol: str
    final_rank: int
    strategy_source_identity: str
    datasets: tuple[RichDatasetResult, ...]
    features: tuple[FeatureDataReadiness, ...]


@dataclass(frozen=True, slots=True)
class RichDataAcquisitionResult:
    plan: RichDataAcquisitionPlan
    symbols: tuple[SymbolRichDataResult, ...]


class RichArchiveBackend(Protocol):
    def acquire_archive(self, request: ArchiveAcquisitionRequest, *, cancelled=None) -> BackendAcquisitionResult: ...


def resolve_rich_data_requirements(
    candidates: FinalCandidateSet,
    config: RichDataAcquisitionConfig,
    registry: FeatureRegistry | None = None,
) -> RichDataAcquisitionPlan:
    """Resolve only explicit features over ``FinalCandidateSet.candidates``."""
    registry = registry or production_feature_registry()
    resolved = registry.resolve(config.enabled_features, config.feature_parameters)
    by_name = {item.definition.name: item for item in resolved}
    raw: list[RichDataRequirement] = []
    for candidate in sorted(candidates.candidates, key=lambda c: (c.final_rank, c.symbol)):
        base = candidate.strategy_data_request
        for name in sorted(by_name):
            params = by_name[name].parameters
            specs: list[tuple[DatasetKind, str | None, timedelta, RequirementRequiredness, str]] = []
            if name == "funding_context":
                specs.append((DatasetKind.FUNDING_RATE, None, timedelta(days=max(1.0, float(params["funding_zscore_window_days"]))), RequirementRequiredness.REQUIRED_FOR_FEATURE, "funding history"))
            elif name == "basis_context":
                warmup = max(timedelta(days=float(params["basis_zscore_window_days"])), interval_to_timedelta(base.strategy_interval))
                specs.extend((kind, base.strategy_interval, warmup, requiredness, reason) for kind, requiredness, reason in (
                    (DatasetKind.MARK_PRICE_KLINES, RequirementRequiredness.REQUIRED_FOR_FEATURE, "mark price basis"),
                    (DatasetKind.INDEX_PRICE_KLINES, RequirementRequiredness.REQUIRED_FOR_FEATURE, "index price basis"),
                    (DatasetKind.PREMIUM_INDEX_KLINES, RequirementRequiredness.OPTIONAL_FOR_FEATURE, "premium basis enrichment"),
                ))
            elif name == "futures_positioning":
                specs.append((DatasetKind.FUTURES_METRICS, None, timedelta(days=max(1.0, float(params["oi_zscore_window_days"]))), RequirementRequiredness.REQUIRED_FOR_FEATURE, "open-interest positioning"))
                if base.strategy_interval != "1h":
                    specs.append((DatasetKind.KLINES, "1h", timedelta(hours=1), RequirementRequiredness.OPTIONAL_FOR_FEATURE, "one-hour price telemetry"))
            elif name == "taker_flow_context":
                interval = normalize_binance_interval(str(params["taker_flow_interval"]))
                if interval != base.strategy_interval:
                    specs.append((DatasetKind.KLINES, interval, timedelta(hours=1), RequirementRequiredness.REQUIRED_FOR_FEATURE, "parameterized taker-flow klines"))
            elif name == "trade_flow_context":
                try:
                    source = DatasetKind[str(params["trade_flow_source"]).upper()]
                except KeyError as exc:
                    raise ValueError("trade_flow_source must be AGG_TRADES or TRADES") from exc
                if source not in {DatasetKind.AGG_TRADES, DatasetKind.TRADES}:
                    raise ValueError("trade_flow_source must be AGG_TRADES or TRADES")
                specs.append((source, None, timedelta(0), RequirementRequiredness.REQUIRED_FOR_FEATURE, "raw trade-flow archive"))
            elif name == "order_book_context":
                specs.extend((kind, None, timedelta(0), RequirementRequiredness.OPTIONAL_FOR_FEATURE, "order-book any-of source") for kind in (DatasetKind.BOOK_TICKER, DatasetKind.BOOK_DEPTH))
            for dataset, interval, warmup, requiredness, reason in specs:
                request = _request(base, dataset, interval, base.start - warmup, base.end)
                raw.append(RichDataRequirement(base.symbol, candidate.final_rank, dataset, interval, request.start, request.end, (name,), (reason,), requiredness, request))
        if config.intrabar_interval and config.intrabar_interval != base.strategy_interval:
            interval = config.intrabar_interval
            request = _request(base, DatasetKind.KLINES, interval, base.start, base.end)
            raw.append(RichDataRequirement(base.symbol, candidate.final_rank, DatasetKind.KLINES, interval, request.start, request.end, ("intrabar",), ("explicit intrabar execution data",), RequirementRequiredness.OPTIONAL_FOR_FEATURE, request))
    merged = _coalesce(raw)
    feature_plans = tuple(FeatureDataRequirementPlan(name, tuple(r for r in merged if name in r.feature_names)) for name in config.enabled_features)
    return RichDataAcquisitionPlan(merged, feature_plans)


def _request(base, dataset, interval, start, end):
    return DataRequest(base.symbol, start, end, interval or base.strategy_interval, datasets=(dataset,), market=base.market, exchange=base.exchange)


def _coalesce(requirements):
    groups = {}
    for item in requirements:
        key = (item.symbol, item.dataset, item.interval)
        if key not in groups:
            groups[key] = item
            continue
        prior = groups[key]
        start, end = min(prior.start, item.start), max(prior.end, item.end)
        requiredness = RequirementRequiredness.REQUIRED_FOR_FEATURE if RequirementRequiredness.REQUIRED_FOR_FEATURE in {prior.requiredness, item.requiredness} else RequirementRequiredness.OPTIONAL_FOR_FEATURE
        groups[key] = replace(prior, start=start, end=end, feature_names=tuple(sorted(set(prior.feature_names + item.feature_names))), reasons=tuple(sorted(set(prior.reasons + item.reasons))), requiredness=requiredness, data_request=_request(prior.data_request, prior.dataset, prior.interval, start, end))
    return tuple(sorted(groups.values(), key=lambda r: (r.final_rank, r.dataset.value, r.interval or "", r.start, r.symbol)))


class SelectiveRichDataAcquirer:
    def __init__(self, store: MarketDataStore, backend: RichArchiveBackend, config=RichDataAcquisitionConfig(), registry=None):
        self.store, self.backend, self.config, self.registry = store, backend, config, registry

    def acquire(self, candidates: FinalCandidateSet, *, cancelled: Callable[[], bool] | None = None):
        plan = resolve_rich_data_requirements(candidates, self.config, self.registry)
        self.store.refresh_catalog()
        results, pending = {}, []
        for req in plan.requirements:
            if cancelled and cancelled():
                results[_key(req)] = RichDatasetResult(req, AcquisitionState.CANCELLED, detail="not attempted")
                continue
            try:
                report = self.store.data_quality_report(req.data_request, req.dataset, interval=req.interval, required=req.requiredness is RequirementRequiredness.REQUIRED_FOR_FEATURE)
            except ValueError as exc:
                results[_key(req)] = RichDatasetResult(req, AcquisitionState.QUALITY_FAILED, detail=f"Data Lake validation failed: {exc}")
                continue
            if report.status is DataQualityStatus.OK:
                results[_key(req)] = self._available(req, report, AcquisitionState.REUSED)
            elif report.has_non_missing_errors():
                results[_key(req)] = RichDatasetResult(req, AcquisitionState.QUALITY_FAILED, quality_status=report.status, detail="existing source is structurally invalid")
            elif report.missing_coverage_ranges():
                pending.append((req, report, report.missing_coverage_ranges()))
            else:
                results[_key(req)] = RichDatasetResult(req, AcquisitionState.QUALITY_FAILED, quality_status=report.status, detail="quality failure has no acquirable gap")
        self._run(pending, results, cancelled)
        return self._result(candidates, plan, results)

    def _run(self, pending, results, cancelled):
        iterator, futures = iter(pending), {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            def schedule():
                while len(futures) < self.config.max_workers:
                    try: req, report, gaps = next(iterator)
                    except StopIteration: return
                    if cancelled and cancelled():
                        results[_key(req)] = RichDatasetResult(req, AcquisitionState.CANCELLED, gaps, report.status, detail="not attempted")
                    else:
                        call = ArchiveAcquisitionRequest(req.data_request, req.dataset, req.interval, gaps)
                        futures[pool.submit(self.backend.acquire_archive, call, cancelled=cancelled)] = (req, report, gaps)
            schedule()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    req, prior, gaps = futures.pop(future)
                    try: outcome = future.result()
                    except Exception as exc: outcome = BackendAcquisitionResult(AcquisitionState.DOWNLOAD_FAILED, str(exc))
                    # Work that reached backend completion remains intact. Cancellation
                    # prevents subsequent scheduling; it does not rewrite completed work.
                    if outcome.state is AcquisitionState.ACQUIRED:
                        results[_key(req)] = self._verify(req, gaps)
                    else:
                        state = AcquisitionState.CANCELLED if cancelled and cancelled() else outcome.state
                        results[_key(req)] = RichDatasetResult(req, state, gaps, prior.status, detail=outcome.detail)
                schedule()

    def _verify(self, req, gaps):
        self.store.refresh_catalog()
        report = self.store.data_quality_report(req.data_request, req.dataset, interval=req.interval, required=req.requiredness is RequirementRequiredness.REQUIRED_FOR_FEATURE)
        if report.status is DataQualityStatus.OK:
            return self._available(req, report, AcquisitionState.ACQUIRED, gaps)
        state = AcquisitionState.QUALITY_FAILED if report.has_non_missing_errors() else AcquisitionState.MISSING
        return RichDatasetResult(req, state, gaps, report.status, detail="post-acquisition Data Lake validation failed")

    def _available(self, req, report, state, gaps=()):
        try: signature = self.store.source_signature(req.data_request, req.dataset, interval=req.interval)
        except (DataNotAvailableError, ValueError) as exc:
            return RichDatasetResult(req, AcquisitionState.QUALITY_FAILED, tuple(gaps), report.status, detail=str(exc))
        return RichDatasetResult(req, state, tuple(gaps), report.status, signature, signature.cache_identity())

    def _result(self, candidates, plan, results):
        symbols = []
        for candidate in sorted(candidates.candidates, key=lambda c: (c.final_rank, c.symbol)):
            datasets = tuple(results[_key(r)] for r in plan.requirements if r.symbol == candidate.symbol)
            features = []
            for feature in self.config.enabled_features:
                relevant = tuple(r for r in datasets if feature in r.requirement.feature_names)
                available = lambda r: r.state in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED}
                if feature == "order_book_context":
                    count = sum(available(r) for r in relevant)
                    readiness = FeatureReadiness.READY if count == 2 else FeatureReadiness.DEGRADED if count == 1 else FeatureReadiness.UNAVAILABLE
                else:
                    required = tuple(r for r in relevant if r.requirement.requiredness is RequirementRequiredness.REQUIRED_FOR_FEATURE)
                    optional = tuple(r for r in relevant if r.requirement.requiredness is RequirementRequiredness.OPTIONAL_FOR_FEATURE)
                    readiness = FeatureReadiness.UNAVAILABLE if any(not available(r) for r in required) else FeatureReadiness.DEGRADED if any(not available(r) for r in optional) else FeatureReadiness.READY
                features.append(FeatureDataReadiness(feature, readiness, relevant))
            symbols.append(SymbolRichDataResult(candidate.symbol, candidate.final_rank, candidate.strategy_source_identity, datasets, tuple(features)))
        return RichDataAcquisitionResult(plan, tuple(symbols))


def _key(requirement):
    return requirement.symbol, requirement.dataset, requirement.interval
