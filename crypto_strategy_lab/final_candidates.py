"""Task 5: pure, immutable boundary from scanner results to Lab requests.

This module deliberately performs no discovery, acquisition, feature execution,
opportunity scoring, or strategy evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Sequence

from crypto_strategy_lab.data.binance.historical_discovery import HistoricalDiscoveryResult
from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionState, CandleAcquisitionResult, SymbolAcquisitionResult,
)
from crypto_strategy_lab.data.binance.universe import DiscoveryRow
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.data.timing import ensure_utc, normalize_binance_interval
from crypto_strategy_lab.opportunity_scoring import (
    OpportunityScoreRow, OpportunityScoringResult, ScoreStatus,
)


class DiscoveryMode(str, Enum):
    LIVE = "LIVE"
    HISTORICAL = "HISTORICAL"


class SelectionKind(str, Enum):
    DISCOVERY_ORDER = "DISCOVERY_ORDER"
    OPPORTUNITY_MODEL = "OPPORTUNITY_MODEL"


class ExclusionReason(str, Enum):
    DISCOVERY_NOT_ELIGIBLE = "discovery_not_eligible"
    ACQUISITION_NOT_READY = "acquisition_not_ready"
    ACQUISITION_SOURCE_MISSING = "acquisition_source_missing"
    STRATEGY_INTERVAL_MISMATCH = "strategy_interval_mismatch"
    OPPORTUNITY_MODEL_MISSING = "opportunity_model_missing"
    OPPORTUNITY_UNSCORABLE = "opportunity_unscorable"
    OPPORTUNITY_DECISION_TIME_MISMATCH = "opportunity_decision_time_mismatch"
    OPPORTUNITY_INTERVAL_MISMATCH = "opportunity_interval_mismatch"
    SCORE_SOURCE_IDENTITY_MISMATCH = "score_source_identity_mismatch"
    CANDIDATE_LIMIT = "candidate_limit"


@dataclass(frozen=True, slots=True)
class CandidateImpactMetrics:
    quote_volume: Decimal | None
    range_percent: Decimal | None
    price_change_percent: Decimal | None
    absolute_price_change_percent: Decimal | None
    spread_percent: Decimal | None
    listing_age: timedelta | None
    reference_period_start: datetime | None
    reference_period_end: datetime | None
    reference_available_at: datetime | None

    def serializable(self) -> dict[str, object]:
        return {field.name: _json(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class OpportunityModelRef:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("opportunity model name and version must not be empty")


@dataclass(frozen=True, slots=True)
class FinalCandidateBoundaryConfig:
    strategy_interval: str = "1h"
    max_candidates: int | None = None
    opportunity_model: OpportunityModelRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_interval", normalize_binance_interval(self.strategy_interval))
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive or None")


@dataclass(frozen=True, slots=True)
class CandidateSelectionReason:
    kind: SelectionKind
    discovery_rank: int
    final_rank: int
    model_name: str | None = None
    model_version: str | None = None
    model_rank: int | None = None
    score: float | None = None

    def serializable(self) -> dict[str, object]:
        return {field.name: _json(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    symbol: str
    reason: ExclusionReason
    detail: str | None = None

    def serializable(self) -> dict[str, object]:
        return {"symbol": self.symbol, "reason": self.reason.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class FinalCandidate:
    symbol: str
    discovery_timestamp: datetime
    discovery_rank: int
    final_rank: int
    impact_metrics: CandidateImpactMetrics
    strategy_data_request: DataRequest
    strategy_interval: str
    strategy_source_signature: SourceSignature
    strategy_source_identity: str
    discovery_source_identity: str | None
    opportunity_score: float | None
    opportunity_model_name: str | None
    opportunity_model_version: str | None
    opportunity_model_rank: int | None
    selection_reason: CandidateSelectionReason
    discovery_mode: DiscoveryMode
    discovery_contract: str

    def serializable(self) -> dict[str, object]:
        request = self.strategy_data_request
        return {
            "symbol": self.symbol, "discovery_timestamp": _json(self.discovery_timestamp),
            "discovery_rank": self.discovery_rank, "final_rank": self.final_rank,
            "impact_metrics": self.impact_metrics.serializable(),
            "strategy_data_request": {
                "symbol": request.symbol, "start": _json(request.start), "end": _json(request.end),
                "strategy_interval": request.strategy_interval,
                "intrabar_interval": request.intrabar_interval,
                "datasets": [item.value for item in request.datasets],
                "market": request.market.value, "exchange": request.exchange,
            },
            "strategy_interval": self.strategy_interval,
            "strategy_source_signature": {
                "dataset": self.strategy_source_signature.dataset.value,
                "digest": self.strategy_source_signature.digest,
                "partition_count": self.strategy_source_signature.partition_count,
                "identity_version": self.strategy_source_signature.identity_version,
            },
            "strategy_source_identity": self.strategy_source_identity,
            "discovery_source_identity": self.discovery_source_identity,
            "opportunity_score": self.opportunity_score,
            "opportunity_model_name": self.opportunity_model_name,
            "opportunity_model_version": self.opportunity_model_version,
            "opportunity_model_rank": self.opportunity_model_rank,
            "selection_reason": self.selection_reason.serializable(),
            "discovery_mode": self.discovery_mode.value,
            "discovery_contract": self.discovery_contract,
        }


@dataclass(frozen=True, slots=True)
class FinalCandidateSet:
    candidates: tuple[FinalCandidate, ...]
    exclusions: tuple[CandidateExclusion, ...]
    config: FinalCandidateBoundaryConfig

    def strategy_requests(self) -> tuple[DataRequest, ...]:
        return tuple(candidate.strategy_data_request for candidate in self.candidates)

    def serializable(self) -> dict[str, object]:
        model = self.config.opportunity_model
        return {
            "config": {"strategy_interval": self.config.strategy_interval,
                       "max_candidates": self.config.max_candidates,
                       "opportunity_model": None if model is None else {"name": model.name, "version": model.version}},
            "candidates": [candidate.serializable() for candidate in self.candidates],
            "exclusions": [exclusion.serializable() for exclusion in self.exclusions],
        }


@dataclass(frozen=True, slots=True)
class _Discovery:
    symbol: str; rank: int | None; eligible: bool; timestamp: datetime
    metrics: CandidateImpactMetrics; source_identity: str | None
    mode: DiscoveryMode; contract: str


def build_final_candidate_set(
    discovery: Sequence[DiscoveryRow] | HistoricalDiscoveryResult,
    acquisition: CandleAcquisitionResult,
    scoring: OpportunityScoringResult | None = None,
    config: FinalCandidateBoundaryConfig = FinalCandidateBoundaryConfig(),
) -> FinalCandidateSet:
    """Normalize completed Task 1--4 outputs; never execute those tasks."""
    normalized = _adapt_discovery(discovery)
    _unique(normalized, "discovery")
    acquisitions = _index_acquisitions(acquisition.symbols)
    scores = _score_rows(scoring)
    exclusions: list[CandidateExclusion] = []
    ready: list[tuple[_Discovery, SymbolAcquisitionResult, OpportunityScoreRow | None]] = []
    model = config.opportunity_model

    discovered_symbols = {row.symbol for row in normalized}
    for symbol in acquisitions.keys() - discovered_symbols:
        exclusions.append(CandidateExclusion(symbol, ExclusionReason.DISCOVERY_NOT_ELIGIBLE,
                                             "symbol is absent from discovery output"))

    for row in sorted(normalized, key=lambda item: item.symbol):
        if not row.eligible or row.rank is None:
            exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.DISCOVERY_NOT_ELIGIBLE))
            continue
        acquired = acquisitions.get(row.symbol)
        if acquired is not None and acquired.rank != row.rank:
            raise ValueError(f"acquisition rank disagrees with discovery rank for {row.symbol}")
        if acquired is None or acquired.state not in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED}:
            detail = "missing acquisition result" if acquired is None else acquired.state.value
            exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.ACQUISITION_NOT_READY, detail))
            continue
        if (acquired.source_signature is None
                or acquired.source_signature.dataset is not DatasetKind.KLINES
                or not acquired.source_signature.digest.strip()
                or acquired.source_signature.partition_count <= 0):
            exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.ACQUISITION_SOURCE_MISSING))
            continue
        if normalize_binance_interval(acquired.strategy_interval) != config.strategy_interval:
            exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.STRATEGY_INTERVAL_MISMATCH))
            continue
        score = None
        if model is not None:
            relevant = [s for s in scores if s.symbol == row.symbol and s.model_name == model.name and s.model_version == model.version]
            exact_time = [s for s in relevant if ensure_utc(s.decision_time) == row.timestamp]
            if not relevant:
                exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.OPPORTUNITY_MODEL_MISSING)); continue
            if not exact_time:
                exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.OPPORTUNITY_DECISION_TIME_MISMATCH)); continue
            score = exact_time[0]
            if normalize_binance_interval(score.strategy_interval) != config.strategy_interval:
                exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.OPPORTUNITY_INTERVAL_MISMATCH)); continue
            if score.status is not ScoreStatus.SCORABLE or score.score is None or score.model_rank is None:
                exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.OPPORTUNITY_UNSCORABLE, score.unscorable_reason)); continue
            if score.discovery_rank != row.rank:
                raise ValueError(f"score rank disagrees with discovery rank for {row.symbol}")
            if score.source_identity != acquired.source_signature.cache_identity():
                exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.SCORE_SOURCE_IDENTITY_MISMATCH)); continue
        ready.append((row, acquired, score))

    key = ((lambda item: (item[2].model_rank, item[0].rank, item[0].symbol)) if model else
           (lambda item: (item[0].rank, item[0].symbol)))
    ordered = sorted(ready, key=key)
    if config.max_candidates is not None:
        for row, _, _ in ordered[config.max_candidates:]:
            exclusions.append(CandidateExclusion(row.symbol, ExclusionReason.CANDIDATE_LIMIT))
        ordered = ordered[:config.max_candidates]
    candidates = tuple(_candidate(item, index, model) for index, item in enumerate(ordered, 1))
    exclusions.sort(key=lambda item: (item.symbol, item.reason.value, item.detail or ""))
    return FinalCandidateSet(candidates, tuple(exclusions), config)


def _candidate(item, final_rank: int, model: OpportunityModelRef | None) -> FinalCandidate:
    row, acquired, score = item
    request = DataRequest(row.symbol, acquired.requested_start, acquired.requested_end,
                          acquired.strategy_interval, datasets=(DatasetKind.KLINES,), market=MarketKind.FUTURES_UM)
    reason = CandidateSelectionReason(
        SelectionKind.OPPORTUNITY_MODEL if model else SelectionKind.DISCOVERY_ORDER,
        row.rank, final_rank, None if model is None else model.name,
        None if model is None else model.version, None if score is None else score.model_rank,
        None if score is None else score.score,
    )
    signature = acquired.source_signature
    return FinalCandidate(row.symbol, row.timestamp, row.rank, final_rank, row.metrics, request,
                          request.strategy_interval, signature, signature.cache_identity(), row.source_identity,
                          None if score is None else score.score, None if model is None else model.name,
                          None if model is None else model.version, None if score is None else score.model_rank,
                          reason, row.mode, row.contract)


def _adapt_discovery(discovery) -> list[_Discovery]:
    if isinstance(discovery, HistoricalDiscoveryResult):
        timestamp = discovery.decision_time.value
        return [_Discovery(row.symbol.strip().upper(), row.rank, row.eligible, timestamp,
            CandidateImpactMetrics(row.quote_volume, row.range_percent, row.price_change_percent,
                row.absolute_price_change_percent, None, None, row.period_start, row.period_end, row.available_at),
            row.source_identity, DiscoveryMode.HISTORICAL, discovery.contract) for row in discovery.snapshot.rows]
    rows = list(discovery)
    timestamps = {ensure_utc(row.discovery_timestamp) for row in rows}
    if len(timestamps) > 1:
        raise ValueError("live discovery rows have mixed discovery timestamps")
    return [_Discovery(row.symbol.strip().upper(), row.preliminary_rank, row.eligible,
        ensure_utc(row.discovery_timestamp), CandidateImpactMetrics(row.quote_volume, row.range_24h_percent,
        row.price_change_24h_percent, None if row.price_change_24h_percent is None else abs(row.price_change_24h_percent),
        row.spread_percent, row.listing_age, None, None, None), None, DiscoveryMode.LIVE,
        "binance_usdm_live_24h_v1") for row in rows]


def _unique(rows: Sequence[_Discovery], label: str) -> None:
    seen: dict[str, int | None] = {}
    for row in rows:
        if not row.symbol:
            raise ValueError(f"{label} contains an empty symbol")
        if row.symbol in seen:
            raise ValueError(f"duplicate {label} symbol {row.symbol}")
        seen[row.symbol] = row.rank


def _index_acquisitions(rows):
    result = {}
    for row in rows:
        symbol = row.symbol.strip().upper()
        if not symbol: raise ValueError("acquisition contains an empty symbol")
        if symbol in result: raise ValueError(f"duplicate acquisition symbol {symbol}")
        result[symbol] = row
    return result


def _score_rows(scoring):
    rows = () if scoring is None else scoring.rows
    seen = set()
    for row in rows:
        symbol = row.symbol.strip().upper()
        if not symbol: raise ValueError("score contains an empty symbol")
        key = (symbol, row.model_name, row.model_version, ensure_utc(row.decision_time))
        if key in seen: raise ValueError(f"duplicate opportunity score row for {symbol}")
        seen.add(key)
    return tuple(rows)


def _json(value):
    if isinstance(value, Enum): return value.value
    if isinstance(value, datetime): return ensure_utc(value).isoformat()
    if isinstance(value, timedelta): return value.total_seconds()
    if isinstance(value, Decimal): return str(value)
    return value
