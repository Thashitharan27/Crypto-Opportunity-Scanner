from dataclasses import fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionState, CandleAcquisitionResult, SymbolAcquisitionResult,
)
from crypto_strategy_lab.data.binance.historical_discovery import (
    DiscoveryDecisionTime, HistoricalDiscoveryConfig, HistoricalDiscoveryResult,
    HistoricalSnapshotRow, HistoricalUniverseSnapshot,
)
from crypto_strategy_lab.data.binance.universe import DiscoveryRow
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.final_candidates import (
    ExclusionReason, FinalCandidate, FinalCandidateBoundaryConfig,
    OpportunityModelRef, SelectionKind, build_final_candidate_set,
)
from crypto_strategy_lab.opportunity_scoring import (
    OpportunityScoreRow, OpportunityScoringResult, ScoreStatus,
)

NOW = datetime(2025, 1, 2, tzinfo=timezone.utc)
START, END = NOW - timedelta(days=5), NOW


def live(symbol, rank, *, eligible=True):
    return DiscoveryRow(symbol, eligible, () if eligible else ("filtered",), timedelta(days=100),
                        Decimal("20000000"), Decimal("10"), Decimal("10.01"), Decimal("0.1"),
                        Decimal("8"), Decimal("-3"), rank if eligible else None, NOW)


def acquisition(symbol, rank, state=AcquisitionState.REUSED, interval="1h", signature=True):
    source = SourceSignature(DatasetKind.KLINES, f"digest-{symbol}", 2) if signature else None
    return SymbolAcquisitionResult(symbol, rank, state, interval, START, END, row_count=120,
                                   source_signature=source)


def score(symbol, rank, model_rank, source, *, status=ScoreStatus.SCORABLE,
          decision=NOW, interval="1h", model="balanced_activity"):
    return OpportunityScoreRow(symbol, rank, decision, interval, START, START, {}, {}, {}, model, "1",
        None if status is ScoreStatus.UNSCORABLE else float(10 - model_rank),
        None if status is ScoreStatus.UNSCORABLE else model_rank,
        None, None, None, None, None, None, "UNKNOWN", None, None, None, source, {}, status,
        "missing" if status is ScoreStatus.UNSCORABLE else None)


def test_live_handoff_request_metrics_signature_and_no_trading_semantics():
    source = acquisition("BTCUSDT", 1, interval="60m")
    result = build_final_candidate_set([live("BTCUSDT", 1)], CandleAcquisitionResult((source,)))
    candidate = result.candidates[0]
    assert candidate.discovery_timestamp == NOW
    assert candidate.impact_metrics.price_change_percent == Decimal("-3")
    assert candidate.impact_metrics.absolute_price_change_percent == Decimal("3")
    assert candidate.impact_metrics.reference_period_start is None
    assert candidate.strategy_source_signature is source.source_signature
    assert candidate.strategy_source_identity == source.source_signature.cache_identity()
    assert candidate.strategy_data_request.symbol == "BTCUSDT"
    assert candidate.strategy_data_request.strategy_interval == "1h"
    assert candidate.strategy_data_request.datasets == (DatasetKind.KLINES,)
    assert candidate.strategy_data_request.market is MarketKind.FUTURES_UM
    assert result.strategy_requests() == (candidate.strategy_data_request,)
    forbidden = {"side", "long", "short", "entry", "entry_eligible", "stop", "target",
                 "take_profit", "position"}
    assert forbidden.isdisjoint({field.name for field in fields(FinalCandidate)})


def test_only_ready_states_cross_and_candidate_limit_is_explicit():
    states = list(AcquisitionState)
    rows = [live(f"S{i}", i + 1) for i in range(len(states))]
    acquired = [acquisition(row.symbol, row.preliminary_rank, state) for row, state in zip(rows, states)]
    result = build_final_candidate_set(rows, CandleAcquisitionResult(tuple(acquired)),
                                       config=FinalCandidateBoundaryConfig(max_candidates=1))
    assert len(result.candidates) == 1
    reasons = {item.symbol: item.reason for item in result.exclusions}
    assert reasons["S1"] is ExclusionReason.CANDIDATE_LIMIT
    assert all(reasons[f"S{i}"] is ExclusionReason.ACQUISITION_NOT_READY for i in range(2, 6))


def test_historical_handoff_uses_decision_time_and_separate_discovery_source():
    period_start, period_end = NOW - timedelta(days=2), NOW - timedelta(days=1)
    historical_row = HistoricalSnapshotRow(
        "BTCUSDT", period_start, period_end, period_end, Decimal("10"), Decimal("12"),
        Decimal("9"), Decimal("11"), Decimal("30000000"), Decimal("27.27"),
        Decimal("10"), Decimal("10"), True, (), 1, "historical-1d-source",
    )
    decision = DiscoveryDecisionTime(NOW)
    discovery = HistoricalDiscoveryResult(
        decision, HistoricalUniverseSnapshot(decision, (historical_row,)), (historical_row,), (),
        HistoricalDiscoveryConfig(),
    )
    acquired = acquisition("BTCUSDT", 1)
    candidate = build_final_candidate_set(discovery, CandleAcquisitionResult((acquired,))).candidates[0]
    assert candidate.discovery_timestamp == NOW
    assert candidate.discovery_source_identity == "historical-1d-source"
    assert candidate.strategy_source_identity != candidate.discovery_source_identity
    assert candidate.impact_metrics.reference_period_start == period_start
    assert candidate.impact_metrics.reference_period_end == period_end
    assert candidate.impact_metrics.spread_percent is None
    assert candidate.impact_metrics.listing_age is None


def test_scores_are_ignored_by_default_but_exact_explicit_model_orders():
    rows = [live("AAA", 1), live("BBB", 2)]
    acquired = tuple(acquisition(row.symbol, row.preliminary_rank) for row in rows)
    scores = OpportunityScoringResult(NOW, tuple(
        score(row.symbol, row.preliminary_rank, 3 - row.preliminary_rank,
              acquired[index].source_signature.cache_identity()) for index, row in enumerate(rows)))
    default = build_final_candidate_set(rows, CandleAcquisitionResult(acquired), scores)
    assert [c.symbol for c in default.candidates] == ["AAA", "BBB"]
    assert all(c.opportunity_score is None and c.selection_reason.kind is SelectionKind.DISCOVERY_ORDER
               for c in default.candidates)
    explicit = build_final_candidate_set(rows, CandleAcquisitionResult(acquired), scores,
        FinalCandidateBoundaryConfig(opportunity_model=OpportunityModelRef("balanced_activity", "1")))
    assert [c.symbol for c in explicit.candidates] == ["BBB", "AAA"]
    assert all(c.selection_reason.kind is SelectionKind.OPPORTUNITY_MODEL for c in explicit.candidates)


def test_explicit_score_failures_never_fall_back():
    row = live("AAA", 1); acquired = acquisition("AAA", 1)
    config = FinalCandidateBoundaryConfig(opportunity_model=OpportunityModelRef("balanced_activity", "1"))
    cases = [
        (None, ExclusionReason.OPPORTUNITY_MODEL_MISSING),
        (score("AAA", 1, 1, acquired.source_signature.cache_identity(), status=ScoreStatus.UNSCORABLE),
         ExclusionReason.OPPORTUNITY_UNSCORABLE),
        (score("AAA", 1, 1, "different"), ExclusionReason.SCORE_SOURCE_IDENTITY_MISMATCH),
        (score("AAA", 1, 1, acquired.source_signature.cache_identity(), decision=NOW-timedelta(hours=1)),
         ExclusionReason.OPPORTUNITY_DECISION_TIME_MISMATCH),
        (score("AAA", 1, 1, acquired.source_signature.cache_identity(), interval="4h"),
         ExclusionReason.OPPORTUNITY_INTERVAL_MISMATCH),
    ]
    for score_row, reason in cases:
        scoring = None if score_row is None else OpportunityScoringResult(NOW, (score_row,))
        result = build_final_candidate_set([row], CandleAcquisitionResult((acquired,)), scoring, config)
        assert not result.candidates
        assert result.exclusions[0].reason is reason


def test_deterministic_serialization_and_rank_invariant():
    rows = [live("BBB", 2), live("AAA", 1)]
    acquired = [acquisition("BBB", 2), acquisition("AAA", 1)]
    first = build_final_candidate_set(rows, CandleAcquisitionResult(tuple(acquired))).serializable()
    second = build_final_candidate_set(rows[::-1], CandleAcquisitionResult(tuple(acquired[::-1]))).serializable()
    assert first == second

    bad = acquisition("AAA", 9)
    try:
        build_final_candidate_set([live("AAA", 1)], CandleAcquisitionResult((bad,)))
    except ValueError as error:
        assert "rank disagrees" in str(error)
    else:
        raise AssertionError("rank mismatch did not fail fast")
