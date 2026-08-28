from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.config import IntrabarMissingPolicy, TiePolicy
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.prepared_backtest import IntrabarExecutionData
from crypto_strategy_lab.research_adapters import native_simulator_config
from crypto_strategy_lab.research_sampling import (
    StrategyResearchSamplingEngine,
    _annotate_episodes,
    _resolved_samples,
    _select_sampling_mode,
    build_context_breakdown,
    build_episode_table,
    build_sampling_summary,
    research_native_config,
)
from crypto_strategy_lab.research_sampling_reporting import (
    _episode_reporting_context,
    research_sampling_enabled,
)
from crypto_strategy_lab.trade import ExitReason, Position, Side, TradePair


def _viable_rows():
    return pd.DataFrame(
        [
            {"research_signal_index": 10, "side": "SHORT", "strategy_profile_key": "bear_short", "pair_net_r": 1.0, "entry_time": "2024-01-01", "exit_time": "2024-01-02", "short_exit_reason": "TP", "plus_di": 10.0, "minus_di": 35.0, "adx": 25.0, "market_regime": "BEAR"},
            {"research_signal_index": 11, "side": "SHORT", "strategy_profile_key": "bear_short", "pair_net_r": -1.0, "entry_time": "2024-01-02", "exit_time": "2024-01-03", "short_exit_reason": "SL", "plus_di": 12.0, "minus_di": 33.0, "adx": 28.0, "market_regime": "BEAR"},
            {"research_signal_index": 12, "side": "SHORT", "strategy_profile_key": "bear_short", "pair_net_r": 1.0, "entry_time": "2024-01-03", "exit_time": "2024-01-04", "short_exit_reason": "TP", "plus_di": 11.0, "minus_di": 31.0, "adx": 31.0, "market_regime": "BEAR"},
            {"research_signal_index": 13, "side": "LONG", "strategy_profile_key": "sideways_long", "pair_net_r": 1.0, "entry_time": "2024-01-04", "exit_time": "2024-01-05", "long_exit_reason": "TP", "plus_di": 22.0, "minus_di": 18.0, "adx": 18.0, "market_regime": "SIDEWAYS"},
            {"research_signal_index": 15, "side": "LONG", "strategy_profile_key": "sideways_long", "pair_net_r": -1.0, "entry_time": "2024-01-06", "exit_time": "2024-01-07", "long_exit_reason": "SL", "plus_di": 24.0, "minus_di": 17.0, "adx": 19.0, "market_regime": "SIDEWAYS"},
        ]
    )


def _intrabar_for_exit_tests(*, event_high=100.2, event_low=99.8, event_minute=20):
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z", periods=45, freq="1min"
    ).to_numpy(dtype="datetime64[ns]")
    opens = np.full(len(timestamps), 100.0)
    highs = np.full(len(timestamps), 100.2)
    lows = np.full(len(timestamps), 99.8)
    highs[event_minute] = event_high
    lows[event_minute] = event_low
    return IntrabarExecutionData(
        timestamps,
        pd.Timedelta(minutes=1),
        opens,
        highs,
        lows,
    )


def _research_exit_engine(pairs, intrabar, *, tie_policy=TiePolicy.PESSIMISTIC):
    engine = object.__new__(StrategyResearchSamplingEngine)
    engine.config = SimpleNamespace(
        use_intrabar_data=True,
        intrabar_timeframe_minutes=1,
        intrabar_missing_policy=IntrabarMissingPolicy.ERROR,
        slippage=0.0,
        tie_policy=tie_policy,
        maker_fee=0.0,
        taker_fee=0.0,
        use_maker_exit=False,
    )
    engine.times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=3, freq="15min"
    ).to_numpy(dtype="datetime64[ns]")
    engine.entry_delta = pd.Timedelta(minutes=15)
    engine.intrabar_data = intrabar
    engine.active_pairs = list(pairs)
    engine.missing_intrabar_intervals = []
    engine.fallback_reasons = []
    engine.last_timeout_exit_time = None
    return engine


def _research_pair(
    pair_id: int,
    side: Side,
    *,
    timeout_minutes: int | None = None,
):
    position = Position(
        side=side,
        entry_time=pd.Timestamp("2026-01-01T00:15:00"),
        entry_index=0,
        entry_price=100.0,
        risk=1.0,
        sl=99.0 if side == Side.LONG else 101.0,
        tp=101.0 if side == Side.LONG else 99.0,
        quantity=1.0,
        risk_amount=1.0,
        entry_notional=100.0,
        atr_at_entry=1.0,
        original_sl=99.0 if side == Side.LONG else 101.0,
    )
    pair = TradePair(
        pair_id=pair_id,
        long=position if side == Side.LONG else None,
        short=position if side == Side.SHORT else None,
        equity_before_trade=1000.0,
        strategy_candle_open_time=pd.Timestamp("2026-01-01T00:00:00Z"),
        strategy_entry_time=pd.Timestamp("2026-01-01T00:15:00Z"),
        strategy_entry_price=100.0,
    )
    pair.profile_timeout_enabled = timeout_minutes is not None
    pair.profile_timeout_minutes = timeout_minutes
    return pair


def _assert_exit_parity(actual_pair, expected_pair):
    actual = actual_pair.position
    expected = expected_pair.position
    assert actual.exit_reason == expected.exit_reason
    assert actual.exit_source == expected.exit_source
    assert actual.exit_time == expected.exit_time
    assert actual.exit_price == pytest.approx(expected.exit_price)
    assert actual.gross_pnl == pytest.approx(expected.gross_pnl)
    assert actual.net_pnl == pytest.approx(expected.net_pnl)
    assert actual.gross_r == pytest.approx(expected.gross_r)
    assert actual.net_r == pytest.approx(expected.net_r)
    assert actual.price_r == pytest.approx(expected.price_r)
    assert actual.ambiguous == expected.ambiguous
    assert actual_pair.profile_timeout_triggered == expected_pair.profile_timeout_triggered
    assert actual_pair.timeout_exit_time == expected_pair.timeout_exit_time


def test_reporting_config_defaults_to_realistic_portfolio_mode():
    run = ResearchRunConfig()
    assert run.reporting.research_sampling_mode == "PORTFOLIO"
    assert run.reporting.research_sampling_interval_candles == 1
    assert not research_sampling_enabled(run.reporting)


def test_research_sampling_config_is_strict_and_persistent():
    run = ResearchRunConfig()
    reporting = replace(
        run.reporting,
        research_sampling_mode="FIXED_INTERVAL",
        research_sampling_interval_candles=3,
    )
    configured = replace(run, reporting=reporting)
    configured.validate()
    assert configured.to_dict()["reporting"]["research_sampling_mode"] == "FIXED_INTERVAL"
    assert research_sampling_enabled(configured.reporting)

    with pytest.raises(ValueError, match="invalid research sampling mode"):
        replace(run, reporting=replace(run.reporting, research_sampling_mode="UNKNOWN")).validate()
    with pytest.raises(ValueError, match="interval must be positive"):
        replace(run, reporting=replace(run.reporting, research_sampling_interval_candles=0)).validate()


def test_research_native_config_preserves_strategy_rules_and_per_trade_execution():
    run = ResearchRunConfig()
    strategy = replace(
        run.strategy,
        profiles={
            key: replace(
                profile,
                enabled=(key == "bear_short"),
                entry_rules=(
                    {"action": "REJECT", "indicator": "ADX", "condition": "INSIDE", "minimum": 0.0, "maximum": 20.0},
                ) if key == "bear_short" else (),
            )
            for key, profile in run.strategy.profiles.items()
        },
    )
    native = native_simulator_config(run.data, run.features, strategy, run.execution)
    research = research_native_config(native, 100)

    assert research.strategy_profiles["bear_short"].enabled
    assert research.strategy_profiles["bear_short"].entry_rules == native.strategy_profiles["bear_short"].entry_rules
    assert research.max_active_pairs == 202
    assert research.max_combined_effective_leverage is None
    assert research.max_effective_leverage_per_leg == native.max_effective_leverage_per_leg
    assert research.risk_per_leg == native.risk_per_leg
    assert research.tie_policy == native.tie_policy
    assert research.enable_daily_entry_schedule is False


def test_research_engine_ignores_open_overlap_but_still_requires_strategy_context():
    engine = object.__new__(StrategyResearchSamplingEngine)
    engine.risk = np.ones(3)
    engine.active_pairs = [object(), object(), object()]
    engine._in_trading_window = lambda i: True
    engine._profile_context = lambda i: ("BEAR", "SHORT", "bear_short", object()) if i != 1 else None

    assert engine._should_enter(0)
    assert not engine._should_enter(1)
    assert engine._should_enter(2)


def test_batched_simple_exit_matches_native_for_overlapping_long_and_short():
    intrabar = _intrabar_for_exit_tests(event_high=101.2, event_low=99.5)
    expected_pairs = [_research_pair(1, Side.LONG), _research_pair(2, Side.SHORT)]
    actual_pairs = [_research_pair(1, Side.LONG), _research_pair(2, Side.SHORT)]
    expected = _research_exit_engine(expected_pairs, intrabar)
    actual = _research_exit_engine(actual_pairs, intrabar)

    DataLakeProductionBacktestEngine._update_positions_to_strategy_index(expected, 1)
    actual._update_positions_to_strategy_index(1)

    for actual_pair, expected_pair in zip(actual_pairs, expected_pairs):
        _assert_exit_parity(actual_pair, expected_pair)
    assert actual_pairs[0].position.exit_reason == ExitReason.TP
    assert actual_pairs[1].position.exit_reason == ExitReason.SL
    stats = actual.research_exit_optimization_stats()
    assert stats["research_batched_simple_position_intervals"] == 2
    assert stats["research_dynamic_exit_position_intervals"] == 0
    assert stats["research_batch_fallback_position_intervals"] == 0


@pytest.mark.parametrize(
    ("tie_policy", "expected_reason"),
    [
        (TiePolicy.PESSIMISTIC, ExitReason.SL),
        (TiePolicy.OPTIMISTIC, ExitReason.TP),
    ],
)
def test_batched_simple_exit_preserves_same_bar_tie_policy(tie_policy, expected_reason):
    intrabar = _intrabar_for_exit_tests(event_high=101.5, event_low=98.5)
    expected_pair = _research_pair(1, Side.LONG)
    actual_pair = _research_pair(1, Side.LONG)
    expected = _research_exit_engine([expected_pair], intrabar, tie_policy=tie_policy)
    actual = _research_exit_engine([actual_pair], intrabar, tie_policy=tie_policy)

    DataLakeProductionBacktestEngine._update_positions_to_strategy_index(expected, 1)
    actual._update_positions_to_strategy_index(1)

    _assert_exit_parity(actual_pair, expected_pair)
    assert actual_pair.position.exit_reason == expected_reason
    assert actual_pair.position.ambiguous


def test_batched_simple_exit_preserves_timeout_priority_over_same_minute_target():
    intrabar = _intrabar_for_exit_tests(event_high=101.5, event_low=99.8)
    expected_pair = _research_pair(1, Side.LONG, timeout_minutes=5)
    actual_pair = _research_pair(1, Side.LONG, timeout_minutes=5)
    expected = _research_exit_engine([expected_pair], intrabar)
    actual = _research_exit_engine([actual_pair], intrabar)

    DataLakeProductionBacktestEngine._update_positions_to_strategy_index(expected, 1)
    actual._update_positions_to_strategy_index(1)

    _assert_exit_parity(actual_pair, expected_pair)
    assert actual_pair.position.exit_reason == ExitReason.PROFILE_TIMEOUT
    assert actual_pair.profile_timeout_triggered


def test_stateful_research_exit_uses_mature_scanner_instead_of_batch_kernel():
    intrabar = _intrabar_for_exit_tests(event_high=100.2, event_low=99.8)
    pair = _research_pair(1, Side.LONG)
    pair.position.trailing_enabled = True
    pair.position.trailing_activation_price = 105.0
    pair.position.trailing_distance_r = 1.0
    pair.position.favourable_price = 100.0
    engine = _research_exit_engine([pair], intrabar)

    engine._update_positions_to_strategy_index(1)

    stats = engine.research_exit_optimization_stats()
    assert stats["research_batched_simple_position_intervals"] == 0
    assert stats["research_dynamic_exit_position_intervals"] == 1
    assert pair.position.is_open


def test_episode_ids_break_on_gap_direction_or_profile_change():
    result = _annotate_episodes(_viable_rows())
    assert result["research_episode_id"].tolist() == [
        "episode-000001", "episode-000001", "episode-000001",
        "episode-000002", "episode-000003",
    ]
    assert result["research_episode_entry_number"].tolist() == [1, 2, 3, 1, 1]
    assert result["research_episode_viable_entries"].tolist() == [3, 3, 3, 1, 1]


def test_sampling_modes_are_episode_anchored_not_global_index_anchored():
    viable = _annotate_episodes(_viable_rows())
    every = _select_sampling_mode(viable, "EVERY_VIABLE_ENTRY", 99)
    fixed = _select_sampling_mode(viable, "FIXED_INTERVAL", 2)
    first = _select_sampling_mode(viable, "EPISODE_FIRST", 99)

    assert every["research_signal_index"].tolist() == [10, 11, 12, 13, 15]
    assert fixed["research_signal_index"].tolist() == [10, 12, 13, 15]
    assert first["research_signal_index"].tolist() == [10, 13, 15]


def test_end_of_data_is_censored_before_outcome_reporting():
    frame = _annotate_episodes(_viable_rows())
    frame.loc[2, "short_exit_reason"] = "END_OF_DATA"
    resolved = _resolved_samples(frame)
    assert resolved["research_signal_index"].tolist() == [10, 11, 13, 15]
    # Episode identity was assigned before censoring, so the cluster membership remains stable.
    assert resolved.loc[resolved["research_signal_index"].eq(11), "research_episode_viable_entries"].iloc[0] == 3


def test_cluster_aware_summary_reports_entries_and_episodes_separately():
    samples = _annotate_episodes(_viable_rows())
    samples["research_sample_id"] = [f"sample-{i}" for i in range(len(samples))]
    samples["research_sampling_mode"] = "EVERY_VIABLE_ENTRY"
    episodes = build_episode_table(samples)
    episodes, episode_reporting = _episode_reporting_context(episodes, samples)
    summary = build_sampling_summary(samples, {"mode": "EVERY_VIABLE_ENTRY"})
    summary.update(episode_reporting)
    context = build_context_breakdown(samples)

    assert len(episodes) == 3
    assert episodes["episode_start_year"].tolist() == [2024, 2024, 2024]
    assert episodes["market_regime"].tolist() == ["BEAR", "SIDEWAYS", "SIDEWAYS"]
    assert summary["synthetic_entries"] == 5
    assert summary["unique_episodes"] == 3
    assert summary["entry_win_rate"] == pytest.approx(3 / 5)
    assert summary["episode_success_rate"] == pytest.approx(2 / 3)
    assert summary["average_sampled_entries_per_episode"] == pytest.approx(5 / 3)
    assert summary["average_viable_entries_per_episode"] == pytest.approx(5 / 3)
    assert summary["episodes_by_year"] == {"2024": 3}
    assert summary["episodes_by_regime"] == {"BEAR": 1, "SIDEWAYS": 2}
    assert summary["bayesian_effective_cluster_units"] == 3
    assert {"research_di_bucket", "research_adx_bucket", "market_regime"} <= set(context["dimension"])
