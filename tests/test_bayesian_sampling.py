from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.bayesian_sampling import (
    _BayesResearchSamplingEngine,
    _normalize_directions,
    _research_native_config,
    _resolved_samples,
    _validate_sample_causality,
    is_sampling_timestamp,
    resolve_sampling_interval,
)
from crypto_strategy_lab.bayesian_sampling_reporting import (
    _stable_empty_samples,
    bayesian_sampling_enabled,
)
from crypto_strategy_lab.config import EntryMode
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.research_adapters import native_simulator_config


def test_default_sampling_uses_whatever_strategy_timeframe_is_selected():
    assert resolve_sampling_interval(15) == (15, 1)
    assert resolve_sampling_interval(60) == (60, 1)
    assert resolve_sampling_interval(240) == (240, 1)
    assert resolve_sampling_interval(1440) == (1440, 1)


def test_custom_sampling_interval_must_stay_on_strategy_grid():
    assert resolve_sampling_interval(60, 240) == (240, 4)
    assert resolve_sampling_interval(15, 60) == (60, 4)
    with pytest.raises(ValueError, match="cannot be lower"):
        resolve_sampling_interval(240, 60)
    with pytest.raises(ValueError, match="integer multiple"):
        resolve_sampling_interval(60, 90)


def test_sampling_grid_is_utc_anchored_not_request_start_anchored():
    assert is_sampling_timestamp("2024-01-01T00:00:00Z", 240)
    assert not is_sampling_timestamp("2024-01-01T01:00:00Z", 240)
    assert is_sampling_timestamp("2024-01-01T04:00:00Z", 240)
    # Starting a requested slice at 02:00 does not move the 4h grid to 02/06/10.
    assert not is_sampling_timestamp("2024-01-01T02:00:00Z", 240)
    assert is_sampling_timestamp("2024-01-01T08:00:00Z", 240)


def test_directions_are_independent_and_validated():
    assert _normalize_directions(("long", "SHORT")) == ("LONG", "SHORT")
    assert _normalize_directions(("LONG", "LONG")) == ("LONG",)
    with pytest.raises(ValueError, match="at least one"):
        _normalize_directions(())
    with pytest.raises(ValueError, match="unsupported"):
        _normalize_directions(("FLAT",))


def test_market_grid_removes_strategy_selection_rules_but_keeps_native_execution_config():
    run = ResearchRunConfig()
    strategy = replace(
        run.strategy,
        enable_daily_entry_schedule=True,
        profiles={
            key: replace(
                profile,
                enabled=False,
                flip_direction=True,
                entry_rules=(
                    {
                        "action": "REJECT",
                        "indicator": "ADX",
                        "condition": "INSIDE",
                        "minimum": 0.0,
                        "maximum": 10.0,
                    },
                ),
            )
            for key, profile in run.strategy.profiles.items()
        },
    )
    native = native_simulator_config(
        run.data,
        run.features,
        strategy,
        run.execution,
    )

    research = _research_native_config(native, 100)

    assert research.entry_mode == EntryMode.EVERY_N_CANDLES
    assert research.entry_interval == 1
    assert research.enable_daily_entry_schedule is False
    assert research.max_active_pairs == 202
    assert research.risk_per_leg == native.risk_per_leg
    assert research.tie_policy == native.tie_policy
    assert all(profile.enabled for profile in research.strategy_profiles.values())
    assert all(not profile.flip_direction for profile in research.strategy_profiles.values())
    assert all(not profile.entry_rules for profile in research.strategy_profiles.values())


def test_end_of_data_samples_are_treated_as_censored_not_losses():
    frame = pd.DataFrame(
        [
            {"side": "LONG", "long_exit_reason": "TP", "pair_net_r": 1.0},
            {"side": "LONG", "long_exit_reason": "END_OF_DATA", "pair_net_r": -0.1},
            {"side": "SHORT", "short_exit_reason": "SL", "pair_net_r": -1.0},
            {"side": "SHORT", "short_exit_reason": "END_OF_DATA", "pair_net_r": 0.2},
        ]
    )

    result = _resolved_samples(frame)

    assert len(result) == 2
    assert result["side"].tolist() == ["LONG", "SHORT"]
    assert result["pair_net_r"].tolist() == [1.0, -1.0]


def test_completed_research_observation_does_not_change_later_research_equity():
    engine = object.__new__(_BayesResearchSamplingEngine)
    pair = SimpleNamespace(
        is_open=False,
        equity_before_trade=1000.0,
        equity_after_trade=None,
    )
    engine.active_pairs = [pair]
    engine.completed_pairs = []
    engine.current_equity = 1000.0

    engine._collect_closed_pairs()

    assert engine.current_equity == 1000.0
    assert pair.equity_after_trade == 1000.0
    assert engine.completed_pairs == [pair]
    assert engine.active_pairs == []


def test_research_entry_cannot_precede_prepared_evidence_availability():
    prepared = SimpleNamespace(
        timestamp=np.array(
            ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            dtype="datetime64[ns]",
        ),
        decision_available_at=np.array(
            ["2024-01-01T01:00:00", "2024-01-01T02:00:00"],
            dtype="datetime64[ns]",
        ),
        __len__=lambda self: 2,
    )

    class Prepared:
        timestamp = prepared.timestamp
        decision_available_at = prepared.decision_available_at

        def __len__(self):
            return 2

    valid = pd.DataFrame(
        [
            {
                "research_signal_index": 0,
                "research_signal_candle_open_time": "2024-01-01T00:00:00Z",
                "entry_time": "2024-01-01T01:00:00Z",
                "exit_time": "2024-01-01T03:00:00Z",
            }
        ]
    )
    _validate_sample_causality(valid, Prepared())

    invalid = valid.copy()
    invalid.loc[0, "entry_time"] = "2024-01-01T00:59:00Z"
    with pytest.raises(ValueError, match="before its evidence"):
        _validate_sample_causality(invalid, Prepared())


def test_engine_entry_cadence_is_timeframe_agnostic_and_ignores_open_overlap():
    engine = object.__new__(_BayesResearchSamplingEngine)
    engine.risk = np.ones(6)
    engine.times = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T01:00:00",
            "2024-01-01T02:00:00",
            "2024-01-01T03:00:00",
            "2024-01-01T04:00:00",
            "2024-01-01T05:00:00",
        ],
        dtype="datetime64[ns]",
    )
    engine.research_sampling_interval_minutes = 240
    engine.active_pairs = [object(), object(), object()]
    engine._in_trading_window = lambda i: True
    engine._profile_context = lambda i: ("BULL", "LONG", "bull_long", object())

    assert engine._should_enter(0)
    assert not engine._should_enter(1)
    assert not engine._should_enter(2)
    assert not engine._should_enter(3)
    assert engine._should_enter(4)


def test_deep_research_is_the_explicit_sampling_opt_in():
    assert not bayesian_sampling_enabled(SimpleNamespace(analysis_level="STANDARD"))
    assert bayesian_sampling_enabled(SimpleNamespace(analysis_level="DEEP"))
    assert bayesian_sampling_enabled(SimpleNamespace(analysis_level="deep_research"))


def test_empty_deep_run_still_has_a_publishable_sampling_schema():
    empty = _stable_empty_samples(pd.DataFrame())
    assert empty.empty
    for required in (
        "bayes_sample_id",
        "side",
        "entry_time",
        "exit_time",
        "pair_net_r",
        "bayes_long_probability",
        "bayes_short_probability",
    ):
        assert required in empty.columns
