from types import SimpleNamespace

import numpy as np

from crypto_strategy_lab.rule_native_engine import (
    RuleAwareDataLakeProductionBacktestEngine,
)


def _engine():
    engine = object.__new__(RuleAwareDataLakeProductionBacktestEngine)
    engine.plus_di_values = np.array([32.0, 18.0])
    engine.minus_di_values = np.array([11.0, 27.0])
    engine.di_pressure_spread_change = np.array([5.0, -3.0])
    engine.long_directional_di_change = np.array([6.0, -4.0])
    engine.long_opposing_di_change = np.array([-2.0, 3.0])
    engine.long_di_pressure_state = np.array(["EXPANDING", "CONTRACTING"], dtype=object)
    engine.short_directional_di_change = np.array([-2.0, 3.0])
    engine.short_opposing_di_change = np.array([6.0, -4.0])
    engine.short_di_pressure_state = np.array(["CONTRACTING", "EXPANDING"], dtype=object)
    engine.config = SimpleNamespace(enable_support_resistance_analysis=True)
    engine._pending_sr_context = None
    return engine


def _sr_context():
    return SimpleNamespace(
        near_support=True,
        near_resistance=False,
        inside_support_zone=True,
        inside_resistance_zone=False,
        support_state="SUPPORT_HELD",
        resistance_state="APPROACHING_RESISTANCE",
        support_held=True,
        resistance_held=False,
        trade_location_rating="GOOD_LOCATION",
        room_in_direction_atr=2.5,
        nearest_support_distance_atr=0.25,
        nearest_resistance_distance_atr=2.5,
        support_rejection_atr=0.8,
        resistance_rejection_atr=0.1,
    )


def test_directional_di_rule_value_uses_absolute_candidate_side_di():
    engine = _engine()
    profile = SimpleNamespace()
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "DIRECTIONAL_DI"
    ) == 32.0
    assert engine._strategy_profile_rule_value(
        0, "SHORT", profile, "DIRECTIONAL_DI"
    ) == 11.0
    assert engine._strategy_profile_rule_value(
        1, "SHORT", profile, "DIRECTIONAL_DI"
    ) == 27.0


def test_pressure_state_rule_value_uses_direction_specific_prepared_state():
    engine = _engine()
    profile = SimpleNamespace()
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "DI_PRESSURE_STATE"
    ) == 1.0
    assert engine._strategy_profile_rule_value(
        0, "SHORT", profile, "DI_PRESSURE_STATE"
    ) == 2.0


def test_numeric_pressure_rule_values_use_prepared_causal_arrays():
    engine = _engine()
    profile = SimpleNamespace()
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "DI_SPREAD_CHANGE"
    ) == 5.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "DIRECTIONAL_DI_CHANGE"
    ) == 6.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "OPPOSING_DI_CHANGE"
    ) == -2.0


def test_support_resistance_categorical_rule_values_use_prepared_context():
    engine = _engine()
    engine._analyze_support_resistance = lambda _i, _direction: _sr_context()
    profile = SimpleNamespace()

    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_NEAR_SUPPORT"
    ) == 1.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_NEAR_RESISTANCE"
    ) == 0.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_SUPPORT_STATE"
    ) == 4.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_TRADE_LOCATION_RATING"
    ) == 1.0


def test_support_resistance_numeric_rule_values_reuse_one_prepared_context():
    engine = _engine()
    calls = []

    def context(i, direction):
        calls.append((i, direction))
        return _sr_context()

    engine._analyze_support_resistance = context
    profile = SimpleNamespace()
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_ROOM_IN_DIRECTION_ATR"
    ) == 2.5
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_SUPPORT_DISTANCE_ATR"
    ) == 0.25
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "SR_SUPPORT_REJECTION_ATR"
    ) == 0.8
    assert calls == [(0, "LONG")]


def test_sr_rule_evidence_is_unavailable_when_sr_calculation_is_disabled():
    engine = _engine()
    engine.config.enable_support_resistance_analysis = False
    engine._analyze_support_resistance = lambda *_args: _sr_context()
    value = engine._strategy_profile_rule_value(
        0, "LONG", SimpleNamespace(), "SR_NEAR_SUPPORT"
    )
    assert np.isnan(value)


def test_global_pressure_allow_list_is_retired_in_current_native_runtime():
    engine = _engine()
    assert engine._di_pressure_filter_result(0) == (True, None)


def test_legacy_sr_preset_filter_is_retired_in_current_native_runtime():
    engine = _engine()
    assert engine._should_reject_for_sr(0, "LONG", _sr_context()) == (False, None)


def test_public_entry_decision_reports_native_post_flip_side():
    engine = _engine()
    engine.close = np.array([100.0])
    profile = SimpleNamespace(flip_direction=True, entry_rules=(), flip_rule_match_mode="ALL")
    engine._strategy_profile_filter_result = lambda index: (True, "passed")
    engine._profile_context = lambda index: ("BULL", "LONG", "BULL_LONG", profile)
    decision = engine.evaluate_prepared_entry(0)
    assert decision.accepted
    assert decision.strategy_profile_key == "BULL_LONG"
    assert decision.side == "SHORT"
    assert decision.reference_price == 100.0
