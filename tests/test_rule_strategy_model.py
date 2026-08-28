from dataclasses import asdict

import pytest

from crypto_strategy_lab.data_lake_config import ExecutionProfileConfig
from crypto_strategy_lab.strategy_profiles import StrategyProfile
from crypto_strategy_lab.strategy_rule_model import (
    MARKET_PERMISSIONS,
    SUPPORT_RESISTANCE_RULE_EVIDENCE,
    compile_profiles,
    decompile_rules,
    new_rule,
    normalize_rule,
    rule_operator_options,
    uses_support_resistance_rules,
)


def test_required_di_spread_rule_compiles_to_reject_when_requirement_fails():
    rule = new_rule(kind="REQUIRED", evidence="DI_SPREAD")
    rule.update(operator="GTE", value=30.0, regime="ALL", side="ALL")

    strategy, execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(rule,),
    )

    assert set(strategy) == {
        "bull_long", "bull_short", "bear_long", "bear_short",
        "sideways_long", "sideways_short",
    }
    for key, profile in strategy.items():
        assert profile.enabled is True
        assert profile.flip_direction is False
        assert profile.reject_rule_match_mode == "ANY"
        assert len(profile.entry_rules) == 1
        native = profile.entry_rules[0]
        assert native["action"] == "REJECT"
        assert native["indicator"] == "DI_SPREAD"
        assert native["condition"] == "OUTSIDE"
        assert native["minimum"] == 30.0
        assert native["maximum"] > 1e300
        mature = StrategyProfile(**{
            **asdict(profile),
            **asdict(execution[key]),
        })
        mature.validate(key)


def test_required_expanding_pressure_compiles_as_categorical_requirement():
    rule = new_rule(kind="REQUIRED", evidence="DI_PRESSURE_STATE")
    rule.update(operator="IS", value="EXPANDING", regime="ALL", side="ALL")

    strategy, execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(rule,),
    )

    native = strategy["bull_long"].entry_rules[0]
    assert native["action"] == "REJECT"
    assert native["indicator"] == "DI_PRESSURE_STATE"
    assert native["condition"] == "OUTSIDE"
    assert native["minimum"] == native["maximum"] == 1.0
    StrategyProfile(**{
        **asdict(strategy["bull_long"]),
        **asdict(execution["bull_long"]),
    }).validate("bull_long")


def test_contracting_pressure_veto_rejects_matching_state():
    rule = new_rule(kind="VETO", evidence="DI_PRESSURE_STATE")
    rule.update(operator="IS", value="CONTRACTING")
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        veto_rules=(rule,),
    )
    native = strategy["bear_short"].entry_rules[0]
    assert native["indicator"] == "DI_PRESSURE_STATE"
    assert native["condition"] == "INSIDE"
    assert native["minimum"] == native["maximum"] == 2.0


def test_pressure_state_supports_only_categorical_operators_and_values():
    with pytest.raises(ValueError, match="unsupported operator"):
        normalize_rule({
            "kind": "REQUIRED",
            "evidence": "DI_PRESSURE_STATE",
            "operator": "GTE",
            "value": "EXPANDING",
        })
    with pytest.raises(ValueError, match="unsupported value"):
        normalize_rule({
            "kind": "REQUIRED",
            "evidence": "DI_PRESSURE_STATE",
            "operator": "IS",
            "value": "UNKNOWN",
        })


def test_numeric_pressure_change_evidence_keeps_numeric_operators():
    rule = new_rule(kind="REQUIRED", evidence="DIRECTIONAL_DI_CHANGE")
    rule.update(operator="GTE", value=4.0)
    normalized = normalize_rule(rule)
    assert normalized["value"] == 4.0
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(normalized,),
    )
    native = strategy["bull_long"].entry_rules[0]
    assert native["indicator"] == "DIRECTIONAL_DI_CHANGE"
    assert native["condition"] == "OUTSIDE"
    assert native["minimum"] == 4.0


def test_directional_di_is_separate_numeric_evidence_with_strict_and_range_operators():
    assert rule_operator_options("DIRECTIONAL_DI") == (
        "GT", "GTE", "LT", "LTE", "BETWEEN", "OUTSIDE"
    )
    rule = new_rule(kind="REQUIRED", evidence="DIRECTIONAL_DI")
    rule.update(operator="GT", value=30.0)
    normalized = normalize_rule(rule)
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(normalized,),
    )
    native = strategy["bull_long"].entry_rules[0]
    assert native["indicator"] == "DIRECTIONAL_DI"
    # Required Directional DI > 30 rejects the inclusive complement (<= 30).
    assert native["condition"] == "INSIDE"
    assert native["minimum"] < -1e300
    assert native["maximum"] == 30.0


def test_required_near_support_compiles_as_categorical_sr_requirement():
    rule = new_rule(kind="REQUIRED", evidence="SR_NEAR_SUPPORT")
    rule.update(operator="IS", value="TRUE", side="LONG")
    strategy, execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(rule,),
    )

    native = strategy["bull_long"].entry_rules[0]
    assert native["indicator"] == "SR_NEAR_SUPPORT"
    assert native["condition"] == "OUTSIDE"
    assert native["minimum"] == native["maximum"] == 1.0
    assert not strategy["bull_short"].entry_rules
    StrategyProfile(**{
        **asdict(strategy["bull_long"]),
        **asdict(execution["bull_long"]),
    }).validate("bull_long")


def test_broken_support_veto_and_room_requirement_compile_through_generic_rules():
    broken = new_rule(kind="VETO", evidence="SR_SUPPORT_STATE")
    broken.update(operator="IS", value="SUPPORT_BROKEN", side="LONG")
    room = new_rule(kind="REQUIRED", evidence="SR_ROOM_IN_DIRECTION_ATR")
    room.update(operator="GTE", value=2.0, side="LONG")
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(room,),
        veto_rules=(broken,),
    )

    native = strategy["bear_long"].entry_rules
    assert [rule["indicator"] for rule in native] == [
        "SR_ROOM_IN_DIRECTION_ATR", "SR_SUPPORT_STATE"
    ]
    assert native[0]["condition"] == "OUTSIDE"
    assert native[0]["minimum"] == 2.0
    assert native[1]["condition"] == "INSIDE"
    assert native[1]["minimum"] == native[1]["maximum"] == 5.0


def test_support_resistance_rule_dependency_is_detected_from_any_rule_group():
    ordinary = new_rule(kind="REQUIRED", evidence="DI_SPREAD")
    sr = new_rule(kind="VETO", evidence="SR_NEAR_RESISTANCE")
    assert not uses_support_resistance_rules((ordinary,))
    assert uses_support_resistance_rules((ordinary,), (sr,))
    assert "SR_ROOM_IN_DIRECTION_ATR" in SUPPORT_RESISTANCE_RULE_EVIDENCE


def test_scoped_veto_only_reaches_matching_regime_and_side():
    rule = new_rule(kind="VETO", evidence="ADX")
    rule.update(operator="LTE", value=20.0, regime="BULL", side="LONG")
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        veto_rules=(rule,),
    )

    assert len(strategy["bull_long"].entry_rules) == 1
    native = strategy["bull_long"].entry_rules[0]
    assert native["condition"] == "INSIDE"
    assert native["maximum"] == 20.0
    assert not strategy["bull_short"].entry_rules
    assert not strategy["bear_long"].entry_rules


def test_long_only_trading_is_expressed_by_permissions_not_direction_override():
    long_permissions = ("BULL_LONG", "BEAR_LONG", "SIDEWAYS_LONG")
    rule = new_rule(kind="REQUIRED", evidence="DI_SPREAD")
    rule.update(operator="GTE", value=30.0, side="LONG")
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=long_permissions,
        required_rules=(rule,),
    )

    assert strategy["bull_long"].enabled is True
    assert strategy["bull_short"].enabled is False
    assert strategy["bear_long"].enabled is True
    assert strategy["bear_short"].enabled is False
    assert strategy["sideways_long"].enabled is True
    assert strategy["sideways_short"].enabled is False
    assert all(profile.flip_direction is False for profile in strategy.values())
    assert len(strategy["bull_long"].entry_rules) == 1
    assert not strategy["bull_short"].entry_rules


def test_forced_long_and_short_direction_modes_are_retired():
    for retired in ("LONG_ONLY", "SHORT_ONLY"):
        with pytest.raises(ValueError, match="unsupported direction mode"):
            compile_profiles(
                direction_mode=retired,
                market_permissions=MARKET_PERMISSIONS,
            )


def test_builder_metadata_round_trips_scoped_rules_without_profile_ui():
    required = new_rule(kind="REQUIRED", evidence="RSI")
    required.update(
        operator="BETWEEN", value=20.0, value2=45.0,
        regime="BULL", side="LONG",
    )
    veto = new_rule(kind="VETO", evidence="SR_TRADE_LOCATION_RATING")
    veto.update(
        operator="IS", value="BAD_LOCATION", regime="ALL", side="SHORT"
    )

    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(required,),
        veto_rules=(veto,),
    )
    recovered = decompile_rules(strategy)

    assert recovered["REQUIRED"] == (normalize_rule(required),)
    assert recovered["VETO"] == (normalize_rule(veto),)
    assert recovered["FLIP"] == ()


def test_one_base_execution_plan_is_copied_to_internal_engine_inputs():
    base = ExecutionProfileConfig(
        stop_loss_multiple=3.0,
        reward_risk_ratio=2.0,
        break_even_enabled=True,
        break_even_activation_r=1.0,
    )
    _strategy, execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
        base_execution=base,
    )
    assert all(profile == base for profile in execution.values())
