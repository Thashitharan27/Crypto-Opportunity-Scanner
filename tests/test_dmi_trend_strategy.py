import numpy as np

from crypto_strategy_lab.rule_native_engine import (
    RuleAwareDataLakeProductionBacktestEngine,
)
from crypto_strategy_lab.strategy_rule_model import (
    MARKET_PERMISSIONS,
    compile_profiles,
    decompile_rules,
    infer_direction_mode,
    new_rule,
    normalize_rule,
)


def test_dmi_trend_adds_fixed_baseline_confirmations_without_changing_permissions():
    strategy, _execution = compile_profiles(
        direction_mode="DMI_TREND",
        market_permissions=MARKET_PERMISSIONS,
    )

    for profile in strategy.values():
        assert profile.enabled is True
        assert [rule["indicator"] for rule in profile.entry_rules] == [
            "ADX",
            "ADX_CHANGE",
            "DI_PRESSURE_STATE",
        ]
        adx, adx_change, pressure = profile.entry_rules
        assert adx["condition"] == "OUTSIDE"
        assert adx["minimum"] == 20.0
        assert adx_change["condition"] == "OUTSIDE"
        assert adx_change["minimum"] == 0.0
        assert pressure["condition"] == "OUTSIDE"
        assert pressure["minimum"] == pressure["maximum"] == 1.0
        assert all(rule["_builder_kind"] == "REQUIRED" for rule in profile.entry_rules)

    assert infer_direction_mode(strategy) == "DMI_TREND"
    assert decompile_rules(strategy) == {
        "REQUIRED": (),
        "VETO": (),
        "FLIP": (),
    }


def test_dmi_trend_keeps_user_rules_separate_and_appends_them_after_baseline():
    user_rule = new_rule(kind="REQUIRED", evidence="DI_SPREAD")
    user_rule.update(operator="GTE", value=25.0, regime="BULL", side="LONG")

    strategy, _execution = compile_profiles(
        direction_mode="DMI_TREND",
        market_permissions=MARKET_PERMISSIONS,
        required_rules=(user_rule,),
    )

    bull_long = strategy["bull_long"].entry_rules
    assert [rule["indicator"] for rule in bull_long] == [
        "ADX",
        "ADX_CHANGE",
        "DI_PRESSURE_STATE",
        "DI_SPREAD",
    ]
    assert [rule["indicator"] for rule in strategy["bull_short"].entry_rules] == [
        "ADX",
        "ADX_CHANGE",
        "DI_PRESSURE_STATE",
    ]
    recovered = decompile_rules(strategy)
    assert recovered["REQUIRED"] == (normalize_rule(user_rule),)


def test_raw_di_direction_remains_the_unfiltered_control_group():
    strategy, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=MARKET_PERMISSIONS,
    )
    assert all(profile.entry_rules == () for profile in strategy.values())
    assert infer_direction_mode(strategy) == "DI"


def test_adx_change_rule_value_is_one_completed_bar_difference():
    engine = object.__new__(RuleAwareDataLakeProductionBacktestEngine)
    engine.adx_values = np.array([19.5, 20.0, 21.25], dtype=float)

    assert np.isnan(
        engine._strategy_profile_rule_value(0, "LONG", None, "ADX_CHANGE")
    )
    assert engine._strategy_profile_rule_value(1, "LONG", None, "ADX_CHANGE") == 0.5
    assert engine._strategy_profile_rule_value(2, "SHORT", None, "ADX_CHANGE") == 1.25
