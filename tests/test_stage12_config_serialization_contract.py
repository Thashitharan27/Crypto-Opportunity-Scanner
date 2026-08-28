import json

import pytest

from crypto_strategy_lab.gui.config_logic import (
    CONFIG_VERSION,
    canonical_config_values,
    default_gui_config,
    load_config_json,
    save_config_json,
)


def test_canonical_config_preserves_shared_settings_and_current_profile_rules():
    values = default_gui_config()
    values.update({
        "market_symbol": "SOLUSDT",
        "risk_mode": "FIXED",
        "fixed_r": 125.0,
        "maker_fee": 0.00017,
        "enable_di_pressure_analysis": True,
        "di_pressure_lookback": 7,
        "enable_support_resistance_analysis": True,
        "sr_filter_mode": "APPLY_ENTRY_RULES",
        "market_regime_method": "BTC_STRUCTURAL",
        "structural_regime_sma_days": 210,
    })
    bull_long = values["strategy_profiles"]["bull_long"]
    bull_long["reward_risk_ratio"] = 2.75
    bull_long["entry_rules"] = [
        {"action": "REJECT", "indicator": "ADX", "condition": "OUTSIDE", "minimum": 22.0, "maximum": 35.0}
    ]

    saved = canonical_config_values(values)

    assert saved["config_version"] == CONFIG_VERSION
    assert saved["market_symbol"] == "SOLUSDT"
    assert saved["risk_mode"] == "FIXED"
    assert saved["fixed_r"] == 125.0
    assert saved["maker_fee"] == 0.00017
    assert saved["di_pressure_lookback"] == 7
    assert saved["sr_filter_mode"] == "APPLY_ENTRY_RULES"
    assert saved["structural_regime_sma_days"] == 210
    assert saved["strategy_profiles"]["bull_long"]["reward_risk_ratio"] == 2.75
    assert saved["strategy_profiles"]["bull_long"]["entry_rules"][0]["indicator"] == "ADX"
    assert "enable_strategy_profiles" not in saved


def test_canonical_config_rejects_retired_and_unknown_keys():
    for key, value in (
        ("sl_mult", 9.0),
        ("enable_partial_take_profit", True),
        ("enable_random_entry", True),
        ("enable_remaining_leg_timeout_after_first_sl", True),
        ("unknown_setting", "reject-me"),
    ):
        values = default_gui_config()
        values[key] = value
        with pytest.raises(ValueError, match="Unknown/retired configuration settings"):
            canonical_config_values(values)


def test_save_load_round_trip_keeps_only_current_v2_contract(tmp_path):
    values = default_gui_config()
    values.update({
        "market_symbol": "ETHUSDT",
        "risk_mode": "PERCENT",
        "percent_r": 0.004,
        "enable_support_resistance_analysis": True,
        "sr_near_distance_atr": 1.25,
    })
    profile = values["strategy_profiles"]["sideways_short"]
    profile["enabled"] = True
    profile["reward_risk_ratio"] = 3.25
    profile["entry_rules"] = [
        {"action": "REJECT", "indicator": "BB_WIDTH", "condition": "OUTSIDE", "minimum": 0.04, "maximum": 0.10}
    ]

    path = tmp_path / "strategy-config.json"
    save_config_json(path, values)
    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_config_json(path)

    assert raw["config_version"] == CONFIG_VERSION
    assert "enable_strategy_profiles" not in raw
    assert "sl_mult" not in raw
    assert raw["market_symbol"] == "ETHUSDT"
    assert raw["sr_near_distance_atr"] == 1.25
    assert raw["strategy_profiles"]["sideways_short"]["reward_risk_ratio"] == 3.25
    assert loaded["market_symbol"] == "ETHUSDT"
    assert loaded["percent_r"] == 0.004
    assert loaded["strategy_profiles"]["sideways_short"]["entry_rules"][0]["indicator"] == "BB_WIDTH"

    raw["sl_mult"] = 99.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown/retired configuration settings: sl_mult"):
        load_config_json(path)
