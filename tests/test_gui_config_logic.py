import json
from pathlib import Path

import pytest

from crypto_strategy_lab.config import RiskMode
from crypto_strategy_lab.gui.config_logic import (
    CONFIG_VERSION,
    build_backtest_config,
    load_config_json,
    parse_percentage,
    save_config_json,
    validate_config_values,
)


def base(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("timestamp,open,high,low,close,volume\n2024-01-01,1,1,1,1,1\n")
    return {"input_csv": str(csv), "output_dir": str(tmp_path / "out"), "use_intrabar_data": False}


def test_percent_conversions():
    assert parse_percentage("0.5%") == pytest.approx(0.005)
    assert parse_percentage("0.05%") == pytest.approx(0.0005)


def test_atr_mode_creates_config(tmp_path):
    cfg = build_backtest_config({**base(tmp_path), "risk_mode": "ATR", "atr_period": 14, "atr_multiplier": 1.5})
    assert cfg.risk_mode == RiskMode.ATR
    assert cfg.atr_period == 14
    assert cfg.atr_multiplier == pytest.approx(1.5)


def test_percentage_mode_creates_config(tmp_path):
    cfg = build_backtest_config({**base(tmp_path), "risk_mode": "PERCENT", "percent_r": parse_percentage("0.20%")})
    assert cfg.risk_mode == RiskMode.PERCENT
    assert cfg.percent_r == pytest.approx(0.002)


def test_fixed_mode_creates_config(tmp_path):
    cfg = build_backtest_config({**base(tmp_path), "risk_mode": "FIXED", "fixed_r": 100})
    assert cfg.risk_mode == RiskMode.FIXED
    assert cfg.fixed_r == pytest.approx(100)


def test_invalid_current_values_and_retired_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Risk per trade"):
        build_backtest_config({**base(tmp_path), "risk_per_leg": 1})
    with pytest.raises(ValueError, match="Unknown/retired configuration settings: sl_mult"):
        build_backtest_config({**base(tmp_path), "sl_mult": 2.0})


def test_saved_configuration_loads_correctly(tmp_path):
    path = tmp_path / "cfg.json"
    save_config_json(path, {**base(tmp_path), "risk_mode": "FIXED", "fixed_r": 123, "risk_per_leg": 0.005})
    loaded = load_config_json(path)
    assert loaded["config_version"] == CONFIG_VERSION
    cfg = build_backtest_config(loaded)
    assert cfg.fixed_r == pytest.approx(123)
    assert cfg.risk_per_leg == pytest.approx(0.005)


def test_optional_report_settings_round_trip(tmp_path):
    path = tmp_path / "report-settings.json"
    save_config_json(
        path,
        {
            **base(tmp_path),
            "save_feature_analysis_reports": False,
            "save_indicator_analysis_reports": False,
            "create_standard_charts": False,
        },
    )
    cfg = build_backtest_config(load_config_json(path))
    assert cfg.save_feature_analysis_reports is False
    assert cfg.save_indicator_analysis_reports is False
    assert cfg.create_standard_charts is False


def test_gui_passes_selected_intrabar_csv_and_enabled_flag(tmp_path):
    strategy = tmp_path / "strategy.csv"
    intrabar = tmp_path / "intrabar.csv"
    content = "timestamp,open,high,low,close,volume\n2024-01-01,1,1,1,1,1\n"
    strategy.write_text(content)
    intrabar.write_text(content)
    cfg = build_backtest_config(
        {
            **base(tmp_path),
            "input_csv": str(strategy),
            "intrabar_csv": str(intrabar),
            "use_intrabar_data": True,
        }
    )
    assert cfg.intrabar_csv == intrabar
    assert cfg.use_intrabar_data is True
    assert cfg.intrabar_timeframe_minutes == 1


def test_default_gui_config_returns_copy_and_does_not_mutate_source():
    from crypto_strategy_lab.gui.config_logic import DEFAULT_GUI_CONFIG, default_gui_config

    first = default_gui_config()
    second = default_gui_config()

    assert first == DEFAULT_GUI_CONFIG
    assert first is not DEFAULT_GUI_CONFIG
    assert first is not second

    first["atr_period"] = 99
    assert DEFAULT_GUI_CONFIG["atr_period"] == 14
    assert second["atr_period"] == 14


def test_standard_analysis_preset_is_default():
    from crypto_strategy_lab.gui.config_logic import default_gui_config

    values = default_gui_config()
    assert values["enable_trade_telemetry"] is False
    assert values["enable_indicator_lifecycle_analysis"] is False
    assert values["save_feature_analysis_reports"] is False
    assert values["analysis_level"] == "STANDARD"
    assert values["save_indicator_analysis_reports"] is True
    assert values["create_standard_charts"] is True


def test_new_gui_defaults_to_btc_structural_regime():
    from crypto_strategy_lab.gui.config_logic import default_gui_config

    values = default_gui_config()
    assert values["market_regime_method"] == "BTC_STRUCTURAL"
    assert values["structural_regime_sma_days"] == 200
    assert values["structural_regime_slope_lookback_days"] == 30


def test_configurable_timeframes_are_passed_to_backtest_config(tmp_path):
    cfg = build_backtest_config(
        {
            **base(tmp_path),
            "strategy_timeframe_minutes": 60,
            "intrabar_timeframe_minutes": 5,
            "telemetry_interval_minutes": 60,
        }
    )
    assert cfg.strategy_timeframe_minutes == 60
    assert cfg.intrabar_timeframe_minutes == 5


def test_intrabar_timeframe_must_be_lower_only_when_enabled(tmp_path):
    values = {**base(tmp_path), "strategy_timeframe_minutes": 5, "intrabar_timeframe_minutes": 5, "use_intrabar_data": True}
    with pytest.raises(ValueError, match="Intrabar timeframe must be smaller"):
        build_backtest_config(values)

    cfg = build_backtest_config({**values, "use_intrabar_data": False})
    assert cfg.use_intrabar_data is False


def test_support_resistance_settings_round_trip_and_validation(tmp_path):
    rules = {
        "sr_long_avoid_near_resistance": True,
        "sr_long_require_near_support": True,
        "sr_long_block_broken_support": True,
        "sr_long_min_room_to_resistance_atr": 1.5,
        "sr_short_avoid_near_support": True,
        "sr_short_require_near_resistance": True,
        "sr_short_block_broken_resistance": True,
        "sr_short_min_room_to_support_atr": 1.75,
    }
    values = {
        **base(tmp_path),
        "enable_support_resistance_analysis": True,
        "sr_filter_mode": "APPLY_ENTRY_RULES",
        **rules,
    }
    assert not validate_config_values(values)
    cfg = build_backtest_config(values)
    for key, value in rules.items():
        assert getattr(cfg, key) == value

    path = tmp_path / "sr-config.json"
    save_config_json(path, values)
    loaded = load_config_json(path)
    assert loaded["sr_filter_mode"] == "APPLY_ENTRY_RULES"
    for key, value in rules.items():
        assert loaded[key] == value
    cfg2 = build_backtest_config(loaded)
    for key, value in rules.items():
        assert getattr(cfg2, key) == value


def test_support_resistance_mode_and_unknown_keys_are_strict(tmp_path):
    values = {
        **base(tmp_path),
        "enable_support_resistance_analysis": True,
        "sr_filter_mode": "analysis only",
    }
    assert "Invalid support/resistance usage mode." in validate_config_values(values)

    path = tmp_path / "unknown-config.json"
    path.write_text(json.dumps({"config_version": CONFIG_VERSION, **values, "unknown_setting": True}))
    with pytest.raises(ValueError, match="Unknown/retired configuration settings: unknown_setting"):
        load_config_json(path)


def test_old_config_version_is_rejected(tmp_path):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"config_version": 1, **base(tmp_path)}))
    with pytest.raises(ValueError, match="Configuration version 2 is required"):
        load_config_json(path)
