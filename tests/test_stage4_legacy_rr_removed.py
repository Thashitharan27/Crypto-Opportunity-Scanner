from dataclasses import fields
from pathlib import Path

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.config_logic import DEFAULT_GUI_CONFIG


REMOVED_FIELDS = {
    "di_reward_risk_ratio",
    "di_long_reward_risk_ratio",
    "di_short_reward_risk_ratio",
    "enable_di_regime_reward_risk",
    "di_regime_bear_return_threshold",
    "di_long_bull_reward_risk_ratio",
    "di_long_bear_reward_risk_ratio",
    "di_long_sideways_reward_risk_ratio",
    "di_short_bull_reward_risk_ratio",
    "di_short_bear_reward_risk_ratio",
    "di_short_sideways_reward_risk_ratio",
    "enable_bull_long_conditional_reward_risk",
    "enable_sideways_long_conditional_reward_risk",
    "enable_sideways_short_conditional_reward_risk",
    "enable_bear_short_conditional_reward_risk",
}


def test_legacy_di_reward_risk_fields_are_not_configurable_or_serialized():
    config_fields = {field.name for field in fields(BacktestConfig)}
    assert REMOVED_FIELDS.isdisjoint(config_fields)
    assert REMOVED_FIELDS.isdisjoint(DEFAULT_GUI_CONFIG)


def test_production_code_has_no_removed_reward_risk_config_references():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "crypto_strategy_lab/config.py",
        "crypto_strategy_lab/engine.py",
        "crypto_strategy_lab/gui/config_logic.py",
        "crypto_strategy_lab/gui/main_window.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        for field in REMOVED_FIELDS:
            assert field not in text, f"stale {field} reference in {rel}"


def test_config_has_no_stale_legacy_regime_ratio_validation_symbol():
    root = Path(__file__).resolve().parents[1]
    config_text = (root / "crypto_strategy_lab/config.py").read_text(encoding="utf-8")
    assert "regime_ratios" not in config_text
