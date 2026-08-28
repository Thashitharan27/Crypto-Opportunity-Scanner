from dataclasses import fields
from pathlib import Path

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.config_logic import DEFAULT_GUI_CONFIG

REMOVED_FIELDS = {
    "short_vwap_minimum_distance_atr",
    "long_momentum_minimum_return",
    "directional_long_adx_maximum",
    "enable_directional_adx_filter",
    "directional_short_adx_minimum",
    "enable_long_momentum_filter",
    "enable_bull_regime_short_filter",
    "bear_regime_adx_minimum",
    "enable_short_vwap_distance_filter",
    "enable_bear_regime_adx_filter",
    "long_momentum_lookback_hours",
    "biased_short_adx_maximum",
    "enable_biased_short_adx_cap",
}


def test_retired_oneoff_entry_filter_fields_are_gone():
    names = {f.name for f in fields(BacktestConfig)}
    assert REMOVED_FIELDS.isdisjoint(names)
    assert REMOVED_FIELDS.isdisjoint(DEFAULT_GUI_CONFIG)


def test_production_code_has_no_retired_oneoff_entry_filter_references():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "crypto_strategy_lab/config.py",
        "crypto_strategy_lab/engine.py",
        "crypto_strategy_lab/output_manager.py",
        "crypto_strategy_lab/gui/config_logic.py",
        "crypto_strategy_lab/gui/main_window.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        for field in REMOVED_FIELDS:
            assert field not in text, f"stale {field} reference in {rel}"


def test_removed_long_momentum_telemetry_array_does_not_return():
    root = Path(__file__).resolve().parents[1]
    engine_text = (root / "crypto_strategy_lab/engine.py").read_text(encoding="utf-8")
    assert "long_momentum_return_values" not in engine_text
