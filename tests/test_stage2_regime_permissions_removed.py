from dataclasses import fields

from PySide6.QtWidgets import QApplication

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.config_logic import DEFAULT_GUI_CONFIG
from crypto_strategy_lab.gui.main_window import MainWindow
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, default_profiles


LEGACY_REGIME_PERMISSION_FIELDS = {
    "enable_regime_direction_filter",
    "allow_bull_long",
    "allow_bull_short",
    "allow_bear_long",
    "allow_bear_short",
    "allow_sideways_long",
    "allow_sideways_short",
}


def app():
    return QApplication.instance() or QApplication([])


def test_legacy_regime_permission_fields_are_removed_from_config_and_gui_defaults():
    config_fields = {field.name for field in fields(BacktestConfig)}
    assert LEGACY_REGIME_PERMISSION_FIELDS.isdisjoint(config_fields)
    assert LEGACY_REGIME_PERMISSION_FIELDS.isdisjoint(DEFAULT_GUI_CONFIG)


def test_main_window_does_not_create_legacy_regime_permission_widgets():
    app()
    window = MainWindow()
    try:
        for name in LEGACY_REGIME_PERMISSION_FIELDS:
            assert not hasattr(window, name)
    finally:
        window.close()


def test_profile_enabled_is_available_for_every_regime_direction_profile():
    profiles = default_profiles()
    assert set(profiles) == set(PROFILE_KEYS)
    assert all(profile.enabled for profile in profiles.values())
