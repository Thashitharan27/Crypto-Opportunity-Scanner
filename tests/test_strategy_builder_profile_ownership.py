"""Regression tests for Strategy Builder market-permission/profile-override ownership."""
from __future__ import annotations

from dataclasses import replace
import os

import pytest

from crypto_strategy_lab.data_lake_config import ResearchRunConfig


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _window():
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.v2_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def coverage(self, _request):
            return []

        def inventory(self, *_args):
            return []

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    return app, MainWindow(service=Service())


def test_market_permission_is_the_only_profile_enabled_control():
    _app, window = _window()
    try:
        editor = window.profile_editor
        assert "enabled" not in editor.strategy_form.widgets

        before = window.build_config()
        editor.permission_checks["bull_long"].click()
        after = window.build_config()

        assert after.strategy.profiles["bull_long"].enabled is False
        assert replace(after.strategy.profiles["bull_long"], enabled=True) == before.strategy.profiles["bull_long"]
    finally:
        window.close()


def test_paste_overrides_never_changes_target_market_permission():
    _app, window = _window()
    try:
        editor = window.profile_editor
        base = ResearchRunConfig()
        profiles = dict(base.strategy.profiles)
        profiles["bull_long"] = replace(
            profiles["bull_long"],
            enabled=True,
            flip_direction=True,
            rsi_period=9,
        )
        profiles["bear_long"] = replace(profiles["bear_long"], enabled=False)
        config = replace(base, strategy=replace(base.strategy, profiles=profiles))
        window.apply_config(config)

        editor.selector.setCurrentText("bull_long")
        editor.copy_profile()
        editor.selector.setCurrentText("bear_long")
        editor.paste_profile()

        result = window.build_config().strategy.profiles["bear_long"]
        assert result.enabled is False
        assert result.flip_direction is True
        assert result.rsi_period == 9
    finally:
        window.close()


def test_native_calculation_overrides_are_collapsed_and_roundtrip_losslessly():
    _app, window = _window()
    try:
        editor = window.profile_editor
        base = ResearchRunConfig()
        profiles = dict(base.strategy.profiles)
        profiles["bull_long"] = replace(
            profiles["bull_long"],
            rsi_period=11,
            momentum_lookback_hours=37,
        )
        config = replace(base, strategy=replace(base.strategy, profiles=profiles))
        window.apply_config(config)

        editor.selector.setCurrentText("bull_long")
        assert editor.native_calculation_overrides.isHidden()
        assert "rsi_period" not in editor.strategy_form.widgets
        assert "momentum_lookback_hours" not in editor.strategy_form.widgets
        assert editor.native_calculation_widgets["rsi_period"].value() == 11
        assert editor.native_calculation_widgets["momentum_lookback_hours"].value() == 37
        assert window.build_config() == config

        editor.native_calculation_widgets["rsi_period"].setValue(13)
        updated = window.build_config()
        assert updated.strategy.profiles["bull_long"].rsi_period == 13
        assert updated.strategy.profiles["bull_long"].momentum_lookback_hours == 37
        assert updated.strategy.profiles["bull_long"].enabled is True
    finally:
        window.close()
