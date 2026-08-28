from __future__ import annotations

import os

import pytest

from crypto_strategy_lab.gui.research_feature_ownership import apply_research_feature_ownership
from crypto_strategy_lab.gui.risk_execution_install import apply_risk_execution_workspace


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.rule_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def coverage(self, _request):
            return []

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    window = MainWindow(service=Service())
    apply_research_feature_ownership(window)
    apply_risk_execution_workspace(window)
    return app, window


def test_risk_page_is_reorganized_around_one_execution_plan():
    app, window = _window()
    try:
        from PySide6.QtWidgets import QGroupBox

        page = window.pages.widget(3)
        titles = {
            box.title()
            for box in page.findChildren(QGroupBox)
            if not box.isHidden()
        }
        assert "Effective Plan" in titles
        assert "1. Account Risk & Position Sizing" in titles
        assert "2. Stop & Position Sizing" in titles
        assert "3. Profit Target" in titles
        assert "4. Trade Management" in titles
        assert "Advanced Trade Management" in titles
        assert "Exposure & Costs" in titles
        assert "Base Trade Management" not in titles

        workspace = window.risk_execution_workspace
        assert window.execution_form.isHidden()
        assert window.base_execution_form.isHidden()
        assert "effective risk budget" in workspace.summary_label.text().lower()
        assert "ATR distance unit" in workspace.summary_label.text()
    finally:
        window.close()
        app.processEvents()


def test_stop_distance_method_shows_only_the_active_parameter():
    app, window = _window()
    try:
        workspace = window.risk_execution_workspace
        mode = window.execution_form.widgets["risk_mode"]
        atr = window.execution_form.widgets["atr_multiplier"]
        percent = window.execution_form.widgets["percent_r"]
        fixed = window.execution_form.widgets["fixed_r"]

        mode.setCurrentIndex(mode.findData("ATR"))
        workspace.refresh_visibility()
        assert not atr.isHidden()
        assert percent.isHidden()
        assert fixed.isHidden()

        mode.setCurrentIndex(mode.findData("PERCENT"))
        workspace.refresh_visibility()
        assert atr.isHidden()
        assert not percent.isHidden()
        assert fixed.isHidden()

        mode.setCurrentIndex(mode.findData("FIXED"))
        workspace.refresh_visibility()
        assert atr.isHidden()
        assert percent.isHidden()
        assert not fixed.isHidden()
    finally:
        window.close()
        app.processEvents()


def test_optional_trade_management_reveals_settings_only_when_enabled():
    app, window = _window()
    try:
        workspace = window.risk_execution_workspace
        enabled = window.base_execution_form.widgets["break_even_enabled"]
        activation = window.base_execution_form.widgets["break_even_activation_r"]
        offset = window.base_execution_form.widgets["break_even_offset_r"]

        enabled.setChecked(False)
        workspace.refresh_visibility()
        assert activation.isHidden()
        assert offset.isHidden()

        enabled.setChecked(True)
        workspace.refresh_visibility()
        assert not activation.isHidden()
        assert not offset.isHidden()
    finally:
        window.close()
        app.processEvents()


def test_sr_target_policy_automatically_owns_support_resistance_dependency():
    app, window = _window()
    try:
        target_mode = window.execution_form.widgets["sr_take_profit_mode"]
        sr_toggle = window.feature_form.widgets["enable_support_resistance_analysis"]
        panel = window.research_features_panel

        sr_toggle.setChecked(False)
        target_mode.setCurrentIndex(target_mode.findData("SR_CAPPED_R"))
        app.processEvents()

        assert sr_toggle.isChecked() is True
        assert sr_toggle.isEnabled() is False
        assert panel.sr_card.status.text() == "REQUIRED BY TARGET POLICY"
        assert not window.execution_form.widgets["sr_take_profit_minimum_r"].isHidden()
        assert window.build_config().features.enable_support_resistance_analysis is True

        target_mode.setCurrentIndex(target_mode.findData("FIXED_R"))
        app.processEvents()
        assert sr_toggle.isEnabled() is True
        assert sr_toggle.isChecked() is False
        assert window.execution_form.widgets["sr_take_profit_minimum_r"].isHidden()
    finally:
        window.close()
        app.processEvents()
