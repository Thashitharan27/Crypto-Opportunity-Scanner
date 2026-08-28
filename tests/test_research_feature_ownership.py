from __future__ import annotations

import os

import pytest

from crypto_strategy_lab.data_lake_config import FeatureConfig
from crypto_strategy_lab.gui.research_feature_ownership import (
    apply_research_feature_ownership,
)
from crypto_strategy_lab.gui.rule_strategy_builder import RuleStrategyBuilder
from crypto_strategy_lab.gui.v2_main_window import DataclassForm, FEATURE_GROUPS
from crypto_strategy_lab.strategy_rule_model import new_rule


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    app = widgets.QApplication.instance() or widgets.QApplication([])

    window = widgets.QWidget()
    window.pages = widgets.QStackedWidget(window)

    strategy_page = widgets.QWidget()
    strategy_layout = widgets.QVBoxLayout(strategy_page)
    strategy_layout.addWidget(widgets.QLabel("Strategy Builder"))
    window.pages.addWidget(strategy_page)

    feature_page = widgets.QWidget()
    feature_layout = widgets.QVBoxLayout(feature_page)
    feature_layout.addWidget(widgets.QLabel("Research Features"))
    window.feature_form = DataclassForm(FeatureConfig(), groups=FEATURE_GROUPS)
    scroll = widgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(window.feature_form)
    feature_layout.addWidget(scroll)
    window.pages.addWidget(feature_page)

    window.rule_builder = RuleStrategyBuilder()
    strategy_layout.addWidget(window.rule_builder)
    return app, window, scroll


def test_research_features_are_reorganized_without_replacing_authoritative_form():
    app, window, scroll = _window()
    try:
        original_form = window.feature_form
        apply_research_feature_ownership(window)

        assert window.feature_form is original_form
        assert window.feature_form.isHidden()
        assert scroll.widget() is window.research_features_panel
        assert window.mean_reversion_research_box is window.research_features_panel.mr_card
        assert window.rule_builder.advanced.title() == "4. Advanced"

        panel = window.research_features_panel
        assert panel.price_card.status.text() == "AUTOMATIC"
        assert panel.di_card.status.text() == "REQUIRED BY STRATEGY"
        assert panel.regime_card.status.text() == "REQUIRED BY STRATEGY"
        assert panel.oi_card.status.text() == "AUTO · RULE-READY"
        assert panel.funding_card.status.text() == "AUTO · RULE-READY"
        assert panel.basis_card.status.text() == "AUTO · RULE-READY"
        assert panel.taker_card.status.text() == "AUTO · RULE-READY"
        assert panel.trade_card.status.text() == "HEAVY · OFF"
        assert panel.book_card.status.text() == "HEAVY · OFF"

        # Re-parenting is presentation-only: the native form still serializes the
        # same FeatureConfig values.
        result = window.feature_form.value(FeatureConfig())
        assert result.atr_period == 14
        assert result.market_regime_method == "BTC_STRUCTURAL"
        assert result.trade_flow_enabled is False
        assert result.order_book_enabled is False

        # Idempotent final composition.
        original_panel = window.research_features_panel
        apply_research_feature_ownership(window)
        assert window.research_features_panel is original_panel
    finally:
        window.close()
        app.processEvents()


def test_regime_settings_show_only_parameters_for_selected_model():
    app, window, _scroll = _window()
    try:
        apply_research_feature_ownership(window)
        panel = window.research_features_panel
        method = panel.widgets["market_regime_method"]

        method.setCurrentIndex(method.findData("ASSET_RETURN"))
        panel.refresh_visibility()
        # isVisible() also depends on whether the top-level test window was shown;
        # isHidden() tests the explicit visibility state set by the panel itself.
        assert not panel.widgets["bull_regime_lookback_days"].isHidden()
        assert not panel.widgets["bull_regime_return_threshold"].isHidden()
        assert panel.widgets["structural_regime_sma_days"].isHidden()
        assert panel.widgets["structural_regime_slope_lookback_days"].isHidden()
        assert "Asset Return" in panel.regime_detail.text()

        method.setCurrentIndex(method.findData("BTC_STRUCTURAL"))
        panel.refresh_visibility()
        assert not panel.widgets["structural_regime_sma_days"].isHidden()
        assert not panel.widgets["structural_regime_slope_lookback_days"].isHidden()
        assert panel.widgets["bull_regime_lookback_days"].isHidden()
        assert panel.widgets["bull_regime_return_threshold"].isHidden()
        assert "BTCUSDT" in panel.regime_detail.text()
    finally:
        window.close()
        app.processEvents()


def test_presets_control_only_explicit_optional_or_heavy_features():
    app, window, _scroll = _window()
    try:
        apply_research_feature_ownership(window)
        panel = window.research_features_panel

        panel.preset.setCurrentIndex(panel.preset.findData("FAST"))
        panel._apply_selected_preset()
        assert window.rule_builder.enable_mr.isChecked() is False
        assert panel.sr_enable.isChecked() is False
        assert panel.trade_enable.isChecked() is False
        assert panel.book_enable.isChecked() is False

        panel.preset.setCurrentIndex(panel.preset.findData("DEEP"))
        panel._apply_selected_preset()
        assert window.rule_builder.enable_mr.isChecked() is True
        assert panel.sr_enable.isChecked() is True
        assert panel.trade_enable.isChecked() is True
        assert panel.book_enable.isChecked() is True
    finally:
        window.close()
        app.processEvents()


def test_sr_rule_makes_support_resistance_a_required_dependency():
    app, window, _scroll = _window()
    try:
        apply_research_feature_ownership(window)
        panel = window.research_features_panel
        panel.sr_enable.setChecked(False)

        rule = new_rule(kind="REQUIRED", evidence="SR_TRADE_LOCATION_RATING")
        window.rule_builder.required_rules.set_rules((rule,))
        panel._sync_sr_requirement()

        assert panel.sr_enable.isChecked() is True
        assert panel.sr_enable.isEnabled() is False
        assert panel.sr_card.status.text() == "REQUIRED BY STRATEGY"
        assert "required by strategy rule" in panel.sr_enable.text().lower()
    finally:
        window.close()
        app.processEvents()
