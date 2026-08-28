import os
import sys

import pytest

qtwidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
QApplication = qtwidgets.QApplication

from crypto_strategy_lab.gui.main_window import MainWindow


def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication(sys.argv)


def test_sr_tab_uses_clear_user_facing_sections_and_correct_hold_semantics():
    app(); window=MainWindow()
    try:
        assert window.enable_support_resistance_analysis.text()=="Enable Support & Resistance"
        assert window.sr_apply_entry_rules.text()=="Filter Entries"
        assert window.sr_entry_rules_box.title()=="Entry Filters"
        assert window.sr_proximity_box.title()=="Price Proximity"
        assert window.sr_interaction_box.title()=="Level Interaction"
        assert window.sr_detection_box.title()=="Level Detection"
        assert window.enable_sr_hold_confirmation.text()=="Confirm level hold after a test"
        assert "does not control when a level is marked BROKEN" in window.enable_sr_hold_confirmation.toolTip()
        assert window.sr_detection_preset.currentText()=="Balanced (Recommended)"
        assert window.sr_detection_advanced.title()=="Advanced Detection Settings"
    finally: window.close()


def test_sr_analysis_only_is_non_invasive_and_entry_filters_enable_only_in_filter_mode():
    app(); window=MainWindow()
    try:
        window.enable_support_resistance_analysis.setChecked(True)
        window.sr_analyze_only.setChecked(True)
        window.update_dynamic()
        assert not window.sr_entry_rules_box.isEnabled()
        assert "NONE — ANALYSIS ONLY" in window.sr_strategy_status.text()
        assert window.values()["sr_filter_mode"]=="ANALYSIS_ONLY"

        window.sr_apply_entry_rules.setChecked(True)
        window.update_dynamic()
        assert window.sr_entry_rules_box.isEnabled()
        assert "ENTRY FILTER ACTIVE" in window.sr_strategy_status.text()
        assert window.values()["sr_filter_mode"]=="APPLY_ENTRY_RULES"
    finally: window.close()


def test_sr_summary_explains_mode_detection_proximity_and_directional_filters():
    app(); window=MainWindow()
    try:
        window.enable_support_resistance_analysis.setChecked(True)
        window.sr_apply_entry_rules.setChecked(True)
        window.sr_long_require_near_support.setChecked(True)
        window.sr_short_avoid_near_support.setChecked(True)
        window.sr_long_min_room_to_resistance_atr.setValue(1.5)
        window.update_dynamic()
        summary=window.sr_summary_label.text()
        assert "Mode: Filter Entries" in summary
        assert "Detection: Balanced (Recommended)" in summary
        assert "Near level: ≤ 0.75 ATR" in summary
        assert "LONG filters: Require near support, Minimum room: 1.50 ATR" in summary
        assert "SHORT filters: Avoid near support" in summary
    finally: window.close()


def test_sr_balanced_preset_still_maps_to_existing_engine_values():
    app(); window=MainWindow()
    try:
        window.sr_detection_preset.setCurrentText("Sensitive")
        assert window.sr_pivot_left.value()==3
        assert window.sr_pivot_right.value()==3
        assert window.sr_lookback_bars.value()==150
        assert window.sr_zone_width_atr.value()==pytest.approx(0.35)
        assert window.sr_break_tolerance_atr.value()==pytest.approx(0.15)

        window.sr_detection_preset.setCurrentText("Balanced (Recommended)")
        assert window.sr_pivot_left.value()==5
        assert window.sr_pivot_right.value()==5
        assert window.sr_lookback_bars.value()==200
        assert window.sr_zone_width_atr.value()==pytest.approx(0.5)
        assert window.sr_break_tolerance_atr.value()==pytest.approx(0.25)
    finally: window.close()
