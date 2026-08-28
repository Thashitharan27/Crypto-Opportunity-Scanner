"""Regression coverage for the hard removal of old pair-management controls."""

from pathlib import Path

from PySide6.QtWidgets import QApplication, QGroupBox

from crypto_strategy_lab.gui.config_logic import default_gui_config
from crypto_strategy_lab.gui.main_window import MainWindow


def app():
    return QApplication.instance() or QApplication([])


def test_hidden_pair_management_widgets_are_not_constructed():
    app()
    window = MainWindow()
    try:
        removed = (
            "both_timeout", "both_timeout_duration", "both_timeout_unit",
            "be_after_sl", "be_mode", "be_offset", "be_same_candle",
            "remaining_leg_timeout", "remaining_leg_timeout_duration", "remaining_leg_timeout_unit",
            "remaining_leg_timeout_profit_extension", "reentry_gate_after_timeout",
            "checkpoint_score_extension", "first_sl_survivor_partial", "zero_score_confirmation",
            "be_label",
        )
        for name in removed:
            assert not hasattr(window, name)
        titles = {box.title() for box in window.findChildren(QGroupBox)}
        assert "Both-Open Timeout" not in titles
        assert "Break-Even After Opposite SL" not in titles
        assert "Remaining Leg Timeout After First SL" not in titles
        assert "Break-Even Calculator" not in titles
    finally:
        window.close()


def test_retired_pair_globals_are_absent_while_profile_timeout_and_be_survive():
    app()
    window = MainWindow()
    try:
        values = default_gui_config()
        profile = values["strategy_profiles"]["bull_long"]
        profile.update({
            "timeout_enabled": True,
            "timeout_minutes": 180,
            "break_even_enabled": True,
            "break_even_activation_r": 1.5,
            "break_even_offset_r": 0.2,
        })
        window.apply_values(values)
        current = window.values()
        for key in (
            "enable_both_open_timeout", "max_both_open_minutes", "enable_be_after_opposite_sl", "be_mode",
            "enable_remaining_leg_timeout_after_first_sl", "enable_remaining_leg_timeout_profit_extension",
            "enable_remaining_leg_checkpoint_score_extension", "enable_first_sl_survivor_partial_close",
            "enable_checkpoint_zero_score_confirmation", "enable_reentry_gate_after_remaining_leg_timeout",
        ):
            assert key not in current
        bull_long = current["strategy_profiles"]["bull_long"]
        assert bull_long["timeout_enabled"] is True
        assert bull_long["timeout_minutes"] == 180
        assert bull_long["break_even_enabled"] is True
        assert bull_long["break_even_activation_r"] == 1.5
        assert bull_long["break_even_offset_r"] == 0.2
    finally:
        window.close()


def test_main_window_source_has_no_hidden_pair_management_plumbing():
    source = (Path(__file__).resolve().parents[1] / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    for fragment in (
        'group("Both-Open Timeout")',
        'group("Break-Even After Opposite SL")',
        'group("Remaining Leg Timeout After First SL")',
        'group("Break-Even Calculator")',
        "self.both_timeout",
        "self.be_after_sl",
        "self.remaining_leg_timeout",
        "_update_checkpoint_score_controls",
    ):
        assert fragment not in source
