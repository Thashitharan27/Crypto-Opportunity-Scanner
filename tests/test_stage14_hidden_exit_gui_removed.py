from pathlib import Path

from PySide6.QtWidgets import QApplication, QGroupBox

from crypto_strategy_lab.gui.config_logic import default_gui_config
from crypto_strategy_lab.gui.main_window import MainWindow


def app():
    return QApplication.instance() or QApplication([])


def test_hidden_global_exit_widgets_are_not_constructed():
    app()
    window = MainWindow()
    try:
        removed = (
            "enable_partial_tp", "tp1_r", "tp1_close_pct", "tp2_r", "tp2_close_pct",
            "stop_loss_r", "after_tp1_stop_mode", "after_tp1_stop_offset_r", "tp2_exit_mode",
            "enable_partial_sl", "sl1_r", "sl1_close_pct", "sl2_r",
            "enable_trailing_profit", "trail_activation_trigger", "trail_activation_r",
            "trail_distance_r", "trail_apply_to", "trail_intrabar_mode",
        )
        for name in removed:
            assert not hasattr(window, name)
        titles = {box.title() for box in window.findChildren(QGroupBox)}
        assert "Partial Stop Loss" not in titles
        assert "Partial Take Profit" not in titles
        assert "Post-TP1 Protective Stop" not in titles
        assert "Independent Trailing Stop" not in titles
    finally:
        window.close()


def test_retired_global_exit_keys_are_absent_while_profiles_own_exit_management():
    app()
    window = MainWindow()
    try:
        values = default_gui_config()
        profile = values["strategy_profiles"]["bull_long"]
        profile.update({
            "partial_stop_enabled": True,
            "sl1_r": 0.75,
            "sl2_r": 2.5,
            "partial_profit_enabled": True,
            "tp1_r": 1.25,
            "tp2_r": 3.0,
            "trailing_enabled": True,
            "trailing_activation_r": 2.5,
            "trailing_distance_r": 0.75,
        })
        window.apply_values(values)
        current = window.values()
        for key in (
            "enable_partial_stop_loss", "enable_partial_take_profit", "enable_trailing_profit",
            "stop_loss_r", "trail_activation_trigger", "trail_apply_to", "trail_intrabar_mode",
        ):
            assert key not in current
        bull_long = current["strategy_profiles"]["bull_long"]
        assert bull_long["partial_stop_enabled"] is True
        assert bull_long["sl1_r"] == 0.75
        assert bull_long["partial_profit_enabled"] is True
        assert bull_long["tp2_r"] == 3.0
        assert bull_long["trailing_enabled"] is True
        assert bull_long["trailing_distance_r"] == 0.75
    finally:
        window.close()


def test_main_window_source_has_no_hidden_global_exit_plumbing():
    source = (Path(__file__).resolve().parents[1] / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    for fragment in (
        'group("Partial Stop Loss")',
        'group("Partial Take Profit")',
        'group("Post-TP1 Protective Stop")',
        'group("Independent Trailing Stop")',
        "self.enable_partial_sl",
        "self.enable_partial_tp",
        "self.enable_trailing_profit",
    ):
        assert fragment not in source
