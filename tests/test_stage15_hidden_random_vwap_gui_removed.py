from pathlib import Path

from PySide6.QtWidgets import QApplication, QGroupBox

from crypto_strategy_lab.gui.config_logic import default_gui_config
from crypto_strategy_lab.gui.main_window import MainWindow


def app():
    return QApplication.instance() or QApplication([])


def test_hidden_random_vwap_widgets_are_not_constructed():
    app()
    window = MainWindow()
    try:
        removed = (
            "vwap_breakout_hours", "vwap_volume_lookback", "vwap_volume_multiplier", "vwap_slope_lookback",
            "vwap_atr_min", "vwap_atr_max", "vwap_confirmation_mode", "vwap_retest_window", "vwap_retest_tolerance",
            "enable_random_entry", "entry_timing_mode", "random_probability", "random_seed", "random_start_mode",
            "randomize_first", "max_random_wait", "enable_random_batch", "random_seed_start", "random_seed_count",
            "enable_coin_flip_sizing", "coin_flip_seed",
        )
        for name in removed:
            assert not hasattr(window, name)
        titles = {box.title() for box in window.findChildren(QGroupBox)}
        assert "VWAP Volume Breakout" not in titles
        assert "Random Entry Timing" not in titles
    finally:
        window.close()


def test_retired_random_and_breakout_keys_are_absent_while_profile_vwap_rule_survives():
    app()
    window = MainWindow()
    try:
        values = default_gui_config()
        profile = values["strategy_profiles"]["bull_short"]
        profile["entry_rules"] = [
            {
                "action": "REJECT",
                "indicator": "VWAP_DISTANCE",
                "condition": "OUTSIDE",
                "minimum": -0.5,
                "maximum": 1.5,
            }
        ]
        window.apply_values(values)
        current = window.values()
        for key in (
            "enable_random_entry", "entry_timing_mode", "random_entry_probability",
            "enable_random_entry_batch", "enable_coin_flip_sizing", "vwap_breakout_lookback_hours",
            "vwap_confirmation_mode",
        ):
            assert key not in current
        assert tuple(current["strategy_profiles"]["bull_short"]["entry_rules"]) == (
            {
                "action": "REJECT",
                "indicator": "VWAP_DISTANCE",
                "condition": "OUTSIDE",
                "minimum": -0.5,
                "maximum": 1.5,
            },
        )
    finally:
        window.close()


def test_main_window_source_has_no_hidden_random_vwap_plumbing():
    source = (Path(__file__).resolve().parents[1] / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    for fragment in (
        'group("VWAP Volume Breakout")',
        'group("Random Entry Timing")',
        "self.vwap_confirmation_mode",
        "self.enable_random_entry",
        "self.enable_coin_flip_sizing",
    ):
        assert fragment not in source
