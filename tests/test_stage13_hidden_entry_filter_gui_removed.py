from pathlib import Path

from PySide6.QtWidgets import QApplication, QGroupBox

from crypto_strategy_lab.gui.main_window import MainWindow


def app():
    return QApplication.instance() or QApplication([])


def test_hidden_global_entry_filter_widgets_are_not_constructed():
    app()
    window = MainWindow()
    try:
        removed = (
            "enable_adx", "adx_mode", "adx_max", "adx_min",
            "enable_bb_width", "bb_width_mode", "bb_width_min", "bb_width_max",
            "skip_monday_entries", "skip_monday_timezone",
            "enable_di_spread", "di_spread_mode", "di_spread_min", "di_spread_max",
        )
        for name in removed:
            assert not hasattr(window, name)
        titles = {box.title() for box in window.findChildren(QGroupBox)}
        assert "Trend Filter" not in titles
        assert "Market Compression Filters" not in titles
    finally:
        window.close()


def test_retired_global_entry_filter_keys_do_not_exist_in_gui_values():
    app()
    window = MainWindow()
    try:
        values = window.values()
        for key in (
            "enable_adx_filter", "adx_filter_mode", "adx_minimum", "adx_maximum",
            "enable_bb_width_filter", "bb_width_filter_mode", "bb_width_minimum", "bb_width_maximum",
            "enable_skip_monday_entries", "skip_monday_timezone",
            "enable_di_spread_filter", "di_spread_filter_mode", "di_spread_minimum", "di_spread_maximum",
        ):
            assert key not in values
        assert "adx_period" in values
    finally:
        window.close()


def test_main_window_source_has_no_hidden_entry_filter_group_plumbing():
    source = (Path(__file__).resolve().parents[1] / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    for fragment in (
        'group("Trend Filter")',
        'group("Market Compression Filters")',
        "self.enable_adx=",
        "self.enable_bb_width=",
        "self.skip_monday_entries=",
        "self.enable_di_spread=",
    ):
        assert fragment not in source
