from PySide6.QtWidgets import QApplication, QGroupBox, QLabel

from crypto_strategy_lab.gui.main_window import MainWindow


def app():
    return QApplication.instance() or QApplication([])


def test_di_direction_pressure_tab_has_direction_pressure_and_mean_reversion_sections():
    app()
    window = MainWindow()
    try:
        names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        page = window.tabs.widget(names.index("DI Direction & Pressure"))
        group_titles = [box.title() for box in page.findChildren(QGroupBox)]

        assert group_titles == ["DI Direction Selection", "DI Pressure Analysis", "Mean Reversion Analysis"]
        assert page.isAncestorOf(window.enable_di_direction_selection)
        assert page.isAncestorOf(window.enable_di_pressure_analysis)
        assert page.isAncestorOf(window.di_pressure_lookback)
        assert page.isAncestorOf(window.enable_mean_reversion_analysis)
        assert page.isAncestorOf(window.mean_reversion_period)
        assert window.di_pressure_lookback.minimum() == 1
        assert window.di_pressure_lookback.maximum() == 100
        assert window.mean_reversion_period.value() == 20
        labels = [label.text() for label in page.findChildren(QLabel)]
        assert any("RECORD ONLY" in text for text in labels)
        assert any("no DI cutoff is hard-coded" in text for text in labels)
    finally:
        window.close()
