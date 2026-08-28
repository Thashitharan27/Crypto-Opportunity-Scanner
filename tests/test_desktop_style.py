from pathlib import Path

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from crypto_strategy_lab.gui.desktop_style import (
    APPLICATION_FONT_FAMILY,
    APPLICATION_FONT_POINT_SIZE,
    apply_application_style,
    apply_shell_style,
)


ROOT = Path(__file__).resolve().parents[1]


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_application_style_owns_shell_typography():
    app = _app()
    apply_application_style(app)

    assert app.font().family() == APPLICATION_FONT_FAMILY
    assert app.font().pointSize() == APPLICATION_FONT_POINT_SIZE


def test_shell_style_owns_navigation_surface_without_restyling_page_actions():
    _app()
    window = QWidget()
    layout = QVBoxLayout(window)
    setup = QPushButton("Setup", window)
    risk = QPushButton("Risk & Execution", window)
    ordinary_action = QPushButton("Open Workbook", window)
    window.current_research = QLabel(window)
    for widget in (setup, risk, ordinary_action, window.current_research):
        layout.addWidget(widget)

    setup.setFlat(True)
    risk.setFlat(True)
    apply_shell_style(window)

    for button in (setup, risk):
        assert button.isFlat() is False
        assert button.minimumWidth() >= 240
        assert button.property("cslNavigationButton") is True
        assert "background-color: #ffffff" in button.styleSheet()
        assert "border-radius: 4px" in button.styleSheet()

    assert ordinary_action.property("cslNavigationButton") is None
    assert ordinary_action.styleSheet() == ""
    assert "border-radius:4px" in window.current_research.styleSheet()
    assert window.property("cslDesktopStyleApplied") is True


def test_active_app_applies_desktop_style_before_window_is_shown():
    source = (ROOT / "app.py").read_text()

    assert "apply_application_style(app)" in source
    assert "apply_shell_style(window)" in source
    assert source.index("apply_shell_style(window)") < source.index("window.show()")
