"""Small application-owned style layer for stable desktop shell presentation.

The research controls keep the platform Qt theme.  Only the application shell
(font plus left navigation chrome) is explicit so minor Qt/native-style changes
do not materially alter the workstation layout.
"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget


APPLICATION_FONT_FAMILY = "Segoe UI"
APPLICATION_FONT_POINT_SIZE = 9

_NAVIGATION_LABELS = {
    "Setup",
    "Strategy Builder",
    "Research Features",
    "Risk Execution",
    "Reports",
    "Review Run",
    "Results Dashboard",
    "Data Library",
    "ChatGPT / MCP",
    "GitHub",
}

NAVIGATION_BUTTON_STYLE = """
QPushButton {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #edf1f5;
    border-radius: 4px;
    padding: 2px 10px;
    min-height: 19px;
}
QPushButton:hover {
    background-color: #f7f9fb;
    border-color: #d9e2ec;
}
QPushButton:pressed {
    background-color: #eef2f6;
    border-color: #c8d4df;
}
QPushButton:focus {
    border-color: #b7c9dc;
}
""".strip()

CURRENT_RESEARCH_STYLE = (
    "background:#f4f7fa; padding:12px; border:1px solid #d9e2ec; "
    "border-radius:4px"
)


def _normalized_navigation_text(text: str) -> str:
    return " ".join(text.replace("&", "").split())


def apply_application_style(app: QApplication) -> None:
    """Apply stable typography before constructing the main window."""

    font = QFont(APPLICATION_FONT_FAMILY)
    font.setPointSize(APPLICATION_FONT_POINT_SIZE)
    app.setFont(font)


def apply_shell_style(window: QWidget) -> None:
    """Make the left application chrome independent of the native Qt style."""

    for button in window.findChildren(QPushButton):
        if _normalized_navigation_text(button.text()) not in _NAVIGATION_LABELS:
            continue
        # The old shell used native flat buttons.  Qt 6.10 and 6.11 render
        # those differently on Windows, so own their visible surface here.
        button.setFlat(False)
        button.setMinimumWidth(240)
        button.setStyleSheet(NAVIGATION_BUTTON_STYLE)
        button.setProperty("cslNavigationButton", True)

    current_research = getattr(window, "current_research", None)
    if isinstance(current_research, QLabel):
        current_research.setMinimumWidth(245)
        current_research.setStyleSheet(CURRENT_RESEARCH_STYLE)

    window.setProperty("cslDesktopStyleApplied", True)
