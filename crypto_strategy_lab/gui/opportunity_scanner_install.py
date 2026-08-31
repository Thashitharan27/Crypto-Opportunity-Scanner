"""Compose Opportunity Scanner into the one active QMainWindow."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFrame, QPushButton, QScrollArea

from .opportunity_scanner_controller import UnconfiguredOpportunityScannerService
from .opportunity_scanner_workspace import OpportunityScannerWorkspace


def apply_opportunity_scanner_workspace(window, service=None):
    if getattr(window,"opportunity_scanner_workspace",None) is not None or not hasattr(window,"pages"): return
    workspace=OpportunityScannerWorkspace(service or getattr(window,"opportunity_scanner_service",None) or UnconfiguredOpportunityScannerService(),window)
    scroll=QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(workspace)
    page=window._page("Opportunity Scanner",scroll); index=window.pages.addWidget(page)
    nav=window.centralWidget().layout().itemAt(0).layout(); button=QPushButton("Opportunity Scanner"); button.setFlat(True); button.clicked.connect(lambda _=False: window.pages.setCurrentIndex(index))
    # Place the scanner before the navigation stretch without disturbing pages.
    nav.insertWidget(max(0,nav.count()-3),button); window.opportunity_scanner_workspace=workspace; window.opportunity_scanner_button=button
    application = QApplication.instance()
    if application is not None:
        application.aboutToQuit.connect(workspace.shutdown)
