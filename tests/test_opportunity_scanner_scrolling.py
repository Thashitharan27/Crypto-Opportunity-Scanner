"""Offscreen regression coverage for the Opportunity Scanner page boundary."""
import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_installer_scrolls_the_workspace_without_changing_nested_tables():
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6.QtCore import Qt

    from crypto_strategy_lab.gui.opportunity_scanner_install import (
        apply_opportunity_scanner_workspace,
    )
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import (
        OpportunityScannerWorkspace,
    )

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Window(widgets.QWidget):
        def __init__(self):
            super().__init__()
            root = widgets.QHBoxLayout(self)
            navigation = widgets.QVBoxLayout()
            navigation.addStretch()
            navigation.addWidget(widgets.QLabel("status"))
            navigation.addWidget(widgets.QPushButton("run"))
            root.addLayout(navigation)
            self.pages = widgets.QStackedWidget()
            root.addWidget(self.pages, 1)

        def centralWidget(self):
            return self

        @staticmethod
        def _page(title, widget):
            page = widgets.QWidget()
            layout = widgets.QVBoxLayout(page)
            layout.addWidget(widgets.QLabel(title))
            layout.addWidget(widget)
            return page

    window = Window()
    apply_opportunity_scanner_workspace(window, service=object())
    workspace = window.opportunity_scanner_workspace
    scroll = window.pages.currentWidget().findChild(widgets.QScrollArea)

    try:
        assert isinstance(workspace, OpportunityScannerWorkspace)
        assert scroll is not None and scroll.widget() is workspace
        assert scroll.widgetResizable()
        assert scroll.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
        assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff

        table_names = (
            "preliminary_table", "final_table", "readiness_table",
            "validation_outcomes", "validation_summary", "validation_rank",
        )
        for name in table_names:
            table = getattr(workspace, name)
            assert isinstance(table, widgets.QAbstractScrollArea)
            assert table.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
            assert table.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded

        # The workspace follows available width, while its natural content height
        # makes a deliberately short page scroll vertically.
        window.resize(900, 420)
        window.show()
        app.processEvents()
        vertical = scroll.verticalScrollBar()
        assert vertical.maximum() > 0
        assert scroll.horizontalScrollBar().maximum() == 0

        workspace.tabs.setCurrentIndex(workspace.tabs.count() - 1)
        vertical.setValue(vertical.maximum())
        app.processEvents()
        validation = workspace.tabs.currentWidget()
        validation_bottom = validation.mapTo(scroll.viewport(), validation.rect().bottomLeft()).y()
        assert 0 < validation_bottom <= scroll.viewport().height()

        # A viewport tall enough for the layout's natural size needs no page scroll.
        window.resize(1200, workspace.sizeHint().height() + 200)
        app.processEvents()
        assert vertical.maximum() == 0

        window.resize(640, window.height())
        app.processEvents()
        assert scroll.horizontalScrollBar().maximum() == 0
    finally:
        workspace.shutdown()
        window.close()
