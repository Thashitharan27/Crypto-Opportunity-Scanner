"""Human-focused Data Library presentation for the active native v3 GUI."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DataLibraryWorkspace(QWidget):
    """Show one archive inventory, with exact validation detail on demand."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        inventory = QGroupBox("Historical Data Inventory")
        inventory_layout = QVBoxLayout(inventory)
        inventory_header = QHBoxLayout()
        inventory_note = QLabel(
            "Full local archive coverage. Run-specific data usage and readiness stay on Setup."
        )
        inventory_note.setWordWrap(True)
        inventory_note.setStyleSheet("color:#52606d")
        self.refresh_inventory = QPushButton("Refresh Inventory")
        self.refresh_inventory.clicked.connect(window.refresh_data_library)
        inventory_header.addWidget(inventory_note, 1)
        inventory_header.addWidget(self.refresh_inventory)
        inventory_layout.addLayout(inventory_header)

        window.library_table.setMinimumHeight(430)
        window.library_table.horizontalHeader().setStretchLastSection(True)
        inventory_layout.addWidget(window.library_table)
        outer.addWidget(inventory)

        self.advanced_validation_toggle = QPushButton(
            "Advanced Validation Diagnostics ▸"
        )
        self.advanced_validation_toggle.setCheckable(True)
        self.advanced_validation_toggle.setStyleSheet(
            "text-align:left; font-weight:600; padding:6px"
        )
        outer.addWidget(self.advanced_validation_toggle)

        self.advanced_validation_panel = QGroupBox("Current Request Validation")
        advanced = QVBoxLayout(self.advanced_validation_panel)
        advanced_note = QLabel(
            "Exact validator row counts, statuses and issue codes for the current request. "
            "Setup already translates these diagnostics into the normal Ready / Not Ready view."
        )
        advanced_note.setWordWrap(True)
        advanced_note.setStyleSheet("color:#52606d")
        advanced.addWidget(advanced_note)
        advanced.addWidget(window.quality)
        window.quality_table.setMinimumHeight(220)
        window.quality_table.horizontalHeader().setStretchLastSection(True)
        advanced.addWidget(window.quality_table)
        self.advanced_validation_panel.hide()
        outer.addWidget(self.advanced_validation_panel)

        self.advanced_validation_toggle.toggled.connect(
            self._set_advanced_validation_visible
        )

        # The legacy request-coverage table and resolution label remain live
        # because existing readiness/result plumbing writes to them. They are
        # no longer researcher-facing Data Library content.
        self._compatibility_sinks = QWidget(self)
        self._compatibility_sinks.hide()
        window.coverage.setParent(self._compatibility_sinks)
        window.coverage.hide()
        window.resolution.setParent(self._compatibility_sinks)
        window.resolution.hide()

        outer.addStretch()

    def _set_advanced_validation_visible(self, expanded: bool) -> None:
        self.advanced_validation_panel.setVisible(expanded)
        self.advanced_validation_toggle.setText(
            "Advanced Validation Diagnostics ▾"
            if expanded
            else "Advanced Validation Diagnostics ▸"
        )
