from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.data_library_install import apply_data_library_workspace
    from crypto_strategy_lab.gui.setup_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def inventory(self, *_args):
            return [
                {
                    "symbol": "BTCUSDT",
                    "dataset": "klines",
                    "interval": "1h",
                    "first_period": "2020-01-01",
                    "last_period": "2026-08-23",
                    "archive_count": 101,
                    "state": "AVAILABLE",
                },
                {
                    "symbol": "BTCUSDT",
                    "dataset": "funding_rate",
                    "interval": None,
                    "first_period": "2020-01-01",
                    "last_period": "2026-08-21",
                    "archive_count": 79,
                    "state": "AVAILABLE",
                },
            ]

        def coverage(self, request):
            base = dict(
                first_period=datetime(2020, 1, 1, tzinfo=timezone.utc),
                last_period=datetime(2026, 8, 21, tzinfo=timezone.utc),
                archive_count=99,
                state="AVAILABLE",
            )
            return [
                {"dataset": "klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "funding_rate", "interval": None, **base},
            ]

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    window = MainWindow(service=Service())
    window._validation_debounce.stop()
    apply_data_library_workspace(window)
    return app, window


def test_active_app_composes_compact_data_library():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "apply_data_library_workspace" in source
    assert source.index("apply_results_dashboard_workspace(window)") < source.index(
        "apply_data_library_workspace(window)"
    )


def test_data_library_has_one_visible_inventory_and_hidden_coverage_sink():
    _app, window = _window()
    try:
        workspace = window.data_library_workspace
        page = window.pages.widget(7)
        assert page.findChild(type(workspace)) is workspace
        groups = [group.title() for group in page.findChildren(type(workspace.advanced_validation_panel))]
        assert "Historical Data Inventory" in groups
        assert "Technical Coverage & Validation Detail" not in groups
        assert window.library_table.rowCount() == 2
        assert window.coverage.isHidden()
        assert window.resolution.isHidden()
        assert workspace.advanced_validation_panel.isHidden()
    finally:
        window.close()


def test_advanced_validation_reveals_only_unique_quality_diagnostics():
    _app, window = _window()
    try:
        workspace = window.data_library_workspace
        assert window.quality_table.parent() is workspace.advanced_validation_panel
        workspace.advanced_validation_toggle.setChecked(True)
        assert not workspace.advanced_validation_panel.isHidden()
        assert workspace.advanced_validation_toggle.text().endswith("▾")
        workspace.advanced_validation_toggle.setChecked(False)
        assert workspace.advanced_validation_panel.isHidden()
        assert workspace.advanced_validation_toggle.text().endswith("▸")
    finally:
        window.close()
