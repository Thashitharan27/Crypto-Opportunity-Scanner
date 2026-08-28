from __future__ import annotations

from datetime import datetime, timezone
import os

import pytest

from crypto_strategy_lab.data import (
    DataQualityIssue,
    DataQualityReport,
    DataQualityStatus,
    DatasetQualityReport,
)


def _dataset_report(interval, *, status=DataQualityStatus.OK, issues=()):
    return DatasetQualityReport(
        dataset="klines",
        symbol="BTCUSDT",
        interval=interval,
        required=True,
        requested_start="2020-01-01T00:00:00+00:00",
        requested_end="2020-12-01T00:00:00+00:00",
        observed_start="2020-01-01T00:00:00+00:00",
        observed_end="2020-12-01T00:00:00+00:00",
        complete_start="2020-01-01T00:00:00+00:00",
        complete_end="2020-12-01T00:00:00+00:00",
        row_count=100,
        source_identity="source",
        status=status,
        issues=tuple(issues),
    )


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.setup_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def inventory(self, *_args):
            return []

        def coverage(self, request):
            base = dict(
                first_period=datetime(2020, 1, 1, tzinfo=timezone.utc),
                last_period=datetime(2026, 8, 21, tzinfo=timezone.utc),
                archive_count=99,
                state="AVAILABLE",
            )
            rows = [
                {"dataset": "klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "funding_rate", "interval": None, **base},
                {"dataset": "metrics", "interval": None, **base},
                {"dataset": "mark_price_klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "index_price_klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "premium_index_klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "klines", "interval": "5m", **base},
                {"dataset": "agg_trades", "interval": None, **base},
                {"dataset": "book_depth", "interval": None, **base},
            ]
            if request.intrabar_timeframe:
                rows.append({"dataset": "klines", "interval": request.intrabar_timeframe, **base})
            return rows

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    window = MainWindow(service=Service())
    window._validation_debounce.stop()
    return app, window


def _table_rows(window):
    return [
        [
            window.run_data_table.item(row, column).text()
            for column in range(window.run_data_table.columnCount())
        ]
        for row in range(window.run_data_table.rowCount())
    ]


def test_setup_is_request_data_used_and_readiness_not_technical_tables():
    _app, window = _window()
    try:
        groups = [group.title() for group in window.pages.widget(0).findChildren(type(window.readiness_state.parent()))]
        # Use direct widget presence as the stable acceptance contract.
        assert window.run_data_table is not None
        assert window.readiness_state.text()
        assert window.coverage.parent() is not window.pages.widget(0)
        assert window.quality_table.parent() is not window.pages.widget(0)
        assert "Technical Coverage & Validation Detail" not in groups
    finally:
        window.close()


def test_data_used_by_run_separates_required_and_optional_sources():
    _app, window = _window()
    try:
        window.strategy_tf.setCurrentText("1 Hour")
        window.intrabar_tf.setCurrentText("1 Minute")
        window.refresh_coverage()
        window._validation_debounce.stop()
        rows = _table_rows(window)
        assert ["Strategy", "Market Price Candles", "1h", "Required", "… Needs validation"] in rows
        assert ["Intrabar / Exits", "Market Price Candles", "1m", "Required", "… Needs validation"] in rows
        assert any(row[:2] == ["Research", "Funding"] and row[3] == "Auto when covered" for row in rows)
        assert any(row[:2] == ["Research", "Basis / Premium Context"] for row in rows)
        assert any(row[:2] == ["Research", "Trade Flow"] and row[4] == "— Off" for row in rows)
        assert any(row[:2] == ["Research", "Order Book"] and row[4] == "— Off" for row in rows)
    finally:
        window.close()


def test_intrabar_internal_gap_is_plain_english_and_offers_strategy_bars_only():
    _app, window = _window()
    try:
        window.strategy_tf.setCurrentText("1 Hour")
        window.intrabar_tf.setCurrentText("1 Minute")
        issue = DataQualityIssue(
            "MISSING_INTERNAL_INTERVAL",
            DataQualityStatus.ERROR,
            "Internal fixed-cadence intervals are missing",
            count=44640,
            first_timestamp="2020-10-01T00:00:00+00:00",
            last_timestamp="2020-10-31T23:59:00+00:00",
        )
        report = DataQualityReport((
            _dataset_report("1h"),
            _dataset_report("1m", status=DataQualityStatus.ERROR, issues=(issue,)),
        ))
        window._validated_request_key = window._request_key(window.request_model())
        window._validated_report = report
        window._render_range_validation(report)
        window._refresh_run_data_view()

        assert window.readiness_state.text() == "NOT READY"
        text = window.range_validation.text()
        assert "Strategy 1h candles are complete" in text
        assert "Intrabar 1m candles are incomplete" in text
        assert "Internal gap: 44,640" in text
        assert "Only the selected intrabar / exit-detail data is blocking this run" in text
        assert window.use_strategy_bars_button.isEnabled()
        assert not window.use_validated_range_button.isEnabled()
        assert any(row[0] == "Intrabar / Exits" and row[4] == "✗ Blocked" for row in _table_rows(window))
    finally:
        window.close()


def test_use_strategy_bars_only_clears_intrabar_requirement():
    _app, window = _window()
    try:
        window.intrabar_tf.setCurrentText("1 Minute")
        assert window.intrabar_tf.currentData() == "1m"
        window._use_strategy_bars_only()
        assert window.intrabar_tf.currentData() is None
    finally:
        window.close()


def test_active_app_launches_setup_readiness_window():
    from app import MainWindow as AppMainWindow
    from crypto_strategy_lab.gui.setup_main_window import MainWindow as SetupMainWindow

    assert issubclass(AppMainWindow, SetupMainWindow)
