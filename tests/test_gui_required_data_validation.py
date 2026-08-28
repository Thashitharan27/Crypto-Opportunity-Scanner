from __future__ import annotations

from datetime import datetime, timezone
import os

import pytest

from crypto_strategy_lab.data import (
    DataQualityIssue,
    DataQualityReport,
    DataQualityStatus,
    DatasetQualityReport,
    DatasetKind,
    MarketKind,
)
from crypto_strategy_lab.gui.v2_controller import GuiApplicationService, GuiResearchRequest


def _dataset_report(*, status=DataQualityStatus.OK, issues=(), complete_start=None, complete_end=None):
    return DatasetQualityReport(
        dataset="klines",
        symbol="BTCUSDT",
        interval="1h",
        required=True,
        requested_start="2020-01-01T00:00:00+00:00",
        requested_end="2020-12-31T00:00:00+00:00",
        observed_start="2020-01-01T02:00:00+00:00",
        observed_end="2020-12-31T00:00:00+00:00",
        complete_start=complete_start or "2020-01-01T00:00:00+00:00",
        complete_end=complete_end or "2020-12-31T00:00:00+00:00",
        row_count=8760,
        source_identity="source",
        status=status,
        issues=tuple(issues),
    )


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.rule_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def inventory(self, *_args):
            return []

        def coverage(self, request):
            return [
                {
                    "dataset": "klines",
                    "interval": request.strategy_timeframe,
                    "first_period": datetime(2019, 12, 1, tzinfo=timezone.utc),
                    "last_period": datetime(2021, 1, 1, tzinfo=timezone.utc),
                    "archive_count": 13,
                    "state": "AVAILABLE",
                }
            ]

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    return app, MainWindow(service=Service())


def _strategy_bars_only(window):
    window.intrabar_tf.setCurrentText(None)
    window.strategy_tf.setCurrentText("1h")


def test_required_data_quality_validates_exact_strategy_and_intrabar_intervals_once_each():
    calls = []
    report = _dataset_report()

    class Store:
        def data_quality_report(self, request, dataset, *, interval, required):
            calls.append((dataset, interval, required, request.start, request.end))
            return report

    service = object.__new__(GuiApplicationService)
    service.store = Store()
    service.progress_callback = None
    request = GuiResearchRequest(
        "binance",
        MarketKind.FUTURES_UM,
        "BTCUSDT",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 12, 31, tzinfo=timezone.utc),
        "1h",
        "1m",
    )

    result = service.required_data_quality(request)

    assert isinstance(result, DataQualityReport)
    assert [item[1] for item in calls] == ["1h", "1m"]
    assert all(item[0] is DatasetKind.KLINES and item[2] is True for item in calls)


def test_setup_distinguishes_archive_catalog_availability_from_validated_continuity():
    _app, window = _window()
    try:
        _strategy_bars_only(window)
        window.refresh_coverage()
        text = window.datasets.text()
        assert "Strategy candles (1h): CATALOG AVAILABLE" in text
        assert "archive coverage" in text
        assert "Validate Selected Range checks actual candle continuity" in text
        assert "VALIDATION PENDING" in window._data_state()
    finally:
        window.close()


def test_leading_gap_is_shown_with_exact_boundary_and_safe_date_adjustment():
    _app, window = _window()
    try:
        from PySide6.QtCore import QDate

        _strategy_bars_only(window)
        window.start.setDate(QDate(2020, 1, 1))
        window.end.setDate(QDate(2020, 12, 31))
        window.refresh_coverage()
        issues = (
            DataQualityIssue(
                "LEADING_COVERAGE_GAP",
                DataQualityStatus.ERROR,
                "Leading fixed-cadence coverage is missing",
                count=2,
                first_timestamp="2020-01-01T00:00:00+00:00",
                last_timestamp="2020-01-01T01:00:00+00:00",
            ),
            DataQualityIssue(
                "LEADING_SOURCE_COVERAGE_GAP",
                DataQualityStatus.ERROR,
                "Catalog source coverage begins after the requested start",
                details={"coverage_start": "2020-01-01T02:00:00+00:00"},
            ),
        )
        report = DataQualityReport((
            _dataset_report(
                status=DataQualityStatus.ERROR,
                issues=issues,
                complete_start="2020-01-01T02:00:00+00:00",
            ),
        ))
        window._validated_request_key = window._request_key(window.request_model())
        window._validated_report = report
        window._render_range_validation(report)

        text = window.range_validation.text()
        assert "Strategy candles (1h): BLOCKED" in text
        assert "Missing at start: 2" in text
        assert "2020-01-01 00:00 UTC" in text
        assert "Earliest validated boundary: 2020-01-01 02:00 UTC" in text
        assert window.use_validated_range_button.isEnabled()
        assert window._validated_safe_dates[0].isoformat() == "2020-01-02"
        assert window._data_state() == "BLOCKED"
    finally:
        window.close()


def test_internal_gap_does_not_offer_misleading_boundary_date_fix():
    _app, window = _window()
    try:
        _strategy_bars_only(window)
        issue = DataQualityIssue(
            "MISSING_INTERNAL_INTERVAL",
            DataQualityStatus.ERROR,
            "Internal fixed-cadence intervals are missing",
            count=1,
            first_timestamp="2020-06-01T12:00:00+00:00",
            last_timestamp="2020-06-01T12:00:00+00:00",
        )
        report = DataQualityReport((
            _dataset_report(status=DataQualityStatus.ERROR, issues=(issue,)),
        ))
        window._render_range_validation(report)
        assert "Internal candle gap: 1" in window.range_validation.text()
        assert not window.use_validated_range_button.isEnabled()
    finally:
        window.close()
