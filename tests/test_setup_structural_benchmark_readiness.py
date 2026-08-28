from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

import pandas as pd
import pytest

from crypto_strategy_lab.data import (
    DataQualityIssue,
    DataQualityReport,
    DataQualityStatus,
    DatasetQualityReport,
    MarketKind,
)
from crypto_strategy_lab.data_lake_config import FeatureConfig
from crypto_strategy_lab.gui.run_readiness import (
    MainWindow,
    required_run_data_quality,
    structural_benchmark_requirement,
)
from crypto_strategy_lab.gui.v2_controller import GuiResearchRequest


def _request():
    return GuiResearchRequest(
        "binance",
        MarketKind.FUTURES_UM,
        "BTCUSDT",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 12, 1, tzinfo=timezone.utc),
        "1h",
        "1m",
    )


def _report(interval, *, start="2020-01-01T00:00:00+00:00", status=DataQualityStatus.OK,
            issues=(), complete_start=None, symbol="BTCUSDT"):
    return DatasetQualityReport(
        dataset="klines",
        symbol=symbol,
        interval=interval,
        required=True,
        requested_start=start,
        requested_end="2020-12-01T00:00:00+00:00",
        observed_start=complete_start or start,
        observed_end="2020-12-01T00:00:00+00:00",
        complete_start=complete_start or start,
        complete_end="2020-12-01T00:00:00+00:00",
        row_count=100,
        source_identity="source",
        status=status,
        issues=tuple(issues),
    )


def test_structural_requirement_matches_runtime_warmup_contract():
    request = _request()
    requirement = structural_benchmark_requirement(request, FeatureConfig())

    assert requirement is not None
    assert requirement.symbol == "BTCUSDT"
    assert requirement.interval == "1h"
    assert requirement.warmup_days == 237
    assert requirement.data_request.start == request.period_start - timedelta(days=237)
    assert requirement.data_request.end == request.period_end


def test_required_run_quality_includes_structural_benchmark_prestart_history():
    request = _request()
    features = FeatureConfig()
    requirement = structural_benchmark_requirement(request, features)
    base = DataQualityReport((_report("1h"), _report("1m")))
    calls = []

    class Store:
        def data_quality_report(self, data_request, dataset, *, interval, required):
            calls.append((data_request, dataset, interval, required))
            issue = DataQualityIssue(
                "LEADING_SOURCE_COVERAGE_GAP",
                DataQualityStatus.ERROR,
                "Source begins after requested start",
                details={"coverage_start": "2019-09-01T00:00:00+00:00"},
            )
            return _report(
                "1h",
                start=requirement.data_request.start.isoformat(),
                status=DataQualityStatus.ERROR,
                issues=(issue,),
                complete_start="2019-09-01T00:00:00+00:00",
            )

    class Service:
        store = Store()

        @staticmethod
        def required_data_quality(_request):
            return base

    result = required_run_data_quality(Service(), request, features)

    assert len(result.datasets) == 3
    assert result.overall_status is DataQualityStatus.ERROR
    assert len(calls) == 1
    assert calls[0][0].start == request.period_start - timedelta(days=237)
    assert calls[0][2] == "1h"
    assert calls[0][3] is True


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
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
                {"dataset": "klines", "interval": "5m", **base},
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


def test_setup_cannot_be_ready_when_structural_warmup_is_missing():
    _app, window = _window()
    try:
        from PySide6.QtCore import QDate

        window.start.setDate(QDate(2020, 1, 1))
        window.end.setDate(QDate(2020, 12, 1))
        window.strategy_tf.setCurrentText("1 Hour")
        window.intrabar_tf.setCurrentText("1 Minute")
        window._validation_debounce.stop()
        request = window.request_model()
        requirement = structural_benchmark_requirement(request, window._current_features())
        issue = DataQualityIssue(
            "LEADING_SOURCE_COVERAGE_GAP",
            DataQualityStatus.ERROR,
            "Source begins after requested start",
            details={"coverage_start": "2019-09-01T00:00:00+00:00"},
        )
        report = DataQualityReport((
            _report("1h"),
            _report("1m"),
            _report(
                "1h",
                start=requirement.data_request.start.isoformat(),
                status=DataQualityStatus.ERROR,
                issues=(issue,),
                complete_start="2019-09-01T00:00:00+00:00",
            ),
        ))
        window._validated_request_key = window._request_key(request)
        window._validated_report = report
        window._render_range_validation(report)
        window._refresh_run_data_view()

        assert window.readiness_state.text() == "NOT READY"
        assert "Market regime benchmark BTCUSDT 1h" in window.range_validation.text()
        assert "237-day warm-up" in window.range_validation.text()
        assert "structural market-regime benchmark is also a required strategy input" in window.range_validation.text()
        assert window.use_validated_range_button.isEnabled()
        expected = (
            pd.Timestamp("2019-09-01T00:00:00Z") + pd.Timedelta(days=237)
        ).date()
        assert window._validated_safe_dates[0] == expected

        rows = [
            [
                window.run_data_table.item(row, column).text()
                for column in range(window.run_data_table.columnCount())
            ]
            for row in range(window.run_data_table.rowCount())
        ]
        assert any(
            row[0] == "Market Regime"
            and row[1] == "BTCUSDT Structural Benchmark"
            and row[4] == "✗ Blocked"
            for row in rows
        )
    finally:
        window.close()


def test_setup_shows_catalog_coverage_when_required_interval_has_no_overlap():
    _app, window = _window()
    try:
        from PySide6.QtCore import QDate

        window.start.setDate(QDate(2019, 1, 1))
        window.end.setDate(QDate(2019, 12, 1))
        window.strategy_tf.setCurrentText("1 Hour")
        window.intrabar_tf.setCurrentText("1 Minute")
        window._validation_debounce.stop()

        missing_issue = DataQualityIssue(
            "DATASET_MISSING",
            DataQualityStatus.ERROR,
            "No canonical rows are available for the requested range",
            count=0,
        )
        missing_intrabar = DatasetQualityReport(
            dataset="klines",
            symbol="BTCUSDT",
            interval="1m",
            required=True,
            requested_start="2019-01-01T00:00:00+00:00",
            requested_end="2019-12-01T00:00:00+00:00",
            observed_start=None,
            observed_end=None,
            complete_start=None,
            complete_end=None,
            row_count=0,
            source_identity=None,
            status=DataQualityStatus.ERROR,
            issues=(missing_issue,),
        )
        report = DataQualityReport((
            _report("1h", start="2019-01-01T00:00:00+00:00"),
            missing_intrabar,
        ))

        window._render_range_validation(report)
        text = window.range_validation.text()

        assert window.readiness_state.text() == "NOT READY"
        assert (
            "No 1m candles overlap the requested range "
            "2019-01-01 00:00 UTC → 2019-12-01 00:00 UTC."
        ) in text
        assert (
            "Available catalog coverage: "
            "2020-01-01 00:00 UTC → 2026-08-21 00:00 UTC."
        ) in text
        assert window.use_strategy_bars_button.isEnabled()
    finally:
        window.close()
