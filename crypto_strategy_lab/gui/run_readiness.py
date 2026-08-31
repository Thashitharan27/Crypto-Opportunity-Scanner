"""Run-faithful Setup readiness for required structural benchmark history.

The base Setup workspace validates the researcher-selected strategy and intrabar
candle slices. Native structural market-regime preparation also requires a 1h
benchmark with pre-start warm-up history. This layer makes that runtime
requirement visible and validates it before Setup can report READY TO RUN.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox, QTableWidgetItem

from crypto_strategy_lab.data import DataQualityReport, DataQualityStatus, DatasetKind
from crypto_strategy_lab.gui.setup_main_window import MainWindow as SetupMainWindow


from crypto_strategy_lab.structural_requirements import (
    STRUCTURAL_BENCHMARK_INTERVAL, STRUCTURAL_REGIME_METHODS,
    StructuralBenchmarkRequirement, structural_benchmark_requirement,
)


def required_run_data_quality(service, request, features) -> DataQualityReport:
    """Validate selected execution candles plus required structural warm-up."""
    base = service.required_data_quality(request)
    requirement = structural_benchmark_requirement(request, features)
    if requirement is None:
        return base
    store = getattr(service, "store", None)
    if store is None or not hasattr(store, "data_quality_report"):
        raise RuntimeError(
            "Structural benchmark readiness requires the application market-data store."
        )
    benchmark = store.data_quality_report(
        requirement.data_request,
        DatasetKind.KLINES,
        interval=requirement.interval,
        required=True,
    )
    return DataQualityReport((*base.datasets, benchmark))


class RequiredRunDataValidationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service, request, features):
        super().__init__()
        self.service = service
        self.request = request
        self.features = features

    @Slot()
    def run(self):
        try:
            self.finished.emit(
                required_run_data_quality(self.service, self.request, self.features)
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(SetupMainWindow):
    """Setup workspace whose READY state matches structural runtime requirements."""

    def __init__(self, startup_status=None, service=None):
        self._validation_requirements_key = None
        super().__init__(startup_status=startup_status, service=service)
        self.feature_form.changed.connect(self._run_requirements_changed)
        self._refresh_run_data_view()

    def _current_features(self):
        config = getattr(self, "config", None)
        base = getattr(config, "features", None)
        form = getattr(self, "feature_form", None)
        if form is None or base is None:
            return base
        try:
            return form.value(base)
        except Exception:
            return base

    def _requirements_key(self, request, features) -> tuple:
        return (
            *self._request_key(request),
            str(getattr(features, "market_regime_method", "")).upper(),
            int(getattr(features, "structural_regime_sma_days", 0) or 0),
            int(getattr(features, "structural_regime_slope_lookback_days", 0) or 0),
        )

    def _run_requirements_changed(self) -> None:
        if not hasattr(self, "_validation_debounce"):
            return
        self._invalidate_range_validation()
        self._schedule_auto_validation()

    @staticmethod
    def _timestamps_equal(left, right) -> bool:
        try:
            return pd.Timestamp(left) == pd.Timestamp(right)
        except Exception:
            return False

    def _current_benchmark_requirement(self):
        try:
            return structural_benchmark_requirement(
                self.request_model(), self._current_features()
            )
        except Exception:
            return None

    def _is_benchmark_report(self, report, requirement=None) -> bool:
        requirement = requirement or self._current_benchmark_requirement()
        if requirement is None:
            return False
        return (
            str(report.dataset).lower() == DatasetKind.KLINES.value
            and str(report.symbol).upper() == requirement.symbol.upper()
            and report.interval == requirement.interval
            and self._timestamps_equal(
                report.requested_start, requirement.data_request.start
            )
        )

    def _benchmark_report(self, requirement=None):
        if self._validated_report is None:
            return None
        try:
            if self._validated_request_key != self._request_key(self.request_model()):
                return None
        except Exception:
            return None
        requirement = requirement or self._current_benchmark_requirement()
        if requirement is None:
            return None
        for report in self._validated_report.datasets:
            if self._is_benchmark_report(report, requirement):
                return report
        return None

    def _refresh_run_data_view(self):
        super()._refresh_run_data_view()
        if not hasattr(self, "run_data_table"):
            return
        requirement = self._current_benchmark_requirement()
        if requirement is None:
            return

        report = self._benchmark_report(requirement)
        if report is None:
            status = "… Needs validation"
        elif report.status is DataQualityStatus.ERROR:
            status = "✗ Blocked"
        else:
            status = "✓ Ready"

        row = min(2, self.run_data_table.rowCount())
        self.run_data_table.insertRow(row)
        values = (
            "Market Regime",
            f"{requirement.symbol} Structural Benchmark",
            requirement.interval,
            f"Required · {requirement.warmup_days}d warm-up",
            status,
        )
        for column, value in enumerate(values):
            self.run_data_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.run_data_table.resizeColumnsToContents()
        self.run_data_table.horizontalHeader().setStretchLastSection(True)

    def validate_selected_range(self):
        """Rescan raw archive metadata before an explicit researcher revalidation."""
        if self._validation_thread is not None:
            return

        refresh_catalog = getattr(self.service, "refresh_catalog", None)
        try:
            if callable(refresh_catalog):
                self._set_readiness(
                    "CHECKING DATA…",
                    "Refreshing the market-data catalog before revalidating repaired/downloaded files.",
                )
                refresh_catalog()
                self._invalidate_range_validation()
                self._load_catalog()
                if hasattr(self, "_validation_debounce"):
                    self._validation_debounce.stop()
        except Exception as exc:
            self._set_readiness(
                "VALIDATION FAILED",
                f"Could not refresh the market-data catalog before revalidation: {exc}",
                state="blocked",
            )
            return

        self._begin_required_data_validation(auto_run=False)

    def _begin_required_data_validation(self, *, auto_run: bool) -> None:
        if self._validation_thread is not None:
            if auto_run:
                self._validation_auto_run = True
            return
        try:
            request = self.request_model()
            features = self._current_features()
            if request.period_start >= request.period_end:
                raise ValueError("Start Date must be before End Date")
        except Exception as exc:
            if auto_run:
                QMessageBox.warning(self, "Invalid research request", str(exc))
            self._set_readiness("NOT READY", str(exc), state="blocked")
            return

        catalog_state, detail = self.data_readiness(
            self._coverage_rows_from_table(),
            request.strategy_timeframe,
            request.intrabar_timeframe,
        )
        if catalog_state == "BLOCKED":
            self._set_readiness("NOT READY", detail, state="blocked")
            self._refresh_run_data_view()
            if auto_run:
                QMessageBox.warning(self, "Required candle data unavailable", detail)
            return

        if not hasattr(self.service, "required_data_quality"):
            self._set_readiness(
                "VALIDATION UNAVAILABLE",
                "The current application service cannot perform exact candle-range validation.",
            )
            if auto_run:
                super().start_run()
            return

        self._validation_request_key = self._request_key(request)
        self._validation_requirements_key = self._requirements_key(request, features)
        self._validation_auto_run = bool(auto_run)
        self.validate_range_button.setEnabled(False)
        if auto_run:
            self.run_button.setEnabled(False)
            self.stage.setText("VALIDATING REQUIRED RUN DATA")
            self.progress.setRange(0, 0)
            if hasattr(self, "run_progress_detail"):
                self.run_progress_detail.setText(
                    "Required execution candles and structural benchmark warm-up are being validated before strategy execution."
                )
        self._set_readiness(
            "CHECKING DATA…",
            "Checking all required run data, including structural market-regime warm-up when enabled. Cached validation is reused.",
        )

        thread = QThread(self)
        worker = RequiredRunDataValidationWorker(self.service, request, features)
        self._validation_thread = thread
        self._validation_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._validation_finished)
        worker.failed.connect(self._validation_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _validation_finished(self, report) -> None:
        captured = self._validation_requirements_key
        try:
            request = self.request_model()
            current = self._requirements_key(request, self._current_features())
        except Exception:
            current = None
        if captured is not None and current != captured:
            # Make the parent stale-request guard reject this result and schedule
            # a fresh validation instead of accepting a result for old features.
            self._validation_request_key = ("STALE_RUN_REQUIREMENTS",)
        super()._validation_finished(report)
        self._validation_requirements_key = None

    def _validation_failed(self, message: str) -> None:
        self._validation_requirements_key = None
        super()._validation_failed(message)

    def _render_range_validation(self, report) -> None:
        strategy_interval = self.strategy_tf.currentData()
        intrabar_interval = self.intrabar_tf.currentData()
        requirement = self._current_benchmark_requirement()
        strategy_error = False
        intrabar_error = False
        benchmark_error = False
        lines: list[str] = []

        for dataset in report.datasets:
            if self._is_benchmark_report(dataset, requirement):
                has_error = dataset.status is DataQualityStatus.ERROR
                benchmark_error = benchmark_error or has_error
                if not has_error:
                    lines.append(
                        f"✓ Market regime benchmark {dataset.symbol} {dataset.interval} has the required "
                        f"{requirement.warmup_days}-day pre-start history."
                    )
                    continue
                lines.append(
                    f"✗ Market regime benchmark {dataset.symbol} {dataset.interval} does not have the required "
                    f"{requirement.warmup_days}-day warm-up before the selected start date."
                )
                for issue in dataset.issues:
                    self._append_issue_line(lines, issue, dataset)
                earliest = self._benchmark_earliest_research_start(dataset, requirement)
                if earliest is not None:
                    lines.append(
                        f"   Earliest research start supported by this benchmark: {earliest.date().isoformat()}."
                    )
                continue

            is_strategy = dataset.interval == strategy_interval
            role = "Strategy" if is_strategy else "Intrabar"
            has_error = dataset.status is DataQualityStatus.ERROR
            if is_strategy:
                strategy_error = strategy_error or has_error
            elif intrabar_interval and dataset.interval == intrabar_interval:
                intrabar_error = intrabar_error or has_error

            if not has_error:
                lines.append(
                    f"✓ {role} {dataset.interval} candles are complete for the selected range."
                )
                continue
            lines.append(f"✗ {role} {dataset.interval} candles are incomplete.")
            for issue in dataset.issues:
                self._append_issue_line(lines, issue, dataset)

        blocked = self._quality_has_errors(report)
        if benchmark_error:
            lines.append(
                "The selected execution candles can be complete while the run is still blocked: "
                "the structural market-regime benchmark is also a required strategy input."
            )
        elif blocked and intrabar_error and not strategy_error:
            lines.append(
                "Only the selected intrabar / exit-detail data is blocking this run. "
                "Use Strategy Bars Only, choose another intrabar timeframe above, or repair/download the missing data."
            )
        elif blocked:
            lines.append(
                "Required strategy data is incomplete. Repair/download the missing candles, use a safe boundary adjustment when offered, or choose another range."
            )
        else:
            lines.append(
                "All required run data is validated. Optional research gaps do not block the backtest."
            )

        self._set_readiness(
            "NOT READY" if blocked else "READY TO RUN",
            "\n".join(lines),
            state="blocked" if blocked else "ready",
        )
        self.use_strategy_bars_button.setEnabled(
            bool(intrabar_interval and intrabar_error and not strategy_error and not benchmark_error)
        )

        self._validated_safe_dates = self._safe_dates_from_report(report)
        self.use_validated_range_button.setEnabled(self._validated_safe_dates is not None)
        if self._validated_safe_dates:
            start_date, end_date = self._validated_safe_dates
            self.use_validated_range_button.setToolTip(
                f"Set the research request to safe validated boundaries: "
                f"{start_date.isoformat()} → {end_date.isoformat()}."
            )
        else:
            self.use_validated_range_button.setToolTip(
                "No boundary-only date adjustment can make all required run data valid."
            )

    def _catalog_coverage_for_report(self, report):
        """Return archive-level coverage for a missing required dataset, if known."""
        catalog = getattr(self.service, "catalog", None)
        coverage = getattr(catalog, "coverage", None)
        if not callable(coverage):
            return None
        try:
            request = self.request_model()
            report_symbol = str(report.symbol).upper()
            if str(request.symbol).upper() != report_symbol:
                request = replace(request, symbol=report_symbol)
            rows = coverage(request)
        except Exception:
            return None

        dataset_key = self._dataset_key(report.dataset)
        for row in rows:
            if self._dataset_key(row.get("dataset")) != dataset_key:
                continue
            if row.get("interval") != report.interval:
                continue
            row_symbol = row.get("symbol")
            if row_symbol and str(row_symbol).upper() != report_symbol:
                continue
            if int(row.get("archive_count") or 0) <= 0:
                continue
            return row
        return None

    def _missing_dataset_line(self, report) -> str:
        interval = report.interval or "event"
        requested = (
            f"{self._time_text(report.requested_start)} → "
            f"{self._time_text(report.requested_end)}"
        )
        coverage = self._catalog_coverage_for_report(report)
        if coverage is None:
            return (
                f"   No {interval} candle archives are available for {report.symbol}. "
                f"Requested range: {requested}."
            )

        first = coverage.get("first_period")
        last = coverage.get("last_period")
        if first not in (None, "") and last not in (None, ""):
            available = f"{self._time_text(first)} → {self._time_text(last)}"
            return (
                f"   No {interval} candles overlap the requested range {requested}. "
                f"Available catalog coverage: {available}."
            )
        if first not in (None, ""):
            return (
                f"   No {interval} candles overlap the requested range {requested}. "
                f"Available catalog coverage starts at {self._time_text(first)}."
            )
        if last not in (None, ""):
            return (
                f"   No {interval} candles overlap the requested range {requested}. "
                f"Available catalog coverage ends at {self._time_text(last)}."
            )
        return (
            f"   No {interval} candles overlap the requested range {requested}. "
            "Matching archives exist, but their time boundaries are unknown; see Data Library."
        )

    def _append_issue_line(self, lines: list[str], issue, report=None) -> None:
        if issue.code == "DATASET_MISSING":
            if report is not None:
                lines.append(self._missing_dataset_line(report))
            else:
                lines.append("   Required candle data is unavailable for the selected range.")
            return

        gap_codes = {
            "LEADING_COVERAGE_GAP",
            "LEADING_SOURCE_COVERAGE_GAP",
            "TRAILING_COVERAGE_GAP",
            "TRAILING_SOURCE_COVERAGE_GAP",
            "MISSING_INTERNAL_INTERVAL",
        }
        if issue.code in gap_codes:
            if issue.first_timestamp or issue.last_timestamp:
                span = (
                    f"{self._time_text(issue.first_timestamp)} → "
                    f"{self._time_text(issue.last_timestamp)}"
                )
            elif issue.details.get("coverage_start"):
                span = self._time_text(issue.details["coverage_start"])
            elif issue.details.get("coverage_end"):
                span = self._time_text(issue.details["coverage_end"])
            else:
                span = "see Data Library technical detail"
            lines.append(
                f"   {self._issue_text(issue).capitalize()}: {issue.count:,} · {span}"
            )
            return

        # A required dataset can be blocked for reasons other than missing-grid
        # coverage (for example malformed/off-grid timestamps, conflicting
        # archives, invalid OHLC, or a schema problem). Never hide those errors
        # behind the generic "candles are incomplete" line.
        if issue.severity is not DataQualityStatus.ERROR:
            return

        message = str(getattr(issue, "message", "") or self._issue_text(issue)).strip()
        timestamp_span = None
        if issue.first_timestamp or issue.last_timestamp:
            timestamp_span = (
                f"{self._time_text(issue.first_timestamp)} → "
                f"{self._time_text(issue.last_timestamp)}"
            )

        count = int(getattr(issue, "count", 0) or 0)
        count_text = f" · {count:,} affected" if count else ""
        span_text = f" · {timestamp_span}" if timestamp_span else ""
        lines.append(
            f"   {message} [{issue.code}]{count_text}{span_text}"
        )

    @staticmethod
    def _ceil_day(timestamp: pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(timestamp)
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        floor = value.normalize()
        return floor if value == floor else floor + pd.Timedelta(days=1)

    def _benchmark_earliest_research_start(self, report, requirement):
        if report.complete_start in (None, ""):
            return None
        start = pd.Timestamp(report.complete_start)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        return self._ceil_day(start + pd.Timedelta(days=requirement.warmup_days))

    def _safe_dates_from_report(self, report):
        boundary_codes = {
            "LEADING_COVERAGE_GAP",
            "LEADING_SOURCE_COVERAGE_GAP",
            "TRAILING_COVERAGE_GAP",
            "TRAILING_SOURCE_COVERAGE_GAP",
        }
        if any(
            issue.code == "MISSING_INTERNAL_INTERVAL"
            and issue.severity is DataQualityStatus.ERROR
            for dataset in report.datasets
            for issue in dataset.issues
        ):
            return None
        if not any(
            issue.code in boundary_codes
            for dataset in report.datasets
            for issue in dataset.issues
        ):
            return None

        requirement = self._current_benchmark_requirement()
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for dataset in report.datasets:
            if dataset.complete_start:
                start = pd.Timestamp(dataset.complete_start)
                if self._is_benchmark_report(dataset, requirement) and requirement is not None:
                    start = start + pd.Timedelta(days=requirement.warmup_days)
                starts.append(start)
            if dataset.complete_end:
                ends.append(pd.Timestamp(dataset.complete_end))
        if not starts or not ends:
            return None

        safe_start = self._ceil_day(max(starts))
        safe_end = min(ends)
        if safe_end.tzinfo is None:
            safe_end = safe_end.tz_localize("UTC")
        else:
            safe_end = safe_end.tz_convert("UTC")
        safe_end = safe_end.normalize()
        if safe_start >= safe_end:
            return None

        current_start = pd.Timestamp(self.request_model().period_start)
        current_end = pd.Timestamp(self.request_model().period_end)
        if current_start.tzinfo is None:
            current_start = current_start.tz_localize("UTC")
        if current_end.tzinfo is None:
            current_end = current_end.tz_localize("UTC")
        if safe_start <= current_start and safe_end >= current_end:
            return None
        return safe_start.date(), safe_end.date()
