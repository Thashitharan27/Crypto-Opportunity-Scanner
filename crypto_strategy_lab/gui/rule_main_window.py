"""Rule-based researcher GUI built on the stable v2 application shell.

The shell/data/results plumbing stays unchanged. Strategy authoring and trade
management use a single rule-based thesis and one base execution configuration;
mature six-way engine inputs are generated only when building the run config.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
from PySide6.QtCore import QObject, QDate, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crypto_strategy_lab.data import DataQualityStatus
from crypto_strategy_lab.data_lake_config import ExecutionProfileConfig
from crypto_strategy_lab.strategy_rule_model import (
    SUPPORT_RESISTANCE_RULE_EVIDENCE,
    common_execution_profile,
    compile_profiles,
    uses_support_resistance_rules,
)
from .rule_strategy_builder import DIRECTION_LABELS, RuleStrategyBuilder
from .v2_main_window import (
    DataclassForm,
    EXECUTION_PROFILE_GROUPS,
    MainWindow as LegacyMainWindow,
    TIMEFRAME_LABELS,
    display_percentage,
    timeframe_label,
    timeframe_minutes,
)


class RequiredDataValidationWorker(QObject):
    """Validate only required execution candles without running the strategy."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service, request):
        super().__init__()
        self.service = service
        self.request = request

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.service.required_data_quality(self.request))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(LegacyMainWindow):
    """Authoritative desktop GUI with profile-free strategy authoring."""

    def __init__(self, startup_status=None, service=None):
        # LegacyMainWindow supplies the battle-tested setup/data/results/reporting
        # shell. During this call our guarded overrides delegate back until the
        # rule widgets exist.
        super().__init__(startup_status=startup_status, service=service)

        self.rule_builder = RuleStrategyBuilder()
        self.base_execution_form = DataclassForm(
            ExecutionProfileConfig(), groups=EXECUTION_PROFILE_GROUPS
        )
        self._replace_strategy_page()
        self._replace_execution_page()
        self.rule_builder.changed.connect(self._refresh_summary_from_widgets)
        self.base_execution_form.changed.connect(self._refresh_summary_from_widgets)
        self.apply_config(self.config)

    def _data_panel(self):
        """Add explicit request validation without making catalog refresh expensive."""
        box = super()._data_panel()
        form = box.layout()

        self.range_validation = QLabel(
            "NOT VALIDATED — catalog availability does not prove every selected candle exists."
        )
        self.range_validation.setWordWrap(True)
        self.range_validation.setStyleSheet("color:#52606d")

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        self.validate_range_button = QPushButton("Validate Selected Range")
        self.validate_range_button.setToolTip(
            "Checks actual required candle timestamps using the normal cached data-quality contract. "
            "It does not run strategy features or the simulator."
        )
        self.use_validated_range_button = QPushButton("Use Safe Available Dates")
        self.use_validated_range_button.setEnabled(False)
        self.validate_range_button.clicked.connect(self.validate_selected_range)
        self.use_validated_range_button.clicked.connect(self._apply_validated_range)
        row.addWidget(self.validate_range_button)
        row.addWidget(self.use_validated_range_button)
        row.addStretch()

        form.addRow("Selected Range Validation", self.range_validation)
        form.addRow("", controls)

        self._validated_request_key = None
        self._validated_report = None
        self._validated_safe_dates = None
        self._validation_thread = None
        self._validation_worker = None
        self._validation_request_key = None
        self._validation_auto_run = False
        return box

    def _replace_page(self, index: int, page: QWidget) -> None:
        old = self.pages.widget(index)
        was_current = self.pages.currentIndex() == index
        self.pages.removeWidget(old)
        old.deleteLater()
        self.pages.insertWidget(index, page)
        if was_current:
            self.pages.setCurrentIndex(index)

    def _replace_strategy_page(self) -> None:
        page = self._page("Strategy Builder", self._scroll(self.rule_builder))
        self._replace_page(1, page)

    def _replace_execution_page(self) -> None:
        note = QLabel(
            "Risk applies to the shared account. Trade management below is one base execution plan "
            "for the strategy; it is no longer duplicated across Bull/Bear/Sideways profiles."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#eef5fb; padding:8px; border:1px solid #c8d9e8"
        )

        management = QGroupBox("Base Trade Management")
        management_layout = QVBoxLayout(management)
        management_layout.addWidget(self.base_execution_form)

        # execution_form is a live shared widget, so setWidget reparents it out of
        # the old page before that page is deleted.
        page = self._page(
            "Risk & Execution",
            note,
            self._scroll(self.execution_form),
            self._scroll(management),
            self._risk_explanation(),
        )
        self._replace_page(3, page)

    @staticmethod
    def _request_key(request) -> tuple:
        market = getattr(request.market, "value", request.market)
        return (
            request.exchange,
            str(market),
            request.symbol.upper(),
            pd.Timestamp(request.period_start).isoformat(),
            pd.Timestamp(request.period_end).isoformat(),
            request.strategy_timeframe,
            request.intrabar_timeframe,
        )

    @staticmethod
    def _time_text(value) -> str:
        if value in (None, "", "—"):
            return "—"
        try:
            return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            try:
                timestamp = pd.Timestamp(value)
                timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
                return timestamp.strftime("%Y-%m-%d %H:%M UTC")
            except (TypeError, ValueError):
                return str(value)

    def _coverage_detail_rows(self) -> list[dict]:
        rows = []
        for row in range(self.coverage.rowCount()):
            if not all(self.coverage.item(row, column) for column in range(6)):
                continue
            rows.append({
                "dataset": self.coverage.item(row, 0).text(),
                "interval": None if self.coverage.item(row, 1).text() == "—" else self.coverage.item(row, 1).text(),
                "first_period": self.coverage.item(row, 2).text(),
                "last_period": self.coverage.item(row, 3).text(),
                "archive_count": self.coverage.item(row, 4).text(),
                "state": self.coverage.item(row, 5).text(),
            })
        return rows

    def _render_catalog_availability(self) -> None:
        if not hasattr(self, "datasets"):
            return
        rows = self._coverage_detail_rows()
        candle_rows = [row for row in rows if row["dataset"].lower() == "klines"]
        lines = []

        def candle_line(role: str, interval: str):
            matches = [row for row in candle_rows if row["interval"] == interval]
            if not matches:
                return f"{role} candles ({interval}): CATALOG UNAVAILABLE"
            row = matches[0]
            return (
                f"{role} candles ({interval}): CATALOG {row['state']} — "
                f"archive coverage {self._time_text(row['first_period'])} → "
                f"{self._time_text(row['last_period'])}"
            )

        strategy_interval = self.strategy_tf.currentData()
        lines.append(candle_line("Strategy", strategy_interval))
        intrabar_interval = self.intrabar_tf.currentData()
        if intrabar_interval:
            lines.append(candle_line("Intrabar", intrabar_interval))

        optional = {}
        for row in rows:
            if row in candle_rows:
                continue
            family = self._dataset_family(row["dataset"])
            optional.setdefault(family, []).append(row["state"])
        for family, states in sorted(optional.items()):
            state = (
                "AVAILABLE" if states and all(item == "AVAILABLE" for item in states)
                else "PARTIAL" if any(item == "AVAILABLE" for item in states)
                else "UNAVAILABLE"
            )
            lines.append(f"{family}: {state}")

        lines.append(
            "Catalog status is archive-level only. Validate Selected Range checks actual candle continuity."
        )
        self.datasets.setText("\n".join(lines))

    def _invalidate_range_validation(self) -> None:
        if not hasattr(self, "range_validation"):
            return
        self._validated_request_key = None
        self._validated_report = None
        self._validated_safe_dates = None
        self.use_validated_range_button.setEnabled(False)
        self.use_validated_range_button.setToolTip("")
        self.range_validation.setText(
            "NOT VALIDATED — click Validate Selected Range, or press Run to validate before execution."
        )
        self.range_validation.setStyleSheet("color:#52606d")

    def refresh_coverage(self):
        super().refresh_coverage()
        if not hasattr(self, "range_validation"):
            return
        self._invalidate_range_validation()
        self._render_catalog_availability()

    def _data_state(self):
        catalog_state, _detail = self.data_readiness(
            self._coverage_rows_from_table(), self.strategy_tf.currentData(),
            self.intrabar_tf.currentData(),
        )
        if catalog_state == "BLOCKED":
            return "BLOCKED"
        if not hasattr(self, "_validated_request_key"):
            return catalog_state
        try:
            key = self._request_key(self.request_model())
        except Exception:
            return catalog_state
        if self._validated_request_key != key or self._validated_report is None:
            return f"CATALOG {catalog_state} — VALIDATION PENDING"
        if any(
            dataset.status is DataQualityStatus.ERROR
            for dataset in self._validated_report.datasets
        ):
            return "BLOCKED"
        return "WARN" if catalog_state == "WARN" else "READY"

    @staticmethod
    def _quality_has_errors(report) -> bool:
        return any(
            dataset.status is DataQualityStatus.ERROR
            for dataset in report.datasets
        )

    def validate_selected_range(self):
        self._begin_required_data_validation(auto_run=False)

    def _begin_required_data_validation(self, *, auto_run: bool) -> None:
        if self._validation_thread is not None:
            return
        try:
            request = self.request_model()
            if request.period_start >= request.period_end:
                raise ValueError("Start Date must be before End Date")
        except Exception as exc:
            QMessageBox.warning(self, "Invalid research request", str(exc))
            return

        catalog_state, detail = self.data_readiness(
            self._coverage_rows_from_table(),
            request.strategy_timeframe,
            request.intrabar_timeframe,
        )
        if catalog_state == "BLOCKED":
            self.range_validation.setText("BLOCKED AT CATALOG — " + detail)
            self.range_validation.setStyleSheet("color:#a61b1b; font-weight:600")
            if auto_run:
                QMessageBox.warning(self, "Required candle data unavailable", detail)
            return

        if not hasattr(self.service, "required_data_quality"):
            if auto_run:
                super().start_run()
            else:
                self.range_validation.setText(
                    "Exact range validation is unavailable from the current application service."
                )
            return

        self._validation_request_key = self._request_key(request)
        self._validation_auto_run = bool(auto_run)
        self.validate_range_button.setEnabled(False)
        if auto_run:
            self.run_button.setEnabled(False)
        self.range_validation.setText(
            "VALIDATING — checking actual required candle timestamps. Cached checks are reused."
        )
        self.range_validation.setStyleSheet("color:#52606d; font-weight:600")
        self.stage.setText("VALIDATING SELECTED CANDLE RANGE")
        self.progress.setRange(0, 0)
        if hasattr(self, "run_progress_detail"):
            self.run_progress_detail.setText(
                "This validates required candles only; strategy simulation has not started."
            )

        thread = QThread(self)
        worker = RequiredDataValidationWorker(self.service, request)
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
        auto_run = self._validation_auto_run
        request_key = self._validation_request_key
        self._validation_thread = None
        self._validation_worker = None
        self._validation_request_key = None
        self._validation_auto_run = False
        self.validate_range_button.setEnabled(True)
        self.run_button.setEnabled(True)

        try:
            current_key = self._request_key(self.request_model())
        except Exception:
            current_key = None
        if current_key != request_key:
            self._invalidate_range_validation()
            self.range_validation.setText(
                "STALE VALIDATION — the request changed while validation was running."
            )
            return

        self._validated_request_key = request_key
        self._validated_report = report
        self.render_data_quality(report)
        self._render_range_validation(report)
        self._refresh_summary_from_widgets()

        if self._quality_has_errors(report):
            self.stage.setText("BLOCKED — REQUIRED CANDLE COVERAGE")
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Blocked")
            if hasattr(self, "run_progress_detail"):
                self.run_progress_detail.setText(
                    "Adjust the selected dates or repair/download the missing required candle data."
                )
            if auto_run:
                QMessageBox.warning(
                    self,
                    "Required candle range is incomplete",
                    "The strategy was not started. The Setup tab now shows the exact validated coverage and any safe date adjustment available.",
                )
            return

        self.stage.setText("Required candle range validated")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat("Validated")
        if hasattr(self, "run_progress_detail"):
            self.run_progress_detail.setText(
                "Required strategy/intrabar candles passed validation."
            )
        if auto_run:
            super().start_run()

    def _validation_failed(self, message: str) -> None:
        auto_run = self._validation_auto_run
        self._validation_thread = None
        self._validation_worker = None
        self._validation_request_key = None
        self._validation_auto_run = False
        self.validate_range_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.range_validation.setText("VALIDATION FAILED — " + message)
        self.range_validation.setStyleSheet("color:#a61b1b; font-weight:600")
        self.stage.setText("VALIDATION FAILED")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Failed")
        if auto_run:
            QMessageBox.warning(self, "Candle validation failed", message)

    def _render_range_validation(self, report) -> None:
        lines = []
        for dataset in report.datasets:
            role = (
                "Strategy"
                if dataset.interval == self.strategy_tf.currentData()
                else "Intrabar"
            )
            state = "BLOCKED" if dataset.status is DataQualityStatus.ERROR else dataset.status.value
            lines.append(
                f"{role} candles ({dataset.interval}): {state} — observed "
                f"{self._time_text(dataset.observed_start)} → {self._time_text(dataset.observed_end)}"
            )
            for issue in dataset.issues:
                if issue.code not in {
                    "LEADING_COVERAGE_GAP",
                    "LEADING_SOURCE_COVERAGE_GAP",
                    "TRAILING_COVERAGE_GAP",
                    "TRAILING_SOURCE_COVERAGE_GAP",
                    "MISSING_INTERNAL_INTERVAL",
                }:
                    continue
                label = {
                    "LEADING_COVERAGE_GAP": "Missing at start",
                    "LEADING_SOURCE_COVERAGE_GAP": "Source starts after request",
                    "TRAILING_COVERAGE_GAP": "Missing at end",
                    "TRAILING_SOURCE_COVERAGE_GAP": "Source ends before request",
                    "MISSING_INTERNAL_INTERVAL": "Internal candle gap",
                }[issue.code]
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
                    span = "see Data Status"
                lines.append(f"  {label}: {issue.count} · {span}")
            if dataset.complete_start and dataset.complete_start != dataset.requested_start:
                lines.append(
                    f"  Earliest validated boundary: {self._time_text(dataset.complete_start)}"
                )
            if dataset.complete_end and dataset.complete_end != dataset.requested_end:
                lines.append(
                    f"  Latest validated boundary: {self._time_text(dataset.complete_end)}"
                )

        has_errors = self._quality_has_errors(report)
        self.range_validation.setText("\n".join(lines) or "No required candle datasets were selected.")
        self.range_validation.setStyleSheet(
            "color:#a61b1b; font-weight:600" if has_errors
            else "color:#1f6f43; font-weight:600"
        )
        self._validated_safe_dates = self._safe_dates_from_report(report)
        self.use_validated_range_button.setEnabled(self._validated_safe_dates is not None)
        if self._validated_safe_dates:
            start_date, end_date = self._validated_safe_dates
            self.use_validated_range_button.setToolTip(
                f"Set the date-only request to the safe validated boundaries: "
                f"{start_date.isoformat()} → {end_date.isoformat()}."
            )
        else:
            self.use_validated_range_button.setToolTip(
                "No boundary-only date adjustment can fix the current validation result."
            )

    def _safe_dates_from_report(self, report):
        boundary_codes = {
            "LEADING_COVERAGE_GAP",
            "LEADING_SOURCE_COVERAGE_GAP",
            "TRAILING_COVERAGE_GAP",
            "TRAILING_SOURCE_COVERAGE_GAP",
        }
        if not any(
            issue.code in boundary_codes
            for dataset in report.datasets
            for issue in dataset.issues
        ):
            return None
        starts = [
            pd.Timestamp(dataset.complete_start)
            for dataset in report.datasets if dataset.complete_start
        ]
        ends = [
            pd.Timestamp(dataset.complete_end)
            for dataset in report.datasets if dataset.complete_end
        ]
        if not starts or not ends:
            return None
        start = max(starts)
        end = min(ends)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        start_floor = start.normalize()
        safe_start = start_floor + pd.Timedelta(days=1) if start != start_floor else start_floor
        safe_end = end.normalize()
        if safe_start >= safe_end:
            return None
        current_start = pd.Timestamp(self.request_model().period_start)
        current_end = pd.Timestamp(self.request_model().period_end)
        if safe_start <= current_start and safe_end >= current_end:
            return None
        return safe_start.date(), safe_end.date()

    def _apply_validated_range(self) -> None:
        if not self._validated_safe_dates:
            return
        start_date, end_date = self._validated_safe_dates
        self.start.blockSignals(True)
        self.end.blockSignals(True)
        try:
            self.start.setDate(QDate(start_date.year, start_date.month, start_date.day))
            self.end.setDate(QDate(end_date.year, end_date.month, end_date.day))
        finally:
            self.start.blockSignals(False)
            self.end.blockSignals(False)
        self.refresh_coverage()

    def start_run(self):
        """Never enter strategy execution before exact required candles validate."""
        if self._validation_thread is not None:
            return
        try:
            request, config = self.request_model(), self.build_config()
            config.validate()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid research request", str(exc))
            return

        catalog_state, detail = self.data_readiness(
            self._coverage_rows_from_table(),
            request.strategy_timeframe,
            request.intrabar_timeframe,
        )
        if catalog_state == "BLOCKED":
            self.range_validation.setText("BLOCKED AT CATALOG — " + detail)
            self.range_validation.setStyleSheet("color:#a61b1b; font-weight:600")
            QMessageBox.warning(self, "Required candle data unavailable", detail)
            return

        key = self._request_key(request)
        if self._validated_request_key == key and self._validated_report is not None:
            if self._quality_has_errors(self._validated_report):
                QMessageBox.warning(
                    self,
                    "Required candle range is incomplete",
                    "The current request is already validated as incomplete. Adjust the dates or source data before running.",
                )
                return
            super().start_run()
            return
        self._begin_required_data_validation(auto_run=True)

    def build_config(self):
        if not hasattr(self, "rule_builder"):
            return super().build_config()

        intrabar = self.request_model().intrabar_timeframe
        data = replace(
            self.config.data,
            strategy_timeframe_minutes=timeframe_minutes(
                self.strategy_tf.currentText()
            ),
            use_intrabar_data=intrabar is not None,
            intrabar_timeframe_minutes=(
                timeframe_minutes(intrabar)
                if intrabar
                else self.config.data.intrabar_timeframe_minutes
            ),
        )
        features = self.feature_form.value(self.config.features)
        execution_base = self.execution_form.value(self.config.execution)
        reporting = replace(
            self.reporting_form.value(self.config.reporting),
            output_dir=self.output_root.text(),
        )

        authored = self.rule_builder.strategy_values()
        required_rules = authored["required_rules"]
        veto_rules = authored["veto_rules"]
        flip_rules = authored["flip_rules"]
        # Rule dependencies are authoritative. A researcher does not need to
        # remember a second S/R enable switch just to use S/R evidence.
        if uses_support_resistance_rules(required_rules, veto_rules, flip_rules):
            features = replace(features, enable_support_resistance_analysis=True)

        base_execution = self.base_execution_form.value(
            common_execution_profile(self.config.execution.profiles)
        )
        strategy_profiles, execution_profiles = compile_profiles(
            direction_mode=authored.pop("direction_mode"),
            market_permissions=authored.pop("market_permissions"),
            required_rules=authored.pop("required_rules"),
            veto_rules=authored.pop("veto_rules"),
            flip_rules=authored.pop("flip_rules"),
            rsi_period=features.mean_reversion_rsi_period,
            momentum_lookback_hours=authored.pop("momentum_lookback_hours"),
            base_execution=base_execution,
        )
        strategy = replace(
            self.config.strategy,
            profiles=strategy_profiles,
            strategy_profile_run_mode="COMBINED_SHARED_CAPITAL",
            **authored,
        )
        execution = replace(execution_base, profiles=execution_profiles)
        result = replace(
            self.config,
            data=data,
            features=features,
            strategy=strategy,
            execution=execution,
            reporting=reporting,
        )
        result.validate()
        return result

    def apply_config(self, config):
        if not hasattr(self, "rule_builder"):
            return super().apply_config(config)

        self._applying_config = True
        try:
            self.config = config
            data = config.data
            self.strategy_tf.setCurrentText(
                timeframe_label(data.strategy_timeframe_minutes)
            )
            self.intrabar_tf.setCurrentText(
                timeframe_label(data.intrabar_timeframe_minutes)
                if data.use_intrabar_data
                else None
            )
            self.feature_form.set_value(config.features)
            self.execution_form.set_value(config.execution)
            self.output_root.setText(config.reporting.output_dir)
            self.reporting_form.set_value(config.reporting)
            self.rule_builder.set_from_strategy(config.strategy)
            self.rule_builder.set_feature_status(config.features)
            self.base_execution_form.set_value(
                common_execution_profile(config.execution.profiles)
            )
        finally:
            self._applying_config = False
        self.rule_builder.refresh_summary()
        self._render_research_summary(config)

    def _refresh_summary_from_widgets(self):
        if not hasattr(self, "rule_builder"):
            return super()._refresh_summary_from_widgets()
        if self._applying_config:
            return
        try:
            config = self.build_config()
            self.rule_builder.set_feature_status(config.features)
            self.rule_builder.refresh_summary()
            self._render_research_summary(config)
        except (ValueError, TypeError, KeyError):
            # Invalid/incomplete edits remain visible in the controls and are
            # reported when Run is pressed; live summary refresh must stay quiet.
            return

    def _render_research_summary(self, config):
        if not hasattr(self, "rule_builder"):
            return super()._render_research_summary(config)

        permissions = self.rule_builder.market_permissions()
        direction = self.rule_builder.direction_mode.currentData()
        required_rules = self.rule_builder.required_rules.rules()
        veto_rules = self.rule_builder.veto_rules.rules()
        flip_rules = self.rule_builder.flip_rules.rules()
        required = len(required_rules)
        veto = len(veto_rules)
        all_rules = (*required_rules, *veto_rules, *flip_rules)
        pressure_evidence = {
            "DI_PRESSURE_STATE",
            "DI_SPREAD_CHANGE",
            "DIRECTIONAL_DI_CHANGE",
            "OPPOSING_DI_CHANGE",
        }
        pressure_rule_count = sum(
            rule["evidence"] in pressure_evidence for rule in all_rules
        )
        pressure_text = (
            f"Rule Evidence ({pressure_rule_count} rule(s))"
            if pressure_rule_count
            else "Available to Rules"
        )
        sr_rule_count = sum(
            rule["evidence"] in SUPPORT_RESISTANCE_RULE_EVIDENCE
            for rule in all_rules
        )
        sr_text = (
            f"Rule Evidence ({sr_rule_count} rule(s))"
            if sr_rule_count
            else "Analyze Only"
            if config.features.enable_support_resistance_analysis
            else "Off"
        )
        intrabar = (
            f"{config.data.intrabar_timeframe_minutes}m exits"
            if config.data.use_intrabar_data
            else "bar-close exits"
        )
        risk = display_percentage(config.execution.risk_per_leg)
        base_execution = self.base_execution_form.value(
            common_execution_profile(config.execution.profiles)
        )

        self.current_research.setText(
            f"CURRENT RESEARCH\n\n{self.symbol.currentText() or 'BTCUSDT'}\n"
            f"{timeframe_label(config.data.strategy_timeframe_minutes)} strategy / {intrabar}\n\n"
            f"Direction  {DIRECTION_LABELS[direction]}\n"
            f"Markets  {len(permissions)} of 6 allowed\n"
            f"Entry rules  {required}\nVeto rules  {veto}\n"
            f"DI Pressure  {pressure_text}\n"
            f"S/R  {sr_text}\n"
            f"MR Context  {'ANALYZE' if config.strategy.enable_mean_reversion_analysis else 'OFF'}\n"
            f"Trade Flow  {'ANALYZE' if config.features.trade_flow_enabled else 'OFF'}\n"
            f"Order Book  {'ANALYZE' if config.features.order_book_enabled else 'OFF'}\n\n"
            f"Base risk  {risk}\nMax trades  {config.execution.max_active_pairs}\n\n"
            f"Data  {self._data_state()}"
        )
        self.risk_explanation.setText(
            f"Base Risk: {risk}. At ${config.execution.initial_equity:,.2f}, planned full-stop "
            f"account risk is ${config.execution.initial_equity * config.execution.risk_per_leg:,.2f}. "
            f"Base stop distance: {base_execution.stop_loss_multiple:g} distance units; "
            f"fixed target: {base_execution.reward_risk_ratio:g}R."
        )

        if hasattr(self, "review_summary"):
            allowed = (
                ", ".join(item.replace("_", " ").title() for item in permissions)
                or "None"
            )
            self.review_summary.setText(
                f"{self.symbol.currentText() or 'BTCUSDT'} — "
                f"{TIMEFRAME_LABELS[timeframe_label(config.data.strategy_timeframe_minutes)]} Research\n\n"
                f"Direction: {DIRECTION_LABELS[direction]}\n"
                f"Allowed markets/sides: {allowed}\n"
                f"Entry rules: {required} required · {veto} veto\n"
                f"DI pressure: {pressure_text}\n"
                f"Mean Reversion: {'Analyze Only' if config.strategy.enable_mean_reversion_analysis else 'Off'}\n"
                f"Support / Resistance: {sr_text}\n\n"
                f"Starting Equity: ${config.execution.initial_equity:,.2f}\n"
                f"Base Risk: {risk}\n"
                f"Stop: {base_execution.stop_loss_multiple:g} distance units · "
                f"Target: {base_execution.reward_risk_ratio:g}R\n"
                f"Maximum Active Trades: {config.execution.max_active_pairs}\n"
                f"Reports: {config.reporting.analysis_level}\n\n"
                f"DATA STATUS: {self._data_state()}"
            )