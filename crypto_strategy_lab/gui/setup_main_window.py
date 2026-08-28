"""Researcher-focused Setup workspace layered over the rule-based GUI.

Setup answers three questions only: what is being tested, what data this run will
actually consume, and whether the selected request is ready to run. Technical
catalog and data-quality tables remain available in Data Library.
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QDate, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crypto_strategy_lab.data import DataQualityStatus, MarketKind
from crypto_strategy_lab.gui.dataset_labels import dataset_family_label
from crypto_strategy_lab.gui.rule_main_window import (
    MainWindow as RuleMainWindow,
    RequiredDataValidationWorker,
)
from crypto_strategy_lab.gui.v2_main_window import (
    INTRABAR_TIMEFRAMES,
    STRATEGY_TIMEFRAMES,
    TIMEFRAME_LABELS,
    TimeframeCombo,
)


class MainWindow(RuleMainWindow):
    """Active GUI with a compact request/data/readiness Setup experience."""

    _dataset_family = staticmethod(dataset_family_label)

    def __init__(self, startup_status=None, service=None):
        super().__init__(startup_status=startup_status, service=service)
        self.feature_form.changed.connect(self._refresh_run_data_view)
        if hasattr(self, "rule_builder"):
            self.rule_builder.changed.connect(self._refresh_run_data_view)
        self._refresh_run_data_view()
        self._schedule_auto_validation()

    # ------------------------------------------------------------------
    # Setup composition
    # ------------------------------------------------------------------
    def _data_panel(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        request_box = QGroupBox("Research Request")
        form = QFormLayout(request_box)
        self.exchange = QComboBox()
        self.exchange.addItem("Binance", "binance")
        self.market = QComboBox()
        self.market.addItem("USD-M Futures", MarketKind.FUTURES_UM)
        self.symbol = QComboBox()
        self.symbol.setEditable(True)
        self.start = QDateEdit(QDate(2024, 1, 1))
        self.start.setCalendarPopup(True)
        self.end = QDateEdit(QDate.currentDate())
        self.end.setCalendarPopup(True)
        self.strategy_tf = TimeframeCombo()
        for value in STRATEGY_TIMEFRAMES:
            self.strategy_tf.addItem(TIMEFRAME_LABELS[value], value)
        self.intrabar_tf = TimeframeCombo()
        self.intrabar_tf.addItem("None — Strategy Bars Only", None)
        for value in INTRABAR_TIMEFRAMES:
            self.intrabar_tf.addItem(TIMEFRAME_LABELS[value], value)

        date_note = QLabel(
            "Research includes data from the start date up to, but not including, "
            "the selected end boundary."
        )
        date_note.setWordWrap(True)
        for label, widget in (
            ("Exchange", self.exchange),
            ("Market", self.market),
            ("Symbol", self.symbol),
            ("Start Date", self.start),
            ("End Date", self.end),
            ("Date Range", date_note),
            ("Strategy Timeframe", self.strategy_tf),
            ("Intrabar / Exit Detail", self.intrabar_tf),
        ):
            form.addRow(label, widget)
        outer.addWidget(request_box)

        used_box = QGroupBox("Data Used By This Run")
        used_layout = QVBoxLayout(used_box)
        note = QLabel(
            "Required execution data and active/automatic research sources are shown here. "
            "The full archive inventory stays in Data Library."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#52606d")
        used_layout.addWidget(note)
        self.run_data_table = QTableWidget(0, 5)
        self.run_data_table.setHorizontalHeaderLabels(
            ["Role", "Data", "Interval", "Use", "Status"]
        )
        self.run_data_table.setMinimumHeight(190)
        used_layout.addWidget(self.run_data_table)
        outer.addWidget(used_box)

        # v2 catalog refresh still writes a summary label. Keep it as a hidden
        # compatibility target; the researcher-facing view is run_data_table.
        self.datasets = QLabel()
        self.datasets.hide()

        self._validated_request_key = None
        self._validated_report = None
        self._validated_safe_dates = None
        self._validation_thread = None
        self._validation_worker = None
        self._validation_request_key = None
        self._validation_auto_run = False

        self._validation_debounce = QTimer(self)
        self._validation_debounce.setSingleShot(True)
        self._validation_debounce.setInterval(650)
        self._validation_debounce.timeout.connect(self._auto_validate_current_request)
        return container

    def _status_panel(self):
        # Keep the technical widgets alive for the existing catalog/quality
        # plumbing, but place them in Data Library instead of Setup.
        self.resolution = QLabel("Requested/effective resolution: not run")
        self.coverage = QTableWidget(0, 6)
        self.coverage.setHorizontalHeaderLabels(
            ["Dataset", "Interval", "First UTC", "Last UTC", "Partitions", "State"]
        )
        self.quality = QLabel("Data quality: not run")
        self.quality_table = QTableWidget(0, 6)
        self.quality_table.setHorizontalHeaderLabels(
            ["Dataset", "Interval", "Required", "Rows", "Status", "Issues"]
        )

        box = QGroupBox("Run Readiness")
        layout = QVBoxLayout(box)
        self.readiness_state = QLabel("CHECKING DATA…")
        self.readiness_state.setStyleSheet("font-size:18px; font-weight:700; color:#52606d")
        layout.addWidget(self.readiness_state)

        self.range_validation = QLabel(
            "Checking the selected strategy and intrabar candle range. Cached validation is reused."
        )
        self.range_validation.setWordWrap(True)
        self.range_validation.setStyleSheet("color:#52606d")
        layout.addWidget(self.range_validation)

        actions = QWidget()
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        self.validate_range_button = QPushButton("Revalidate")
        self.validate_range_button.setToolTip(
            "Re-check the exact required candle timestamps. The normal quality cache is reused."
        )
        self.use_strategy_bars_button = QPushButton("Use Strategy Bars Only")
        self.use_strategy_bars_button.setEnabled(False)
        self.use_validated_range_button = QPushButton("Use Safe Available Dates")
        self.use_validated_range_button.setEnabled(False)
        self.open_data_library_button = QPushButton("Open Data Library")
        self.validate_range_button.clicked.connect(self.validate_selected_range)
        self.use_strategy_bars_button.clicked.connect(self._use_strategy_bars_only)
        self.use_validated_range_button.clicked.connect(self._apply_validated_range)
        self.open_data_library_button.clicked.connect(self._open_data_library)
        for button in (
            self.validate_range_button,
            self.use_strategy_bars_button,
            self.use_validated_range_button,
            self.open_data_library_button,
        ):
            row.addWidget(button)
        row.addStretch()
        layout.addWidget(actions)
        return box

    def _data_library_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(super()._data_library_panel())

        technical = QGroupBox("Technical Coverage & Validation Detail")
        detail = QVBoxLayout(technical)
        note = QLabel(
            "Low-level catalog rows, partition counts, row counts and validator issue codes live here. "
            "Setup translates these diagnostics into run readiness."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#52606d")
        detail.addWidget(note)
        detail.addWidget(self.resolution)
        detail.addWidget(self.coverage)
        detail.addWidget(self.quality)
        detail.addWidget(self.quality_table)
        layout.addWidget(technical)
        return container

    # ------------------------------------------------------------------
    # Run data summary
    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_key(value) -> str:
        raw = getattr(value, "value", value)
        text = str(raw).lower()
        return text.split(".", 1)[1] if text.startswith("datasetkind.") else text

    def _catalog_state(self, dataset: str, interval: str | None = None) -> str:
        matches = []
        for row in self._coverage_detail_rows():
            if self._dataset_key(row["dataset"]) != dataset:
                continue
            if interval is not None and row["interval"] != interval:
                continue
            matches.append(str(row["state"]).upper())
        if not matches:
            return "UNAVAILABLE"
        if all(state == "AVAILABLE" for state in matches):
            return "AVAILABLE"
        if any(state == "AVAILABLE" for state in matches):
            return "PARTIAL"
        return "UNAVAILABLE"

    def _validated_interval_state(self, interval: str | None) -> str | None:
        if not interval or self._validated_report is None:
            return None
        try:
            if self._validated_request_key != self._request_key(self.request_model()):
                return None
        except Exception:
            return None
        matches = [item for item in self._validated_report.datasets if item.interval == interval]
        if not matches:
            return None
        return "BLOCKED" if any(item.status is DataQualityStatus.ERROR for item in matches) else "READY"

    @staticmethod
    def _friendly_status(state: str, *, optional: bool = False) -> str:
        state = str(state).upper()
        if state == "READY":
            return "✓ Ready"
        if state == "BLOCKED":
            return "✗ Blocked"
        if state == "AVAILABLE":
            return "✓ Available" if optional else "… Needs validation"
        if state == "PARTIAL":
            return "△ Partial"
        if state == "OFF":
            return "— Off"
        return "○ Unavailable" if optional else "✗ Unavailable"

    def _required_candle_status(self, interval: str | None) -> str:
        validated = self._validated_interval_state(interval)
        if validated:
            return self._friendly_status(validated)
        return self._friendly_status(self._catalog_state("klines", interval))

    def _add_run_data_row(self, role: str, data: str, interval: str, use: str, status: str):
        row = self.run_data_table.rowCount()
        self.run_data_table.insertRow(row)
        for column, value in enumerate((role, data, interval, use, status)):
            self.run_data_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _refresh_run_data_view(self):
        if not hasattr(self, "run_data_table"):
            return
        self.run_data_table.setRowCount(0)
        strategy_interval = self.strategy_tf.currentData()
        intrabar_interval = self.intrabar_tf.currentData()
        self._add_run_data_row(
            "Strategy",
            "Market Price Candles",
            strategy_interval or "—",
            "Required",
            self._required_candle_status(strategy_interval),
        )
        if intrabar_interval:
            self._add_run_data_row(
                "Intrabar / Exits",
                "Market Price Candles",
                intrabar_interval,
                "Required",
                self._required_candle_status(intrabar_interval),
            )
        else:
            self._add_run_data_row(
                "Intrabar / Exits", "Strategy bars", "—", "Off", "— Strategy bars only"
            )

        try:
            features = self.feature_form.value(self.config.features)
        except Exception:
            features = self.config.features

        funding = self._catalog_state("funding_rate")
        self._add_run_data_row(
            "Research", "Funding", "Event", "Auto when covered",
            self._friendly_status(funding, optional=True),
        )

        positioning = self._catalog_state("metrics")
        self._add_run_data_row(
            "Research", "Futures Positioning / OI", "Event", "Auto when covered",
            self._friendly_status(positioning, optional=True),
        )

        mark = self._catalog_state("mark_price_klines", strategy_interval)
        index = self._catalog_state("index_price_klines", strategy_interval)
        premium = self._catalog_state("premium_index_klines", strategy_interval)
        if mark == "AVAILABLE" and index == "AVAILABLE":
            basis_status = "✓ Available" if premium == "AVAILABLE" else "✓ Available · premium optional missing"
        elif mark == "AVAILABLE" or index == "AVAILABLE":
            basis_status = "△ Partial"
        else:
            basis_status = "○ Unavailable"
        self._add_run_data_row(
            "Research", "Basis / Premium Context", strategy_interval or "—",
            "Auto when covered", basis_status,
        )

        taker_interval = str(getattr(features, "taker_flow_interval", "5m"))
        taker_state = self._catalog_state("klines", taker_interval)
        self._add_run_data_row(
            "Research", "Taker Flow", taker_interval, "Auto when covered",
            self._friendly_status(taker_state, optional=True),
        )

        if bool(getattr(features, "trade_flow_enabled", False)):
            source = str(getattr(features, "trade_flow_source", "AGG_TRADES")).lower()
            source_key = "trades" if source == "trades" else "agg_trades"
            trade_state = self._catalog_state(source_key)
            self._add_run_data_row(
                "Research", "Trade Flow", "Event", "Enabled",
                self._friendly_status(trade_state, optional=True),
            )
        else:
            self._add_run_data_row("Research", "Trade Flow", "—", "Off", "— Off")

        if bool(getattr(features, "order_book_enabled", False)):
            ticker = self._catalog_state("book_ticker")
            depth = self._catalog_state("book_depth")
            order_state = "AVAILABLE" if "AVAILABLE" in {ticker, depth} else "UNAVAILABLE"
            self._add_run_data_row(
                "Research", "Order Book", "Event", "Enabled",
                self._friendly_status(order_state, optional=True),
            )
        else:
            self._add_run_data_row("Research", "Order Book", "—", "Off", "— Off")

        self.run_data_table.resizeColumnsToContents()
        self.run_data_table.horizontalHeader().setStretchLastSection(True)

    def _render_catalog_availability(self) -> None:
        self._refresh_run_data_view()
        if hasattr(self, "datasets"):
            self.datasets.setText("Run-specific data usage is shown in Data Used By This Run.")

    # ------------------------------------------------------------------
    # Automatic exact-range validation + plain-English readiness
    # ------------------------------------------------------------------
    def _set_readiness(self, title: str, detail: str, *, state: str = "pending") -> None:
        styles = {
            "ready": "font-size:18px; font-weight:700; color:#1f6f43",
            "blocked": "font-size:18px; font-weight:700; color:#a61b1b",
            "pending": "font-size:18px; font-weight:700; color:#52606d",
        }
        self.readiness_state.setText(title)
        self.readiness_state.setStyleSheet(styles[state])
        self.range_validation.setText(detail)
        self.range_validation.setStyleSheet(
            "color:#a61b1b; font-weight:600" if state == "blocked"
            else "color:#1f6f43; font-weight:600" if state == "ready"
            else "color:#52606d"
        )

    def _invalidate_range_validation(self) -> None:
        if not hasattr(self, "range_validation"):
            return
        self._validated_request_key = None
        self._validated_report = None
        self._validated_safe_dates = None
        self.use_validated_range_button.setEnabled(False)
        self.use_validated_range_button.setToolTip("")
        self.use_strategy_bars_button.setEnabled(False)
        self._set_readiness(
            "CHECKING DATA…",
            "The selected request changed. Exact candle continuity will be checked automatically; cached validation is reused.",
        )
        self._refresh_run_data_view()

    def refresh_coverage(self):
        super().refresh_coverage()
        if hasattr(self, "_validation_debounce"):
            self._schedule_auto_validation()

    def _schedule_auto_validation(self) -> None:
        if not hasattr(self, "_validation_debounce"):
            return
        if self._validation_thread is None:
            self._validation_debounce.start()

    def _auto_validate_current_request(self) -> None:
        if self._validation_thread is not None:
            return
        try:
            request = self.request_model()
            key = self._request_key(request)
        except Exception:
            return
        if self._validated_request_key == key and self._validated_report is not None:
            return
        self._begin_required_data_validation(auto_run=False)

    def _begin_required_data_validation(self, *, auto_run: bool) -> None:
        if self._validation_thread is not None:
            if auto_run:
                self._validation_auto_run = True
            return
        try:
            request = self.request_model()
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
        self._validation_auto_run = bool(auto_run)
        self.validate_range_button.setEnabled(False)
        if auto_run:
            self.run_button.setEnabled(False)
            self.stage.setText("VALIDATING SELECTED CANDLE RANGE")
            self.progress.setRange(0, 0)
            if hasattr(self, "run_progress_detail"):
                self.run_progress_detail.setText(
                    "Required candle validation is running before strategy execution."
                )
        self._set_readiness(
            "CHECKING DATA…",
            "Checking exact required candle timestamps. This uses the normal quality cache and does not run the strategy simulator.",
        )

        from PySide6.QtCore import QThread

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
            self._set_readiness(
                "CHECKING DATA…",
                "The request changed while validation was running. Rechecking the new selection…",
            )
            self._schedule_auto_validation()
            return

        self._validated_request_key = request_key
        self._validated_report = report
        self.render_data_quality(report)
        self._render_range_validation(report)
        self._refresh_run_data_view()
        self._refresh_summary_from_widgets()

        if self._quality_has_errors(report):
            if auto_run:
                self.stage.setText("BLOCKED — REQUIRED CANDLE COVERAGE")
                self.progress.setRange(0, 100)
                self.progress.setValue(0)
                self.progress.setFormat("Blocked")
                if hasattr(self, "run_progress_detail"):
                    self.run_progress_detail.setText(
                        "Setup shows what is missing and the available corrective actions."
                    )
                QMessageBox.warning(
                    self,
                    "Required candle range is incomplete",
                    "The strategy was not started. Setup now shows the blocking data gap and available actions.",
                )
            return

        if auto_run:
            self.stage.setText("Required candle range validated")
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("Validated")
            if hasattr(self, "run_progress_detail"):
                self.run_progress_detail.setText(
                    "Required strategy/intrabar candles passed validation."
                )
            super().start_run()

    def _validation_failed(self, message: str) -> None:
        auto_run = self._validation_auto_run
        self._validation_thread = None
        self._validation_worker = None
        self._validation_request_key = None
        self._validation_auto_run = False
        self.validate_range_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self._set_readiness("VALIDATION FAILED", message, state="blocked")
        if auto_run:
            self.stage.setText("VALIDATION FAILED")
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Failed")
            QMessageBox.warning(self, "Candle validation failed", message)

    @staticmethod
    def _issue_text(issue) -> str:
        labels = {
            "LEADING_COVERAGE_GAP": "missing at the start",
            "LEADING_SOURCE_COVERAGE_GAP": "source begins after the requested start",
            "TRAILING_COVERAGE_GAP": "missing at the end",
            "TRAILING_SOURCE_COVERAGE_GAP": "source ends before the requested end",
            "MISSING_INTERNAL_INTERVAL": "internal gap",
        }
        return labels.get(issue.code, issue.code.replace("_", " ").lower())

    def _render_range_validation(self, report) -> None:
        strategy_interval = self.strategy_tf.currentData()
        intrabar_interval = self.intrabar_tf.currentData()
        strategy_error = False
        intrabar_error = False
        lines: list[str] = []

        for dataset in report.datasets:
            is_strategy = dataset.interval == strategy_interval
            role = "Strategy" if is_strategy else "Intrabar"
            has_error = dataset.status is DataQualityStatus.ERROR
            if is_strategy:
                strategy_error = strategy_error or has_error
            elif intrabar_interval and dataset.interval == intrabar_interval:
                intrabar_error = intrabar_error or has_error

            if not has_error:
                lines.append(f"✓ {role} {dataset.interval} candles are complete for the selected range.")
                continue

            lines.append(f"✗ {role} {dataset.interval} candles are incomplete.")
            for issue in dataset.issues:
                if issue.code not in {
                    "LEADING_COVERAGE_GAP",
                    "LEADING_SOURCE_COVERAGE_GAP",
                    "TRAILING_COVERAGE_GAP",
                    "TRAILING_SOURCE_COVERAGE_GAP",
                    "MISSING_INTERNAL_INTERVAL",
                }:
                    continue
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

        blocked = self._quality_has_errors(report)
        if blocked and intrabar_error and not strategy_error:
            lines.append(
                "Only the selected intrabar / exit-detail data is blocking this run. "
                "Use Strategy Bars Only, choose another intrabar timeframe above, or repair/download the missing data."
            )
        elif blocked:
            lines.append(
                "Required strategy data is incomplete. Repair/download the missing candles, use a safe boundary adjustment when offered, or choose another range."
            )
        else:
            lines.append("All required execution candles are validated. Optional research gaps do not block the backtest.")

        self._set_readiness(
            "NOT READY" if blocked else "READY TO RUN",
            "\n".join(lines),
            state="blocked" if blocked else "ready",
        )
        self.use_strategy_bars_button.setEnabled(
            bool(intrabar_interval and intrabar_error and not strategy_error)
        )

        self._validated_safe_dates = self._safe_dates_from_report(report)
        self.use_validated_range_button.setEnabled(self._validated_safe_dates is not None)
        if self._validated_safe_dates:
            start_date, end_date = self._validated_safe_dates
            self.use_validated_range_button.setToolTip(
                f"Set the date-only request to safe validated boundaries: "
                f"{start_date.isoformat()} → {end_date.isoformat()}."
            )
        else:
            self.use_validated_range_button.setToolTip(
                "Internal gaps cannot be fixed safely by changing only the outer date boundaries."
            )

    # ------------------------------------------------------------------
    # Corrective actions
    # ------------------------------------------------------------------
    def _use_strategy_bars_only(self) -> None:
        index = self.intrabar_tf.findData(None)
        if index >= 0:
            self.intrabar_tf.setCurrentIndex(index)

    def _open_data_library(self) -> None:
        for index in range(self.pages.count()):
            page = self.pages.widget(index)
            if any(label.text() == "Data Library" for label in page.findChildren(QLabel)):
                self.pages.setCurrentIndex(index)
                return

    def start_run(self):
        if self._validation_thread is not None:
            self._validation_auto_run = True
            self.run_button.setEnabled(False)
            self._set_readiness(
                "CHECKING DATA…",
                "Validation is already running. The backtest will start automatically if the selected required data passes.",
            )
            return
        super().start_run()
