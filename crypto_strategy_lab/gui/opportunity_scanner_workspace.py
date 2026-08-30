"""Focused Opportunity Scanner QWidget for the active GUI shell."""
from __future__ import annotations

from datetime import timezone
from dataclasses import replace
from decimal import Decimal
from threading import Event
import time

from PySide6.QtCore import QDateTime, QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from crypto_strategy_lab.data.binance.selective_acquisition import SelectiveCandleAcquisitionConfig
from crypto_strategy_lab.data.binance.universe import DiscoveryConfig
from crypto_strategy_lab.final_candidates import FinalCandidateBoundaryConfig
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.opportunity_scoring import SCORING_MODELS
from .opportunity_scanner_controller import (HISTORICAL_REPLAY_CADENCES,
    HistoricalRangeRunner, HistoricalReplayFailure, OpportunityScanCancelled,
    historical_decision_points, build_request)

NATIVE_INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d")
RICH_FEATURES = ("funding_context", "basis_context", "futures_positioning",
                 "taker_flow_context", "trade_flow_context", "order_book_context")


class _ScanWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(object)
    def __init__(self, service, request, cancelled):
        super().__init__(); self.service = service; self.request = request; self._is_cancelled = cancelled
    @Slot()
    def run(self):
        try:
            method=getattr(self.service,"run_with_progress",None)
            self.finished.emit(method(self.request,self._is_cancelled,self.progress.emit) if method else self.service.run(self.request,self._is_cancelled))
        except OpportunityScanCancelled: self.cancelled.emit()
        except Exception as exc:
            if self._is_cancelled(): self.cancelled.emit()
            else: self.failed.emit(str(exc))


class _RangeWorker(QObject):
    finished = Signal(object); failed = Signal(object); cancelled = Signal(); progress = Signal(object)
    def __init__(self, runner, points, request_factory, cancelled):
        super().__init__(); self.runner=runner; self.points=points; self.request_factory=request_factory; self._cancelled=cancelled
    @Slot()
    def run(self):
        try: self.finished.emit(self.runner.run(self.points,self.request_factory,self._cancelled,self.progress.emit))
        except OpportunityScanCancelled: self.cancelled.emit()
        except HistoricalReplayFailure as exc: self.failed.emit(exc)
        except Exception as exc: self.failed.emit(exc)


class OpportunityScannerWorkspace(QWidget):
    """Configuration and immutable publication viewer; contains no pipeline logic."""
    def __init__(self, service, parent=None):
        super().__init__(parent); self.service = service; self._thread = None; self._cancel = Event()
        defaults, candle_defaults, final_defaults = DiscoveryConfig(), SelectiveCandleAcquisitionConfig(), FinalCandidateBoundaryConfig()
        root = QVBoxLayout(self)
        config = QGroupBox("Scan configuration"); form = QFormLayout(config); self._form = form
        self.market = QComboBox(); self.market.addItem("Binance USD-M Futures", "futures_um")
        self.mode = QComboBox(); self.mode.addItem("Live", "LIVE"); self.mode.addItem("Historical", "HISTORICAL")
        self.decision_time = QDateTimeEdit(QDateTime.currentDateTimeUtc()); self.decision_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'"); self.decision_time.setTimeSpec(Qt.UTC); self.decision_time.setEnabled(False)
        self.execution = QComboBox(); self.execution.addItem("Single timestamp", "SINGLE"); self.execution.addItem("Date/time range", "RANGE")
        self.range_start = QDateTimeEdit(QDateTime.currentDateTimeUtc()); self.range_end = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        for edit in (self.range_start,self.range_end): edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'"); edit.setTimeSpec(Qt.UTC)
        self.cadence = QComboBox(); self.cadence.addItems(tuple(HISTORICAL_REPLAY_CADENCES))
        self.planned = QLabel("Planned scans: 1")
        self.listing_age = QSpinBox(); self.listing_age.setRange(0, 10000); self.listing_age.setValue(defaults.minimum_listing_age.days)
        self.volume = QDoubleSpinBox(); self.volume.setRange(0, 1e15); self.volume.setDecimals(2); self.volume.setValue(float(defaults.minimum_quote_volume))
        self.spread = QDoubleSpinBox(); self.spread.setRange(0, 100); self.spread.setDecimals(6); self.spread.setValue(float(defaults.maximum_spread_percent))
        self.preliminary_size = QSpinBox(); self.preliminary_size.setRange(1, 10000); self.preliminary_size.setValue(candle_defaults.shortlist_size)
        self.final_size = QSpinBox(); self.final_size.setRange(1, 10000); self.final_size.setValue(final_defaults.max_candidates or 10)
        self.model = QComboBox(); self.model.addItem("Discovery Order / No Opportunity Model", None)
        for definition in SCORING_MODELS: self.model.addItem(f"{definition.name} v{definition.version}", definition)
        self.timeframe = QComboBox(); self.timeframe.addItems(NATIVE_INTERVALS); self.timeframe.setCurrentText(candle_defaults.strategy_interval)
        for label, widget in (("Market",self.market),("Scan mode",self.mode),("Historical execution",self.execution),("Decision timestamp",self.decision_time),("Start decision timestamp UTC",self.range_start),("End decision timestamp UTC",self.range_end),("Replay cadence",self.cadence),("",self.planned),("Minimum listing age (days, Live only)",self.listing_age),("Minimum quote volume",self.volume),("Maximum spread (%, Live only)",self.spread),("Preliminary shortlist size",self.preliminary_size),("Final candidate size",self.final_size),("Scoring model",self.model),("Strategy timeframe",self.timeframe)): form.addRow(label,widget)
        rich = QWidget(); rich_layout = QHBoxLayout(rich); rich_layout.setContentsMargins(0,0,0,0); self.feature_checks = {}
        available = set(production_feature_registry().names())
        for name in RICH_FEATURES:
            if name in available:
                check=QCheckBox(name.replace("_"," ").title()); self.feature_checks[name]=check; rich_layout.addWidget(check)
        form.addRow("Rich features", rich); root.addWidget(config)
        actions=QHBoxLayout(); self.run_button=QPushButton("Run Scan"); self.cancel_button=QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.status=QLabel("Ready")
        actions.addWidget(self.run_button); actions.addWidget(self.cancel_button); actions.addWidget(self.status,1); root.addLayout(actions)
        self.range_progress=QProgressBar(); self.range_progress.setTextVisible(True); self.range_progress.hide(); root.addWidget(self.range_progress)
        self.progress_text=QLabel("Stage: —\nElapsed: 00:00:00"); root.addWidget(self.progress_text)
        self.summary=QLabel("No completed scan loaded."); self.summary.setWordWrap(True); root.addWidget(self.summary)
        self.tabs=QTabWidget(); self.preliminary_table=QTableWidget(); self.final_table=QTableWidget(); self.readiness_table=QTableWidget()
        self.tabs.addTab(self.preliminary_table,"Preliminary Candidates"); self.tabs.addTab(self.final_table,"Final Candidates"); self.tabs.addTab(self.readiness_table,"Data Readiness"); root.addWidget(self.tabs,1)
        self._timer=QTimer(self); self._timer.timeout.connect(self._update_elapsed)
        self.mode.currentIndexChanged.connect(self._mode_changed); self.execution.currentIndexChanged.connect(self._mode_changed)
        for widget in (self.range_start,self.range_end): widget.dateTimeChanged.connect(self._update_planned)
        self.cadence.currentIndexChanged.connect(self._update_planned)
        self.run_button.clicked.connect(self.start_scan); self.cancel_button.clicked.connect(self.cancel_scan)
        self._mode_changed()

    def _mode_changed(self):
        historical = self.mode.currentData() == "HISTORICAL"
        ranged = historical and self.execution.currentData() == "RANGE"
        self._set_row_visible(self.execution, historical)
        self._set_row_visible(self.decision_time, historical and not ranged)
        for widget in (self.range_start,self.range_end,self.cadence,self.planned):
            self._set_row_visible(widget, ranged)
        self.decision_time.setEnabled(historical and not ranged)
        self.listing_age.setEnabled(not historical)
        self.spread.setEnabled(not historical)
        self.listing_age.setToolTip("Live discovery only; Task 2 has no historical listing-date source.")
        self.spread.setToolTip("Live discovery only; Task 2 uses completed daily candles, not live books.")
        if not ranged:
            self.range_progress.hide()
            self.range_progress.setValue(0)
            self.progress_text.setText("Stage: —\nElapsed: 00:00:00")
        self._update_planned()

    def _set_row_visible(self, field, visible):
        """Hide both parts of a form row on Qt versions without setRowVisible."""
        if hasattr(self._form, "setRowVisible"):
            self._form.setRowVisible(field, visible)
            return
        field.setVisible(visible)
        label = self._form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    @staticmethod
    def _utc(edit):
        """Return exactly the second displayed by the UTC editor."""
        value = edit.dateTime().toUTC().toPython()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.replace(microsecond=0)

    def decision_points(self):
        return historical_decision_points(self._utc(self.range_start),self._utc(self.range_end),HISTORICAL_REPLAY_CADENCES[self.cadence.currentText()])

    def _update_planned(self, *_):
        try: self.planned.setText(f"Planned scans: {len(self.decision_points())}")
        except ValueError as exc: self.planned.setText(str(exc))

    def request(self):
        decision = self._utc(self.decision_time) if self.mode.currentData()=="HISTORICAL" else None
        return build_request(mode=self.mode.currentData(), decision_time=decision,
            minimum_listing_age_days=self.listing_age.value(), minimum_quote_volume=Decimal(str(self.volume.value())),
            maximum_spread_percent=Decimal(str(self.spread.value())), preliminary_size=self.preliminary_size.value(),
            final_size=self.final_size.value(), strategy_interval=self.timeframe.currentText(), model=self.model.currentData(),
            enabled_features=tuple(name for name,check in self.feature_checks.items() if check.isChecked()))

    def start_scan(self):
        if self._thread is not None: return
        ranged=self.mode.currentData()=="HISTORICAL" and self.execution.currentData()=="RANGE"
        try:
            points=self.decision_points() if ranged else ()
            request=self.request()
            if ranged: request=replace(request,decision_time=points[0])
        except Exception as exc: self.status.setText(str(exc)); return
        self._cancel.clear(); self._started=time.monotonic(); self._range_total=len(points); self._range_completed=0
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True); self.status.setText("Scanning…"); self._timer.start(250)
        self.range_progress.setVisible(ranged)
        if ranged: self.range_progress.setRange(0,len(points)); self.range_progress.setValue(0); self.range_progress.setFormat(f"0 / {len(points)}")
        request_factory=lambda decision, template=request: replace(template,decision_time=decision)
        thread=QThread(self); worker=(_RangeWorker(HistoricalRangeRunner(self.service,monotonic=time.monotonic),points,request_factory,self._cancel.is_set) if ranged else _ScanWorker(self.service,request,self._cancel.is_set)); worker.moveToThread(thread); thread.started.connect(worker.run); worker.progress.connect(self._progress); worker.finished.connect(self._completed); worker.failed.connect(self._failed); worker.cancelled.connect(self._cancelled); worker.finished.connect(thread.quit); worker.failed.connect(thread.quit); worker.cancelled.connect(thread.quit); thread.finished.connect(self._thread_finished)
        self._thread=thread; self._worker=worker; thread.start()

    def cancel_scan(self): self._cancel.set(); self.status.setText("Cancelling…")
    def _completed(self,result):
        if self._cancel.is_set():
            self.status.setText("Cancelled")
            return
        if hasattr(result,"completed"):
            if result.last: self.render(result.last)
            count=len(result.completed); average=result.elapsed_seconds/count
            self.status.setText(f"Completed {count} / {len(result.decision_points)} historical scans")
            self.progress_text.setText(f"Elapsed: {self._duration(result.elapsed_seconds)}\nAverage: {average:.1f}s / scan\nFirst: {result.decision_points[0].isoformat()}\nLast: {result.decision_points[-1].isoformat()}")
        else: self.render(result); self.status.setText("Completed")
    def _cancelled(self): self.status.setText(f"Cancelled — {self._range_completed} / {self._range_total} completed" if self._range_total else "Cancelled")
    def _failed(self,error):
        if isinstance(error,HistoricalReplayFailure): self.status.setText(f"Failed at {error.decision_time.isoformat()} — {len(error.completed)} / {self._range_total} completed: {error}")
        else: self.status.setText(f"Failed: {error}")
    def _progress(self,event):
        self._range_completed=event.completed_scans
        if event.total_scans>1:
            self.range_progress.setValue(event.completed_scans); self.range_progress.setFormat(f"{event.completed_scans} / {event.total_scans}")
        replay=f"Historical replay: {event.completed_scans} / {event.total_scans}\n" if event.total_scans>1 else ""
        decision=f"\nDecision: {event.decision_timestamp:%Y-%m-%d %H:%M:%S} UTC" if event.decision_timestamp else ""
        average="" if event.average_scan_seconds is None else f"\nAverage / completed scan: {event.average_scan_seconds:.1f}s"
        eta="" if event.total_scans==1 else ("\nETA: calculating…" if event.eta_seconds is None else f"\nETA (estimate): {self._duration(event.eta_seconds)}")
        self.progress_text.setText(f"{replay}Stage: {event.stage_index}/{event.stage_count} — {event.message}{decision}\nElapsed: {self._duration(event.elapsed_seconds)}{average}{eta}")
    def _update_elapsed(self):
        if self._thread is not None:
            lines=self.progress_text.text().splitlines(); elapsed=f"Elapsed: {self._duration(time.monotonic()-self._started)}"
            self.progress_text.setText("\n".join(elapsed if line.startswith("Elapsed:") else line for line in lines))
    @staticmethod
    def _duration(seconds):
        seconds=max(0,int(seconds)); return f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}"
    def _thread_finished(self):
        self._timer.stop()
        self._thread.deleteLater(); self._thread=None; self._worker=None; self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)

    def shutdown(self):
        """Cooperatively stop backend work before Qt destroys its QThread."""
        thread = self._thread
        if thread is not None and thread.isRunning():
            self._cancel.set()
            self.status.setText("Cancelling…")
            thread.quit()
            thread.wait()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    @staticmethod
    def _fill(table, frame, columns):
        shown=[c for c in columns if c in frame.columns]; table.clear(); table.setColumnCount(len(shown)); table.setHorizontalHeaderLabels([c.replace("_"," ").title() for c in shown]); table.setRowCount(len(frame))
        for row,(_,record) in enumerate(frame.iterrows()):
            for column,name in enumerate(shown):
                value=record[name]; table.setItem(row,column,QTableWidgetItem("" if value is None or str(value)=="nan" else str(value)))
        table.resizeColumnsToContents()

    def render(self,result):
        s=result.summary; m=result.manifest; scan=m.get("opportunity_scan",{}); hashes=m.get("hashes",{})
        self.summary.setText(" | ".join((f"Run ID: {m.get('run_id','—')}",f"Scan: {s.get('scan_timestamp','—')}",f"Decision: {s.get('decision_timestamp','—')}",f"Mode: {s.get('discovery_mode','—')}",f"Eligible: {s.get('discovery_eligible_count',0)}",f"Rejected: {s.get('discovery_rejected_count',0)}",f"Preliminary: {s.get('preliminary_candidate_count',0)}",f"Final: {s.get('final_candidate_count',0)}",f"Commit: {m.get('code_commit','—')}",f"Semantic: {hashes.get('semantic_input_hash','—')}",f"Sources: {scan.get('source_identity_digest','—')}",f"Folder: {result.run_dir}")))
        self._fill(self.preliminary_table,result.preliminary,("discovery_rank","symbol","range_percent","absolute_price_change_percent","quote_volume","spread_percent","model_rank","score","acquisition_state","quality_status","detail"))
        self._fill(self.final_table,result.final,("final_rank","symbol","discovery_rank","opportunity_model_name","opportunity_model_version","opportunity_model_rank","opportunity_score","strategy_interval","quality_status","acquisition_state"))
        self._fill(self.readiness_table,result.readiness,("symbol","feature_name","dataset","feature_readiness","acquisition_state","requiredness_for_feature","quality_status","detail"))
