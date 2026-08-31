"""Focused Opportunity Scanner QWidget for the active GUI shell."""
from __future__ import annotations

from datetime import timezone
from dataclasses import replace
from decimal import Decimal
from threading import Event
import time

from PySide6.QtCore import QDateTime, QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
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

class _ValidationWorker(QObject):
    finished=Signal(object); failed=Signal(str); cancelled=Signal(); progress=Signal(object)
    def __init__(self,service,run_dirs,config_path,horizon,cancelled):
        super().__init__(); self.service,self.run_dirs=service,run_dirs
        self.config_path,self.horizon,self._cancelled=config_path,horizon,cancelled
    @Slot()
    def run(self):
        try:
            result=self.service.validate(self.run_dirs,self.config_path,self.horizon,self._cancelled,self.progress.emit)
            if self._cancelled(): self.cancelled.emit()
            else: self.finished.emit(result)
        except Exception as exc:
            if self._cancelled(): self.cancelled.emit()
            else: self.failed.emit(str(exc))


class OpportunityScannerWorkspace(QWidget):
    """Configuration and immutable publication viewer; contains no pipeline logic."""
    def __init__(self, service, parent=None):
        super().__init__(parent); self.service = service; self._thread = None; self._cancel = Event()
        self._validation_thread=None; self._validation_worker=None; self._validation_cancel=Event(); self._validation_scan_run_dirs=()
        self._validation_started=0.0; self._validation_progress_state={}
        defaults, candle_defaults, final_defaults = DiscoveryConfig(), SelectiveCandleAcquisitionConfig(), FinalCandidateBoundaryConfig()
        root = QVBoxLayout(self)
        data_hub_path = getattr(service, "binance_data_hub_project_path", None)
        self.data_hub_status = QLabel(
            f"Binance Data Hub:\n{data_hub_path}" if data_hub_path else
            "Binance Data Hub:\nNot configured — downloads unavailable"
        )
        self.data_hub_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.data_hub_status)
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
        validation=QWidget(); validation_layout=QVBoxLayout(validation); validation_form=QFormLayout()
        config_row=QWidget(); config_layout=QHBoxLayout(config_row); config_layout.setContentsMargins(0,0,0,0)
        self.validation_config=QLineEdit(); self.validation_browse=QPushButton("Browse"); config_layout.addWidget(self.validation_config,1); config_layout.addWidget(self.validation_browse)
        self.validation_horizon=QLineEdit("24h"); validation_form.addRow("Strategy config",config_row); validation_form.addRow("Entry evaluation horizon",self.validation_horizon); validation_layout.addLayout(validation_form)
        validation_actions=QHBoxLayout(); self.validate_button=QPushButton("Validate Final Candidates"); self.validation_cancel=QPushButton("Cancel"); self.validate_button.setEnabled(False); self.validation_cancel.setEnabled(False); validation_actions.addWidget(self.validate_button); validation_actions.addWidget(self.validation_cancel); validation_actions.addStretch(1); validation_layout.addLayout(validation_actions)
        self.validation_progress=QLabel("Strategy validation: —\nCurrent symbol: —\nStage: —\nElapsed: 00:00:00\nETA: calculating…"); validation_layout.addWidget(self.validation_progress)
        self.validation_views=QTabWidget(); self.validation_outcomes=QTableWidget(); self.validation_summary=QTableWidget(); self.validation_rank=QTableWidget(); self.validation_views.addTab(self.validation_outcomes,"Candidate Outcomes"); self.validation_views.addTab(self.validation_summary,"Summary"); self.validation_views.addTab(self.validation_rank,"Rank Performance"); validation_layout.addWidget(self.validation_views,1)
        self.tabs.addTab(validation,"Strategy Validation")
        self._timer=QTimer(self); self._timer.timeout.connect(self._update_elapsed)
        self._validation_timer=QTimer(self); self._validation_timer.timeout.connect(self._update_validation_elapsed)
        self.mode.currentIndexChanged.connect(self._mode_changed); self.execution.currentIndexChanged.connect(self._mode_changed)
        for widget in (self.range_start,self.range_end): widget.dateTimeChanged.connect(self._update_planned)
        self.cadence.currentIndexChanged.connect(self._update_planned)
        self.run_button.clicked.connect(self.start_scan); self.cancel_button.clicked.connect(self.cancel_scan)
        self.validation_browse.clicked.connect(self._browse_validation_config)
        self.validate_button.clicked.connect(self.start_validation); self.validation_cancel.clicked.connect(self.cancel_validation)
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
        self._sync_controls()

    def _sync_controls(self):
        scanning=self._thread is not None; validating=self._validation_thread is not None
        historical=self.mode.currentData()=="HISTORICAL"
        self.run_button.setEnabled(not scanning and not validating)
        self.validate_button.setEnabled(historical and bool(self._validation_scan_run_dirs) and not scanning and not validating)
        self.validation_cancel.setEnabled(validating)

    def _browse_validation_config(self):
        path,_=QFileDialog.getOpenFileName(self,"Select v3 ResearchRunConfig",self.validation_config.text(),"JSON (*.json)")
        if path: self.validation_config.setText(path)

    def render_validation(self, result):
        """Render already-computed validation facts; no strategy logic lives in Qt."""
        self._fill(self.validation_outcomes,result.outcomes,("decision_timestamp","final_rank","symbol","population","valid_entry","side","result","completed_trade_count","wins","losses","neutrals","average_r"))
        self._fill(self.validation_summary,result.summary,("population","candidate_observations","candidate_to_entry_conversion","unique_trade_count","unique_wins","unique_losses","unique_neutrals","resolved_unique_trade_win_rate","average_r_per_unique_trade"))
        rank=result.by_rank.copy(); top=result.top_k.copy()
        if not top.empty: top["final_rank"]="Top "+top.top_k.astype(str)
        self._fill(self.validation_rank,__import__("pandas").concat([rank,top],ignore_index=True),("population","final_rank","candidate_observations","candidate_to_entry_conversion","unique_trade_count","unique_wins","unique_losses","unique_neutrals","resolved_unique_trade_win_rate","average_r_per_unique_trade"))

    def start_validation(self):
        if self._validation_thread is not None or self._thread is not None: return
        validation_service=getattr(self.service,"validation_service",None)
        if validation_service is None: self.validation_progress.setText("Validation service is not configured."); return
        try:
            from crypto_strategy_lab.historical_strategy_validation import load_validation_config
            path=self.validation_config.text().strip(); load_validation_config(path)
            horizon=__import__("pandas").Timedelta(self.validation_horizon.text().strip())
            if horizon <= __import__("pandas").Timedelta(0): raise ValueError("Entry evaluation horizon must be positive")
            if not self._validation_scan_run_dirs: raise ValueError("A complete Historical scan is required")
        except Exception as exc: self.validation_progress.setText(f"Cannot start validation: {exc}"); return
        self._validation_cancel.clear(); self._validation_started=time.monotonic(); self._validation_progress_state={"symbol_index":0,"symbol_total":0,"symbol":"—","stage":"Starting","eta":None}
        thread=QThread(self); worker=_ValidationWorker(validation_service,self._validation_scan_run_dirs,path,str(horizon),self._validation_cancel.is_set); worker.moveToThread(thread)
        thread.started.connect(worker.run); worker.progress.connect(self._validation_progress_event); worker.finished.connect(self._validation_completed); worker.failed.connect(self._validation_failed); worker.cancelled.connect(self._validation_cancelled)
        worker.finished.connect(thread.quit); worker.failed.connect(thread.quit); worker.cancelled.connect(thread.quit); thread.finished.connect(self._validation_thread_finished)
        self._validation_thread,self._validation_worker=thread,worker; self._sync_controls(); self._validation_timer.start(250); thread.start()

    def cancel_validation(self):
        self._validation_cancel.set(); self.validation_progress.setText("Cancelling after current native symbol run…")

    def _validation_progress_event(self,event):
        self._validation_progress_state.update(event); self._update_validation_elapsed()
    def _update_validation_elapsed(self):
        if self._validation_thread is None: return
        event=self._validation_progress_state; index,total=event.get("symbol_index",0),event.get("symbol_total",0); eta=event.get("eta")
        self.validation_progress.setText(f"Strategy validation: {index} / {total} symbols\nCurrent symbol: {event.get('symbol','—')}\nNative stage: {event.get('native_stage',event.get('stage','—'))}\nElapsed: {self._duration(time.monotonic()-self._validation_started)}\nETA: {'calculating…' if eta is None else self._duration(eta)}")
    def _validation_completed(self,result): self.render_validation(result); self.validation_progress.setText(f"Completed — {result.run_dir}")
    def _validation_failed(self,error): self.validation_progress.setText(f"Validation failed: {error}")
    def _validation_cancelled(self): self.validation_progress.setText("Validation cancelled; completed native runs were preserved.")
    def _validation_thread_finished(self):
        self._validation_timer.stop()
        self._validation_thread.deleteLater(); self._validation_thread=None; self._validation_worker=None
        self._sync_controls()

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
        if self._thread is not None or self._validation_thread is not None: return
        self._validation_scan_run_dirs=(); self.validate_button.setEnabled(False)
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
        self._sync_controls()

    def cancel_scan(self): self._cancel.set(); self.status.setText("Cancelling…")
    def _completed(self,result):
        if self._cancel.is_set():
            self.status.setText("Cancelled")
            return
        if hasattr(result,"completed"):
            if result.last: self.render(result.last)
            self._validation_scan_run_dirs=tuple(item.run_dir for item in result.completed)
            count=len(result.completed); average=result.elapsed_seconds/count
            self.status.setText(f"Completed {count} / {len(result.decision_points)} historical scans")
            self.progress_text.setText(f"Elapsed: {self._duration(result.elapsed_seconds)}\nAverage: {average:.1f}s / scan\nFirst: {result.decision_points[0].isoformat()}\nLast: {result.decision_points[-1].isoformat()}")
        else:
            self.render(result); self.status.setText("Completed")
            if result.manifest.get("opportunity_scan",{}).get("discovery_mode")=="HISTORICAL": self._validation_scan_run_dirs=(result.run_dir,)
        self._sync_controls()
    def _cancelled(self):
        self._validation_scan_run_dirs=(); self.validate_button.setEnabled(False); self.status.setText(f"Cancelled — {self._range_completed} / {self._range_total} completed" if self._range_total else "Cancelled")
    def _failed(self,error):
        self._validation_scan_run_dirs=(); self.validate_button.setEnabled(False)
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
        self._sync_controls()

    def shutdown(self):
        """Cooperatively stop backend work before Qt destroys its QThread."""
        thread = self._thread
        if thread is not None and thread.isRunning():
            self._cancel.set()
            self.status.setText("Cancelling…")
            thread.quit()
            thread.wait()
        validation_thread=self._validation_thread
        if validation_thread is not None and validation_thread.isRunning():
            self._validation_cancel.set(); validation_thread.quit(); validation_thread.wait()

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
