"""Focused Opportunity Scanner QWidget for the active GUI shell."""
from __future__ import annotations

from datetime import timedelta, timezone
from decimal import Decimal
from threading import Event

from PySide6.QtCore import QDateTime, QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from crypto_strategy_lab.data.binance.selective_acquisition import SelectiveCandleAcquisitionConfig
from crypto_strategy_lab.data.binance.universe import DiscoveryConfig
from crypto_strategy_lab.final_candidates import FinalCandidateBoundaryConfig
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.opportunity_scoring import SCORING_MODELS
from .opportunity_scanner_controller import (HistoricalRangeRunner,
    OpportunityScanCancelled, build_request, historical_decision_points)

NATIVE_INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d")
RICH_FEATURES = ("funding_context", "basis_context", "futures_positioning",
                 "taker_flow_context", "trade_flow_context", "order_book_context")


class _ScanWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    def __init__(self, service, request, cancelled):
        super().__init__(); self.service = service; self.request = request; self._is_cancelled = cancelled
    @Slot()
    def run(self):
        try: self.finished.emit(self.service.run(self.request, self._is_cancelled))
        except OpportunityScanCancelled: self.cancelled.emit()
        except Exception as exc:
            if self._is_cancelled(): self.cancelled.emit()
            else: self.failed.emit(str(exc))


class _RangeWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    progress = Signal(str, int, int, object, object)
    def __init__(self, service, request, points, cancelled):
        super().__init__(); self.runner=HistoricalRangeRunner(service); self.request=request; self.points=points; self._is_cancelled=cancelled
    @Slot()
    def run(self):
        try:
            result=self.runner.run(self.request,self.points,self._is_cancelled,self.progress.emit)
            self.finished.emit(result)
        except OpportunityScanCancelled as exc: self.cancelled.emit(str(exc))
        except Exception as exc: self.failed.emit(str(exc))


class OpportunityScannerWorkspace(QWidget):
    """Configuration and immutable publication viewer; contains no pipeline logic."""
    def __init__(self, service, parent=None):
        super().__init__(parent); self.service = service; self._thread = None; self._cancel = Event()
        defaults, candle_defaults, final_defaults = DiscoveryConfig(), SelectiveCandleAcquisitionConfig(), FinalCandidateBoundaryConfig()
        root = QVBoxLayout(self)
        config = QGroupBox("Scan configuration"); form = QFormLayout(config); self.form=form
        self.market = QComboBox(); self.market.addItem("Binance USD-M Futures", "futures_um")
        self.mode = QComboBox(); self.mode.addItem("Live", "LIVE"); self.mode.addItem("Historical", "HISTORICAL")
        self.historical_execution=QComboBox(); self.historical_execution.addItem("Single", "SINGLE"); self.historical_execution.addItem("Range", "RANGE")
        self.decision_time = QDateTimeEdit(QDateTime.currentDateTimeUtc()); self.decision_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'"); self.decision_time.setTimeSpec(Qt.UTC); self.decision_time.setEnabled(False)
        self.range_start=QDateTimeEdit(QDateTime.currentDateTimeUtc()); self.range_end=QDateTimeEdit(QDateTime.currentDateTimeUtc().addDays(2))
        for editor in (self.range_start,self.range_end): editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'"); editor.setTimeSpec(Qt.UTC)
        self.replay_cadence=QSpinBox(); self.replay_cadence.setRange(1,1000000); self.replay_cadence.setValue(24); self.replay_cadence.setSuffix(" hours")
        self.planned_scans=QLabel("0")
        self.listing_age = QSpinBox(); self.listing_age.setRange(0, 10000); self.listing_age.setValue(defaults.minimum_listing_age.days)
        self.volume = QDoubleSpinBox(); self.volume.setRange(0, 1e15); self.volume.setDecimals(2); self.volume.setValue(float(defaults.minimum_quote_volume))
        self.spread = QDoubleSpinBox(); self.spread.setRange(0, 100); self.spread.setDecimals(6); self.spread.setValue(float(defaults.maximum_spread_percent))
        self.preliminary_size = QSpinBox(); self.preliminary_size.setRange(1, 10000); self.preliminary_size.setValue(candle_defaults.shortlist_size)
        self.final_size = QSpinBox(); self.final_size.setRange(1, 10000); self.final_size.setValue(final_defaults.max_candidates or 10)
        self.model = QComboBox(); self.model.addItem("Discovery Order / No Opportunity Model", None)
        for definition in SCORING_MODELS: self.model.addItem(f"{definition.name} v{definition.version}", definition)
        self.timeframe = QComboBox(); self.timeframe.addItems(NATIVE_INTERVALS); self.timeframe.setCurrentText(candle_defaults.strategy_interval)
        for label, widget in (("Market",self.market),("Scan mode",self.mode),("Historical execution",self.historical_execution),("Decision timestamp",self.decision_time),("Start decision timestamp",self.range_start),("End decision timestamp",self.range_end),("Replay cadence",self.replay_cadence),("Planned scans",self.planned_scans),("Minimum listing age (days, Live only)",self.listing_age),("Minimum quote volume",self.volume),("Maximum spread (%, Live only)",self.spread),("Preliminary shortlist size",self.preliminary_size),("Final candidate size",self.final_size),("Scoring model",self.model),("Strategy timeframe",self.timeframe)): form.addRow(label,widget)
        rich = QWidget(); rich_layout = QHBoxLayout(rich); rich_layout.setContentsMargins(0,0,0,0); self.feature_checks = {}
        available = set(production_feature_registry().names())
        for name in RICH_FEATURES:
            if name in available:
                check=QCheckBox(name.replace("_"," ").title()); self.feature_checks[name]=check; rich_layout.addWidget(check)
        form.addRow("Rich features", rich); root.addWidget(config)
        actions=QHBoxLayout(); self.run_button=QPushButton("Run Scan"); self.cancel_button=QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.status=QLabel("Ready")
        actions.addWidget(self.run_button); actions.addWidget(self.cancel_button); actions.addWidget(self.status,1); root.addLayout(actions)
        self.range_progress=QProgressBar(); self.range_progress.setTextVisible(True); self.range_progress.setFormat("%v / %m"); self.range_progress.hide(); root.addWidget(self.range_progress)
        self.eta=QLabel(""); self.eta.hide(); root.addWidget(self.eta)
        self.summary=QLabel("No completed scan loaded."); self.summary.setWordWrap(True); root.addWidget(self.summary)
        self.tabs=QTabWidget(); self.preliminary_table=QTableWidget(); self.final_table=QTableWidget(); self.readiness_table=QTableWidget()
        self.tabs.addTab(self.preliminary_table,"Preliminary Candidates"); self.tabs.addTab(self.final_table,"Final Candidates"); self.tabs.addTab(self.readiness_table,"Data Readiness"); root.addWidget(self.tabs,1)
        self.mode.currentIndexChanged.connect(self._mode_changed); self.historical_execution.currentIndexChanged.connect(self._mode_changed)
        self.range_start.dateTimeChanged.connect(self._update_planned); self.range_end.dateTimeChanged.connect(self._update_planned); self.replay_cadence.valueChanged.connect(self._update_planned)
        self.run_button.clicked.connect(self.start_scan); self.cancel_button.clicked.connect(self.cancel_scan)
        self._mode_changed()

    def _show_row(self, field, visible):
        """Hide the complete form row (including the QFormLayout-created label)."""
        if hasattr(self.form,"setRowVisible"): self.form.setRowVisible(field,visible)
        else:
            field.setVisible(visible); label=self.form.labelForField(field)
            if label is not None: label.setVisible(visible)

    def _mode_changed(self):
        historical = self.mode.currentData() == "HISTORICAL"
        ranged = historical and self.historical_execution.currentData()=="RANGE"
        self._show_row(self.historical_execution,historical)
        self._show_row(self.decision_time,historical and not ranged)
        for field in (self.range_start,self.range_end,self.replay_cadence,self.planned_scans): self._show_row(field,ranged)
        self.decision_time.setEnabled(historical)
        self.listing_age.setEnabled(not historical)
        self.spread.setEnabled(not historical)
        self.listing_age.setToolTip("Live discovery only; Task 2 has no historical listing-date source.")
        self.spread.setToolTip("Live discovery only; Task 2 uses completed daily candles, not live books.")
        if not ranged:
            self.range_progress.hide(); self.range_progress.setValue(0); self.eta.hide(); self.eta.setText("")
        self._update_planned()

    @staticmethod
    def _utc(editor):
        # Convert the represented instant to UTC; do not merely relabel its zone.
        return editor.dateTime().toUTC().toPython().astimezone(timezone.utc).replace(microsecond=0)

    def _range_points(self):
        return historical_decision_points(self._utc(self.range_start),self._utc(self.range_end),timedelta(hours=self.replay_cadence.value()))

    def _update_planned(self):
        try: self.planned_scans.setText(str(len(self._range_points())))
        except ValueError: self.planned_scans.setText("0")

    def request(self):
        decision = self._utc(self.decision_time) if self.mode.currentData()=="HISTORICAL" else None
        return build_request(mode=self.mode.currentData(), decision_time=decision,
            minimum_listing_age_days=self.listing_age.value(), minimum_quote_volume=Decimal(str(self.volume.value())),
            maximum_spread_percent=Decimal(str(self.spread.value())), preliminary_size=self.preliminary_size.value(),
            final_size=self.final_size.value(), strategy_interval=self.timeframe.currentText(), model=self.model.currentData(),
            enabled_features=tuple(name for name,check in self.feature_checks.items() if check.isChecked()))

    def start_scan(self):
        if self._thread is not None: return
        try: request=self.request()
        except Exception as exc: self.status.setText(str(exc)); return
        self._cancel.clear(); self.run_button.setEnabled(False); self.cancel_button.setEnabled(True); self.status.setText("Scanning…")
        thread=QThread(self)
        ranged=request.mode=="HISTORICAL" and self.historical_execution.currentData()=="RANGE"
        if ranged:
            try: points=self._range_points()
            except ValueError as exc: self.status.setText(str(exc)); self.run_button.setEnabled(True); self.cancel_button.setEnabled(False); return
            worker=_RangeWorker(self.service,request,points,self._cancel.is_set); worker.progress.connect(self._range_updated)
            worker.cancelled.connect(self._range_cancelled)
            self.range_progress.setRange(0,len(points)); self.range_progress.setValue(0); self.range_progress.show(); self.eta.setText("ETA: calculating…"); self.eta.show()
        else: worker=_ScanWorker(self.service,request,self._cancel.is_set); worker.cancelled.connect(self._cancelled)
        worker.moveToThread(thread); thread.started.connect(worker.run); worker.finished.connect(self._completed); worker.failed.connect(self._failed); worker.finished.connect(thread.quit); worker.failed.connect(thread.quit); worker.cancelled.connect(thread.quit); thread.finished.connect(self._thread_finished)
        self._thread=thread; self._worker=worker; thread.start()

    def cancel_scan(self): self._cancel.set(); self.status.setText("Cancelling…")
    def _completed(self,result):
        if self._cancel.is_set():
            self.status.setText("Cancelled")
            return
        if isinstance(result, tuple):
            total=len(result)
            if result: self.render(result[-1])
            self.status.setText(f"Completed {total} / {total}")
            return
        self.render(result); self.status.setText("Completed")
    def _cancelled(self): self.status.setText("Cancelled")
    def _range_cancelled(self,message): self.status.setText(message.capitalize())
    def _range_updated(self,kind,completed,total,decision,event):
        if kind=="completed":
            self.range_progress.setValue(completed)
            remaining=total-completed
            self.eta.setText("ETA: complete" if not remaining else f"ETA: {remaining} scan{'s' if remaining != 1 else ''} remaining")
            self.status.setText(f"Completed {completed} / {total} — {decision.isoformat()}")
    def _failed(self,message): self.status.setText(f"Failed: {message}")
    def _thread_finished(self):
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
