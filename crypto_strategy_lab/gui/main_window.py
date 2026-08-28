"""Main PySide6 window for the backtester."""
from __future__ import annotations
import re, time, traceback
from dataclasses import replace
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QSettings, QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import *

from crypto_strategy_lab.loader import load_ohlcv_csv
from .config_logic import *
from .worker import BacktestWorker
from .portfolio_worker import PortfolioWorker
from .profile_editor import StrategyProfilesWidget
from .chatgpt_connection import ChatGPTIntegrationWidget
from .github_manager import GitHubIntegrationWidget
from crypto_strategy_lab.output_manager import planned_run_dir
from crypto_strategy_lab.support_resistance_analysis import build_sr_event_context_summary
from crypto_strategy_lab.report_workbooks import build_performance_breakdowns
from ..paths import DATA_DIR

REPORT_TARGETS = {
    "backtest": "backtest_report.xlsx", "indicators": "indicator_analysis.xlsx",
    "sr": "support_resistance_analysis.xlsx", "trades": "trade_list.csv",
    "charts": "charts", "output": ".",
}


def report_button_states(run_dir: Path) -> dict[str, bool]:
    """Return report availability independently of Qt for easy UI testing."""
    return {name: (run_dir / relative).exists() for name, relative in REPORT_TARGETS.items()}

class PolicyComboBox(QComboBox):
    """Show friendly policy labels while preserving stable config values."""
    def currentText(self):
        return str(self.currentData() or "")
    def setCurrentText(self,value):
        self.setCurrentIndex(max(0,self.findData(value)))

class MainWindow(QMainWindow):
    def __init__(self, startup_status=None):
        super().__init__(); self.setWindowTitle("Crypto Strategy Lab"); self.resize(1280, 860)
        startup_status = startup_status or (lambda _message: None)
        self.market_data_folder=DATA_DIR
        self.settings = QSettings("LongShortCrypto", "Backtester"); self.worker=None; self.thread=None; self.portfolio_worker=None; self.portfolio_thread=None; self.started=0; self.last_summary={}; self.output_dir=Path("output"); self.completed_run_dir=None; self._pending_ui_results=None; self._run_failed=False
        self._shared_adx_period = int(DEFAULT_GUI_CONFIG["adx_period"])
        self.tabs=QTabWidget(); self.setCentralWidget(self.tabs)
        self.tabs.setDocumentMode(True); self.tabs.setMovable(False)
        self.setStyleSheet("QGroupBox{font-weight:600;margin-top:10px;padding-top:10px} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px} QPushButton{padding:6px 12px} QTabBar::tab{padding:8px 14px} QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{min-height:24px}")
        startup_status("Building interface...")
        self._build_config(); self._build_profiles(); self._build_summary(); self._build_portfolio_tab(); self._build_log(); self.reset_defaults(); self._restore_settings()
        startup_status("Initializing integrations...")
        self.github_tab=GitHubIntegrationWidget(self, self._git_work_active)
        self.tabs.addTab(self.github_tab,"GitHub")
        self.chatgpt_tab=ChatGPTIntegrationWidget(self.settings, lambda: self.output_folder.text().strip() or "output", self)
        self.tabs.addTab(self.chatgpt_tab,"ChatGPT")

    def start_post_show_tasks(self):
        """Schedule optional integrations after the main window is visible."""
        QTimer.singleShot(0, self.chatgpt_tab.auto_start_connection)

    def _git_work_active(self):
        """Prevent source updates while calculation worker threads are running."""
        return bool((self.thread and self.thread.isRunning()) or
                    (self.portfolio_thread and self.portfolio_thread.isRunning()))

    def closeEvent(self,event):
        self.chatgpt_tab.shutdown()
        event.accept()
    def _build_profiles(self):
        self.profile_editor=StrategyProfilesWidget(); self.tabs.addTab(self.profile_editor,"Strategy Profiles")
        self.profile_editor.changed.connect(self.update_dynamic)
        self.profile_editor.list.currentRowChanged.connect(self.update_dynamic)
    def _line(self, text=""):
        w=QLineEdit(text); return w
    def _spin(self, v, mn=-1e12, mx=1e12, dec=6):
        s=QDoubleSpinBox(); s.setRange(mn,mx); s.setDecimals(dec); s.setValue(v); return s
    def _build_config(self):
        page=QWidget(); outer=QVBoxLayout(page); scroll=QScrollArea(); scroll.setWidgetResizable(True); inner=QWidget(); form=QVBoxLayout(inner); self.config_controls=[]
        toolbar=QHBoxLayout(); toolbar.setContentsMargins(0,0,0,4); self.setup_toolbar=toolbar
        self.new_run_btn=QPushButton("New Run"); self.save_btn=QPushButton("Save Config"); self.load_btn=QPushButton("Load Config")
        for button in (self.new_run_btn,self.save_btn,self.load_btn): toolbar.addWidget(button)
        toolbar.addStretch(1)
        self.run_btn=QPushButton("Run Backtest"); self.cancel_btn=QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        toolbar.addWidget(self.run_btn); toolbar.addWidget(self.cancel_btn); outer.addLayout(toolbar)
        self.new_run_btn.clicked.connect(self.new_run); self.save_btn.clicked.connect(self.save_config); self.load_btn.clicked.connect(self.load_config)
        self.run_btn.clicked.connect(self.run_backtest); self.cancel_btn.clicked.connect(lambda: self.worker and self.worker.cancel())
        def group(title): g=QGroupBox(title); l=QFormLayout(g); form.addWidget(g); return l
        data=group("Data")
        self.market_symbol=QComboBox(); self.market_symbol.setEditable(True); self.market_symbol.addItems(["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"]); self.market_symbol.setCurrentText("XRPUSDT"); data.addRow("Trading Pair",self.market_symbol)
        self.strategy_timeframe=QComboBox(); self.strategy_timeframe.addItems(["1m","5m","15m","30m","1h","4h"]); data.addRow("Strategy Timeframe",self.strategy_timeframe)
        self.input_csv=self._line(); self.input_csv.setReadOnly(True); b=QPushButton("Browse"); b.clicked.connect(self.browse_csv); row=QHBoxLayout(); row.addWidget(self.input_csv); row.addWidget(b); data.addRow("Strategy CSV", row)
        self.intrabar_timeframe=QComboBox(); self.intrabar_timeframe.addItems(["1m","5m","15m","30m","1h","4h"])
        self.intrabar_csv=self._line(); self.intrabar_csv.setReadOnly(True); bi=QPushButton("Browse"); bi.clicked.connect(self.browse_intrabar_csv); self.intrabar_csv_row=QHBoxLayout(); self.intrabar_csv_row.addWidget(self.intrabar_csv); self.intrabar_csv_row.addWidget(bi)
        self.input_csv.setPlaceholderText("No matching file in the shared Binance Data Hub folder")
        self.intrabar_csv.setPlaceholderText("No matching file in the shared Binance Data Hub folder")
        self.use_intrabar=QCheckBox("Use lower-timeframe data to resolve exits"); self.use_intrabar.setChecked(True)
        self.data_help=QLabel(); self.data_help.setWordWrap(True)
        data.addRow("",self.use_intrabar); data.addRow("Intrabar Timeframe",self.intrabar_timeframe); data.addRow("Intrabar CSV",self.intrabar_csv_row)
        self.shared_data_note=QLabel(f"Shared candle library: {self.market_data_folder}"); self.shared_data_note.setWordWrap(True); data.addRow("Shared Data",self.shared_data_note)
        self.strategy_timeframe.currentTextChanged.connect(self._timeframe_changed); self.intrabar_timeframe.currentTextChanged.connect(self.update_dynamic); self.use_intrabar.toggled.connect(self.update_dynamic)
        self.market_symbol.currentTextChanged.connect(self._sync_dataset_paths); self.strategy_timeframe.currentTextChanged.connect(self._sync_dataset_paths); self.intrabar_timeframe.currentTextChanged.connect(self._sync_dataset_paths)
        self.use_intrabar.toggled.connect(self._sync_dataset_paths)
        self.run_name=self._line(); self.run_name.setPlaceholderText("Optional run name prefix"); data.addRow("Run Name", self.run_name)
        self.output_folder=self._line(); self.output_folder.setReadOnly(True); bo=QPushButton("Browse"); bo.clicked.connect(self.browse_output); row=QHBoxLayout(); row.addWidget(self.output_folder); row.addWidget(bo); data.addRow("Output Folder", row)
        self.planned_output=QLabel("Output run folder: not calculated yet"); self.planned_output.setWordWrap(True); data.addRow("Next Run Folder", self.planned_output)
        self.dataset_info=QLabel("No CSV loaded."); data.addRow("Dataset Information", self.dataset_info); val=QPushButton("Validate Data"); val.clicked.connect(self.validate_data); data.addRow(val)
        strat=group("Core Strategy")
        self.entry_mode=QComboBox(); self.entry_mode.addItem("Wait until current trade closes","WAIT_UNTIL_CLOSED"); self.entry_mode.addItem("Check every N candles","EVERY_N_CANDLES"); self.entry_interval=QSpinBox(); self.entry_interval.setRange(1,999999); self.max_pairs=QSpinBox(); self.max_pairs.setRange(1,999999); self.tie=QComboBox(); self.tie.addItem("Conservative (stop first)","PESSIMISTIC"); self.tie.addItem("Optimistic (target first)","OPTIMISTIC")
        for lab,w in [("Entry Mode",self.entry_mode),("Entry Interval",self.entry_interval),("Maximum Active Pairs",self.max_pairs),("Tie Policy",self.tie)]: strat.addRow(lab,w)
        self.entry_mode.currentIndexChanged.connect(lambda:self.entry_interval.setEnabled(self.entry_mode.currentData()=="EVERY_N_CANDLES"))
        self.entry_mode.currentTextChanged.connect(self.update_dynamic)
        self.enable_support_resistance_analysis=QCheckBox("Enable Support/Resistance Analysis"); self.sr_pivot_left=QSpinBox(); self.sr_pivot_left.setRange(1,1000); self.sr_pivot_left.setValue(5); self.sr_pivot_right=QSpinBox(); self.sr_pivot_right.setRange(1,1000); self.sr_pivot_right.setValue(5); self.sr_lookback_bars=QSpinBox(); self.sr_lookback_bars.setRange(10,10000); self.sr_lookback_bars.setValue(200); self.sr_zone_width_atr=self._spin(0.5,0.0,10.0,3); self.sr_near_distance_atr=self._spin(0.75,0.0,10.0,3); self.enable_sr_hold_confirmation=QCheckBox("Enable"); self.sr_hold_confirmation_bars=QSpinBox(); self.sr_hold_confirmation_bars.setRange(1,100); self.sr_hold_confirmation_bars.setValue(3); self.sr_hold_confirmation_atr=self._spin(0.25,0.0,10.0,3); self.sr_break_tolerance_atr=self._spin(0.25,0.0,10.0,3); self.sr_break_basis=QComboBox(); self.sr_break_basis.addItems(["CLOSE","WICK"]); self.sr_filter_mode=PolicyComboBox()
        self.sr_filter_mode.addItem("Analysis Only", "ANALYSIS_ONLY")
        self.sr_filter_mode.addItem("Apply Entry Rules", "APPLY_ENTRY_RULES")
        self.sr_long_avoid_near_resistance=QCheckBox("Avoid buying near resistance")
        self.sr_long_require_near_support=QCheckBox("Require price near support")
        self.sr_long_block_broken_support=QCheckBox("Block if support structure is broken")
        self.sr_long_min_room_to_resistance_atr=self._spin(0.0,0.0,1000.0,2)
        self.sr_short_avoid_near_support=QCheckBox("Avoid selling near support")
        self.sr_short_require_near_resistance=QCheckBox("Require price near resistance")
        self.sr_short_block_broken_resistance=QCheckBox("Block if resistance structure is broken")
        self.sr_short_min_room_to_support_atr=self._spin(0.0,0.0,1000.0,2)
        self.enable_di_direction_selection=QCheckBox("Enable DI direction selection"); self.enable_di_direction_selection.setChecked(True)
        self.enable_di_pressure_analysis=QCheckBox("Analyze DI expansion / contraction"); self.enable_di_pressure_analysis.setChecked(True)
        self.di_pressure_lookback=QSpinBox(); self.di_pressure_lookback.setRange(1,100); self.di_pressure_lookback.setValue(3)
        self.enable_mean_reversion_analysis=QCheckBox("Analyze mean reversion"); self.enable_mean_reversion_analysis.setChecked(True)
        self.mean_reversion_period=QSpinBox(); self.mean_reversion_period.setRange(2,1000); self.mean_reversion_period.setValue(20)
        sched=group("Entry Schedule")
        self.enable_daily_schedule=QCheckBox("Enable Daily Scheduled Entry")
        self.daily_entry_time=self._line("00:00")
        self.daily_entry_timezone=self._line("UTC")
        self.daily_entry_missed_policy=QComboBox(); self.daily_entry_missed_policy.addItems(["SKIP_DAY","NEXT_AVAILABLE_CANDLE"])
        self.next_entry_summary=QLabel("Next eligible entry time: 00:00 UTC when enabled"); self.next_entry_summary.setWordWrap(True)
        help_text=QLabel("When enabled, the strategy attempts an entry once per day at the selected time.\n\nIf a trade is open at that time and SKIP_DAY is selected, no entry is opened later that day."); help_text.setWordWrap(True)
        for lab,w in [("",self.enable_daily_schedule),("Daily Entry Time",self.daily_entry_time),("Entry Timezone",self.daily_entry_timezone),("Missed Entry Policy",self.daily_entry_missed_policy),("Summary",self.next_entry_summary),("",help_text)]: sched.addRow(lab,w)
        self.enable_daily_schedule.toggled.connect(self.update_dynamic)
        risk=group("Account Risk & Leverage"); self.account_form=risk
        self.risk_mode=PolicyComboBox(); self.risk_mode.addItem("ATR volatility","ATR"); self.risk_mode.addItem("Percent of price","PERCENT"); self.risk_mode.addItem("Fixed price distance","FIXED")
        self.trading_start=self._line(); self.trading_end=self._line(); self.max_lev_leg=self._line("3"); self.max_lev_combined=self._line("5"); self.missing_policy=PolicyComboBox(); self.missing_policy.addItem("Use strategy candle for affected interval","WARN_AND_USE_15M"); self.missing_policy.addItem("Stop the run","ERROR"); self.missing_policy.addItem("Continue with available intrabar candles","WARN_AND_CONTINUE"); self.zero_cost=QCheckBox("Run Zero-Cost Comparison"); self.atr_period=QSpinBox(); self.atr_period.setRange(1,99999); self.atr_mult=self._spin(1,0); self.percent_r=self._line("0.20%"); self.fixed_r=self._spin(100,0); self.equity=self._spin(1000,0,1e12,2); self.risk_leg=self._line("1%")
        self.risk_formula=QLabel(); self.risk_formula.setWordWrap(True); self.risk_warn=QLabel(); self.risk_warn.setWordWrap(True)
        self.account_risk_help=QLabel("Account risk controls how many dollars are planned to be lost at the initial full stop. The selected Strategy Profile can scale this with its Profile Risk Multiplier."); self.account_risk_help.setWordWrap(True)
        for lab,w in [("Starting Equity",self.equity),("Base Risk Per Trade",self.risk_leg),("Effective Account Risk",self.risk_warn),("Maximum Leverage Per Trade",self.max_lev_leg),("Maximum Portfolio Leverage",self.max_lev_combined),("",self.account_risk_help)]: risk.addRow(lab,w)
        basis=group("Stop Distance Basis"); self.distance_basis_form=basis
        self.distance_basis_help=QLabel("This defines one price-distance unit. Strategy Profile stop distances multiply this unit; it does not change the account-risk percentage by itself."); self.distance_basis_help.setWordWrap(True)
        for lab,w in [("Distance Basis",self.risk_mode),("ATR Period",self.atr_period),("ATR Unit Multiplier",self.atr_mult),("Percentage Distance Unit",self.percent_r),("Fixed Distance Unit",self.fixed_r),("Distance Unit",self.risk_formula),("",self.distance_basis_help)]: basis.addRow(lab,w)
        self.risk_mode.currentTextChanged.connect(self.update_dynamic); self.risk_leg.textChanged.connect(self.update_dynamic); self.equity.valueChanged.connect(self.update_dynamic); self.atr_period.valueChanged.connect(self.update_dynamic); self.atr_mult.valueChanged.connect(self.update_dynamic); self.percent_r.textChanged.connect(self.update_dynamic); self.fixed_r.valueChanged.connect(self.update_dynamic)
        period=group("Backtest Period"); self.period_form=period
        self.entire_dataset=QCheckBox("Use entire dataset"); self.entire_dataset.setChecked(True)
        self.trading_start.setPlaceholderText("YYYY-MM-DD (optional)"); self.trading_end.setPlaceholderText("YYYY-MM-DD (optional)")
        for lab,w in [("",self.entire_dataset),("Start Date",self.trading_start),("End Date",self.trading_end)]: period.addRow(lab,w)
        self.entire_dataset.toggled.connect(self.update_dynamic)
        intrabar=group("Intrabar Execution Rules"); self.intrabar_form=intrabar
        for lab,w in [("Missing Data Policy",self.missing_policy),("",self.data_help)]: intrabar.addRow(lab,w)
        self.missing_policy.currentIndexChanged.connect(self.update_dynamic)
        telemetry=group("Trade Telemetry")
        self.enable_trade_telemetry=QCheckBox("Enable Trade Telemetry"); self.telemetry_interval=QSpinBox(); self.telemetry_interval.setRange(1,999999); self.telemetry_interval.setSuffix(" minutes"); self.save_full_telemetry=QCheckBox("Save Full Telemetry CSV"); self.save_journey_summary=QCheckBox("Save Journey Summary"); self.save_journey_charts=QCheckBox("Save Journey Charts"); self.telemetry_estimate=QLabel("Estimated telemetry rows: calculated after data validation when practical."); self.telemetry_estimate.setWordWrap(True)
        telemetry.addRow(QLabel("Records how ATR, ADX, DI Spread, and Bollinger Band Width change while each trade is active."));
        for lab,w in [("",self.enable_trade_telemetry),("Telemetry Interval",self.telemetry_interval),("",self.save_full_telemetry),("",self.save_journey_summary),("",self.save_journey_charts),("Estimate",self.telemetry_estimate)]: telemetry.addRow(lab,w)
        self.enable_trade_telemetry.toggled.connect(self.update_dynamic); self.telemetry_interval.valueChanged.connect(self.update_dynamic)
        lifecycle=group("Indicator Lifecycle Analysis")
        self.enable_lifecycle=QCheckBox("Enable Indicator Lifecycle Analysis"); self.lifecycle_phases=QSpinBox(); self.lifecycle_phases.setRange(4,4); self.lifecycle_checkpoints=self._line("15,30,60"); self.lifecycle_min_sample=QSpinBox(); self.lifecycle_min_sample.setRange(1,100000); self.lifecycle_charts=QCheckBox("Create lifecycle charts"); self.lifecycle_flat_threshold=self._spin(5.0,0,100,2); self.lifecycle_flat_threshold.setSuffix(" %")
        for lab,w in [("",self.enable_lifecycle),("Lifecycle phases",self.lifecycle_phases),("Early checkpoints (minutes)",self.lifecycle_checkpoints),("Minimum bucket sample",self.lifecycle_min_sample),("",self.lifecycle_charts),("Flat-pattern threshold",self.lifecycle_flat_threshold)]: lifecycle.addRow(lab,w)
        reports=group("Optional Output Reports")
        self.save_feature_reports=QCheckBox("Create trailing/partial-exit diagnostic reports"); self.save_indicator_reports=QCheckBox("Create ADX/BB-width/DI-spread analysis reports"); self.create_standard_charts=QCheckBox("Create standard performance charts")
        reports_help=QLabel("Core Excel report, trade list, equity curve, configuration, and log are always saved."); reports_help.setWordWrap(True); reports.addRow(reports_help)
        for w in [self.save_feature_reports,self.save_indicator_reports,self.create_standard_charts]: reports.addRow(w)
        self.analysis_level=QComboBox(); self.analysis_level.addItems(["Fast","Standard (Recommended)","Research"])
        self.analysis_description=QLabel(); self.analysis_description.setWordWrap(True)
        self.analysis_advanced=QPushButton("Show Advanced Settings"); self.analysis_advanced.setCheckable(True)
        telemetry.insertRow(0,"Analysis Level",self.analysis_level); telemetry.insertRow(1,"",self.analysis_description); telemetry.insertRow(2,"",self.analysis_advanced)
        self.analysis_detail_widgets=[self.enable_trade_telemetry,self.telemetry_interval,self.save_full_telemetry,self.save_journey_summary,self.save_journey_charts,self.telemetry_estimate]
        self.analysis_level.currentTextChanged.connect(self._apply_analysis_preset); self.analysis_advanced.toggled.connect(self._set_analysis_advanced)
        fees=group("Fees and Execution")
        self.maker=self._line("0.02%"); self.taker=self._line("0.05%"); self.maker_entry=QCheckBox("Use Maker Fee for Entry"); self.maker_exit=QCheckBox("Use Maker Fee for Exit"); self.slippage=self._line("0.05%"); self.cost=QLabel(); self.cost.setWordWrap(True)
        for lab,w in [("Maker Fee",self.maker),("Taker Fee",self.taker),("",self.maker_entry),("",self.maker_exit),("Slippage",self.slippage),("Round-trip Cost",self.cost),("",self.zero_cost)]: fees.addRow(lab,w)
        for w in [self.maker,self.taker,self.slippage]: w.textChanged.connect(self.update_dynamic)
        self.maker_entry.toggled.connect(self.update_dynamic); self.maker_exit.toggled.connect(self.update_dynamic)
        controls=group("Run Status")
        self.progress=QProgressBar(); self.status=QLabel("Ready"); self.elapsed=QLabel("Elapsed: 0s"); controls.addRow(self.progress); controls.addRow(self.status); controls.addRow(self.elapsed)
        for w in [self.run_name,self.output_folder]: w.textChanged.connect(self.update_planned_output)
        data.parentWidget().setTitle("Data & Output"); strat.parentWidget().setTitle("Entry Timing & Simulation"); sched.parentWidget().setTitle("Scheduled Entry"); fees.parentWidget().setTitle("Execution Costs"); telemetry.parentWidget().setTitle("Reports & Analysis"); lifecycle.parentWidget().setTitle("Advanced Indicator Analysis"); reports.parentWidget().setTitle("Report Files")
        scroll.setWidget(inner); outer.addWidget(scroll); self.backtest_setup_page=page; self.tabs.addTab(page,"Backtest Setup"); self.config_controls=inner.findChildren(QWidget); self._build_di_strategy_tab(); self._build_support_resistance_tab(); self.update_dynamic()

    def _build_support_resistance_tab(self):
        page=QWidget(); outer=QVBoxLayout(page); scroll=QScrollArea(); scroll.setWidgetResizable(True); inner=QWidget(); layout=QVBoxLayout(inner)

        usage=QGroupBox("Support & Resistance"); usage_layout=QVBoxLayout(usage)
        self.enable_support_resistance_analysis.setText("Enable Support & Resistance")
        self.enable_support_resistance_analysis.setToolTip("Calculate, store, and report support/resistance context.")
        usage_layout.addWidget(self.enable_support_resistance_analysis)
        usage_layout.addWidget(QLabel("Usage"))
        self.sr_analyze_only=QRadioButton("Analyze Only")
        self.sr_apply_entry_rules=QRadioButton("Filter Entries")
        usage_layout.addWidget(self.sr_analyze_only)
        analysis_help=QLabel("Record S/R context and reports. This mode never rejects trades."); analysis_help.setWordWrap(True); usage_layout.addWidget(analysis_help)
        usage_layout.addWidget(self.sr_apply_entry_rules)
        filter_help=QLabel("Use the selected LONG/SHORT rules below to reject entries."); filter_help.setWordWrap(True); usage_layout.addWidget(filter_help)
        self.sr_strategy_status=QLabel(); self.sr_strategy_status.setWordWrap(True)
        self.sr_strategy_status.setStyleSheet("font-weight: 600; padding: 4px;")
        usage_layout.addWidget(self.sr_strategy_status)
        layout.addWidget(usage)

        entry_box=QGroupBox("Entry Filters"); entry_layout=QVBoxLayout(entry_box)
        columns=QHBoxLayout(); long_box=QGroupBox("LONG"); lf=QFormLayout(long_box); short_box=QGroupBox("SHORT"); sf=QFormLayout(short_box)
        self.sr_long_avoid_near_resistance.setText("Avoid entry near resistance")
        self.sr_long_require_near_support.setText("Require entry near support")
        self.sr_long_block_broken_support.setText("Reject after support break")
        self.sr_short_avoid_near_support.setText("Avoid entry near support")
        self.sr_short_require_near_resistance.setText("Require entry near resistance")
        self.sr_short_block_broken_resistance.setText("Reject after resistance break")
        self.sr_long_avoid_near_resistance.setToolTip("Reject LONG entries when price is near resistance.")
        self.sr_long_require_near_support.setToolTip("Allow LONG entries only when price is near support.")
        self.sr_short_avoid_near_support.setToolTip("Reject SHORT entries when price is near support.")
        self.sr_short_require_near_resistance.setToolTip("Allow SHORT entries only when price is near resistance.")
        self.sr_long_min_room_to_resistance_atr.setSuffix(" ATR"); self.sr_short_min_room_to_support_atr.setSuffix(" ATR")
        lf.addRow(self.sr_long_require_near_support); lf.addRow(self.sr_long_avoid_near_resistance); lf.addRow(self.sr_long_block_broken_support); lf.addRow("Minimum room to resistance", self.sr_long_min_room_to_resistance_atr)
        sf.addRow(self.sr_short_require_near_resistance); sf.addRow(self.sr_short_avoid_near_support); sf.addRow(self.sr_short_block_broken_resistance); sf.addRow("Minimum room to support", self.sr_short_min_room_to_support_atr)
        columns.addWidget(long_box); columns.addWidget(short_box); entry_layout.addLayout(columns)
        self.sr_trade_context_note=QLabel("Analyze Only is active. Entry filters are saved but do not reject trades."); self.sr_trade_context_note.setWordWrap(True); entry_layout.addWidget(self.sr_trade_context_note)
        self.sr_entry_rules_box=entry_box; layout.addWidget(entry_box)

        proximity=QGroupBox("Price Proximity"); pf=QFormLayout(proximity)
        self.sr_near_distance_atr.setSuffix(" ATR")
        self.sr_near_distance_atr.setToolTip("Price within this ATR distance of the closest S/R zone is considered near that level.")
        pf.addRow("Near-Level Distance",self.sr_near_distance_atr)
        proximity_help=QLabel("Defines when price is classified as near the closest support or resistance zone."); proximity_help.setWordWrap(True); pf.addRow("",proximity_help)
        self.sr_proximity_box=proximity; layout.addWidget(proximity)

        interaction=QGroupBox("Level Interaction"); interaction_layout=QVBoxLayout(interaction)
        hold_widget=QWidget(); hf=QFormLayout(hold_widget); hf.setContentsMargins(0,0,0,0)
        self.enable_sr_hold_confirmation.setText("Confirm level hold after a test")
        self.enable_sr_hold_confirmation.setToolTip("After price tests a zone, mark it HELD only after sufficient rejection within the confirmation window. This does not control when a level is marked BROKEN.")
        self.sr_hold_confirmation_bars.setToolTip("Maximum candles after a zone test in which the required rejection may confirm that the level held.")
        self.sr_hold_confirmation_atr.setToolTip("Minimum rejection away from the tested zone required to classify the level as held.")
        self.sr_hold_confirmation_atr.setSuffix(" ATR")
        hf.addRow(self.enable_sr_hold_confirmation); hf.addRow("Confirmation Window",self.sr_hold_confirmation_bars); hf.addRow("Required Rejection",self.sr_hold_confirmation_atr)
        interaction_layout.addWidget(hold_widget)
        break_box=QGroupBox("Break Detection"); bf=QFormLayout(break_box)
        self.sr_break_basis.setToolTip("Use candle closes or wicks to decide whether price has moved beyond a zone far enough to mark it broken.")
        self.sr_break_tolerance_atr.setToolTip("ATR distance beyond the zone required before the structure is marked broken.")
        self.sr_break_tolerance_atr.setSuffix(" ATR")
        bf.addRow("Break Basis",self.sr_break_basis); bf.addRow("Break Tolerance",self.sr_break_tolerance_atr)
        interaction_layout.addWidget(break_box)
        self.sr_interaction_box=interaction; self.sr_break_detection_box=interaction; layout.addWidget(interaction)

        detection=QGroupBox("Level Detection"); detection_layout=QVBoxLayout(detection)
        preset_row=QFormLayout(); self.sr_detection_preset=QComboBox(); self.sr_detection_preset.addItems(["Conservative","Balanced (Recommended)","Sensitive","Custom"]); self.sr_detection_preset.setCurrentText("Balanced (Recommended)")
        preset_row.addRow("Detection Sensitivity",self.sr_detection_preset); detection_layout.addLayout(preset_row)
        preset_help=QLabel("Use a preset for normal testing. Raw pivot settings are available below for deliberate research only."); preset_help.setWordWrap(True); detection_layout.addWidget(preset_help)
        self.sr_detection_advanced=QGroupBox("Advanced Detection Settings"); self.sr_detection_advanced.setCheckable(True); self.sr_detection_advanced.setChecked(False)
        af=QFormLayout(); self.sr_zone_width_atr.setSuffix(" ATR"); self.sr_zone_width_atr.setToolTip("Merge nearby detected swing levels into one zone when they are within this ATR distance.")
        af.addRow("Pivot Left",self.sr_pivot_left); af.addRow("Pivot Right",self.sr_pivot_right); af.addRow("Lookback Bars",self.sr_lookback_bars); af.addRow("Zone Merge Width",self.sr_zone_width_atr)
        advanced_content=QWidget(); advanced_content.setLayout(af); advanced_wrapper=QVBoxLayout(self.sr_detection_advanced); advanced_wrapper.addWidget(advanced_content); self.sr_detection_advanced.toggled.connect(advanced_content.setVisible); advanced_content.setVisible(False)
        detection_layout.addWidget(self.sr_detection_advanced)
        self.sr_detection_box=detection; self.sr_advanced_box=detection; layout.addWidget(detection)

        summary=QGroupBox("Current Configuration"); sl=QVBoxLayout(summary); self.sr_summary_label=QLabel(); self.sr_summary_label.setWordWrap(True); sl.addWidget(self.sr_summary_label); layout.addWidget(summary); layout.addStretch(1)
        self._sr_detection_presets={"Conservative":{"pivot_left":8,"pivot_right":8,"lookback":300,"zone_width_atr":0.75,"break_tolerance_atr":0.35},"Balanced (Recommended)":{"pivot_left":5,"pivot_right":5,"lookback":200,"zone_width_atr":0.5,"break_tolerance_atr":0.25},"Sensitive":{"pivot_left":3,"pivot_right":3,"lookback":150,"zone_width_atr":0.35,"break_tolerance_atr":0.15}}
        self.sr_detection_preset.currentTextChanged.connect(self._apply_sr_detection_preset)
        for c in (self.sr_pivot_left,self.sr_pivot_right,self.sr_lookback_bars,self.sr_zone_width_atr,self.sr_break_tolerance_atr): c.valueChanged.connect(self._mark_sr_preset_custom)
        self.sr_analyze_only.toggled.connect(lambda checked: checked and self.sr_filter_mode.setCurrentText("ANALYSIS_ONLY")); self.sr_apply_entry_rules.toggled.connect(lambda checked: checked and self.sr_filter_mode.setCurrentText("APPLY_ENTRY_RULES"))
        for c in (self.enable_support_resistance_analysis,self.enable_sr_hold_confirmation,self.sr_analyze_only,self.sr_apply_entry_rules): c.toggled.connect(self.update_dynamic)
        for c in (self.sr_filter_mode,self.sr_break_basis): c.currentTextChanged.connect(self.update_dynamic)
        for c in (self.sr_near_distance_atr,self.sr_zone_width_atr,self.sr_hold_confirmation_bars,self.sr_hold_confirmation_atr,self.sr_break_tolerance_atr,self.sr_pivot_left,self.sr_pivot_right,self.sr_lookback_bars): c.valueChanged.connect(self.update_dynamic)
        for c in (self.sr_long_avoid_near_resistance,self.sr_long_require_near_support,self.sr_long_block_broken_support,self.sr_short_avoid_near_support,self.sr_short_require_near_resistance,self.sr_short_block_broken_resistance): c.toggled.connect(self.update_dynamic)
        for c in (self.sr_long_min_room_to_resistance_atr,self.sr_short_min_room_to_support_atr): c.valueChanged.connect(self.update_dynamic)
        outer.addWidget(scroll); scroll.setWidget(inner); self.tabs.addTab(page,"Support & Resistance"); self._update_sr_usage_radios(); self._update_sr_tab_state()

    def _update_sr_usage_radios(self):
        analysis=self.sr_filter_mode.currentText()=="ANALYSIS_ONLY"
        for control,value in ((self.sr_analyze_only,analysis),(self.sr_apply_entry_rules,not analysis)):
            control.blockSignals(True); control.setChecked(value); control.blockSignals(False)

    def _apply_sr_detection_preset(self,name):
        preset=self._sr_detection_presets.get(name)
        if not preset: return
        self._applying_sr_preset=True
        try:
            self.sr_pivot_left.setValue(preset["pivot_left"]); self.sr_pivot_right.setValue(preset["pivot_right"]); self.sr_lookback_bars.setValue(preset["lookback"]); self.sr_zone_width_atr.setValue(preset["zone_width_atr"]); self.sr_break_tolerance_atr.setValue(preset["break_tolerance_atr"])
        finally:
            self._applying_sr_preset=False

    def _mark_sr_preset_custom(self,*args):
        if getattr(self,"_applying_sr_preset",False): return
        if self.sr_detection_preset.currentText()!="Custom":
            self.sr_detection_preset.blockSignals(True); self.sr_detection_preset.setCurrentText("Custom"); self.sr_detection_preset.blockSignals(False)

    def _sync_sr_preset_from_values(self):
        current={"pivot_left":self.sr_pivot_left.value(),"pivot_right":self.sr_pivot_right.value(),"lookback":self.sr_lookback_bars.value(),"zone_width_atr":round(self.sr_zone_width_atr.value(),3),"break_tolerance_atr":round(self.sr_break_tolerance_atr.value(),3)}
        matched="Custom"
        for name,preset in self._sr_detection_presets.items():
            if preset=={**preset,**{k:current[k] for k in preset if k in current}} and current==preset:
                matched=name; break
        self._applying_sr_preset=True
        try: self.sr_detection_preset.setCurrentText(matched)
        finally: self._applying_sr_preset=False

    def _update_sr_tab_state(self):
        if not hasattr(self,"sr_summary_label"): return
        self._update_sr_usage_radios()
        enabled=self.enable_support_resistance_analysis.isChecked()
        applying=self.sr_filter_mode.currentText()!="ANALYSIS_ONLY"
        self.sr_analyze_only.setEnabled(enabled); self.sr_apply_entry_rules.setEnabled(enabled)
        for box in (self.sr_proximity_box,self.sr_interaction_box,self.sr_detection_box): box.setEnabled(enabled)
        self.sr_entry_rules_box.setEnabled(enabled and applying)
        confirmation=enabled and self.enable_sr_hold_confirmation.isChecked()
        self.sr_hold_confirmation_bars.setEnabled(confirmation); self.sr_hold_confirmation_atr.setEnabled(confirmation)
        if not enabled:
            note="Support & Resistance is disabled."
            impact="DISABLED"
        elif not applying:
            note="Analyze Only is active. Entry filters are saved but do not reject trades."
            impact="NONE — ANALYSIS ONLY"
        else:
            note="Filter Entries is active. Selected LONG/SHORT rules may reject entries."
            impact="ENTRY FILTER ACTIVE"
        self.sr_trade_context_note.setText(note)
        self.sr_strategy_status.setText(f"Trading impact: {impact}")
        long_rules=[label for c,label in ((self.sr_long_require_near_support,"Require near support"),(self.sr_long_avoid_near_resistance,"Avoid near resistance"),(self.sr_long_block_broken_support,"Reject after support break")) if c.isChecked()]
        short_rules=[label for c,label in ((self.sr_short_require_near_resistance,"Require near resistance"),(self.sr_short_avoid_near_support,"Avoid near support"),(self.sr_short_block_broken_resistance,"Reject after resistance break")) if c.isChecked()]
        if self.sr_long_min_room_to_resistance_atr.value() > 0: long_rules.append(f"Minimum room: {self.sr_long_min_room_to_resistance_atr.value():.2f} ATR")
        if self.sr_short_min_room_to_support_atr.value() > 0: short_rules.append(f"Minimum room: {self.sr_short_min_room_to_support_atr.value():.2f} ATR")
        long_text=", ".join(long_rules) or "None"
        short_text=", ".join(short_rules) or "None"
        mode="Filter Entries" if applying else "Analyze Only"
        preset=self.sr_detection_preset.currentText()
        self.sr_summary_label.setText(
            f"Mode: {mode}\n"
            f"Detection: {preset}\n"
            f"Near level: ≤ {self.sr_near_distance_atr.value():.2f} ATR\n\n"
            f"LONG filters: {long_text}\n"
            f"SHORT filters: {short_text}"
        )

    def _build_di_strategy_tab(self):
        page=QWidget(); outer=QVBoxLayout(page); scroll=QScrollArea(); scroll.setWidgetResizable(True); inner=QWidget(); form=QVBoxLayout(inner)
        intro=QLabel("DI-direction strategy settings live here. Shared data, risk, fees, execution, telemetry, and output settings remain on the Configuration tab.")
        intro.setWordWrap(True); form.addWidget(intro)
        direction_box=QGroupBox("DI Direction Selection"); direction_form=QFormLayout(direction_box)
        rule=QLabel("Current Rule\n+DI above -DI → LONG\n-DI above +DI → SHORT"); rule.setWordWrap(True)
        direction_form.addRow("",self.enable_di_direction_selection); direction_form.addRow("",rule); form.addWidget(direction_box)
        pressure_box=QGroupBox("DI Pressure Analysis"); pressure_form=QFormLayout(pressure_box)
        mode=QLabel("Analysis Mode: RECORD ONLY\nDoes not filter or reject trades.")
        help_text=QLabel("DI direction chooses LONG or SHORT from +DI versus -DI. DI Pressure Analysis measures whether directional pressure is strengthening or weakening before entry. It is analysis-only and does not filter trades. DI Spread entry filtering is configured separately under Strategy Profiles → Rules → DI Spread."); help_text.setWordWrap(True)
        pressure_form.addRow("",self.enable_di_pressure_analysis); pressure_form.addRow("Lookback",self.di_pressure_lookback); pressure_form.addRow("",mode); pressure_form.addRow("",help_text); form.addWidget(pressure_box)
        mean_box=QGroupBox("Mean Reversion Analysis"); mean_form=QFormLayout(mean_box)
        mean_mode=QLabel("Analysis Mode: RECORD ONLY\nDoes not filter or reject trades.")
        mean_help=QLabel("Uses a causal EMA as the recent mean and measures the entry close's signed distance from that mean in configured ATR units. Alignment and motion are recorded separately so you can test whether mean reversion adds value at any DI-pressure level. Reports cover all DI buckets; no DI cutoff is hard-coded."); mean_help.setWordWrap(True)
        self.enable_mean_reversion_analysis.setToolTip("Record mean-reversion telemetry only. This setting never changes trade selection or direction.")
        self.mean_reversion_period.setToolTip("EMA period used as the recent price mean. Distance from the EMA is normalized by the configured ATR.")
        mean_form.addRow("",self.enable_mean_reversion_analysis); mean_form.addRow("Mean EMA Period",self.mean_reversion_period); mean_form.addRow("",mean_mode); mean_form.addRow("",mean_help); form.addWidget(mean_box)
        form.addStretch(1)
        scroll.setWidget(inner); outer.addWidget(scroll); self.di_strategy_page=page
        self.config_controls += inner.findChildren(QWidget)
        self.tabs.addTab(page,"DI Direction & Pressure")
        self.analysis_level.setCurrentText("Standard (Recommended)"); self._apply_analysis_preset(); self._set_analysis_advanced(False)
    def _build_portfolio_tab(self):
        page=QWidget(); layout=QVBoxLayout(page); box=QGroupBox("Shared-Equity Portfolio"); form=QFormLayout(box)
        self.portfolio_assets=[]; asset_widget=QWidget(); self.portfolio_asset_layout=QVBoxLayout(asset_widget); self.portfolio_asset_layout.setContentsMargins(0,0,0,0); form.addRow("Assets",asset_widget)
        add_asset=QPushButton("Add Asset"); add_asset.clicked.connect(lambda:self._add_portfolio_asset()); form.addRow("",add_asset)
        self.portfolio_initial_equity=self._spin(1000,1,1e12,2); self.portfolio_risk_per_asset=QLineEdit("1%"); self.portfolio_maximum_total_risk=QLineEdit("5%")
        self.portfolio_output_folder=QLineEdit("output"); self.portfolio_output_folder.setReadOnly(True); output_btn=QPushButton("Browse"); output_btn.clicked.connect(self._browse_portfolio_output); output_row=QHBoxLayout(); output_row.addWidget(self.portfolio_output_folder); output_row.addWidget(output_btn)
        self.portfolio_help=QLabel(); self.portfolio_help.setWordWrap(True)
        for label,control in [("Starting Equity",self.portfolio_initial_equity),("Base Risk Per Asset",self.portfolio_risk_per_asset),("Maximum Total Portfolio Risk",self.portfolio_maximum_total_risk),("Output Folder",output_row),("",self.portfolio_help)]: form.addRow(label,control)
        self.portfolio_risk_per_asset.textChanged.connect(self._update_portfolio_help); self.portfolio_maximum_total_risk.textChanged.connect(self._update_portfolio_help)
        for symbol in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"): self._add_portfolio_asset(symbol)
        layout.addWidget(box); buttons=QHBoxLayout(); self.portfolio_run_btn=QPushButton("Run Portfolio Backtest"); self.portfolio_run_btn.clicked.connect(self.run_portfolio_backtest); self.portfolio_open_btn=QPushButton("Open Portfolio Output"); self.portfolio_open_btn.clicked.connect(self._open_portfolio_output); buttons.addWidget(self.portfolio_run_btn); buttons.addWidget(self.portfolio_open_btn); layout.addLayout(buttons)
        self.portfolio_progress=QProgressBar(); self.portfolio_status=QLabel("Select configurations for at least two enabled assets."); layout.addWidget(self.portfolio_progress); layout.addWidget(self.portfolio_status)
        self.portfolio_summary_table=QTableWidget(0,2); self.portfolio_summary_table.setHorizontalHeaderLabels(["Portfolio Metric","Value"]); self.portfolio_summary_table.horizontalHeader().setStretchLastSection(True); layout.addWidget(self.portfolio_summary_table); self.tabs.addTab(page,"Portfolio")
    def _add_portfolio_asset(self,symbol=""):
        row=QWidget(); line=QHBoxLayout(row); line.setContentsMargins(0,0,0,0); enabled=QCheckBox("Include"); enabled.setChecked(True); pair=QComboBox(); pair.setEditable(True); pair.addItems(["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"]); pair.setCurrentText(symbol or "BTCUSDT"); config=QLineEdit(); config.setReadOnly(True); config.setPlaceholderText("Select this asset's saved configuration"); browse=QPushButton("Browse"); remove=QPushButton("Remove")
        line.addWidget(enabled); line.addWidget(pair); line.addWidget(config,1); line.addWidget(browse); line.addWidget(remove); entry={"row":row,"enabled":enabled,"pair":pair,"config":config}; self.portfolio_assets.append(entry); self.portfolio_asset_layout.addWidget(row)
        browse.clicked.connect(lambda:self._browse_portfolio_config(config)); remove.clicked.connect(lambda:self._remove_portfolio_asset(entry)); enabled.toggled.connect(self._update_portfolio_help); pair.currentTextChanged.connect(self._update_portfolio_help); self._update_portfolio_help()
    def _remove_portfolio_asset(self,entry):
        if entry not in self.portfolio_assets:return
        self.portfolio_assets.remove(entry); self.portfolio_asset_layout.removeWidget(entry["row"]); entry["row"].deleteLater(); self._update_portfolio_help()
    def _update_portfolio_help(self,*_):
        if not hasattr(self,"portfolio_help"):return
        count=sum(item["enabled"].isChecked() for item in self.portfolio_assets)
        try:risk=parse_percentage(self.portfolio_risk_per_asset.text()); requested=format_percentage(risk*count,2); cap=format_percentage(parse_percentage(self.portfolio_maximum_total_risk.text()),2)
        except Exception:requested=cap="invalid"
        self.portfolio_help.setText(f"{count} assets enabled. One open trade per asset would request approximately {requested} total risk. The hard portfolio limit is {cap}; new entries that would exceed it are blocked and reported. Each asset uses its own saved strategy configuration, while all accepted trades share one account.")
    def _browse_portfolio_config(self,target):
        path,_=QFileDialog.getOpenFileName(self,"Select Saved Strategy Configuration","","JSON (*.json)")
        if path: target.setText(path)
    def _browse_portfolio_output(self):
        path=QFileDialog.getExistingDirectory(self,"Select Portfolio Output Folder")
        if path: self.portfolio_output_folder.setText(path)
    def _open_portfolio_output(self):
        path=getattr(self,"portfolio_output_dir",Path(self.portfolio_output_folder.text() or "output"))
        if Path(path).exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
    def run_portfolio_backtest(self):
        try:
            components=[]; seen=set()
            for item in self.portfolio_assets:
                if not item["enabled"].isChecked():continue
                symbol=item["pair"].currentText().strip().upper().replace("/",""); path=item["config"].text().strip()
                if not symbol:raise ValueError("Every enabled asset needs a trading pair.")
                if symbol in seen:raise ValueError(f"{symbol} is included more than once.")
                if not Path(path).is_file():raise ValueError(f"Select a valid saved configuration for {symbol}.")
                values=load_config_json(path); configured=str(values.get("market_symbol") or self._pair_from_path(values.get("input_csv") or "") or "").upper()
                if configured and configured != symbol:raise ValueError(f"{symbol} row uses a {configured} configuration. Select the matching configuration.")
                seen.add(symbol); components.append((symbol,path))
            if len(components)<2:raise ValueError("Enable at least two assets and select their saved configurations.")
            risk=parse_percentage(self.portfolio_risk_per_asset.text())
            if not 0 < risk < 1: raise ValueError("Risk per asset must be above 0% and below 100%.")
            maximum_total_risk=parse_percentage(self.portfolio_maximum_total_risk.text())
            if not risk <= maximum_total_risk < 1: raise ValueError("Maximum total portfolio risk must be at least the per-asset risk and below 100%.")
            output=self.portfolio_output_folder.text().strip() or "output"; Path(output).mkdir(parents=True,exist_ok=True)
        except Exception as exc: QMessageBox.warning(self,"Portfolio Validation",str(exc)); return
        self.portfolio_thread=QThread(); self.portfolio_worker=PortfolioWorker(components,output,self.portfolio_initial_equity.value(),risk,maximum_total_risk); self.portfolio_worker.moveToThread(self.portfolio_thread); self.portfolio_thread.started.connect(self.portfolio_worker.run); self.portfolio_thread.finished.connect(self._portfolio_thread_finished); self.portfolio_worker.status.connect(self._on_portfolio_status); self.portfolio_worker.log.connect(self.append_log); self.portfolio_worker.finished.connect(self._on_portfolio_finished); self.portfolio_worker.failed.connect(self._on_portfolio_failed); self.portfolio_run_btn.setEnabled(False); self.new_run_btn.setEnabled(False); self.portfolio_progress.setValue(0); self.portfolio_thread.start()
    def _on_portfolio_status(self,text,percent): self.portfolio_status.setText(text); self.portfolio_progress.setValue(percent)
    def _on_portfolio_finished(self,summary,trades,equity,out):
        self.portfolio_output_dir=Path(out); self.portfolio_summary_table.setRowCount(len(summary))
        for row,(key,value) in enumerate(summary.items()): self.portfolio_summary_table.setItem(row,0,QTableWidgetItem(str(key))); self.portfolio_summary_table.setItem(row,1,QTableWidgetItem(str(value)))
        self.portfolio_status.setText(f"Completed: {out}"); self._play_completion_sound(); self._cleanup_portfolio_thread()
    def _on_portfolio_failed(self,message,tb): QMessageBox.critical(self,"Portfolio Backtest Error",message); self.append_log(tb); self._cleanup_portfolio_thread()
    def _cleanup_portfolio_thread(self):
        if self.portfolio_thread is not None: self.portfolio_thread.quit()
        else: self.portfolio_run_btn.setEnabled(True)
    def _portfolio_thread_finished(self):
        thread=self.portfolio_thread; self.portfolio_thread=None; self.portfolio_worker=None; self.portfolio_run_btn.setEnabled(True); self.new_run_btn.setEnabled(not bool(self.thread and self.thread.isRunning()))
        if thread is not None: thread.deleteLater()
    def _build_summary(self):
        page=QScrollArea(); page.setWidgetResizable(True); page.setObjectName("summaryScrollArea")
        content=QWidget(); l=QVBoxLayout(content); l.setAlignment(Qt.AlignTop)
        overview=QGroupBox("Performance Overview"); overview_layout=QGridLayout(overview); self.kpi_labels={}
        kpis=("Ending Equity","Total Return","Total Trades","Win Rate","Profit Factor","Maximum Drawdown","Average Net R","Total Net R","Total Fees","Signals Traded / Evaluated")
        for index,label in enumerate(kpis):
            card=QGroupBox(label); card_layout=QVBoxLayout(card); value=QLabel("—"); value.setAlignment(Qt.AlignCenter); value.setStyleSheet("font-size: 18px; font-weight: 600; padding: 4px;")
            card_layout.addWidget(value); self.kpi_labels[label]=value; overview_layout.addWidget(card,index//5,index%5)
        self.starting_equity_label=QLabel("Starting Equity: —"); overview_layout.addWidget(self.starting_equity_label,2,0,1,5)
        l.addWidget(overview)
        reports_box=QGroupBox("Run Reports"); reports_layout=QGridLayout(reports_box)
        labels={"output":"Open Output Folder","backtest":"Open Backtest Report","indicators":"Open Indicator Analysis","sr":"Open S/R Analysis","trades":"Open Trade List","charts":"Open Charts Folder"}
        self.report_buttons={}
        for index,(name,label) in enumerate(labels.items()):
            button=QPushButton(label); button.setEnabled(False); button.clicked.connect(lambda _checked=False, key=name: self._open_report(key))
            self.report_buttons[name]=button; reports_layout.addWidget(button,index//3,index%3)
        l.addWidget(reports_box)
        self.comparison_box=QGroupBox("Direction / Regime Performance"); comparison_layout=QVBoxLayout(self.comparison_box)
        self.combo_table=QTableWidget(0,9); self.combo_table.setHorizontalHeaderLabels(["Market Regime","Direction","Trades","Wins","Losses","Win Rate","Average Net R","Total Net R","Net PnL"]); self.combo_table.setSortingEnabled(False); self.combo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.combo_table.horizontalHeader().setStretchLastSection(True); self.combo_table.verticalHeader().setVisible(False)
        comparison_layout.addWidget(self.combo_table); l.addWidget(self.comparison_box)
        sr_summary_box=QGroupBox("Support / Resistance"); sr_summary_layout=QVBoxLayout(sr_summary_box)
        self.sr_summary_panel_label=QLabel("No backtest run yet."); self.sr_summary_panel_label.setWordWrap(True)
        sr_summary_layout.addWidget(self.sr_summary_panel_label); l.addWidget(sr_summary_box)
        page.setWidget(content); self.summary_scroll_area=page; self.summary_content=content
        self.tabs.addTab(page,"Summary")

    def _size_summary_table(self):
        height=self.combo_table.horizontalHeader().height()+self.combo_table.frameWidth()*2
        height += sum(self.combo_table.rowHeight(row) for row in range(self.combo_table.rowCount()))
        self.combo_table.setFixedHeight(max(80,height+2))

    def _refresh_report_buttons(self):
        states=report_button_states(self.completed_run_dir) if self.completed_run_dir is not None else {name:False for name in REPORT_TARGETS}
        for name,button in self.report_buttons.items(): button.setEnabled(states[name])

    def _open_report(self,name):
        if self.completed_run_dir is None: return
        target=(self.completed_run_dir/REPORT_TARGETS[name]).resolve()
        if target.exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
    def _build_log(self):
        # Keep an internal buffer for diagnostics and saved log files without
        # constructing a large interactive Log tab that the workflow does not use.
        self.log=QTextEdit(readOnly=True)
    @staticmethod
    def _timeframe_minutes(label):
        return int(label[:-1]) * (60 if label.endswith("h") else 1)
    @staticmethod
    def _timeframe_label(minutes):
        return f"{minutes // 60}h" if minutes >= 60 and minutes % 60 == 0 else f"{minutes}m"
    def _timeframe_changed(self):
        strategy_minutes=self._timeframe_minutes(self.strategy_timeframe.currentText())
        if strategy_minutes == 1:
            self.use_intrabar.setChecked(False)
        if hasattr(self,"telemetry_interval") and self.telemetry_interval.value() % strategy_minutes:
            self.telemetry_interval.setValue(strategy_minutes)
        self.update_dynamic()
        self.update_planned_output()

    def _base_values(self):
        return {
            "config_version": 2,
            "strategy_timeframe_minutes": self._timeframe_minutes(self.strategy_timeframe.currentText()),
            "intrabar_timeframe_minutes": self._timeframe_minutes(self.intrabar_timeframe.currentText()),
            "enable_indicator_lifecycle_analysis": self.enable_lifecycle.isChecked(),
            "lifecycle_phases": self.lifecycle_phases.value(),
            "lifecycle_early_checkpoints": [int(v.strip()) for v in self.lifecycle_checkpoints.text().split(",") if v.strip()],
            "lifecycle_minimum_bucket_sample": self.lifecycle_min_sample.value(),
            "create_lifecycle_charts": self.lifecycle_charts.isChecked(),
            "lifecycle_flat_pattern_threshold_pct": self.lifecycle_flat_threshold.value(),
            "run_name": self.run_name.text().strip(),
            "input_csv": self.input_csv.text(),
            "intrabar_csv": self.intrabar_csv.text(), "use_intrabar_data": self.use_intrabar.isChecked(),
            "trading_start_date": self.trading_start.text() or None, "trading_end_date": self.trading_end.text() or None,
            "max_effective_leverage_per_leg": self.max_lev_leg.text() or None,
            "max_combined_effective_leverage": self.max_lev_combined.text() or None,
            "intrabar_missing_policy": self.missing_policy.currentText(), "zero_cost_comparison": self.zero_cost.isChecked(),
            "output_dir": self.output_folder.text(), "entry_mode": self.entry_mode.currentData(), "entry_interval": self.entry_interval.value(),
            "enable_daily_entry_schedule": self.enable_daily_schedule.isChecked(), "daily_entry_time": self.daily_entry_time.text().strip(),
            "daily_entry_timezone": self.daily_entry_timezone.text().strip(), "daily_entry_missed_policy": self.daily_entry_missed_policy.currentText(),
            "max_active_pairs": self.max_pairs.value(), "tie_policy": self.tie.currentData(),
            "risk_mode": self.risk_mode.currentText(), "atr_period": self.atr_period.value(), "atr_multiplier": self.atr_mult.value(),
            "percent_r": parse_percentage(self.percent_r.text()), "fixed_r": self.fixed_r.value(), "initial_equity": self.equity.value(),
            "risk_per_leg": parse_percentage(self.risk_leg.text()), "maker_fee": parse_percentage(self.maker.text()),
            "taker_fee": parse_percentage(self.taker.text()), "use_maker_entry": self.maker_entry.isChecked(),
            "use_maker_exit": self.maker_exit.isChecked(), "slippage": parse_percentage(self.slippage.text()),
            "adx_period": self._shared_adx_period,
            "enable_trade_telemetry": self.enable_trade_telemetry.isChecked(), "save_full_telemetry_csv": self.save_full_telemetry.isChecked(),
            "save_trade_journey_summary": self.save_journey_summary.isChecked(), "save_trade_journey_charts": self.save_journey_charts.isChecked(),
            "telemetry_interval_minutes": self.telemetry_interval.value(),
        }
    def values(self):
        values = self._base_values()
        values["market_symbol"]=self.market_symbol.currentText().strip().upper().replace("/","")
        if self.entire_dataset.isChecked():
            values["trading_start_date"]=None; values["trading_end_date"]=None
        values["analysis_level"]=self.analysis_level.currentText().split(" ",1)[0].upper()
        values.update({"enable_support_resistance_analysis":self.enable_support_resistance_analysis.isChecked(),"sr_pivot_left":self.sr_pivot_left.value(),"sr_pivot_right":self.sr_pivot_right.value(),"sr_lookback_bars":self.sr_lookback_bars.value(),"sr_zone_width_atr":self.sr_zone_width_atr.value(),"sr_near_distance_atr":self.sr_near_distance_atr.value(),"enable_sr_hold_confirmation":self.enable_sr_hold_confirmation.isChecked(),"sr_hold_confirmation_bars":self.sr_hold_confirmation_bars.value(),"sr_hold_confirmation_atr":self.sr_hold_confirmation_atr.value(),"sr_break_tolerance_atr":self.sr_break_tolerance_atr.value(),"sr_break_basis":self.sr_break_basis.currentText(),"sr_filter_mode":self.sr_filter_mode.currentData(),"sr_long_avoid_near_resistance":self.sr_long_avoid_near_resistance.isChecked(),"sr_long_require_near_support":self.sr_long_require_near_support.isChecked(),"sr_long_block_broken_support":self.sr_long_block_broken_support.isChecked(),"sr_long_min_room_to_resistance_atr":self.sr_long_min_room_to_resistance_atr.value(),"sr_short_avoid_near_support":self.sr_short_avoid_near_support.isChecked(),"sr_short_require_near_resistance":self.sr_short_require_near_resistance.isChecked(),"sr_short_block_broken_resistance":self.sr_short_block_broken_resistance.isChecked(),"sr_short_min_room_to_support_atr":self.sr_short_min_room_to_support_atr.value()})
        values.update({"enable_di_direction_selection":self.enable_di_direction_selection.isChecked(),"enable_di_pressure_analysis":self.enable_di_pressure_analysis.isChecked(),"di_pressure_lookback":self.di_pressure_lookback.value(),"enable_mean_reversion_analysis":self.enable_mean_reversion_analysis.isChecked(),"mean_reversion_period":self.mean_reversion_period.value()})
        values.update(self.profile_editor.values())
        return values

    def _apply_analysis_preset(self,*_):
        level=self.analysis_level.currentText()
        fast=level=="Fast"; research=level=="Research"
        self.enable_trade_telemetry.setChecked(research); self.save_full_telemetry.setChecked(research); self.save_journey_summary.setChecked(research); self.save_journey_charts.setChecked(research)
        self.enable_lifecycle.setChecked(research); self.lifecycle_charts.setChecked(research)
        self.save_feature_reports.setChecked(research); self.save_indicator_reports.setChecked(not fast); self.create_standard_charts.setChecked(not fast)
        descriptions={"Fast":"Core results only. Best for quick parameter checks.","Standard (Recommended)":"Performance charts and indicator summaries without heavy candle-by-candle telemetry.","Research":"Full telemetry, trade journeys, lifecycle analysis, diagnostics, and all charts. Slowest and produces the most files."}
        self.analysis_description.setText(descriptions[level])

    def _set_analysis_advanced(self,shown):
        self.analysis_advanced.setChecked(bool(shown)); self.analysis_advanced.setText("Hide Advanced Settings" if shown else "Show Advanced Settings")
        for widget in self.analysis_detail_widgets:
            widget.setVisible(shown); label=widget.parentWidget().layout().labelForField(widget) if isinstance(widget.parentWidget().layout(),QFormLayout) else None
            if label: label.setVisible(shown)
        self.enable_lifecycle.parentWidget().setVisible(shown); self.save_feature_reports.parentWidget().setVisible(shown)

    def reset_defaults(self):
        self.apply_values(default_gui_config())

    def new_run(self):
        """Confirm and prepare a fresh configuration without touching persisted data."""
        if self._git_work_active(): return
        if not self._confirm_new_run(): return
        self.apply_values(default_gui_config())
        self._clear_displayed_results()

    def _confirm_new_run(self):
        prompt=QMessageBox(self); prompt.setWindowTitle("Start a new run?")
        prompt.setText("Start a new run?")
        prompt.setInformativeText(
            "This will reset the backtest setup and strategy settings\n"
            "to their default values.\n\n"
            "Existing backtest results and market data will not be deleted.")
        prompt.addButton(QMessageBox.Cancel)
        confirm=prompt.addButton("New Run",QMessageBox.AcceptRole)
        prompt.setDefaultButton(QMessageBox.Cancel); prompt.exec()
        return prompt.clickedButton() is confirm

    def _clear_displayed_results(self):
        """Clear in-memory result presentation, never persisted run artifacts."""
        self.last_summary={}; self._pending_ui_results=None; self.completed_run_dir=None
        self._refresh_report_buttons()
        self.populate_summary({},pd.DataFrame())
        self.sr_summary_panel_label.setText("No backtest run yet.")
        self.progress.setValue(0); self.status.setText("Ready"); self.elapsed.setText("Elapsed: 0s")
        self.update_dynamic(); self.update_planned_output()
    def _restore_settings(self):
        self.output_folder.setText(self.settings.value("last_output", self.output_folder.text())); self._sync_dataset_paths()
    def browse_csv(self):
        p,_=QFileDialog.getOpenFileName(self,"Select CSV",self.input_csv.text(),"CSV files (*.csv)");
        if p:
            self.input_csv.setText(p); self.settings.setValue("last_csv",p); pair=self._pair_from_path(p)
            if pair: self.market_symbol.setCurrentText(pair)
            self.validate_data()
    def browse_intrabar_csv(self):
        p,_=QFileDialog.getOpenFileName(self,"Select Intrabar CSV",self.intrabar_csv.text(),"CSV files (*.csv)");
        if p: self.intrabar_csv.setText(p)
    @staticmethod
    def _pair_from_path(path):
        match=re.search(r"([A-Z0-9]+USDT)(?:[_-]|$)",Path(path).stem.upper())
        return match.group(1) if match else None
    def _sync_dataset_paths(self,*_):
        if getattr(self,"_applying_values",False): return
        symbol=self.market_symbol.currentText().strip().upper().replace("/","")
        def matching_file(timeframe):
            expected=self.market_data_folder/f"{symbol}_{timeframe}.csv"
            if expected.exists(): return str(expected.resolve())
            if self.market_data_folder.is_dir():
                match=next((path for path in self.market_data_folder.glob("*.csv") if path.name.lower()==expected.name.lower()),None)
                if match: return str(match.resolve())
            return ""
        self.input_csv.setText(matching_file(self.strategy_timeframe.currentText()))
        self.intrabar_csv.setText(matching_file(self.intrabar_timeframe.currentText()) if self.use_intrabar.isChecked() else "")
        if hasattr(self,"dataset_info"): self.dataset_info.setText("Matching shared dataset selected." if self.input_csv.text() else f"No matching dataset in {self.market_data_folder}. Open Binance Data Hub to download it.")
        if hasattr(self,"planned_output"): self.update_planned_output()
    def browse_output(self):
        p=QFileDialog.getExistingDirectory(self,"Select Output Folder",self.output_folder.text());
        if p: self.output_folder.setText(p); self.settings.setValue("last_output",p)
    def validate_data(self):
        try:
            strategy_pair=self._pair_from_path(self.input_csv.text()); intrabar_pair=self._pair_from_path(self.intrabar_csv.text()) if self.use_intrabar.isChecked() else None; selected=self.market_symbol.currentText().strip().upper().replace("/","")
            if strategy_pair and intrabar_pair and strategy_pair != intrabar_pair: raise ValueError(f"Dataset pair mismatch: strategy file is {strategy_pair}, but intrabar file is {intrabar_pair}.")
            if strategy_pair and selected and strategy_pair != selected: raise ValueError(f"Selected pair is {selected}, but the strategy filename identifies {strategy_pair}.")
            df=load_ohlcv_csv(self.input_csv.text(), expected_timeframe_minutes=self._timeframe_minutes(self.strategy_timeframe.currentText()), label="Strategy data", strict_timeframe=True); self._validated_strategy_data=df; sm=df.attrs.get("summary"); miss=sm.missing_candles; tf=f"{sm.detected_timeframe_minutes} minutes"; self.dataset_info.setText(f"Total candles: {len(df):,}\nStart date: {df.timestamp.min()}\nEnd date: {df.timestamp.max()}\nDetected timeframe: {tf}\nMissing candles: {miss}\nRows removed: see log/console\nDuplicate candles removed: see log/console"); self.append_log("Data validation passed."); return True
        except Exception as e: QMessageBox.warning(self,"Invalid CSV",str(e)); self.append_log(traceback.format_exc()); return False
    def update_planned_output(self):
        try:
            cfg=build_backtest_config(self.values(), require_paths=False); self.planned_output.setText(str(planned_run_dir(cfg).resolve()))
        except Exception:
            self.planned_output.setText("Output run folder: unavailable until configuration is valid")
    def update_dynamic(self):
        self.entry_interval.setEnabled(self.entry_mode.currentData()=="EVERY_N_CANDLES")
        self.max_pairs.setEnabled(self.entry_mode.currentData()=="EVERY_N_CANDLES")
        self._update_sr_tab_state()
        if hasattr(self,"strategy_timeframe"):
            strategy=self._timeframe_minutes(self.strategy_timeframe.currentText())
            intrabar=self._timeframe_minutes(self.intrabar_timeframe.currentText())
            available=strategy > 1
            self.use_intrabar.setEnabled(available)
            intrabar_enabled=available and self.use_intrabar.isChecked()
            self.intrabar_timeframe.setEnabled(intrabar_enabled)
            self.intrabar_csv.setEnabled(intrabar_enabled)
        self.missing_policy.setEnabled(intrabar_enabled)
        policy_help={
            "WARN_AND_USE_15M":f"If {self.intrabar_timeframe.currentText()} data is incomplete, that affected interval is evaluated using its {self.strategy_timeframe.currentText()} strategy candle.",
            "ERROR":f"The run stops if any required {self.intrabar_timeframe.currentText()} candle is missing during an open trade.",
            "WARN_AND_CONTINUE":f"Available {self.intrabar_timeframe.currentText()} candles are evaluated and missing portions are skipped.",
        }.get(self.missing_policy.currentData(),"")
        self.data_help.setText(f"ATR, entry price, SL, and TP are calculated from {self.strategy_timeframe.currentText()} candles.\n" + (f"{self.intrabar_timeframe.currentText()} candles determine the exact exit sequence. {policy_help}\n" if self.use_intrabar.isChecked() else "Intrabar exit resolution is disabled.\n") + "Fees are charged on full notional, not margin; leverage changes required margin but does not reduce trading fees.")
        m=getattr(self,'risk_mode',None) and self.risk_mode.currentText(); self.atr_period.setVisible(m=="ATR"); self.atr_mult.setVisible(m=="ATR"); self.percent_r.setVisible(m=="PERCENT"); self.fixed_r.setVisible(m=="FIXED")
        try:
            distance_text={
                "ATR":f"1 volatility unit = ATR({self.atr_period.value()}) × {self.atr_mult.value():g}",
                "PERCENT":f"1 distance unit = Entry Price × {format_percentage(parse_percentage(self.percent_r.text()),2)}",
                "FIXED":f"1 distance unit = {self.fixed_r.value():g} price units",
            }.get(m,"")
            self.risk_formula.setText(distance_text)
        except Exception:
            self.risk_formula.setText("Distance unit unavailable until the value is valid")
        if hasattr(self,"distance_basis_form"):
            self.distance_basis_form.setRowVisible(self.atr_period,m=="ATR"); self.distance_basis_form.setRowVisible(self.atr_mult,m=="ATR")
            self.distance_basis_form.setRowVisible(self.percent_r,m=="PERCENT"); self.distance_basis_form.setRowVisible(self.fixed_r,m=="FIXED")
        if hasattr(self,"period_form"):
            custom_period=not self.entire_dataset.isChecked()
            self.period_form.setRowVisible(self.trading_start,custom_period); self.period_form.setRowVisible(self.trading_end,custom_period)
        try:
            r=parse_percentage(self.risk_leg.text()); multiplier=1.0; profile_name="Selected profile"
            if hasattr(self,"profile_editor"):
                profile_name=self.profile_editor.current.replace("_"," ").title()
                multiplier=float(self.profile_editor.profiles[self.profile_editor.current].risk_multiplier)
            planned=r*multiplier; planned_cash=self.equity.value()*planned
            self.risk_warn.setText(f"Base {r*100:.2f}% × {profile_name} {multiplier:g}x = {planned*100:.2f}% account risk (${planned_cash:,.2f} at ${self.equity.value():,.2f} equity)" + (" — warning: exceeds 5%." if planned>0.05 else ""))
        except Exception: pass
        if hasattr(self,"enable_daily_schedule"):
            en=self.enable_daily_schedule.isChecked(); self.daily_entry_time.setEnabled(en); self.daily_entry_timezone.setEnabled(en); self.daily_entry_missed_policy.setEnabled(en); self.next_entry_summary.setText(f"Next eligible entry time: {self.daily_entry_time.text() or '00:00'} {self.daily_entry_timezone.text() or 'UTC'}" if en else "Daily schedule disabled; existing entry mode controls entries.")
        if hasattr(self,"enable_trade_telemetry"):
            enabled=self.enable_trade_telemetry.isChecked(); self.telemetry_interval.setEnabled(enabled); self.save_full_telemetry.setEnabled(enabled); self.save_journey_summary.setEnabled(enabled); self.save_journey_charts.setEnabled(enabled)
        try:
            maker=parse_percentage(self.maker.text()); taker=parse_percentage(self.taker.text()); slip=parse_percentage(self.slippage.text())
            entry_fee=maker if self.maker_entry.isChecked() else taker; exit_fee=maker if self.maker_exit.isChecked() else taker
            fees=entry_fee+exit_fee; slippage_cost=slip*2; cost=fees+slippage_cost
            self.cost.setText(f"Estimated cost for one trade (entry + final exit): {format_percentage(cost,4)} of notional — fees {format_percentage(fees,4)} + slippage {format_percentage(slippage_cost,4)}. Partial exits and changing exit value can alter the actual cost.")
        except Exception: self.cost.setText("Invalid execution-cost input")
    def run_backtest(self):
        try: vals=self.values(); cfg=build_backtest_config(vals); cfg=replace(cfg, save_feature_analysis_reports=self.save_feature_reports.isChecked(), save_indicator_analysis_reports=self.save_indicator_reports.isChecked(), create_standard_charts=self.create_standard_charts.isChecked()); Path(vals['output_dir']).mkdir(parents=True,exist_ok=True); cfg=replace(cfg, output_run_dir=planned_run_dir(cfg)); self.planned_output.setText(str(cfg.output_run_dir.resolve()))
        except Exception as e: QMessageBox.warning(self,"Validation Problems",str(e)); return
        if not self.validate_data(): return
        self._run_failed=False; self._pending_ui_results=None; self.output_dir=cfg.output_run_dir; self.thread=QThread(); self.worker=BacktestWorker(cfg, self._validated_strategy_data); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.thread.finished.connect(self._thread_finished); self.worker.status.connect(self.on_status); self.worker.log.connect(self.append_log); self.worker.finished.connect(self.on_finished); self.worker.failed.connect(self.on_failed); self.started=time.time(); self._set_backtest_running(True); self.thread.start()
    def _set_backtest_running(self,running):
        """Keep setup actions consistent with the single-backtest worker state."""
        self.run_btn.setEnabled(not running)
        self.run_btn.setText("Running..." if running else "Run Backtest")
        self.cancel_btn.setEnabled(running)
        self.new_run_btn.setEnabled(not running and not bool(self.portfolio_thread and self.portfolio_thread.isRunning()))
        self.load_btn.setEnabled(not running)
    def on_status(self,s,p): self.status.setText(s); self.progress.setValue(p); self.elapsed.setText(f"Elapsed: {int(time.time()-self.started)}s")
    def on_finished(self,summary,trades,equity,out):
        self.last_summary=summary; self.output_dir=Path(out); self.completed_run_dir=self.output_dir; self._pending_ui_results=(summary,trades,equity,out)
        self._refresh_report_buttons()
        self.cleanup_thread()
        # The backtest and all output files are complete at this point.  Do not
        # hold the run at 99% while optional GUI views are being refreshed.
        self.progress.setValue(100); self.status.setText(f"Completed | Results saved to {self.output_dir}"); self._play_completion_sound()
        QTimer.singleShot(0,self._finish_summary_view)

    @staticmethod
    def _play_completion_sound():
        """Play the operating system's standard notification sound."""
        try:
            QApplication.beep()
        except Exception:
            # Audio availability must never turn a successful backtest into an error.
            pass
    def _finish_summary_view(self):
        if self._pending_ui_results is None: return
        summary,trades,_,_=self._pending_ui_results
        try:
            self.populate_summary(summary,trades)
        except Exception:
            self.append_log("Results display warning (summary):\n"+traceback.format_exc())
        try:
            self._update_sr_summary_panel(trades)
        except Exception:
            self.append_log("Results display warning (S/R summary):\n"+traceback.format_exc())
        QTimer.singleShot(0,self._finish_results_view)
    def _finish_results_view(self):
        if self._pending_ui_results is None: return
        try:
            self.update_dynamic()
        except Exception:
            self.append_log("Results display warning (chart/window):\n"+traceback.format_exc())
        finally:
            self._pending_ui_results=None
            self.progress.setValue(100); self.status.setText(f"Completed | Results saved to {self.output_dir}")
            if self.thread is None: self.run_btn.setEnabled(True)
    def on_failed(self,msg,tb): self._run_failed=True; QMessageBox.critical(self,"Backtest Error",msg); self.append_log(tb); self.cleanup_thread()
    def cleanup_thread(self):
        self.cancel_btn.setEnabled(False)
        if self.thread is None:
            self.run_btn.setEnabled(True)
            return
        self.thread.quit()

    def _thread_finished(self):
        thread=self.thread
        self.thread=None; self.worker=None
        self._set_backtest_running(False)
        if self._pending_ui_results is None:
            self.run_btn.setEnabled(True)
            if self._run_failed: self.status.setText("Backtest failed; see the Log tab for details.")
        if thread is not None: thread.deleteLater()
    def _update_sr_summary_panel(self,trades):
        mode_raw=self.sr_filter_mode.currentText() if hasattr(self,"sr_filter_mode") else "ANALYSIS_ONLY"
        mode={
            "ANALYSIS_ONLY":"Analysis Only",
            "APPLY_ENTRY_RULES":"Apply Entry Rules",
        }.get(mode_raw,mode_raw)
        mode_note="\nNo trades filtered; support/resistance was measured for analysis only." if mode == "Analysis Only" else ""
        if trades is None or trades.empty or ("long_sr_context" not in trades.columns and "short_sr_context" not in trades.columns):
            self.sr_summary_panel_label.setText(f"Mode: {mode}{mode_note}\n\nSupport/Resistance analysis was not enabled for this run.")
            return
        report=build_sr_event_context_summary(trades)
        candidates=report[report["trade_count"]>=3] if not report.empty else report
        if candidates.empty:
            self.sr_summary_panel_label.setText(f"Mode: {mode}{mode_note}\n\nNot enough trades per S/R context yet for a reliable best/weakest comparison (need at least 3 per context).")
            return
        ranked=candidates.dropna(subset=["win_rate"])
        best=ranked.sort_values("win_rate",ascending=False).iloc[0]
        worst=ranked.sort_values("win_rate",ascending=True).iloc[0]
        def describe(row):
            context=str(row["context"]).replace("_"," ").title()
            r_value=row.get("avg_r")
            r_text=f"{r_value:+.2f}R" if r_value == r_value else "n/a"
            return f"{row['direction']} {context} — {format_percentage(row['win_rate'],1)} WR, {r_text}"
        self.sr_summary_panel_label.setText(
            f"Mode: {mode}{mode_note}\n\nBest S/R Context: {describe(best)}\nWeakest S/R Context: {describe(worst)}"
        )

    @staticmethod
    def _money(value):
        return "—" if value is None else f"{float(value):,.2f}"

    def populate_summary(self,s,trades=None):
        percentage=lambda value: "—" if value is None else format_percentage(value,2)
        number=lambda value: "—" if value is None else f"{float(value):.2f}"
        r_value=lambda value: "—" if value is None else f"{float(value):.2f}R"
        values={
            "Ending Equity":self._money(s.get("ending_equity")), "Total Return":percentage(s.get("total_return_percentage")),
            "Total Trades":f"{int(s.get('total_trades',s.get('total_pairs',0)) or 0):,}", "Win Rate":percentage(s.get("win_rate")),
            "Profit Factor":number(s.get("profit_factor")), "Maximum Drawdown":percentage(s.get("maximum_drawdown_percentage")),
            "Average Net R":r_value(s.get("average_net_r")), "Total Net R":r_value(s.get("total_net_r")),
            "Total Fees":self._money(s.get("total_fees")),
            "Signals Traded / Evaluated":f"{int(s.get('signals_traded',0)):,} / {int(s.get('signals_evaluated',0)):,}" if s.get("signals_traded") is not None or s.get("signals_evaluated") is not None else "—",
        }
        for label,text in values.items(): self.kpi_labels[label].setText(text)
        self.starting_equity_label.setText(f"Starting Equity: {self._money(self.equity.value())}")
        profiles=s.get("isolated_profile_comparison",[])
        if profiles:
            self.comparison_box.setTitle("Profile Performance")
            self.combo_table.setColumnCount(5)
            self.combo_table.setHorizontalHeaderLabels(["Profile","Trades","Win Rate","Profit Factor","Net Profit"]); self.combo_table.setRowCount(len(profiles))
            for row,item in enumerate(profiles):
                values=[item.get("profile","").replace("_"," ").title(),item.get("trades",0),format_percentage(item.get("win_rate",0),2),f"{float(item.get('profit_factor',0)):.2f}",self._money(item.get("net_profit"))]
                for column,value in enumerate(values): self.combo_table.setItem(row,column,QTableWidgetItem(str(value)))
        else:
            self.comparison_box.setTitle("Direction / Regime Performance")
            self.combo_table.setColumnCount(9); self.combo_table.setHorizontalHeaderLabels(["Market Regime","Direction","Trades","Wins","Losses","Win Rate","Average Net R","Total Net R","Net PnL"])
            _,breakdown=build_performance_breakdowns(trades if trades is not None else pd.DataFrame())
            lookup={(str(item.market_regime).upper(),str(item.direction).upper()):item for item in breakdown.itertuples()}
            rows=[(regime,direction) for regime in ("BULL","BEAR","SIDEWAYS") for direction in ("LONG","SHORT")]; self.combo_table.setRowCount(len(rows))
            for row,(regime,direction) in enumerate(rows):
                item=lookup.get((regime,direction)); raw=[regime.title(),direction.title(),getattr(item,"trades",0),getattr(item,"wins",0),getattr(item,"losses",0),format_percentage(getattr(item,"win_rate",0),2),f"{getattr(item,'average_r',0):.2f}R",f"{getattr(item,'total_r',0):.2f}R",self._money(getattr(item,"net_pnl",0))]
                for column,text in enumerate(raw): self.combo_table.setItem(row,column,QTableWidgetItem(str(text)))
        self.combo_table.resizeRowsToContents(); self._size_summary_table()
    def save_config(self):
        p,_=QFileDialog.getSaveFileName(self,"Save Configuration","backtest_config.json","JSON (*.json)");
        if p:
            values=self.values(); values.update({"save_feature_analysis_reports":self.save_feature_reports.isChecked(),"save_indicator_analysis_reports":self.save_indicator_reports.isChecked(),"create_standard_charts":self.create_standard_charts.isChecked()}); save_config_json(p,values)
    def load_config(self):
        if self.thread and self.thread.isRunning(): return
        p,_=QFileDialog.getOpenFileName(self,"Load Configuration","","JSON (*.json)");
        if p: self.apply_values(load_config_json(p))
    def apply_values(self,d):
        self._applying_values=True
        values = {**default_gui_config(), **d}
        self.profile_editor.apply_values(values)
        self.market_symbol.setCurrentText(str(values.get("market_symbol") or self._pair_from_path(values.get("input_csv","")) or "XRPUSDT"))
        self.run_name.setText(str(values.get("run_name", "")))
        self.strategy_timeframe.setCurrentText(self._timeframe_label(int(values["strategy_timeframe_minutes"])))
        self.intrabar_timeframe.setCurrentText(self._timeframe_label(int(values["intrabar_timeframe_minutes"])))
        self.input_csv.setText(str(values["input_csv"]))
        self.intrabar_csv.setText(str(values["intrabar_csv"]))
        self.use_intrabar.setChecked(bool(values["use_intrabar_data"]) and int(values["strategy_timeframe_minutes"]) > 1)
        self.output_folder.setText(str(values["output_dir"]))
        self.entry_mode.setCurrentIndex(max(0,self.entry_mode.findData(values["entry_mode"])))
        self.entry_interval.setValue(int(values["entry_interval"]))
        self.enable_support_resistance_analysis.setChecked(bool(values.get("enable_support_resistance_analysis",False))); self.sr_pivot_left.setValue(int(values.get("sr_pivot_left",5))); self.sr_pivot_right.setValue(int(values.get("sr_pivot_right",5))); self.sr_lookback_bars.setValue(int(values.get("sr_lookback_bars",200))); self.sr_zone_width_atr.setValue(float(values.get("sr_zone_width_atr",0.5))); self.sr_near_distance_atr.setValue(float(values.get("sr_near_distance_atr",0.75))); self.enable_sr_hold_confirmation.setChecked(bool(values.get("enable_sr_hold_confirmation",False))); self.sr_hold_confirmation_bars.setValue(int(values.get("sr_hold_confirmation_bars",3))); self.sr_hold_confirmation_atr.setValue(float(values.get("sr_hold_confirmation_atr",0.25))); self.sr_break_tolerance_atr.setValue(float(values.get("sr_break_tolerance_atr",0.25))); self.sr_break_basis.setCurrentText(str(values.get("sr_break_basis","CLOSE"))); self.sr_filter_mode.setCurrentIndex(max(0,self.sr_filter_mode.findData(str(values.get("sr_filter_mode","ANALYSIS_ONLY")))))
        self.sr_long_avoid_near_resistance.setChecked(bool(values.get("sr_long_avoid_near_resistance",False))); self.sr_long_require_near_support.setChecked(bool(values.get("sr_long_require_near_support",False))); self.sr_long_block_broken_support.setChecked(bool(values.get("sr_long_block_broken_support",False))); self.sr_long_min_room_to_resistance_atr.setValue(float(values.get("sr_long_min_room_to_resistance_atr",0.0))); self.sr_short_avoid_near_support.setChecked(bool(values.get("sr_short_avoid_near_support",False))); self.sr_short_require_near_resistance.setChecked(bool(values.get("sr_short_require_near_resistance",False))); self.sr_short_block_broken_resistance.setChecked(bool(values.get("sr_short_block_broken_resistance",False))); self.sr_short_min_room_to_support_atr.setValue(float(values.get("sr_short_min_room_to_support_atr",0.0)))
        if hasattr(self,"sr_detection_preset"): self._sync_sr_preset_from_values()
        self.enable_di_direction_selection.setChecked(bool(values.get("enable_di_direction_selection",True))); self.enable_di_pressure_analysis.setChecked(bool(values.get("enable_di_pressure_analysis",True))); self.di_pressure_lookback.setValue(int(values.get("di_pressure_lookback",3))); self.enable_mean_reversion_analysis.setChecked(bool(values.get("enable_mean_reversion_analysis",True))); self.mean_reversion_period.setValue(int(values.get("mean_reversion_period",20)))
        self.enable_daily_schedule.setChecked(bool(values.get("enable_daily_entry_schedule", False)))
        self.daily_entry_time.setText(str(values.get("daily_entry_time", "00:00")))
        self.daily_entry_timezone.setText(str(values.get("daily_entry_timezone", "UTC")))
        self.daily_entry_missed_policy.setCurrentText(str(values.get("daily_entry_missed_policy", "SKIP_DAY")))
        self.max_pairs.setValue(int(values["max_active_pairs"]))
        self.tie.setCurrentIndex(max(0,self.tie.findData(values["tie_policy"])))
        self.risk_mode.setCurrentText(values["risk_mode"])
        self.atr_period.setValue(int(values["atr_period"]))
        self.atr_mult.setValue(float(values["atr_multiplier"]))
        self.trading_start.setText(str(values["trading_start_date"] or ""))
        self.trading_end.setText(str(values["trading_end_date"] or ""))
        self.entire_dataset.setChecked(not values["trading_start_date"] and not values["trading_end_date"])
        self.max_lev_leg.setText(str(values["max_effective_leverage_per_leg"] or ""))
        self.max_lev_combined.setText(str(values["max_combined_effective_leverage"] or ""))
        self.missing_policy.setCurrentText(values["intrabar_missing_policy"])
        self.zero_cost.setChecked(bool(values["zero_cost_comparison"]))
        self.percent_r.setText(format_percentage(float(values["percent_r"])))
        self.fixed_r.setValue(float(values["fixed_r"]))
        self.equity.setValue(float(values["initial_equity"]))
        self.risk_leg.setText(format_percentage(float(values["risk_per_leg"])))
        self.maker.setText(format_percentage(float(values["maker_fee"])))
        self.taker.setText(format_percentage(float(values["taker_fee"])))
        self.maker_entry.setChecked(bool(values["use_maker_entry"]))
        self.maker_exit.setChecked(bool(values["use_maker_exit"]))
        self.slippage.setText(format_percentage(float(values["slippage"])))
        self._shared_adx_period = int(values.get("adx_period", 14))
        level=str(values.get("analysis_level","STANDARD")).upper(); self.analysis_level.setCurrentText({"FAST":"Fast","RESEARCH":"Research"}.get(level,"Standard (Recommended)"))
        self.enable_trade_telemetry.setChecked(bool(values.get("enable_trade_telemetry", False))); self.telemetry_interval.setValue(int(values.get("telemetry_interval_minutes", 15))); self.save_full_telemetry.setChecked(bool(values.get("save_full_telemetry_csv", False))); self.save_journey_summary.setChecked(bool(values.get("save_trade_journey_summary", False))); self.save_journey_charts.setChecked(bool(values.get("save_trade_journey_charts", False)))
        self.enable_lifecycle.setChecked(bool(values.get("enable_indicator_lifecycle_analysis",False))); self.lifecycle_phases.setValue(int(values.get("lifecycle_phases",4))); self.lifecycle_checkpoints.setText(",".join(str(v) for v in values.get("lifecycle_early_checkpoints",[15,30,60]))); self.lifecycle_min_sample.setValue(int(values.get("lifecycle_minimum_bucket_sample",20))); self.lifecycle_charts.setChecked(bool(values.get("create_lifecycle_charts",False))); self.lifecycle_flat_threshold.setValue(float(values.get("lifecycle_flat_pattern_threshold_pct",5.0)))
        self.save_feature_reports.setChecked(bool(values.get("save_feature_analysis_reports",False))); self.save_indicator_reports.setChecked(bool(values.get("save_indicator_analysis_reports",False))); self.create_standard_charts.setChecked(bool(values.get("create_standard_charts",False)))
        self._applying_values=False; self.update_dynamic()
        self.update_planned_output()
    def append_log(self,t): self.log.append(str(t))
    def save_log(self):
        p,_=QFileDialog.getSaveFileName(self,"Save Log","backtest.log","Log (*.log *.txt)");
        if p: Path(p).write_text(self.log.toPlainText())
