"""Six-profile overview with one reusable strategy-profile editor."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from PySide6.QtCore import Signal
from PySide6.QtWidgets import *
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, StrategyProfile, default_profiles, profiles_to_dict, normalize_profiles

LABELS={k:k.replace("_"," ").title() for k in PROFILE_KEYS}

class StrategyProfilesWidget(QWidget):
    changed=Signal()
    def __init__(self):
        super().__init__(); self.profiles=default_profiles(); self.current="bull_long"; self._loading=False
        root=QVBoxLayout(self); top=QHBoxLayout(); self.mode=QComboBox(); self.mode.addItem("Combined shared account","COMBINED_SHARED_CAPITAL"); self.mode.addItem("Each profile separately","ISOLATED_PROFILES"); self.mode.addItem("Combined + separate comparison","BOTH"); top.addWidget(QLabel("Test mode")); top.addWidget(self.mode); top.addStretch(); root.addLayout(top)
        definition=QGroupBox("Market Regime Definition"); definition_form=QFormLayout(definition)
        self.regime_method=QComboBox(); self.regime_method.addItem("BTC structural trend (market-wide)","BTC_STRUCTURAL"); self.regime_method.addItem("Selected asset structural trend","ASSET_STRUCTURAL"); self.regime_method.addItem("Asset trailing return","ASSET_RETURN")
        self.structural_sma_days=QSpinBox(); self.structural_sma_days.setRange(2,3650); self.structural_sma_days.setSuffix(" days"); self.structural_sma_days.setValue(200)
        self.structural_slope_days=QSpinBox(); self.structural_slope_days.setRange(1,3650); self.structural_slope_days.setSuffix(" days"); self.structural_slope_days.setValue(30)
        self.regime_lookback=QSpinBox(); self.regime_lookback.setRange(1,3650); self.regime_lookback.setSuffix(" days")
        self.bull_threshold=QDoubleSpinBox(); self.bull_threshold.setRange(-99.99,10000); self.bull_threshold.setSuffix(" %"); self.bull_threshold.setDecimals(2)
        self.adx_period=QSpinBox(); self.adx_period.setRange(1,1000); self.bb_period=QSpinBox(); self.bb_period.setRange(2,1000); self.bb_stddevs=QDoubleSpinBox(); self.bb_stddevs.setRange(.01,20); self.bb_stddevs.setDecimals(2)
        for label,widget in (("Regime method",self.regime_method),("Trend average",self.structural_sma_days),("Average slope",self.structural_slope_days),("Return lookback",self.regime_lookback),("Bull/Bear threshold magnitude",self.bull_threshold),("ADX period",self.adx_period),("Bollinger period",self.bb_period),("Bollinger deviations",self.bb_stddevs)): definition_form.addRow(label,widget)
        self.definition_form=definition_form
        definition.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum); root.addWidget(definition)
        self.mode_help=QLabel(); self.mode_help.setWordWrap(True); self.mode_help.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum); root.addWidget(self.mode_help)
        split=QSplitter(); split.setMinimumHeight(450); self.list=QListWidget(); self.list.setMinimumWidth(280); split.addWidget(self.list); editor=QScrollArea(); editor.setWidgetResizable(True); body=QWidget(); self.editor_layout=QVBoxLayout(body); editor.setWidget(body); split.addWidget(editor); split.setStretchFactor(1,1); root.addWidget(split,1)
        self.controls={}; self.control_forms={}; self.sections={}

        self._section("Profile Settings")
        self._check("enabled","Profile enabled")
        self._check("flip_direction","Flip entry direction (Long ↔ Short)")
        self.controls["flip_direction"].setToolTip("Apply this profile's filters to the original DI signal, then enter in the opposite direction.")
        self._number("risk_multiplier",1,.01,100)
        self.form.labelForField(self.controls["risk_multiplier"]).setText("Profile Risk Multiplier")
        self.controls["risk_multiplier"].setSuffix(" x")
        self.controls["risk_multiplier"].setToolTip("Scales Base Risk Per Trade for this profile. 1.0x uses the base account risk; 0.5x uses half; 2.0x uses double.")

        self._section("Entry Rules")
        self._choice("flip_rule_match_mode",(("Any flip rule (OR)","ANY"),("All flip rules (AND)","ALL")))
        self.form.labelForField(self.controls["flip_rule_match_mode"]).setText("Flip rules match")
        self._choice("reject_rule_match_mode",(("Any reject rule (OR)","ANY"),("All reject rules (AND)","ALL")))
        self.form.labelForField(self.controls["reject_rule_match_mode"]).setText("Reject rules match")
        self._integer("rsi_period",14,1,1000)
        self.form.labelForField(self.controls["rsi_period"]).setText("RSI period")
        self._integer("momentum_lookback_hours",24,1,87600)
        self.form.labelForField(self.controls["momentum_lookback_hours"]).setText("Return lookback")
        self.controls["momentum_lookback_hours"].setSuffix(" hours")
        self.entry_rules_table=QTableWidget(0,5)
        self.entry_rules_table.setHorizontalHeaderLabels(["Action","Indicator","Condition","Minimum","Maximum"])
        self.entry_rules_table.horizontalHeader().setStretchLastSection(True)
        self.entry_rules_table.setMaximumHeight(240)
        self.entry_rules_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rule_buttons=QHBoxLayout()
        self.add_rule_btn=QPushButton("+ Add rule")
        self.remove_rule_btn=QPushButton("Remove selected")
        rule_buttons.addWidget(self.add_rule_btn)
        rule_buttons.addWidget(self.remove_rule_btn)
        rule_buttons.addStretch()
        self.form.addRow("Rules",self.entry_rules_table)
        self.form.addRow("",rule_buttons)
        filter_note=QLabel("Rules are evaluated in this order: Reject, then Flip, then Normal. Choose Inside or Outside for each range. Any means OR; All means AND. Percentage values use decimals: 0.025 means 2.5%.")
        filter_note.setWordWrap(True)
        self.form.addRow(filter_note)

        self._section("Exit Strategy")
        self._subsection("Base Exit","The normal stop and fixed target used when optional exit methods are disabled.")
        self._number("stop_loss_multiple",2,.001,1000)
        self._number("reward_risk_ratio",1,.01,100)
        self.form.labelForField(self.controls["stop_loss_multiple"]).setText("Stop Distance")
        self.form.labelForField(self.controls["reward_risk_ratio"]).setText("Profit Target")
        self.controls["stop_loss_multiple"].setDecimals(3)
        self.controls["stop_loss_multiple"].setSuffix(" distance units")
        self.controls["stop_loss_multiple"].setToolTip("Distance from entry measured in the Backtest Setup distance unit. With ATR basis, 2.0 means 2 × the configured ATR volatility unit.")
        self.controls["reward_risk_ratio"].setSuffix(" x stop (R)")
        self.controls["reward_risk_ratio"].setToolTip("Fixed final target as a multiple of the full initial stop distance. 1R equals the initial full stop distance. Disabled while partial take-profit is active.")

        self._subsection("Stop Loss","Optional staged loss handling.")
        self._check("partial_stop_enabled","Use partial stop-loss")
        self._number("sl1_r",.5,.001,1000)
        self._number("sl1_close_pct",50,.01,99.99)
        self._number("sl2_r",2,.001,1000)
        for key,label in (("sl1_r","First stop distance"),("sl1_close_pct","Position closed at first stop"),("sl2_r","Final stop distance")):
            self.form.labelForField(self.controls[key]).setText(label)
        for key in ("sl1_r","sl2_r"):
            self.controls[key].setDecimals(3)
            self.controls[key].setSuffix(" distance units")
        self.controls["sl1_close_pct"].setDecimals(2)
        self.controls["sl1_close_pct"].setSuffix(" %")

        self._subsection("Profit Taking","Profit targets use trade R, where 1R equals the full initial stop distance.")
        self._check("partial_profit_enabled","Use partial take-profit")
        self._number("tp1_r",1,.001,1000)
        self._number("tp1_close_pct",50,.01,99.99)
        self._number("tp2_r",2,.001,1000)
        for key,label in (("tp1_r","First profit target"),("tp1_close_pct","Position closed at first target"),("tp2_r","Final profit target")):
            self.form.labelForField(self.controls[key]).setText(label)
        for key in ("tp1_r","tp2_r"):
            self.controls[key].setDecimals(3)
            self.controls[key].setSuffix(" R")
        self.controls["tp1_close_pct"].setDecimals(2)
        self.controls["tp1_close_pct"].setSuffix(" %")
        self.controls["tp1_close_pct"].setToolTip("The remaining position exits at the final profit target. TP1 and TP2 are measured in trade R (1R = full initial stop distance).")

        self._subsection("Protection","Protection thresholds use trade R, where 1R equals the full initial stop distance.")
        self._check("break_even_enabled","Break-even protection")
        self._number("break_even_activation_r",1,.001,1000)
        self._number("break_even_offset_r",0,-1000,1000)
        self._check("trailing_enabled","Trailing stop")
        self._number("trailing_activation_r",3,.001,1000)
        self._number("trailing_distance_r",1,.001,1000)
        for key,label in (("break_even_activation_r","Break-even activates at"),("break_even_offset_r","Break-even profit lock"),("trailing_activation_r","Trailing activates at"),("trailing_distance_r","Trailing distance")):
            self.form.labelForField(self.controls[key]).setText(label)
        for key in ("trailing_activation_r","trailing_distance_r","break_even_activation_r","break_even_offset_r"):
            self.controls[key].setDecimals(3)
            self.controls[key].setSuffix(" R")

        self._subsection("Time Exit","Optional maximum duration for a trade.")
        self._check("timeout_enabled","Use maximum holding time")
        self._integer("timeout_minutes",480,1,1000000)
        self.form.labelForField(self.controls["timeout_minutes"]).setText("Maximum holding time")
        self.controls["timeout_minutes"].setSuffix(" min")
        self.controls["timeout_minutes"].setToolTip("480 minutes equals 8 hours.")

        self._subsection("Advanced Profit Management","Advanced alternatives/extensions. Details stay hidden until the feature is enabled.")
        self._check("r_step_trailing_enabled","R-step staircase / runner")
        self.controls["r_step_trailing_enabled"].setToolTip("Cannot be combined with partial take-profit, trailing stop, or ATR checkpoint extension.")
        self._number("r_step_activation_r",2,.001,1000)
        self._number("r_step_distance_r",2,.001,1000)
        self._number("r_step_size_r",1,.001,1000)
        self._number("r_step_maximum_r",0,0,1000)
        self._number("r_step_activation_close_pct",0,0,99.99)
        self._check("atr_checkpoint_tp_extension_enabled","Distance-unit checkpoint TP extension")
        self.controls["atr_checkpoint_tp_extension_enabled"].setToolTip("Advanced extension measured in the configured price-distance unit, not trade R. Extends profit management when DI and Bollinger checkpoint conditions are met.")
        self._number("atr_checkpoint_di_spread_minimum",30,0,1000)
        self._number("atr_checkpoint_bb_width_minimum",.03,0,1000)
        self._number("atr_checkpoint_profit_lock_start",3,.001,1000)
        self._number("atr_checkpoint_profit_lock_distance",1,.001,1000)
        for key,label in (("r_step_activation_r","Staircase activation"),("r_step_distance_r","Stop distance behind checkpoint"),("r_step_size_r","Checkpoint step"),("r_step_maximum_r","Maximum target (0 = runner)"),("r_step_activation_close_pct","Close at activation"),("atr_checkpoint_di_spread_minimum","Checkpoint DI spread minimum"),("atr_checkpoint_bb_width_minimum","Checkpoint BB width minimum"),("atr_checkpoint_profit_lock_start","Profit lock starts at"),("atr_checkpoint_profit_lock_distance","Profit lock distance")):
            self.form.labelForField(self.controls[key]).setText(label)
        for key in ("r_step_activation_r","r_step_distance_r","r_step_size_r","r_step_maximum_r"):
            self.controls[key].setSuffix(" R")
        for key in ("atr_checkpoint_profit_lock_start","atr_checkpoint_profit_lock_distance"):
            self.controls[key].setSuffix(" distance units")
        self.controls["r_step_activation_close_pct"].setSuffix(" %")
        self.controls["atr_checkpoint_bb_width_minimum"].setDecimals(6)

        self._section("Profile Actions")
        buttons=QHBoxLayout()
        copy_btn=QPushButton("Copy Profile")
        paste_btn=QPushButton("Paste Profile")
        reset_btn=QPushButton("Reset Profile")
        copy_strategy_btn=QPushButton("Apply Strategy to All Profiles")
        buttons.addWidget(copy_btn)
        buttons.addWidget(paste_btn)
        buttons.addWidget(reset_btn)
        buttons.addWidget(copy_strategy_btn)
        self.form.addRow(buttons)
        self.editor_layout.addStretch()
        self.clipboard=None
        copy_btn.clicked.connect(lambda:setattr(self,"clipboard",deepcopy(self.profiles[self.current]))); paste_btn.clicked.connect(self._paste); reset_btn.clicked.connect(self._reset); copy_strategy_btn.clicked.connect(self._apply_strategy_to_all); self.list.currentRowChanged.connect(self._select); self.mode.currentTextChanged.connect(self.changed); self.mode.currentIndexChanged.connect(self._update_mode_help)
        for widget in (self.regime_lookback,self.bull_threshold,self.structural_sma_days,self.structural_slope_days,self.adx_period,self.bb_period,self.bb_stddevs): widget.valueChanged.connect(self.changed)
        self.regime_method.currentIndexChanged.connect(self._update_regime_controls); self.regime_method.currentIndexChanged.connect(self.changed)
        for key in ("partial_stop_enabled","partial_profit_enabled","trailing_enabled","break_even_enabled","timeout_enabled","r_step_trailing_enabled","atr_checkpoint_tp_extension_enabled"): self.controls[key].toggled.connect(self._update_management_controls)
        self.add_rule_btn.clicked.connect(self._add_entry_rule); self.remove_rule_btn.clicked.connect(self._remove_entry_rule)
        self._update_mode_help(); self._update_regime_controls(); self._refresh_list(); self.list.setCurrentRow(0)
    def _update_regime_controls(self,*_):
        structural=self.regime_method.currentData() in ("BTC_STRUCTURAL","ASSET_STRUCTURAL")
        for widget in (self.structural_sma_days,self.structural_slope_days): self.definition_form.setRowVisible(widget,structural)
        for widget in (self.regime_lookback,self.bull_threshold): self.definition_form.setRowVisible(widget,not structural)
    def _update_mode_help(self,*_):
        text={
            "COMBINED_SHARED_CAPITAL":"Runs all enabled profiles together once using one shared account. Open trades can block opportunities from other profiles.",
            "ISOLATED_PROFILES":"Runs each enabled profile independently. Every profile starts with the full configured starting equity; no shared-account run is performed.",
            "BOTH":"Runs the shared account first, then every enabled profile independently. Reports performance comparisons and opportunities blocked in the shared-account run.",
        }
        self.mode_help.setText(text.get(self.mode.currentData(),""))
    def _section(self,title):
        box=QGroupBox(title); self.form=QFormLayout(box); self.editor_layout.addWidget(box); self.sections[title]=box; return box
    def _subsection(self,title,help_text=""):
        label=QLabel(title); font=label.font(); font.setBold(True); label.setFont(font)
        if help_text:
            label.setToolTip(help_text)
        self.form.addRow(label)
        if help_text:
            note=QLabel(help_text); note.setWordWrap(True); note.setStyleSheet("color: palette(mid);")
            self.form.addRow(note)
    def _check(self,key,label): w=QCheckBox(label); self.controls[key]=w; self.control_forms[key]=self.form; self.form.addRow(w); w.toggled.connect(self._store)
    def _number(self,key,value,lo,hi): w=QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(3); w.setValue(value); self.controls[key]=w; self.control_forms[key]=self.form; self.form.addRow(key.replace('_',' ').title(),w); w.valueChanged.connect(self._store)
    def _integer(self,key,value,lo,hi): w=QSpinBox(); w.setRange(lo,hi); w.setValue(value); self.controls[key]=w; self.control_forms[key]=self.form; self.form.addRow(key.replace('_',' ').title(),w); w.valueChanged.connect(self._store)
    def _choice(self,key,values):
        w=QComboBox()
        for value in values:
            label,data=value if isinstance(value,tuple) else (value,value); w.addItem(label,data)
        self.controls[key]=w; self.control_forms[key]=self.form; self.form.addRow(key.replace('_',' ').title(),w); w.currentIndexChanged.connect(self._store)
    def _range(self,prefix,title,maximum,percent):
        self._check(prefix+"_enabled",f"Enable {title}"); self._number(prefix+"_minimum",0,-maximum,maximum); self._number(prefix+"_maximum",maximum,-maximum,maximum)
        self.form.labelForField(self.controls[prefix+"_minimum"]).setText(f"{title} minimum")
        self.form.labelForField(self.controls[prefix+"_maximum"]).setText(f"{title} maximum")
        if percent:
            self.controls[prefix+"_minimum"].setSuffix("  (decimal)"); self.controls[prefix+"_maximum"].setSuffix("  (decimal)")
    def _values(self):
        return {k:(w.isChecked() if isinstance(w,QCheckBox) else (w.currentData() if isinstance(w,QComboBox) else w.value())) for k,w in self.controls.items()}
    def _store(self,*_):
        if self._loading:return
        values=self._values(); values["entry_rules"]=tuple(self._entry_rules_values()); self.profiles[self.current]=StrategyProfile(**values); self._refresh_list(); self.changed.emit()
    def _entry_rules_values(self):
        rules=[]
        for row in range(self.entry_rules_table.rowCount()):
            action=self.entry_rules_table.cellWidget(row,0); indicator=self.entry_rules_table.cellWidget(row,1); condition=self.entry_rules_table.cellWidget(row,2); minimum=self.entry_rules_table.cellWidget(row,3); maximum=self.entry_rules_table.cellWidget(row,4)
            rules.append({"action":action.currentData(),"indicator":indicator.currentData(),"condition":condition.currentData(),"minimum":minimum.value(),"maximum":maximum.value()})
        return rules
    def _add_entry_rule(self,checked=False,rule=None):
        rule=rule or {"action":"FLIP","indicator":"CLOSE_LOCATION","condition":"INSIDE","minimum":0.0,"maximum":1.0}; row=self.entry_rules_table.rowCount(); self.entry_rules_table.insertRow(row)
        action=QComboBox(); action.addItem("Flip","FLIP"); action.addItem("Reject","REJECT")
        indicator=QComboBox(); condition=QComboBox(); condition.addItem("Inside range","INSIDE"); condition.addItem("Outside range","OUTSIDE")
        for label,data in (("DI Spread","DI_SPREAD"),("ADX","ADX"),("ATR %","ATR_PCT"),("RSI","RSI"),("BB Width","BB_WIDTH"),("Close Location","CLOSE_LOCATION"),("Trailing Return","MOMENTUM"),("VWAP Distance (ATR)","VWAP_DISTANCE")): indicator.addItem(label,data)
        action.setCurrentIndex(max(0,action.findData(str(rule.get("action","FLIP")))))
        indicator.setCurrentIndex(max(0,indicator.findData(str(rule.get("indicator","CLOSE_LOCATION")))))
        condition.setCurrentIndex(max(0,condition.findData(str(rule.get("condition","INSIDE")))))
        minimum=QDoubleSpinBox(); maximum=QDoubleSpinBox()
        for widget in (minimum,maximum): widget.setRange(-1000,1000); widget.setDecimals(6)
        minimum.setValue(float(rule.get("minimum",0))); maximum.setValue(float(rule.get("maximum",1)))
        for column,widget in enumerate((action,indicator,condition,minimum,maximum)): self.entry_rules_table.setCellWidget(row,column,widget)
        action.currentIndexChanged.connect(self._store); indicator.currentIndexChanged.connect(self._store); condition.currentIndexChanged.connect(self._store); minimum.valueChanged.connect(self._store); maximum.valueChanged.connect(self._store)
        self.entry_rules_table.selectRow(row); self._store()
    def _remove_entry_rule(self):
        row=self.entry_rules_table.currentRow()
        if row<0 and self.entry_rules_table.rowCount(): row=self.entry_rules_table.rowCount()-1
        if row>=0: self.entry_rules_table.removeRow(row); self._store()
    def _load_entry_rules(self,rules):
        self.entry_rules_table.setRowCount(0)
        for rule in rules: self._add_entry_rule(rule=rule)
    def _update_management_controls(self,*_):
        partial_stop=self.controls["partial_stop_enabled"].isChecked()
        for key in ("sl1_r","sl1_close_pct","sl2_r"):
            self._show_control(key,partial_stop)

        partial_profit=self.controls["partial_profit_enabled"].isChecked()
        for key in ("tp1_r","tp1_close_pct","tp2_r"):
            self._show_control(key,partial_profit)
        self.controls["reward_risk_ratio"].setEnabled(not partial_profit)

        break_even=self.controls["break_even_enabled"].isChecked()
        for key in ("break_even_activation_r","break_even_offset_r"):
            self._show_control(key,break_even)

        trailing=self.controls["trailing_enabled"].isChecked()
        for key in ("trailing_activation_r","trailing_distance_r"):
            self._show_control(key,trailing)

        timeout=self.controls["timeout_enabled"].isChecked()
        self._show_control("timeout_minutes",timeout)

        r_step=self.controls["r_step_trailing_enabled"].isChecked()
        for key in ("r_step_activation_r","r_step_distance_r","r_step_size_r","r_step_maximum_r","r_step_activation_close_pct"):
            self._show_control(key,r_step)

        atr_extension=self.controls["atr_checkpoint_tp_extension_enabled"].isChecked()
        for key in ("atr_checkpoint_di_spread_minimum","atr_checkpoint_bb_width_minimum","atr_checkpoint_profit_lock_start","atr_checkpoint_profit_lock_distance"):
            self._show_control(key,atr_extension)

        # Match StrategyProfile.validate(): R-step is mutually exclusive with
        # partial take-profit, trailing stop, and ATR checkpoint extension.
        r_step_conflict=partial_profit or trailing or atr_extension
        self.controls["r_step_trailing_enabled"].setEnabled(r_step or not r_step_conflict)
        for key in ("partial_profit_enabled","trailing_enabled","atr_checkpoint_tp_extension_enabled"):
            self.controls[key].setEnabled(not r_step)
    def _show_control(self,key,visible):
        control=self.controls[key]; control.setEnabled(visible); self.control_forms[key].setRowVisible(control,visible)
    def _load(self):
        self._loading=True; p=self.profiles[self.current]
        for k,w in self.controls.items():
            v=getattr(p,k)
            if isinstance(w,QCheckBox): w.setChecked(v)
            elif isinstance(w,QComboBox): w.setCurrentIndex(max(0,w.findData(str(v))))
            else: w.setValue(v)
        self._load_entry_rules(p.entry_rules); self._loading=False; self._update_management_controls()
    def _select(self,row):
        if row<0:return
        self.current=PROFILE_KEYS[row]; self._load()
    def _refresh_list(self):
        row=self.list.currentRow(); self.list.blockSignals(True); self.list.clear()
        for k in PROFILE_KEYS:
            p=self.profiles[k]; flips=sum(r.get("action")=="FLIP" for r in p.entry_rules); rejects=sum(r.get("action")=="REJECT" for r in p.entry_rules); self.list.addItem(f"{'✓' if p.enabled else '○'} {LABELS[k]}   {p.reward_risk_ratio:g}:1   {flips} flip / {rejects} reject")
        self.list.setCurrentRow(max(0,row)); self.list.blockSignals(False)
    def _paste(self):
        if self.clipboard is not None:self.profiles[self.current]=deepcopy(self.clipboard);self._load();self._refresh_list();self.changed.emit()
    def _reset(self): self.profiles[self.current]=StrategyProfile();self._load();self._refresh_list();self.changed.emit()
    def _apply_strategy_to_all(self):
        """Copy the current profile's complete baseline strategy to every profile.

        Profile identity controls stay profile-specific: enabled and flip_direction
        are preserved for each target profile. Every other StrategyProfile field is
        copied from the current profile, so new strategy fields are included
        automatically without maintaining a separate allow-list.
        """
        source=deepcopy(self.profiles[self.current])
        for key in PROFILE_KEYS:
            if key==self.current:
                continue
            target=self.profiles[key]
            self.profiles[key]=replace(source, enabled=target.enabled, flip_direction=target.flip_direction)
        self._refresh_list();self.changed.emit()
    def values(self): return {"strategy_profile_run_mode":self.mode.currentData(),"market_regime_method":self.regime_method.currentData(),"structural_regime_sma_days":self.structural_sma_days.value(),"structural_regime_slope_lookback_days":self.structural_slope_days.value(),"bull_regime_lookback_days":self.regime_lookback.value(),"bull_regime_return_threshold":self.bull_threshold.value()/100.0,"adx_period":self.adx_period.value(),"bb_period":self.bb_period.value(),"bb_stddevs":self.bb_stddevs.value(),"strategy_profiles":profiles_to_dict(self.profiles)}
    def apply_values(self,values):
        self.profiles=normalize_profiles(values.get("strategy_profiles",{})); index=self.mode.findData(str(values.get("strategy_profile_run_mode","COMBINED_SHARED_CAPITAL"))); self.mode.setCurrentIndex(max(0,index)); method=self.regime_method.findData(str(values.get("market_regime_method","ASSET_RETURN"))); self.regime_method.setCurrentIndex(max(0,method)); self.structural_sma_days.setValue(int(values.get("structural_regime_sma_days",200))); self.structural_slope_days.setValue(int(values.get("structural_regime_slope_lookback_days",30))); self.regime_lookback.setValue(int(values.get("bull_regime_lookback_days",90))); self.bull_threshold.setValue(float(values.get("bull_regime_return_threshold",.20))*100); self.adx_period.setValue(int(values.get("adx_period",14))); self.bb_period.setValue(int(values.get("bb_period",20))); self.bb_stddevs.setValue(float(values.get("bb_stddevs",2))); self._update_regime_controls(); self._load(); self._refresh_list()
