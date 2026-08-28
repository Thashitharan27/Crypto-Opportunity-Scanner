"""Composition-based Task-18 desktop GUI for native v2 research runs."""
from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, QDate, QSettings, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from crypto_strategy_lab.data import MarketKind
from crypto_strategy_lab.data_lake_config import (
    ExecutionConfig, ExecutionProfileConfig, FeatureConfig, PROFILE_KEYS,
    ReportingConfig, ResearchRunConfig, StrategyConfig, StrategyProfileConfig,
)
from crypto_strategy_lab.paths import CACHE_DIR, MARKET_DATA_ROOT, OUTPUT_DIR
from crypto_strategy_lab.strategy_profiles import RULE_INDICATORS
from .chatgpt_connection import ChatGPTIntegrationWidget
from .github_manager import GitHubIntegrationWidget
from .v2_controller import GuiApplicationService, GuiResearchRequest
from .ux_presentation import (
    ENUM_LABELS, PROFILE_LABELS, REPORT_PRESETS, apply_report_preset,
    clone_profile_pair, display_percentage, metadata,
)


STRATEGY_TIMEFRAMES = ("15m", "1h", "4h", "1d")
INTRABAR_TIMEFRAMES = ("1m", "5m", "15m")
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
TIMEFRAME_LABELS = {
    "1m": "1 Minute", "5m": "5 Minutes", "15m": "15 Minutes",
    "1h": "1 Hour", "4h": "4 Hours", "1d": "1 Day",
}

STRATEGY_GROUPS = (
    ("Profiles", ("strategy_profile_run_mode",), None),
    ("Direction", (
        "enable_di_direction_selection", "enable_di_pressure_analysis",
        "di_pressure_allow_expanding", "di_pressure_allow_contracting",
        "di_pressure_allow_mixed",
    ), None),
    ("Entry Filters", (
        "enable_mean_reversion_analysis", "sr_filter_mode",
        "sr_long_avoid_near_resistance", "sr_long_require_near_support",
        "sr_long_block_broken_support", "sr_long_min_room_to_resistance_atr",
        "sr_short_avoid_near_support", "sr_short_require_near_resistance",
        "sr_short_block_broken_resistance", "sr_short_min_room_to_support_atr",
    ), None),
    ("Entry Timing / Schedule", (
        "entry_mode", "entry_interval", "enable_daily_entry_schedule",
        "daily_entry_time", "daily_entry_timezone", "daily_entry_missed_policy",
    ), None),
)

FEATURE_GROUPS = (
    ("Price / Volatility", ("atr_period", "bb_period", "bb_stddevs"), None),
    ("DI", ("adx_period", "di_pressure_lookback"), None),
    ("Mean Reversion", (
        "mean_reversion_period", "mean_reversion_mean_type",
        "mean_reversion_bb_stddevs", "mean_reversion_rsi_period",
        "mean_reversion_rsi_oversold", "mean_reversion_rsi_overbought",
        "mean_reversion_require_reentry", "mean_reversion_track_atr_distance",
        "mean_reversion_track_motion",
    ), None),
    ("Regime", (
        "market_regime_method", "structural_regime_sma_days",
        "structural_regime_slope_lookback_days", "bull_regime_lookback_days",
        "bull_regime_return_threshold",
    ), None),
    ("Support / Resistance", (
        "enable_support_resistance_analysis", "sr_timeframe_minutes",
        "sr_pivot_left", "sr_pivot_right", "sr_lookback_bars",
        "sr_zone_width_atr", "sr_near_distance_atr",
        "enable_sr_hold_confirmation", "sr_hold_confirmation_bars",
        "sr_hold_confirmation_atr", "sr_break_tolerance_atr", "sr_break_basis",
    ), None),
    ("Open Interest", ("oi_zscore_window_days", "oi_zscore_min_samples"), "AUTO WHEN AVAILABLE"),
    ("Funding", (
        "funding_zscore_window_days", "funding_zscore_min_samples", "funding_extreme_zscore",
    ), "AUTO WHEN AVAILABLE"),
    ("Positioning", ("basis_zscore_window_days",), "AUTO WHEN AVAILABLE"),
    ("Taker Flow", ("taker_flow_interval",), "AUTO WHEN AVAILABLE"),
    ("Trade Flow", (
        "trade_flow_enabled", "trade_flow_source", "trade_flow_base_interval",
        "trade_flow_windows", "large_trade_quote_threshold",
    ), None),
    ("Order Book", (
        "order_book_enabled", "order_book_base_interval",
        "book_ticker_max_age_seconds", "book_depth_max_age_seconds",
    ), None),
)

EXECUTION_GROUPS = (
    ("Risk", (
        "initial_equity", "risk_mode", "fixed_r", "percent_r", "atr_multiplier",
        "risk_per_leg", "max_effective_leverage_per_leg",
        "max_combined_effective_leverage", "max_active_pairs",
    ), None),
    ("Take Profit", (
        "sr_take_profit_mode", "sr_take_profit_maximum_r",
        "sr_take_profit_minimum_r", "sr_take_profit_buffer_r",
        "sr_take_profit_no_level_policy",
    ), None),
    ("Fees", (
        "maker_fee", "taker_fee", "use_maker_entry", "use_maker_exit",
        "zero_cost_comparison",
    ), None),
    ("Slippage", ("slippage",), None),
    ("Tie / Same-bar Policy", ("tie_policy",), None),
)

STRATEGY_PROFILE_GROUPS = (
    ("Profile Direction Exception", ("flip_direction",), None),
    ("Advanced Rule Matching", (
        "flip_rule_match_mode", "reject_rule_match_mode",
    ), "These settings control how native FLIP/REJECT exception rules are combined."),
)

EXECUTION_PROFILE_GROUPS = (
    ("Risk", ("risk_multiplier",), None),
    ("Stop Loss", ("stop_loss_multiple",), None),
    ("Take Profit", ("reward_risk_ratio",), None),
    ("Break-even", (
        "break_even_enabled", "break_even_activation_r", "break_even_offset_r",
    ), None),
    ("Trailing", (
        "trailing_enabled", "trailing_activation_r", "trailing_distance_r",
    ), None),
    ("Partials", (
        "partial_stop_enabled", "sl1_r", "sl1_close_pct", "sl2_r",
        "partial_profit_enabled", "tp1_r", "tp1_close_pct", "tp2_r",
    ), None),
    ("Timeout", ("timeout_enabled", "timeout_minutes"), None),
    ("Advanced Profit Management", (
        "r_step_trailing_enabled", "r_step_activation_r", "r_step_distance_r",
        "r_step_size_r", "r_step_maximum_r", "r_step_activation_close_pct",
        "atr_checkpoint_tp_extension_enabled", "atr_checkpoint_di_spread_minimum",
        "atr_checkpoint_bb_width_minimum", "atr_checkpoint_profit_lock_start",
        "atr_checkpoint_profit_lock_distance",
    ), None),
)


def timeframe_minutes(value: str) -> int:
    try:
        return TIMEFRAME_MINUTES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported native timeframe: {value}") from exc


def timeframe_label(minutes: int) -> str:
    reverse = {value: key for key, value in TIMEFRAME_MINUTES.items()}
    try:
        return reverse[int(minutes)]
    except KeyError as exc:
        raise ValueError(f"Unsupported native timeframe minutes: {minutes}") from exc


class TupleEditor(QPlainTextEdit):
    """Structured JSON tuple editor preserving tuple values losslessly."""

    def set_tuple(self, value: tuple) -> None:
        self.setPlainText(json.dumps(list(value), indent=2))

    def tuple_value(self) -> tuple:
        value = json.loads(self.toPlainText() or "[]")
        if not isinstance(value, list):
            raise ValueError("Tuple fields must be represented by a JSON array")
        return tuple(value)


class EntryRuleEditor(QTableWidget):
    """Ordered, lossless native advanced-exception editor."""

    changed = Signal()
    COLUMNS = ("action", "indicator", "condition", "minimum", "maximum")
    INDICATOR_LABELS = {
        "DI_SPREAD": "Trend · DI Spread",
        "ADX": "Trend · ADX",
        "MOMENTUM": "Trend · Momentum",
        "ATR_PCT": "Price / Volatility · ATR %",
        "BB_WIDTH": "Price / Volatility · BB Width",
        "VWAP_DISTANCE": "Price / Volatility · VWAP Distance",
        "CLOSE_LOCATION": "Price / Volatility · Close Location",
        "RSI": "Mean Reversion · RSI",
    }

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLUMNS), parent)
        self.setHorizontalHeaderLabels(("Action", "Evidence", "Condition", "Minimum", "Maximum"))
        self._payloads = []
        self.itemChanged.connect(lambda _item: self.changed.emit())

    def set_tuple(self, rules: tuple) -> None:
        from copy import deepcopy

        self.blockSignals(True)
        try:
            self._payloads = deepcopy(list(rules))
            self.setRowCount(len(rules))
            for row, rule in enumerate(rules):
                for column, key in enumerate(self.COLUMNS):
                    value = rule.get(key, "") if isinstance(rule, dict) else ""
                    if key in ("action", "indicator", "condition"):
                        editor = QComboBox()
                        choices = (
                            {"FLIP": "Flip Direction", "REJECT": "Reject Entry"}
                            if key == "action"
                            else {name: self.INDICATOR_LABELS.get(name, name.replace("_", " ").title()) for name in RULE_INDICATORS}
                            if key == "indicator"
                            else {"INSIDE": "Inside Range", "OUTSIDE": "Outside Range"}
                        )
                        if not value:
                            editor.addItem(f"Select {key}…", "")
                        for native, label in choices.items():
                            editor.addItem(label, native)
                        index = editor.findData(value)
                        if index < 0 and value:
                            editor.addItem(str(value), value)
                            index = editor.count() - 1
                        editor.setCurrentIndex(max(index, 0))
                        editor.currentIndexChanged.connect(lambda _index: self.changed.emit())
                        self.setCellWidget(row, column, editor)
                    else:
                        self.setItem(row, column, QTableWidgetItem(str(value)))
        finally:
            self.blockSignals(False)

    def tuple_value(self) -> tuple:
        from copy import deepcopy

        result = deepcopy(self._payloads)
        for row, rule in enumerate(result):
            if not isinstance(rule, dict):
                continue
            for column, key in enumerate(self.COLUMNS):
                cell = self.cellWidget(row, column)
                text = (
                    cell.currentData()
                    if isinstance(cell, QComboBox)
                    else self.item(row, column).text()
                    if self.item(row, column)
                    else ""
                )
                original = rule.get(key, "")
                if str(text) == str(original):
                    continue
                old = rule.get(key)
                if isinstance(old, bool):
                    value = str(text).lower() in ("true", "1", "yes")
                elif key in ("minimum", "maximum") and text != "":
                    numeric = float(text)
                    value = int(numeric) if isinstance(old, int) and numeric.is_integer() else numeric
                elif isinstance(old, int):
                    value = int(text)
                elif isinstance(old, float):
                    value = float(text)
                else:
                    value = text
                rule[key] = value
        return tuple(result)

    def add_rule(self):
        self._payloads.append({
            "action": "FLIP", "indicator": "RSI", "condition": "INSIDE",
            "minimum": 0.0, "maximum": 100.0,
        })
        self.set_tuple(tuple(self._payloads))
        self.changed.emit()

    def remove_selected(self):
        rows = sorted({index.row() for index in self.selectedIndexes()}, reverse=True)
        for row in rows:
            self.removeRow(row)
            self._payloads.pop(row)
        if rows:
            self.changed.emit()


class LosslessDoubleSpinBox(QDoubleSpinBox):
    """Friendly numeric display that retains an untouched native float exactly."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._native_value = 0.0
        self._edited = False
        self.valueChanged.connect(self._mark_edited)

    def _mark_edited(self, _value):
        self._edited = True

    def set_native_value(self, native: float, scale: float) -> None:
        self.blockSignals(True)
        try:
            self._native_value = native
            self._edited = False
            self.setValue(native * scale)
        finally:
            self.blockSignals(False)

    def native_value(self, scale: float) -> float:
        return self.value() / scale if self._edited else self._native_value


class DataclassForm(QWidget):
    """Lossless editor for every scalar/tuple field of one native dataclass."""

    changed = Signal()

    CHOICES = {
        "strategy_profile_run_mode": ("COMBINED_SHARED_CAPITAL", "ISOLATED_PROFILES", "BOTH"),
        "entry_mode": ("WAIT_UNTIL_CLOSED", "EVERY_N_CANDLES"),
        "sr_filter_mode": ("ANALYSIS_ONLY", "APPLY_ENTRY_RULES"),
        "daily_entry_missed_policy": ("SKIP_DAY", "NEXT_AVAILABLE_CANDLE"),
        "market_regime_method": ("BTC_STRUCTURAL", "ASSET_STRUCTURAL", "ASSET_RETURN"),
        "trade_flow_source": ("AGG_TRADES", "TRADES"),
        "sr_break_basis": ("CLOSE", "WICK"),
        "mean_reversion_mean_type": ("SMA", "EMA"),
        "risk_mode": ("ATR", "FIXED", "PERCENT"),
        "tie_policy": ("PESSIMISTIC", "OPTIMISTIC", "INTRABAR"),
        "sr_take_profit_mode": ("FIXED_R", "SR_CAPPED_R"),
        "sr_take_profit_no_level_policy": ("USE_FIXED_TP", "REJECT_TRADE"),
        "flip_rule_match_mode": ("ANY", "ALL"),
        "reject_rule_match_mode": ("ANY", "ALL"),
    }
    OPTIONAL_NUMERIC = {
        "large_trade_quote_threshold": float,
        "max_effective_leverage_per_leg": float,
        "max_combined_effective_leverage": float,
    }

    def __init__(self, value, *, excluded=(), groups=None, parent=None):
        super().__init__(parent)
        self.cls = type(value)
        self.widgets: dict[str, QWidget] = {}
        available = [item.name for item in fields(value) if item.name not in excluded]
        groups = groups or ((self.cls.__name__, tuple(available), None),)
        assigned = [name for _title, names, _note in groups for name in names]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(available):
            raise ValueError(f"Grouped form fields must represent {self.cls.__name__} exactly once")
        self.section_titles = tuple(title for title, _names, _note in groups)
        self._forms = {}
        self._dependents = {
            "trade_flow_enabled": (
                "trade_flow_source", "trade_flow_base_interval", "trade_flow_windows",
                "large_trade_quote_threshold",
            ),
            "order_book_enabled": (
                "order_book_base_interval", "book_ticker_max_age_seconds", "book_depth_max_age_seconds",
            ),
            "enable_support_resistance_analysis": (
                "sr_timeframe_minutes", "sr_pivot_left", "sr_pivot_right", "sr_lookback_bars",
                "sr_zone_width_atr", "sr_near_distance_atr", "enable_sr_hold_confirmation",
                "sr_hold_confirmation_bars", "sr_hold_confirmation_atr", "sr_break_tolerance_atr",
                "sr_break_basis",
            ),
            "break_even_enabled": ("break_even_activation_r", "break_even_offset_r"),
            "trailing_enabled": ("trailing_activation_r", "trailing_distance_r"),
            "partial_profit_enabled": ("tp1_r", "tp1_close_pct", "tp2_r"),
            "partial_stop_enabled": ("sl1_r", "sl1_close_pct", "sl2_r"),
            "timeout_enabled": ("timeout_minutes",),
            "r_step_trailing_enabled": (
                "r_step_activation_r", "r_step_distance_r", "r_step_size_r",
                "r_step_maximum_r", "r_step_activation_close_pct",
            ),
            "atr_checkpoint_tp_extension_enabled": (
                "atr_checkpoint_di_spread_minimum", "atr_checkpoint_bb_width_minimum",
                "atr_checkpoint_profit_lock_start", "atr_checkpoint_profit_lock_distance",
            ),
        }
        outer = QVBoxLayout(self)
        for title, names, note in groups:
            group = QGroupBox(title)
            form = QFormLayout(group)
            if note:
                status = QLabel(note)
                status.setObjectName("availabilityStatus")
                status.setWordWrap(True)
                form.addRow(status)
            for name in names:
                widget = self._widget(name, getattr(value, name))
                self.widgets[name] = widget
                info = metadata(name)
                widget.setToolTip(info.help)
                self._forms[name] = form
                form.addRow(info.label, widget)
            outer.addWidget(group)
        outer.addStretch()
        for controller, dependents in self._dependents.items():
            if controller in self.widgets:
                self.widgets[controller].toggled.connect(
                    lambda checked, names=dependents: self._set_visible(names, checked)
                )
        for widget in self.widgets.values():
            signal = (
                widget.toggled if isinstance(widget, QCheckBox)
                else widget.valueChanged if isinstance(widget, (QSpinBox, QDoubleSpinBox))
                else widget.currentIndexChanged if isinstance(widget, QComboBox)
                else widget.textChanged
            )
            signal.connect(lambda *_args: self.changed.emit())
        self.set_value(value)

    def _set_visible(self, names, visible):
        for name in names:
            if name in self.widgets:
                widget = self.widgets[name]
                form = self._forms[name]
                label = form.labelForField(widget)
                widget.setVisible(visible)
                if label:
                    label.setVisible(visible)

    def _widget(self, name: str, value: Any) -> QWidget:
        if name in self.CHOICES:
            widget = QComboBox()
            for native in self.CHOICES[name]:
                widget.addItem(ENUM_LABELS.get(name, {}).get(native, native), native)
            return widget
        if isinstance(value, bool):
            return QCheckBox()
        if isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-2_000_000_000, 2_000_000_000)
            return widget
        if isinstance(value, float):
            info = metadata(name)
            widget = LosslessDoubleSpinBox()
            widget.setRange(-1e12, 1e12)
            widget.setDecimals(info.decimals)
            widget.setSuffix(info.unit if info.unit != "$" else "")
            widget.setPrefix("$" if info.unit == "$" else "")
            return widget
        if isinstance(value, tuple):
            return TupleEditor()
        return QLineEdit()

    def set_value(self, value) -> None:
        for name, widget in self.widgets.items():
            raw = getattr(value, name)
            if isinstance(widget, QCheckBox):
                widget.setChecked(raw)
            elif isinstance(widget, LosslessDoubleSpinBox):
                widget.set_native_value(raw, metadata(name).scale)
            elif isinstance(widget, QSpinBox):
                widget.setValue(raw)
            elif isinstance(widget, QComboBox):
                index = widget.findData(raw)
                if index < 0:
                    widget.addItem(str(raw), raw)
                    index = widget.count() - 1
                widget.setCurrentIndex(index)
            elif isinstance(widget, TupleEditor):
                widget.set_tuple(tuple(raw))
            else:
                widget.setText("" if raw is None else str(raw))
        for controller, dependents in self._dependents.items():
            if controller in self.widgets:
                self._set_visible(dependents, self.widgets[controller].isChecked())

    def value(self, base=None):
        base = base or self.cls()
        values = {}
        for name, widget in self.widgets.items():
            old = getattr(base, name)
            if isinstance(widget, QCheckBox):
                raw = widget.isChecked()
            elif isinstance(widget, LosslessDoubleSpinBox):
                raw = widget.native_value(metadata(name).scale)
            elif isinstance(widget, QSpinBox):
                raw = widget.value()
            elif isinstance(widget, QComboBox):
                raw = widget.currentData()
            elif isinstance(widget, TupleEditor):
                raw = widget.tuple_value()
            else:
                text = widget.text()
                if old is None and not text:
                    raw = None
                elif isinstance(old, (int, float)):
                    raw = type(old)(text)
                elif name in self.OPTIONAL_NUMERIC:
                    raw = self.OPTIONAL_NUMERIC[name](text)
                else:
                    raw = text
            values[name] = raw
        return replace(base, **values)


class ProfileSelector(QComboBox):
    """Shows friendly profile names while preserving native profile keys."""

    def setCurrentText(self, text):
        index = self.findData(text)
        super().setCurrentIndex(index if index >= 0 else self.findText(text))


class TimeframeCombo(QComboBox):
    """Friendly visible timeframes with native interval values as user data."""

    def currentText(self):
        return self.currentData()

    def setCurrentText(self, text):
        index = self.findData(text)
        super().setCurrentIndex(index if index >= 0 else self.findText(text))


class NativeProfileEditor(QWidget):
    """Market permission plus explicit profile-specific exceptions over one base strategy."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "The base entry thesis is shared. Market environments only decide where it may trade; "
            "profile overrides are for genuine regime/direction exceptions, not duplicate filters."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        permission_box = QGroupBox("1. Market Environments — Where can this strategy trade?")
        permission = QGridLayout(permission_box)
        permission.addWidget(QLabel("Market state"), 0, 0)
        permission.addWidget(QLabel("LONG"), 0, 1)
        permission.addWidget(QLabel("SHORT"), 0, 2)
        self.permission_checks: dict[str, QCheckBox] = {}
        regimes = (("Bull", "bull"), ("Bear", "bear"), ("Sideways", "sideways"))
        for row, (label, native) in enumerate(regimes, 1):
            permission.addWidget(QLabel(label), row, 0)
            for column, direction in enumerate(("long", "short"), 1):
                key = f"{native}_{direction}"
                check = QCheckBox(f"Trade {direction.title()}")
                check.toggled.connect(
                    lambda checked, profile_key=key: self._toggle_permission(profile_key, checked)
                )
                check.clicked.connect(
                    lambda _checked, profile_key=key: self.selector.setCurrentText(profile_key)
                )
                self.permission_checks[key] = check
                permission.addWidget(check, row, column)
        layout.addWidget(permission_box)

        selected = QGroupBox("2. Profile Overrides — Only exceptions from the base strategy")
        selected_layout = QVBoxLayout(selected)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Environment"))
        self.selector = ProfileSelector()
        for key in PROFILE_KEYS:
            self.selector.addItem(PROFILE_LABELS[key], key)
        selector_row.addWidget(self.selector, 1)
        selected_layout.addLayout(selector_row)
        self.profile_summary = QLabel()
        self.profile_summary.setWordWrap(True)
        self.profile_summary.setStyleSheet(
            "background:#f7f9fb; padding:10px; border:1px solid #d9e2ec"
        )
        selected_layout.addWidget(self.profile_summary)

        actions = QHBoxLayout()
        self.copy_button = QPushButton("Copy Overrides")
        self.paste_button = QPushButton("Paste Overrides")
        self.reset_button = QPushButton("Reset to Native Defaults")
        self.apply_all_button = QPushButton("Apply Strategy Exceptions to All")
        for button in (
            self.copy_button, self.paste_button, self.reset_button, self.apply_all_button
        ):
            actions.addWidget(button)
        selected_layout.addLayout(actions)

        self.show_profile_details = QCheckBox(
            "Show advanced profile exceptions and execution overrides"
        )
        selected_layout.addWidget(self.show_profile_details)
        self.profile_details = QWidget()
        detail_layout = QVBoxLayout(self.profile_details)

        tabs = QTabWidget()
        self.strategy_form = DataclassForm(
            StrategyProfileConfig(),
            excluded={"enabled", "entry_rules", "rsi_period", "momentum_lookback_hours"},
            groups=STRATEGY_PROFILE_GROUPS,
        )
        strategy_page = QWidget()
        strategy_layout = QVBoxLayout(strategy_page)
        strategy_layout.addWidget(self.strategy_form)

        self.show_native_calculation_overrides = QCheckBox(
            "Show native calculation overrides (legacy / profile-specific)"
        )
        strategy_layout.addWidget(self.show_native_calculation_overrides)
        self.native_calculation_overrides = QGroupBox("Native Calculation Overrides")
        native_calc_layout = QVBoxLayout(self.native_calculation_overrides)
        native_calc_note = QLabel(
            "Normal indicator calculation belongs on Research Features. These profile-scoped fields "
            "remain here only because they are part of the native v3 advanced-rule contract. Change "
            "them only when intentionally reproducing a profile-specific or legacy rule setup."
        )
        native_calc_note.setWordWrap(True)
        native_calc_note.setStyleSheet("color:#52606d")
        native_calc_layout.addWidget(native_calc_note)
        native_calc_form = QFormLayout()
        self.native_calculation_widgets: dict[str, QSpinBox] = {}
        for name in ("rsi_period", "momentum_lookback_hours"):
            widget = QSpinBox()
            widget.setRange(-2_000_000_000, 2_000_000_000)
            widget.setToolTip(metadata(name).help)
            widget.valueChanged.connect(lambda _value: self._notify_changed())
            self.native_calculation_widgets[name] = widget
            native_calc_form.addRow(metadata(name).label, widget)
        native_calc_layout.addLayout(native_calc_form)
        self.native_calculation_overrides.setVisible(False)
        self.show_native_calculation_overrides.toggled.connect(
            self.native_calculation_overrides.setVisible
        )
        strategy_layout.addWidget(self.native_calculation_overrides)

        rule_title = QLabel("Advanced Exceptions — Native Profile Rule Builder")
        rule_title.setStyleSheet("font-weight:bold; margin-top:8px")
        strategy_layout.addWidget(rule_title)
        rule_note = QLabel(
            "Use this only when the centralized Entry Evidence section cannot express a real native "
            "profile exception. These rules retain the existing FLIP/REJECT engine semantics."
        )
        rule_note.setWordWrap(True)
        rule_note.setStyleSheet("color:#52606d")
        strategy_layout.addWidget(rule_note)
        self.entry_rules = EntryRuleEditor()
        strategy_layout.addWidget(self.entry_rules)
        rule_actions = QHBoxLayout()
        add_rule = QPushButton("+ Add Exception")
        remove_rule = QPushButton("Remove Selected")
        add_rule.clicked.connect(self.entry_rules.add_rule)
        remove_rule.clicked.connect(self.entry_rules.remove_selected)
        rule_actions.addWidget(add_rule)
        rule_actions.addWidget(remove_rule)
        strategy_layout.addLayout(rule_actions)

        self.execution_form = DataclassForm(
            ExecutionProfileConfig(), groups=EXECUTION_PROFILE_GROUPS
        )
        tabs.addTab(strategy_page, "Advanced Strategy Exceptions")
        tabs.addTab(self.execution_form, "Execution Override")
        detail_layout.addWidget(tabs)
        selected_layout.addWidget(self.profile_details)
        self.profile_details.setVisible(False)
        self.show_profile_details.toggled.connect(self.profile_details.setVisible)
        layout.addWidget(selected)

        self.profile_cards = {}
        self._strategy = {}
        self._execution = {}
        self._current = PROFILE_KEYS[0]
        self._rendering = False
        self._clipboard = None

        self.selector.currentIndexChanged.connect(
            lambda: self._select(self.selector.currentData())
        )
        self.copy_button.clicked.connect(self.copy_profile)
        self.paste_button.clicked.connect(self.paste_profile)
        self.reset_button.clicked.connect(self.reset_profile)
        self.apply_all_button.clicked.connect(self.apply_strategy_to_all)
        self.strategy_form.changed.connect(self._notify_changed)
        self.execution_form.changed.connect(self._notify_changed)
        self.entry_rules.changed.connect(self._notify_changed)

    @staticmethod
    def _strategy_exception_count(profile: StrategyProfileConfig) -> int:
        default = StrategyProfileConfig()
        return sum((
            profile.flip_direction != default.flip_direction,
            bool(profile.entry_rules),
            profile.flip_rule_match_mode != default.flip_rule_match_mode,
            profile.reject_rule_match_mode != default.reject_rule_match_mode,
        ))

    @staticmethod
    def _native_calculation_override_count(profile: StrategyProfileConfig) -> int:
        default = StrategyProfileConfig()
        return sum((
            profile.rsi_period != default.rsi_period,
            profile.momentum_lookback_hours != default.momentum_lookback_hours,
        ))

    @staticmethod
    def _execution_override_count(profile: ExecutionProfileConfig) -> int:
        default = ExecutionProfileConfig()
        return sum(
            getattr(profile, item.name) != getattr(default, item.name)
            for item in fields(ExecutionProfileConfig)
        )

    def _toggle_permission(self, key: str, checked: bool) -> None:
        if self._rendering or not self._strategy:
            return
        self._store()
        self._strategy[key] = replace(self._strategy[key], enabled=checked)
        self._render_permission_matrix()
        self._render_profile_summary()
        self.changed.emit()

    def _notify_changed(self):
        if self._rendering:
            return
        self._store()
        self._render_permission_matrix()
        self._render_profile_summary()
        self.changed.emit()

    def _store(self) -> None:
        if not self._strategy:
            return
        key = self._current
        strategy = self.strategy_form.value(self._strategy[key])
        self._strategy[key] = replace(
            strategy,
            entry_rules=self.entry_rules.tuple_value(),
            rsi_period=self.native_calculation_widgets["rsi_period"].value(),
            momentum_lookback_hours=self.native_calculation_widgets["momentum_lookback_hours"].value(),
        )
        self._execution[key] = self.execution_form.value(self._execution[key])

    def _render(self, key: str) -> None:
        self._rendering = True
        try:
            self._current = key
            self.strategy_form.set_value(self._strategy[key])
            for name, widget in self.native_calculation_widgets.items():
                widget.blockSignals(True)
                widget.setValue(getattr(self._strategy[key], name))
                widget.blockSignals(False)
            self.entry_rules.set_tuple(tuple(self._strategy[key].entry_rules))
            self.execution_form.set_value(self._execution[key])
            self._render_permission_matrix()
            self._render_profile_summary()
        finally:
            self._rendering = False

    def _render_permission_matrix(self):
        for key, check in self.permission_checks.items():
            if key not in self._strategy:
                continue
            check.blockSignals(True)
            check.setChecked(self._strategy[key].enabled)
            check.blockSignals(False)

    def _render_profile_summary(self):
        if self._current not in self._strategy:
            return
        strategy = self._strategy[self._current]
        execution = self._execution[self._current]
        strategy_count = self._strategy_exception_count(strategy)
        native_calc_count = self._native_calculation_override_count(strategy)
        execution_count = self._execution_override_count(execution)
        parts = [
            "Uses base entry thesis — no strategy exceptions"
            if not strategy_count else f"{strategy_count} strategy exception(s) from the base thesis"
        ]
        if native_calc_count:
            parts.append(f"{native_calc_count} native calculation override(s)")
        parts.append(
            "Standard native execution"
            if not execution_count else f"{execution_count} execution override(s) from native defaults"
        )
        self.profile_summary.setText(
            f"{PROFILE_LABELS[self._current]} — {'TRADE' if strategy.enabled else 'OFF'}\n"
            + " · ".join(parts) + "."
        )

    def _select(self, key: str) -> None:
        self._store()
        self._render(key)

    def set_profiles(self, strategy, execution) -> None:
        self.selector.blockSignals(True)
        try:
            self._strategy = dict(strategy)
            self._execution = dict(execution)
            self._render(self.selector.currentData() or PROFILE_KEYS[0])
        finally:
            self.selector.blockSignals(False)

    def profiles(self):
        self._store()
        return dict(self._strategy), dict(self._execution)

    def copy_profile(self):
        self._store()
        self._clipboard = clone_profile_pair(
            self._strategy[self._current], self._execution[self._current]
        )

    def paste_profile(self):
        if self._clipboard:
            current_enabled = self._strategy[self._current].enabled
            copied_strategy, copied_execution = clone_profile_pair(*self._clipboard)
            self._strategy[self._current] = replace(copied_strategy, enabled=current_enabled)
            self._execution[self._current] = copied_execution
            self._render(self._current)
            self.changed.emit()

    def reset_profile(self):
        enabled = self._strategy[self._current].enabled
        self._strategy[self._current] = replace(StrategyProfileConfig(), enabled=enabled)
        self._execution[self._current] = ExecutionProfileConfig()
        self._render(self._current)
        self.changed.emit()

    def apply_strategy_to_all(self):
        self._store()
        source = self._strategy[self._current]
        for key in PROFILE_KEYS:
            current_enabled = self._strategy[key].enabled
            copied = clone_profile_pair(source, self._execution[key])[0]
            self._strategy[key] = replace(copied, enabled=current_enabled)
        self._render(self._current)
        self.changed.emit()


class StrategyWorkspace(QWidget):
    """Single researcher-facing strategy builder over authoritative native widgets.

    Research Features owns how evidence is calculated. This workspace is the only
    normal UI surface that decides how already-calculated evidence affects entries.
    It never invents unsupported Direction/Confirmation/Veto/Ranking semantics.
    """

    EVIDENCE_SOURCES = (
        "DI Direction", "DI Pressure", "Mean Reversion", "Support / Resistance",
        "Open Interest", "Funding", "Positioning / Basis", "Taker Flow",
        "Trade Flow", "Order Book",
    )

    def __init__(
        self, profile_editor: NativeProfileEditor, strategy_form: DataclassForm,
        parent=None,
    ):
        super().__init__(parent)
        self.profile_editor = profile_editor
        self.strategy_form = strategy_form
        self.widgets = strategy_form.widgets
        layout = QVBoxLayout(self)

        thesis_box = QGroupBox("Strategy Summary")
        thesis_layout = QVBoxLayout(thesis_box)
        self.thesis_summary = QLabel()
        self.thesis_summary.setWordWrap(True)
        self.thesis_summary.setStyleSheet(
            "font-size:13px; background:#f7f9fb; padding:12px; border:1px solid #d9e2ec"
        )
        thesis_layout.addWidget(self.thesis_summary)
        layout.addWidget(thesis_box)

        mode_box = QGroupBox("Strategy Test Mode")
        mode_layout = QVBoxLayout(mode_box)
        mode_form = QFormLayout()
        mode_form.addRow("How enabled environments are tested", self.widgets["strategy_profile_run_mode"])
        mode_layout.addLayout(mode_form)
        mode_note = QLabel(
            "This controls shared-capital versus isolated testing. It is not an entry condition."
        )
        mode_note.setWordWrap(True)
        mode_note.setStyleSheet("color:#52606d")
        mode_layout.addWidget(mode_note)
        layout.addWidget(mode_box)

        layout.addWidget(profile_editor)

        entry_box = QGroupBox("3. Base Entry Evidence — Single source of truth for entry decisions")
        entry = QVBoxLayout(entry_box)
        framework_note = QLabel(
            "Research Features controls how evidence is calculated. Strategy Builder controls how "
            "that evidence is used. New Binance evidence starts as Analyze Only and is promoted to "
            "Direction, Required, Confirmation, Veto/Avoid or Ranking only after causal outcome research "
            "and an explicit native strategy implementation."
        )
        framework_note.setWordWrap(True)
        framework_note.setStyleSheet(
            "background:#eef5fb; padding:10px; border:1px solid #c8d9e8"
        )
        entry.addWidget(framework_note)

        map_box = QGroupBox("Entry Evidence")
        evidence_grid = QGridLayout(map_box)
        evidence_grid.addWidget(QLabel("Evidence"), 0, 0)
        evidence_grid.addWidget(QLabel("Current role"), 0, 1)
        evidence_grid.addWidget(QLabel("Condition / usage"), 0, 2)
        evidence_grid.addWidget(QLabel("Notes"), 0, 3)
        self.evidence_role_labels: dict[str, QLabel] = {}
        self.evidence_note_labels: dict[str, QLabel] = {}
        self.evidence_setting_widgets: dict[str, QWidget] = {}

        for row, source in enumerate(self.EVIDENCE_SOURCES, 1):
            source_label = QLabel(source)
            source_label.setStyleSheet("font-weight:bold")
            role = QLabel("—")
            role.setMinimumWidth(125)
            note = QLabel("—")
            note.setWordWrap(True)
            self.evidence_role_labels[source] = role
            self.evidence_note_labels[source] = note
            evidence_grid.addWidget(source_label, row, 0)
            evidence_grid.addWidget(role, row, 1)

            if source == "DI Direction":
                setting = self._toggle_setting(
                    self.widgets["enable_di_direction_selection"], "Use DI to choose trade direction"
                )
            elif source == "DI Pressure":
                setting = QWidget()
                row_layout = QHBoxLayout(setting)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(self.widgets["enable_di_pressure_analysis"])
                row_layout.addWidget(QLabel("Use pressure"))
                for name, label in (
                    ("di_pressure_allow_expanding", "Expanding"),
                    ("di_pressure_allow_contracting", "Contracting"),
                    ("di_pressure_allow_mixed", "Mixed"),
                ):
                    row_layout.addWidget(QLabel(label))
                    row_layout.addWidget(self.widgets[name])
                row_layout.addStretch()
                self.pressure_states = setting
            elif source == "Mean Reversion":
                setting = self._toggle_setting(
                    self.widgets["enable_mean_reversion_analysis"], "Calculate / attach MR context"
                )
            elif source == "Support / Resistance":
                setting = self.widgets["sr_filter_mode"]
            else:
                setting = QLabel("Managed on Research Features")
                setting.setStyleSheet("color:#52606d")
            self.evidence_setting_widgets[source] = setting
            evidence_grid.addWidget(setting, row, 2)
            evidence_grid.addWidget(note, row, 3)

        evidence_grid.setColumnStretch(2, 1)
        evidence_grid.setColumnStretch(3, 2)
        entry.addWidget(map_box)

        self.sr_filter_details = QGroupBox("Support / Resistance Veto / Avoid Conditions")
        sr_form = QFormLayout(self.sr_filter_details)
        for name in (
            "sr_long_avoid_near_resistance", "sr_long_require_near_support",
            "sr_long_block_broken_support", "sr_long_min_room_to_resistance_atr",
            "sr_short_avoid_near_support", "sr_short_require_near_resistance",
            "sr_short_block_broken_resistance", "sr_short_min_room_to_support_atr",
        ):
            sr_form.addRow(metadata(name).label, self.widgets[name])
        entry.addWidget(self.sr_filter_details)

        legend = QLabel(
            "Role model: Analyze Only records evidence without changing trades. Required can reject "
            "an entry when a native condition fails. Veto / Avoid rejects otherwise valid entries. "
            "Confirmation and Ranking are reserved until the native engine explicitly supports them."
        )
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#52606d; padding:6px")
        entry.addWidget(legend)
        layout.addWidget(entry_box)

        self.show_advanced_strategy = QCheckBox(
            "Show advanced entry timing and schedule settings"
        )
        layout.addWidget(self.show_advanced_strategy)
        self.advanced_strategy = QGroupBox("4. Advanced Entry Timing")
        advanced = QFormLayout(self.advanced_strategy)
        for name in (
            "entry_mode", "entry_interval", "enable_daily_entry_schedule",
            "daily_entry_time", "daily_entry_timezone", "daily_entry_missed_policy",
        ):
            advanced.addRow(metadata(name).label, self.widgets[name])
        self.advanced_strategy.setVisible(False)
        self.show_advanced_strategy.toggled.connect(self.advanced_strategy.setVisible)
        layout.addWidget(self.advanced_strategy)
        layout.addStretch()

        self.widgets["enable_di_pressure_analysis"].toggled.connect(
            self._refresh_visibility
        )
        self.widgets["sr_filter_mode"].currentIndexChanged.connect(
            lambda _index: self._refresh_visibility()
        )
        strategy_form.changed.connect(self.refresh_from_widgets)
        profile_editor.changed.connect(self.refresh_from_widgets)
        self._refresh_visibility()
        self.refresh_from_widgets()

    @staticmethod
    def _toggle_setting(widget: QCheckBox, text: str) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(widget)
        row_layout.addWidget(QLabel(text))
        row_layout.addStretch()
        return row

    def _feature_config(self) -> FeatureConfig:
        root = self.window()
        form = getattr(root, "feature_form", None)
        base_config = getattr(root, "config", None)
        base = getattr(base_config, "features", FeatureConfig())
        if form is None:
            return base
        try:
            return form.value(base)
        except (ValueError, TypeError, KeyError):
            return base

    def evidence_roles(self) -> dict[str, tuple[str, str]]:
        """Describe current native entry roles; this method never changes config."""
        strategy = self.strategy_form.value(StrategyConfig())
        features = self._feature_config()

        roles: dict[str, tuple[str, str]] = {}
        roles["DI Direction"] = (
            ("Direction", "Chooses the candidate side using native DI direction selection.")
            if strategy.enable_di_direction_selection
            else ("Not Used", "Profile/native direction remains in control.")
        )

        if not strategy.enable_di_pressure_analysis:
            roles["DI Pressure"] = ("Off", "DI pressure context and its entry filter are disabled.")
        else:
            allowed = (
                strategy.di_pressure_allow_expanding,
                strategy.di_pressure_allow_contracting,
                strategy.di_pressure_allow_mixed,
            )
            roles["DI Pressure"] = (
                ("Analyze Only", "All pressure states are allowed, so pressure cannot reject entries.")
                if all(allowed)
                else ("Required", "Disallowed pressure states are rejected by the existing native filter.")
            )

        roles["Mean Reversion"] = (
            ("Analyze Only", "Recorded for research; it does not confirm, choose direction or block entries by itself.")
            if strategy.enable_mean_reversion_analysis
            else ("Off", "Mean Reversion research context is disabled.")
        )

        if strategy.sr_filter_mode == "APPLY_ENTRY_RULES":
            sr_note = "Existing S/R location rules may reject entries."
            if not features.enable_support_resistance_analysis:
                sr_note += " Enable S/R calculation on Research Features so location context is available."
            roles["Support / Resistance"] = ("Veto / Avoid", sr_note)
        elif features.enable_support_resistance_analysis:
            roles["Support / Resistance"] = (
                "Analyze Only", "S/R location context is calculated but does not block entries."
            )
        else:
            roles["Support / Resistance"] = (
                "Off", "S/R calculation is disabled on Research Features."
            )

        for source in ("Open Interest", "Funding", "Positioning / Basis", "Taker Flow"):
            roles[source] = (
                "Analyze Only",
                "Attached automatically when source coverage exists; it cannot change entries yet.",
            )
        roles["Trade Flow"] = (
            ("Analyze Only", "Enabled on Research Features; collected for causal outcome research only.")
            if features.trade_flow_enabled
            else ("Off", "Enable Trade Flow on Research Features to collect it for research.")
        )
        roles["Order Book"] = (
            ("Analyze Only", "Enabled on Research Features; collected for causal outcome research only.")
            if features.order_book_enabled
            else ("Off", "Enable Order Book on Research Features to collect it for research.")
        )
        return roles

    @staticmethod
    def _market_permission_summary(profiles) -> str:
        parts = []
        for regime, prefix in (("Bull", "bull"), ("Bear", "bear"), ("Sideways", "sideways")):
            sides = []
            if profiles[f"{prefix}_long"].enabled:
                sides.append("Long")
            if profiles[f"{prefix}_short"].enabled:
                sides.append("Short")
            parts.append(f"{regime}: {' + '.join(sides) if sides else 'Off'}")
        return "; ".join(parts)

    def _refresh_visibility(self):
        enabled = self.widgets["enable_di_pressure_analysis"].isChecked()
        for name in (
            "di_pressure_allow_expanding", "di_pressure_allow_contracting",
            "di_pressure_allow_mixed",
        ):
            self.widgets[name].setVisible(enabled)
        self.sr_filter_details.setVisible(
            self.widgets["sr_filter_mode"].currentData() == "APPLY_ENTRY_RULES"
        )

    def refresh_from_widgets(self):
        self._refresh_visibility()
        if not self.profile_editor._strategy:
            self.thesis_summary.setText(
                "Configure the strategy to see a plain-English summary here."
            )
            return
        try:
            profiles, _execution = self.profile_editor.profiles()
            roles = self.evidence_roles()
        except (ValueError, TypeError, KeyError):
            return

        for source, (role, note) in roles.items():
            self.evidence_role_labels[source].setText(role)
            self.evidence_note_labels[source].setText(note)

        market_text = self._market_permission_summary(profiles)
        profile_overrides = sum(
            self.profile_editor._strategy_exception_count(profile)
            for profile in profiles.values()
        )
        override_text = (
            f"{profile_overrides} strategy exception(s) across enabled environments."
            if profile_overrides
            else "No profile-specific strategy exceptions."
        )
        self.thesis_summary.setText(
            f"<b>Markets</b><br>{market_text}<br><br>"
            f"<b>Direction</b><br>DI Direction: {roles['DI Direction'][0]}<br><br>"
            f"<b>Required</b><br>DI Pressure: {roles['DI Pressure'][0]}<br><br>"
            f"<b>Supporting Evidence</b><br>Mean Reversion: {roles['Mean Reversion'][0]}; "
            f"OI, Funding, Positioning and Taker Flow remain Analyze Only.<br><br>"
            f"<b>Avoid / Veto</b><br>Support / Resistance: {roles['Support / Resistance'][0]}<br><br>"
            f"<b>Ranking</b><br>No native confidence/ranking role yet.<br><br>"
            f"<b>Profile Overrides</b><br>{override_text}"
        )


class RunWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service, request, config):
        super().__init__()
        self.service = service
        self.request = request
        self.config = config

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.service.run(self.request, self.config))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """The single authoritative GUI, inheriting directly from QMainWindow."""

    def __init__(self, startup_status=None, service=None):
        super().__init__()
        self.setWindowTitle("Crypto Strategy Lab — Research Workstation")
        self.resize(1500, 920)
        self.service = service or GuiApplicationService(MARKET_DATA_ROOT, CACHE_DIR)
        self.config = ResearchRunConfig()
        self._manifest = None
        self._run_dir = None
        self._thread = None
        self._applying_config = False

        root = QWidget()
        shell = QHBoxLayout(root)
        self.setCentralWidget(root)
        nav = QVBoxLayout()
        shell.addLayout(nav)
        self.pages = QStackedWidget()
        shell.addWidget(self.pages, 1)

        self.profile_editor = NativeProfileEditor()
        self.strategy_form = DataclassForm(
            StrategyConfig(), excluded={"profiles"}, groups=STRATEGY_GROUPS
        )
        self.feature_form = DataclassForm(FeatureConfig(), groups=FEATURE_GROUPS)
        self.execution_form = DataclassForm(
            ExecutionConfig(), excluded={"profiles"}, groups=EXECUTION_GROUPS
        )
        self.reporting_form = DataclassForm(
            ReportingConfig(), excluded={"output_dir"}
        )
        self.strategy_workspace = StrategyWorkspace(
            self.profile_editor, self.strategy_form
        )

        setup = self._page("Setup", self._data_panel(), self._status_panel())
        strategy = self._page(
            "Strategy Builder", self._scroll(self.strategy_workspace)
        )
        features = self._page("Research Features", self._scroll(self.feature_form))
        risk = self._page(
            "Risk & Execution", self._scroll(self.execution_form),
            self._risk_explanation(),
        )
        reports = self._page(
            "Reports & Diagnostics", self._report_presets(),
            self._scroll(self.reporting_form),
        )
        review = self._page("Review & Run", self._review_panel(), self._run_panel())
        results = self._page("Results Dashboard", self._results_panel())
        library = self._page("Data Library", self._data_library_panel())
        chat = ChatGPTIntegrationWidget(
            QSettings("CryptoStrategyLab", "CryptoStrategyLab"),
            lambda: self.output_root.text(),
        )
        github = GitHubIntegrationWidget()
        groups = (
            ("NEW RESEARCH", (
                ("Setup", setup), ("Strategy Builder", strategy),
                ("Research Features", features), ("Risk & Execution", risk),
                ("Reports", reports), ("Review & Run", review),
            )),
            ("RESULTS", (("Results Dashboard", results),)),
            ("DATA", (("Data Library", library),)),
            ("TOOLS", (("ChatGPT / MCP", chat), ("GitHub", github))),
        )
        for heading, entries in groups:
            title = QLabel(heading)
            title.setStyleSheet(
                "font-weight:bold; color:#52606d; margin-top:10px"
            )
            nav.addWidget(title)
            for label, page in entries:
                index = self.pages.addWidget(page)
                button = QPushButton(label)
                button.setFlat(True)
                button.clicked.connect(
                    lambda _=False, i=index: self.pages.setCurrentIndex(i)
                )
                nav.addWidget(button)
        nav.addStretch()
        self.current_research = QLabel()
        self.current_research.setMinimumWidth(245)
        self.current_research.setStyleSheet(
            "background:#f4f7fa; padding:12px; border:1px solid #d9e2ec"
        )
        nav.addWidget(self.current_research)
        quick_run = QPushButton("RUN BACKTEST")
        quick_run.clicked.connect(self.start_run)
        nav.addWidget(quick_run)
        self._connect_request_refresh()
        self._connect_live_summary()
        self.service.refresh_catalog()
        self._load_catalog()
        self.apply_config(self.config)
        if startup_status:
            startup_status("Native v2 research GUI ready")

    @staticmethod
    def _page(title, *widgets):
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel(title)
        heading.setStyleSheet("font-size:22px; font-weight:bold; margin:8px")
        layout.addWidget(heading)
        for widget in widgets:
            layout.addWidget(widget)
        return page

    def _status_panel_clone(self):
        box = QGroupBox("Technical Catalog Detail")
        layout = QVBoxLayout(box)
        note = QLabel(
            "Availability, friendly coverage dates, interval and partition counts are supplied "
            "by the catalog service. Raw archive paths are never displayed."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        return box

    def _data_library_panel(self):
        box = QGroupBox("Catalog Coverage")
        layout = QVBoxLayout(box)
        refresh = QPushButton("Refresh Catalog View")
        refresh.clicked.connect(self.refresh_data_library)
        layout.addWidget(refresh)
        self.library_table = QTableWidget(0, 7)
        self.library_table.setHorizontalHeaderLabels([
            "Symbol", "Dataset Family", "Interval", "First Available UTC",
            "Last Available UTC", "Partitions", "State",
        ])
        layout.addWidget(self.library_table)
        return box

    def refresh_data_library(self):
        if not hasattr(self.service.catalog, "inventory"):
            return
        rows = self.service.catalog.inventory(self.market.currentData())
        self.library_table.setRowCount(len(rows))
        for row_number, row in enumerate(rows):
            count = row.get("archive_count", 0)
            first = row.get("first_period")
            last = row.get("last_period")
            state = (
                "UNAVAILABLE" if not count
                else "AVAILABLE" if first is not None and last is not None
                else "PARTIAL"
            )
            values = (
                row.get("symbol", "—"), self._dataset_family(row.get("dataset", "")),
                row.get("interval") or "Event data", first or "—", last or "—",
                count, row.get("state", state),
            )
            for column, value in enumerate(values):
                self.library_table.setItem(
                    row_number, column, QTableWidgetItem(str(value))
                )

    @staticmethod
    def _dataset_family(dataset):
        value = str(dataset).lower()
        if "kline" in value:
            return "Candles"
        if "fund" in value:
            return "Funding"
        if "metric" in value or "interest" in value or "ratio" in value:
            return "Futures Positioning"
        if "agg" in value or value == "trades":
            return "Trades"
        if "book" in value or "depth" in value:
            return "Order Book"
        return str(dataset).replace("_", " ").title()

    def _risk_explanation(self):
        box = QGroupBox("Effective Risk")
        layout = QVBoxLayout(box)
        self.risk_explanation = QLabel()
        self.risk_explanation.setWordWrap(True)
        layout.addWidget(self.risk_explanation)
        return box

    def _report_presets(self):
        box = QGroupBox("Report Preset")
        layout = QHBoxLayout(box)
        self.report_preset = QComboBox()
        self.report_preset.addItem("Quick — core artifacts", "QUICK")
        self.report_preset.addItem("Standard — Recommended", "STANDARD")
        self.report_preset.addItem(
            "Deep Research — full diagnostics", "DEEP_RESEARCH"
        )
        apply_button = QPushButton("Apply Preset")
        apply_button.clicked.connect(self.apply_reporting_preset)
        layout.addWidget(self.report_preset)
        layout.addWidget(apply_button)
        return box

    def apply_reporting_preset(self):
        current = self.reporting_form.value(self.config.reporting)
        updated = apply_report_preset(current, self.report_preset.currentData())
        self.reporting_form.set_value(updated)

    def _review_panel(self):
        box = QGroupBox("Experiment Review")
        layout = QVBoxLayout(box)
        self.review_summary = QLabel()
        self.review_summary.setWordWrap(True)
        layout.addWidget(self.review_summary)
        return box

    @staticmethod
    def _scroll(widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _data_panel(self):
        box = QGroupBox("Data / Request")
        form = QFormLayout(box)
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
        self.datasets = QLabel("Catalog not loaded")
        self.datasets.setWordWrap(True)
        date_note = QLabel(
            "Research includes data from the start date up to, but not including, "
            "the selected end boundary."
        )
        date_note.setWordWrap(True)
        for label, widget in (
            ("Exchange", self.exchange), ("Market", self.market),
            ("Symbol", self.symbol), ("Start Date", self.start),
            ("End Date", self.end), ("Date Range", date_note),
            ("Strategy Timeframe", self.strategy_tf),
            ("Intrabar / Exit Detail", self.intrabar_tf),
            ("Research Data Availability", self.datasets),
        ):
            form.addRow(label, widget)
        return box

    def _config_tabs(self):
        tabs = QTabWidget()
        tabs.addTab(self._scroll(self.profile_editor), "Profiles")
        tabs.addTab(self._scroll(self.strategy_form), "Strategy")
        tabs.addTab(self._scroll(self.feature_form), "Research Features")
        tabs.addTab(self._scroll(self.execution_form), "Execution")
        return tabs

    def _run_panel(self):
        box = QGroupBox("Run")
        layout = QHBoxLayout(box)
        self.output_root = QLineEdit(str(OUTPUT_DIR / "data_lake_v2"))
        browse = QPushButton("Output Root…")
        browse.clicked.connect(self._browse_output)
        self.save = QPushButton("Save Config…")
        self.load = QPushButton("Load Config…")
        self.run_button = QPushButton("Run")
        self.progress = QProgressBar()
        self.stage = QLabel("Ready")
        self.save.clicked.connect(self.save_config)
        self.load.clicked.connect(self.load_config)
        self.run_button.clicked.connect(self.start_run)
        for widget in (
            QLabel("Output root"), self.output_root, browse, self.save, self.load,
            self.run_button, self.progress, self.stage,
        ):
            layout.addWidget(widget)
        return box

    def _status_panel(self):
        box = QGroupBox("Data Status")
        layout = QVBoxLayout(box)
        self.resolution = QLabel("Requested/effective resolution: not run")
        self.coverage = QTableWidget(0, 6)
        self.coverage.setHorizontalHeaderLabels([
            "Dataset", "Interval", "First UTC", "Last UTC", "Partitions", "State"
        ])
        self.quality = QLabel("Data quality: not run")
        self.quality_table = QTableWidget(0, 6)
        self.quality_table.setHorizontalHeaderLabels([
            "Dataset", "Interval", "Required", "Rows", "Status", "Issues"
        ])
        for widget in (
            self.resolution, self.coverage, self.quality, self.quality_table
        ):
            layout.addWidget(widget)
        return box

    def _results_panel(self):
        box = QGroupBox("Results / Canonical Artifacts")
        layout = QVBoxLayout(box)
        cards = QGridLayout()
        layout.addLayout(cards)
        self.kpi_cards = {}
        labels = (
            "Trades", "Wins", "Losses", "Win Rate", "Net R", "Average R",
            "Net PnL", "Ending Equity", "Profit Factor", "Maximum Drawdown", "Fees",
        )
        for index, label in enumerate(labels):
            card = QLabel(f"{label}\n—")
            card.setFrameShape(QLabel.Box)
            card.setMinimumHeight(55)
            cards.addWidget(card, index // 4, index % 4)
            self.kpi_cards[label] = card
        self.summary = QLabel("No completed run")
        layout.addWidget(self.summary)
        self.timings = QLabel("Run timings: —")
        layout.addWidget(self.timings)
        self.artifact_buttons = {}
        for key in (
            "workbook", "trade_csv", "summary", "trades", "signals",
            "feature_context", "telemetry", "data_quality",
        ):
            button = QPushButton(key.replace("_", " ").title())
            button.setEnabled(False)
            button.clicked.connect(
                lambda _=False, name=key: self.open_artifact(name)
            )
            self.artifact_buttons[key] = button
            layout.addWidget(button)
        self.open_folder = QPushButton("Output Folder")
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self._run_dir))
            )
        )
        layout.addWidget(self.open_folder)
        return box

    def _connect_request_refresh(self):
        self.symbol.currentTextChanged.connect(self.refresh_coverage)
        self.exchange.currentIndexChanged.connect(self.refresh_coverage)
        self.market.currentIndexChanged.connect(self.refresh_coverage)
        self.start.dateChanged.connect(self.refresh_coverage)
        self.end.dateChanged.connect(self.refresh_coverage)
        self.strategy_tf.currentTextChanged.connect(self.refresh_coverage)
        self.intrabar_tf.currentTextChanged.connect(self.refresh_coverage)

    def _connect_live_summary(self):
        for form in (
            self.strategy_form, self.feature_form, self.execution_form,
            self.reporting_form,
        ):
            form.changed.connect(self._refresh_summary_from_widgets)
        self.profile_editor.changed.connect(self._refresh_summary_from_widgets)
        self.output_root.textChanged.connect(self._refresh_summary_from_widgets)

    def _refresh_summary_from_widgets(self):
        if self._applying_config:
            return
        try:
            config = self.build_config()
            self.strategy_workspace.refresh_from_widgets()
            self._render_research_summary(config)
        except (ValueError, TypeError, KeyError):
            pass

    def request_model(self):
        def utc(widget):
            return pd.Timestamp(widget.date().toPython(), tz="UTC").to_pydatetime()

        intrabar = self.intrabar_tf.currentData()
        return GuiResearchRequest(
            self.exchange.currentData(), self.market.currentData(),
            self.symbol.currentText(), utc(self.start), utc(self.end),
            self.strategy_tf.currentText(), intrabar,
        )

    def build_config(self):
        strategy_profiles, execution_profiles = self.profile_editor.profiles()
        intrabar = self.request_model().intrabar_timeframe
        data = replace(
            self.config.data,
            strategy_timeframe_minutes=timeframe_minutes(
                self.strategy_tf.currentText()
            ),
            use_intrabar_data=intrabar is not None,
            intrabar_timeframe_minutes=(
                timeframe_minutes(intrabar)
                if intrabar else self.config.data.intrabar_timeframe_minutes
            ),
        )
        strategy = replace(
            self.strategy_form.value(self.config.strategy),
            profiles=strategy_profiles,
        )
        execution = replace(
            self.execution_form.value(self.config.execution),
            profiles=execution_profiles,
        )
        return replace(
            self.config,
            data=data,
            features=self.feature_form.value(self.config.features),
            strategy=strategy,
            execution=execution,
            reporting=replace(
                self.reporting_form.value(self.config.reporting),
                output_dir=self.output_root.text(),
            ),
        )

    def apply_config(self, config):
        self._applying_config = True
        try:
            self.config = config
            data = config.data
            self.strategy_tf.setCurrentText(
                timeframe_label(data.strategy_timeframe_minutes)
            )
            self.intrabar_tf.setCurrentText(
                timeframe_label(data.intrabar_timeframe_minutes)
                if data.use_intrabar_data else None
            )
            self.profile_editor.set_profiles(
                config.strategy.profiles, config.execution.profiles
            )
            self.strategy_form.set_value(config.strategy)
            self.feature_form.set_value(config.features)
            self.execution_form.set_value(config.execution)
            self.output_root.setText(config.reporting.output_dir)
            self.reporting_form.set_value(config.reporting)
        finally:
            self._applying_config = False
        self.strategy_workspace.refresh_from_widgets()
        self._render_research_summary(config)

    def _render_research_summary(self, config):
        enabled = sum(
            profile.enabled for profile in config.strategy.profiles.values()
        )
        intrabar = (
            f"{config.data.intrabar_timeframe_minutes}m exits"
            if config.data.use_intrabar_data else "bar-close exits"
        )
        risk = display_percentage(config.execution.risk_per_leg)
        text = (
            f"CURRENT RESEARCH\n\n{self.symbol.currentText() or 'BTCUSDT'}\n"
            f"{timeframe_label(config.data.strategy_timeframe_minutes)} strategy / {intrabar}\n\n"
            f"Environments  {enabled} of 6 ON\n"
            f"DI Direction  {'ON' if config.strategy.enable_di_direction_selection else 'OFF'}\n"
            f"MR Context  {'ANALYZE' if config.strategy.enable_mean_reversion_analysis else 'OFF'}\n"
            f"Trade Flow  {'ANALYZE' if config.features.trade_flow_enabled else 'OFF'}\n"
            f"Order Book  {'ANALYZE' if config.features.order_book_enabled else 'OFF'}\n\n"
            f"Base risk  {risk}\nMax trades  {config.execution.max_active_pairs}\n\n"
            f"Data  {self._data_state()}"
        )
        self.current_research.setText(text)
        self.risk_explanation.setText(
            f"Base Risk: {risk}\nAt ${config.execution.initial_equity:,.2f}, planned "
            f"base full-stop loss is "
            f"${config.execution.initial_equity * config.execution.risk_per_leg:,.2f}. "
            "Profile multipliers use the existing execution configuration."
        )
        if hasattr(self, "review_summary"):
            mode = ENUM_LABELS["strategy_profile_run_mode"].get(
                config.strategy.strategy_profile_run_mode,
                config.strategy.strategy_profile_run_mode,
            )
            enabled_names = [
                PROFILE_LABELS[key] for key, profile in config.strategy.profiles.items()
                if profile.enabled
            ]
            pressure_states = (
                [
                    label for allowed, label in (
                        (config.strategy.di_pressure_allow_expanding, "Expanding"),
                        (config.strategy.di_pressure_allow_contracting, "Contracting"),
                        (config.strategy.di_pressure_allow_mixed, "Mixed"),
                    ) if allowed
                ]
                if config.strategy.enable_di_pressure_analysis else []
            )
            sr_text = (
                "Veto / Avoid"
                if config.strategy.sr_filter_mode == "APPLY_ENTRY_RULES"
                else "Analyze Only"
            )
            self.review_summary.setText(
                f"{self.symbol.currentText() or 'BTCUSDT'} — "
                f"{TIMEFRAME_LABELS[timeframe_label(config.data.strategy_timeframe_minutes)]} Research\n\n"
                f"Allowed environments: {', '.join(enabled_names) if enabled_names else 'None'}\n"
                f"Direction: {'DI selection' if config.strategy.enable_di_direction_selection else 'Profile/native direction'}\n"
                f"DI pressure: {', '.join(pressure_states) if pressure_states else 'Off'}\n"
                f"Mean Reversion: {'Analyze Only' if config.strategy.enable_mean_reversion_analysis else 'Off'}\n"
                f"Support / Resistance: {sr_text}\n\n"
                f"Profile Test: {mode}\n"
                f"Starting Equity: ${config.execution.initial_equity:,.2f}\n"
                f"Base Risk: {risk}\n"
                f"Maximum Active Trades: {config.execution.max_active_pairs}\n"
                f"Reports: {config.reporting.analysis_level}\n\n"
                f"DATA STATUS: {self._data_state()}"
            )

    def _data_state(self):
        return self.data_readiness(
            self._coverage_rows_from_table(), self.strategy_tf.currentData(),
            self.intrabar_tf.currentData(),
        )[0]

    @staticmethod
    def data_readiness(rows, strategy_interval, intrabar_interval=None):
        """Classify catalog coverage using only canonical execution klines as required data."""
        required = {strategy_interval}
        if intrabar_interval:
            required.add(intrabar_interval)

        def dataset_name(row):
            dataset = row.get("dataset", "")
            return str(getattr(dataset, "value", dataset)).lower()

        execution_candles = [
            row for row in rows if dataset_name(row) == "klines"
        ]
        missing = []
        for interval in required:
            matches = [
                row for row in execution_candles
                if row.get("interval") == interval
            ]
            if not matches or not any(
                row.get("state") == "AVAILABLE" for row in matches
            ):
                missing.append(interval)
        if missing:
            return (
                "BLOCKED",
                "Required candle coverage unavailable: "
                + ", ".join(sorted(missing)),
            )
        optional = [
            row for row in rows
            if row not in execution_candles and row.get("state") != "AVAILABLE"
        ]
        if optional:
            return (
                "WARN",
                "Required candles are ready; optional research coverage is partial or unavailable.",
            )
        return "READY", "Required execution data is available."

    def _coverage_rows_from_table(self):
        return [
            {
                "dataset": self.coverage.item(row, 0).text(),
                "interval": (
                    None if self.coverage.item(row, 1).text() == "—"
                    else self.coverage.item(row, 1).text()
                ),
                "state": self.coverage.item(row, 5).text(),
            }
            for row in range(self.coverage.rowCount())
            if all(self.coverage.item(row, column) for column in (0, 1, 5))
        ]

    def _load_catalog(self):
        current = self.symbol.currentText()
        symbols = self.service.catalog.symbols()
        self.symbol.blockSignals(True)
        self.symbol.clear()
        self.symbol.addItems(symbols or [current or "BTCUSDT"])
        if current:
            self.symbol.setCurrentText(current)
        self.symbol.blockSignals(False)
        self.refresh_coverage()

    def refresh_coverage(self):
        if not self.symbol.currentText() or self.start.date() >= self.end.date():
            return
        rows = self.service.catalog.coverage(self.request_model())
        self.coverage.setRowCount(len(rows))
        grouped = {}
        for row in rows:
            grouped.setdefault(
                self._dataset_family(row["dataset"]), []
            ).append(row["state"])
        self.datasets.setText(
            "\n".join(
                f"{family}: "
                f"{('AVAILABLE' if all(x == 'AVAILABLE' for x in states) else 'PARTIAL' if any(x == 'AVAILABLE' for x in states) else 'UNAVAILABLE')}"
                for family, states in sorted(grouped.items())
            ) or "Required candle data: UNAVAILABLE"
        )
        for row_number, row in enumerate(rows):
            values = (
                row["dataset"], row["interval"] or "—",
                row["first_period"] or "—", row["last_period"] or "—",
                row["archive_count"], row["state"],
            )
            for column, value in enumerate(values):
                self.coverage.setItem(
                    row_number, column, QTableWidgetItem(str(value))
                )
        self.refresh_data_library()
        if hasattr(self, "current_research"):
            self._refresh_summary_from_widgets()

    def start_run(self):
        try:
            request, config = self.request_model(), self.build_config()
            config.validate()
        except Exception as exc:
            QMessageBox.warning(
                self, "Invalid research request", str(exc)
            )
            return
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.stage.setText("Native research executing")
        self._thread = QThread(self)
        self._worker = RunWorker(self.service, request, config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _finished(self, result):
        self.run_button.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.stage.setText("COMPLETED")
        self._run_dir = Path(result.output_dir)
        self._manifest, summary = self.service.completed_runs.read(self._run_dir)
        self.summary.setText(
            "\n".join(
                f"{key}: {summary.get(key, '—')}" for key in (
                    "ending_equity", "net_pnl", "total_return_percentage",
                    "total_trades", "wins", "losses", "win_rate",
                    "total_net_r", "average_net_r", "profit_factor",
                    "maximum_drawdown_percentage", "total_fees",
                )
            )
        )
        keys = (
            "total_trades", "wins", "losses", "win_rate", "total_net_r",
            "average_net_r", "net_pnl", "ending_equity", "profit_factor",
            "maximum_drawdown_percentage", "total_fees",
        )
        for (label, card), key in zip(self.kpi_cards.items(), keys):
            card.setText(f"{label}\n{summary.get(key, '—')}")
        stage_timings = self._manifest.get(
            "execution_result", {}
        ).get("stage_timings", {})
        self.timings.setText(
            "Run timings: " + (
                " · ".join(
                    f"{name.replace('_', ' ').title()} {value}s"
                    for name, value in stage_timings.items()
                ) or "not recorded"
            )
        )
        self.render_resolution(self._manifest)
        self.render_data_quality(result.data_quality)
        self.open_folder.setEnabled(True)
        for key, button in self.artifact_buttons.items():
            button.setEnabled(key in self._manifest.get("artifacts", {}))

    def render_data_quality(self, report):
        if report is None:
            self.quality.setText("Data quality: NOT AVAILABLE")
            self.quality_table.setRowCount(0)
            return
        self.quality.setText("Data quality: " + report.overall_status.value)
        self.quality_table.setRowCount(len(report.datasets))
        for row, dataset in enumerate(report.datasets):
            values = (
                dataset.dataset, dataset.interval or "event", dataset.required,
                dataset.row_count, dataset.status.value,
                "; ".join(issue.code for issue in dataset.issues) or "—",
            )
            for column, value in enumerate(values):
                self.quality_table.setItem(
                    row, column, QTableWidgetItem(str(value))
                )

    def render_resolution(self, manifest):
        request = manifest["request"]
        requested = request.get("requested_intrabar_interval") or "none"
        effective = request.get("effective_intrabar_interval") or "none"
        fallback = "FALLBACK" if requested != effective else "AS REQUESTED"
        self.resolution.setText(
            f"Requested: {requested} | Effective: {effective} | {fallback}"
        )

    def _failed(self, message):
        self.run_button.setEnabled(True)
        self.progress.setRange(0, 100)
        self.stage.setText("FAILED: " + message)

    def open_artifact(self, key):
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(
                str(self.service.completed_runs.artifact_path(
                    self._run_dir, self._manifest, key
                ))
            )
        )

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Output root", self.output_root.text()
        )
        if path:
            self.output_root.setText(path)

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save native v3 config", "research-v3.json", "JSON (*.json)"
        )
        if path:
            self.service.save_config(Path(path), self.build_config())

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load native v3 config", "", "JSON (*.json)"
        )
        if path:
            self.apply_config(self.service.load_config(Path(path)))

    def start_post_show_tasks(self):
        pass
