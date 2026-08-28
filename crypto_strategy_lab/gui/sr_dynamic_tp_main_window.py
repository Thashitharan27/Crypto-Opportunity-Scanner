"""GUI layer for optional support/resistance-capped take-profit targets."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel

import crypto_strategy_lab.gui.worker as worker_module
import crypto_strategy_lab.portfolio as portfolio_module
from crypto_strategy_lab.sr_dynamic_tp_engine import SRDynamicTPBacktestEngine
from crypto_strategy_lab.gui.enhanced_config import SR_DYNAMIC_TP_DEFAULTS
from crypto_strategy_lab.gui.enhanced_main_window import MainWindow as EnhancedMainWindow


# The normal app already routes through EnhancedMainWindow. Replace only the
# calculation engine so all existing DI/MR/HTF-SR behaviour is preserved.
worker_module.BacktestEngine = SRDynamicTPBacktestEngine
portfolio_module.BacktestEngine = SRDynamicTPBacktestEngine


class MainWindow(EnhancedMainWindow):
    """Enhanced main window with S/R-aware TP and optional market presets."""

    @staticmethod
    def _timeframe_minutes(label):
        """Support minute, hour, and daily strategy data."""
        text = str(label).strip().lower()
        if text.endswith("d"):
            return int(text[:-1]) * 1440
        if text.endswith("h"):
            return int(text[:-1]) * 60
        if text.endswith("m"):
            return int(text[:-1])
        raise ValueError(f"Unsupported timeframe: {label}")

    @staticmethod
    def _timeframe_label(minutes):
        minutes = int(minutes)
        if minutes >= 1440 and minutes % 1440 == 0:
            return f"{minutes // 1440}d"
        if minutes >= 60 and minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"

    def _build_config(self):
        super()._build_config()

        # Daily candles are useful for exchange-traded assets and higher-timeframe
        # swing research while preserving every existing crypto timeframe/default.
        if self.strategy_timeframe.findText("1d") < 0:
            self.strategy_timeframe.addItem("1d")

        self.market_preset_box = QGroupBox("Market / Strategy Preset")
        preset_form = QFormLayout(self.market_preset_box)
        self.market_preset = QComboBox()
        self.market_preset.addItem("Crypto / existing settings", "CRYPTO")
        self.market_preset.addItem("Crypto Swing (4H baseline)", "CRYPTO_SWING")
        self.market_preset.addItem("Sri Lanka Stocks (CSE daily)", "CSE_DAILY")
        self.market_preset_status = QLabel(
            "Crypto / existing settings: no automatic changes are applied."
        )
        self.market_preset_status.setWordWrap(True)
        preset_form.addRow("Preset", self.market_preset)
        preset_form.addRow("", self.market_preset_status)

        # Backtest Setup contains the toolbar followed by the main scroll area.
        # Put the small preset box between them so it stays visible and removable.
        self.backtest_setup_page.layout().insertWidget(1, self.market_preset_box)
        self.market_preset.currentIndexChanged.connect(self._apply_market_preset)

    def _set_asset_structural_regime(self):
        if not hasattr(self, "profile_editor"):
            return
        idx = self.profile_editor.regime_method.findData("ASSET_STRUCTURAL")
        if idx >= 0:
            self.profile_editor.regime_method.setCurrentIndex(idx)

    def _apply_market_preset(self, *_):
        if not hasattr(self, "market_preset"):
            return
        mode = str(self.market_preset.currentData() or "CRYPTO")
        if mode == "CRYPTO":
            self.market_preset_status.setText(
                "Crypto / existing settings: no automatic changes are applied."
            )
            return

        if mode == "CRYPTO_SWING":
            # Research-first slower swing baseline: 4H entries and the asset's
            # own structural regime. Existing Strategy Profile exits remain
            # untouched so the slower horizon can be compared without silently
            # changing stop/target semantics.
            self.strategy_timeframe.setCurrentText("4h")
            self.use_intrabar.setChecked(False)
            self.enable_daily_schedule.setChecked(False)
            self._set_asset_structural_regime()

            # Keep S/R on the 4H strategy timeframe for the baseline. The optional
            # resampled Daily S/R path is deliberately not forced by this preset;
            # it can be tested separately after its timestamp handling is hardened.
            self.enable_support_resistance_analysis.setChecked(True)
            if hasattr(self, "sr_analyze_only"):
                self.sr_analyze_only.setChecked(True)
            if hasattr(self, "sr_timeframe"):
                idx = self.sr_timeframe.findData(0)
                if idx >= 0:
                    self.sr_timeframe.setCurrentIndex(idx)

            # DI/MR remain descriptive on the first swing baseline. Ensure all DI
            # pressure states are admitted so an older filtered config cannot turn
            # the preset into an accidental optimization.
            self.enable_di_pressure_analysis.setChecked(True)
            for name in (
                "di_pressure_allow_expanding",
                "di_pressure_allow_contracting",
                "di_pressure_allow_mixed",
            ):
                control = getattr(self, name, None)
                if control is not None:
                    control.setChecked(True)
            self.enable_mean_reversion_analysis.setChecked(True)

            self.input_csv.setPlaceholderText(
                "Select 4H crypto OHLCV CSV for swing research"
            )
            self.market_preset_status.setText(
                "Crypto Swing preset active: 4H entries, intrabar exits OFF, scheduled entry OFF, "
                "Market Regime = Selected asset structural trend, and S/R uses the same 4H strategy "
                "timeframe in analysis-only mode. DI expansion/contraction admits all states and MR remains "
                "analysis telemetry. Existing Strategy Profile stop/TP/trailing settings are intentionally "
                "preserved. Daily S/R remains available for separate testing but is not forced by this preset."
            )
            self.update_dynamic()
            self.update_planned_output()
            return

        # CSE research starts from daily OHLCV and the selected asset's own
        # structural regime. These are existing engine features, not a new
        # stock-specific calculation path.
        if self.strategy_timeframe.findText("1d") < 0:
            self.strategy_timeframe.addItem("1d")
        self.strategy_timeframe.setCurrentText("1d")
        self.use_intrabar.setChecked(False)
        self.enable_daily_schedule.setChecked(False)
        self._set_asset_structural_regime()

        self.input_csv.setPlaceholderText(
            "Browse CSE daily OHLCV CSV: Date, Open, High, Low, Close, Volume"
        )
        self.market_preset_status.setText(
            "CSE daily preset active: Strategy Timeframe = 1d, intrabar exits = OFF, "
            "scheduled entry = OFF, and Market Regime = Selected asset structural trend. "
            "ATR, DI, MR, S/R and Strategy Profiles use the same engine as crypto. Review "
            "fees/slippage for your stock assumptions. SHORT profiles remain research-only "
            "unless the instrument/venue actually permits short selling."
        )
        self.update_dynamic()
        self.update_planned_output()

    def _build_support_resistance_tab(self):
        super()._build_support_resistance_tab()

        self.sr_take_profit_box = QGroupBox("Take Profit from S/R Room")
        tp_form = QFormLayout(self.sr_take_profit_box)

        self.sr_take_profit_mode = QComboBox()
        self.sr_take_profit_mode.addItem("Fixed R (baseline)", "FIXED_R")
        self.sr_take_profit_mode.addItem("Cap TP at next S/R level", "SR_CAPPED_R")

        self.sr_take_profit_maximum_r = QDoubleSpinBox()
        self.sr_take_profit_maximum_r.setRange(0.1, 100.0)
        self.sr_take_profit_maximum_r.setDecimals(2)
        self.sr_take_profit_maximum_r.setSingleStep(0.1)
        self.sr_take_profit_maximum_r.setSuffix(" R")

        self.sr_take_profit_minimum_r = QDoubleSpinBox()
        self.sr_take_profit_minimum_r.setRange(0.1, 100.0)
        self.sr_take_profit_minimum_r.setDecimals(2)
        self.sr_take_profit_minimum_r.setSingleStep(0.1)
        self.sr_take_profit_minimum_r.setSuffix(" R")

        self.sr_take_profit_buffer_r = QDoubleSpinBox()
        self.sr_take_profit_buffer_r.setRange(0.0, 10.0)
        self.sr_take_profit_buffer_r.setDecimals(2)
        self.sr_take_profit_buffer_r.setSingleStep(0.05)
        self.sr_take_profit_buffer_r.setSuffix(" R")

        self.sr_take_profit_no_level_policy = QComboBox()
        self.sr_take_profit_no_level_policy.addItem("Use normal fixed TP", "USE_FIXED_TP")
        self.sr_take_profit_no_level_policy.addItem("Reject trade", "REJECT_TRADE")

        self.sr_take_profit_status = QLabel()
        self.sr_take_profit_status.setWordWrap(True)
        help_text = QLabel(
            "When S/R-capped mode is active, the engine measures the next resistance for LONG or next support "
            "for SHORT in initial-stop R units. The final target is min(normal strategy TP, Maximum TP, "
            "available S/R room − buffer). If that result is below Minimum TP, the entry is rejected. "
            "Fixed R remains the default baseline."
        )
        help_text.setWordWrap(True)

        tp_form.addRow("Take Profit Mode", self.sr_take_profit_mode)
        tp_form.addRow("Maximum TP", self.sr_take_profit_maximum_r)
        tp_form.addRow("Minimum Acceptable TP", self.sr_take_profit_minimum_r)
        tp_form.addRow("S/R Target Buffer", self.sr_take_profit_buffer_r)
        tp_form.addRow("If No Opposing S/R", self.sr_take_profit_no_level_policy)
        tp_form.addRow("", self.sr_take_profit_status)
        tp_form.addRow("", help_text)

        parent_layout = self.sr_detection_box.parentWidget().layout()
        anchor = getattr(self, "sr_entry_rules_box", self.sr_detection_box)
        index = parent_layout.indexOf(anchor)
        parent_layout.insertWidget(index + 1 if index >= 0 else parent_layout.count(), self.sr_take_profit_box)

        self.sr_take_profit_mode.currentIndexChanged.connect(self.update_dynamic)
        self.sr_take_profit_maximum_r.valueChanged.connect(self.update_dynamic)
        self.sr_take_profit_minimum_r.valueChanged.connect(self.update_dynamic)
        self.sr_take_profit_buffer_r.valueChanged.connect(self.update_dynamic)
        self.sr_take_profit_no_level_policy.currentIndexChanged.connect(self.update_dynamic)
        self.enable_support_resistance_analysis.toggled.connect(self.update_dynamic)

    def values(self):
        values = super().values()
        values.update(
            {
                "sr_take_profit_mode": str(self.sr_take_profit_mode.currentData() or "FIXED_R"),
                "sr_take_profit_maximum_r": self.sr_take_profit_maximum_r.value(),
                "sr_take_profit_minimum_r": self.sr_take_profit_minimum_r.value(),
                "sr_take_profit_buffer_r": self.sr_take_profit_buffer_r.value(),
                "sr_take_profit_no_level_policy": str(
                    self.sr_take_profit_no_level_policy.currentData() or "USE_FIXED_TP"
                ),
            }
        )
        return values

    def apply_values(self, values):
        merged = {**SR_DYNAMIC_TP_DEFAULTS, **values}
        super().apply_values(merged)

        mode_index = self.sr_take_profit_mode.findData(str(merged["sr_take_profit_mode"]).upper())
        self.sr_take_profit_mode.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self.sr_take_profit_maximum_r.setValue(float(merged["sr_take_profit_maximum_r"]))
        self.sr_take_profit_minimum_r.setValue(float(merged["sr_take_profit_minimum_r"]))
        self.sr_take_profit_buffer_r.setValue(float(merged["sr_take_profit_buffer_r"]))
        policy_index = self.sr_take_profit_no_level_policy.findData(
            str(merged["sr_take_profit_no_level_policy"]).upper()
        )
        self.sr_take_profit_no_level_policy.setCurrentIndex(policy_index if policy_index >= 0 else 0)
        self.update_dynamic()

    def update_dynamic(self):
        super().update_dynamic()
        if not hasattr(self, "sr_take_profit_mode"):
            return

        sr_enabled = self.enable_support_resistance_analysis.isChecked()
        mode = str(self.sr_take_profit_mode.currentData() or "FIXED_R")
        if not sr_enabled and mode == "SR_CAPPED_R":
            self.sr_take_profit_mode.blockSignals(True)
            self.sr_take_profit_mode.setCurrentIndex(self.sr_take_profit_mode.findData("FIXED_R"))
            self.sr_take_profit_mode.blockSignals(False)
            mode = "FIXED_R"

        self.sr_take_profit_mode.setEnabled(sr_enabled)
        active = sr_enabled and mode == "SR_CAPPED_R"
        for control in (
            self.sr_take_profit_maximum_r,
            self.sr_take_profit_minimum_r,
            self.sr_take_profit_buffer_r,
            self.sr_take_profit_no_level_policy,
        ):
            control.setEnabled(active)

        maximum_r = self.sr_take_profit_maximum_r.value()
        minimum_r = self.sr_take_profit_minimum_r.value()
        if not sr_enabled:
            text = "S/R TP: DISABLED — fixed Strategy Profile TP is used."
        elif mode == "FIXED_R":
            text = "S/R TP Mode: FIXED R BASELINE — S/R does not change the target."
        elif minimum_r > maximum_r:
            text = "S/R TP Mode: INVALID — Minimum Acceptable TP cannot exceed Maximum TP."
        else:
            no_level = (
                "use the normal fixed TP"
                if self.sr_take_profit_no_level_policy.currentData() == "USE_FIXED_TP"
                else "reject the trade"
            )
            text = (
                f"S/R TP Mode: ACTIVE — target up to {maximum_r:.2f}R, reject below {minimum_r:.2f}R, "
                f"leave {self.sr_take_profit_buffer_r.value():.2f}R before the level; if no level, {no_level}."
            )
        self.sr_take_profit_status.setText(text)