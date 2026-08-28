"""Researcher-facing Risk & Execution composition.

The native ExecutionConfig and ExecutionProfileConfig forms remain authoritative.
This module only reorganizes their existing widgets into a mode-aware workspace so
researchers see the controls that matter for the selected execution plan.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class FormCard(QGroupBox):
    """Small card that can hide individual form rows without replacing widgets."""

    def __init__(self, title: str, *, note: str = "", parent=None):
        super().__init__(title, parent)
        outer = QVBoxLayout(self)
        if note:
            note_label = QLabel(note)
            note_label.setWordWrap(True)
            note_label.setStyleSheet("color:#52606d")
            outer.addWidget(note_label)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(8, 4, 8, 4)
        outer.addWidget(self.body)
        self.rows: dict[str, tuple[QLabel | None, QWidget]] = {}

    def add_field(self, name: str, label: str, widget: QWidget) -> None:
        label_widget = QLabel(label)
        self.form.addRow(label_widget, widget)
        self.rows[name] = (label_widget, widget)

    def add_control(self, name: str, widget: QWidget) -> None:
        self.form.addRow(widget)
        self.rows[name] = (None, widget)

    def set_row_visible(self, name: str, visible: bool) -> None:
        row = self.rows.get(name)
        if row is None:
            return
        label, widget = row
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)


class DisclosureCard(QGroupBox):
    """Advanced settings card collapsed by default."""

    def __init__(self, title: str, *, note: str, expanded: bool = False, parent=None):
        super().__init__(title, parent)
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        explanation = QLabel(note)
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color:#52606d")
        top.addWidget(explanation, 1)
        self.toggle = QToolButton()
        self.toggle.setText("Show settings")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        top.addWidget(self.toggle)
        outer.addLayout(top)

        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(8, 4, 8, 4)
        outer.addWidget(self.body)
        self.rows: dict[str, tuple[QLabel | None, QWidget]] = {}
        self.toggle.toggled.connect(self._toggle_body)
        self._toggle_body(expanded)

    def _toggle_body(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self.toggle.setText("Hide settings" if checked else "Show settings")

    def add_field(self, name: str, label: str, widget: QWidget) -> None:
        label_widget = QLabel(label)
        self.form.addRow(label_widget, widget)
        self.rows[name] = (label_widget, widget)

    def add_control(self, name: str, widget: QWidget) -> None:
        self.form.addRow(widget)
        self.rows[name] = (None, widget)

    def set_row_visible(self, name: str, visible: bool) -> None:
        row = self.rows.get(name)
        if row is None:
            return
        label, widget = row
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)


class RiskExecutionWorkspace(QWidget):
    """Friendly composition over the two authoritative execution dataclass forms."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.execution_form = window.execution_form
        self.base_form = window.base_execution_form
        self.account = self.execution_form.widgets
        self.trade = self.base_form.widgets

        # Keep the authoritative forms alive for build/apply/round-trip behavior,
        # but remove their old long-form presentation from the active page.
        self.execution_form.setParent(window)
        self.base_form.setParent(window)
        self.execution_form.hide()
        self.base_form.hide()

        self._prepare_widgets()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.plan_box = QGroupBox("Effective Plan")
        plan_layout = QVBoxLayout(self.plan_box)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "font-size:13px; background:#f7f9fb; padding:10px; border:1px solid #d9e2ec"
        )
        plan_layout.addWidget(self.summary_label)
        layout.addWidget(self.plan_box)

        self.account_card = FormCard(
            "1. Account Risk & Position Sizing",
            note="These controls define the account risk budget and how many positions may be open at once.",
        )
        self.account_card.add_field("initial_equity", "Starting Equity", self.account["initial_equity"])
        self.account_card.add_field("risk_per_leg", "Risk Per Trade", self.account["risk_per_leg"])
        self.account_card.add_field("max_active_pairs", "Maximum Active Trades", self.account["max_active_pairs"])
        layout.addWidget(self.account_card)

        self.stop_card = FormCard(
            "2. Stop & Position Sizing",
            note="Choose one distance method. Only the parameter used by that method is shown.",
        )
        self.stop_card.add_field("risk_mode", "Stop Distance Method", self.account["risk_mode"])
        self.stop_card.add_field("atr_multiplier", "ATR Distance Multiplier", self.account["atr_multiplier"])
        self.stop_card.add_field("percent_r", "Price Distance", self.account["percent_r"])
        self.stop_card.add_field("fixed_r", "Fixed Price Distance", self.account["fixed_r"])
        self.stop_card.add_field("stop_loss_multiple", "Stop Multiplier", self.trade["stop_loss_multiple"])
        layout.addWidget(self.stop_card)

        self.target_card = FormCard(
            "3. Profit Target",
            note="Fixed R is the baseline. S/R-Constrained Target can cap the target and reject trades that do not have enough room.",
        )
        self.target_card.add_field("sr_take_profit_mode", "Target Policy", self.account["sr_take_profit_mode"])
        self.target_card.add_field("reward_risk_ratio", "Base Profit Target", self.trade["reward_risk_ratio"])
        self.target_card.add_field("sr_take_profit_minimum_r", "Minimum Acceptable Target", self.account["sr_take_profit_minimum_r"])
        self.target_card.add_field("sr_take_profit_maximum_r", "Maximum S/R Target Cap", self.account["sr_take_profit_maximum_r"])
        self.target_card.add_field("sr_take_profit_buffer_r", "Buffer Before S/R", self.account["sr_take_profit_buffer_r"])
        self.target_card.add_field("sr_take_profit_no_level_policy", "If No Opposing S/R Exists", self.account["sr_take_profit_no_level_policy"])
        self.sr_dependency_note = QLabel(
            "S/R target policy automatically requires causal Support / Resistance calculation; no separate enable step is needed."
        )
        self.sr_dependency_note.setWordWrap(True)
        self.sr_dependency_note.setStyleSheet("color:#52606d; margin:2px 8px 6px 8px")
        self.target_card.layout().addWidget(self.sr_dependency_note)
        layout.addWidget(self.target_card)

        self.management_card = FormCard(
            "4. Trade Management",
            note="Optional management stays compact while disabled. Enabling a feature reveals only its relevant settings.",
        )
        self.management_card.add_control("break_even_enabled", self.trade["break_even_enabled"])
        self.management_card.add_field("break_even_activation_r", "Break-even Activation", self.trade["break_even_activation_r"])
        self.management_card.add_field("break_even_offset_r", "Break-even Offset", self.trade["break_even_offset_r"])

        self.management_card.add_control("trailing_enabled", self.trade["trailing_enabled"])
        self.management_card.add_field("trailing_activation_r", "Trailing Activation", self.trade["trailing_activation_r"])
        self.management_card.add_field("trailing_distance_r", "Trailing Distance", self.trade["trailing_distance_r"])

        self.management_card.add_control("partial_profit_enabled", self.trade["partial_profit_enabled"])
        self.management_card.add_field("tp1_r", "First Profit Level", self.trade["tp1_r"])
        self.management_card.add_field("tp1_close_pct", "First Profit Close", self.trade["tp1_close_pct"])
        self.management_card.add_field("tp2_r", "Final Profit Level", self.trade["tp2_r"])

        self.management_card.add_control("partial_stop_enabled", self.trade["partial_stop_enabled"])
        self.management_card.add_field("sl1_r", "First Stop Level", self.trade["sl1_r"])
        self.management_card.add_field("sl1_close_pct", "First Stop Close", self.trade["sl1_close_pct"])
        self.management_card.add_field("sl2_r", "Final Stop Level", self.trade["sl2_r"])

        self.management_card.add_control("timeout_enabled", self.trade["timeout_enabled"])
        self.management_card.add_field("timeout_minutes", "Maximum Holding Time", self.trade["timeout_minutes"])
        layout.addWidget(self.management_card)

        self.advanced_card = DisclosureCard(
            "Advanced Trade Management",
            note="Less-common research controls: risk override, R-step trailing and ATR-checkpoint target extension.",
            expanded=False,
        )
        self.advanced_card.add_field("risk_multiplier", "Risk Multiplier", self.trade["risk_multiplier"])
        self.advanced_card.add_control("r_step_trailing_enabled", self.trade["r_step_trailing_enabled"])
        self.advanced_card.add_field("r_step_activation_r", "R-step Activation", self.trade["r_step_activation_r"])
        self.advanced_card.add_field("r_step_distance_r", "R-step Distance", self.trade["r_step_distance_r"])
        self.advanced_card.add_field("r_step_size_r", "R-step Size", self.trade["r_step_size_r"])
        self.advanced_card.add_field("r_step_maximum_r", "R-step Maximum", self.trade["r_step_maximum_r"])
        self.advanced_card.add_field("r_step_activation_close_pct", "Activation Close %", self.trade["r_step_activation_close_pct"])
        self.advanced_card.add_control("atr_checkpoint_tp_extension_enabled", self.trade["atr_checkpoint_tp_extension_enabled"])
        self.advanced_card.add_field("atr_checkpoint_di_spread_minimum", "Checkpoint Minimum DI Spread", self.trade["atr_checkpoint_di_spread_minimum"])
        self.advanced_card.add_field("atr_checkpoint_bb_width_minimum", "Checkpoint Minimum BB Width", self.trade["atr_checkpoint_bb_width_minimum"])
        self.advanced_card.add_field("atr_checkpoint_profit_lock_start", "Profit Lock Start", self.trade["atr_checkpoint_profit_lock_start"])
        self.advanced_card.add_field("atr_checkpoint_profit_lock_distance", "Profit Lock Distance", self.trade["atr_checkpoint_profit_lock_distance"])
        layout.addWidget(self.advanced_card)

        self.costs_card = DisclosureCard(
            "Exposure & Costs",
            note="Safety limits and execution assumptions are important, but normally do not need to occupy the main workflow.",
            expanded=False,
        )
        self.costs_card.add_field("max_effective_leverage_per_leg", "Maximum Effective Leverage Per Trade", self.account["max_effective_leverage_per_leg"])
        self.costs_card.add_field("max_combined_effective_leverage", "Maximum Combined Effective Leverage", self.account["max_combined_effective_leverage"])
        self.costs_card.add_field("maker_fee", "Maker Fee", self.account["maker_fee"])
        self.costs_card.add_field("taker_fee", "Taker Fee", self.account["taker_fee"])
        self.costs_card.add_control("use_maker_entry", self.account["use_maker_entry"])
        self.costs_card.add_control("use_maker_exit", self.account["use_maker_exit"])
        self.costs_card.add_field("slippage", "Slippage", self.account["slippage"])
        self.costs_card.add_field("tie_policy", "Same-bar Resolution", self.account["tie_policy"])
        self.costs_card.add_control("zero_cost_comparison", self.account["zero_cost_comparison"])
        layout.addWidget(self.costs_card)
        layout.addStretch()

        self.account["risk_mode"].currentIndexChanged.connect(self.refresh_visibility)
        self.account["sr_take_profit_mode"].currentIndexChanged.connect(self.refresh_visibility)
        for name in (
            "break_even_enabled", "trailing_enabled", "partial_profit_enabled",
            "partial_stop_enabled", "timeout_enabled", "r_step_trailing_enabled",
            "atr_checkpoint_tp_extension_enabled",
        ):
            self.trade[name].toggled.connect(self.refresh_visibility)

        self.refresh_visibility()
        self.refresh_summary_from_widgets()

    def _prepare_widgets(self) -> None:
        for name, text in {
            "break_even_enabled": "Enable break-even",
            "trailing_enabled": "Enable trailing stop",
            "partial_profit_enabled": "Enable partial profit-taking",
            "partial_stop_enabled": "Enable staged stop-loss",
            "timeout_enabled": "Enable maximum holding time",
            "r_step_trailing_enabled": "Enable R-step trailing",
            "atr_checkpoint_tp_extension_enabled": "Enable ATR-checkpoint target extension",
            "use_maker_entry": "Assume maker entry",
            "use_maker_exit": "Assume maker exit",
            "zero_cost_comparison": "Also calculate zero-cost comparison",
        }.items():
            widget = self.trade.get(name) or self.account.get(name)
            if isinstance(widget, QCheckBox):
                widget.setText(text)

        # These suffixes describe the composition better than the old generic
        # "distance units" wording and do not affect stored native values.
        if hasattr(self.trade["stop_loss_multiple"], "setSuffix"):
            self.trade["stop_loss_multiple"].setSuffix(" ×")
        if hasattr(self.account["atr_multiplier"], "setSuffix"):
            self.account["atr_multiplier"].setSuffix(" × ATR")

    def refresh_visibility(self, *_args) -> None:
        mode = str(self.account["risk_mode"].currentData() or "ATR")
        self.stop_card.set_row_visible("atr_multiplier", mode == "ATR")
        self.stop_card.set_row_visible("percent_r", mode == "PERCENT")
        self.stop_card.set_row_visible("fixed_r", mode == "FIXED")

        sr_target = str(self.account["sr_take_profit_mode"].currentData() or "FIXED_R") == "SR_CAPPED_R"
        for name in (
            "sr_take_profit_minimum_r", "sr_take_profit_maximum_r",
            "sr_take_profit_buffer_r", "sr_take_profit_no_level_policy",
        ):
            self.target_card.set_row_visible(name, sr_target)
        self.sr_dependency_note.setVisible(sr_target)

        toggles = {
            "break_even_enabled": ("break_even_activation_r", "break_even_offset_r"),
            "trailing_enabled": ("trailing_activation_r", "trailing_distance_r"),
            "partial_profit_enabled": ("tp1_r", "tp1_close_pct", "tp2_r"),
            "partial_stop_enabled": ("sl1_r", "sl1_close_pct", "sl2_r"),
            "timeout_enabled": ("timeout_minutes",),
        }
        for controller, fields in toggles.items():
            enabled = self.trade[controller].isChecked()
            for name in fields:
                self.management_card.set_row_visible(name, enabled)

        advanced = {
            "r_step_trailing_enabled": (
                "r_step_activation_r", "r_step_distance_r", "r_step_size_r",
                "r_step_maximum_r", "r_step_activation_close_pct",
            ),
            "atr_checkpoint_tp_extension_enabled": (
                "atr_checkpoint_di_spread_minimum", "atr_checkpoint_bb_width_minimum",
                "atr_checkpoint_profit_lock_start", "atr_checkpoint_profit_lock_distance",
            ),
        }
        for controller, fields in advanced.items():
            enabled = self.trade[controller].isChecked()
            for name in fields:
                self.advanced_card.set_row_visible(name, enabled)
        self.refresh_summary_from_widgets()

    @staticmethod
    def _distance_description(execution) -> str:
        mode = str(execution.risk_mode).upper()
        if mode == "ATR":
            return f"{execution.atr_multiplier:g}× ATR distance unit"
        if mode == "PERCENT":
            return f"{execution.percent_r * 100:g}% of price distance unit"
        return f"{execution.fixed_r:g} fixed-price distance unit"

    @staticmethod
    def _management_description(base) -> str:
        enabled = []
        if base.break_even_enabled:
            enabled.append("break-even")
        if base.trailing_enabled:
            enabled.append("trailing")
        if base.partial_profit_enabled:
            enabled.append("partial profit")
        if base.partial_stop_enabled:
            enabled.append("staged stop")
        if base.timeout_enabled:
            enabled.append("timeout")
        if base.r_step_trailing_enabled:
            enabled.append("R-step trailing")
        if base.atr_checkpoint_tp_extension_enabled:
            enabled.append("ATR-checkpoint extension")
        return ", ".join(enabled) if enabled else "no optional trade management"

    def update_summary(self, execution, base) -> None:
        effective_risk = float(execution.risk_per_leg) * float(base.risk_multiplier)
        risk_dollars = float(execution.initial_equity) * effective_risk
        stop_mult = float(base.sl2_r if base.partial_stop_enabled else base.stop_loss_multiple)
        if str(execution.sr_take_profit_mode).upper() == "SR_CAPPED_R":
            target = (
                f"S/R-constrained target (base {base.reward_risk_ratio:g}R, "
                f"minimum {execution.sr_take_profit_minimum_r:g}R, "
                f"cap {execution.sr_take_profit_maximum_r:g}R)"
            )
        else:
            target = f"fixed {base.reward_risk_ratio:g}R target"
        multiplier = (
            f" · risk multiplier {base.risk_multiplier:g}×"
            if abs(float(base.risk_multiplier) - 1.0) > 1e-12
            else ""
        )
        self.summary_label.setText(
            f"${execution.initial_equity:,.2f} equity · base risk {execution.risk_per_leg * 100:.2f}%"
            f"{multiplier} → effective risk budget {effective_risk * 100:.2f}% (${risk_dollars:,.2f}). "
            f"Stop distance uses {self._distance_description(execution)} with a {stop_mult:g}× stop multiplier. "
            f"Profit policy: {target}. Maximum active trades: {execution.max_active_pairs}. "
            f"Management: {self._management_description(base)}."
        )

    def refresh_summary_from_widgets(self) -> None:
        try:
            execution = self.execution_form.value()
            base = self.base_form.value()
        except (TypeError, ValueError):
            return
        self.update_summary(execution, base)
