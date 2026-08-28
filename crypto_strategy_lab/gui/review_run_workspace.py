"""Compact final-review workspace for the active native v3 GUI."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crypto_strategy_lab.strategy_rule_model import (
    SUPPORT_RESISTANCE_RULE_EVIDENCE,
    common_execution_profile,
)
from .rule_strategy_builder import DIRECTION_LABELS
from .v2_main_window import TIMEFRAME_LABELS, display_percentage, timeframe_label


PRESSURE_EVIDENCE = {
    "DI_PRESSURE_STATE",
    "DI_SPREAD_CHANGE",
    "DIRECTIONAL_DI_CHANGE",
    "OPPOSING_DI_CHANGE",
}


class ReviewRunWorkspace(QWidget):
    """One purpose: show exactly what will run, whether it is ready, and run it."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.plan_values: dict[str, QLabel] = {}
        self.risk_values: dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.plan_box = QGroupBox("Research Plan")
        plan = QGridLayout(self.plan_box)
        self.plan_heading = QLabel()
        self.plan_heading.setStyleSheet("font-size:16px; font-weight:700")
        plan.addWidget(self.plan_heading, 0, 0, 1, 2)
        for row, (key, label) in enumerate(
            (
                ("direction", "Direction"),
                ("markets", "Markets"),
                ("entry", "Entry Filters"),
                ("pressure", "DI Pressure"),
                ("mr", "Mean Reversion"),
                ("sr", "Support / Resistance"),
                ("extra", "Additional Research"),
            ),
            1,
        ):
            title = QLabel(label)
            title.setStyleSheet("color:#52606d")
            value = QLabel("—")
            value.setWordWrap(True)
            self.plan_values[key] = value
            plan.addWidget(title, row, 0)
            plan.addWidget(value, row, 1)
        plan.setColumnStretch(1, 1)
        outer.addWidget(self.plan_box)

        self.risk_box = QGroupBox("Risk & Execution")
        risk_layout = QGridLayout(self.risk_box)
        for row, (key, label) in enumerate(
            (
                ("equity", "Starting Equity"),
                ("risk", "Base Risk / Trade"),
                ("stop_target", "Stop / Target"),
                ("max_trades", "Maximum Active Trades"),
                ("concurrent", "Base Planned Concurrent Risk"),
            )
        ):
            title = QLabel(label)
            title.setStyleSheet("color:#52606d")
            value = QLabel("—")
            self.risk_values[key] = value
            risk_layout.addWidget(title, row, 0)
            risk_layout.addWidget(value, row, 1)
        risk_layout.setColumnStretch(1, 1)
        outer.addWidget(self.risk_box)

        self.readiness_box = QGroupBox("Run Readiness")
        readiness = QVBoxLayout(self.readiness_box)
        self.readiness_title = QLabel("CHECKING DATA…")
        self.readiness_title.setStyleSheet(
            "font-size:18px; font-weight:700; color:#52606d"
        )
        readiness.addWidget(self.readiness_title)
        self.readiness_detail = QLabel()
        self.readiness_detail.setWordWrap(True)
        self.readiness_detail.setStyleSheet("color:#52606d")
        readiness.addWidget(self.readiness_detail)

        self.go_setup = QPushButton("Go to Setup")
        self.go_setup.clicked.connect(lambda: self.window.pages.setCurrentIndex(0))
        readiness.addWidget(self.go_setup, 0, Qt.AlignmentFlag.AlignRight)

        self.progress_status = getattr(window, "run_progress_status", None)
        if self.progress_status is not None:
            readiness.addWidget(self.progress_status)
            self.progress_status.hide()
        outer.addWidget(self.readiness_box)

        output_box = QGroupBox("Output")
        output = QGridLayout(output_box)
        output.addWidget(QLabel("Saved as"), 0, 0)
        output.addWidget(QLabel("Canonical completed-run artifact set"), 0, 1)
        output.addWidget(QLabel("Folder"), 1, 0)
        self.output_path = QLabel()
        self.output_path.setWordWrap(True)
        self.output_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output.addWidget(self.output_path, 1, 1)
        output.setColumnStretch(1, 1)
        outer.addWidget(output_box)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(window.save)
        actions.addWidget(window.load)
        actions.addStretch()
        window.run_button.setMinimumWidth(170)
        window.run_button.setMinimumHeight(36)
        window.run_button.setStyleSheet("font-weight:700")
        actions.addWidget(window.run_button)
        outer.addLayout(actions)
        outer.addStretch()

        window.output_root.textChanged.connect(self.output_path.setText)
        self.output_path.setText(window.output_root.text())

    @staticmethod
    def _count_rule_evidence(rules, evidence_names: set[str]) -> int:
        return sum(rule.get("evidence") in evidence_names for rule in rules)

    def refresh(self, config=None) -> None:
        if config is None:
            try:
                config = self.window.build_config()
            except Exception:
                config = self.window.config

        symbol = self.window.symbol.currentText() or "BTCUSDT"
        tf = TIMEFRAME_LABELS[timeframe_label(config.data.strategy_timeframe_minutes)]
        self.plan_heading.setText(f"{symbol} · {tf}")

        builder = getattr(self.window, "rule_builder", None)
        if builder is not None:
            direction = builder.direction_mode.currentData()
            direction_text = DIRECTION_LABELS.get(direction, str(direction))
            permissions = builder.market_permissions()
            required_rules = tuple(builder.required_rules.rules())
            veto_rules = tuple(builder.veto_rules.rules())
            flip_rules = tuple(builder.flip_rules.rules())
        else:
            direction_text = "DI Direction" if config.strategy.enable_di_direction_selection else "Native direction"
            permissions = [
                key for key, profile in config.strategy.profiles.items() if profile.enabled
            ]
            required_rules = veto_rules = flip_rules = ()

        if len(permissions) == 6:
            markets_text = "All 6 environments"
        elif permissions:
            markets_text = ", ".join(
                item.replace("_", " ").title() for item in permissions
            )
        else:
            markets_text = "None"

        if not required_rules and not veto_rules and not flip_rules:
            entry_text = f"None — {direction_text} only"
        else:
            parts = []
            if required_rules:
                parts.append(f"{len(required_rules)} required")
            if veto_rules:
                parts.append(f"{len(veto_rules)} veto")
            if flip_rules:
                parts.append(f"{len(flip_rules)} flip")
            entry_text = " · ".join(parts)

        all_rules = (*required_rules, *veto_rules, *flip_rules)
        pressure_count = self._count_rule_evidence(all_rules, PRESSURE_EVIDENCE)
        if pressure_count:
            pressure_text = f"Used by {pressure_count} entry rule(s)"
        elif getattr(config.strategy, "enable_di_pressure_analysis", False):
            pressure_text = "Research context only"
        else:
            pressure_text = "Off"

        mr_text = (
            "Research context only"
            if config.strategy.enable_mean_reversion_analysis
            else "Off"
        )
        sr_count = self._count_rule_evidence(
            all_rules, set(SUPPORT_RESISTANCE_RULE_EVIDENCE)
        )
        if sr_count:
            sr_text = f"Used by {sr_count} entry / veto rule(s)"
        elif config.features.enable_support_resistance_analysis:
            sr_text = "Research context only"
        else:
            sr_text = "Off"

        extras = []
        if config.features.trade_flow_enabled:
            extras.append("Detailed Trade Flow")
        if config.features.order_book_enabled:
            extras.append("Historical Order Book")
        extra_text = " · ".join(extras) if extras else "None"

        self.plan_values["direction"].setText(direction_text)
        self.plan_values["markets"].setText(markets_text)
        self.plan_values["entry"].setText(entry_text)
        self.plan_values["pressure"].setText(pressure_text)
        self.plan_values["mr"].setText(mr_text)
        self.plan_values["sr"].setText(sr_text)
        self.plan_values["extra"].setText(extra_text)

        risk_pct = float(config.execution.risk_per_leg)
        equity = float(config.execution.initial_equity)
        max_trades = int(config.execution.max_active_pairs)
        risk_dollars = equity * risk_pct
        concurrent_pct = risk_pct * max_trades
        concurrent_dollars = equity * concurrent_pct
        base_execution = common_execution_profile(config.execution.profiles)
        if hasattr(self.window, "base_execution_form"):
            try:
                base_execution = self.window.base_execution_form.value(base_execution)
            except Exception:
                pass

        self.risk_values["equity"].setText(f"${equity:,.2f}")
        self.risk_values["risk"].setText(
            f"{display_percentage(risk_pct)} · ${risk_dollars:,.2f}"
        )
        self.risk_values["stop_target"].setText(
            f"{base_execution.stop_loss_multiple:g} distance units · {base_execution.reward_risk_ratio:g}R"
        )
        self.risk_values["max_trades"].setText(str(max_trades))
        self.risk_values["concurrent"].setText(
            f"{display_percentage(concurrent_pct)} · ${concurrent_dollars:,.2f}"
        )

        self.output_path.setText(self.window.output_root.text())
        self.refresh_readiness()

    def refresh_readiness(self) -> None:
        source_title = getattr(self.window, "readiness_state", None)
        source_detail = getattr(self.window, "range_validation", None)
        title = source_title.text() if source_title is not None else str(self.window._data_state())
        detail = source_detail.text() if source_detail is not None else ""
        upper = title.upper()
        data_state = str(self.window._data_state()).upper()
        blocked = "BLOCK" in upper or "NOT READY" in upper or data_state == "BLOCKED"
        ready = "READY TO RUN" in upper or (data_state == "READY" and "CHECK" not in upper)
        validation_running = getattr(self.window, "_validation_thread", None) is not None
        run_running = getattr(self.window, "_thread", None) is not None

        if blocked:
            style = "font-size:18px; font-weight:700; color:#a61b1b"
            self.readiness_detail.setStyleSheet("color:#a61b1b; font-weight:600")
        elif ready:
            style = "font-size:18px; font-weight:700; color:#1f6f43"
            self.readiness_detail.setStyleSheet("color:#1f6f43; font-weight:600")
        else:
            style = "font-size:18px; font-weight:700; color:#52606d"
            self.readiness_detail.setStyleSheet("color:#52606d")

        self.readiness_title.setText(title)
        self.readiness_title.setStyleSheet(style)
        self.readiness_detail.setText(detail)
        self.go_setup.setVisible(blocked)

        if run_running:
            self.window.run_button.setText("Running…")
            self.window.run_button.setEnabled(False)
        elif validation_running:
            self.window.run_button.setText("Validating…")
            self.window.run_button.setEnabled(False)
        elif blocked:
            self.window.run_button.setText("Run Backtest")
            self.window.run_button.setEnabled(False)
        elif ready:
            self.window.run_button.setText("Run Backtest")
            self.window.run_button.setEnabled(True)
        else:
            self.window.run_button.setText("Validate & Run")
            self.window.run_button.setEnabled(True)

        if self.progress_status is not None:
            if validation_running or run_running:
                self.progress_status.show()
            elif not any(word in self.window.stage.text().upper() for word in ("COMPLETED", "FAILED")):
                self.progress_status.hide()
