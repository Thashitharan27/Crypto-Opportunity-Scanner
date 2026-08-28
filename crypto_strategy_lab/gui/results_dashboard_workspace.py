"""Human-focused completed-run dashboard for the active native v3 GUI."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

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


PRIMARY_ARTIFACTS = (
    ("workbook", "Open Workbook"),
    ("trade_csv", "Open Trade List"),
)

RESEARCH_ARTIFACTS = (
    ("summary", "Summary JSON"),
    ("trades", "Trades Parquet"),
    ("signals", "Signals Parquet"),
    ("feature_context", "Feature Context"),
    ("data_quality", "Data Quality"),
    ("source_archives", "Source Provenance"),
)

TIMING_LABELS = (
    ("data_features", "Data & Features"),
    ("prepared_cache", "Prepared Frame"),
    ("engine_init", "Engine Init"),
    ("simulation", "Simulation Engine"),
    ("strategy_simulation_total", "Simulation Pipeline"),
    ("reporting", "Reporting"),
)

INTERVAL_LABELS = {
    "1m": "1 Minute",
    "5m": "5 Minutes",
    "15m": "15 Minutes",
    "1h": "1 Hour",
    "4h": "4 Hours",
    "1d": "1 Day",
}


def _number(value, default=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _signed(value: float, suffix: str = "", decimals: int = 1) -> str:
    return f"{value:+,.{decimals}f}{suffix}"


def _money(value: float, *, signed: bool = False) -> str:
    if value < 0:
        prefix = "-"
    elif signed and value > 0:
        prefix = "+"
    else:
        prefix = ""
    return f"{prefix}${abs(value):,.2f}"


def _duration(seconds) -> str:
    value = max(0.0, _number(seconds))
    if value < 60:
        return f"{value:.1f}s"
    minutes, secs = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _profit_factor(value) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isinf(result) and result > 0:
        return "∞"
    if not math.isfinite(result):
        return "—"
    return f"{result:.2f}"


def _quality_label(raw: object) -> tuple[str, str]:
    status = str(raw or "NOT_AVAILABLE").upper()
    if status in {"OK", "PASS", "PASSED"}:
        return "PASS", "#1f6f43"
    if status in {"WARN", "WARNING"}:
        return "WARN", "#9a6700"
    if status in {"ERROR", "FAIL", "FAILED", "BLOCKED"}:
        return "FAIL", "#a61b1b"
    return "NOT AVAILABLE", "#52606d"


class ResultsDashboardWorkspace(QWidget):
    """Quick answer to how a run performed, whether it is trustworthy, and where it lives."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.metric_values: dict[str, QLabel] = {}
        self.artifact_buttons: dict[str, QPushButton] = {}
        self.timing_values: dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        header = QGroupBox("Completed Run")
        header_layout = QVBoxLayout(header)
        self.run_title = QLabel("No completed run yet")
        self.run_title.setStyleSheet("font-size:17px; font-weight:700")
        header_layout.addWidget(self.run_title)
        self.run_context = QLabel("Run a backtest to populate this dashboard.")
        self.run_context.setWordWrap(True)
        self.run_context.setStyleSheet("color:#52606d")
        header_layout.addWidget(self.run_context)
        self.quality_status = QLabel("Data Quality  —")
        self.quality_status.setStyleSheet("font-weight:700; color:#52606d")
        header_layout.addWidget(self.quality_status)
        outer.addWidget(header)

        performance = QGroupBox("Performance")
        performance_grid = QGridLayout(performance)
        metrics = (
            ("trades", "Trades"),
            ("win_rate", "Win Rate"),
            ("net_r", "Net R"),
            ("avg_r", "Avg R / Trade"),
            ("net_pnl", "Net P&L"),
            ("return", "Total Return"),
            ("ending_equity", "Ending Equity"),
            ("profit_factor", "Profit Factor"),
        )
        for index, (key, title) in enumerate(metrics):
            card = QGroupBox(title)
            card_layout = QVBoxLayout(card)
            value = QLabel("—")
            value.setStyleSheet("font-size:17px; font-weight:700")
            value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            value.setWordWrap(True)
            card_layout.addWidget(value)
            self.metric_values[key] = value
            performance_grid.addWidget(card, index // 4, index % 4)
        for column in range(4):
            performance_grid.setColumnStretch(column, 1)
        outer.addWidget(performance)

        risk = QGroupBox("Risk & Costs")
        risk_grid = QGridLayout(risk)
        self.drawdown_value = QLabel("—")
        self.fees_value = QLabel("—")
        self.funding_value = QLabel("—")
        self.funding_value.setToolTip(
            "Net perpetual funding cashflow (received minus paid). Already included in Net P&L."
        )
        for column, (title, value) in enumerate(
            (
                ("Maximum Drawdown", self.drawdown_value),
                ("Fees", self.fees_value),
                ("Funding P&L", self.funding_value),
            )
        ):
            title_label = QLabel(title)
            title_label.setStyleSheet("color:#52606d")
            value.setStyleSheet("font-size:16px; font-weight:700")
            risk_grid.addWidget(title_label, 0, column)
            risk_grid.addWidget(value, 1, column)
            risk_grid.setColumnStretch(column, 1)
        outer.addWidget(risk)

        actions = QHBoxLayout()
        for key, label in PRIMARY_ARTIFACTS:
            button = QPushButton(label)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, name=key: self.window.open_artifact(name))
            self.artifact_buttons[key] = button
            actions.addWidget(button)
        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(lambda: self.window.open_folder.click())
        actions.addWidget(self.open_folder_button)
        actions.addStretch()
        outer.addLayout(actions)

        self.research_toggle = QPushButton("Research Artifacts ▸")
        self.research_toggle.setCheckable(True)
        self.research_toggle.setStyleSheet("text-align:left; font-weight:600")
        outer.addWidget(self.research_toggle)
        self.research_content = QWidget()
        research_grid = QGridLayout(self.research_content)
        research_grid.setContentsMargins(8, 0, 8, 0)
        for index, (key, label) in enumerate(RESEARCH_ARTIFACTS):
            button = QPushButton(label)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, name=key: self.window.open_artifact(name))
            self.artifact_buttons[key] = button
            research_grid.addWidget(button, index // 3, index % 3)
        for column in range(3):
            research_grid.setColumnStretch(column, 1)
        self.research_content.hide()
        outer.addWidget(self.research_content)
        self.research_toggle.toggled.connect(self._toggle_research)

        self.performance_toggle = QPushButton("Run Performance ▸")
        self.performance_toggle.setCheckable(True)
        self.performance_toggle.setStyleSheet("text-align:left; font-weight:600")
        outer.addWidget(self.performance_toggle)
        self.performance_content = QWidget()
        timing_grid = QGridLayout(self.performance_content)
        timing_grid.setContentsMargins(8, 0, 8, 0)
        for row, (key, title) in enumerate(TIMING_LABELS):
            name = QLabel(title)
            name.setStyleSheet("color:#52606d")
            value = QLabel("—")
            self.timing_values[key] = value
            timing_grid.addWidget(name, row, 0)
            timing_grid.addWidget(value, row, 1)
        timing_grid.setColumnStretch(1, 1)
        self.performance_content.hide()
        outer.addWidget(self.performance_content)
        self.performance_toggle.toggled.connect(self._toggle_performance)
        outer.addStretch()

    def _toggle_research(self, checked: bool) -> None:
        self.research_content.setVisible(checked)
        self.research_toggle.setText("Research Artifacts ▾" if checked else "Research Artifacts ▸")

    def _toggle_performance(self, checked: bool) -> None:
        self.performance_content.setVisible(checked)
        self.performance_toggle.setText("Run Performance ▾" if checked else "Run Performance ▸")

    def _read_quality(self, manifest: dict, run_dir: Path | None) -> object:
        if run_dir is None:
            return "NOT_AVAILABLE"
        entry = manifest.get("artifacts", {}).get("data_quality", {})
        relative = entry.get("path")
        if not relative:
            return "NOT_AVAILABLE"
        path = run_dir / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return "NOT_AVAILABLE"
        return payload.get("status", "NOT_AVAILABLE") if isinstance(payload, dict) else "NOT_AVAILABLE"

    def _read_funding_net_pnl(
        self,
        manifest: dict,
        run_dir: Path | None,
        summary: dict,
    ) -> float | None:
        """Read funding from summary when available, otherwise from the completed trade CSV.

        The CSV fallback makes the dashboard useful for runs created immediately
        after funding accounting was introduced, before summary.json gained any
        dedicated funding aggregate. Older pre-funding runs remain explicitly
        unavailable rather than being displayed as a misleading $0.00.
        """
        if "total_funding_net_pnl" in summary:
            try:
                value = float(summary["total_funding_net_pnl"])
            except (TypeError, ValueError):
                value = math.nan
            if math.isfinite(value):
                return value

        if run_dir is None:
            return None
        entry = manifest.get("artifacts", {}).get("trade_csv", {})
        relative = entry.get("path")
        if not relative:
            return None
        path = run_dir / relative
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "pair_funding_net_pnl" not in reader.fieldnames:
                    return None
                total = 0.0
                for row in reader:
                    try:
                        value = float(row.get("pair_funding_net_pnl", ""))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        total += value
                return total
        except (OSError, csv.Error):
            return None

    def refresh_completed_run(self) -> None:
        manifest = dict(getattr(self.window, "_manifest", {}) or {})
        run_dir_value = getattr(self.window, "_run_dir", None)
        run_dir = Path(run_dir_value) if run_dir_value else None
        if not manifest or run_dir is None:
            self._clear()
            return

        try:
            _manifest, summary = self.window.service.completed_runs.read(run_dir)
        except Exception:
            summary = {}

        request = manifest.get("request", {})
        symbol = str(request.get("symbol") or run_dir.name.split("_", 1)[0] or "Run")
        interval = str(request.get("requested_strategy_interval") or "").strip()
        interval_text = INTERVAL_LABELS.get(interval, interval or "Strategy timeframe")
        self.run_title.setText(f"{symbol} · {interval_text}")
        start = str(request.get("start") or "").split("T", 1)[0]
        end = str(request.get("end") or "").split("T", 1)[0]
        run_id = str(manifest.get("run_id") or "")
        context_bits = ["Completed"]
        if start and end:
            context_bits.append(f"{start} → {end}")
        if run_id:
            context_bits.append(f"Run {run_id}")
        self.run_context.setText("  ·  ".join(context_bits))

        quality, color = _quality_label(self._read_quality(manifest, run_dir))
        self.quality_status.setText(f"Data Quality  {quality}")
        self.quality_status.setStyleSheet(f"font-weight:700; color:{color}")

        trades = int(_number(summary.get("total_trades"), 0))
        wins = int(_number(summary.get("wins"), 0))
        losses = int(_number(summary.get("losses"), 0))
        win_rate = _number(summary.get("win_rate")) * 100.0
        net_r = _number(summary.get("total_net_r"))
        avg_r = _number(summary.get("average_net_r"))
        net_pnl = _number(summary.get("net_pnl"))
        total_return = _number(summary.get("total_return_percentage"))
        ending_equity = _number(summary.get("ending_equity"))
        drawdown = _number(summary.get("maximum_drawdown_percentage"))
        fees = _number(summary.get("total_fees"))
        funding_net = self._read_funding_net_pnl(manifest, run_dir, summary)

        self.metric_values["trades"].setText(f"{trades:,}\n{wins:,} W · {losses:,} L")
        self.metric_values["win_rate"].setText(f"{win_rate:.1f}%")
        self.metric_values["net_r"].setText(_signed(net_r, "R", 1))
        self.metric_values["avg_r"].setText(_signed(avg_r, "R", 3))
        self.metric_values["net_pnl"].setText(_money(net_pnl, signed=True))
        self.metric_values["return"].setText(_signed(total_return, "%", 1))
        self.metric_values["ending_equity"].setText(_money(ending_equity))
        self.metric_values["profit_factor"].setText(_profit_factor(summary.get("profit_factor")))
        self.drawdown_value.setText(_signed(drawdown, "%", 1))
        self.fees_value.setText(_money(fees))
        self.funding_value.setText(
            _money(funding_net, signed=True) if funding_net is not None else "—"
        )

        artifacts = manifest.get("artifacts", {})
        for key, button in self.artifact_buttons.items():
            button.setEnabled(key in artifacts)
        self.open_folder_button.setEnabled(True)

        timings = manifest.get("execution_result", {}).get("stage_timings", {})
        for key, value in self.timing_values.items():
            value.setText(_duration(timings.get(key)) if key in timings else "—")

    def _clear(self) -> None:
        self.run_title.setText("No completed run yet")
        self.run_context.setText("Run a backtest to populate this dashboard.")
        self.quality_status.setText("Data Quality  —")
        self.quality_status.setStyleSheet("font-weight:700; color:#52606d")
        for value in self.metric_values.values():
            value.setText("—")
        self.drawdown_value.setText("—")
        self.fees_value.setText("—")
        self.funding_value.setText("—")
        for value in self.timing_values.values():
            value.setText("—")
        for button in self.artifact_buttons.values():
            button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
