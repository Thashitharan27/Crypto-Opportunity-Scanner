"""Minimal researcher-facing output workspace for native v3 runs.

The active GUI intentionally exposes one stable completed-run output contract.
Question-specific research is performed on demand from canonical artifacts
instead of accumulating permanent report toggles and telemetry modes.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ReportsDiagnosticsWorkspace(QWidget):
    """Run details plus research-only sampling and canonical output guidance."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.form = window.reporting_form
        self.widgets = self.form.widgets
        self._syncing_output = False
        self._syncing_bayes = False
        self._syncing_research_sampling = False

        # Keep the stable shell's ReportingConfig form alive as the one native
        # config boundary, but expose only the researcher-facing controls below.
        self.form.setParent(window)
        self.form.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self._run_details_box())
        layout.addWidget(self._research_sampling_box())
        layout.addWidget(self._bayes_sampling_box())
        layout.addWidget(self._saved_outputs_box())
        layout.addWidget(self._research_model_box())
        layout.addStretch()

        self.window.output_root.textChanged.connect(self._source_output_changed)
        self.refresh_from_form()

    def _run_details_box(self) -> QGroupBox:
        box = QGroupBox("Run Details")
        form = QFormLayout(box)

        self.run_name = self.widgets["run_name"]
        self.output_dir = QLineEdit(self.window.output_root.text())
        self.output_dir.setPlaceholderText(
            "Folder where completed run directories are saved"
        )
        self.output_dir.textChanged.connect(self._workspace_output_changed)

        output_row = QWidget()
        row = QHBoxLayout(output_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.output_dir, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.window._browse_output)
        row.addWidget(browse)

        form.addRow("Run Name", self.run_name)
        form.addRow("Output Folder", output_row)
        return box

    def _research_sampling_box(self) -> QGroupBox:
        box = QGroupBox("Research Sampling Mode")
        form = QFormLayout(box)

        self.research_sampling_mode = QComboBox()
        self.research_sampling_mode.addItem(
            "Portfolio — realistic backtest", "PORTFOLIO"
        )
        self.research_sampling_mode.addItem(
            "Every Viable Entry — resilience", "EVERY_VIABLE_ENTRY"
        )
        self.research_sampling_mode.addItem(
            "Fixed Interval — every N viable candles", "FIXED_INTERVAL"
        )
        self.research_sampling_mode.addItem(
            "Episode First — first viable entry only", "EPISODE_FIRST"
        )
        self.research_sampling_mode.setToolTip(
            "Portfolio keeps normal capital/open-trade constraints. Research modes run a "
            "separate strategy-valid sample population with overlapping independent trades."
        )
        self.research_sampling_mode.currentIndexChanged.connect(
            self._research_sampling_mode_changed
        )

        self.research_sampling_interval = self.widgets[
            "research_sampling_interval_candles"
        ]
        if isinstance(self.research_sampling_interval, QSpinBox):
            self.research_sampling_interval.setRange(1, 10000)
            self.research_sampling_interval.setSuffix(" candles")
        self.research_sampling_interval.setToolTip(
            "Used only by Fixed Interval. Sampling restarts at the first viable entry "
            "of each uninterrupted strategy episode."
        )
        value_changed = getattr(self.research_sampling_interval, "valueChanged", None)
        if value_changed is not None:
            value_changed.connect(lambda _value: self.refresh_review_summary())

        form.addRow("Mode", self.research_sampling_mode)
        form.addRow("Fixed Interval", self.research_sampling_interval)

        note = QLabel(
            "Research modes do not replace the normal portfolio result. They open a separate "
            "synthetic trade whenever the configured strategy is viable, preserve the same "
            "Entry/Veto rules and SL/TP/timeout/intrabar execution, and ignore only portfolio "
            "overlap, combined exposure and compounding. Samples are grouped into uninterrupted "
            "episodes so repeated entries from one market move are not presented as independent "
            "confirmations. Equity curve, drawdown, exposure and compounded return are explicitly "
            "invalid for this research population."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#fff8e6; padding:10px; border:1px solid #ead9a3"
        )
        form.addRow(note)
        return box

    def _reporting_text_value(self, name: str, default: str) -> str:
        widget = self.widgets.get(name)
        if widget is None:
            return default
        current_data = getattr(widget, "currentData", None)
        if callable(current_data):
            value = current_data()
            if value not in (None, ""):
                return str(value)
        text = getattr(widget, "text", None)
        if callable(text):
            return str(text())
        current_text = getattr(widget, "currentText", None)
        if callable(current_text):
            return str(current_text())
        return default

    def _set_reporting_text(self, name: str, value: str) -> None:
        widget = self.widgets.get(name)
        if widget is None:
            return
        set_text = getattr(widget, "setText", None)
        if callable(set_text):
            set_text(value)
            return
        find_data = getattr(widget, "findData", None)
        set_index = getattr(widget, "setCurrentIndex", None)
        if callable(find_data) and callable(set_index):
            index = find_data(value)
            if index >= 0:
                set_index(index)
                return
        find_text = getattr(widget, "findText", None)
        if callable(find_text) and callable(set_index):
            index = find_text(value)
            if index >= 0:
                set_index(index)

    def _research_sampling_mode_changed(self, _index: int) -> None:
        if self._syncing_research_sampling:
            return
        self._syncing_research_sampling = True
        try:
            mode = str(self.research_sampling_mode.currentData())
            self._set_reporting_text("research_sampling_mode", mode)
            self.research_sampling_interval.setEnabled(mode == "FIXED_INTERVAL")
        finally:
            self._syncing_research_sampling = False
        self.refresh_review_summary()

    def _bayes_sampling_box(self) -> QGroupBox:
        box = QGroupBox("Bayesian Market-Grid Sampling")
        layout = QVBoxLayout(box)
        self.bayes_sampling = QCheckBox(
            "Generate direction-neutral market-grid LONG + SHORT observations"
        )
        self.bayes_sampling.setToolTip(
            "At every completed candle of the selected strategy timeframe, evaluate "
            "independent hypothetical LONG and SHORT trades with the native execution engine."
        )
        self.bayes_sampling.toggled.connect(self._bayes_sampling_changed)
        layout.addWidget(self.bayes_sampling)
        note = QLabel(
            "Separate from Research Sampling Mode. This existing Bayesian dataset deliberately "
            "ignores strategy Entry/Veto selection and labels both LONG and SHORT market-grid "
            "outcomes. Strategy-valid research samples instead carry research_episode_id as the "
            "cluster key for correlation-aware downstream analysis."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#eef5fb; padding:10px; border:1px solid #c8d9e8"
        )
        layout.addWidget(note)
        return box

    def _analysis_level_value(self) -> str:
        return self._reporting_text_value("analysis_level", "STANDARD")

    def _set_analysis_level(self, value: str) -> None:
        self._set_reporting_text("analysis_level", value)

    def _bayes_sampling_changed(self, checked: bool) -> None:
        if self._syncing_bayes:
            return
        self._syncing_bayes = True
        try:
            self._set_analysis_level("DEEP" if checked else "STANDARD")
        finally:
            self._syncing_bayes = False
        self.refresh_review_summary()

    @staticmethod
    def _saved_outputs_box() -> QGroupBox:
        box = QGroupBox("Completed Run Output")
        layout = QVBoxLayout(box)
        note = QLabel(
            "Every completed run saves one clean output set; optional research populations are "
            "published separately and never replace the authoritative strategy trades.\n\n"
            "Human review\n"
            "• backtest_report.xlsx — dashboard, monthly/yearly performance, market regime and direction × regime\n"
            "• trade_list.csv — easy-to-open completed portfolio trade list\n"
            "• summary.json — compact portfolio run statistics\n"
            "• data_quality.json — required input-data quality\n\n"
            "Research & reproducibility\n"
            "• artifacts/trades.parquet — authoritative completed portfolio strategy trades\n"
            "• artifacts/feature_context.parquet — causal feature/research state for every strategy row\n"
            "• artifacts/signals.parquet — entered and rejected portfolio decisions\n"
            "• artifacts/research_sampling_trades.parquet — optional strategy-valid overlapping samples\n"
            "• artifacts/research_sampling_episodes.parquet — correlation-cluster/episode outcomes\n"
            "• research_sampling_context.csv — DI/ADX/regime/MR/funding/OI sample breakdowns\n"
            "• research_sampling_summary.json — entry-level + episode-level resilience metrics\n"
            "• artifacts/bayes_research_samples.parquet — optional direction-neutral market grid\n"
            "• provenance/source_archives.parquet — exact selected Binance archive provenance\n"
            "• run_manifest.json — hashes, config, provenance, artifact catalog and completion marker"
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#f7f9fb; padding:10px; border:1px solid #d9e2ec"
        )
        layout.addWidget(note)
        return box

    @staticmethod
    def _research_model_box() -> QGroupBox:
        box = QGroupBox("On-Demand Research")
        layout = QVBoxLayout(box)
        note = QLabel(
            "DI, ADX, volatility, funding, open interest, positioning, taker flow, "
            "support/resistance and state-transition questions should be analysed "
            "from the canonical completed-run artifacts when needed. Research sampling "
            "adds a denser opportunity population while episode IDs retain the correlation "
            "structure needed to avoid treating repeated entries as independent experiments."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#52606d")
        layout.addWidget(note)
        return box

    def _workspace_output_changed(self, text: str) -> None:
        if self._syncing_output:
            return
        self._syncing_output = True
        try:
            self.window.output_root.setText(text)
        finally:
            self._syncing_output = False

    def _source_output_changed(self, text: str) -> None:
        if self._syncing_output:
            return
        self._syncing_output = True
        try:
            self.output_dir.setText(text)
        finally:
            self._syncing_output = False
        self.refresh_review_summary()

    def refresh_from_form(self) -> None:
        self._source_output_changed(self.window.output_root.text())

        self._syncing_research_sampling = True
        try:
            mode = self._reporting_text_value("research_sampling_mode", "PORTFOLIO").upper()
            index = self.research_sampling_mode.findData(mode)
            self.research_sampling_mode.blockSignals(True)
            self.research_sampling_mode.setCurrentIndex(max(index, 0))
            self.research_sampling_mode.blockSignals(False)
            self.research_sampling_interval.setEnabled(mode == "FIXED_INTERVAL")
        finally:
            self._syncing_research_sampling = False

        self._syncing_bayes = True
        try:
            self.bayes_sampling.blockSignals(True)
            self.bayes_sampling.setChecked(
                self._analysis_level_value().upper() in {"DEEP", "DEEP_RESEARCH"}
            )
            self.bayes_sampling.blockSignals(False)
        finally:
            self._syncing_bayes = False
        self.refresh_review_summary()

    def refresh_review_summary(self) -> None:
        summary = getattr(self.window, "review_summary", None)
        if summary is None:
            return
        text = summary.text()
        if not text:
            return
        lines = text.splitlines()
        mode = (
            str(self.research_sampling_mode.currentData())
            if getattr(self, "research_sampling_mode", None) is not None
            else "PORTFOLIO"
        )
        if mode == "FIXED_INTERVAL":
            value = getattr(self.research_sampling_interval, "value", lambda: 1)()
            resilience = f"Research sampling: FIXED INTERVAL · every {value} viable candles"
        else:
            resilience = f"Research sampling: {mode.replace('_', ' ')}"
        bayes = (
            "Bayes grid: ON"
            if getattr(self, "bayes_sampling", None) is not None and self.bayes_sampling.isChecked()
            else "Bayes grid: OFF"
        )
        for index, line in enumerate(lines):
            if line.startswith("Reports:") or line.startswith("Output:"):
                lines[index] = f"Output: Canonical completed-run set · {resilience} · {bayes}"
                summary.setText("\n".join(lines))
                return