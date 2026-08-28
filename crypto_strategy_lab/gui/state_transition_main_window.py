"""Top-level GUI composition for automatic state-transition research reports.

This layer preserves the existing enhanced/SR-aware main window and only swaps
its single-backtest worker for the research-reporting wrapper. It also exposes a
visible Summary report button for the generated state-transition folder.
"""
from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QPushButton

from crypto_strategy_lab.gui import main_window as base_main_window_module
from crypto_strategy_lab.gui.state_transition_worker import StateTransitionBacktestWorker

# BaseMainWindow.run_backtest resolves BacktestWorker from its defining module at
# runtime. Replacing that module global preserves every enhanced/SR GUI layer
# while adding the post-run research export. Report-target state is intentionally
# kept local to this subclass so importing it does not mutate base-GUI behavior.
base_main_window_module.BacktestWorker = StateTransitionBacktestWorker

from crypto_strategy_lab.gui.sr_dynamic_tp_main_window import MainWindow as SRDynamicTPMainWindow


_STATE_REPORT_TARGET = "state_transition_research"


class MainWindow(SRDynamicTPMainWindow):
    """Existing full GUI plus a visible State Research report shortcut."""

    def _build_summary(self):
        super()._build_summary()
        if "state" in self.report_buttons:
            return
        button = QPushButton("Open State Research")
        button.setEnabled(False)
        button.clicked.connect(lambda _checked=False: self._open_report("state"))
        self.report_buttons["state"] = button

        # The existing six report buttons occupy two rows of three. Put the
        # research shortcut on the next row without restructuring the base UI.
        reports_layout = self.report_buttons["output"].parentWidget().layout()
        reports_layout.addWidget(button, 2, 0)

    def _refresh_report_buttons(self):
        super()._refresh_report_buttons()
        button = self.report_buttons.get("state")
        if button is None:
            return
        available = bool(
            self.completed_run_dir is not None
            and (self.completed_run_dir / _STATE_REPORT_TARGET).exists()
        )
        button.setEnabled(available)

    def _open_report(self, name):
        if name != "state":
            return super()._open_report(name)
        if self.completed_run_dir is None:
            return
        target = (self.completed_run_dir / _STATE_REPORT_TARGET).resolve()
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    @staticmethod
    def _percentage_points(value, decimals: int = 2) -> str:
        """Format statistics that are already stored as percentage points."""
        if value is None:
            return "—"
        return f"{float(value):.{decimals}f}%"

    def populate_summary(self, summary, trades=None):
        """Render ratio metrics and percentage-point metrics with correct units."""
        super().populate_summary(summary, trades)
        self.kpi_labels["Total Return"].setText(
            self._percentage_points(summary.get("total_return_percentage"))
        )
        self.kpi_labels["Maximum Drawdown"].setText(
            self._percentage_points(summary.get("maximum_drawdown_percentage"))
        )
