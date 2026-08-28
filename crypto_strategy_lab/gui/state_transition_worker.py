"""GUI worker wrapper that adds research-only state-transition exports.

The existing BacktestWorker remains unchanged. This subclass listens for the
worker's successful completion signal and writes the Markov/state-transition
reports before the main window processes the same completion signal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Slot

from crypto_strategy_lab.loader import load_backtest_data
from crypto_strategy_lab.state_transition_research import generate_state_transition_reports
from crypto_strategy_lab.gui.worker import BacktestWorker


class StateTransitionBacktestWorker(BacktestWorker):
    """BacktestWorker with automatic research-only state-transition reports."""

    def __init__(self, config, strategy_data: pd.DataFrame | None = None):
        super().__init__(config, strategy_data)
        # Connect before MainWindow attaches its finished handler, so report
        # files are normally present by the time the GUI refreshes run outputs.
        self.finished.connect(self._write_state_transition_reports)

    @Slot(dict, object, object, object)
    def _write_state_transition_reports(self, _summary, trades, _equity, run_dir) -> None:
        try:
            # Reuse already-supplied strategy data when available. Otherwise
            # reload through the normal loader so timestamp/date filtering and
            # timeframe handling remain identical to the backtest input path.
            data, _intrabar = load_backtest_data(self.config, self.strategy_data)
            generate_state_transition_reports(data, trades, Path(run_dir))
            self._log("State-transition research reports saved")
        except Exception as exc:  # Research output must never fail a completed backtest.
            self._log(f"WARNING: state-transition research report failed: {exc}")
