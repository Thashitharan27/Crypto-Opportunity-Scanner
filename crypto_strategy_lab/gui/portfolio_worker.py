"""Background worker for shared-equity portfolio runs."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from crypto_strategy_lab.portfolio import run_portfolio


class PortfolioWorker(QObject):
    status = Signal(str, int)
    log = Signal(str)
    finished = Signal(dict, object, object, object)
    failed = Signal(str, str)

    def __init__(self, components, output_root, initial_equity, risk_per_asset, maximum_total_risk=0.05):
        super().__init__()
        self.components = components
        self.output_root = output_root
        self.initial_equity = initial_equity
        self.risk_per_asset = risk_per_asset
        self.maximum_total_risk = maximum_total_risk

    @Slot()
    def run(self):
        try:
            component_count = len(self.components)
            active_index = {"value": 0}

            def progress(label, done, total, completed, opened):
                labels = [str(item[0]).upper() for item in self.components]
                try:
                    index = labels.index(str(label).upper())
                except ValueError:
                    index = active_index["value"]
                active_index["value"] = index
                ratio = done / total if total else 0.0
                percent = int(((index + ratio) / component_count) * 90)
                self.status.emit(
                    f"Running {label}: {done:,}/{total:,} candles; "
                    f"{completed:,} completed trades",
                    percent,
                )

            self.log.emit("Starting shared-equity portfolio run.")
            summary, trades, equity, run_dir = run_portfolio(
                self.components,
                self.output_root,
                self.initial_equity,
                self.risk_per_asset,
                self.maximum_total_risk,
                progress,
            )
            self.status.emit("Portfolio reports saved", 100)
            self.log.emit(f"Portfolio results saved to {run_dir}")
            self.finished.emit(summary, trades, equity, run_dir)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
