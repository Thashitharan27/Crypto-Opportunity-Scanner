"""Lightweight presentation for native research-run progress.

The data/simulation layers emit only coarse stages and cache-partition events.
This module owns all ETA formatting and Qt presentation so the research hot path
never performs UI work or per-row/per-trade logging.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def format_duration(seconds: float | int | None) -> str:
    """Human-friendly approximate duration used only for cache-build ETA."""
    if seconds is None:
        return ""
    value = max(0, int(round(float(seconds))))
    if value < 60:
        return f"{value}s"
    minutes, secs = divmod(value, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class RunProgressRelay(QObject):
    """Thread-safe bridge from worker callbacks to widgets on the GUI thread."""

    event = Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.event.connect(self.render)

    @Slot(object)
    def render(self, event):
        payload = dict(event or {})
        kind = payload.get("kind", "stage")
        progress = self.window.progress
        stage = self.window.stage
        detail = self.window.run_progress_detail
        status = getattr(self.window, "run_progress_status", None)
        if status is not None:
            status.show()

        if kind == "cache":
            completed = max(0, int(payload.get("completed", 0) or 0))
            total = max(0, int(payload.get("total", 0) or 0))
            built = max(0, int(payload.get("built", 0) or 0))
            reused = max(0, int(payload.get("reused", 0) or 0))
            label = str(payload.get("label") or "Research cache")
            stage.setText(f"CACHE PREPARATION — {label}")
            if total:
                progress.setRange(0, total)
                progress.setValue(min(completed, total))
                progress.setFormat("%v / %m partitions")
            else:
                progress.setRange(0, 0)
                progress.setFormat("")

            if payload.get("mode") == "validation":
                parts = [f"Validated {completed} of {total} source partitions"]
            else:
                parts = [f"Built {built}", f"Reused {reused}"]
            elapsed = float(payload.get("elapsed_seconds", 0.0) or 0.0)
            remaining = max(0, total - completed)
            # Wait for more than one partition before estimating. This avoids
            # displaying a wildly unstable ETA from the first archive alone.
            if completed >= 2 and remaining and elapsed >= 0.25:
                eta = (elapsed / completed) * remaining
                parts.append(f"Approx. remaining {format_duration(eta)}")
            current = payload.get("current")
            if current:
                parts.append(str(current))
            detail.setText("  •  ".join(parts))
            return

        if kind == "complete":
            progress.setRange(0, 100)
            progress.setValue(100)
            progress.setFormat("Complete")
            stage.setText(str(payload.get("label") or "COMPLETED"))
            detail.setText(str(payload.get("detail") or "All run stages finished."))
            return

        if kind == "failed":
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setFormat("Failed")
            stage.setText(str(payload.get("label") or "FAILED"))
            detail.setText(str(payload.get("detail") or "The run did not complete."))
            return

        # Normal preparation/simulation/reporting stages intentionally remain
        # indeterminate: inventing a percentage would be misleading and would
        # require extra hot-path instrumentation.
        progress.setRange(0, 0)
        progress.setFormat("")
        stage.setText(str(payload.get("label") or "Native research executing"))
        detail.setText(str(payload.get("detail") or ""))


def install_run_progress(window) -> None:
    """Upgrade the existing Run strip without changing MainWindow architecture."""
    if hasattr(window, "_run_progress_relay"):
        return

    run_box = window.progress.parentWidget()
    layout = run_box.layout()
    layout.removeWidget(window.progress)
    layout.removeWidget(window.stage)

    status = QWidget(run_box)
    status_layout = QVBoxLayout(status)
    status_layout.setContentsMargins(8, 0, 0, 0)
    status_layout.setSpacing(3)

    window.stage.setStyleSheet("font-weight:600")
    window.progress.setMinimumWidth(380)
    window.progress.setRange(0, 100)
    window.progress.setValue(0)
    window.progress.setFormat("Ready")
    window.run_progress_detail = QLabel(
        "Cache building will show partition progress and an approximate ETA. "
        "Normal simulation remains lightweight and shows only its current stage."
    )
    window.run_progress_detail.setWordWrap(True)
    window.run_progress_detail.setStyleSheet("color:#52606d")

    status_layout.addWidget(window.stage)
    status_layout.addWidget(window.progress)
    status_layout.addWidget(window.run_progress_detail)
    layout.addWidget(status, 1)
    window.run_progress_status = status
    # Idle Review & Run should emphasize readiness, not an inactive progress bar.
    # Validation or runner events reveal this same status widget when activity starts.
    status.hide()

    relay = RunProgressRelay(window)
    window._run_progress_relay = relay
    # GuiApplicationService copies this observer onto its MarketDataStore only
    # for the duration of a run. Signal emission is one event per coarse stage
    # or cache partition; there is no timer and no per-row/per-trade callback.
    window.service.progress_callback = relay.event.emit
