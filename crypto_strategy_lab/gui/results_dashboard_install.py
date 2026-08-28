"""Final GUI composition for the human-focused Results Dashboard."""
from __future__ import annotations

from PySide6.QtCore import QObject, Slot

from .completed_run_research import install_completed_run_research_actions
from .results_dashboard_workspace import ResultsDashboardWorkspace


def _preserve_legacy_result_widgets(window, workspace) -> None:
    """Keep native completion sinks alive while removing their old presentation."""
    widgets = []
    widgets.extend(getattr(window, "kpi_cards", {}).values())
    widgets.append(getattr(window, "summary", None))
    widgets.append(getattr(window, "timings", None))
    widgets.extend(getattr(window, "artifact_buttons", {}).values())
    widgets.append(getattr(window, "open_folder", None))
    for widget in widgets:
        if widget is None:
            continue
        widget.setParent(workspace)
        widget.hide()


class _ResultsCompletionBridge(QObject):
    """Refresh the dashboard from the GUI thread after native completion."""

    def __init__(self, workspace, original_finished):
        super().__init__(workspace)
        self.workspace = workspace
        self.original_finished = original_finished

    @Slot(object)
    def finished(self, result):
        value = self.original_finished(result)
        self.workspace.refresh_completed_run()
        return value


def apply_results_dashboard_workspace(window) -> None:
    """Replace the legacy debug-like result dump without changing completion semantics."""
    if getattr(window, "results_dashboard_workspace", None) is not None:
        return
    required = (
        "_page",
        "_replace_page",
        "_finished",
        "open_artifact",
        "open_folder",
        "artifact_buttons",
        "kpi_cards",
        "summary",
        "timings",
    )
    if not all(hasattr(window, name) for name in required):
        return

    workspace = ResultsDashboardWorkspace(window)
    _preserve_legacy_result_widgets(window, workspace)
    page = window._page("Results Dashboard", workspace)
    window._replace_page(6, page)
    window.results_dashboard_workspace = workspace

    # Keep the final completion callback QObject-bound. RunWorker emits from its
    # worker thread, so replacing _finished with a plain nested function would
    # allow dashboard/QWidget updates to execute off the GUI thread.
    completion_bridge = _ResultsCompletionBridge(workspace, window._finished)
    window._results_completion_bridge = completion_bridge
    window._finished = completion_bridge.finished

    install_completed_run_research_actions(window, workspace)
    workspace.refresh_completed_run()
