"""Final GUI composition for the compact Review & Run workspace."""
from __future__ import annotations

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QPushButton

from .review_run_workspace import ReviewRunWorkspace


def _retire_sidebar_run_shortcut(window) -> None:
    """Hide the legacy duplicate run action; Review & Run owns execution UX."""
    for button in window.findChildren(QPushButton):
        if button is window.run_button:
            continue
        if button.text().strip().upper() == "RUN BACKTEST":
            button.setEnabled(False)
            button.hide()


def _complete_run_thread_lifecycle(window, workspace, thread, worker=None) -> None:
    """Clear completed run handles before rendering the idle action state."""
    if getattr(window, "_thread", None) is thread:
        window._thread = None
    if worker is not None and getattr(window, "_worker", None) is worker:
        window._worker = None
    workspace.refresh_readiness()


class _ReviewRunCompletionBridge(QObject):
    """Keep run-completion UI work on the GUI thread.

    RunWorker emits from its worker QThread.  The active GUI installs wrappers
    around the native completion handlers, so those wrappers must remain QObject
    slots; replacing them with plain Python callables loses Qt receiver affinity
    and can execute QWidget updates on the worker thread.
    """

    def __init__(self, window, workspace, original_finished, original_failed):
        super().__init__(workspace)
        self.window = window
        self.workspace = workspace
        self.original_finished = original_finished
        self.original_failed = original_failed
        self._finishing_thread = None
        self._finishing_worker = None

    def _watch_thread_finish(self, thread, worker) -> None:
        if thread is None:
            self.workspace.refresh_readiness()
            return

        self._finishing_thread = thread
        self._finishing_worker = worker

        # Queue QObject destruction while the worker event loop is still alive.
        # The native RunWorker has no parent because it lives in another thread.
        if worker is not None:
            try:
                worker.deleteLater()
            except RuntimeError:
                pass

        try:
            if thread.isFinished():
                self.thread_finished()
                return
            thread.finished.connect(self.thread_finished)
        except RuntimeError:
            self.thread_finished()

    @Slot(object)
    def finished(self, result):
        thread = getattr(self.window, "_thread", None)
        worker = getattr(self.window, "_worker", None)
        value = self.original_finished(result)
        self.window.run_button.setText("Run Backtest")
        # The native handler enables the button when simulation/reporting is
        # complete. Keep it disabled until QThread itself has actually stopped.
        self.workspace.refresh_readiness()
        self._watch_thread_finish(thread, worker)
        return value

    @Slot(str)
    def failed(self, message):
        thread = getattr(self.window, "_thread", None)
        worker = getattr(self.window, "_worker", None)
        value = self.original_failed(message)
        self.window.run_button.setText("Run Backtest")
        self.workspace.refresh_readiness()
        self._watch_thread_finish(thread, worker)
        return value

    @Slot()
    def thread_finished(self) -> None:
        thread = self._finishing_thread
        worker = self._finishing_worker
        self._finishing_thread = None
        self._finishing_worker = None

        if thread is not None:
            try:
                thread.deleteLater()
            except RuntimeError:
                pass
        _complete_run_thread_lifecycle(
            self.window, self.workspace, thread, worker
        )


def apply_review_run_workspace(window) -> None:
    """Replace the legacy review/run strip without changing run semantics."""
    if getattr(window, "review_run_workspace", None) is not None:
        return
    required = (
        "_page",
        "_replace_page",
        "save",
        "load",
        "run_button",
        "output_root",
        "review_summary",
        "_render_research_summary",
    )
    if not all(hasattr(window, name) for name in required):
        return

    workspace = ReviewRunWorkspace(window)
    _retire_sidebar_run_shortcut(window)

    # Keep native compatibility targets alive when the legacy page is deleted.
    # They remain authoritative sinks used by config/reporting plumbing but are
    # no longer researcher-facing controls on this page.
    window.review_summary.setParent(workspace)
    window.review_summary.hide()
    window.output_root.setParent(workspace)
    window.output_root.hide()

    page = window._page("Review & Run", workspace)
    window._replace_page(5, page)
    window.review_run_workspace = workspace

    original_render_summary = window._render_research_summary

    def render_summary_and_review(config):
        result = original_render_summary(config)
        workspace.refresh(config)
        return result

    window._render_research_summary = render_summary_and_review

    if hasattr(window, "_set_readiness"):
        original_set_readiness = window._set_readiness

        def set_readiness_and_review(title, detail, *, state="pending"):
            result = original_set_readiness(title, detail, state=state)
            workspace.refresh_readiness()
            return result

        window._set_readiness = set_readiness_and_review

    # Preserve QObject receiver affinity for the worker completion signals.
    # start_run connects to whatever window._finished/_failed reference exists
    # at run time; these must therefore be QObject-bound slots, not plain lambdas
    # or nested functions that can run in the worker thread.
    completion_bridge = _ReviewRunCompletionBridge(
        window, workspace, window._finished, window._failed
    )
    window._review_run_completion_bridge = completion_bridge
    window._finished = completion_bridge.finished
    window._failed = completion_bridge.failed

    original_apply_config = window.apply_config

    def apply_config_and_review(config):
        result = original_apply_config(config)
        workspace.refresh(config)
        return result

    window.apply_config = apply_config_and_review
    workspace.refresh()
