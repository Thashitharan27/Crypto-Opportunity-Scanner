from unittest.mock import Mock

from PySide6.QtWidgets import QMainWindow

from crypto_strategy_lab.gui.github_manager import GitHubIntegrationWidget, GitManager, GitStatus
from crypto_strategy_lab.gui.github_sync_install import apply_github_sync_safety


def test_active_v3_window_binds_github_source_changes_to_run_thread(qapp, monkeypatch):
    monkeypatch.setattr(GitManager, "status", lambda self, fetch=False: GitStatus("main", "origin", ()))
    monkeypatch.setattr(
        GitHubIntegrationWidget,
        "_async",
        lambda self, fn, done, description: done(fn()),
    )

    window = QMainWindow()
    widget = GitHubIntegrationWidget(window)
    thread = Mock()
    thread.isRunning.return_value = True
    window._thread = thread

    apply_github_sync_safety(window)

    assert window.github_sync is widget
    assert widget.active_work() is True
    thread.isRunning.return_value = False
    assert widget.active_work() is False
    window.close()
