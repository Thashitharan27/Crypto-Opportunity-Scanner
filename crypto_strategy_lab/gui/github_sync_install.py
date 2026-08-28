"""Wire the reusable GitHub Sync widget to the active v3 window lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from PySide6.QtWidgets import QMessageBox

from .github_manager import GitHubIntegrationWidget


_DEPENDENCY_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "poetry.lock",
    "uv.lock",
    "pipfile",
    "pipfile.lock",
    "environment.yml",
    "environment.yaml",
}


@dataclass(frozen=True)
class GitUpdateResult:
    output: str
    before_head: str
    after_head: str
    changed_files: tuple[str, ...]

    @property
    def updated(self) -> bool:
        return bool(self.before_head and self.after_head and self.before_head != self.after_head)

    @property
    def dependency_files(self) -> tuple[str, ...]:
        return tuple(path for path in self.changed_files if is_dependency_file(path))


def is_dependency_file(path: str) -> bool:
    """Return whether a changed repository path can require environment refresh."""
    normalized = path.replace("\\", "/").strip("/")
    lower = normalized.lower()
    name = PurePosixPath(lower).name
    return (
        (name.startswith("requirements") and name.endswith(".txt"))
        or lower.startswith("requirements/")
        or name in _DEPENDENCY_NAMES
    )


def _pull_with_metadata(manager, pull_fn) -> GitUpdateResult:
    """Run the existing safe pull and capture exactly what changed."""
    before = manager._run("rev-parse", "HEAD").stdout.strip()
    output = pull_fn()
    after = manager._run("rev-parse", "HEAD").stdout.strip()
    changed_files: tuple[str, ...] = ()
    if before and after and before != after:
        diff = manager._run("diff", "--name-only", f"{before}..{after}").stdout
        changed_files = tuple(line.strip() for line in diff.splitlines() if line.strip())
    return GitUpdateResult(output, before, after, changed_files)


def _mark_restart_required(widget: GitHubIntegrationWidget, detail: str) -> None:
    """Leave the current process running but make its stale-code state explicit."""
    widget.state.setText("● Update installed — restart required")
    widget.status_detail.setText(detail)
    # The repository changed underneath this running Python process. Avoid more
    # Git mutations from the stale process; the user can close and reopen the app.
    widget.pull_btn.setEnabled(False)
    widget.refresh.setEnabled(False)
    widget.review.setEnabled(False)
    widget.commit_btn.setEnabled(False)


def apply_github_sync_safety(window) -> None:
    """Bind Git actions to run safety and a deliberate manual restart workflow."""
    if getattr(window, "_github_sync_safety_installed", False):
        return

    widget = window.findChild(GitHubIntegrationWidget)
    if widget is None:
        return

    def active_work() -> bool:
        thread = getattr(window, "_thread", None)
        return bool(thread is not None and thread.isRunning())

    widget.active_work = active_work
    window.github_sync = widget

    original_pull = widget.manager.pull

    def pull_with_metadata():
        return _pull_with_metadata(widget.manager, original_pull)

    widget.manager.pull = pull_with_metadata

    def update_completed(result: GitUpdateResult) -> None:
        widget._log("Update completed")
        if not result.updated:
            widget.refresh_status(False)
            return

        dependency_files = result.dependency_files
        if dependency_files:
            changed = "\n".join(f"• {path}" for path in dependency_files)
            _mark_restart_required(
                widget,
                "Dependency files changed. Close Crypto Strategy Lab, refresh the Python environment, then reopen it.",
            )
            QMessageBox.information(
                widget,
                "Update installed — dependency refresh required",
                "The GitHub update was installed, but Python dependency files changed:\n\n"
                f"{changed}\n\n"
                "Close Crypto Strategy Lab. For the current Windows setup, run:\n"
                ".venv\\Scripts\\python.exe -m pip install -r requirements.txt\n\n"
                "Then reopen Crypto Strategy Lab.",
            )
            return

        _mark_restart_required(
            widget,
            "Close and reopen Crypto Strategy Lab to use the updated version.",
        )
        QMessageBox.information(
            widget,
            "Update installed",
            "The GitHub update was installed successfully.\n\n"
            "Close and reopen Crypto Strategy Lab to use the new version.",
        )

    widget._pulled = update_completed
    window._github_sync_safety_installed = True
