"""Safe Git operations and the GitHub sync widget.

Git commands are passed as argument lists (never through a shell). The manager is
usable without Qt so its safety rules can be tested in temporary repositories.
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


CONFLICT_CODES = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
SENSITIVE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "credentials.json",
    "secrets.json",
    "*cloudflared*credential*",
    "*tunnel*credential*",
)
CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True)
class GitChange:
    code: str
    path: str

    @property
    def label(self) -> str:
        if self.code in CONFLICT_CODES or "U" in self.code:
            return "Conflict"
        if self.code == "??":
            return "Untracked"
        key = next((c for c in self.code if c not in " ?"), "?")
        return {
            "M": "Modified",
            "A": "Added",
            "D": "Deleted",
            "R": "Renamed",
        }.get(key, self.code)

    @property
    def can_restore(self) -> bool:
        return self.label in {"Modified", "Deleted"}

    @property
    def can_remove_untracked(self) -> bool:
        return self.code == "??"


@dataclass(frozen=True)
class GitStatus:
    branch: str
    remote_url: str
    changes: tuple[GitChange, ...]
    ahead: int = 0
    behind: int = 0

    @property
    def conflicts(self):
        return any(c.label == "Conflict" for c in self.changes)

    @property
    def state(self):
        if self.conflicts:
            return "Merge conflict"
        if self.ahead and self.behind:
            return "Local and remote have diverged"
        if self.behind:
            return "Remote updates available"
        if self.ahead:
            return "Local commits not pushed"
        if self.changes:
            return "Local changes"
        return "Up to date"

    @property
    def can_pull(self):
        return not self.changes and not (self.ahead and self.behind) and not self.conflicts

    @property
    def can_push(self):
        return not self.conflicts and not (self.ahead and self.behind)

    @property
    def sync_summary(self) -> str:
        branch = self.branch or "—"
        changes = len(self.changes)
        change_word = "change" if changes == 1 else "changes"
        update_word = "update" if self.behind == 1 else "updates"
        return f"{branch} • {changes} local {change_word} • {self.behind} GitHub {update_word} available"

    @property
    def guidance(self) -> str:
        if self.conflicts:
            return "Merge conflicts require manual Git resolution before syncing."
        if self.changes and self.behind:
            return (
                f"Update blocked — review {len(self.changes)} local change(s) first. "
                f"GitHub also has {self.behind} update(s) waiting."
            )
        if self.changes:
            return f"Update blocked — review {len(self.changes)} local change(s) first."
        if self.ahead and self.behind:
            return "Local and GitHub history have diverged; resolve this manually before syncing."
        if self.behind:
            return f"{self.behind} update(s) are available from GitHub."
        if self.ahead:
            return f"{self.ahead} local commit(s) have not been pushed yet."
        return "Local checkout matches GitHub."


class GitError(RuntimeError):
    pass


def project_root() -> Path:
    """Determine the checkout from this module's location, not a machine path."""
    return Path(__file__).resolve().parents[2]


def parse_porcelain(output: str) -> tuple[GitChange, ...]:
    changes = []
    for line in output.splitlines():
        if len(line) >= 3:
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            changes.append(GitChange(line[:2], path.strip('"')))
    return tuple(changes)


def parse_ahead_behind(output: str) -> tuple[int, int]:
    left, right = output.strip().split()
    return int(left), int(right)


def is_sensitive(path: str) -> bool:
    name = Path(path).name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_PATTERNS)


class GitManager:
    def __init__(self, root: Path | None = None, git: str = "git"):
        self.root = Path(root or project_root()).resolve()
        self.git = git

    def _run(self, *args: str, check=True) -> subprocess.CompletedProcess[str]:
        try:
            creationflags = CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                [self.git, *args],
                cwd=self.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise GitError("Git is not installed or not available in PATH.") from exc
        if check and result.returncode:
            raise GitError((result.stderr or result.stdout or "Git command failed.").strip())
        return result

    def validate(self):
        if not (self.root / ".git").exists():
            raise GitError("The application folder is not a Git repository.")
        try:
            self._run("--version")
        except GitError as exc:
            if "not installed" in str(exc):
                raise
            raise GitError("Git is not installed or not available in PATH.") from exc
        return True

    def branch(self):
        return self._run("branch", "--show-current").stdout.strip()

    def remote_url(self):
        return self._run("remote", "get-url", "origin").stdout.strip()

    def changes(self):
        return parse_porcelain(self._run("status", "--porcelain").stdout)

    def ignored(self, path: str):
        return self._run("check-ignore", "-q", "--", path, check=False).returncode == 0

    def upstream(self):
        result = self._run(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else f"origin/{self.branch()}"

    def status(self, fetch=False):
        self.validate()
        if fetch:
            self._run("fetch", "origin")
        branch, remote, changes = self.branch(), self.remote_url(), self.changes()
        counts = self._run(
            "rev-list",
            "--left-right",
            "--count",
            f"HEAD...{self.upstream()}",
            check=False,
        )
        ahead, behind = parse_ahead_behind(counts.stdout) if counts.returncode == 0 else (0, 0)
        return GitStatus(branch, remote, changes, ahead, behind)

    def pull(self):
        status = self.status()
        if status.conflicts:
            raise GitError("Merge conflicts detected.\nResolve them before continuing.")
        if status.changes:
            raise GitError(
                "Cannot update while local changes are present.\n\n"
                "Review, commit, or discard the local changes first."
            )
        if status.ahead and status.behind:
            raise GitError(
                "Local and remote branches have diverged. Resolve this manually before pulling."
            )
        return self._run("pull", "--ff-only").stdout.strip()

    def validate_commit(self, paths, message):
        if not message.strip():
            raise GitError("A non-empty commit message is required.")
        if not paths:
            raise GitError("Select at least one file to commit.")
        for path in paths:
            if is_sensitive(path):
                raise GitError(
                    "This file appears to contain local credentials or secrets and cannot be committed from the GUI."
                )
            if self.ignored(path):
                raise GitError(f"Ignored files cannot be committed from the GUI: {path}")

    def commit(self, paths, message):
        self.validate_commit(paths, message)
        self._run("add", "--", *paths)
        result = self._run("commit", "-m", message.strip(), "--", *paths)
        return self._run("rev-parse", "--short", "HEAD").stdout.strip(), result.stdout.strip()

    def push(self, branch):
        return self._run("push", "origin", branch).stdout.strip()

    def diff(self, change: GitChange):
        path = self.root / change.path
        if change.code == "??":
            if (
                not path.is_file()
                or path.stat().st_size > 256_000
                or b"\0" in path.read_bytes()[:8192]
            ):
                return "Binary or large untracked file; preview unavailable."
            return path.read_text(errors="replace")
        work = self._run("diff", "--", change.path, check=False).stdout
        staged = self._run("diff", "--cached", "--", change.path, check=False).stdout
        return "WORKING TREE\n" + (work or "(none)") + "\n\nSTAGED\n" + (staged or "(none)")

    def _current_change(self, change: GitChange) -> GitChange:
        current = next((item for item in self.changes() if item.path == change.path), None)
        if current is None:
            raise GitError("The selected local change no longer exists. Refresh and try again.")
        return current

    def restore_tracked(self, change: GitChange):
        current = self._current_change(change)
        if not current.can_restore:
            raise GitError("Only modified or deleted tracked files can be restored from the GUI.")
        self._run(
            "restore",
            "--source=HEAD",
            "--staged",
            "--worktree",
            "--",
            current.path,
        )
        return current.path

    def remove_untracked(self, change: GitChange):
        current = self._current_change(change)
        if not current.can_remove_untracked:
            raise GitError("Only untracked files can be removed with this action.")
        target = (self.root / current.path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise GitError("Refusing to remove a path outside the repository.") from exc
        if not target.is_file():
            raise GitError("Only a single untracked file can be removed from the GUI.")
        target.unlink()
        return current.path


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn())
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class GitHubIntegrationWidget(QWidget):
    """Thin asynchronous UI over :class:`GitManager`."""

    def __init__(self, parent=None, active_work: Callable[[], bool] | None = None):
        super().__init__(parent)
        self.manager = GitManager()
        self.active_work = active_work or (lambda: False)
        self.pool = QThreadPool.globalInstance()
        self.current = None
        self.activity = []
        self._build()
        # The first visible state should already reflect GitHub, not stale local refs.
        self.refresh_status(True)

    def _build(self):
        layout = QVBoxLayout(self)
        title = QLabel("GitHub Sync")
        title.setStyleSheet("font-size:18px;font-weight:600")
        layout.addWidget(title)

        self.sync_summary = QLabel("Checking local repository and GitHub…")
        self.sync_summary.setStyleSheet("font-size:14px;font-weight:600")
        layout.addWidget(self.sync_summary)

        details = QFormLayout()
        self.repo = QLabel(self.manager.root.name)
        self.branch_label = QLabel("—")
        self.remote = QLabel("origin")
        details.addRow("Repository", self.repo)
        details.addRow("Branch", self.branch_label)
        details.addRow("Remote", self.remote)
        layout.addLayout(details)

        status_box = QGroupBox("Sync Status")
        status_layout = QVBoxLayout(status_box)
        self.state = QLabel("● Checking…")
        self.status_detail = QLabel("Fetching current GitHub status…")
        self.status_detail.setWordWrap(True)
        status_layout.addWidget(self.state)
        status_layout.addWidget(self.status_detail)
        layout.addWidget(status_box)

        row = QHBoxLayout()
        self.refresh = QPushButton("Refresh")
        self.pull_btn = QPushButton("Update from GitHub")
        row.addWidget(self.refresh)
        row.addWidget(self.pull_btn)
        row.addStretch()
        layout.addLayout(row)

        self.changes_heading = QLabel("Local Changes")
        self.changes_heading.setStyleSheet("font-weight:600")
        layout.addWidget(self.changes_heading)
        self.changes_hint = QLabel("No local changes detected.")
        self.changes_hint.setWordWrap(True)
        layout.addWidget(self.changes_hint)
        self.change_list = QListWidget()
        layout.addWidget(self.change_list, 1)

        row = QHBoxLayout()
        self.review = QPushButton("Review Changes")
        self.commit_btn = QPushButton("Commit & Push…")
        row.addWidget(self.review)
        row.addWidget(self.commit_btn)
        row.addStretch()
        layout.addLayout(row)

        self.more_btn = QPushButton("More ▸")
        self.more_btn.setCheckable(True)
        layout.addWidget(self.more_btn)
        self.more_panel = QWidget()
        more_layout = QHBoxLayout(self.more_panel)
        more_layout.setContentsMargins(0, 0, 0, 0)
        folder = QPushButton("Open Repository Folder")
        web = QPushButton("Open GitHub Repository")
        logs = QPushButton("Git Activity")
        for button in (folder, web, logs):
            more_layout.addWidget(button)
        more_layout.addStretch()
        self.more_panel.setVisible(False)
        layout.addWidget(self.more_panel)

        self.refresh.clicked.connect(lambda: self.refresh_status(True))
        self.pull_btn.clicked.connect(self.pull)
        self.review.clicked.connect(self.review_changes)
        self.commit_btn.clicked.connect(self.commit_push)
        self.more_btn.toggled.connect(self._toggle_more)
        folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.manager.root)))
        )
        web.clicked.connect(self.open_remote)
        logs.clicked.connect(self.open_logs)

    def _toggle_more(self, shown):
        self.more_panel.setVisible(bool(shown))
        self.more_btn.setText("More ▾" if shown else "More ▸")

    def _log(self, text):
        self.activity = (self.activity + [f"[GIT] {text}"])[-200:]

    def _async(self, fn, done, description):
        self._log(description)
        task = _Task(fn)
        task.signals.done.connect(done)
        task.signals.failed.connect(self._error)
        self.pool.start(task)

    def _error(self, message):
        self.state.setText("● Git error")
        self.status_detail.setText(message.splitlines()[0])
        self._log("Error: " + message.splitlines()[0])
        QMessageBox.warning(self, "GitHub Sync", message)

    def refresh_status(self, fetch=True):
        self.state.setText("● Checking…")
        self.status_detail.setText("Fetching current GitHub status…" if fetch else "Refreshing local status…")
        self._async(
            lambda: self.manager.status(fetch),
            self._show_status,
            "Checking repository status",
        )

    def _show_status(self, status):
        self.current = status
        self.branch_label.setText(status.branch or "—")
        self.remote.setText(status.remote_url)
        self.repo.setText(self._repository_name(status.remote_url))
        self.state.setText("● " + status.state)
        self.status_detail.setText(status.guidance)
        self.sync_summary.setText(status.sync_summary)

        self.change_list.clear()
        for change in status.changes:
            item = QListWidgetItem(f"{change.label:<10}  {change.path}")
            item.setData(256, change)
            self.change_list.addItem(item)

        count = len(status.changes)
        self.changes_hint.setText(
            "No local changes detected."
            if not count
            else f"{count} local change(s). Review them before updating from GitHub."
        )

        busy = self.active_work()
        has_update = status.behind > 0
        self.pull_btn.setEnabled(status.can_pull and has_update and not busy)
        self.pull_btn.setText("Update from GitHub" if has_update else "Up to date")
        self.review.setEnabled(bool(status.changes))
        self.commit_btn.setEnabled(status.can_push and bool(status.changes) and not busy)

    @staticmethod
    def _repository_name(url):
        value = url.rstrip("/").removesuffix(".git")
        return value.split(":", 1)[-1] if "@" in value else "/".join(value.split("/")[-2:])

    def pull(self):
        if self.active_work():
            return self._error("Update is disabled while a backtest or portfolio run is active.")
        self._async(
            self.manager.pull,
            self._pulled,
            "Updating from GitHub with fast-forward only",
        )

    def _pulled(self, _):
        self._log("Update completed")
        self.refresh_status(False)
        QMessageBox.information(
            self,
            "Update downloaded",
            "Update downloaded successfully.\n\n"
            "Restart Crypto Strategy Lab to use the new version.",
        )

    def review_changes(self):
        if not self.current or not self.current.changes:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Review Local Changes")
        dialog.resize(960, 650)
        outer = QVBoxLayout(dialog)
        body = QHBoxLayout()
        files = QListWidget()
        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        body.addWidget(files, 1)
        body.addWidget(preview, 3)
        outer.addLayout(body, 1)

        for change in self.current.changes:
            item = QListWidgetItem(f"{change.label}: {change.path}")
            item.setData(256, change)
            files.addItem(item)

        actions = QHBoxLayout()
        restore = QPushButton("Restore Tracked File")
        remove = QPushButton("Remove Untracked File")
        restore.setEnabled(False)
        remove.setEnabled(False)
        actions.addWidget(restore)
        actions.addWidget(remove)
        actions.addStretch()
        close = QPushButton("Close")
        actions.addWidget(close)
        outer.addLayout(actions)

        def selected(item):
            preview.setPlainText("Loading preview…")
            change = item.data(256)
            restore.setEnabled(change.can_restore and not self.active_work())
            remove.setEnabled(change.can_remove_untracked and not self.active_work())
            self._async(
                lambda: self.manager.diff(change),
                preview.setPlainText,
                "Reviewing local change",
            )

        def selected_change():
            item = files.currentItem()
            return item.data(256) if item else None

        def finish_cleanup(path):
            self._log(f"Local change cleaned: {path}")
            dialog.accept()
            self.refresh_status(False)

        def restore_selected():
            change = selected_change()
            if not change:
                return
            answer = QMessageBox.question(
                dialog,
                "Restore tracked file?",
                f"Discard all local changes to:\n\n{change.path}\n\n"
                "The file will be restored to the current Git commit.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self._async(
                lambda: self.manager.restore_tracked(change),
                finish_cleanup,
                "Restoring tracked file",
            )

        def remove_selected():
            change = selected_change()
            if not change:
                return
            answer = QMessageBox.question(
                dialog,
                "Remove untracked file?",
                f"Permanently delete this untracked file?\n\n{change.path}\n\n"
                "This cannot be undone by Git because the file is not tracked.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self._async(
                lambda: self.manager.remove_untracked(change),
                finish_cleanup,
                "Removing untracked file",
            )

        files.itemClicked.connect(selected)
        restore.clicked.connect(restore_selected)
        remove.clicked.connect(remove_selected)
        close.clicked.connect(dialog.reject)
        dialog.exec()

    def commit_push(self):
        if self.active_work():
            return self._error(
                "Commit and push is disabled while a backtest or portfolio run is active."
            )
        if not self.current:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Commit & Push")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Branch: {self.current.branch}\n\nFiles to commit:"))
        checks = []
        for change in self.current.changes:
            box = QCheckBox(f"{change.label}: {change.path}")
            blocked = is_sensitive(change.path)
            box.setEnabled(not blocked)
            box.setToolTip(
                "This file appears sensitive and cannot be selected."
                if blocked
                else "Ignored paths are also rejected before staging."
            )
            layout.addWidget(box)
            checks.append((box, change.path))
        message = QLineEdit()
        message.setPlaceholderText("Required commit message")
        layout.addWidget(QLabel("Commit message:"))
        layout.addWidget(message)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Commit & Push")
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        if dialog.exec() != QDialog.Accepted:
            return
        paths = [path for box, path in checks if box.isChecked()]
        branch = self.current.branch

        def operation():
            commit, _ = self.manager.commit(paths, message.text())
            self.manager.push(branch)
            return commit

        self._async(operation, self._pushed, "Committing selected files and pushing")

    def _pushed(self, commit):
        self._log(f"Commit {commit} created")
        self._log("Push completed")
        self.refresh_status(True)
        QMessageBox.information(
            self,
            "GitHub Sync",
            f"Commit {commit} was pushed successfully.",
        )

    def open_remote(self):
        if not self.current:
            return
        url = self.current.remote_url
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url.split(":", 1)[1]
        if url.startswith("https://") or url.startswith("http://"):
            QDesktopServices.openUrl(QUrl(url.removesuffix(".git")))
        else:
            self._error("The origin remote is not a GitHub web URL.")

    def open_logs(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Git Activity")
        dialog.resize(700, 400)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit("\n".join(self.activity))
        text.setReadOnly(True)
        layout.addWidget(text)
        dialog.exec()
