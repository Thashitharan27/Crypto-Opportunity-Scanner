from types import SimpleNamespace

from crypto_strategy_lab.gui import github_sync_install
from crypto_strategy_lab.gui.github_sync_install import (
    GitUpdateResult,
    apply_github_sync_safety,
    is_dependency_file,
)


class _Field:
    def __init__(self):
        self.text = None
        self.enabled = True

    def setText(self, value):
        self.text = value

    def setEnabled(self, value):
        self.enabled = bool(value)


class _Manager:
    def __init__(self, changed_files=(), updated=True):
        self.changed_files = tuple(changed_files)
        self.pull_calls = 0
        self.updated = updated

    def pull(self):
        self.pull_calls += 1
        return "Fast-forward" if self.updated else "Already up to date."

    def _run(self, *args, **_kwargs):
        if args == ("rev-parse", "HEAD"):
            head = "old" if self.pull_calls == 0 or not self.updated else "new"
            return SimpleNamespace(stdout=head + "\n")
        if args[:2] == ("diff", "--name-only"):
            return SimpleNamespace(stdout="\n".join(self.changed_files) + "\n")
        raise AssertionError(args)


class _Widget:
    def __init__(self, manager):
        self.manager = manager
        self.state = _Field()
        self.status_detail = _Field()
        self.pull_btn = _Field()
        self.refresh = _Field()
        self.review = _Field()
        self.commit_btn = _Field()
        self.logs = []
        self.refresh_calls = []
        self.errors = []

    def _log(self, value):
        self.logs.append(value)

    def refresh_status(self, fetch):
        self.refresh_calls.append(fetch)

    def _error(self, value):
        self.errors.append(value)


class _Window:
    def __init__(self, widget):
        self.widget = widget
        self._thread = None

    def findChild(self, _cls):
        return self.widget


def test_dependency_file_detection_covers_current_and_common_python_metadata():
    assert is_dependency_file("requirements.txt")
    assert is_dependency_file("requirements-dev.txt")
    assert is_dependency_file("requirements/base.txt")
    assert is_dependency_file("pyproject.toml")
    assert is_dependency_file("uv.lock")
    assert not is_dependency_file("app.py")
    assert not is_dependency_file("crypto_strategy_lab/gui/github_manager.py")


def test_normal_update_stays_open_and_requires_manual_restart(monkeypatch):
    manager = _Manager(("app.py", "crypto_strategy_lab/gui/example.py"))
    widget = _Widget(manager)
    window = _Window(widget)
    messages = []

    monkeypatch.setattr(
        github_sync_install.QMessageBox,
        "information",
        lambda *_args: messages.append(_args[-1]),
    )

    apply_github_sync_safety(window)
    result = widget.manager.pull()
    assert isinstance(result, GitUpdateResult)
    assert result.updated
    assert not result.dependency_files

    widget._pulled(result)

    assert widget.state.text == "● Update installed — restart required"
    assert "Close and reopen Crypto Strategy Lab" in widget.status_detail.text
    assert not widget.pull_btn.enabled
    assert not widget.refresh.enabled
    assert not widget.review.enabled
    assert not widget.commit_btn.enabled
    assert messages
    assert "Close and reopen Crypto Strategy Lab" in messages[0]
    assert not hasattr(window, "_restart_after_exit")


def test_dependency_update_stays_open_and_requires_environment_refresh(monkeypatch):
    manager = _Manager(("requirements.txt", "app.py"))
    widget = _Widget(manager)
    window = _Window(widget)
    messages = []

    monkeypatch.setattr(
        github_sync_install.QMessageBox,
        "information",
        lambda *_args: messages.append(_args[-1]),
    )

    apply_github_sync_safety(window)
    result = widget.manager.pull()
    assert result.dependency_files == ("requirements.txt",)

    widget._pulled(result)

    assert widget.state.text == "● Update installed — restart required"
    assert "Dependency files changed" in widget.status_detail.text
    assert messages
    assert "Close Crypto Strategy Lab" in messages[0]
    assert "pip install -r requirements.txt" in messages[0]
    assert not hasattr(window, "_restart_after_exit")


def test_no_actual_update_refreshes_normally(monkeypatch):
    manager = _Manager(updated=False)
    widget = _Widget(manager)
    window = _Window(widget)

    monkeypatch.setattr(
        github_sync_install.QMessageBox,
        "information",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected message")),
    )

    apply_github_sync_safety(window)
    result = widget.manager.pull()
    assert not result.updated

    widget._pulled(result)

    assert widget.refresh_calls == [False]
    assert not hasattr(window, "_restart_after_exit")


def test_app_entrypoint_has_no_relaunch_hook():
    source = open("app.py", encoding="utf-8").read()
    assert "launch_replacement" not in source
    assert "_restart_after_exit" not in source
