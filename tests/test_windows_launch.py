from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from crypto_strategy_lab.gui import chatgpt_connection
from crypto_strategy_lab.gui import github_manager
from crypto_strategy_lab.gui.github_manager import CREATE_NO_WINDOW, GitManager


ROOT = Path(__file__).resolve().parents[1]


def test_no_console_launcher_quotes_paths_and_uses_project_venv():
    launcher = (ROOT / "Crypto Strategy Lab.vbs").read_text()
    assert 'files.BuildPath(root, ".venv\\Scripts\\pythonw.exe")' in launcher
    assert 'files.BuildPath(root, "app.py")' in launcher
    assert 'shell.CurrentDirectory = root' in launcher
    assert 'shell.Run """" & pythonw & """ """ & appPath & """", 0, False' in launcher
    assert "cmd.exe" not in launcher.lower()
    assert "powershell.exe" not in launcher.lower()


def test_qprocess_create_modifier_adds_no_window_only_on_windows(monkeypatch):
    process = SimpleNamespace(setCreateProcessArgumentsModifier=Mock())
    monkeypatch.setattr(chatgpt_connection.sys, "platform", "win32")
    chatgpt_connection.configure_hidden_process(process)
    modifier = process.setCreateProcessArgumentsModifier.call_args.args[0]
    arguments = SimpleNamespace(flags=4)
    modifier(arguments)
    assert arguments.flags == 4 | chatgpt_connection.CREATE_NO_WINDOW


def test_qprocess_without_create_modifier_still_initializes(monkeypatch):
    class SignalStub:
        def connect(self, callback):
            self.callback = callback

    class ProcessWithoutModifier:
        MergedChannels = object()

        def __init__(self, parent=None):
            # Mirror the normal QProcess signal surface. This test intentionally
            # removes only setCreateProcessArgumentsModifier.
            self.readyReadStandardOutput = SignalStub()
            self.started = SignalStub()
            self.errorOccurred = SignalStub()
            self.finished = SignalStub()

        def setProcessChannelMode(self, mode):
            self.channel_mode = mode

    monkeypatch.setattr(chatgpt_connection.sys, "platform", "win32")
    monkeypatch.setattr(chatgpt_connection, "QProcess", ProcessWithoutModifier)

    manager = chatgpt_connection.ChatGPTConnectionManager(lambda: ".")

    assert manager.mcp.channel_mode is ProcessWithoutModifier.MergedChannels
    assert manager.tunnel.channel_mode is ProcessWithoutModifier.MergedChannels


def test_git_commands_use_argument_list_without_a_shell(monkeypatch, tmp_path):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr("crypto_strategy_lab.gui.github_manager.subprocess.run", run)
    monkeypatch.setattr(github_manager, "os", SimpleNamespace(name="nt"))
    GitManager(tmp_path)._run("status")
    args, kwargs = run.call_args
    assert args[0] == ["git", "status"]
    assert kwargs["shell"] is False
    assert kwargs["creationflags"] == CREATE_NO_WINDOW
