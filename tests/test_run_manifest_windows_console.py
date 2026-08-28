from crypto_strategy_lab import run_manifest


def test_windows_git_provenance_uses_hidden_process_flag():
    assert run_manifest._git_subprocess_kwargs("nt") == {
        "creationflags": 0x08000000
    }


def test_non_windows_git_provenance_does_not_set_windows_flags():
    assert run_manifest._git_subprocess_kwargs("posix") == {}


def test_capture_code_provenance_applies_process_kwargs_to_all_git_calls(monkeypatch, tmp_path):
    calls = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["rev-parse", "HEAD"]:
            return Result("abc123\n")
        if command[1] == "status":
            return Result(" M app.py\n")
        if command[1] == "diff":
            return Result(b"diff-bytes")
        raise AssertionError(command)

    monkeypatch.setattr(
        run_manifest,
        "_git_subprocess_kwargs",
        lambda platform_name=None: {"creationflags": 0x08000000},
    )
    monkeypatch.setattr(run_manifest.subprocess, "run", fake_run)

    result = run_manifest.capture_code_provenance(tmp_path)

    assert result["code_commit"] == "abc123"
    assert result["code_dirty"] is True
    assert len(calls) == 3
    assert all(kwargs["creationflags"] == 0x08000000 for _, kwargs in calls)
