from __future__ import annotations

import pytest

pytest.importorskip("PySide6", exc_type=ImportError)

from crypto_strategy_lab.data import DataQualityIssue, DataQualityStatus
from crypto_strategy_lab.gui.run_readiness import MainWindow


class _IssueRenderHarness:
    @staticmethod
    def _time_text(value) -> str:
        return str(value)

    @staticmethod
    def _issue_text(issue) -> str:
        return issue.code.replace("_", " ").lower()

    @staticmethod
    def _missing_dataset_line(_report) -> str:
        return "missing dataset detail"


def test_unrecognized_blocking_issue_is_not_hidden_from_readiness():
    issue = DataQualityIssue(
        "CONFLICTING_ARCHIVE_OVERLAP",
        DataQualityStatus.ERROR,
        "Overlapping archives disagree for a logical key",
        count=2,
    )
    lines: list[str] = []

    MainWindow._append_issue_line(_IssueRenderHarness(), lines, issue)

    assert len(lines) == 1
    assert "Overlapping archives disagree for a logical key" in lines[0]
    assert "CONFLICTING_ARCHIVE_OVERLAP" in lines[0]
    assert "2 affected" in lines[0]


def test_unrecognized_warning_does_not_clutter_blocking_readiness():
    issue = DataQualityIssue(
        "IDENTICAL_ARCHIVE_OVERLAP",
        DataQualityStatus.WARN,
        "Overlapping source rows are identical",
        count=2,
    )
    lines: list[str] = []

    MainWindow._append_issue_line(_IssueRenderHarness(), lines, issue)

    assert lines == []


def test_explicit_revalidate_refreshes_catalog_before_quality_validation():
    calls: list[object] = []

    class Service:
        @staticmethod
        def refresh_catalog():
            calls.append("refresh_catalog")

    class Debounce:
        @staticmethod
        def stop():
            calls.append("stop_debounce")

    class Harness:
        _validation_thread = None
        service = Service()
        _validation_debounce = Debounce()

        @staticmethod
        def _set_readiness(*_args, **_kwargs):
            calls.append("set_readiness")

        @staticmethod
        def _invalidate_range_validation():
            calls.append("invalidate")

        @staticmethod
        def _load_catalog():
            calls.append("load_catalog")

        @staticmethod
        def _begin_required_data_validation(*, auto_run):
            calls.append(("validate", auto_run))

    MainWindow.validate_selected_range(Harness())

    assert calls.index("refresh_catalog") < calls.index("load_catalog")
    assert calls.index("load_catalog") < calls.index(("validate", False))
    assert "invalidate" in calls
    assert "stop_debounce" in calls
