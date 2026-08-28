from __future__ import annotations

import json
from types import SimpleNamespace

from tools.data_lake_reporting_profile import (
    ProfiledReporter,
    _artifact_summary,
    _profile_rows,
)


class _FakeReporter:
    def __init__(self) -> None:
        self.begun = False
        self.reported = False

    def begin(self, request, config) -> None:
        self.begun = True

    def report(self, result, context) -> None:
        self.reported = True
        total = 0
        for value in range(5000):
            total += value * value
        result.total = total


def test_profiled_reporter_profiles_only_report_call() -> None:
    delegate = _FakeReporter()
    reporter = ProfiledReporter(delegate)
    reporter.begin(object(), object())
    result = SimpleNamespace()

    reporter.report(result, object())

    assert delegate.begun is True
    assert delegate.reported is True
    assert reporter.elapsed_seconds is not None
    assert reporter.elapsed_seconds >= 0
    assert result.total > 0
    rows = _profile_rows(reporter.profile, "cumulative", 20)
    assert any(row["function"] == "report" for row in rows)


def test_artifact_summary_extracts_reporting_shape(tmp_path) -> None:
    manifest = {
        "execution_result": {"stage_timings": {"reporting": 3.25}},
        "research": {
            "artifact_write_seconds": 1.5,
            "trade_row_count": 12,
            "feature_context_row_count": 8760,
            "trade_columns": ["a", "b"],
            "feature_context_columns": ["x", "y", "z"],
            "trade_context_parity_columns": ["x", "z"],
        },
        "artifacts": {
            "trades": {
                "path": "artifacts/trades.parquet",
                "format": "parquet",
                "rows": 12,
                "bytes": 1234,
                "sha256": "ignored",
            }
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    reporting, artifacts = _artifact_summary(tmp_path)

    assert reporting == {
        "manifest_reporting_seconds": 3.25,
        "research_artifact_write_seconds": 1.5,
        "trade_rows": 12,
        "feature_context_rows": 8760,
        "trade_columns": 2,
        "feature_context_columns": 3,
        "trade_context_parity_columns": 2,
    }
    assert artifacts == {
        "trades": {
            "path": "artifacts/trades.parquet",
            "format": "parquet",
            "rows": 12,
            "bytes": 1234,
        }
    }
