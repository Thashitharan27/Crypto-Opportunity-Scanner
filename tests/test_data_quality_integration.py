from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from crypto_strategy_lab.data import (
    DataQualityError,
    DataQualityIssue,
    DataQualityReport,
    DataQualityStatus,
    DataRequest,
    DatasetKind,
    DatasetQualityReport,
    MarketDataStore,
)
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.research_runner import ResearchRunner
import crypto_strategy_lab.research_runner as runner_module


UTC = timezone.utc


def _request(interval="1h"):
    return DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 1, 4, tzinfo=UTC),
        strategy_interval=interval,
    )


def _rows(conflicting=False):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for hour in range(4):
        open_ms = int((start + pd.Timedelta(hours=hour)).timestamp() * 1000)
        close = 2.5 if conflicting and hour == 1 else 2.0
        rows.append(
            f"{open_ms},2,3,1,{close},10,{open_ms + 3_599_999},20,2,4,8,0"
        )
    return rows


def _write_zip(root: Path, frequency: str, name_date: str, rows):
    directory = root / "raw" / "futures" / "um" / frequency / "klines" / "BTCUSDT" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"BTCUSDT-1h-{name_date}"
    path = directory / f"{filename}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{filename}.csv", "\n".join(rows) + "\n")


def test_warm_quality_cache_hit_does_not_reload_canonical_rows(tmp_path, monkeypatch):
    root = tmp_path / "lake"
    _write_zip(root, "daily", "2026-01-01", _rows())
    store = MarketDataStore(root, tmp_path / "cache")
    store.refresh_catalog()

    first = store.data_quality_report(
        _request(), DatasetKind.KLINES, interval="1h", required=True
    )
    assert first.status is DataQualityStatus.OK
    assert first.cache_hit is False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("warm quality hit reloaded canonical rows")

    monkeypatch.setattr(store, "load_dataset", forbidden)
    second = store.data_quality_report(
        _request(), DatasetKind.KLINES, interval="1h", required=True
    )
    assert second.status is DataQualityStatus.OK
    assert second.cache_hit is True


def test_store_quality_report_surfaces_identical_archive_overlap(tmp_path):
    root = tmp_path / "lake"
    _write_zip(root, "daily", "2026-01-01", _rows())
    _write_zip(root, "monthly", "2026-01", _rows())
    store = MarketDataStore(root, tmp_path / "cache")
    store.refresh_catalog()

    report = store.data_quality_report(
        _request(), DatasetKind.KLINES, interval="1h", required=True
    )
    codes = {issue.code for issue in report.issues}
    assert "ARCHIVE_OVERLAP" in codes
    assert "IDENTICAL_ARCHIVE_OVERLAP" in codes
    assert "CONFLICTING_ARCHIVE_OVERLAP" not in codes
    assert report.status is DataQualityStatus.WARN


def test_store_quality_report_rejects_conflicting_archive_overlap(tmp_path):
    root = tmp_path / "lake"
    _write_zip(root, "daily", "2026-01-01", _rows())
    _write_zip(root, "monthly", "2026-01", _rows(conflicting=True))
    store = MarketDataStore(root, tmp_path / "cache")
    store.refresh_catalog()

    report = store.data_quality_report(
        _request(), DatasetKind.KLINES, interval="1h", required=True
    )
    assert any(issue.code == "CONFLICTING_ARCHIVE_OVERLAP" for issue in report.issues)
    assert report.status is DataQualityStatus.ERROR


def _fatal_report(request):
    issue = DataQualityIssue(
        code="MISSING_INTERNAL_INTERVAL",
        severity=DataQualityStatus.ERROR,
        message="required interval missing",
    )
    dataset = DatasetQualityReport(
        dataset="klines",
        symbol=request.symbol,
        interval=request.strategy_interval,
        required=True,
        requested_start=pd.Timestamp(request.start).isoformat(),
        requested_end=pd.Timestamp(request.end).isoformat(),
        observed_start=None,
        observed_end=None,
        complete_start=None,
        complete_end=None,
        row_count=0,
        source_identity="bad-source",
        status=DataQualityStatus.ERROR,
        issues=(issue,),
    )
    return DataQualityReport((dataset,))


def test_fatal_quality_error_stops_runner_before_simulator_and_reporters(monkeypatch):
    config = ResearchRunConfig()
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_interval="15m",
        intrabar_interval="1m",
    )
    report = _fatal_report(request)

    def fail_load(*_args, **_kwargs):
        raise DataQualityError("bad required data", report)

    monkeypatch.setattr(runner_module, "load_backtest_bundle", fail_load)

    class Simulator:
        called = False
        def run(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("simulator must not run after fatal data quality")

    class Reporter:
        called = False
        def report(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("reporter must not run after failed preparation")

    simulator, reporter = Simulator(), Reporter()
    runner = ResearchRunner(object(), object(), object(), object(), simulator, (reporter,))
    with pytest.raises(DataQualityError):
        runner.run(request, config, refresh_catalog=False)
    assert simulator.called is False
    assert reporter.called is False
