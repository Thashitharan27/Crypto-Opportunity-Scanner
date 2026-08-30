from datetime import datetime, timedelta, timezone
from email.message import Message
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from crypto_strategy_lab.data.binance.universe import BinanceUsdMDiscoveryClient
from crypto_strategy_lab.scanner_operations import (
    DiskMonitor, DiskMonitoringConfig, HealthReporter, HealthStatus,
    ScannerHealthSnapshot, ScannerOperationalConfig,
    aggregate_acquisition_metrics, retry_delay_seconds, retryable_exception,
)


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return HTTPError("https://example.test", code, "failure", headers, None)


@pytest.mark.parametrize("error", [ConnectionError(), TimeoutError(), URLError("down"),
                                    http_error(408), http_error(429), http_error(503)])
def test_retryable_failures_are_explicit(error):
    assert retryable_exception(error)


@pytest.mark.parametrize("error", [ValueError(), KeyError(), http_error(400), http_error(418)])
def test_deterministic_and_ban_failures_do_not_retry(error):
    assert not retryable_exception(error)


def test_exponential_retry_after_and_cap_are_deterministic():
    base, cap = timedelta(seconds=2), timedelta(seconds=10)
    assert retry_delay_seconds(ConnectionError(), 1, base, cap) == 2
    assert retry_delay_seconds(ConnectionError(), 3, base, cap) == 8
    assert retry_delay_seconds(http_error(429, 9), 1, base, cap) == 9
    assert retry_delay_seconds(http_error(429, 99), 1, base, cap) == 10


def test_operational_configuration_validation(tmp_path):
    with pytest.raises(ValueError):
        ScannerOperationalConfig(retry_backoff_cap=timedelta(hours=2))
    with pytest.raises(ValueError):
        DiskMonitoringConfig(warning_free_percent=5, critical_free_percent=10)
    assert ScannerOperationalConfig(health_path=tmp_path / "health.json").health_path.is_absolute()


def test_binance_transport_emits_optional_header_telemetry():
    observed = []

    class Response(BytesIO):
        status = 200
        headers = {"Retry-After": "3", "X-MBX-USED-WEIGHT-1M": "41"}
        def __enter__(self): return self
        def __exit__(self, *_): self.close()

    client = BinanceUsdMDiscoveryClient(
        transport=lambda url, timeout: Response(b"[]"), telemetry=observed.append
    )
    assert client.tickers_24h() == []
    assert observed == [{"endpoint": "/fapi/v1/ticker/24hr", "http_status": 200,
                         "retry_after_seconds": "3", "used_weight": "41"}]


def test_binance_http_error_emits_available_rate_limit_telemetry():
    observed = []
    error = http_error(429, 7)
    error.headers["X-MBX-USED-WEIGHT-1M"] = "88"
    client = BinanceUsdMDiscoveryClient(
        transport=lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        telemetry=observed.append,
    )
    with pytest.raises(HTTPError) as raised:
        client.tickers_24h()
    assert raised.value is error
    assert observed == [{
        "endpoint": "/fapi/v1/ticker/24hr", "http_status": 429,
        "retry_after_seconds": "7", "used_weight": "88",
    }]


def snapshot(status="STARTING"):
    return ScannerHealthSnapshot(1, "2026-01-01T00:00:00+00:00",
                                 "2026-01-01T00:00:00+00:00", status)


def test_health_publication_is_atomic_and_versioned(tmp_path, monkeypatch):
    path = tmp_path / "status" / "health.json"
    replacements = []
    import crypto_strategy_lab.scanner_operations as operations
    real_replace = operations.os.replace
    monkeypatch.setattr(operations.os, "replace", lambda source, target:
                        (replacements.append((Path(source), Path(target))), real_replace(source, target))[1])
    HealthReporter(path).publish(snapshot(HealthStatus.HEALTHY.value))
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1 and payload["status"] == "HEALTHY"
    assert replacements[0][1] == path
    assert not list(path.parent.glob("*.tmp"))


def test_disk_monitor_is_read_only_cadenced_and_classified(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir(); (cache / "kept.bin").write_bytes(b"1234")
    calls = []
    monitor = DiskMonitor(DiskMonitoringConfig(
        paths={"paper": tmp_path, "output": tmp_path / "output"}, cache_path=cache,
        sample_every_cycles=3, warning_free_percent=20, critical_free_percent=10,
    ), usage=lambda path: (calls.append(path), SimpleNamespace(total=100, used=85, free=15))[1])
    first = monitor.sample(force=True); second = monitor.sample()
    assert first == second and first["paper"]["level"] == "WARNING"
    assert first["cache"]["size_bytes"] == 4 and (cache / "kept.bin").read_bytes() == b"1234"
    assert len(calls) == 1  # same volume is deduplicated


def test_disk_monitor_deduplicates_by_device_not_posix_anchor(tmp_path):
    same_a, same_b, other = (tmp_path / name for name in ("a", "b", "other"))
    calls = []
    devices = {same_a: 11, same_b: 11, other: 22}

    def stat(path):
        return SimpleNamespace(st_dev=devices[Path(path)])

    def usage(path):
        calls.append(Path(path))
        free = 50 if Path(path) in {same_a, same_b} else 5
        return SimpleNamespace(total=100, used=100 - free, free=free)

    monitor = DiskMonitor(
        DiskMonitoringConfig(
            paths={"same_a": same_a, "same_b": same_b, "other": other},
            sample_every_cycles=1, warning_free_percent=20,
            critical_free_percent=10,
        ),
        usage=usage,
        filesystem_stat=stat,
    )
    sample = monitor.sample()
    assert calls == [same_a, other]
    assert sample["same_a"] is sample["same_b"]
    assert sample["same_a"]["level"] == "OK"
    assert sample["other"]["level"] == "CRITICAL"


def test_cache_walk_tolerates_disappearing_file(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    raced = cache / "raced.bin"
    raced.write_bytes(b"data")
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        if Path(path) == raced:
            raise FileNotFoundError(str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    sample = DiskMonitor(DiskMonitoringConfig(cache_path=cache)).sample(force=True)
    assert sample["cache"]["size_bytes"] == 0
    assert sample["cache"]["entries_disappeared"] == 1


def test_acquisition_metrics_use_existing_state_and_readiness_values():
    candles = [
        {"acquisition_state": "REUSED", "row_count": 5,
         "acquisition_ranges": "[]"},
        {"acquisition_state": "DOWNLOAD_FAILED", "row_count": 0,
         "acquisition_ranges": '[["a","b"],["c","d"]]'},
    ]
    # Dataset A is coalesced for two features; feature one uses two datasets.
    rich = [
        {"symbol": "BTCUSDT", "dataset": "A", "interval": "5m",
         "requested_start": "s", "requested_end": "e", "acquisition_state": "ACQUIRED",
         "feature_name": "one", "feature_readiness": "READY"},
        {"symbol": "BTCUSDT", "dataset": "A", "interval": "5m",
         "requested_start": "s", "requested_end": "e", "acquisition_state": "ACQUIRED",
         "feature_name": "two", "feature_readiness": "DEGRADED"},
        {"symbol": "BTCUSDT", "dataset": "B", "interval": "1h",
         "requested_start": "s", "requested_end": "e", "acquisition_state": "MISSING",
         "feature_name": "one", "feature_readiness": "READY"},
    ]
    result = aggregate_acquisition_metrics(candles, rich)
    assert result["strategy_candle_acquisition"] == {
        "reused": 1, "acquired": 0, "missing": 0, "quality_failed": 0,
        "download_failed": 1, "cancelled": 0, "requested_symbols": 2,
        "rows_available": 5, "coverage_gaps_attempted": 2,
    }
    assert result["rich_dataset_acquisition"]["dataset_requirements"] == 2
    assert result["rich_dataset_acquisition"]["acquired"] == 1
    assert result["rich_dataset_acquisition"]["missing"] == 1
    assert result["rich_feature_readiness"] == {"ready": 1, "degraded": 1, "unavailable": 0}


def test_acquisition_metrics_reject_conflicting_denormalized_feature_rows():
    rows = [
        {"symbol": "BTCUSDT", "dataset": name, "interval": "5m",
         "requested_start": "s", "requested_end": "e", "acquisition_state": "ACQUIRED",
         "feature_name": "feature", "feature_readiness": readiness}
        for name, readiness in (("A", "READY"), ("B", "DEGRADED"))
    ]
    with pytest.raises(ValueError, match="conflicting feature readiness"):
        aggregate_acquisition_metrics([], rows)


def test_windows_launchers_are_local_quoted_and_scanner_named():
    root = Path(__file__).parents[1]
    vbs = (root / "Crypto Opportunity Scanner.vbs").read_text()
    debug = (root / "Debug Launcher.bat").read_text()
    assert ".venv\\Scripts\\pythonw.exe" in vbs and "Crypto Opportunity Scanner" in vbs
    assert 'shell.Run """" & pythonw' in vbs and "cmd.exe" not in vbs and "powershell" not in vbs.lower()
    assert ".venv\\Scripts\\python.exe" in debug and "Crypto Opportunity Scanner" in debug
    assert "exit /b %exit_code%" in debug and "pause" in debug
