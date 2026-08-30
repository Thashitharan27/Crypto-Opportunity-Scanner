"""Operational observation and bounded recovery policy for the PAPER scanner.

Nothing in this module participates in candidate, signal, or lifecycle decisions.
Health publication is deliberately best-effort and replaceable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError


class HealthStatus(str, Enum):
    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class DiskMonitoringConfig:
    paths: Mapping[str, Path] = field(default_factory=dict)
    cache_path: Path | None = None
    sample_every_cycles: int = 10
    warning_free_percent: float | None = 10.0
    critical_free_percent: float | None = 5.0
    max_cache_files: int = 100_000

    def __post_init__(self) -> None:
        if self.sample_every_cycles <= 0 or self.max_cache_files <= 0:
            raise ValueError("disk sampling cadence and cache bound must be positive")
        warning, critical = self.warning_free_percent, self.critical_free_percent
        for value in (warning, critical):
            if value is not None and not 0 <= value <= 100:
                raise ValueError("disk thresholds must be percentages")
        if warning is not None and critical is not None and critical > warning:
            raise ValueError("critical disk threshold cannot exceed warning threshold")
        object.__setattr__(self, "paths", {k: Path(v) for k, v in self.paths.items()})
        if self.cache_path is not None:
            object.__setattr__(self, "cache_path", Path(self.cache_path))


@dataclass(frozen=True, slots=True)
class ScannerOperationalConfig:
    health_path: Path | None = None
    retry_backoff_cap: timedelta = timedelta(minutes=5)
    crash_recovery_delay: timedelta = timedelta(seconds=30)
    disk: DiskMonitoringConfig = field(default_factory=DiskMonitoringConfig)

    def __post_init__(self) -> None:
        if self.retry_backoff_cap < timedelta(0) or self.retry_backoff_cap > timedelta(hours=1):
            raise ValueError("retry backoff cap must be between zero and one hour")
        if self.crash_recovery_delay < timedelta(0) or self.crash_recovery_delay > timedelta(hours=1):
            raise ValueError("crash recovery delay must be between zero and one hour")
        if self.health_path is not None:
            object.__setattr__(self, "health_path", Path(self.health_path))


def retryable_exception(exc: Exception) -> bool:
    """Conservative retry classification; validation/local-state errors stay closed."""
    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (ConnectionError, TimeoutError, URLError))


def retry_delay_seconds(exc: Exception, attempt: int, base: timedelta, cap: timedelta) -> float:
    delay = base.total_seconds() * (2 ** max(0, attempt - 1))
    if isinstance(exc, HTTPError) and exc.code == 429:
        try:
            delay = max(delay, float(exc.headers.get("Retry-After", "")))
        except (TypeError, ValueError):
            pass
    return min(delay, cap.total_seconds())


@dataclass(frozen=True, slots=True)
class ScannerHealthSnapshot:
    schema_version: int
    generated_at: str
    runtime_started_at: str
    status: str
    last_cycle_id: str | None = None
    last_scan_run_id: str | None = None
    last_cycle_status: str | None = None
    last_successful_scan_at: str | None = None
    last_failure_at: str | None = None
    consecutive_failed_cycles: int = 0
    last_error_category: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    disk: Mapping[str, Any] = field(default_factory=dict)


class HealthReporter:
    """Atomically publishes latest health; callers decide that failures are non-fatal."""
    def __init__(self, path: Path | None):
        self.path = path

    def publish(self, snapshot: ScannerHealthSnapshot) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":")) + "\n"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class DiskMonitor:
    """Read-only, cadence-controlled filesystem observer with bounded cache walk."""
    def __init__(self, config: DiskMonitoringConfig, usage: Callable[[Path], Any] = shutil.disk_usage):
        self.config, self.usage, self.samples, self.latest = config, usage, 0, {}

    def sample(self, force: bool = False) -> Mapping[str, Any]:
        self.samples += 1
        if not force and self.latest and self.samples % self.config.sample_every_cycles:
            return self.latest
        volumes, result = {}, {}
        for name, configured in self.config.paths.items():
            path = configured if configured.exists() else configured.parent
            key = str(path.resolve().anchor or path.resolve())
            if key not in volumes:
                usage = self.usage(path)
                free_percent = usage.free * 100.0 / usage.total if usage.total else None
                volumes[key] = {"total_bytes": usage.total, "free_bytes": usage.free,
                                "free_percent": free_percent,
                                "level": self._level(free_percent)}
            result[name] = volumes[key]
        if self.config.cache_path is not None:
            size = files = 0
            if self.config.cache_path.exists():
                for entry in self.config.cache_path.rglob("*"):
                    if entry.is_file():
                        size += entry.stat().st_size; files += 1
                        if files >= self.config.max_cache_files:
                            break
            result["cache"] = {"size_bytes": size, "files_sampled": files,
                               "truncated": files >= self.config.max_cache_files}
        self.latest = result
        return result

    def _level(self, free: float | None) -> str:
        if free is None: return "UNKNOWN"
        if self.config.critical_free_percent is not None and free <= self.config.critical_free_percent: return "CRITICAL"
        if self.config.warning_free_percent is not None and free <= self.config.warning_free_percent: return "WARNING"
        return "OK"


def aggregate_acquisition_metrics(candle_results=(), dataset_results=(), readiness_results=()) -> dict[str, Any]:
    """Count existing Task 3/6 enum values without reinterpreting validity."""
    def value(item: Any, name: str) -> str | None:
        raw = item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)
        return getattr(raw, "value", raw)

    candle = list(candle_results)
    datasets = list(dataset_results)
    readiness = list(readiness_results)
    states = ("REUSED", "ACQUIRED", "MISSING", "QUALITY_FAILED", "DOWNLOAD_FAILED", "CANCELLED")
    candle_counts = {state.lower(): sum(value(item, "state") == state for item in candle) for state in states}
    dataset_counts = {state.lower(): sum(value(item, "state") == state for item in datasets) for state in states}
    candle_counts.update({
        "requested_symbols": len(candle),
        "rows_available": sum(int(getattr(item, "row_count", 0) or 0) for item in candle),
        "coverage_gaps_attempted": sum(len(getattr(item, "acquisition_ranges", ()) or ()) for item in candle),
    })
    ready_states = ("READY", "DEGRADED", "UNAVAILABLE")
    return {"strategy_candles": candle_counts,
            "rich_datasets": {"dataset_requirements": len(datasets), **dataset_counts},
            "feature_readiness": {state.lower(): sum(value(item, "readiness") == state for item in readiness)
                                  for state in ready_states}}
