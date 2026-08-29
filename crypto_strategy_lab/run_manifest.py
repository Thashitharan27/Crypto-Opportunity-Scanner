"""Canonical, immutable completed-run provenance contract."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import uuid
from enum import Enum

RUN_MANIFEST_CONTRACT = "crypto_strategy_lab_run_v1"
RUN_MANIFEST_VERSION = 1
CATALOG_SNAPSHOT_CONTRACT = "selected_source_catalog_v1"
PREPARED_CACHE_CONTRACT = "prepared_backtest_frame_v1"
FEATURE_RESEARCH_ARTIFACT_CONTRACT = "feature_research_v1"
FEATURE_RESEARCH_ARTIFACT_VERSION = 1
OPPORTUNITY_SCAN_ARTIFACT_CONTRACT = "opportunity_scan_v1"
OPPORTUNITY_SCAN_ARTIFACT_VERSION = 1
_CREATE_NO_WINDOW = 0x08000000


class RunArtifactError(ValueError):
    """A completed run or cataloged immutable artifact is invalid."""


class RunType(str, Enum):
    BACKTEST = "BACKTEST"
    OPPORTUNITY_SCAN = "OPPORTUNITY_SCAN"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_entry(path: Path, run_dir: Path, fmt: str, rows: int | None = None,
                  schema_version: int = 1, **extra: Any) -> dict[str, Any]:
    """Catalog one immutable artifact using the common run SHA-256 contract."""
    return {"path": path.relative_to(run_dir).as_posix(), "format": fmt,
            "schema_version": schema_version, "rows": rows,
            "bytes": path.stat().st_size, "sha256": file_sha256(path), **extra}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def new_run_identity() -> tuple[str, str]:
    return uuid.uuid4().hex, datetime.now(timezone.utc).isoformat()


def _git_subprocess_kwargs(platform_name: str | None = None) -> dict[str, int]:
    """Hide internal Git helper consoles when provenance runs inside the Windows GUI."""
    name = os.name if platform_name is None else platform_name
    return {"creationflags": _CREATE_NO_WINDOW} if name == "nt" else {}


def capture_code_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Capture code identity without treating generated outputs as code changes."""
    repo = Path(repo or Path(__file__).resolve().parents[1])
    process_kwargs = _git_subprocess_kwargs()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            **process_kwargs,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            **process_kwargs,
        ).stdout
        lines = [line for line in status.splitlines() if line.strip()]
        tracked_changes = [line for line in lines if not line.startswith("?? ")]
        untracked_source_paths = sorted(
            line[3:]
            for line in lines
            if line.startswith("?? ")
            and Path(line[3:]).suffix in {".py", ".toml", ".yaml", ".yml"}
            and Path(line[3:]).name != ".env"
        )
        dirty = bool(tracked_changes or untracked_source_paths)
        result: dict[str, Any] = {
            "code_commit": commit,
            "code_dirty": dirty,
            "code_provenance_status": (
                "DIRTY_WORKTREE" if dirty else "CLEAN_COMMIT"
            ),
            "reproducibility_status": "PARTIAL" if dirty else "REPRODUCIBLE",
        }
        if dirty:
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                **process_kwargs,
            ).stdout
            result["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
            result["untracked_source_paths"] = untracked_source_paths
        return result
    except (OSError, subprocess.SubprocessError):
        return {
            "code_commit": None,
            "code_dirty": None,
            "code_provenance_status": "GIT_UNAVAILABLE",
            "reproducibility_status": "PARTIAL",
        }


def runtime_provenance(repo: Path | None = None) -> dict[str, Any]:
    requirements = (
        Path(repo or Path(__file__).resolve().parents[1]) / "requirements.txt"
    )
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "project_config_contract": "research_run_config_v3",
        "project_config_version": 3,
        "requirements_sha256": (
            file_sha256(requirements) if requirements.is_file() else None
        ),
    }


def config_snapshot(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def config_hashes(
    config: Any, effective_intrabar_interval: str | None
) -> dict[str, str]:
    """Hash semantic config scopes without letting reporting invalidate execution."""
    snapshot = config_snapshot(config)
    data = snapshot["data"]
    execution_data = {
        "strategy_timeframe_minutes": data.get("strategy_timeframe_minutes"),
        "use_intrabar_data": data.get("use_intrabar_data"),
        "requested_intrabar_timeframe_minutes": data.get(
            "intrabar_timeframe_minutes"
        ),
        "effective_intrabar_interval": effective_intrabar_interval,
        "intrabar_missing_policy": data.get("intrabar_missing_policy"),
    }
    return {
        "strategy_hash": canonical_sha256(snapshot["strategy"]),
        "execution_hash": canonical_sha256(
            {"execution": snapshot["execution"], "execution_data": execution_data}
        ),
        "feature_config_hash": canonical_sha256(snapshot["features"]),
        "data_config_hash": canonical_sha256(snapshot["data"]),
    }


@lru_cache(maxsize=None)
def _canonical_contract_for_dataset(dataset: Any) -> Mapping[str, Any]:
    """Resolve the canonical adapter contract once per DatasetKind."""
    from .data.schemas import DatasetKind
    from .data.binance.events import (
        BookDepthArchiveAdapter,
        BookTickerArchiveAdapter,
        FundingRateArchiveAdapter,
        FuturesMetricsArchiveAdapter,
    )
    from .data.binance.klines import KlineArchiveAdapter, KlineLikeArchiveAdapter
    from .data.binance.trades import AggTradesArchiveAdapter, TradesArchiveAdapter

    adapters = {
        DatasetKind.KLINES: KlineArchiveAdapter(),
        DatasetKind.MARK_PRICE_KLINES: KlineLikeArchiveAdapter(
            DatasetKind.MARK_PRICE_KLINES
        ),
        DatasetKind.INDEX_PRICE_KLINES: KlineLikeArchiveAdapter(
            DatasetKind.INDEX_PRICE_KLINES
        ),
        DatasetKind.PREMIUM_INDEX_KLINES: KlineLikeArchiveAdapter(
            DatasetKind.PREMIUM_INDEX_KLINES
        ),
        DatasetKind.FUTURES_METRICS: FuturesMetricsArchiveAdapter(),
        DatasetKind.FUNDING_RATE: FundingRateArchiveAdapter(),
        DatasetKind.AGG_TRADES: AggTradesArchiveAdapter(),
        DatasetKind.TRADES: TradesArchiveAdapter(),
        DatasetKind.BOOK_TICKER: BookTickerArchiveAdapter(),
        DatasetKind.BOOK_DEPTH: BookDepthArchiveAdapter(),
    }
    return adapters[dataset].canonical_contract()


def _real_canonical_partition_identity(record: Any) -> str | None:
    """Return the exact identity used by MarketDataStore canonical L1."""
    from .data.schemas import ArchiveRecord

    if not isinstance(record, ArchiveRecord):
        return None
    from .data.source_identity import canonical_partition_identity

    return canonical_partition_identity(
        record, dict(_canonical_contract_for_dataset(record.dataset))
    )


def _utc_iso(value: Any) -> str | None:
    """Serialize a source boundary as one host-independent UTC instant."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def source_record(record: Any) -> dict[str, Any]:
    value = {
        "exchange": record.exchange,
        "market": getattr(record.market, "value", record.market),
        "dataset": getattr(record.dataset, "value", record.dataset),
        "symbol": record.symbol,
        "interval": record.interval,
        "frequency": record.frequency,
        "period_start": _utc_iso(record.period_start),
        "period_end": _utc_iso(record.period_end),
        "size_bytes": int(record.size_bytes),
        "mtime_ns": int(record.mtime_ns),
        "raw_archive_fingerprint": record.fingerprint,
    }
    canonical_identity = _real_canonical_partition_identity(record)
    if canonical_identity is None:
        canonical_identity = canonical_sha256(
            {"provenance_test_double_v1": value}
        )
    value["canonical_partition_identity"] = canonical_identity
    return value


def selected_source_snapshot(
    records: tuple[Any, ...] | list[Any],
) -> tuple[list[dict[str, Any]], str]:
    values = [source_record(item) for item in records]
    unique = {canonical_json(item): item for item in values}
    rows = sorted(unique.values(), key=canonical_json)
    return rows, canonical_sha256(rows)


def load_completed_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunArtifactError("completed run manifest is missing or corrupt") from exc
    if (
        manifest.get("run_manifest_contract") != RUN_MANIFEST_CONTRACT
        or manifest.get("run_manifest_version") != RUN_MANIFEST_VERSION
        or manifest.get("run_status") != "COMPLETED"
    ):
        raise RunArtifactError("incompatible or incomplete run manifest")
    raw_type = manifest.get("run_type", RunType.BACKTEST.value)
    try:
        run_type = RunType(raw_type)
    except ValueError as exc:
        raise RunArtifactError(f"unknown run type: {raw_type}") from exc
    if run_type is RunType.BACKTEST:
        research = manifest.get("research")
        if not isinstance(research, Mapping):
            raise RunArtifactError("completed run research contract is missing")
        if (research.get("artifact_contract") != FEATURE_RESEARCH_ARTIFACT_CONTRACT
                or research.get("artifact_version") != FEATURE_RESEARCH_ARTIFACT_VERSION):
            raise RunArtifactError("incompatible feature research artifact version")
    else:
        scan = manifest.get("opportunity_scan")
        if not isinstance(scan, Mapping):
            raise RunArtifactError("completed opportunity scan contract is missing")
        if (scan.get("artifact_contract") != OPPORTUNITY_SCAN_ARTIFACT_CONTRACT
                or scan.get("artifact_version") != OPPORTUNITY_SCAN_ARTIFACT_VERSION):
            raise RunArtifactError("incompatible opportunity scan artifact version")
        required_payload = {
            "scan_timestamp", "decision_timestamp", "discovery_mode",
            "discovery_contract", "universe_definition", "counts",
            "source_identity_digest", "source_identities_artifact",
        }
        if not required_payload <= set(scan):
            raise RunArtifactError("opportunity scan payload is incomplete")
        artifacts = manifest.get("artifacts")
        required_artifacts = {
            "universe_snapshot", "discovery_rejections",
            "preliminary_candidates", "final_candidates",
            "final_candidate_exclusions", "rich_data_readiness",
            "opportunity_summary", "source_identities",
        }
        if not isinstance(artifacts, Mapping) or not required_artifacts <= set(artifacts):
            raise RunArtifactError("opportunity scan artifact catalog is incomplete")
        for name, entry in artifacts.items():
            if (not isinstance(entry, Mapping)
                    or not {"path", "format", "schema_version", "rows", "bytes", "sha256"} <= set(entry)
                    or not isinstance(entry.get("sha256"), str)
                    or len(entry["sha256"]) != 64):
                raise RunArtifactError(f"invalid opportunity scan artifact entry: {name}")
        if manifest.get("config", {}).get("scoring") is not None and "opportunity_scores" not in artifacts:
            raise RunArtifactError("scoring-enabled opportunity scan is missing scores")
    return manifest


def artifact_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    name: str,
    *,
    verify: bool = True,
) -> Path:
    entry = manifest.get("artifacts", {}).get(name)
    if not isinstance(entry, Mapping) or not entry.get("path"):
        raise RunArtifactError(f"artifact is not cataloged: {name}")
    relative = Path(entry["path"])
    root = Path(run_dir).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts:
        raise RunArtifactError("unsafe artifact catalog path")

    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RunArtifactError("artifact catalog path contains a symlink")
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RunArtifactError(f"artifact is missing: {name}") from exc
    if root not in path.parents or not path.is_file():
        raise RunArtifactError("artifact escapes completed run")
    if verify and file_sha256(path) != entry.get("sha256"):
        raise RunArtifactError(f"artifact integrity error: {name}")
    return path
