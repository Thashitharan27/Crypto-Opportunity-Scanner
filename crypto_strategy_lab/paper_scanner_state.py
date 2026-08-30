"""Versioned, atomic persistence for the paper-only scanner ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from crypto_strategy_lab.candidate_lifecycle import (
    CandidateLifecycleRecord,
    CandidateLifecyclePolicy,
    DEFAULT_LIFECYCLE_POLICY,
    LifecycleCursor,
)
from crypto_strategy_lab.run_manifest import atomic_json


STATE_VERSION = 2


class PaperScannerStateError(RuntimeError):
    """State cannot safely be used for duplicate protection."""


@dataclass(slots=True)
class PaperScannerState:
    emitted_signal_ids: list[str] = field(default_factory=list)
    paper_entries: list[dict[str, Any]] = field(default_factory=list)
    last_completed_cycle: dict[str, Any] | None = None
    last_successful_scan_run_id: str | None = None
    lifecycle_policy: CandidateLifecyclePolicy = DEFAULT_LIFECYCLE_POLICY
    candidate_lifecycle: list[CandidateLifecycleRecord] = field(default_factory=list)
    lifecycle_cursor: LifecycleCursor | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "emitted_signal_ids": self.emitted_signal_ids,
            "paper_entries": self.paper_entries,
            "last_completed_cycle": self.last_completed_cycle,
            "last_successful_scan_run_id": self.last_successful_scan_run_id,
            "lifecycle_policy": {
                "identity": self.lifecycle_policy.identity,
                "config": json.loads(self.lifecycle_policy.canonical_json()),
            },
            "candidate_lifecycle": [item.serializable() for item in self.candidate_lifecycle],
            "lifecycle_cursor": (
                self.lifecycle_cursor.serializable() if self.lifecycle_cursor else None
            ),
        }


class PaperScannerStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> PaperScannerState:
        if not self.path.exists():
            return PaperScannerState()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            version = value.get("version")
            if version not in (1, STATE_VERSION):
                raise PaperScannerStateError(
                    f"unsupported paper scanner state version: {value.get('version')!r}"
                )
            ids, entries = value["emitted_signal_ids"], value["paper_entries"]
            if not isinstance(ids, list) or not all(
                isinstance(item, str) for item in ids
            ):
                raise ValueError("emitted_signal_ids is invalid")
            if len(ids) != len(set(ids)) or not isinstance(entries, list):
                raise ValueError("paper scanner ledger is invalid")
            entry_ids: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("paper entry must be an object")
                if entry.get("record_type") != "PAPER_ENTRY":
                    raise ValueError("paper entry record_type is invalid")
                signal_id = entry.get("signal_id")
                signal_timestamp = entry.get("signal_candle_timestamp")
                if not isinstance(signal_id, str) or not signal_id:
                    raise ValueError("paper entry signal_id is invalid")
                if not isinstance(signal_timestamp, str) or not signal_timestamp:
                    raise ValueError("paper entry signal timestamp is invalid")
                try:
                    parsed_timestamp = datetime.fromisoformat(
                        signal_timestamp.replace("Z", "+00:00")
                    )
                except ValueError as exc:
                    raise ValueError("paper entry signal timestamp is invalid") from exc
                if (
                    parsed_timestamp.tzinfo is None
                    or parsed_timestamp.utcoffset() is None
                ):
                    raise ValueError(
                        "paper entry signal timestamp must be timezone-aware"
                    )
                entry_ids.append(signal_id)
            # The ledger and duplicate index are one atomic invariant.  Loading
            # mismatched collections could either suppress a signal without a
            # durable paper record or, worse, re-emit an existing paper entry.
            if entry_ids != ids:
                raise ValueError("paper entry ledger and duplicate index disagree")
            lifecycle_policy = DEFAULT_LIFECYCLE_POLICY
            lifecycle: list[CandidateLifecycleRecord] = []
            lifecycle_cursor: LifecycleCursor | None = None
            if version == STATE_VERSION:
                policy_value = value["lifecycle_policy"]
                lifecycle_policy = CandidateLifecyclePolicy(**policy_value["config"])
                lifecycle_policy.validate_executable()
                if policy_value["identity"] != lifecycle_policy.identity:
                    raise ValueError("lifecycle policy identity is invalid")
                lifecycle = [
                    CandidateLifecycleRecord.from_dict(item)
                    for item in value["candidate_lifecycle"]
                ]
                if len({item.symbol for item in lifecycle}) != len(lifecycle):
                    raise ValueError("candidate lifecycle symbols are not unique")
                cursor_value = value["lifecycle_cursor"]
                lifecycle_cursor = (
                    LifecycleCursor.from_dict(cursor_value)
                    if cursor_value is not None else None
                )
            # v1 migration intentionally starts with no inferred membership: v1
            # did not durably retain the last candidate snapshot.
            return PaperScannerState(
                ids,
                entries,
                value.get("last_completed_cycle"),
                value.get("last_successful_scan_run_id"),
                lifecycle_policy,
                lifecycle,
                lifecycle_cursor,
            )
        except PaperScannerStateError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PaperScannerStateError(
                f"paper scanner state is corrupt: {self.path}"
            ) from exc

    def save(self, state: PaperScannerState) -> None:
        # run_manifest.atomic_json is temp + fsync + os.replace, never in-place.
        atomic_json(self.path, state.serializable())
