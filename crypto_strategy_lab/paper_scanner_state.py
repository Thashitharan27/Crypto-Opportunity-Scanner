"""Versioned, atomic persistence for the paper-only scanner ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from crypto_strategy_lab.run_manifest import atomic_json


STATE_VERSION = 1


class PaperScannerStateError(RuntimeError):
    """State cannot safely be used for duplicate protection."""


@dataclass(slots=True)
class PaperScannerState:
    emitted_signal_ids: list[str] = field(default_factory=list)
    paper_entries: list[dict[str, Any]] = field(default_factory=list)
    last_completed_cycle: dict[str, Any] | None = None
    last_successful_scan_run_id: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "emitted_signal_ids": self.emitted_signal_ids,
            "paper_entries": self.paper_entries,
            "last_completed_cycle": self.last_completed_cycle,
            "last_successful_scan_run_id": self.last_successful_scan_run_id,
        }


class PaperScannerStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> PaperScannerState:
        if not self.path.exists():
            return PaperScannerState()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("version") != STATE_VERSION:
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
                entry_ids.append(signal_id)
            # The ledger and duplicate index are one atomic invariant.  Loading
            # mismatched collections could either suppress a signal without a
            # durable paper record or, worse, re-emit an existing paper entry.
            if entry_ids != ids:
                raise ValueError("paper entry ledger and duplicate index disagree")
            return PaperScannerState(
                ids,
                entries,
                value.get("last_completed_cycle"),
                value.get("last_successful_scan_run_id"),
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
