"""Append-only, secret-free JSONL audit reporting for paper scans."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Callable


class PaperScannerAuditLog:
    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def append(
        self,
        event_type: str,
        cycle_id: str,
        *,
        scan_run_id: str | None = None,
        symbol: str | None = None,
        signal_id: str | None = None,
        detail: str = "",
    ) -> None:
        row = {
            "timestamp": self.clock().astimezone(timezone.utc).isoformat(),
            "event_type": event_type,
            "cycle_id": cycle_id,
            "scan_run_id": scan_run_id,
            "symbol": symbol,
            "signal_id": signal_id,
            "detail": detail,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
