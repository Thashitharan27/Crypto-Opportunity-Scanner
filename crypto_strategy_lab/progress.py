"""Best-effort progress notifications for long-running research work.

Progress callbacks are deliberately observational only. They must never change
research semantics or make a run fail, and callers should emit only coarse stage
or partition-level events rather than hot-path row/trade events.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


ProgressCallback = Callable[[dict[str, Any]], None]


def emit_progress(callback: ProgressCallback | None, **event: Any) -> None:
    """Deliver one best-effort progress event without affecting the work itself."""
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        # UI/observer failures must never interrupt data preparation or simulation.
        return
