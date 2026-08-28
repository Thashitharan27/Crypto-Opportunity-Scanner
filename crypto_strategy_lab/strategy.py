"""Entry strategy extension points."""

from __future__ import annotations

import numpy as np


def custom_entry_signal(index: int, data: dict[str, np.ndarray], active_pairs: int) -> bool:
    """Override this function to add custom entries without changing the engine."""
    return False
