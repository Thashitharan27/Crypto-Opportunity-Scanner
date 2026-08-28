"""Analysis-only mean-reversion helpers.

Mean reversion is intentionally telemetry-only. It never accepts, rejects, or
changes a trade. The recent mean is an EMA, while distance is normalized by the
configured ATR so values remain comparable across symbols and timeframes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DI_PRESSURE_BUCKETS: tuple[tuple[float, float | None], ...] = (
    (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
    (30, 35), (35, 40), (40, 45), (45, 50), (50, None),
)


def ema(values, period: int) -> np.ndarray:
    """Return a causal EMA with warm-up matching the requested period."""
    if period <= 0:
        raise ValueError("mean reversion period must be positive")
    series = pd.Series(np.asarray(values, dtype=float))
    return series.ewm(span=period, adjust=False, min_periods=period).mean().to_numpy(float)


def distance_from_mean_atr(close, mean, atr) -> np.ndarray:
    """Signed close-minus-mean distance in ATR units."""
    close = np.asarray(close, dtype=float)
    mean = np.asarray(mean, dtype=float)
    atr = np.asarray(atr, dtype=float)
    return np.divide(
        close - mean,
        atr,
        out=np.full(len(close), np.nan, dtype=float),
        where=np.isfinite(close) & np.isfinite(mean) & np.isfinite(atr) & (atr > 0),
    )


def classify_state(distance: float) -> str:
    if not np.isfinite(distance):
        return "UNKNOWN"
    if distance <= -1.5:
        return "STRONGLY_BELOW_MEAN"
    if distance < -0.5:
        return "BELOW_MEAN"
    if distance < 0.5:
        return "NEAR_MEAN"
    if distance < 1.5:
        return "ABOVE_MEAN"
    return "STRONGLY_ABOVE_MEAN"


def classify_motion(distance: float, previous_distance: float) -> str:
    if not np.isfinite(distance) or not np.isfinite(previous_distance):
        return "UNKNOWN"
    current_abs = abs(distance)
    previous_abs = abs(previous_distance)
    if current_abs < previous_abs - 1e-12:
        return "TOWARD_MEAN"
    if current_abs > previous_abs + 1e-12:
        return "AWAY_FROM_MEAN"
    return "FLAT"


def classify_alignment(distance: float, direction: str | None) -> str:
    """Classify whether the direction is on the mean-reversion side of price.

    Motion is deliberately not folded into alignment. That lets reports compare
    a stretched price that is already reverting with one that is still extending.
    """
    if not np.isfinite(distance) or direction not in ("LONG", "SHORT"):
        return "UNKNOWN"
    if abs(distance) < 0.5:
        return "NEUTRAL"
    reversion_direction = "LONG" if distance < 0 else "SHORT"
    return "FAVORS_REVERSION" if direction == reversion_direction else "AGAINST_REVERSION"


def classify_strength(distance: float) -> tuple[int, str]:
    if not np.isfinite(distance):
        return -1, "UNKNOWN"
    stretch = abs(distance)
    if stretch < 0.5:
        return 0, "NEUTRAL"
    if stretch < 1.0:
        return 1, "WEAK"
    if stretch < 1.5:
        return 2, "MODERATE"
    return 3, "STRONG"


def di_pressure_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "UNKNOWN"
    for lo, hi in DI_PRESSURE_BUCKETS:
        if value >= lo and (hi is None or value < hi):
            return f"{lo:g}+" if hi is None else f"{lo:g}-{hi:g}"
    return "UNKNOWN"
