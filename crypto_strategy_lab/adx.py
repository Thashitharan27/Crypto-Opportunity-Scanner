"""Wilder Average Directional Index indicator."""
from __future__ import annotations

import numpy as np


def wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    """TradingView-compatible Wilder RMA seeded with an SMA."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period <= 0:
        raise ValueError("period must be positive")
    finite = np.isfinite(values)
    for start in range(0, max(0, len(values) - period + 1)):
        window = values[start:start + period]
        if np.isfinite(window).all():
            seed = start + period - 1
            out[seed] = float(np.mean(window))
            for i in range(seed + 1, len(values)):
                out[i] = (out[i - 1] * (period - 1) + values[i]) / period if finite[i] else out[i - 1]
            break
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ADX, +DI, -DI) using Wilder's DMI/ADX calculation."""
    high = np.asarray(high, dtype=float); low = np.asarray(low, dtype=float); close = np.asarray(close, dtype=float)
    n = len(close)
    tr = np.full(n, np.nan); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    tr_rma = wilder_rma(tr, period)
    plus_rma = wilder_rma(plus_dm.astype(float), period)
    minus_rma = wilder_rma(minus_dm.astype(float), period)
    plus_di = np.divide(100 * plus_rma, tr_rma, out=np.full(n, np.nan), where=tr_rma != 0)
    minus_di = np.divide(100 * minus_rma, tr_rma, out=np.full(n, np.nan), where=tr_rma != 0)
    denom = plus_di + minus_di
    dx = np.divide(100 * np.abs(plus_di - minus_di), denom, out=np.full(n, np.nan), where=denom != 0)
    return wilder_rma(dx, period), plus_di, minus_di
