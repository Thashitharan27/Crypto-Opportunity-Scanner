"""TradingView-style ATR using Wilder's RMA smoothing."""

from __future__ import annotations

import numpy as np


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.empty_like(close, dtype=float)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum.reduce((high - low, np.abs(high - prev_close), np.abs(low - prev_close)))


def rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder RMA seeded with the first full-period SMA, matching TradingView."""
    if period <= 0:
        raise ValueError("period must be positive")
    out = np.full(values.shape, np.nan, dtype=float)
    if len(values) < period:
        return out
    seed_values = values[:period]
    seed = float(np.mean(seed_values[~np.isnan(seed_values)])) if np.isfinite(seed_values).any() else np.nan
    out[period - 1] = seed
    alpha = 1.0 / period
    for i in range(period, len(values)):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    return rma(true_range(high.astype(float), low.astype(float), close.astype(float)), period)
