"""Causal Bollinger + RSI mean-reversion research helpers.

The model is analysis-only.  It records statistical stretch, momentum state,
and optional band re-entry confirmation without accepting, rejecting, flipping,
or resizing any trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_strategy_lab.mean_reversion import ema


def moving_mean(values, period: int, mean_type: str = "SMA") -> np.ndarray:
    """Return a causal SMA or EMA with a full-period warm-up."""
    if period <= 0:
        raise ValueError("mean reversion period must be positive")
    kind = str(mean_type).upper()
    values = np.asarray(values, dtype=float)
    if kind == "EMA":
        return ema(values, period)
    if kind != "SMA":
        raise ValueError("mean_type must be SMA or EMA")
    return pd.Series(values, dtype="float64").rolling(period, min_periods=period).mean().to_numpy(float)


def bollinger_envelope(values, mean, period: int, stddevs: float = 2.0):
    """Return rolling population sigma plus upper/lower bands around ``mean``."""
    if period <= 0 or stddevs <= 0:
        raise ValueError("Bollinger settings must be positive")
    values = np.asarray(values, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = pd.Series(values, dtype="float64").rolling(period, min_periods=period).std(ddof=0).to_numpy(float)
    upper = mean + float(stddevs) * sigma
    lower = mean - float(stddevs) * sigma
    return sigma, upper, lower


def bb_zscore(close, mean, sigma) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.divide(
        close - mean,
        sigma,
        out=np.full(len(close), np.nan, dtype=float),
        where=np.isfinite(close) & np.isfinite(mean) & np.isfinite(sigma) & (sigma > 0),
    )


def classify_bb_location(close: float, mean: float, lower: float, upper: float) -> str:
    if not all(np.isfinite(v) for v in (close, mean, lower, upper)):
        return "UNKNOWN"
    if close < lower:
        return "BELOW_LOWER_BAND"
    if close > upper:
        return "ABOVE_UPPER_BAND"
    if close < mean:
        return "LOWER_HALF"
    if close > mean:
        return "UPPER_HALF"
    return "AT_MEAN"


def classify_rsi_state(value: float, oversold: float, overbought: float) -> str:
    if not np.isfinite(value):
        return "UNKNOWN"
    if value <= oversold:
        return "OVERSOLD"
    if value >= overbought:
        return "OVERBOUGHT"
    return "NEUTRAL"


def bollinger_reentry_flags(close, lower, upper, rsi_values, oversold: float, overbought: float):
    """Confirm the first close back inside a band after an RSI-confirmed excursion.

    An excursion stays armed while price remains outside the same band, allowing a
    multi-candle stretch to be confirmed on the eventual re-entry candle.  Every
    decision uses only current and earlier completed candles.
    """
    close = np.asarray(close, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rsi_values = np.asarray(rsi_values, dtype=float)
    long_reentry = np.zeros(len(close), dtype=bool)
    short_reentry = np.zeros(len(close), dtype=bool)
    long_armed = False
    short_armed = False

    for i in range(len(close)):
        if not all(np.isfinite(v) for v in (close[i], lower[i], upper[i], rsi_values[i])):
            continue

        if close[i] < lower[i]:
            if rsi_values[i] <= oversold:
                long_armed = True
        elif long_armed:
            long_reentry[i] = True
            long_armed = False

        if close[i] > upper[i]:
            if rsi_values[i] >= overbought:
                short_armed = True
        elif short_armed:
            short_reentry[i] = True
            short_armed = False

        # Crossing to the opposite extreme invalidates a stale unconfirmed setup.
        if close[i] > upper[i]:
            long_armed = False
        if close[i] < lower[i]:
            short_armed = False

    return long_reentry, short_reentry


def classify_signal(
    close: float,
    lower: float,
    upper: float,
    rsi_value: float,
    oversold: float,
    overbought: float,
    long_reentry: bool,
    short_reentry: bool,
    require_reentry: bool = True,
) -> str:
    """Classify a research state without changing the trading decision."""
    if not all(np.isfinite(v) for v in (close, lower, upper, rsi_value)):
        return "UNKNOWN"

    potential_long = close < lower and rsi_value <= oversold
    potential_short = close > upper and rsi_value >= overbought

    if require_reentry:
        if long_reentry:
            return "STRONG_LONG"
        if short_reentry:
            return "STRONG_SHORT"
        if potential_long:
            return "POTENTIAL_LONG"
        if potential_short:
            return "POTENTIAL_SHORT"
        return "NEUTRAL"

    if potential_long or long_reentry:
        return "STRONG_LONG"
    if potential_short or short_reentry:
        return "STRONG_SHORT"
    return "NEUTRAL"


def signal_direction(signal: str) -> str:
    signal = str(signal).upper()
    if signal.endswith("_LONG"):
        return "LONG"
    if signal.endswith("_SHORT"):
        return "SHORT"
    return "NONE"


def signal_alignment(signal: str, direction: str | None) -> str:
    expected = signal_direction(signal)
    if expected == "NONE":
        return "NEUTRAL"
    if direction not in ("LONG", "SHORT"):
        return "UNKNOWN"
    return "FAVORS_REVERSION" if direction == expected else "AGAINST_REVERSION"
