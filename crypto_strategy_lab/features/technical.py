"""Causal core technical features prepared before simulator execution.

This module is the forward Data Lake path for indicators that historically lived
inside :class:`BacktestEngine`.  Every output row depends only on the current or
an earlier completed strategy candle.  ``available_at`` is therefore inherited
from the current kline's availability timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.adx import adx
from crypto_strategy_lab.atr import atr
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.indicators import lag

from .base import FeatureDefinition, ParameterDefinition


CORE_DIRECTIONAL_FEATURE_NAME = "core_directional"
CORE_DIRECTIONAL_FEATURE_VERSION = "1"


def _pressure_state(directional_change: np.ndarray, opposing_change: np.ndarray) -> np.ndarray:
    state = np.full(len(directional_change), "UNKNOWN", dtype=object)
    finite = np.isfinite(directional_change) & np.isfinite(opposing_change)
    state[finite] = "MIXED"
    state[finite & (directional_change > 0) & (opposing_change < 0)] = "EXPANDING"
    state[finite & (directional_change < 0) & (opposing_change > 0)] = "CONTRACTING"
    return state


@dataclass(frozen=True, slots=True)
class CoreDirectionalFeatureProvider:
    """Prepare ATR, ADX/DMI and DI-pressure inputs from completed klines."""

    definition: FeatureDefinition = FeatureDefinition(
        name=CORE_DIRECTIONAL_FEATURE_NAME,
        version=CORE_DIRECTIONAL_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        parameters={
            "atr_period": ParameterDefinition(int, 14),
            "adx_period": ParameterDefinition(int, 14),
            "di_pressure_lookback": ParameterDefinition(int, 3),
        },
        output_columns=(
            "atr",
            "atr_pct",
            "adx",
            "plus_di",
            "minus_di",
            "di_spread",
            "di_spread_1",
            "di_spread_3",
            "di_spread_5",
            "di_spread_change",
            "di_ratio",
            "plus_di_change",
            "minus_di_change",
            "di_pressure_spread_change",
            "long_directional_di_change",
            "long_opposing_di_change",
            "long_di_pressure_state",
            "short_directional_di_change",
            "short_opposing_di_change",
            "short_di_pressure_state",
        ),
        warmup_bars=30,
        availability_rule="current_completed_kline_available_at",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[DatasetKind, pd.DataFrame],
        parameters: Mapping[str, object],
    ) -> pd.DataFrame:
        try:
            source = datasets[DatasetKind.KLINES].copy()
        except KeyError as exc:
            raise ValueError("core_directional requires canonical kline data") from exc

        required = {"period_start", "available_at", "high", "low", "close"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")
        if source.empty:
            raise ValueError("Cannot calculate technical features from an empty kline frame")

        atr_period = int(parameters.get("atr_period", 14))
        adx_period = int(parameters.get("adx_period", 14))
        pressure_lookback = int(parameters.get("di_pressure_lookback", 3))
        if atr_period <= 0 or adx_period <= 0 or pressure_lookback <= 0:
            raise ValueError("ATR period, ADX period and DI pressure lookback must be positive")

        source = source.sort_values("period_start", kind="stable").drop_duplicates(
            "period_start", keep="last"
        ).reset_index(drop=True)
        high = pd.to_numeric(source["high"], errors="raise").to_numpy(float)
        low = pd.to_numeric(source["low"], errors="raise").to_numpy(float)
        close = pd.to_numeric(source["close"], errors="raise").to_numpy(float)

        atr_values = atr(high, low, close, atr_period)
        adx_values, plus_di, minus_di = adx(high, low, close, adx_period)
        spread = np.abs(plus_di - minus_di)
        spread_1 = lag(spread, 1)
        spread_3 = lag(spread, 3)
        spread_5 = lag(spread, 5)
        spread_change = spread - spread_5
        maximum = np.maximum(plus_di, minus_di)
        minimum = np.minimum(plus_di, minus_di)
        ratio = np.divide(
            maximum,
            minimum,
            out=np.full(len(maximum), np.nan, dtype=float),
            where=np.isfinite(minimum) & (minimum != 0),
        )
        atr_pct = np.divide(
            atr_values,
            close,
            out=np.full(len(close), np.nan, dtype=float),
            where=np.isfinite(atr_values) & (close != 0),
        )

        old_plus = lag(plus_di, pressure_lookback)
        old_minus = lag(minus_di, pressure_lookback)
        plus_change = plus_di - old_plus
        minus_change = minus_di - old_minus
        pressure_spread_change = spread - lag(spread, pressure_lookback)

        long_directional_change = plus_change
        long_opposing_change = minus_change
        short_directional_change = minus_change
        short_opposing_change = plus_change

        output = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(source["period_start"], utc=True),
                "available_at": pd.to_datetime(source["available_at"], utc=True),
                "atr": atr_values,
                "atr_pct": atr_pct,
                "adx": adx_values,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "di_spread": spread,
                "di_spread_1": spread_1,
                "di_spread_3": spread_3,
                "di_spread_5": spread_5,
                "di_spread_change": spread_change,
                "di_ratio": ratio,
                "plus_di_change": plus_change,
                "minus_di_change": minus_change,
                "di_pressure_spread_change": pressure_spread_change,
                "long_directional_di_change": long_directional_change,
                "long_opposing_di_change": long_opposing_change,
                "long_di_pressure_state": _pressure_state(
                    long_directional_change, long_opposing_change
                ),
                "short_directional_di_change": short_directional_change,
                "short_opposing_di_change": short_opposing_change,
                "short_di_pressure_state": _pressure_state(
                    short_directional_change, short_opposing_change
                ),
            }
        )

        # The feature row must never become available before the candle that
        # supplies its current value. Lagged inputs are all older than this row.
        period_start = pd.to_datetime(source["period_start"], utc=True)
        if bool((output["available_at"] < period_start).any()):
            raise ValueError("Technical feature availability precedes its source candle")

        output.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                "atr_period": atr_period,
                "adx_period": adx_period,
                "di_pressure_lookback": pressure_lookback,
                "effective_warmup_bars": max(
                    atr_period,
                    adx_period * 2,
                    pressure_lookback + 1,
                    6,
                ),
                "request_cache_key": request.cache_key(),
            }
        )
        return output


def prepare_core_directional_features(
    request: DataRequest,
    canonical_klines: pd.DataFrame,
    *,
    atr_period: int,
    adx_period: int,
    di_pressure_lookback: int,
) -> pd.DataFrame:
    """Convenience entry point used by the Data Lake backtest bundle."""

    provider = CoreDirectionalFeatureProvider()
    return provider.compute(
        request,
        {DatasetKind.KLINES: canonical_klines},
        {
            "atr_period": atr_period,
            "adx_period": adx_period,
            "di_pressure_lookback": di_pressure_lookback,
        },
    )
