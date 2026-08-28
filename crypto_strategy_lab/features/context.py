"""Causal volatility and mean-reversion context prepared before simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.indicators import bollinger_bands, lag
from crypto_strategy_lab.mean_reversion import (
    classify_motion,
    classify_state,
    classify_strength,
    distance_from_mean_atr,
    ema,
)

from .base import FeatureDefinition, ParameterDefinition
from .technical import CORE_DIRECTIONAL_FEATURE_NAME


MARKET_CONTEXT_FEATURE_NAME = "market_context"
MARKET_CONTEXT_FEATURE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class MarketContextFeatureProvider:
    """Prepare Bollinger/volatility and mean-reversion arrays once per data slice."""

    definition: FeatureDefinition = FeatureDefinition(
        name=MARKET_CONTEXT_FEATURE_NAME,
        version=MARKET_CONTEXT_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        required_features=(CORE_DIRECTIONAL_FEATURE_NAME,),
        parameters={
            "bb_period": ParameterDefinition(int, 20),
            "bb_stddevs": ParameterDefinition(float, 2.0),
            "mean_reversion_period": ParameterDefinition(int, 20),
        },
        output_columns=(
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_width_pct",
            "bb_width_1",
            "bb_width_3",
            "bb_width_5",
            "bb_width_change",
            "bb_width_change_pct",
            "mean_reversion_mean",
            "mean_reversion_distance_atr",
            "mean_reversion_distance_atr_previous",
            "mean_reversion_distance_change_atr",
            "mean_reversion_state",
            "mean_reversion_motion",
            "mean_reversion_strength",
            "mean_reversion_strength_label",
            "session_vwap",
            "close_location",
        ),
        warmup_bars=30,
        availability_rule="max_current_kline_and_core_directional_available_at",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[DatasetKind, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        try:
            source = datasets[DatasetKind.KLINES].copy()
        except KeyError as exc:
            raise ValueError("market_context requires canonical kline data") from exc
        if not feature_frames or CORE_DIRECTIONAL_FEATURE_NAME not in feature_frames:
            raise ValueError("market_context requires prepared core_directional features")
        directional = feature_frames[CORE_DIRECTIONAL_FEATURE_NAME].reset_index(drop=True)

        required = {"period_start", "available_at", "high", "low", "close", "volume"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")
        source = source.sort_values("period_start", kind="stable").drop_duplicates(
            "period_start", keep="last"
        ).reset_index(drop=True)
        if len(source) != len(directional):
            raise ValueError("market_context dependency rows do not match kline rows")

        source_times = pd.to_datetime(source["period_start"], utc=True).reset_index(drop=True)
        dependency_times = pd.to_datetime(directional["timestamp"], utc=True).reset_index(drop=True)
        if not source_times.equals(dependency_times):
            raise ValueError("market_context dependency timestamps do not match klines")

        bb_period = int(parameters.get("bb_period", 20))
        bb_stddevs = float(parameters.get("bb_stddevs", 2.0))
        mean_period = int(parameters.get("mean_reversion_period", 20))
        if bb_period <= 0 or bb_stddevs <= 0 or mean_period <= 0:
            raise ValueError("Bollinger and mean-reversion settings must be positive")

        high = pd.to_numeric(source["high"], errors="raise").to_numpy(float)
        low = pd.to_numeric(source["low"], errors="raise").to_numpy(float)
        close = pd.to_numeric(source["close"], errors="raise").to_numpy(float)
        volume = pd.to_numeric(source["volume"], errors="raise").to_numpy(float)
        atr_values = pd.to_numeric(directional["atr"], errors="raise").to_numpy(float)

        bb_middle, bb_upper, bb_lower, bb_width, bb_width_pct = bollinger_bands(
            close, bb_period, bb_stddevs
        )
        bb_width_1 = lag(bb_width, 1)
        bb_width_3 = lag(bb_width, 3)
        bb_width_5 = lag(bb_width, 5)
        bb_width_change = bb_width - bb_width_5
        bb_width_change_pct = np.divide(
            bb_width_change,
            bb_width_5,
            out=np.full(len(bb_width), np.nan, dtype=float),
            where=np.isfinite(bb_width_5) & (bb_width_5 != 0),
        )

        mean = ema(close, mean_period)
        distance = distance_from_mean_atr(close, mean, atr_values)
        previous_distance = lag(distance, 1)
        distance_change = distance - previous_distance
        mean_state = np.array([classify_state(value) for value in distance], dtype=object)
        mean_motion = np.array(
            [classify_motion(value, previous) for value, previous in zip(distance, previous_distance)],
            dtype=object,
        )
        strengths = [classify_strength(value) for value in distance]
        mean_strength = np.array([item[0] for item in strengths], dtype=int)
        mean_strength_label = np.array([item[1] for item in strengths], dtype=object)

        typical = (high + low + close) / 3.0
        sessions = source_times.dt.floor("D")
        cumulative_weighted = pd.Series(typical * volume).groupby(sessions).cumsum().to_numpy(float)
        cumulative_volume = pd.Series(volume).groupby(sessions).cumsum().to_numpy(float)
        session_vwap = np.divide(
            cumulative_weighted,
            cumulative_volume,
            out=np.full(len(close), np.nan, dtype=float),
            where=cumulative_volume > 0,
        )
        candle_range = high - low
        close_location = np.divide(
            close - low,
            candle_range,
            out=np.full(len(close), np.nan, dtype=float),
            where=np.isfinite(candle_range) & (candle_range != 0),
        )

        source_available = pd.to_datetime(source["available_at"], utc=True)
        dependency_available = pd.to_datetime(directional["available_at"], utc=True)
        available = pd.concat(
            [source_available.reset_index(drop=True), dependency_available.reset_index(drop=True)],
            axis=1,
        ).max(axis=1)

        output = pd.DataFrame(
            {
                "timestamp": source_times,
                "available_at": available,
                "bb_middle": bb_middle,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_width": bb_width,
                "bb_width_pct": bb_width_pct,
                "bb_width_1": bb_width_1,
                "bb_width_3": bb_width_3,
                "bb_width_5": bb_width_5,
                "bb_width_change": bb_width_change,
                "bb_width_change_pct": bb_width_change_pct,
                "mean_reversion_mean": mean,
                "mean_reversion_distance_atr": distance,
                "mean_reversion_distance_atr_previous": previous_distance,
                "mean_reversion_distance_change_atr": distance_change,
                "mean_reversion_state": mean_state,
                "mean_reversion_motion": mean_motion,
                "mean_reversion_strength": mean_strength,
                "mean_reversion_strength_label": mean_strength_label,
                "session_vwap": session_vwap,
                "close_location": close_location,
            }
        )
        if bool((output["available_at"] < source_times).any()):
            raise ValueError("Market-context feature availability precedes its source candle")
        output.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                "bb_period": bb_period,
                "bb_stddevs": bb_stddevs,
                "mean_reversion_period": mean_period,
                "effective_warmup_bars": max(bb_period, mean_period, 6),
                "request_cache_key": request.cache_key(),
                "core_directional_cache_key": directional.attrs.get("feature_cache_key"),
            }
        )
        return output
