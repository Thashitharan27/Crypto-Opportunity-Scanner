"""Production-equivalent cached Bollinger, VWAP and MR-v2 context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.indicators import bollinger_bands, lag, rsi
from crypto_strategy_lab.mean_reversion import (
    classify_motion,
    classify_state,
    classify_strength,
    distance_from_mean_atr,
)
from crypto_strategy_lab.mean_reversion_v2 import (
    bb_zscore,
    bollinger_envelope,
    bollinger_reentry_flags,
    classify_bb_location,
    classify_rsi_state,
    classify_signal,
    moving_mean,
    signal_direction,
)

from .base import FeatureDefinition, ParameterDefinition
from .technical import CORE_DIRECTIONAL_FEATURE_NAME


PRODUCTION_CONTEXT_FEATURE_NAME = "production_market_context"
PRODUCTION_CONTEXT_FEATURE_VERSION = "2"


@dataclass(frozen=True, slots=True)
class ProductionContextFeatureProvider:
    """Prepare the exact stateless context used by EnhancedBacktestEngine."""

    definition: FeatureDefinition = FeatureDefinition(
        name=PRODUCTION_CONTEXT_FEATURE_NAME,
        version=PRODUCTION_CONTEXT_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        required_features=(CORE_DIRECTIONAL_FEATURE_NAME,),
        parameters={
            "bb_period": ParameterDefinition(int, 20),
            "bb_stddevs": ParameterDefinition(float, 2.0),
            "mean_reversion_period": ParameterDefinition(int, 20),
            "mean_reversion_mean_type": ParameterDefinition(lambda value: str(value).upper(), "SMA"),
            "mean_reversion_bb_stddevs": ParameterDefinition(float, 2.0),
            "mean_reversion_rsi_period": ParameterDefinition(int, 14),
            "mean_reversion_rsi_oversold": ParameterDefinition(float, 30.0),
            "mean_reversion_rsi_overbought": ParameterDefinition(float, 70.0),
            "mean_reversion_require_reentry": ParameterDefinition(bool, True),
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
            "mean_reversion_sigma",
            "mean_reversion_bb_upper",
            "mean_reversion_bb_lower",
            "mean_reversion_bb_zscore",
            "mean_reversion_bb_location",
            "mean_reversion_rsi",
            "mean_reversion_rsi_state",
            "mean_reversion_long_reentry",
            "mean_reversion_short_reentry",
            "mean_reversion_reentry_confirmation",
            "mean_reversion_signal",
            "mean_reversion_signal_direction",
            "mean_reversion_setup_strength",
            "bb_reentry",
            "mr_signal",
            "mr_signal_direction",
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
            raise ValueError("production_market_context requires canonical kline data") from exc
        if not feature_frames or CORE_DIRECTIONAL_FEATURE_NAME not in feature_frames:
            raise ValueError("production_market_context requires prepared core_directional features")
        directional = feature_frames[CORE_DIRECTIONAL_FEATURE_NAME].reset_index(drop=True)

        required = {"period_start", "available_at", "high", "low", "close", "volume"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")
        source = source.sort_values("period_start", kind="stable").drop_duplicates(
            "period_start", keep="last"
        ).reset_index(drop=True)
        if len(source) != len(directional):
            raise ValueError("production context dependency rows do not match kline rows")
        source_times = pd.to_datetime(source["period_start"], utc=True).reset_index(drop=True)
        dependency_times = pd.to_datetime(directional["timestamp"], utc=True).reset_index(drop=True)
        if not source_times.equals(dependency_times):
            raise ValueError("production context dependency timestamps do not match klines")

        bb_period = int(parameters.get("bb_period", 20))
        bb_stddevs = float(parameters.get("bb_stddevs", 2.0))
        mean_period = int(parameters.get("mean_reversion_period", 20))
        mean_type = str(parameters.get("mean_reversion_mean_type", "SMA")).upper()
        mr_stddevs = float(parameters.get("mean_reversion_bb_stddevs", 2.0))
        rsi_period = int(parameters.get("mean_reversion_rsi_period", 14))
        oversold = float(parameters.get("mean_reversion_rsi_oversold", 30.0))
        overbought = float(parameters.get("mean_reversion_rsi_overbought", 70.0))
        require_reentry = bool(parameters.get("mean_reversion_require_reentry", True))
        if bb_period <= 0 or bb_stddevs <= 0 or mean_period <= 0 or mr_stddevs <= 0 or rsi_period <= 0:
            raise ValueError("Production context periods/deviations must be positive")
        if mean_type not in {"SMA", "EMA"}:
            raise ValueError("mean_reversion_mean_type must be SMA or EMA")
        if not 0 <= oversold < overbought <= 100:
            raise ValueError("MR RSI thresholds must satisfy 0 <= oversold < overbought <= 100")

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

        mean = moving_mean(close, mean_period, mean_type)
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

        sigma, mr_upper, mr_lower = bollinger_envelope(close, mean, mean_period, mr_stddevs)
        zscore = bb_zscore(close, mean, sigma)
        mr_rsi = rsi(close, rsi_period)
        long_reentry, short_reentry = bollinger_reentry_flags(
            close,
            mr_lower,
            mr_upper,
            mr_rsi,
            oversold,
            overbought,
        )
        bb_location = np.array(
            [classify_bb_location(c, m, lo, hi) for c, m, lo, hi in zip(close, mean, mr_lower, mr_upper)],
            dtype=object,
        )
        rsi_state = np.array(
            [classify_rsi_state(value, oversold, overbought) for value in mr_rsi], dtype=object
        )
        signals = np.array(
            [
                classify_signal(c, lo, hi, rsi_value, oversold, overbought, long_flag, short_flag, require_reentry)
                for c, lo, hi, rsi_value, long_flag, short_flag in zip(
                    close, mr_lower, mr_upper, mr_rsi, long_reentry, short_reentry
                )
            ],
            dtype=object,
        )
        signal_directions = np.array([signal_direction(value) for value in signals], dtype=object)
        reentry_direction = np.full(len(close), "NONE", dtype=object)
        reentry_direction[short_reentry] = "SHORT"
        reentry_direction[long_reentry] = "LONG"
        setup_strength = np.array(
            [
                "STRONG" if str(value).startswith("STRONG_")
                else "POTENTIAL" if str(value).startswith("POTENTIAL_")
                else "NEUTRAL" if value == "NEUTRAL"
                else "UNKNOWN"
                for value in signals
            ],
            dtype=object,
        )
        confirmed_signal = np.where(reentry_direction != "NONE", "CONFIRMED", "NO_SIGNAL").astype(object)

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
                "mean_reversion_sigma": sigma,
                "mean_reversion_bb_upper": mr_upper,
                "mean_reversion_bb_lower": mr_lower,
                "mean_reversion_bb_zscore": zscore,
                "mean_reversion_bb_location": bb_location,
                "mean_reversion_rsi": mr_rsi,
                "mean_reversion_rsi_state": rsi_state,
                "mean_reversion_long_reentry": long_reentry,
                "mean_reversion_short_reentry": short_reentry,
                "mean_reversion_reentry_confirmation": reentry_direction,
                "mean_reversion_signal": signals,
                "mean_reversion_signal_direction": signal_directions,
                "mean_reversion_setup_strength": setup_strength,
                "bb_reentry": reentry_direction,
                "mr_signal": confirmed_signal,
                "mr_signal_direction": reentry_direction,
                "session_vwap": session_vwap,
                "close_location": close_location,
            }
        )
        if bool((output["available_at"] < source_times).any()):
            raise ValueError("Production context availability precedes its source candle")
        output.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                "bb_period": bb_period,
                "bb_stddevs": bb_stddevs,
                "mean_reversion_period": mean_period,
                "mean_reversion_mean_type": mean_type,
                "mean_reversion_bb_stddevs": mr_stddevs,
                "mean_reversion_rsi_period": rsi_period,
                "mean_reversion_rsi_oversold": oversold,
                "mean_reversion_rsi_overbought": overbought,
                "mean_reversion_require_reentry": require_reentry,
                "effective_warmup_bars": max(bb_period, mean_period, rsi_period, 6),
                "request_cache_key": request.cache_key(),
                "core_directional_cache_key": directional.attrs.get("feature_cache_key"),
            }
        )
        return output
