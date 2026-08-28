"""Causal support/resistance context prepared before simulator execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.atr import atr
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.timing import interval_to_timedelta
from crypto_strategy_lab.higher_timeframe_sr import HigherTimeframeSRDetector, resample_ohlc_for_sr
from crypto_strategy_lab.support_resistance import (
    LocationClassification,
    SRContext,
    SupportResistanceDetector,
    TradeLocationRating,
)

from .base import FeatureDefinition, ParameterDefinition
from .technical import CORE_DIRECTIONAL_FEATURE_NAME


SUPPORT_RESISTANCE_FEATURE_NAME = "support_resistance"
SUPPORT_RESISTANCE_FEATURE_VERSION = "2"

_SR_FIELDS = (
    "nearest_support_price", "nearest_support_bar_index", "nearest_support_distance_atr",
    "nearest_support_distance_price", "nearest_resistance_price", "nearest_resistance_bar_index",
    "nearest_resistance_distance_atr", "nearest_resistance_distance_price", "price_location",
    "trade_location_rating", "near_support", "near_resistance", "inside_support_zone",
    "inside_resistance_zone", "room_in_direction_atr", "support_state", "resistance_state",
    "support_tested", "resistance_tested", "support_held", "resistance_held",
    "support_rejection_atr", "resistance_rejection_atr", "support_test_count",
    "resistance_test_count", "bars_since_support_test", "bars_since_resistance_test",
    "support_last_test_index", "resistance_last_test_index", "confirmation_rating",
    "support_zone_low", "support_zone_high", "resistance_zone_low", "resistance_zone_high",
)


def _primitive(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _flatten(prefix: str, context: SRContext) -> dict[str, object]:
    return {f"{prefix}_{field}": _primitive(getattr(context, field)) for field in _SR_FIELDS}


def _optional_float(value):
    return None if pd.isna(value) else float(value)


def _optional_int(value):
    return None if pd.isna(value) else int(value)


def _float_or_nan(value):
    return np.nan if pd.isna(value) else float(value)


@dataclass(frozen=True, slots=True)
class SupportResistanceFeatureProvider:
    """Prepare confirmed-pivot S/R context for both possible trade directions."""

    definition: FeatureDefinition = FeatureDefinition(
        name=SUPPORT_RESISTANCE_FEATURE_NAME,
        version=SUPPORT_RESISTANCE_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        required_features=(CORE_DIRECTIONAL_FEATURE_NAME,),
        parameters={
            "atr_period": ParameterDefinition(int, 14),
            "sr_timeframe_minutes": ParameterDefinition(int, 0),
            "sr_pivot_left": ParameterDefinition(int, 5),
            "sr_pivot_right": ParameterDefinition(int, 5),
            "sr_lookback_bars": ParameterDefinition(int, 200),
            "sr_zone_width_atr": ParameterDefinition(float, 0.5),
            "sr_near_distance_atr": ParameterDefinition(float, 0.75),
            "enable_sr_hold_confirmation": ParameterDefinition(bool, False),
            "sr_hold_confirmation_bars": ParameterDefinition(int, 3),
            "sr_hold_confirmation_atr": ParameterDefinition(float, 0.25),
            "sr_break_tolerance_atr": ParameterDefinition(float, 0.25),
            "sr_break_basis": ParameterDefinition(lambda value: str(value).upper(), "CLOSE"),
        },
        output_columns=(
            *tuple(
                f"{direction}_{field}"
                for direction in ("long", "short")
                for field in _SR_FIELDS
            ),
            "sr_completed_candle_time",
        ),
        warmup_bars=10,
        availability_rule="confirmed_pivots_through_latest_completed_configured_sr_candle",
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
            raise ValueError("support_resistance requires canonical kline data") from exc
        if not feature_frames or CORE_DIRECTIONAL_FEATURE_NAME not in feature_frames:
            raise ValueError("support_resistance requires prepared core_directional features")
        directional = feature_frames[CORE_DIRECTIONAL_FEATURE_NAME].reset_index(drop=True)

        required = {"period_start", "available_at", "open", "high", "low", "close"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")
        source = source.sort_values("period_start", kind="stable").drop_duplicates(
            "period_start", keep="last"
        ).reset_index(drop=True)
        if len(source) != len(directional):
            raise ValueError("S/R dependency rows do not match kline rows")
        source_times = pd.to_datetime(source["period_start"], utc=True).reset_index(drop=True)
        dependency_times = pd.to_datetime(directional["timestamp"], utc=True).reset_index(drop=True)
        if not source_times.equals(dependency_times):
            raise ValueError("S/R dependency timestamps do not match klines")

        detector_config = {
            "pivot_left": int(parameters.get("sr_pivot_left", 5)),
            "pivot_right": int(parameters.get("sr_pivot_right", 5)),
            "lookback_bars": int(parameters.get("sr_lookback_bars", 200)),
            "zone_width_atr": float(parameters.get("sr_zone_width_atr", 0.5)),
            "near_distance_atr": float(parameters.get("sr_near_distance_atr", 0.75)),
            "enable_hold_confirmation": bool(parameters.get("enable_sr_hold_confirmation", False)),
            "hold_confirmation_bars": int(parameters.get("sr_hold_confirmation_bars", 3)),
            "hold_confirmation_atr": float(parameters.get("sr_hold_confirmation_atr", 0.25)),
            "break_tolerance_atr": float(parameters.get("sr_break_tolerance_atr", 0.25)),
            "break_basis": str(parameters.get("sr_break_basis", "CLOSE")).upper(),
        }
        strategy_minutes = int(interval_to_timedelta(request.strategy_interval).total_seconds() // 60)
        configured_minutes = int(parameters.get("sr_timeframe_minutes", 0) or 0)
        effective_minutes = configured_minutes or strategy_minutes
        atr_period = int(parameters.get("atr_period", 14))
        if effective_minutes < strategy_minutes or effective_minutes % strategy_minutes:
            raise ValueError("S/R timeframe must be the strategy timeframe or an integer multiple")
        if atr_period <= 0:
            raise ValueError("S/R ATR period must be positive")

        open_ = pd.to_numeric(source["open"], errors="raise").to_numpy(float)
        high = pd.to_numeric(source["high"], errors="raise").to_numpy(float)
        low = pd.to_numeric(source["low"], errors="raise").to_numpy(float)
        close = pd.to_numeric(source["close"], errors="raise").to_numpy(float)
        atr_values = pd.to_numeric(directional["atr"], errors="raise").to_numpy(float)
        source_available = pd.to_datetime(source["available_at"], utc=True).reset_index(drop=True)
        dependency_available = pd.to_datetime(directional["available_at"], utc=True).reset_index(drop=True)
        available = pd.concat([source_available, dependency_available], axis=1).max(axis=1)

        rows: list[dict[str, object]] = []
        if effective_minutes == strategy_minutes:
            detector = SupportResistanceDetector(**detector_config)
            for i in range(len(source)):
                long_context = detector.analyze_price_location(
                    i, open_, high, low, close, atr_values, "LONG"
                )
                short_context = detector.analyze_price_location(
                    i, open_, high, low, close, atr_values, "SHORT"
                )
                row = {
                    "timestamp": source_times.iloc[i],
                    "available_at": available.iloc[i],
                    "sr_completed_candle_time": available.iloc[i],
                }
                row.update(_flatten("long", long_context))
                row.update(_flatten("short", short_context))
                rows.append(row)
        else:
            legacy = pd.DataFrame(
                {
                    "timestamp": source_times,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )
            htf = resample_ohlc_for_sr(legacy, strategy_minutes, effective_minutes)
            htf_open = htf["open"].to_numpy(float)
            htf_high = htf["high"].to_numpy(float)
            htf_low = htf["low"].to_numpy(float)
            htf_close = htf["close"].to_numpy(float)
            htf_atr = atr(htf_high, htf_low, htf_close, atr_period)
            htf_end = pd.to_datetime(htf["end_time"], utc=True)
            htf_end_ns = htf_end.to_numpy(dtype="datetime64[ns]")
            detector = HigherTimeframeSRDetector(**detector_config)
            for i in range(len(source)):
                available_i = pd.Timestamp(available.iloc[i])
                needle = np.datetime64(available_i.tz_convert("UTC").tz_localize(None).to_datetime64(), "ns")
                htf_i = int(np.searchsorted(htf_end_ns, needle, side="right") - 1)
                if htf_i < 0:
                    long_context = detector._default_context()
                    short_context = detector._default_context()
                    completed = pd.NaT
                else:
                    long_context = detector.analyze_external_price(
                        htf_i, htf_open, htf_high, htf_low, htf_close, htf_atr,
                        "LONG", float(close[i]),
                    )
                    short_context = detector.analyze_external_price(
                        htf_i, htf_open, htf_high, htf_low, htf_close, htf_atr,
                        "SHORT", float(close[i]),
                    )
                    completed = htf_end.iloc[htf_i]
                row = {
                    "timestamp": source_times.iloc[i],
                    "available_at": available_i,
                    "sr_completed_candle_time": completed,
                }
                row.update(_flatten("long", long_context))
                row.update(_flatten("short", short_context))
                rows.append(row)

        output = pd.DataFrame(rows)
        output["available_at"] = pd.to_datetime(output["available_at"], utc=True)
        output["sr_completed_candle_time"] = pd.to_datetime(
            output["sr_completed_candle_time"], utc=True, errors="coerce"
        )
        if bool((output["available_at"] < source_times).any()):
            raise ValueError("S/R feature availability precedes its source candle")
        completed = output["sr_completed_candle_time"].dropna()
        if not completed.empty:
            aligned_available = output.loc[completed.index, "available_at"]
            if bool((completed > aligned_available).any()):
                raise ValueError("S/R context uses a higher-timeframe candle not yet completed")
        output.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                **{f"parameter_{key}": value for key, value in detector_config.items()},
                "parameter_sr_timeframe_minutes": effective_minutes,
                "parameter_atr_period": atr_period,
                "effective_warmup_bars": max(
                    detector_config["pivot_left"] + detector_config["pivot_right"] + 1,
                    detector_config["hold_confirmation_bars"] + 1,
                ),
                "request_cache_key": request.cache_key(),
                "core_directional_cache_key": directional.attrs.get("feature_cache_key"),
            }
        )
        return output


class PreparedSupportResistanceContextReader:
    """O(1) adapter exposing cached S/R rows through the legacy detector API."""

    def __init__(self, frame) -> None:
        self.frame = frame.reset_index(drop=True) if isinstance(frame, pd.DataFrame) else None
        self.arrays = None if self.frame is not None else frame

    @staticmethod
    def _context_from_row(row, prefix: str) -> SRContext:
        def value(field: str):
            return row[f"{prefix}_{field}"]

        return SRContext(
            nearest_support_price=_optional_float(value("nearest_support_price")),
            nearest_support_bar_index=_optional_int(value("nearest_support_bar_index")),
            nearest_support_distance_atr=_float_or_nan(value("nearest_support_distance_atr")),
            nearest_support_distance_price=_float_or_nan(value("nearest_support_distance_price")),
            nearest_resistance_price=_optional_float(value("nearest_resistance_price")),
            nearest_resistance_bar_index=_optional_int(value("nearest_resistance_bar_index")),
            nearest_resistance_distance_atr=_float_or_nan(value("nearest_resistance_distance_atr")),
            nearest_resistance_distance_price=_float_or_nan(value("nearest_resistance_distance_price")),
            price_location=LocationClassification(str(value("price_location"))),
            trade_location_rating=TradeLocationRating(str(value("trade_location_rating"))),
            near_support=bool(value("near_support")),
            near_resistance=bool(value("near_resistance")),
            inside_support_zone=bool(value("inside_support_zone")),
            inside_resistance_zone=bool(value("inside_resistance_zone")),
            room_in_direction_atr=_float_or_nan(value("room_in_direction_atr")),
            support_state=str(value("support_state")),
            resistance_state=str(value("resistance_state")),
            support_tested=bool(value("support_tested")),
            resistance_tested=bool(value("resistance_tested")),
            support_held=bool(value("support_held")),
            resistance_held=bool(value("resistance_held")),
            support_rejection_atr=_float_or_nan(value("support_rejection_atr")),
            resistance_rejection_atr=_float_or_nan(value("resistance_rejection_atr")),
            support_test_count=int(value("support_test_count")),
            resistance_test_count=int(value("resistance_test_count")),
            bars_since_support_test=_optional_int(value("bars_since_support_test")),
            bars_since_resistance_test=_optional_int(value("bars_since_resistance_test")),
            support_last_test_index=_optional_int(value("support_last_test_index")),
            resistance_last_test_index=_optional_int(value("resistance_last_test_index")),
            confirmation_rating=str(value("confirmation_rating")),
            support_zone_low=_optional_float(value("support_zone_low")),
            support_zone_high=_optional_float(value("support_zone_high")),
            resistance_zone_low=_optional_float(value("resistance_zone_low")),
            resistance_zone_high=_optional_float(value("resistance_zone_high")),
        )

    def analyze_price_location(
        self,
        index: int,
        _open_prices,
        _high_prices,
        _low_prices,
        _close_prices,
        _atr_values,
        direction: str,
    ) -> SRContext:
        length = len(self.frame) if self.frame is not None else len(next(iter(self.arrays.values())))
        if not 0 <= int(index) < length:
            raise IndexError(f"Prepared S/R index out of range: {index}")
        normalized = str(direction).upper()
        if normalized not in {"LONG", "SHORT"}:
            raise ValueError(f"Unsupported S/R direction: {direction}")
        prefix = normalized.lower()
        row = self.frame.iloc[int(index)] if self.frame is not None else {
            name: values[int(index)] for name, values in self.arrays.items()
        }
        return self._context_from_row(row, prefix)


SR_CONTEXT_FIELDS = _SR_FIELDS
