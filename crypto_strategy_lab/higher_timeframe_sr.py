"""Causal higher-timeframe support/resistance helpers.

Higher-timeframe candles are resampled from the strategy dataset. Only complete
higher-timeframe candles whose close time is known at the strategy entry are
made available to the S/R detector, so no future candle information leaks into
entry-time structure analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_strategy_lab.support_resistance import (
    LocationClassification,
    SRContext,
    SRInteractionState,
    SupportResistanceDetector,
)


def resample_ohlc_for_sr(data: pd.DataFrame, strategy_minutes: int, target_minutes: int) -> pd.DataFrame:
    """Aggregate strategy candles into complete, aligned higher-timeframe candles."""
    if target_minutes <= strategy_minutes:
        raise ValueError("target S/R timeframe must be higher than strategy timeframe")
    if target_minutes % strategy_minutes:
        raise ValueError("target S/R timeframe must be an integer multiple of strategy timeframe")

    frame = data[["timestamp", "open", "high", "low", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    factor = target_minutes // strategy_minutes
    rule = f"{target_minutes}min"
    indexed = frame.set_index("timestamp")
    grouped = indexed.resample(rule, label="left", closed="left", origin="epoch")
    result = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    result["source_bars"] = grouped["close"].count()
    result = result[result["source_bars"].eq(factor)].dropna(subset=["open", "high", "low", "close"]).reset_index()
    result["end_time"] = result["timestamp"] + pd.Timedelta(minutes=target_minutes)
    return result


class HigherTimeframeSRDetector(SupportResistanceDetector):
    """S/R detector that evaluates current strategy price against completed HTF structure."""

    def analyze_external_price(
        self,
        index: int,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        atr_values: np.ndarray,
        direction: str,
        evaluation_price: float,
    ) -> SRContext:
        direction = str(direction).upper()
        if index < self.swing_detector.pivot_left + self.swing_detector.pivot_right:
            return self._default_context()
        current_atr = float(atr_values[index])
        current_price = float(evaluation_price)
        if not np.isfinite(current_atr) or current_atr <= 0 or not np.isfinite(current_price):
            return self._default_context()

        # Advance structure and interaction state using only completed HTF bars.
        self._advance_to(index, open_prices, high_prices, low_prices, close_prices, atr_values)
        support_levels = self._find_support_levels(high_prices, low_prices, index, current_atr)
        resistance_levels = self._find_resistance_levels(high_prices, low_prices, index, current_atr)
        nearest_support = self._nearest_level(support_levels, current_price, below=True)
        nearest_resistance = self._nearest_level(resistance_levels, current_price, below=False)
        support_dist_price, support_dist_atr = self._calculate_distance(current_price, nearest_support, current_atr)
        resistance_dist_price, resistance_dist_atr = self._calculate_distance(current_price, nearest_resistance, current_atr)
        location = self._classify_location(nearest_support, nearest_resistance, support_dist_atr, resistance_dist_atr)
        rating = self._rate_location(location, direction)
        room = self._calculate_room_in_direction(nearest_support, nearest_resistance, current_price, direction, current_atr)

        support_metrics = self._interaction_metrics(nearest_support, index, True)
        resistance_metrics = self._interaction_metrics(nearest_resistance, index, False)
        if nearest_support is None:
            support_metrics = self._interaction_metrics_for_active_state(
                index, True, SRInteractionState.SUPPORT_BROKEN.value, current_atr
            )
        if nearest_resistance is None:
            resistance_metrics = self._interaction_metrics_for_active_state(
                index, False, SRInteractionState.RESISTANCE_BROKEN.value, current_atr
            )
        confirmation_rating = self._confirmation_rating(
            direction, support_metrics["state"], resistance_metrics["state"]
        )

        return SRContext(
            nearest_support_price=nearest_support.price if nearest_support else None,
            nearest_support_bar_index=nearest_support.bar_index if nearest_support else None,
            nearest_support_distance_atr=support_dist_atr,
            nearest_support_distance_price=support_dist_price,
            nearest_resistance_price=nearest_resistance.price if nearest_resistance else None,
            nearest_resistance_bar_index=nearest_resistance.bar_index if nearest_resistance else None,
            nearest_resistance_distance_atr=resistance_dist_atr,
            nearest_resistance_distance_price=resistance_dist_price,
            price_location=location if isinstance(location, LocationClassification) else LocationClassification.NO_STRUCTURE,
            trade_location_rating=rating,
            near_support=np.isfinite(support_dist_atr) and support_dist_atr <= self.near_distance_atr,
            near_resistance=np.isfinite(resistance_dist_atr) and resistance_dist_atr <= self.near_distance_atr,
            inside_support_zone=bool(nearest_support and nearest_support.zone_bottom <= current_price <= nearest_support.zone_top),
            inside_resistance_zone=bool(nearest_resistance and nearest_resistance.zone_bottom <= current_price <= nearest_resistance.zone_top),
            room_in_direction_atr=room,
            support_state=support_metrics["state"],
            resistance_state=resistance_metrics["state"],
            support_tested=support_metrics["tested"],
            resistance_tested=resistance_metrics["tested"],
            support_held=support_metrics["held"],
            resistance_held=resistance_metrics["held"],
            support_rejection_atr=support_metrics["rejection_atr"],
            resistance_rejection_atr=resistance_metrics["rejection_atr"],
            support_test_count=support_metrics["test_count"],
            resistance_test_count=resistance_metrics["test_count"],
            bars_since_support_test=support_metrics["bars_since_test"],
            bars_since_resistance_test=resistance_metrics["bars_since_test"],
            support_last_test_index=support_metrics["last_test_index"],
            resistance_last_test_index=resistance_metrics["last_test_index"],
            confirmation_rating=confirmation_rating,
            support_zone_low=nearest_support.zone_bottom if nearest_support else None,
            support_zone_high=nearest_support.zone_top if nearest_support else None,
            resistance_zone_low=nearest_resistance.zone_bottom if nearest_resistance else None,
            resistance_zone_high=nearest_resistance.zone_top if nearest_resistance else None,
        )
