"""Execution adapter for Data Lake v2 prepared features.

The mature event-driven simulator remains responsible for orders, exits, risk
and portfolio state. Stateless market context is prepared once outside it and
injected here so repeated profile runs do not recalculate the same indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import crypto_strategy_lab.engine as engine_module
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.features.market_regime import structural_regime_values
from crypto_strategy_lab.mean_reversion import classify_alignment


_REQUIRED_TECHNICAL_COLUMNS = {
    "timestamp", "available_at", "atr", "atr_pct", "adx", "plus_di", "minus_di",
    "di_spread", "di_spread_1", "di_spread_3", "di_spread_5", "di_spread_change",
    "di_ratio", "plus_di_change", "minus_di_change", "di_pressure_spread_change",
    "long_directional_di_change", "long_opposing_di_change", "long_di_pressure_state",
    "short_directional_di_change", "short_opposing_di_change", "short_di_pressure_state",
}

_REQUIRED_CONTEXT_COLUMNS = {
    "timestamp", "available_at", "bb_middle", "bb_upper", "bb_lower", "bb_width",
    "bb_width_pct", "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
    "bb_width_change_pct", "mean_reversion_mean", "mean_reversion_distance_atr",
    "mean_reversion_distance_atr_previous", "mean_reversion_distance_change_atr",
    "mean_reversion_state", "mean_reversion_motion", "mean_reversion_strength",
    "mean_reversion_strength_label", "session_vwap", "close_location",
}


class DataLakeBacktestEngine(BacktestEngine):
    """BacktestEngine with Data-Lake-prepared stateless context."""

    def __init__(
        self,
        *args,
        structural_benchmark: pd.DataFrame | None = None,
        technical_features: pd.DataFrame | None = None,
        context_features: pd.DataFrame | None = None,
        **kwargs,
    ):
        self.structural_benchmark = structural_benchmark
        self.technical_features = technical_features
        self.context_features = context_features

        if technical_features is None:
            # Diagnostic fallback only. Forward Data Lake callers always supply
            # prepared directional features.
            super().__init__(*args, **kwargs)
            self.technical_feature_source = "legacy_engine_fallback"
            self.context_feature_source = "legacy_engine_fallback"
            return

        data = args[0] if args else kwargs.get("data")
        config = args[1] if len(args) > 1 else kwargs.get("config")
        if data is None or config is None:
            raise TypeError("DataLakeBacktestEngine requires strategy data and config")

        technical = self._validate_technical_features(data, config, technical_features)
        context = (
            self._validate_context_features(data, config, context_features)
            if context_features is not None
            else None
        )
        self.technical_features = technical
        self.context_features = context

        prepared_atr = technical["atr"].to_numpy(float)
        prepared_adx = technical["adx"].to_numpy(float)
        prepared_plus = technical["plus_di"].to_numpy(float)
        prepared_minus = technical["minus_di"].to_numpy(float)
        expected_atr_period = int(technical.attrs["atr_period"])
        expected_adx_period = int(technical.attrs["adx_period"])

        original_atr = engine_module.atr
        original_adx = engine_module.adx
        original_bb = engine_module.bollinger_bands
        original_ema = engine_module.ema
        original_distance = engine_module.distance_from_mean_atr

        def _atr_from_features(_high, _low, _close, period=14):
            if int(period) != expected_atr_period:
                raise ValueError(
                    f"Prepared ATR period is {expected_atr_period}, engine requested {period}"
                )
            return prepared_atr.copy()

        def _adx_from_features(_high, _low, _close, period=14):
            if int(period) != expected_adx_period:
                raise ValueError(
                    f"Prepared ADX period is {expected_adx_period}, engine requested {period}"
                )
            return prepared_adx.copy(), prepared_plus.copy(), prepared_minus.copy()

        engine_module.atr = _atr_from_features
        engine_module.adx = _adx_from_features

        if context is not None:
            expected_bb_period = int(context.attrs["bb_period"])
            expected_bb_stddevs = float(context.attrs["bb_stddevs"])
            expected_mean_period = int(context.attrs["mean_reversion_period"])
            prepared_bb = tuple(
                context[column].to_numpy(float)
                for column in ("bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_width_pct")
            )
            prepared_mean = context["mean_reversion_mean"].to_numpy(float)
            prepared_distance = context["mean_reversion_distance_atr"].to_numpy(float)

            def _bb_from_features(_close, period=20, stddevs=2.0):
                if int(period) != expected_bb_period or not np.isclose(float(stddevs), expected_bb_stddevs):
                    raise ValueError(
                        "Prepared Bollinger settings do not match engine request: "
                        f"prepared=({expected_bb_period}, {expected_bb_stddevs}), "
                        f"requested=({period}, {stddevs})"
                    )
                return tuple(values.copy() for values in prepared_bb)

            def _ema_from_features(_values, period):
                if int(period) != expected_mean_period:
                    raise ValueError(
                        f"Prepared mean-reversion period is {expected_mean_period}, engine requested {period}"
                    )
                return prepared_mean.copy()

            def _distance_from_features(_close, _mean, _atr):
                return prepared_distance.copy()

            engine_module.bollinger_bands = _bb_from_features
            engine_module.ema = _ema_from_features
            engine_module.distance_from_mean_atr = _distance_from_features

        try:
            super().__init__(*args, **kwargs)
        finally:
            engine_module.atr = original_atr
            engine_module.adx = original_adx
            engine_module.bollinger_bands = original_bb
            engine_module.ema = original_ema
            engine_module.distance_from_mean_atr = original_distance

        # Directional provider is the source of truth after construction.
        self.atr_values = prepared_atr
        self.adx_values = prepared_adx
        self.plus_di_values = prepared_plus
        self.minus_di_values = prepared_minus
        self.atr_pct_values = technical["atr_pct"].to_numpy(float)
        self.di_spread = technical["di_spread"].to_numpy(float)
        self.di_spread_1 = technical["di_spread_1"].to_numpy(float)
        self.di_spread_3 = technical["di_spread_3"].to_numpy(float)
        self.di_spread_5 = technical["di_spread_5"].to_numpy(float)
        self.di_spread_change = technical["di_spread_change"].to_numpy(float)
        self.di_ratio = technical["di_ratio"].to_numpy(float)
        self.technical_feature_source = (
            f"{technical.attrs.get('feature_name', 'core_directional')}@"
            f"{technical.attrs.get('feature_version', 'unknown')}"
        )

        if context is not None:
            self.bb_middle = context["bb_middle"].to_numpy(float)
            self.bb_upper = context["bb_upper"].to_numpy(float)
            self.bb_lower = context["bb_lower"].to_numpy(float)
            self.bb_width = context["bb_width"].to_numpy(float)
            self.bb_width_pct = context["bb_width_pct"].to_numpy(float)
            self.bb_width_1 = context["bb_width_1"].to_numpy(float)
            self.bb_width_3 = context["bb_width_3"].to_numpy(float)
            self.bb_width_5 = context["bb_width_5"].to_numpy(float)
            self.bb_width_change = context["bb_width_change"].to_numpy(float)
            self.bb_width_change_pct = context["bb_width_change_pct"].to_numpy(float)
            self.mean_reversion_mean = context["mean_reversion_mean"].to_numpy(float)
            self.mean_reversion_distance_atr = context["mean_reversion_distance_atr"].to_numpy(float)
            self.mean_reversion_distance_atr_previous = context[
                "mean_reversion_distance_atr_previous"
            ].to_numpy(float)
            self.session_vwap = context["session_vwap"].to_numpy(float)
            self.close_location_values = context["close_location"].to_numpy(float)
            self.context_feature_source = (
                f"{context.attrs.get('feature_name', 'market_context')}@"
                f"{context.attrs.get('feature_version', 'unknown')}"
            )
        else:
            self.context_feature_source = "legacy_engine_fallback"

    @staticmethod
    def _aligned_feature_frame(data, frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
        if len(data) != len(frame):
            raise ValueError(
                f"Prepared {label} rows do not match strategy rows: {len(frame)} != {len(data)}"
            )
        prepared = frame.reset_index(drop=True).copy()
        data_times = pd.to_datetime(data["timestamp"], utc=True).reset_index(drop=True)
        feature_times = pd.to_datetime(prepared["timestamp"], utc=True).reset_index(drop=True)
        if not data_times.equals(feature_times):
            raise ValueError(f"Prepared {label} timestamps do not match strategy candles")
        return prepared, data_times

    @classmethod
    def _validate_technical_features(cls, data, config, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(_REQUIRED_TECHNICAL_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Prepared technical feature frame is missing columns: {missing}")
        features, data_times = cls._aligned_feature_frame(data, frame, "technical feature")
        atr_period = int(features.attrs.get("atr_period", -1))
        adx_period = int(features.attrs.get("adx_period", -1))
        pressure_lookback = int(features.attrs.get("di_pressure_lookback", -1))
        if atr_period != int(config.atr_period):
            raise ValueError(f"Prepared ATR period {atr_period} != config {config.atr_period}")
        if adx_period != int(config.adx_period):
            raise ValueError(f"Prepared ADX period {adx_period} != config {config.adx_period}")
        if pressure_lookback != int(config.di_pressure_lookback):
            raise ValueError(
                f"Prepared DI pressure lookback {pressure_lookback} != config {config.di_pressure_lookback}"
            )
        cls._assert_causal_availability(features, data_times, config, "technical feature")
        return features

    @classmethod
    def _validate_context_features(cls, data, config, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(_REQUIRED_CONTEXT_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Prepared context feature frame is missing columns: {missing}")
        features, data_times = cls._aligned_feature_frame(data, frame, "market context")
        if int(features.attrs.get("bb_period", -1)) != int(config.bb_period):
            raise ValueError("Prepared Bollinger period does not match config")
        if not np.isclose(float(features.attrs.get("bb_stddevs", np.nan)), float(config.bb_stddevs)):
            raise ValueError("Prepared Bollinger deviations do not match config")
        if int(features.attrs.get("mean_reversion_period", -1)) != int(config.mean_reversion_period):
            raise ValueError("Prepared mean-reversion period does not match config")
        cls._assert_causal_availability(features, data_times, config, "market context")
        return features

    @staticmethod
    def _assert_causal_availability(features, data_times, config, label: str) -> None:
        available = pd.to_datetime(features["available_at"], utc=True)
        candle_end = data_times + pd.Timedelta(minutes=int(config.strategy_timeframe_minutes))
        if bool((available > candle_end).any()):
            raise ValueError(f"Prepared {label} uses data unavailable at candle completion")

    def _di_pressure_snapshot(self, i, direction):
        """Read precomputed causal DI-pressure telemetry for one signal candle."""
        if self.technical_features is None:
            return super()._di_pressure_snapshot(i, direction)
        row = self.technical_features.iloc[i]
        plus = float(row["plus_di"])
        minus = float(row["minus_di"])
        lookback = int(self.config.di_pressure_lookback)
        result = {
            "plus_di": plus, "minus_di": minus, "directional_di": np.nan,
            "opposing_di": np.nan, "plus_di_change": np.nan, "minus_di_change": np.nan,
            "directional_di_change": np.nan, "opposing_di_change": np.nan,
            "di_spread": float(row["di_spread"]), "di_spread_change": np.nan,
            "di_pressure_state": "UNKNOWN", "di_pressure_lookback": lookback,
        }
        if direction not in ("LONG", "SHORT") or not np.isfinite(plus) or not np.isfinite(minus):
            return result
        if direction == "LONG":
            result["directional_di"], result["opposing_di"] = plus, minus
            directional_change = float(row["long_directional_di_change"])
            opposing_change = float(row["long_opposing_di_change"])
            pressure_state = str(row["long_di_pressure_state"])
        else:
            result["directional_di"], result["opposing_di"] = minus, plus
            directional_change = float(row["short_directional_di_change"])
            opposing_change = float(row["short_opposing_di_change"])
            pressure_state = str(row["short_di_pressure_state"])
        if not self.config.enable_di_pressure_analysis:
            return result
        plus_change = float(row["plus_di_change"])
        minus_change = float(row["minus_di_change"])
        if not np.isfinite(plus_change) or not np.isfinite(minus_change):
            return result
        result.update(
            plus_di_change=plus_change,
            minus_di_change=minus_change,
            directional_di_change=directional_change,
            opposing_di_change=opposing_change,
            di_spread_change=float(row["di_pressure_spread_change"]),
            di_pressure_state=pressure_state,
        )
        return result

    def _mean_reversion_snapshot(self, i, di_direction, trade_direction=None):
        """Read prepared MR state/motion and calculate only direction-dependent alignment."""
        if self.context_features is None and hasattr(self, "prepared_frame"):
            result = {
                "mean_reversion_enabled": bool(self.config.enable_mean_reversion_analysis),
                "mean_reversion_period": int(self.config.mean_reversion_period),
                "mean_price": np.nan, "mean_distance_atr": np.nan,
                "mean_distance_atr_previous": np.nan, "mean_distance_change_atr": np.nan,
                "mean_reversion_state": "UNKNOWN", "mean_reversion_motion": "UNKNOWN",
                "mean_reversion_alignment": "UNKNOWN", "mean_reversion_di_alignment": "UNKNOWN",
                "mean_reversion_trade_alignment": "UNKNOWN", "mean_reversion_strength": -1,
                "mean_reversion_strength_label": "UNKNOWN",
            }
            if not self.config.enable_mean_reversion_analysis:
                return result
            distance = float(self.mean_reversion_distance_atr[i])
            result.update(
                mean_price=float(self.mean_reversion_mean[i]),
                mean_distance_atr=distance,
                mean_distance_atr_previous=float(self.mean_reversion_distance_atr_previous[i]),
                mean_distance_change_atr=float(self.mean_reversion_distance_change_atr[i]),
                mean_reversion_state=str(self.mean_reversion_state[i]),
                mean_reversion_motion=str(self.mean_reversion_motion[i]),
                mean_reversion_strength=int(self.mean_reversion_strength[i]),
                mean_reversion_strength_label=str(self.mean_reversion_strength_label[i]),
            )
            di_alignment = classify_alignment(distance, di_direction)
            trade_alignment = classify_alignment(distance, trade_direction or di_direction)
            result.update(mean_reversion_alignment=di_alignment,
                          mean_reversion_di_alignment=di_alignment,
                          mean_reversion_trade_alignment=trade_alignment)
            return result
        if self.context_features is None:
            return super()._mean_reversion_snapshot(i, di_direction, trade_direction)
        result = {
            "mean_reversion_enabled": bool(self.config.enable_mean_reversion_analysis),
            "mean_reversion_period": int(self.config.mean_reversion_period),
            "mean_price": np.nan, "mean_distance_atr": np.nan,
            "mean_distance_atr_previous": np.nan, "mean_distance_change_atr": np.nan,
            "mean_reversion_state": "UNKNOWN", "mean_reversion_motion": "UNKNOWN",
            "mean_reversion_alignment": "UNKNOWN", "mean_reversion_di_alignment": "UNKNOWN",
            "mean_reversion_trade_alignment": "UNKNOWN", "mean_reversion_strength": -1,
            "mean_reversion_strength_label": "UNKNOWN",
        }
        if not self.config.enable_mean_reversion_analysis:
            return result
        row = self.context_features.iloc[i]
        distance = float(row["mean_reversion_distance_atr"])
        previous = float(row["mean_reversion_distance_atr_previous"])
        result.update(
            mean_price=float(row["mean_reversion_mean"]),
            mean_distance_atr=distance,
            mean_distance_atr_previous=previous,
            mean_distance_change_atr=float(row["mean_reversion_distance_change_atr"]),
            mean_reversion_state=str(row["mean_reversion_state"]),
            mean_reversion_motion=str(row["mean_reversion_motion"]),
            mean_reversion_strength=int(row["mean_reversion_strength"]),
            mean_reversion_strength_label=str(row["mean_reversion_strength_label"]),
        )
        di_alignment = classify_alignment(distance, di_direction)
        trade_alignment = classify_alignment(distance, trade_direction or di_direction)
        result["mean_reversion_alignment"] = di_alignment
        result["mean_reversion_di_alignment"] = di_alignment
        result["mean_reversion_trade_alignment"] = trade_alignment
        return result

    def _market_regime_array(self):
        if self.config.market_regime_method == "ASSET_RETURN":
            threshold = abs(float(self.config.bull_regime_return_threshold))
            return np.array(
                [
                    None if not np.isfinite(value) else (
                        "BULL" if value >= threshold else (
                            "BEAR" if value <= -threshold else "SIDEWAYS"
                        )
                    )
                    for value in self.bull_regime_return_values
                ],
                dtype=object,
            )
        if self.structural_benchmark is None or self.structural_benchmark.empty:
            label = "Asset" if self.config.market_regime_method == "ASSET_STRUCTURAL" else "BTC"
            raise ValueError(f"{label} structural regime requires a prepared Data Lake benchmark frame")
        return structural_regime_values(
            self.times,
            self.structural_benchmark,
            sma_days=int(self.config.structural_regime_sma_days),
            slope_lookback_days=int(self.config.structural_regime_slope_lookback_days),
        )
