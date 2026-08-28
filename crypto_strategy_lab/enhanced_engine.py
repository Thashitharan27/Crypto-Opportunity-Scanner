"""Backtest engine extensions for DI-pressure filters, MR telemetry, and higher-timeframe S/R."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_strategy_lab.atr import atr
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.higher_timeframe_sr import HigherTimeframeSRDetector, resample_ohlc_for_sr
from crypto_strategy_lab.indicators import lag, rsi
from crypto_strategy_lab.mean_reversion import distance_from_mean_atr
from crypto_strategy_lab.mean_reversion_v2 import (
    bb_zscore,
    bollinger_envelope,
    bollinger_reentry_flags,
    classify_bb_location,
    classify_rsi_state,
    classify_signal,
    moving_mean,
    signal_alignment,
    signal_direction,
)


class EnhancedBacktestEngine(BacktestEngine):
    """Add optional DI-pressure filtering plus enhanced research calculations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        mean_type = str(getattr(config, "mean_reversion_mean_type", "EMA")).upper()
        period = int(config.mean_reversion_period)
        stddevs = float(getattr(config, "mean_reversion_bb_stddevs", 2.0))
        rsi_period = int(getattr(config, "mean_reversion_rsi_period", 14))
        oversold = float(getattr(config, "mean_reversion_rsi_oversold", 30.0))
        overbought = float(getattr(config, "mean_reversion_rsi_overbought", 70.0))

        self.mean_reversion_mean = moving_mean(self.close, period, mean_type)
        self.mean_reversion_distance_atr = distance_from_mean_atr(
            self.close, self.mean_reversion_mean, self.atr_values
        )
        self.mean_reversion_distance_atr_previous = lag(self.mean_reversion_distance_atr, 1)
        self.mean_reversion_sigma, self.mean_reversion_bb_upper, self.mean_reversion_bb_lower = bollinger_envelope(
            self.close, self.mean_reversion_mean, period, stddevs
        )
        self.mean_reversion_bb_zscore = bb_zscore(
            self.close, self.mean_reversion_mean, self.mean_reversion_sigma
        )
        self.mean_reversion_rsi_values = rsi(self.close, rsi_period)
        self.mean_reversion_long_reentry, self.mean_reversion_short_reentry = bollinger_reentry_flags(
            self.close,
            self.mean_reversion_bb_lower,
            self.mean_reversion_bb_upper,
            self.mean_reversion_rsi_values,
            oversold,
            overbought,
        )

        # Optional S/R timeframe. 0 keeps legacy same-timeframe behavior.
        configured_sr_tf = int(getattr(config, "sr_timeframe_minutes", 0) or 0)
        self.sr_timeframe_minutes = configured_sr_tf or int(config.strategy_timeframe_minutes)
        self.sr_uses_higher_timeframe = bool(
            config.enable_support_resistance_analysis
            and self.sr_timeframe_minutes > int(config.strategy_timeframe_minutes)
        )
        self.sr_htf_frame = None
        self.sr_htf_end_times = np.array([], dtype="datetime64[ns]")
        if self.sr_uses_higher_timeframe:
            htf = resample_ohlc_for_sr(
                self.data,
                int(config.strategy_timeframe_minutes),
                self.sr_timeframe_minutes,
            )
            self.sr_htf_frame = htf
            self.sr_htf_open = htf["open"].to_numpy(float)
            self.sr_htf_high = htf["high"].to_numpy(float)
            self.sr_htf_low = htf["low"].to_numpy(float)
            self.sr_htf_close = htf["close"].to_numpy(float)
            self.sr_htf_atr = atr(self.sr_htf_high, self.sr_htf_low, self.sr_htf_close, config.atr_period)
            self.sr_htf_end_times = pd.to_datetime(htf["end_time"], utc=True).to_numpy(dtype="datetime64[ns]")
            self.sr_detector = HigherTimeframeSRDetector(
                pivot_left=config.sr_pivot_left,
                pivot_right=config.sr_pivot_right,
                lookback_bars=config.sr_lookback_bars,
                zone_width_atr=config.sr_zone_width_atr,
                near_distance_atr=config.sr_near_distance_atr,
                enable_hold_confirmation=config.enable_sr_hold_confirmation,
                hold_confirmation_bars=config.sr_hold_confirmation_bars,
                hold_confirmation_atr=config.sr_hold_confirmation_atr,
                break_tolerance_atr=config.sr_break_tolerance_atr,
                break_basis=config.sr_break_basis,
            )

    def _di_pressure_filter_result(self, i):
        """Return whether the classified DI-pressure state is allowed to enter."""
        if not self.config.enable_di_pressure_analysis:
            return True, None

        allowed = {
            "EXPANDING": bool(getattr(self.config, "di_pressure_allow_expanding", True)),
            "CONTRACTING": bool(getattr(self.config, "di_pressure_allow_contracting", True)),
            "MIXED": bool(getattr(self.config, "di_pressure_allow_mixed", True)),
        }
        # All three selected preserves the historical record-only behaviour exactly,
        # including warm-up rows that can still classify as UNKNOWN.
        if all(allowed.values()):
            return True, None

        direction = self._selected_direction(i)
        state = str(self._di_pressure_snapshot(i, direction).get("di_pressure_state", "UNKNOWN")).upper()
        if allowed.get(state, False):
            return True, None
        return False, f"DI_PRESSURE_{state}_FILTERED"

    def _entry_filter_result(self, i, execution_i=None):
        passed, reason = super()._entry_filter_result(i, execution_i)
        if not passed:
            return passed, reason
        pressure_passed, pressure_reason = self._di_pressure_filter_result(i)
        if not pressure_passed:
            return False, pressure_reason
        return True, reason

    def _latest_completed_sr_index(self, strategy_index: int) -> int:
        if not self.sr_uses_higher_timeframe or not len(self.sr_htf_end_times):
            return -1
        entry_time = pd.Timestamp(self._entry_time(strategy_index))
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")
        needle = np.datetime64(entry_time.tz_localize(None).to_datetime64(), "ns")
        return int(np.searchsorted(self.sr_htf_end_times, needle, side="right") - 1)

    def _analyze_support_resistance(self, i, direction):
        if not self.sr_uses_higher_timeframe:
            return super()._analyze_support_resistance(i, direction)
        if self.sr_detector is None:
            return None
        htf_i = self._latest_completed_sr_index(i)
        if htf_i < 0:
            return self.sr_detector._default_context()
        try:
            return self.sr_detector.analyze_external_price(
                htf_i,
                self.sr_htf_open,
                self.sr_htf_high,
                self.sr_htf_low,
                self.sr_htf_close,
                self.sr_htf_atr,
                direction,
                float(self.close[i]),
            )
        except Exception as exc:
            self.log(f"Higher-timeframe S/R analysis failed at strategy index {i}: {exc}")
            return None

    @staticmethod
    def _confirmed_alignment(reentry_direction: str, trade_direction: str | None) -> str:
        if reentry_direction not in ("LONG", "SHORT"):
            return "NO_SIGNAL"
        if trade_direction not in ("LONG", "SHORT"):
            return "UNKNOWN"
        return "AGREE" if trade_direction == reentry_direction else "DISAGREE"

    def _mean_reversion_snapshot(self, i, di_direction, trade_direction=None):
        result = super()._mean_reversion_snapshot(i, di_direction, trade_direction)
        config = self.config
        result.update(
            {
                "mean_reversion_mean_type": str(getattr(config, "mean_reversion_mean_type", "EMA")).upper(),
                "mean_reversion_bb_stddevs": float(getattr(config, "mean_reversion_bb_stddevs", 2.0)),
                "mean_reversion_rsi_period": int(getattr(config, "mean_reversion_rsi_period", 14)),
                "mean_reversion_rsi_oversold": float(getattr(config, "mean_reversion_rsi_oversold", 30.0)),
                "mean_reversion_rsi_overbought": float(getattr(config, "mean_reversion_rsi_overbought", 70.0)),
                "mean_reversion_require_reentry": bool(getattr(config, "mean_reversion_require_reentry", True)),
                "mean_reversion_track_atr_distance": bool(getattr(config, "mean_reversion_track_atr_distance", True)),
                "mean_reversion_track_motion": bool(getattr(config, "mean_reversion_track_motion", True)),
                "mean_reversion_bb_upper": np.nan,
                "mean_reversion_bb_lower": np.nan,
                "mean_reversion_bb_sigma": np.nan,
                "mean_reversion_bb_zscore": np.nan,
                "mean_reversion_bb_location": "UNKNOWN",
                "mean_reversion_rsi": np.nan,
                "mean_reversion_rsi_state": "UNKNOWN",
                "mean_reversion_long_reentry": False,
                "mean_reversion_short_reentry": False,
                "mean_reversion_reentry_confirmation": "NONE",
                "mean_reversion_signal": "UNKNOWN",
                "mean_reversion_signal_direction": "NONE",
                "mean_reversion_setup_strength": "UNKNOWN",
                "mean_reversion_distance_alignment": result.get("mean_reversion_alignment", "UNKNOWN"),
                "mean_reversion_signal_di_alignment": "UNKNOWN",
                "mean_reversion_signal_trade_alignment": "UNKNOWN",
                "bb_reentry": "NONE",
                "mr_signal": "NO_SIGNAL",
                "mr_signal_direction": "NONE",
                "mr_trade_alignment": "NO_SIGNAL",
            }
        )
        if not config.enable_mean_reversion_analysis:
            return result

        mean = float(self.mean_reversion_mean[i])
        upper = float(self.mean_reversion_bb_upper[i])
        lower = float(self.mean_reversion_bb_lower[i])
        sigma = float(self.mean_reversion_sigma[i])
        zscore = float(self.mean_reversion_bb_zscore[i])
        rsi_value = float(self.mean_reversion_rsi_values[i])
        close = float(self.close[i])
        long_reentry = bool(self.mean_reversion_long_reentry[i])
        short_reentry = bool(self.mean_reversion_short_reentry[i])
        oversold = float(getattr(config, "mean_reversion_rsi_oversold", 30.0))
        overbought = float(getattr(config, "mean_reversion_rsi_overbought", 70.0))
        require_reentry = bool(getattr(config, "mean_reversion_require_reentry", True))

        signal = classify_signal(
            close, lower, upper, rsi_value, oversold, overbought,
            long_reentry, short_reentry, require_reentry,
        )
        signal_dir = signal_direction(signal)
        di_alignment = signal_alignment(signal, di_direction)
        trade_alignment = signal_alignment(signal, trade_direction or di_direction)
        reentry_direction = "LONG" if long_reentry else ("SHORT" if short_reentry else "NONE")
        confirmed_signal = "CONFIRMED" if reentry_direction != "NONE" else "NO_SIGNAL"
        confirmed_trade_alignment = self._confirmed_alignment(reentry_direction, trade_direction or di_direction)

        result.update(
            {
                "mean_price": mean if np.isfinite(mean) else np.nan,
                "mean_reversion_bb_upper": upper if np.isfinite(upper) else np.nan,
                "mean_reversion_bb_lower": lower if np.isfinite(lower) else np.nan,
                "mean_reversion_bb_sigma": sigma if np.isfinite(sigma) else np.nan,
                "mean_reversion_bb_zscore": zscore if np.isfinite(zscore) else np.nan,
                "mean_reversion_bb_location": classify_bb_location(close, mean, lower, upper),
                "mean_reversion_rsi": rsi_value if np.isfinite(rsi_value) else np.nan,
                "mean_reversion_rsi_state": classify_rsi_state(rsi_value, oversold, overbought),
                "mean_reversion_long_reentry": long_reentry,
                "mean_reversion_short_reentry": short_reentry,
                "mean_reversion_reentry_confirmation": reentry_direction,
                "mean_reversion_signal": signal,
                "mean_reversion_signal_direction": signal_dir,
                "mean_reversion_setup_strength": "STRONG" if signal.startswith("STRONG_") else ("POTENTIAL" if signal.startswith("POTENTIAL_") else ("NEUTRAL" if signal == "NEUTRAL" else "UNKNOWN")),
                "mean_reversion_signal_di_alignment": di_alignment,
                "mean_reversion_signal_trade_alignment": trade_alignment,
                "bb_reentry": reentry_direction,
                "mr_signal": confirmed_signal,
                "mr_signal_direction": reentry_direction,
                "mr_trade_alignment": confirmed_trade_alignment,
                "mean_reversion_alignment": di_alignment,
                "mean_reversion_di_alignment": di_alignment,
                "mean_reversion_trade_alignment": trade_alignment,
            }
        )

        if not bool(getattr(config, "mean_reversion_track_atr_distance", True)):
            result.update(
                {
                    "mean_distance_atr": np.nan,
                    "mean_distance_atr_previous": np.nan,
                    "mean_distance_change_atr": np.nan,
                    "mean_reversion_state": "UNKNOWN",
                    "mean_reversion_strength": -1,
                    "mean_reversion_strength_label": "UNKNOWN",
                    "mean_reversion_distance_alignment": "UNKNOWN",
                }
            )
        if not bool(getattr(config, "mean_reversion_track_motion", True)):
            result["mean_distance_change_atr"] = np.nan
            result["mean_reversion_motion"] = "UNKNOWN"
        return result

    @staticmethod
    def _timeframe_label(minutes: int) -> str:
        if minutes == 1440:
            return "1d"
        if minutes >= 60 and minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"

    def _build_result_row(self, p, row_kind, positions):
        """Include enhanced MR and S/R source telemetry in trade_list.csv."""
        row = super()._build_result_row(p, row_kind, positions)
        fields = (
            "mean_reversion_mean_type", "mean_reversion_bb_stddevs", "mean_reversion_rsi_period",
            "mean_reversion_rsi_oversold", "mean_reversion_rsi_overbought", "mean_reversion_require_reentry",
            "mean_reversion_bb_upper", "mean_reversion_bb_lower", "mean_reversion_bb_sigma",
            "mean_reversion_bb_zscore", "mean_reversion_bb_location", "mean_reversion_rsi",
            "mean_reversion_rsi_state", "mean_reversion_long_reentry", "mean_reversion_short_reentry",
            "mean_reversion_reentry_confirmation", "mean_reversion_signal", "mean_reversion_signal_direction",
            "mean_reversion_setup_strength", "mean_reversion_distance_alignment", "mean_reversion_signal_di_alignment",
            "mean_reversion_signal_trade_alignment", "bb_reentry", "mr_signal", "mr_signal_direction", "mr_trade_alignment",
        )
        defaults = {
            "mean_reversion_bb_location": "UNKNOWN", "mean_reversion_rsi_state": "UNKNOWN",
            "mean_reversion_reentry_confirmation": "NONE", "mean_reversion_signal": "UNKNOWN",
            "mean_reversion_signal_direction": "NONE", "mean_reversion_setup_strength": "UNKNOWN",
            "mean_reversion_distance_alignment": "UNKNOWN", "mean_reversion_signal_di_alignment": "UNKNOWN",
            "mean_reversion_signal_trade_alignment": "UNKNOWN", "bb_reentry": "NONE",
            "mr_signal": "NO_SIGNAL", "mr_signal_direction": "NONE", "mr_trade_alignment": "NO_SIGNAL",
        }
        for field in fields:
            row[field] = getattr(p, field, defaults.get(field, np.nan))

        row["sr_timeframe_minutes"] = int(self.sr_timeframe_minutes)
        row["sr_timeframe"] = self._timeframe_label(int(self.sr_timeframe_minutes))
        row["sr_timeframe_source"] = "HIGHER_TIMEFRAME_RESAMPLED" if self.sr_uses_higher_timeframe else "STRATEGY_TIMEFRAME"
        if self.sr_uses_higher_timeframe:
            strategy_i = int(np.searchsorted(self.times, np.datetime64(pd.Timestamp(p.strategy_candle_open_time).to_datetime64()), side="right") - 1)
            htf_i = self._latest_completed_sr_index(max(0, strategy_i))
            row["sr_last_completed_candle_time"] = pd.Timestamp(self.sr_htf_end_times[htf_i], tz="UTC") if htf_i >= 0 else pd.NaT
        else:
            row["sr_last_completed_candle_time"] = pd.Timestamp(p.strategy_entry_time)
        return row