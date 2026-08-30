"""Event-driven dual long/short backtesting engine with 15m strategy and optional 1m exits."""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from crypto_strategy_lab.atr import atr
from crypto_strategy_lab.adx import adx
from crypto_strategy_lab.config import BacktestConfig, EntryMode, IntrabarMissingPolicy, RiskMode, TiePolicy, DailyEntryMissedPolicy
from crypto_strategy_lab.indicators import bollinger_bands, lag, rsi
from crypto_strategy_lab.mean_reversion import ema, distance_from_mean_atr, classify_state, classify_motion, classify_alignment, classify_strength
from crypto_strategy_lab.strategy_profiles import profile_key
from crypto_strategy_lab.support_resistance import SupportResistanceDetector
from crypto_strategy_lab.trade import ExitReason, ExitSource, Position, Side, TradePair

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, config: BacktestConfig, intrabar_data: pd.DataFrame | None = None, progress_callback: Callable[[int, int, int, int], None] | None = None, progress_interval: int = 50):
        self.data=data.reset_index(drop=True); self.intrabar_data=intrabar_data.reset_index(drop=True) if intrabar_data is not None else None; self.config=config; self.progress_callback=progress_callback; self.progress_interval=max(1, int(progress_interval))
        self.high=self.data.high.to_numpy(float); self.low=self.data.low.to_numpy(float); self.close=self.data.close.to_numpy(float); self.open=self.data.open.to_numpy(float); self.volume=self.data["volume"].to_numpy(float) if "volume" in self.data else np.ones(len(self.data),float); self.times=self.data.timestamp.to_numpy()
        self.atr_values=atr(self.high,self.low,self.close,self.config.atr_period); self.adx_values,self.plus_di_values,self.minus_di_values=adx(self.high,self.low,self.close,self.config.adx_period); self.bb_middle,self.bb_upper,self.bb_lower,self.bb_width,self.bb_width_pct=bollinger_bands(self.close,self.config.bb_period,self.config.bb_stddevs); self.bb_width_1=lag(self.bb_width,1); self.bb_width_3=lag(self.bb_width,3); self.bb_width_5=lag(self.bb_width,5); self.bb_width_change=self.bb_width-self.bb_width_5; self.bb_width_change_pct=np.divide(self.bb_width_change,self.bb_width_5,out=np.full(len(self.bb_width),np.nan,float),where=np.isfinite(self.bb_width_5)&(self.bb_width_5!=0)); self.di_spread=np.abs(self.plus_di_values-self.minus_di_values); self.di_spread_1=lag(self.di_spread,1); self.di_spread_3=lag(self.di_spread,3); self.di_spread_5=lag(self.di_spread,5); self.di_spread_change=self.di_spread-self.di_spread_5; mx=np.maximum(self.plus_di_values,self.minus_di_values); mn=np.minimum(self.plus_di_values,self.minus_di_values); self.di_ratio=np.divide(mx,mn,out=np.full(len(mx),np.nan,float),where=np.isfinite(mn)&(mn!=0)); self.bull_regime_return_values=self._trailing_return_array(config.bull_regime_lookback_days); self.market_regime_values=self._market_regime_array(); self.atr_pct_values=np.divide(self.atr_values,self.close,out=np.full(len(self.close),np.nan,float),where=np.isfinite(self.atr_values)&(self.close!=0)); candle_range=self.high-self.low; self.close_location_values=np.divide(self.close-self.low,candle_range,out=np.full(len(self.close),np.nan,float),where=np.isfinite(candle_range)&(candle_range!=0)); self.risk=self._risk_array()
        self.active_pairs=[]; self.completed_pairs=[]; self.telemetry_rows=[]; self.skipped_signals=[]; self.skipped_daily_entries=[]; self.signals_evaluated=0; self.daily_entry_opportunities=0; self.daily_entries_on_schedule=0; self.daily_entries_next_available=0; self.pending_daily_entry=None; self.next_pair_id=1; self.current_equity=config.initial_equity; self.missing_intrabar_intervals=[]; self.fallback_reasons=[]
        self.entry_delta=pd.Timedelta(minutes=config.strategy_timeframe_minutes)
        self.session_vwap=self._utc_session_vwap()
        self.mean_reversion_mean=ema(self.close,config.mean_reversion_period)
        self.mean_reversion_distance_atr=distance_from_mean_atr(self.close,self.mean_reversion_mean,self.atr_values)
        self.mean_reversion_distance_atr_previous=lag(self.mean_reversion_distance_atr,1)
        self.profile_rsi_values={period:rsi(self.close,period) for period in {p.rsi_period for p in self.config.strategy_profiles.values()}}
        self.profile_momentum_values={hours:self._trailing_return_hours_array(hours) for hours in {p.momentum_lookback_hours for p in self.config.strategy_profiles.values()}}
        self.sr_detector = SupportResistanceDetector(
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
        ) if config.enable_support_resistance_analysis else None
        self._pending_sr_context = None
        self.last_timeout_exit_time=None
        self.trading_start=pd.Timestamp(config.trading_start_date, tz="UTC") if config.trading_start_date else None
        self.trading_end=pd.Timestamp(config.trading_end_date, tz="UTC") if config.trading_end_date else None
        self.first_valid_atr_timestamp=self._first_valid_atr_timestamp()
        self.warmup_candle_count=int((self.data.timestamp < self.trading_start).sum()) if self.trading_start is not None else 0
        self.daily_entry_tz=ZoneInfo(config.daily_entry_timezone)
        hh, mm = [int(part) for part in str(config.daily_entry_time).split(":", 1)]
        self.daily_entry_minutes = hh * 60 + mm

    def _market_structure_snapshot(self, i):
        """Confirmed 2-left/2-right swing structure known at candle ``i``.

        Strength fields are telemetry only.  They never reject an entry.  A
        pivot at index ``j`` becomes available only at ``j + 2``, preventing
        future candles from leaking into the direction decision.
        """
        span=2; lookback=20
        result={
            "market_structure_direction":"ABSTAIN",
            "market_structure_reason":"INSUFFICIENT_CONFIRMED_SWINGS",
            "market_structure_lookback":lookback,
            "market_structure_pivot_span":span,
        }
        if i < span * 2:
            return result
        start=max(span,i-lookback+1); stop=i-span
        swing_highs=[]; swing_lows=[]
        for j in range(start,stop+1):
            left_high=self.high[j-span:j]; right_high=self.high[j+1:j+span+1]
            left_low=self.low[j-span:j]; right_low=self.low[j+1:j+span+1]
            if self.high[j] > np.max(left_high) and self.high[j] > np.max(right_high): swing_highs.append(j)
            if self.low[j] < np.min(left_low) and self.low[j] < np.min(right_low): swing_lows.append(j)
        result["market_structure_confirmed_high_count"]=len(swing_highs)
        result["market_structure_confirmed_low_count"]=len(swing_lows)
        if len(swing_highs)<2 or len(swing_lows)<2:
            return result
        previous_high_i,latest_high_i=swing_highs[-2:]; previous_low_i,latest_low_i=swing_lows[-2:]
        previous_high=float(self.high[previous_high_i]); latest_high=float(self.high[latest_high_i])
        previous_low=float(self.low[previous_low_i]); latest_low=float(self.low[latest_low_i])
        high_change=latest_high-previous_high; low_change=latest_low-previous_low
        if high_change>0 and low_change>0:
            direction="LONG"; reason="HIGHER_HIGH_AND_HIGHER_LOW"; sign=1.0
        elif high_change<0 and low_change<0:
            direction="SHORT"; reason="LOWER_HIGH_AND_LOWER_LOW"; sign=-1.0
        elif high_change==0 or low_change==0:
            direction="ABSTAIN"; reason="EQUAL_SWING_BOUNDARY"; sign=np.nan
        elif high_change>0:
            direction="ABSTAIN"; reason="HIGHER_HIGH_AND_LOWER_LOW"; sign=np.nan
        else:
            direction="ABSTAIN"; reason="LOWER_HIGH_AND_HIGHER_LOW"; sign=np.nan
        atr_value=float(self.atr_values[i]) if np.isfinite(self.atr_values[i]) and self.atr_values[i]>0 else np.nan
        high_pct=high_change/previous_high if previous_high else np.nan; low_pct=low_change/previous_low if previous_low else np.nan
        directional_high=sign*high_change/atr_value if np.isfinite(sign) and np.isfinite(atr_value) else np.nan
        directional_low=sign*low_change/atr_value if np.isfinite(sign) and np.isfinite(atr_value) else np.nan
        previous_range=previous_high-previous_low; latest_range=latest_high-latest_low
        breakout=(float(self.close[i])-previous_high)/atr_value if direction=="LONG" and np.isfinite(atr_value) else ((previous_low-float(self.close[i]))/atr_value if direction=="SHORT" and np.isfinite(atr_value) else np.nan)
        result.update({
            "market_structure_direction":direction,"market_structure_reason":reason,
            "market_structure_previous_swing_high":previous_high,"market_structure_latest_swing_high":latest_high,
            "market_structure_previous_swing_low":previous_low,"market_structure_latest_swing_low":latest_low,
            "market_structure_previous_swing_high_time":pd.Timestamp(self.times[previous_high_i]),"market_structure_latest_swing_high_time":pd.Timestamp(self.times[latest_high_i]),
            "market_structure_previous_swing_low_time":pd.Timestamp(self.times[previous_low_i]),"market_structure_latest_swing_low_time":pd.Timestamp(self.times[latest_low_i]),
            "market_structure_latest_swing_high_age":i-latest_high_i,"market_structure_latest_swing_low_age":i-latest_low_i,
            "market_structure_high_displacement":high_change,"market_structure_low_displacement":low_change,
            "market_structure_high_displacement_pct":high_pct,"market_structure_low_displacement_pct":low_pct,
            "market_structure_directional_high_displacement_atr":directional_high,"market_structure_directional_low_displacement_atr":directional_low,
            "market_structure_minimum_displacement_atr":min(directional_high,directional_low) if np.isfinite(directional_high) and np.isfinite(directional_low) else np.nan,
            "market_structure_maximum_displacement_atr":max(directional_high,directional_low) if np.isfinite(directional_high) and np.isfinite(directional_low) else np.nan,
            "market_structure_latest_to_previous_range_ratio":latest_range/previous_range if previous_range>0 else np.nan,
            "market_structure_breakout_distance_atr":breakout,
            "market_structure_breakout_confirmed_by_close":bool((direction=="LONG" and self.close[i]>previous_high) or (direction=="SHORT" and self.close[i]<previous_low)),
        })
        return result

    def _analyze_support_resistance(self, i, direction):
        """Analyze price location relative to support/resistance levels."""
        if self.sr_detector is None:
            return None
        try:
            return self.sr_detector.analyze_price_location(
                i,
                self.open,
                self.high,
                self.low,
                self.close,
                self.atr_values,
                direction,
            )
        except Exception as e:
            self.log(f"SR analysis failed at index {i}: {e}")
            return None

    def _selected_direction(self, i):
        """Select direction solely from DI values known at candle ``i``."""
        if not self.config.enable_di_direction_selection:
            return None
        plus=float(self.plus_di_values[i]); minus=float(self.minus_di_values[i])
        if not np.isfinite(plus) or not np.isfinite(minus) or plus == minus:
            return None
        return "LONG" if plus > minus else "SHORT"

    def _di_pressure_snapshot(self, i, direction):
        """Return analysis-only DI telemetry using candle i and i-lookback."""
        plus=float(self.plus_di_values[i]); minus=float(self.minus_di_values[i])
        lookback=self.config.di_pressure_lookback
        result={"plus_di":plus,"minus_di":minus,"directional_di":np.nan,"opposing_di":np.nan,
                "plus_di_change":np.nan,"minus_di_change":np.nan,"directional_di_change":np.nan,
                "opposing_di_change":np.nan,"di_spread":float(self.di_spread[i]),"di_spread_change":np.nan,
                "di_pressure_state":"UNKNOWN","di_pressure_lookback":lookback}
        if direction not in ("LONG","SHORT") or not np.isfinite(plus) or not np.isfinite(minus): return result
        result["directional_di"],result["opposing_di"]=(plus,minus) if direction=="LONG" else (minus,plus)
        if not self.config.enable_di_pressure_analysis or i < lookback: return result
        old_plus=float(self.plus_di_values[i-lookback]); old_minus=float(self.minus_di_values[i-lookback])
        if not np.isfinite(old_plus) or not np.isfinite(old_minus): return result
        result["plus_di_change"]=plus-old_plus; result["minus_di_change"]=minus-old_minus
        old_directional,old_opposing=(old_plus,old_minus) if direction=="LONG" else (old_minus,old_plus)
        dc=result["directional_di_change"]=result["directional_di"]-old_directional
        oc=result["opposing_di_change"]=result["opposing_di"]-old_opposing
        result["di_spread_change"]=result["di_spread"]-abs(old_plus-old_minus)
        result["di_pressure_state"]="EXPANDING" if dc>0 and oc<0 else ("CONTRACTING" if dc<0 and oc>0 else "MIXED")
        return result

    def _mean_reversion_snapshot(self, i, di_direction, trade_direction=None):
        """Return analysis-only entry-time mean-reversion telemetry."""
        result={
            "mean_reversion_enabled":bool(self.config.enable_mean_reversion_analysis),
            "mean_reversion_period":int(self.config.mean_reversion_period),
            "mean_price":np.nan,"mean_distance_atr":np.nan,"mean_distance_atr_previous":np.nan,
            "mean_distance_change_atr":np.nan,"mean_reversion_state":"UNKNOWN",
            "mean_reversion_motion":"UNKNOWN","mean_reversion_alignment":"UNKNOWN",
            "mean_reversion_di_alignment":"UNKNOWN","mean_reversion_trade_alignment":"UNKNOWN",
            "mean_reversion_strength":-1,"mean_reversion_strength_label":"UNKNOWN",
        }
        if not self.config.enable_mean_reversion_analysis:
            return result
        mean=float(self.mean_reversion_mean[i]); distance=float(self.mean_reversion_distance_atr[i]); previous=float(self.mean_reversion_distance_atr_previous[i])
        result["mean_price"]=mean if np.isfinite(mean) else np.nan
        result["mean_distance_atr"]=distance if np.isfinite(distance) else np.nan
        result["mean_distance_atr_previous"]=previous if np.isfinite(previous) else np.nan
        if np.isfinite(distance) and np.isfinite(previous): result["mean_distance_change_atr"]=distance-previous
        result["mean_reversion_state"]=classify_state(distance)
        result["mean_reversion_motion"]=classify_motion(distance,previous)
        di_alignment=classify_alignment(distance,di_direction)
        trade_alignment=classify_alignment(distance,trade_direction or di_direction)
        strength,strength_label=classify_strength(distance)
        result["mean_reversion_alignment"]=di_alignment
        result["mean_reversion_di_alignment"]=di_alignment
        result["mean_reversion_trade_alignment"]=trade_alignment
        result["mean_reversion_strength"]=strength
        result["mean_reversion_strength_label"]=strength_label
        return result

    def _first_valid_atr_timestamp(self):
        idx=np.where(np.isfinite(self.atr_values))[0]
        return pd.Timestamp(self.times[int(idx[0])]) if len(idx) else None
    def _utc_session_vwap(self):
        """UTC-midnight anchored VWAP, calculated only from each completed candle."""
        timestamps=pd.to_datetime(self.data.timestamp,utc=True)
        typical=(self.high+self.low+self.close)/3.0
        weighted=typical*self.volume
        sessions=timestamps.dt.floor("D")
        cumulative_weighted=pd.Series(weighted).groupby(sessions).cumsum().to_numpy(float)
        cumulative_volume=pd.Series(self.volume).groupby(sessions).cumsum().to_numpy(float)
        return np.divide(cumulative_weighted,cumulative_volume,out=np.full(len(self.data),np.nan),where=cumulative_volume>0)
    def run(self)->pd.DataFrame:
        total=len(self.times)
        self._emit_progress(0,total)
        for i in range(total):
            self.current_index=i
            active_at_candle_start = bool(self.active_pairs)
            self._update_positions_to_strategy_index(i); self._record_active_telemetry(i); self._collect_closed_pairs()
            decision = self._entry_decision(i, active_at_candle_start)
            if decision:
                self.signals_evaluated += 1
                passed, reason = self._entry_filter_result(decision["indicator_index"], decision["execution_index"])
                if passed: self._open_pair(decision["execution_index"], passed, reason, decision)
                else:
                    self._record_skipped_signal(decision["indicator_index"], reason)
                    if self.config.enable_daily_entry_schedule: self._record_skipped_daily_entry(decision["scheduled_timestamp"], "FILTER_REJECTED", reason)
            processed=i+1
            if processed == total or processed % self.progress_interval == 0:
                self._emit_progress(processed,total)
        self._force_close_end(); self._collect_closed_pairs(force=True); self._emit_progress(total,total); return self.results_frame()
    def _emit_progress(self, processed_candles, total_candles):
        if self.progress_callback is not None:
            self.progress_callback(processed_candles, total_candles, len(self.completed_pairs), self.next_pair_id - 1)
    def _risk_array(self):
        if self.config.risk_mode==RiskMode.FIXED: return np.full(len(self.times), self.config.fixed_r, float)
        if self.config.risk_mode==RiskMode.PERCENT: return self.close*self.config.percent_r
        return self.atr_values*self.config.atr_multiplier
    def _trailing_return_array(self, lookback_days):
        """Trailing close-to-close return using only candles known at each index."""
        result=np.full(len(self.times),np.nan,float)
        times=pd.DatetimeIndex(pd.to_datetime(self.times,utc=True))
        targets=times-pd.Timedelta(days=lookback_days)
        prior=np.searchsorted(times.asi8,targets.asi8,side="right")-1
        valid=prior>=0
        result[valid]=self.close[valid]/self.close[prior[valid]]-1.0
        return result
    def _trailing_return_hours_array(self, lookback_hours):
        """Trailing close-to-close return using only candles known at each index."""
        result=np.full(len(self.times),np.nan,float)
        times=pd.DatetimeIndex(pd.to_datetime(self.times,utc=True))
        targets=times-pd.Timedelta(hours=lookback_hours)
        prior=np.searchsorted(times.asi8,targets.asi8,side="right")-1
        valid=prior>=0
        result[valid]=self.close[valid]/self.close[prior[valid]]-1.0
        return result
    def _market_regime_array(self):
        if self.config.market_regime_method == "ASSET_RETURN":
            threshold = abs(float(self.config.bull_regime_return_threshold))
            return np.array([None if not np.isfinite(v) else ("BULL" if v >= threshold else ("BEAR" if v <= -threshold else "SIDEWAYS")) for v in self.bull_regime_return_values], dtype=object)
        benchmark_path = self.config.structural_regime_benchmark_csv
        if benchmark_path is None:
            strategy_path = Path(self.config.input_csv)
            benchmark_path = strategy_path if self.config.market_regime_method == "ASSET_STRUCTURAL" else strategy_path.with_name("BTCUSDT_1h.csv")
        benchmark_path = Path(benchmark_path)
        if not benchmark_path.is_file():
            label = "Asset" if self.config.market_regime_method == "ASSET_STRUCTURAL" else "BTC"
            raise ValueError(f"{label} structural regime requires benchmark data: {benchmark_path}")
        benchmark = pd.read_csv(benchmark_path)
        benchmark.columns = [str(c).strip().lower().replace(" ", "_") for c in benchmark.columns]
        time_col = next((c for c in ("timestamp", "open_time", "time", "datetime", "date") if c in benchmark.columns), None)
        if time_col is None or "close" not in benchmark.columns:
            raise ValueError("Structural regime data must contain timestamp and close columns")
        raw = benchmark[time_col]
        numeric = pd.to_numeric(raw, errors="coerce")
        unit = "ms" if numeric.notna().mean() > .9 else None
        benchmark["timestamp"] = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp")
        daily = benchmark.set_index("timestamp")["close"].resample("1D").last().dropna().to_frame()
        days = self.config.structural_regime_sma_days
        slope = self.config.structural_regime_slope_lookback_days
        daily["sma"] = daily["close"].rolling(days, min_periods=days).mean()
        daily["prior_sma"] = daily["sma"].shift(slope)
        daily["regime"] = np.where((daily["close"] > daily["sma"]) & (daily["sma"] > daily["prior_sma"]), "BULL", np.where((daily["close"] < daily["sma"]) & (daily["sma"] < daily["prior_sma"]), "BEAR", "SIDEWAYS"))
        daily.loc[daily[["sma", "prior_sma"]].isna().any(axis=1), "regime"] = None
        # A UTC daily candle is usable only from the following midnight.
        available = daily.reset_index()[["timestamp", "regime"]]
        available["timestamp"] += pd.Timedelta(days=1)
        target = pd.DataFrame({"timestamp": pd.to_datetime(self.times, utc=True)})
        mapped = pd.merge_asof(target.sort_values("timestamp"), available.sort_values("timestamp"), on="timestamp", direction="backward")
        return mapped["regime"].to_numpy(object)
    def _regime_at(self, i):
        value=self.market_regime_values[i]
        return None if value is None or pd.isna(value) else str(value)
    def _structural_sma_arrays(self, sma_days, slope_lookback_days):
        """SMA and its value at a prior date, using only completed strategy candles."""
        candles=max(1,int(round(sma_days*1440/self.config.strategy_timeframe_minutes)))
        sma=pd.Series(self.close,dtype=float).rolling(candles,min_periods=candles).mean().to_numpy()
        prior_sma=np.full(len(sma),np.nan,float)
        times=pd.DatetimeIndex(pd.to_datetime(self.times,utc=True))
        targets=times-pd.Timedelta(days=slope_lookback_days)
        prior=np.searchsorted(times.asi8,targets.asi8,side="right")-1
        valid=prior>=0
        prior_sma[valid]=sma[prior[valid]]
        return sma,prior_sma
    def _entry_time(self,i): return pd.Timestamp(self.times[i]) + self.entry_delta
    def _execution_time(self,i): return pd.Timestamp(self.times[i]) if self.config.enable_daily_entry_schedule else self._entry_time(i)
    def _price_index(self,i): return i
    def _indicator_index_for_execution(self,i): return i - 1 if self.config.enable_daily_entry_schedule else i
    def _in_trading_window(self,i):
        et=self._execution_time(i)
        return (self.trading_start is None or et >= self.trading_start) and (self.trading_end is None or et <= self.trading_end)

    def _is_scheduled_candle(self, i):
        ts = pd.Timestamp(self.times[i])
        if ts.tzinfo is None: ts = ts.tz_localize("UTC")
        local = ts.tz_convert(self.daily_entry_tz)
        return local.hour * 60 + local.minute == self.daily_entry_minutes

    def _active_pair_for_skip(self):
        if not self.active_pairs: return None
        return self.active_pairs[0]

    def _record_skipped_daily_entry(self, scheduled_ts, reason, detail=""):
        pair = self._active_pair_for_skip()
        positions = pair.positions() if pair is not None else []
        exit_times = [pd.Timestamp(pos.exit_time) for pos in positions if pos.exit_time is not None]
        self.skipped_daily_entries.append({"scheduled_timestamp": scheduled_ts, "timezone": self.config.daily_entry_timezone, "reason": reason, "active_trade_id": pair.pair_id if pair is not None else None, "active_trade_entry_time": pair.strategy_entry_time if pair is not None else None, "active_trade_expected_or_actual_exit_time": max(exit_times) if exit_times else None, "detail": detail})

    def _daily_entry_decision(self, i, active_at_candle_start):
        if self.pending_daily_entry and not self.active_pairs:
            if self._base_entry_allowed(i):
                d=self.pending_daily_entry; self.pending_daily_entry=None; d.update({"execution_index": i, "indicator_index": i-1, "actual_entry_timestamp": pd.Timestamp(self.times[i]), "entry_schedule_status": "NEXT_AVAILABLE_CANDLE"}); return d
        if not self._is_scheduled_candle(i): return None
        scheduled_ts = pd.Timestamp(self.times[i])
        self.daily_entry_opportunities += 1
        if i <= 0 or not np.isfinite(self.risk[i-1]) or self.risk[i-1] <= 0:
            self._record_skipped_daily_entry(scheduled_ts, "INDICATOR_WARMUP") ; return None
        if not self._in_trading_window(i):
            self._record_skipped_daily_entry(scheduled_ts, "OUTSIDE_TRADING_DATE_RANGE") ; return None
        if active_at_candle_start or len(self.active_pairs) >= self.config.max_active_pairs:
            self._record_skipped_daily_entry(scheduled_ts, "ACTIVE_TRADE")
            if self.config.daily_entry_missed_policy == DailyEntryMissedPolicy.NEXT_AVAILABLE_CANDLE:
                self.pending_daily_entry={"scheduled_timestamp": scheduled_ts}
            return None
        return {"execution_index": i, "indicator_index": i-1, "scheduled_timestamp": scheduled_ts, "actual_entry_timestamp": scheduled_ts, "entry_schedule_status": "ON_TIME"}

    def _base_entry_allowed(self, i):
        return i > 0 and np.isfinite(self.risk[i-1]) and self.risk[i-1] > 0 and len(self.active_pairs) < self.config.max_active_pairs and self._in_trading_window(i)

    def _entry_decision(self, i, active_at_candle_start=False):
        if self.config.enable_daily_entry_schedule:
            return self._daily_entry_decision(i, active_at_candle_start)
        return {"execution_index": i, "indicator_index": i, "scheduled_timestamp": None, "actual_entry_timestamp": self._entry_time(i), "entry_schedule_status": None} if self._should_enter(i) else None
    def _should_enter(self,i):
        if not np.isfinite(self.risk[i]) or self.risk[i] <= 0 or len(self.active_pairs) >= self.config.max_active_pairs or not self._in_trading_window(i):
            return False
        if self.last_timeout_exit_time is not None and self._entry_time(i) <= self.last_timeout_exit_time:
            return False
        if self.config.entry_mode == EntryMode.WAIT_UNTIL_CLOSED:
            return not self.active_pairs
        if self.config.entry_mode == EntryMode.EVERY_N_CANDLES:
            return i % self.config.entry_interval == 0
        return False
    def _entry_filter_result(self, i, execution_i=None):
        self._pending_sr_context = None
        profile_result = self._strategy_profile_filter_result(i, execution_i)
        if not profile_result[0]:
            return profile_result
        if self.config.enable_support_resistance_analysis:
            direction = self._selected_direction(i)
            sr_reject, sr_reason = self._should_reject_for_sr(i, direction, None)
            if sr_reject:
                return False, sr_reason
        return True, profile_result[1]

    def _should_reject_for_sr(self, i, direction, sr_context=None):
        """Apply independent, trader-facing S/R entry constraints."""
        if not self.config.enable_support_resistance_analysis:
            return False, None
        if sr_context is None:
            if direction is None:
                return False, None
            sr_context = self._analyze_support_resistance(i, direction)
            self._pending_sr_context = (i, direction, sr_context)
        if sr_context is None:
            return False, None

        mode = self.config.sr_filter_mode
        if mode == "ANALYSIS_ONLY":
            return False, None
        if mode != "APPLY_ENTRY_RULES":
            return False, None

        room = sr_context.room_in_direction_atr
        if direction == "LONG":
            if self.config.sr_long_avoid_near_resistance and sr_context.near_resistance:
                return True, "SR_LONG_NEAR_RESISTANCE"
            if self.config.sr_long_require_near_support and not sr_context.near_support:
                return True, "SR_LONG_NOT_NEAR_SUPPORT"
            if self.config.sr_long_block_broken_support and sr_context.support_state == "SUPPORT_BROKEN":
                return True, "SR_LONG_SUPPORT_BROKEN"
            minimum = self.config.sr_long_min_room_to_resistance_atr
            if minimum > 0 and room < minimum:
                return True, "SR_LONG_INSUFFICIENT_ROOM_TO_RESISTANCE"
        elif direction == "SHORT":
            if self.config.sr_short_avoid_near_support and sr_context.near_support:
                return True, "SR_SHORT_NEAR_SUPPORT"
            if self.config.sr_short_require_near_resistance and not sr_context.near_resistance:
                return True, "SR_SHORT_NOT_NEAR_RESISTANCE"
            if self.config.sr_short_block_broken_resistance and sr_context.resistance_state == "RESISTANCE_BROKEN":
                return True, "SR_SHORT_RESISTANCE_BROKEN"
            minimum = self.config.sr_short_min_room_to_support_atr
            if minimum > 0 and room < minimum:
                return True, "SR_SHORT_INSUFFICIENT_ROOM_TO_SUPPORT"
        return False, None

    def _profile_context(self, i):
        plus=float(self.plus_di_values[i]); minus=float(self.minus_di_values[i]); regime=self._regime_at(i)
        if not all(np.isfinite(v) for v in (plus,minus)) or regime is None:
            return None
        direction=self._selected_direction(i)
        if direction is None: return None
        key=profile_key(regime,direction)
        return regime,direction,key,self.config.strategy_profiles[key]

    def _strategy_profile_filter_result(self, i, execution_i=None):
        context=self._profile_context(i)
        if context is None:
            plus=float(self.plus_di_values[i]); minus=float(self.minus_di_values[i]); regime=self._regime_at(i)
            if not all(np.isfinite(v) for v in (plus,minus)) or regime is None:
                return False,"Strategy profile classification indicator warm-up incomplete"
            return False,"Strategy profile direction unavailable"
        regime,direction,key,profile=context
        if not profile.enabled: return False,f"Strategy profile {key} is disabled"
        if profile.entry_rules:
            rejected=self._strategy_profile_rule_group_match(i,direction,profile,"REJECT",profile.reject_rule_match_mode)
            if rejected: return False,f"Strategy profile {key} rejected by entry rules"
            flipped=self._strategy_profile_rule_group_match(i,direction,profile,"FLIP",profile.flip_rule_match_mode)
            action="will be flipped" if flipped else "will trade in its normal direction"
            return True,f"Strategy profile {key} passed; flip rules {'matched' if flipped else 'did not match'}: entry {action}"
        return True,f"Strategy profile {key} passed"

    def _strategy_profile_rule_value(self, i, direction, profile, indicator):
        if indicator=="DI_SPREAD": return float(self.di_spread[i])
        if indicator=="ADX": return float(self.adx_values[i])
        if indicator=="ATR_PCT": return float(self.atr_pct_values[i])
        if indicator=="RSI": return float(self.profile_rsi_values[profile.rsi_period][i])
        if indicator=="BB_WIDTH": return float(self.bb_width[i])
        if indicator=="CLOSE_LOCATION": return float(self.close_location_values[i])
        if indicator=="MOMENTUM": return float(self.profile_momentum_values[profile.momentum_lookback_hours][i])
        atr_value=float(self.atr_values[i]); vwap=float(self.session_vwap[i])
        if not np.isfinite(atr_value) or atr_value <= 0 or not np.isfinite(vwap): return np.nan
        return ((float(self.close[i])-vwap) if direction=="LONG" else (vwap-float(self.close[i])))/atr_value

    def _strategy_profile_entry_rule_matches(self, i, direction, profile, rule):
        value=self._strategy_profile_rule_value(i,direction,profile,rule["indicator"])
        if not np.isfinite(value): return False
        inside=float(rule["minimum"]) <= value <= float(rule["maximum"])
        return inside if rule.get("condition","INSIDE")=="INSIDE" else not inside

    def _strategy_profile_rule_group_match(self, i, direction, profile, action, mode):
        rules=[rule for rule in profile.entry_rules if rule.get("action")==action]
        if not rules: return False
        matches=[self._strategy_profile_entry_rule_matches(i,direction,profile,rule) for rule in rules]
        return all(matches) if mode=="ALL" else any(matches)

    def _record_skipped_signal(self, i, reason):
        row={"strategy_candle_open_time": self.times[i], "strategy_entry_time": self._entry_time(i), "strategy_entry_price": float(self.close[i]), "adx": float(self.adx_values[i]) if np.isfinite(self.adx_values[i]) else np.nan, "plus_di": float(self.plus_di_values[i]) if np.isfinite(self.plus_di_values[i]) else np.nan, "minus_di": float(self.minus_di_values[i]) if np.isfinite(self.minus_di_values[i]) else np.nan, "di_spread": float(self.di_spread[i]) if np.isfinite(self.di_spread[i]) else np.nan, "market_regime_return": float(self.bull_regime_return_values[i]) if np.isfinite(self.bull_regime_return_values[i]) else np.nan, "bb_width": float(self.bb_width[i]) if np.isfinite(self.bb_width[i]) else np.nan, "entry_filter_passed": False, "entry_filter_reason": reason, "adx_filter_passed": False, "adx_filter_reason": reason}
        selected=self._selected_direction(i)
        row.update(self._di_pressure_snapshot(i, selected))
        row.update(self._mean_reversion_snapshot(i, selected, selected))
        self.skipped_signals.append(row)
    def _cap_qty(self, qty, entry_price, equity):
        capped=False; cap_qty=qty
        if self.config.max_effective_leverage_per_leg is not None: cap_qty=min(cap_qty, self.config.max_effective_leverage_per_leg*equity/entry_price)
        if self.config.max_combined_effective_leverage is not None: cap_qty=min(cap_qty, self.config.max_combined_effective_leverage*equity/(self._entry_leg_count()*entry_price))
        capped=cap_qty < qty - 1e-12
        return cap_qty,capped
    def _entry_leg_count(self):
        return 1
    def _active_positions(self, pair):
        return pair.positions()

    def _resolved_profile_direction(self, i, profile_context=None):
        """Return the native post-profile direction used for entry execution."""
        context = profile_context or self._profile_context(i)
        if context is None:
            raise ValueError("Cannot resolve direction without a Strategy Profile context")
        _, original_direction, _, profile = context
        rule_flip = bool(profile.entry_rules and self._strategy_profile_rule_group_match(
            i, original_direction, profile, "FLIP", profile.flip_rule_match_mode
        ))
        direction = original_direction
        if profile.flip_direction or rule_flip:
            direction = "SHORT" if original_direction == "LONG" else "LONG"
        return original_direction, direction

    def _open_pair(self, i, entry_filter_passed=True, entry_filter_reason="Strategy profile passed", schedule=None):
        ind_i = schedule["indicator_index"] if schedule else i
        raw = self.open[i] if self.config.enable_daily_entry_schedule else self.close[i]
        profile_context = self._profile_context(ind_i)
        if profile_context is None:
            raise ValueError("Cannot open a trade without a current Strategy Profile context")
        regime, _, active_profile_key, active_profile = profile_context
        original_direction, direction = self._resolved_profile_direction(ind_i, profile_context)
        if direction != original_direction:
            entry_filter_reason = (
                f"{entry_filter_reason}; direction flipped after profile rules: "
                f"{original_direction} -> {direction}"
            )

        side = Side.LONG if direction == "LONG" else Side.SHORT
        side_sign = 1 if side == Side.LONG else -1
        entry = raw * (1 + self.config.slippage) if side == Side.LONG else raw * (1 - self.config.slippage)
        partial_sl_enabled = active_profile.partial_stop_enabled
        partial_tp_enabled = active_profile.partial_profit_enabled
        sl1_r = active_profile.sl1_r
        sl1_close_pct = active_profile.sl1_close_pct
        sl2_r = active_profile.sl2_r
        tp1_r = active_profile.tp1_r
        tp1_close_pct = active_profile.tp1_close_pct
        tp2_r = active_profile.tp2_r
        r = float(self.risk[ind_i])
        stop_mult = sl2_r if partial_sl_enabled else active_profile.stop_loss_multiple
        stop = stop_mult * r
        risk_amt = self.current_equity * self.config.risk_per_leg * active_profile.risk_multiplier
        uncapped = risk_amt / stop
        qty, capped = self._cap_qty(uncapped, entry, self.current_equity)
        target_distance = stop * active_profile.reward_risk_ratio
        sl = entry - side_sign * stop
        tp = entry + side_sign * target_distance
        entry_fee_rate = self.config.maker_fee if self.config.use_maker_entry else self.config.taker_fee
        entry_fee = entry * qty * entry_fee_rate
        pos = Position(
            side, self._execution_time(i), i, entry, stop, sl, tp, qty, risk_amt, entry * qty,
            float(self.atr_values[ind_i]), uncapped, qty * entry / self.current_equity,
            distance_unit=r, entry_fee=entry_fee, fees=entry_fee, original_sl=sl,
            be_enabled=active_profile.break_even_enabled,
            be_mode="R_OFFSET" if active_profile.break_even_offset_r else "ENTRY_PRICE",
            be_offset_r=active_profile.break_even_offset_r,
        )

        pos.atr_checkpoint_extension_enabled = bool(active_profile.atr_checkpoint_tp_extension_enabled)
        if pos.atr_checkpoint_extension_enabled:
            pos.atr_checkpoint_di_spread_minimum = active_profile.atr_checkpoint_di_spread_minimum
            pos.atr_checkpoint_bb_width_minimum = active_profile.atr_checkpoint_bb_width_minimum
            pos.atr_checkpoint_profit_lock_start = active_profile.atr_checkpoint_profit_lock_start
            pos.atr_checkpoint_profit_lock_distance = active_profile.atr_checkpoint_profit_lock_distance
            pos.atr_checkpoint_initial_tp = pos.tp
            pos.atr_checkpoint_final_tp_r = target_distance / r

        pos.r_step_trailing_enabled = bool(active_profile.r_step_trailing_enabled)
        if pos.r_step_trailing_enabled:
            pos.r_step_activation_r = active_profile.r_step_activation_r
            pos.r_step_distance_r = active_profile.r_step_distance_r
            pos.r_step_size_r = active_profile.r_step_size_r
            pos.r_step_maximum_r = active_profile.r_step_maximum_r
            pos.r_step_next_checkpoint_r = pos.r_step_activation_r
            pos.r_step_initial_tp = pos.tp
            pos.r_step_activation_close_pct = active_profile.r_step_activation_close_pct
            if pos.r_step_activation_close_pct > 0:
                pos.partial_tp_enabled = True
                pos.original_quantity = pos.quantity
                pos.remaining_quantity = pos.quantity
                pos.tp1_quantity = pos.quantity * pos.r_step_activation_close_pct / 100.0
                pos.tp1_price = pos.entry_price + side_sign * pos.r_step_activation_r * pos.risk
                pos.r_step_activation_quantity = pos.tp1_quantity
                pos.r_step_runner_quantity = pos.quantity - pos.tp1_quantity
            if pos.r_step_maximum_r > 0:
                pos.tp = pos.entry_price + side_sign * pos.r_step_maximum_r * pos.risk

        if partial_tp_enabled:
            pos.partial_tp_enabled = True
            pos.original_quantity = pos.quantity
            pos.remaining_quantity = pos.quantity
            pos.tp1_quantity = pos.quantity * tp1_close_pct / 100.0
            pos.tp2_quantity = pos.quantity - pos.tp1_quantity
            pos.tp1_price = pos.entry_price + side_sign * tp1_r * stop
            pos.tp2_price = pos.entry_price + side_sign * tp2_r * stop
            pos.tp = pos.tp2_price
            pos.final_active_stop = pos.sl
        if partial_sl_enabled:
            pos.partial_sl_enabled = True
            pos.original_quantity = pos.quantity
            pos.remaining_quantity = pos.quantity
            pos.sl1_quantity = pos.quantity * sl1_close_pct / 100.0
            pos.sl1_price = pos.entry_price - side_sign * sl1_r * r
            pos.sl2_price = pos.entry_price - side_sign * sl2_r * r
            pos.sl = pos.sl2_price
            pos.original_sl = pos.sl2_price
            pos.final_active_stop = pos.sl2_price

        pos.be_enabled = bool(active_profile.break_even_enabled)
        pos.be_offset_r = active_profile.break_even_offset_r
        pos.profile_break_even_activation_r = (
            active_profile.break_even_activation_r if active_profile.break_even_enabled else None
        )
        pos.trailing_enabled = bool(active_profile.trailing_enabled)
        if pos.trailing_enabled:
            pos.trailing_distance_r = active_profile.trailing_distance_r
            pos.trailing_activation_price = pos.entry_price + side_sign * pos.risk * active_profile.trailing_activation_r
            pos.favourable_price = pos.entry_price

        long = pos if side == Side.LONG else None
        short = pos if side == Side.SHORT else None

        # Capture support/resistance data
        # Capture support/resistance data
        if self.config.enable_support_resistance_analysis:
            pending = self._pending_sr_context
            for pos in [long, short]:
                if pos is None:
                    continue
                direction = pos.side.value
                sr_context = pending[2] if pending is not None and pending[0] == ind_i and pending[1] == direction else self._analyze_support_resistance(ind_i, direction)
                if sr_context is None:
                    continue
                pos.sr_nearest_support = sr_context.nearest_support_price
                pos.sr_nearest_resistance = sr_context.nearest_resistance_price
                pos.sr_support_distance_atr = sr_context.nearest_support_distance_atr
                pos.sr_resistance_distance_atr = sr_context.nearest_resistance_distance_atr
                pos.sr_support_distance_price = sr_context.nearest_support_distance_price
                pos.sr_resistance_distance_price = sr_context.nearest_resistance_distance_price
                pos.sr_near_support = sr_context.near_support
                pos.sr_near_resistance = sr_context.near_resistance
                pos.sr_inside_support_zone = sr_context.inside_support_zone
                pos.sr_inside_resistance_zone = sr_context.inside_resistance_zone
                pos.sr_location = sr_context.price_location.value
                pos.sr_trade_location_rating = sr_context.trade_location_rating.value
                pos.sr_room_in_direction_atr = sr_context.room_in_direction_atr
                pos.sr_support_state = sr_context.support_state
                pos.sr_resistance_state = sr_context.resistance_state
                pos.sr_support_tested = sr_context.support_tested
                pos.sr_resistance_tested = sr_context.resistance_tested
                pos.sr_support_held = sr_context.support_held
                pos.sr_resistance_held = sr_context.resistance_held
                pos.sr_support_rejection_atr = sr_context.support_rejection_atr
                pos.sr_resistance_rejection_atr = sr_context.resistance_rejection_atr
                pos.sr_support_test_count = sr_context.support_test_count
                pos.sr_resistance_test_count = sr_context.resistance_test_count
                pos.sr_bars_since_support_test = sr_context.bars_since_support_test
                pos.sr_bars_since_resistance_test = sr_context.bars_since_resistance_test
                pos.sr_support_last_test_index = sr_context.support_last_test_index
                pos.sr_resistance_last_test_index = sr_context.resistance_last_test_index
                pos.sr_support_last_test_time = pd.Timestamp(self.times[sr_context.support_last_test_index]) if sr_context.support_last_test_index is not None else None
                pos.sr_resistance_last_test_time = pd.Timestamp(self.times[sr_context.resistance_last_test_index]) if sr_context.resistance_last_test_index is not None else None
                pos.sr_confirmation_rating = sr_context.confirmation_rating
                pos.sr_support_zone_low = sr_context.support_zone_low
                pos.sr_support_zone_high = sr_context.support_zone_high
                pos.sr_resistance_zone_low = sr_context.resistance_zone_low
                pos.sr_resistance_zone_high = sr_context.resistance_zone_high
                if direction == "LONG":
                    pos.sr_level_price = sr_context.nearest_support_price
                    pos.sr_zone_low = sr_context.support_zone_low
                    pos.sr_zone_high = sr_context.support_zone_high
                else:
                    pos.sr_level_price = sr_context.nearest_resistance_price
                    pos.sr_zone_low = sr_context.resistance_zone_low
                    pos.sr_zone_high = sr_context.resistance_zone_high
        
        pair = TradePair(
            self.next_pair_id, long, short, self.current_equity, pd.Timestamp(self.times[i]),
            self._execution_time(i), raw, capped
        )
        pair.trade_direction = direction
        pair.daily_schedule_enabled = self.config.enable_daily_entry_schedule
        pair.scheduled_entry_time = self.config.daily_entry_time
        pair.scheduled_entry_timezone = self.config.daily_entry_timezone
        pair.scheduled_entry_timestamp = (schedule or {}).get("scheduled_timestamp")
        pair.actual_entry_timestamp = (schedule or {}).get("actual_entry_timestamp", self._execution_time(i))
        pair.entry_schedule_status = (schedule or {}).get("entry_schedule_status")
        pair.strategy_profile_key = active_profile_key
        pair.applied_stop_loss_multiple = stop_mult
        pair.applied_partial_sl_enabled = partial_sl_enabled
        pair.applied_sl1_r = sl1_r
        pair.applied_sl1_close_pct = sl1_close_pct
        pair.applied_sl2_r = sl2_r
        pair.applied_partial_tp_enabled = partial_tp_enabled
        pair.applied_tp1_r = tp1_r
        pair.applied_tp1_close_pct = tp1_close_pct
        pair.applied_tp2_r = tp2_r
        pair.profile_timeout_enabled = bool(active_profile.timeout_enabled)
        pair.profile_timeout_minutes = int(active_profile.timeout_minutes) if pair.profile_timeout_enabled else None
        pair.di_sizing_direction = original_direction
        pair.sizing_direction = direction
        pair.long_size_multiplier = 1.0 if direction == "LONG" else 0.0
        pair.short_size_multiplier = 1.0 if direction == "SHORT" else 0.0
        for key, value in self._di_pressure_snapshot(ind_i, original_direction).items():
            setattr(pair, key, value)
        for key, value in self._mean_reversion_snapshot(ind_i, original_direction, direction).items():
            setattr(pair, key, value)
        pair.di_reward_risk_regime = regime
        pair.di_applied_long_reward_risk_ratio = active_profile.reward_risk_ratio if direction == "LONG" else np.nan
        pair.di_applied_short_reward_risk_ratio = active_profile.reward_risk_ratio if direction == "SHORT" else np.nan
        pair.market_regime_return = float(self.bull_regime_return_values[ind_i]) if np.isfinite(self.bull_regime_return_values[ind_i]) else np.nan
        pair.entry_atr_pct = float(self.atr_pct_values[ind_i]) if np.isfinite(self.atr_pct_values[ind_i]) else np.nan
        pair.entry_close_location = float(self.close_location_values[ind_i]) if np.isfinite(self.close_location_values[ind_i]) else np.nan
        profile_rsi = float(self.profile_rsi_values[active_profile.rsi_period][ind_i])
        profile_momentum = float(self.profile_momentum_values[active_profile.momentum_lookback_hours][ind_i])
        pair.entry_rsi = profile_rsi if np.isfinite(profile_rsi) else np.nan
        pair.directional_momentum_return = profile_momentum if np.isfinite(profile_momentum) else np.nan
        pair.long_momentum_return = pair.directional_momentum_return
        pair.market_regime = self._regime_at(ind_i)
        pair.market_regime_method = self.config.market_regime_method
        pair.bull_regime = pair.market_regime == "BULL"
        if self.config.enable_daily_entry_schedule:
            if pair.entry_schedule_status == "ON_TIME":
                self.daily_entries_on_schedule += 1
            elif pair.entry_schedule_status == "NEXT_AVAILABLE_CANDLE":
                self.daily_entries_next_available += 1
        pair.adx = float(self.adx_values[ind_i]) if np.isfinite(self.adx_values[ind_i]) else np.nan
        pair.plus_di = float(self.plus_di_values[ind_i]) if np.isfinite(self.plus_di_values[ind_i]) else np.nan
        pair.minus_di = float(self.minus_di_values[ind_i]) if np.isfinite(self.minus_di_values[ind_i]) else np.nan
        self._attach_market_state(pair, ind_i)
        pair.entry_filter_passed = entry_filter_passed
        pair.entry_filter_reason = entry_filter_reason
        self.active_pairs.append(pair)
        self._record_pair_telemetry(pair, i)
        self.next_pair_id += 1

    def _attach_market_state(self, pair, i):
        fields = {"bb_middle":self.bb_middle,"bb_upper":self.bb_upper,"bb_lower":self.bb_lower,"bb_width":self.bb_width,"bb_width_pct":self.bb_width_pct,"bb_width_1":self.bb_width_1,"bb_width_3":self.bb_width_3,"bb_width_5":self.bb_width_5,"bb_width_entry_5bar_change":self.bb_width_change,"bb_width_entry_5bar_change_pct":self.bb_width_change_pct,"di_spread":self.di_spread,"di_ratio":self.di_ratio,"di_spread_1":self.di_spread_1,"di_spread_3":self.di_spread_3,"di_spread_5":self.di_spread_5,"di_spread_entry_5bar_change":self.di_spread_change,"utc_session_vwap":self.session_vwap}
        for name, arr in fields.items():
            setattr(pair, name, float(arr[i]) if np.isfinite(arr[i]) else np.nan)
        pair.short_vwap_distance_atr = ((pair.utc_session_vwap-float(self.close[i]))/float(self.atr_values[i])) if np.isfinite(pair.utc_session_vwap) and np.isfinite(self.atr_values[i]) and self.atr_values[i] > 0 else np.nan
    def _update_positions_to_strategy_index(self,i):
        for pair in self.active_pairs:
            first = pair.positions()[0]
            if i > first.entry_index:
                self._scan_pair_exit(pair,i)
    def _scan_pair_exit(self,pair,i):
        if not self.config.use_intrabar_data or self.intrabar_data is None:
            if self._maybe_timeout_pair_at(pair, i, pd.Timestamp(self.times[i]), float(self.open[i]), ExitSource.FALLBACK_15M): return
            for pos in pair.positions():
                if pos.is_open: self._scan_exit(pos,i)
            return
        # Only inspect the strategy interval currently being processed. Starting
        # every scan at pair entry replays old intrabars with today's ratcheted
        # trailing stop and can therefore manufacture an exit in the past.
        start=max(pd.Timestamp(pair.strategy_entry_time), pd.Timestamp(self.times[i])); end=pd.Timestamp(self.times[i])+self.entry_delta
        if start.floor(f"{self.config.intrabar_timeframe_minutes}min") != start:
            for pos in pair.positions():
                if pos.is_open: self._fallback_exit(pos,i,"timestamp_alignment_failure")
            return
        sub=self.intrabar_data[(self.intrabar_data.timestamp>=start)&(self.intrabar_data.timestamp<end)]
        incomplete=sub.empty or (sub.timestamp.iloc[0] > start + pd.Timedelta(minutes=self.config.intrabar_timeframe_minutes)) or self._has_missing_intrabar(sub,start,end)
        if incomplete:
            reason="no_overlapping_intrabar_rows" if sub.empty else "intrabar_gap"
            for pos in pair.positions():
                if pos.is_open: pos.missing_intrabar_data=True
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.ERROR:
                raise ValueError(f"Missing {self.config.intrabar_timeframe_minutes}-minute intrabar candles during open trade")
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.WARN_AND_USE_15M:
                for pos in pair.positions():
                    if pos.is_open: self._fallback_exit(pos,i,reason)
                return
            if sub.empty: return
        for j,row in sub.iterrows():
            if self._maybe_timeout_pair_at(pair, j, pd.Timestamp(row.timestamp), float(row.open), ExitSource.INTRABAR):
                break
            before=tuple(pos.is_open for pos in pair.positions())
            for pos in pair.positions():
                if pos.is_open and not (pos.be_active_after is not None and pd.Timestamp(row.timestamp) < pd.Timestamp(pos.be_active_after)):
                    self._maybe_exit_bar(pos,j,row.high,row.low,row.timestamp,ExitSource.INTRABAR)
            if not pair.is_open or before != tuple(pos.is_open for pos in pair.positions()):
                pass
        if self.intrabar_data.timestamp.max() < end - pd.Timedelta(minutes=self.config.intrabar_timeframe_minutes):
            for pos in pair.positions():
                if pos.is_open: pos.missing_intrabar_data=True
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.ERROR:
                raise ValueError(f"Missing {self.config.intrabar_timeframe_minutes}-minute intrabar candles during open trade")
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.WARN_AND_USE_15M:
                for pos in pair.positions():
                    if pos.is_open: self._fallback_exit(pos,i,"end_of_intrabar_data")

    def _scan_exit(self,pos,i):
        # Trailing state is stateful; never apply its current stop to a candle
        # which was already processed with an older stop.
        start=max(pd.Timestamp(pos.entry_time), pd.Timestamp(self.times[i])); end=pd.Timestamp(self.times[i])+self.entry_delta
        if not self.config.use_intrabar_data:
            return self._fallback_exit(pos,i,"intrabar_disabled")
        if self.intrabar_data is None:
            return self._fallback_exit(pos,i,"intrabar_file_missing")
        if start.floor(f"{self.config.intrabar_timeframe_minutes}min") != start:
            return self._fallback_exit(pos,i,"timestamp_alignment_failure")
        sub=self.intrabar_data[(self.intrabar_data.timestamp>=start)&(self.intrabar_data.timestamp<end)]
        if sub.empty:
            reason="end_of_intrabar_data" if self.intrabar_data.timestamp.max() < start else "no_overlapping_intrabar_rows"
            pos.missing_intrabar_data=True
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.ERROR: raise ValueError(f"Missing {self.config.intrabar_timeframe_minutes}-minute intrabar candles during open trade")
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.WARN_AND_USE_15M: return self._fallback_exit(pos,i,reason)
            return False
        expected=pd.Timedelta(minutes=self.config.intrabar_timeframe_minutes)
        if sub.timestamp.iloc[0] > start + expected:
            pos.missing_intrabar_data=True
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.ERROR: raise ValueError(f"Missing {self.config.intrabar_timeframe_minutes}-minute intrabar candles during open trade")
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.WARN_AND_USE_15M: return self._fallback_exit(pos,i,"timestamp_alignment_failure")
        if self._has_missing_intrabar(sub,start,end):
            pos.missing_intrabar_data=True
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.ERROR: raise ValueError("Missing intrabar candles during open trade")
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.WARN_AND_USE_15M: return self._fallback_exit(pos,i,"intrabar_gap")
        for j,row in sub.iterrows():
            if self._maybe_exit_bar(pos,j,row.high,row.low,row.timestamp,ExitSource.INTRABAR): return True
        if self.intrabar_data.timestamp.max() < end - pd.Timedelta(minutes=self.config.intrabar_timeframe_minutes):
            pos.missing_intrabar_data=True
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.ERROR: raise ValueError(f"Missing {self.config.intrabar_timeframe_minutes}-minute intrabar candles during open trade")
            if self.config.intrabar_missing_policy==IntrabarMissingPolicy.WARN_AND_USE_15M: return self._fallback_exit(pos,i,"end_of_intrabar_data")
        return False


    def _maybe_timeout_pair_at(self, pair, i, timestamp, raw_open, source):
        if not getattr(pair, "profile_timeout_enabled", False):
            return False
        minutes = getattr(pair, "profile_timeout_minutes", None)
        if minutes is None:
            return False
        timeout_at = pd.Timestamp(pair.strategy_entry_time) + pd.Timedelta(minutes=minutes)
        timestamp = pd.Timestamp(timestamp)
        if timestamp < timeout_at:
            return False
        for pos in pair.positions():
            if pos.is_open:
                slip = 1-self.config.slippage if pos.side == Side.LONG else 1+self.config.slippage
                self._close_position(pos, i, float(raw_open)*slip, ExitReason.PROFILE_TIMEOUT, source, timestamp)
        pair.profile_timeout_triggered = True
        pair.timeout_minutes = int(minutes)
        pair.timeout_exit_time = timestamp
        self.last_timeout_exit_time = timestamp
        return True
    def _fallback_exit(self,pos,i,reason):
        pos.fallback_reason=reason; self.fallback_reasons.append(reason)
        return self._maybe_exit_ohlc(pos,i,ExitSource.FALLBACK_15M)
    def _has_missing_intrabar(self,sub,start,end):
        expected=pd.Timedelta(minutes=self.config.intrabar_timeframe_minutes)
        diffs=sub.timestamp.diff().dropna(); gaps=diffs[diffs>expected]
        for idx,d in gaps.items(): self.missing_intrabar_intervals.append((sub.loc[idx-1,'timestamp'],sub.loc[idx,'timestamp'])); print(f"WARNING: Missing intrabar data {sub.loc[idx-1,'timestamp']} to {sub.loc[idx,'timestamp']}")
        return not gaps.empty
    def _maybe_exit_ohlc(self,pos,i,source): return self._maybe_exit_bar(pos,i,self.high[i],self.low[i],self.times[i],source)
    def _maybe_exit_bar(self,pos,i,high,low,timestamp,source):
        # Break-even is a Protection rule, independent of the selected profit
        # taking method. A newly raised BE stop is deliberately not eligible
        # until the next processed bar/intrabar candle, avoiding invented OHLC
        # paths inside the candle that triggered it.
        if pos.r_step_trailing_enabled:
            changed=self._maybe_r_step_trailing_exit(pos,i,float(high),float(low),timestamp,source)
            if pos.is_open: self._maybe_activate_break_even(pos,float(high),float(low),timestamp)
            return changed
        if pos.partial_sl_enabled and pos.partial_tp_enabled:
            changed=self._maybe_combined_partial_exit(pos,i,float(high),float(low),timestamp,source)
            if pos.is_open: self._maybe_activate_break_even(pos,float(high),float(low),timestamp)
            return changed
        if pos.partial_sl_enabled:
            changed=self._maybe_partial_sl_exit(pos,i,float(high),float(low),timestamp,source)
            if pos.is_open: self._maybe_activate_break_even(pos,float(high),float(low),timestamp)
            return changed
        if pos.partial_tp_enabled:
            changed=self._maybe_partial_exit(pos,i,float(high),float(low),timestamp,source)
            if pos.is_open: self._maybe_activate_break_even(pos,float(high),float(low),timestamp)
            return changed
        if pos.atr_checkpoint_extension_enabled:
            self._apply_atr_checkpoint_extensions(pos, float(high), float(low), timestamp)
        if pos.trailing_enabled:
            changed=self._maybe_trailing_exit(pos,i,float(high),float(low),timestamp,source)
            if pos.is_open: self._maybe_activate_break_even(pos,float(high),float(low),timestamp)
            return changed
        hit_tp=high>=pos.tp if pos.side==Side.LONG else low<=pos.tp; hit_sl=low<=pos.sl if pos.side==Side.LONG else high>=pos.sl
        if not(hit_tp or hit_sl):
            self._maybe_activate_break_even(pos,float(high),float(low),timestamp)
            return False
        if hit_tp and hit_sl: pos.ambiguous=True; use_tp=self.config.tie_policy==TiePolicy.OPTIMISTIC
        else: use_tp=hit_tp
        raw=pos.tp if use_tp else pos.sl; slip=1-self.config.slippage if pos.side==Side.LONG else 1+self.config.slippage
        reason=ExitReason.TP if use_tp else (pos.be_exit_reason if pos.be_triggered else (ExitReason.ATR_CHECKPOINT_PROFIT_LOCK if pos.atr_checkpoint_profit_lock_r is not None else ExitReason.SL))
        self._close_position(pos,i,raw*slip,reason,source,timestamp); return True

    def _maybe_activate_break_even(self,pos,high,low,timestamp):
        activation=pos.profile_break_even_activation_r
        if activation is None or pos.be_triggered or not pos.is_open:
            return False
        reached=(
            high>=pos.entry_price+activation*pos.risk
            if pos.side==Side.LONG
            else low<=pos.entry_price-activation*pos.risk
        )
        if not reached:
            return False
        new_sl=(
            pos.entry_price+pos.be_offset_r*pos.risk
            if pos.side==Side.LONG
            else pos.entry_price-pos.be_offset_r*pos.risk
        )
        improves=new_sl>pos.sl if pos.side==Side.LONG else new_sl<pos.sl
        if not improves:
            return False
        pos.sl=new_sl
        pos.final_active_stop=new_sl
        pos.be_triggered=True
        pos.be_trigger_time=pd.Timestamp(timestamp)
        pos.be_stop_price=new_sl
        pos.be_exit_reason=ExitReason.BE_R_OFFSET if pos.be_offset_r else ExitReason.BE
        return True

    def _maybe_r_step_trailing_exit(self, pos, i, high, low, timestamp, source):
        """Trail a qualifying bull long in discrete R steps.

        The original fixed TP is ignored. At each favourable checkpoint the
        stop is placed ``distance_r`` behind it. A zero maximum leaves the
        position open until the staircase stop (or end of data) closes it.
        """
        is_long = pos.side == Side.LONG
        hit_current_stop = low <= pos.sl if is_long else high >= pos.sl
        maximum_r = pos.r_step_maximum_r
        hit_maximum = bool(maximum_r > 0 and (
            high >= pos.entry_price + maximum_r * pos.risk if is_long
            else low <= pos.entry_price - maximum_r * pos.risk
        ))
        if hit_current_stop or hit_maximum:
            if hit_current_stop and hit_maximum:
                pos.ambiguous = True
                use_target = self.config.tie_policy == TiePolicy.OPTIMISTIC
            else:
                use_target = hit_maximum
            raw = (
                pos.entry_price + (1 if is_long else -1) * maximum_r * pos.risk
                if use_target else pos.sl
            )
            slip = 1-self.config.slippage if is_long else 1+self.config.slippage
            reason = ExitReason.TP if use_target else (
                ExitReason.R_STEP_TRAILING_STOP if pos.r_step_trailing_active else ExitReason.SL
            )
            self._close_position(pos, i, raw*slip, reason, source, timestamp)
            return True

        favourable_r = ((high-pos.entry_price) if is_long else (pos.entry_price-low)) / pos.risk
        while favourable_r + 1e-12 >= pos.r_step_next_checkpoint_r:
            checkpoint_r = float(pos.r_step_next_checkpoint_r)
            if (
                not pos.r_step_activation_partial_taken
                and pos.r_step_activation_close_pct > 0
                and checkpoint_r + 1e-12 >= pos.r_step_activation_r
            ):
                raw = pos.entry_price + (1 if is_long else -1) * pos.r_step_activation_r * pos.risk
                execution = raw * (1-self.config.slippage if is_long else 1+self.config.slippage)
                self._partial_fill(pos, pos.r_step_activation_quantity, execution, "tp1", i, timestamp, source)
                pos.r_step_activation_partial_taken = True
            lock_r = checkpoint_r - pos.r_step_distance_r
            candidate = pos.entry_price + (1 if is_long else -1) * lock_r * pos.risk
            improves = candidate > pos.sl if is_long else candidate < pos.sl
            pos.r_step_checkpoint_count += 1
            pos.r_step_last_checkpoint_r = checkpoint_r
            pos.r_step_last_checkpoint_time = pd.Timestamp(timestamp)
            if improves:
                pos.sl = candidate
                pos.final_active_stop = candidate
                pos.r_step_locked_r = lock_r
            pos.r_step_trailing_active = True
            pos.r_step_next_checkpoint_r = checkpoint_r + pos.r_step_size_r

        # Pessimistic same-bar handling: a newly raised stop may also have been
        # touched inside this bar after the checkpoint.
        hit_raised_stop = low <= pos.sl if is_long else high >= pos.sl
        if pos.r_step_trailing_active and hit_raised_stop:
            slip = 1-self.config.slippage if is_long else 1+self.config.slippage
            self._close_position(pos, i, pos.sl*slip, ExitReason.R_STEP_TRAILING_STOP, source, timestamp)
            return True
        return False

    def _checkpoint_indicator_index(self, timestamp):
        """Latest fully closed strategy candle at an intrabar checkpoint."""
        needle=np.datetime64(pd.Timestamp(timestamp).tz_localize(None), "ns")
        strategy_i=int(np.searchsorted(self.times, needle, side="right")-1)
        strategy_i=max(0,min(strategy_i,len(self.times)-1))
        return max(0,strategy_i-1)

    def _apply_atr_checkpoint_extensions(self, pos, high, low, timestamp):
        """Extend the biased leg one ATR at a time when DI and BB checks pass."""
        direction=1 if pos.side==Side.LONG else -1
        unit=pos.distance_unit or pos.risk
        favourable_r=((high-pos.entry_price) if pos.side==Side.LONG else (pos.entry_price-low))/unit
        while pos.is_open and favourable_r + 1e-12 >= pos.atr_checkpoint_next_r:
            checkpoint_r=float(pos.atr_checkpoint_next_r)
            indicator_i=self._checkpoint_indicator_index(timestamp)
            plus=float(self.plus_di_values[indicator_i]); minus=float(self.minus_di_values[indicator_i])
            directional_spread=(plus-minus) if pos.side==Side.LONG else (minus-plus)
            width=float(self.bb_width[indicator_i])
            passed=(
                np.isfinite(directional_spread)
                and directional_spread >= pos.atr_checkpoint_di_spread_minimum
                and np.isfinite(width)
                and width >= pos.atr_checkpoint_bb_width_minimum
            )
            pos.atr_checkpoint_count += 1
            pos.atr_checkpoint_last_time=pd.Timestamp(timestamp)
            pos.atr_checkpoint_last_r=checkpoint_r
            pos.atr_checkpoint_last_di_spread=float(directional_spread) if np.isfinite(directional_spread) else None
            pos.atr_checkpoint_last_bb_width=float(width) if np.isfinite(width) else None
            pos.atr_checkpoint_last_passed=bool(passed)
            if not passed:
                pos.atr_checkpoint_fail_count += 1
                # A failed checkpoint leaves the current TP/SL unchanged and
                # permanently ends extension for this position.
                pos.atr_checkpoint_extension_enabled=False
                break
            pos.atr_checkpoint_pass_count += 1
            new_tp_r=checkpoint_r+2.0
            pos.tp=pos.entry_price+direction*new_tp_r*unit
            pos.atr_checkpoint_final_tp_r=new_tp_r
            if checkpoint_r >= pos.atr_checkpoint_profit_lock_start:
                lock_r=checkpoint_r-pos.atr_checkpoint_profit_lock_distance
                candidate=pos.entry_price+direction*lock_r*unit
                improves=candidate>pos.sl if pos.side==Side.LONG else candidate<pos.sl
                if improves:
                    pos.sl=candidate
                    pos.final_active_stop=candidate
                    pos.atr_checkpoint_profit_lock_r=lock_r
            pos.atr_checkpoint_next_r=checkpoint_r+1.0

    def _maybe_partial_sl_exit(self,pos,i,high,low,timestamp,source):
        """Close SL1 percentage once, then leave the remainder for TP or SL2.

        If Break-even Protection was activated on an earlier processed bar, its
        stop takes priority over the deeper SL ladder for the remaining size.
        """
        if pos.trailing_enabled and pos.trailing_active:
            if self._maybe_trailing_exit(pos,i,high,low,timestamp,source): return True
        is_long=pos.side==Side.LONG
        adverse=lambda price: low<=price if is_long else high>=price
        tp_hit=high>=pos.tp if is_long else low<=pos.tp
        protective_stop_hit=bool(pos.be_triggered and adverse(pos.sl))
        sl1_hit=adverse(pos.sl1_price) and not pos.sl1_hit
        sl2_hit=adverse(pos.sl2_price)
        slip=1-self.config.slippage if is_long else 1+self.config.slippage
        sl1_execution=pos.sl1_price*slip
        if tp_hit and (not (protective_stop_hit or sl1_hit or sl2_hit) or self.config.tie_policy==TiePolicy.OPTIMISTIC):
            raw=pos.tp; price=raw*slip
            self._partial_sl_fill(pos,pos.remaining_quantity,price,"tp",i,timestamp,source)
            self._finalize_partial_sl(pos,ExitReason.TP)
            return True
        if protective_stop_hit:
            self._partial_sl_fill(pos,pos.remaining_quantity,pos.sl*slip,"stop",i,timestamp,source)
            self._finalize_partial_sl(pos,pos.be_exit_reason or ExitReason.BE)
            return True
        if sl2_hit:
            if not pos.sl1_hit:
                self._partial_sl_fill(pos,pos.sl1_quantity,sl1_execution,"sl1",i,timestamp,source)
            self._partial_sl_fill(pos,pos.remaining_quantity,pos.sl2_price*slip,"sl2",i,timestamp,source)
            self._finalize_partial_sl(pos,ExitReason.SL)
            return True
        if sl1_hit:
            self._partial_sl_fill(pos,pos.sl1_quantity,sl1_execution,"sl1",i,timestamp,source)
            return True
        if pos.trailing_enabled and not pos.trailing_active:
            if self._maybe_trailing_exit(pos,i,high,low,timestamp,source): return True
        return False

    def _maybe_combined_partial_exit(self,pos,i,high,low,timestamp,source):
        """Resolve TP1/TP2 and SL1/SL2 ladders.

        Profit Taking never moves the stop. If Break-even Protection activated
        on an earlier processed bar, that protective stop takes priority over
        the deeper partial-stop ladder for the remaining size.
        """
        if pos.trailing_enabled and pos.trailing_active:
            if self._maybe_trailing_exit(pos,i,high,low,timestamp,source): return True
        is_long=pos.side==Side.LONG
        adverse=lambda price: low<=price if is_long else high>=price
        favourable=lambda price: high>=price if is_long else low<=price
        tp1_hit=favourable(pos.tp1_price) and not pos.tp1_hit
        tp2_hit=favourable(pos.tp2_price) and pos.tp1_hit
        protective_stop_hit=bool(pos.be_triggered and adverse(pos.sl))
        sl1_hit=not pos.sl1_hit and adverse(pos.sl1_price)
        sl2_hit=adverse(pos.sl2_price)
        slip=1-self.config.slippage if is_long else 1+self.config.slippage

        def execute_losses():
            changed=False
            if protective_stop_hit:
                self._partial_fill(pos,pos.remaining_quantity,pos.sl*slip,"stop",i,timestamp,source)
                self._finalize_partial(pos,pos.be_exit_reason or ExitReason.BE)
                return True
            if sl2_hit:
                if not pos.sl1_hit:
                    changed=self._partial_sl_fill(pos,pos.sl1_quantity,pos.sl1_price*slip,"sl1",i,timestamp,source) or changed
                if pos.is_open:
                    self._partial_fill(pos,pos.remaining_quantity,pos.sl2_price*slip,"stop",i,timestamp,source)
                    self._finalize_partial(pos,ExitReason.SL)
                return True
            if sl1_hit:
                changed=self._partial_sl_fill(pos,pos.sl1_quantity,pos.sl1_price*slip,"sl1",i,timestamp,source) or changed
                if not pos.is_open:
                    self._finalize_partial(pos,ExitReason.SL)
            return changed

        if self.config.tie_policy==TiePolicy.PESSIMISTIC and (protective_stop_hit or sl1_hit or sl2_hit):
            return execute_losses()

        changed=False
        if tp1_hit:
            changed=self._partial_fill(pos,pos.tp1_quantity,pos.tp1_price,"tp1",i,timestamp,source) or changed
            if not pos.is_open:
                self._finalize_partial(pos,ExitReason.TP)
                return True
            tp2_hit=favourable(pos.tp2_price)
        if pos.is_open and pos.tp1_hit and tp2_hit:
            self._partial_fill(pos,pos.remaining_quantity,pos.tp2_price,"tp2",i,timestamp,source)
            self._finalize_partial(pos,ExitReason.TP)
            return True
        if pos.is_open:
            changed=execute_losses() or changed
        if pos.is_open and pos.trailing_enabled and not pos.trailing_active:
            if self._maybe_trailing_exit(pos,i,high,low,timestamp,source): return True
        return changed

    def _partial_sl_fill(self,pos,quantity,price,stage,i,timestamp,source):
        quantity=min(max(0.0,quantity),pos.remaining_quantity)
        if quantity <= 0: return False
        gross=((price-pos.entry_price) if pos.side==Side.LONG else (pos.entry_price-price))*quantity
        fee=price*quantity*(self.config.maker_fee if self.config.use_maker_exit else self.config.taker_fee)
        pos.remaining_quantity=max(0.0,pos.remaining_quantity-quantity); pos.realized_pnl+=gross-fee; pos.exit_fee+=fee; pos.fees=pos.entry_fee+pos.exit_fee
        if stage=="sl1":
            pos.sl1_hit=True; pos.sl1_exit_time=pd.Timestamp(timestamp); pos.sl1_exit_price=price; pos.sl1_gross_pnl=gross; pos.sl1_fees=fee; pos.sl1_net_pnl=gross-fee
        else:
            pos.stop_exit_time=pd.Timestamp(timestamp); pos.stop_exit_price=price; pos.stop_exit_quantity=quantity; pos.stop_gross_pnl=gross; pos.stop_fees=fee; pos.stop_net_pnl=gross-fee
        if pos.remaining_quantity <= 1e-12:
            pos.remaining_quantity=0.0; pos.exit_time=pd.Timestamp(timestamp); pos.exit_index=i; pos.exit_price=price; pos.exit_source=source
        return True

    def _finalize_partial_sl(self,pos,reason):
        pos.exit_reason=reason
        if pos.sl1_hit and reason==ExitReason.TP:
            pos.final_exit_reason="SL1_THEN_TP"
        elif pos.sl1_hit and reason==ExitReason.SL:
            pos.final_exit_reason="SL1_THEN_SL2"
        elif pos.sl1_hit:
            pos.final_exit_reason=f"SL1_THEN_{reason.value}"
        else:
            pos.final_exit_reason=reason.value
        pos.gross_pnl=(pos.sl1_gross_pnl or 0)+(pos.stop_gross_pnl or 0)
        pos.net_pnl=pos.gross_pnl-pos.fees; pos.gross_r=pos.gross_pnl/pos.risk_amount; pos.net_r=pos.net_pnl/pos.risk_amount
        pos.quantity=pos.original_quantity
        move=(pos.exit_price-pos.entry_price) if pos.side==Side.LONG else (pos.entry_price-pos.exit_price); pos.price_r=move/pos.risk

    def _partial_fill(self, pos, quantity, price, stage, i, timestamp, source):
        quantity=min(max(0.0, quantity), pos.remaining_quantity)
        if quantity <= 0: return False
        gross=((price-pos.entry_price) if pos.side==Side.LONG else (pos.entry_price-price))*quantity
        fee=price*quantity*(self.config.maker_fee if self.config.use_maker_exit else self.config.taker_fee)
        net=gross-fee; pos.remaining_quantity=max(0.0,pos.remaining_quantity-quantity)
        setattr(pos,f"{stage}_exit_time",pd.Timestamp(timestamp)); setattr(pos,f"{stage}_exit_price",price)
        setattr(pos,f"{stage}_gross_pnl",gross); setattr(pos,f"{stage}_fees",fee); setattr(pos,f"{stage}_net_pnl",net)
        pos.realized_pnl += net; pos.exit_fee += fee; pos.fees=pos.entry_fee+pos.exit_fee
        if stage=="tp1":
            pos.tp1_hit=True; pos.tp1_quantity=quantity
        elif stage=="tp2":
            pos.tp2_hit=True; pos.tp2_quantity=quantity
        else: pos.stop_exit_quantity=quantity
        if pos.remaining_quantity <= 1e-12:
            pos.remaining_quantity=0.0; pos.exit_time=pd.Timestamp(timestamp); pos.exit_index=i; pos.exit_price=price; pos.exit_source=source
        return True

    def _maybe_partial_exit(self,pos,i,high,low,timestamp,source):
        """Resolve unknown OHLC paths consistently.

        PESSIMISTIC orders an already-active stop before favourable targets.
        OPTIMISTIC orders TP1, then fixed TP2 (or trailing update), before the
        stop. Targets are monotonic, so TP2 is never processed before TP1.
        """
        trailing_prechecked=pos.trailing_enabled and pos.trailing_active
        if trailing_prechecked:
            if self._maybe_trailing_exit(pos,i,high,low,timestamp,source): return True
        stop_hit=False if trailing_prechecked else (low<=pos.sl if pos.side==Side.LONG else high>=pos.sl)
        tp1_hit=(high>=pos.tp1_price if pos.side==Side.LONG else low<=pos.tp1_price) and not pos.tp1_hit
        tp2_hit=(high>=pos.tp2_price if pos.side==Side.LONG else low<=pos.tp2_price)
        if stop_hit and self.config.tie_policy==TiePolicy.PESSIMISTIC:
            return self._finish_partial_stop(pos,i,timestamp,source)
        changed=False
        if tp1_hit:
            changed=self._partial_fill(pos,pos.tp1_quantity,pos.tp1_price,"tp1",i,timestamp,source)
        if pos.is_open and pos.tp1_hit and tp2_hit:
            self._partial_fill(pos,pos.remaining_quantity,pos.tp2_price,"tp2",i,timestamp,source); self._finalize_partial(pos,ExitReason.TP); return True
        if pos.is_open and stop_hit:
            return self._finish_partial_stop(pos,i,timestamp,source)
        if pos.is_open and pos.trailing_enabled and not pos.trailing_active:
            if self._maybe_trailing_exit(pos,i,high,low,timestamp,source): return True
        return changed

    def _finish_partial_stop(self,pos,i,timestamp,source):
        raw=pos.sl; price=raw*(1-self.config.slippage if pos.side==Side.LONG else 1+self.config.slippage)
        self._partial_fill(pos,pos.remaining_quantity,price,"stop",i,timestamp,source)
        reason=pos.be_exit_reason if pos.be_triggered else ExitReason.SL; self._finalize_partial(pos,reason)
        return True

    def _finalize_partial(self,pos,reason):
        pos.exit_reason=reason
        stages=[]
        if pos.sl1_hit: stages.append("SL1")
        if pos.tp1_hit: stages.append("TP1")
        if pos.tp2_hit: stages.append("TP2")
        elif reason==ExitReason.SL: stages.append("SL")
        elif reason==ExitReason.TP and not pos.tp1_hit: stages.append("TP")
        elif reason in (ExitReason.BE,ExitReason.BE_R_OFFSET): stages.append(reason.value)
        elif reason==ExitReason.TRAILING_STOP: stages.append("TRAILING_STOP")
        elif reason==ExitReason.R_STEP_TRAILING_STOP: stages.append("R_STEP_TRAILING_STOP")
        pos.final_exit_reason="_THEN_".join(stages) or reason.value
        pos.gross_pnl=sum(v or 0 for v in (pos.sl1_gross_pnl,pos.tp1_gross_pnl,pos.tp2_gross_pnl,pos.stop_gross_pnl))
        pos.net_pnl=pos.gross_pnl-pos.fees; pos.gross_r=pos.gross_pnl/pos.risk_amount; pos.net_r=pos.net_pnl/pos.risk_amount
        pos.quantity=pos.original_quantity
        move=(pos.exit_price-pos.entry_price) if pos.side==Side.LONG else (pos.entry_price-pos.exit_price); pos.price_r=move/pos.risk
    def _maybe_trailing_exit(self, pos, i, high, low, timestamp, source):
        """Apply current Strategy Profile trailing with pessimistic OHLC ordering.

        The stop that existed at bar open has priority. Only after that check can
        the favourable extreme activate or ratchet the trail. A newly raised stop
        becomes eligible on the next processed bar, avoiding invented intrabar paths.
        """
        is_long = pos.side == Side.LONG
        old_stop = (
            max(pos.sl, pos.be_stop_price or -np.inf, pos.trailing_stop or -np.inf)
            if is_long else
            min(pos.sl, pos.be_stop_price or np.inf, pos.trailing_stop or np.inf)
        )
        stop_hit = low <= old_stop if is_long else high >= old_stop
        if stop_hit:
            reason = (
                ExitReason.TRAILING_STOP
                if pos.trailing_active and pos.trailing_stop is not None and abs(old_stop-pos.trailing_stop) < 1e-9
                else (
                    pos.be_exit_reason
                    if pos.be_triggered and pos.be_stop_price is not None and abs(old_stop-pos.be_stop_price) < 1e-9
                    else ExitReason.SL
                )
            )
            return self._close_at_stop(pos, i, old_stop, reason, source, timestamp)

        activation_hit = (
            (high >= pos.trailing_activation_price if is_long else low <= pos.trailing_activation_price)
            if pos.trailing_activation_price is not None else False
        )
        if not pos.trailing_active and activation_hit:
            pos.trailing_active = True
            pos.trailing_activation_time = pd.Timestamp(timestamp)
        if pos.trailing_active:
            extreme = high if is_long else low
            pos.favourable_price = max(pos.favourable_price, extreme) if is_long else min(pos.favourable_price, extreme)
            distance_r = float(pos.trailing_distance_r)
            candidate = (
                pos.favourable_price - pos.risk*distance_r
                if is_long else
                pos.favourable_price + pos.risk*distance_r
            )
            pos.trailing_stop = (
                max(pos.trailing_stop or -np.inf, candidate)
                if is_long else
                min(pos.trailing_stop or np.inf, candidate)
            )
            active_stop = (
                max(pos.sl, pos.be_stop_price or -np.inf, pos.trailing_stop)
                if is_long else
                min(pos.sl, pos.be_stop_price or np.inf, pos.trailing_stop)
            )
            pos.sl = active_stop
            pos.final_active_stop = active_stop
        return False

    def _close_at_stop(self,pos,i,raw,reason,source,timestamp):
        slip=1-self.config.slippage if pos.side==Side.LONG else 1+self.config.slippage
        self._close_position(pos,i,raw*slip,reason,source,timestamp)
        if reason == ExitReason.TRAILING_STOP:
            pos.trailing_exit_price=pos.exit_price; pos.trailing_profit_r=pos.price_r
        return True
    def _close_position(self,pos,i,exit_price,reason,source=None,timestamp=None):
        if pos.partial_sl_enabled and pos.partial_tp_enabled and pos.remaining_quantity > 0:
            exit_time=pd.Timestamp(timestamp if timestamp is not None else self.times[i])
            self._partial_fill(pos,pos.remaining_quantity,exit_price,"stop",i,exit_time,source or ExitSource.FALLBACK_15M)
            self._finalize_partial(pos,reason)
            return
        if pos.partial_sl_enabled and pos.remaining_quantity > 0:
            exit_time=pd.Timestamp(timestamp if timestamp is not None else self.times[i])
            self._partial_sl_fill(pos,pos.remaining_quantity,exit_price,"sl2",i,exit_time,source or ExitSource.FALLBACK_15M)
            self._finalize_partial_sl(pos,reason)
            return
        if pos.partial_tp_enabled and pos.remaining_quantity > 0:
            exit_time=pd.Timestamp(timestamp if timestamp is not None else self.times[i])
            self._partial_fill(pos,pos.remaining_quantity,exit_price,"stop",i,exit_time,source or ExitSource.FALLBACK_15M)
            self._finalize_partial(pos,reason)
            return
        rate=self.config.maker_fee if self.config.use_maker_exit else self.config.taker_fee; final_gross=(exit_price-pos.entry_price)*pos.quantity if pos.side==Side.LONG else (pos.entry_price-exit_price)*pos.quantity; gross=pos.first_sl_partial_gross_pnl+final_gross; exit_fee=pos.exit_fee+exit_price*pos.quantity*rate
        exit_time=pd.Timestamp(timestamp if timestamp is not None else self.times[i])
        if exit_time < pd.Timestamp(pos.entry_time):
            raise ValueError(f"Exit timestamp {exit_time} precedes entry timestamp {pos.entry_time}")
        if reason == ExitReason.TRAILING_STOP and pos.trailing_activation_time is not None and exit_time < pd.Timestamp(pos.trailing_activation_time):
            raise ValueError(f"Trailing-stop exit timestamp {exit_time} precedes activation timestamp {pos.trailing_activation_time}")
        pos.exit_time=exit_time; pos.exit_index=i; pos.exit_price=exit_price; pos.exit_reason=reason; pos.exit_source=source or (ExitSource.END_OF_DATA if reason==ExitReason.END_OF_DATA else ExitSource.FALLBACK_15M); pos.gross_pnl=gross; pos.exit_fee=exit_fee; pos.fees=pos.entry_fee+exit_fee; pos.net_pnl=gross-pos.fees; pos.gross_r=gross/pos.risk_amount; pos.net_r=pos.net_pnl/pos.risk_amount; move=(exit_price-pos.entry_price) if pos.side==Side.LONG else (pos.entry_price-exit_price); pos.price_r=move/pos.risk
    def _force_close_end(self):
        last=len(self.times)-1
        for pair in self.active_pairs:
            for pos in pair.positions():
                if pos.is_open: self._close_position(pos,last,self.close[last],ExitReason.END_OF_DATA,ExitSource.END_OF_DATA,pd.Timestamp(self.times[last]) + self.entry_delta)
    def _collect_closed_pairs(self,force=False):
        still=[]
        for p in self.active_pairs:
            if force or not p.is_open:
                self.current_equity+=sum(pos.net_pnl for pos in p.positions()); p.equity_after_trade=self.current_equity; self.completed_pairs.append(p)
            else: still.append(p)
        self.active_pairs=still
    def _result_rows_for_pair(self, p):
        positions = list(p.positions())
        if not positions:
            return []
        pos = positions[0]
        return [(pos.side.value.lower(), [pos])]

    def _build_result_row(self, p, row_kind, positions):
        primary = positions[0]
        profile_key_value = getattr(p, "strategy_profile_key", None)
        profile = self.config.strategy_profiles.get(profile_key_value)
        fees = sum(pos.fees for pos in positions)
        gross = sum(pos.gross_pnl for pos in positions)
        net = sum(pos.net_pnl for pos in positions)
        risk_base = sum(pos.risk_amount for pos in positions)
        exit_t = max(pd.Timestamp(pos.exit_time) for pos in positions)
        hold = exit_t - pd.Timestamp(p.strategy_entry_time)
        entry_notional = sum(pos.entry_notional for pos in positions)

        partial_sl = bool(getattr(p, "applied_partial_sl_enabled", False))
        partial_tp = bool(getattr(p, "applied_partial_tp_enabled", False))
        sl1_r = float(getattr(p, "applied_sl1_r", 0.0))
        sl1_pct = float(getattr(p, "applied_sl1_close_pct", 0.0))
        sl2_r = float(getattr(p, "applied_sl2_r", 0.0))
        tp1_r = float(getattr(p, "applied_tp1_r", 0.0))
        tp1_pct = float(getattr(p, "applied_tp1_close_pct", 0.0))
        tp2_r = float(getattr(p, "applied_tp2_r", 0.0))
        stop_mult = float(getattr(p, "applied_stop_loss_multiple", 0.0))
        applied_rr = float(profile.reward_risk_ratio) if profile is not None else np.nan

        if partial_tp:
            winning_trade_r = (tp1_pct/100.0) * tp1_r + (1-tp1_pct/100.0) * tp2_r
        else:
            winning_trade_r = applied_rr if np.isfinite(applied_rr) else np.nan
        expected_profit = winning_trade_r * primary.risk * primary.quantity if np.isfinite(winning_trade_r) else np.nan
        estimated_fees = entry_notional * (
            (self.config.maker_fee if self.config.use_maker_entry else self.config.taker_fee)
            + (self.config.maker_fee if self.config.use_maker_exit else self.config.taker_fee)
        )
        fee_pct = fees / expected_profit * 100 if expected_profit and expected_profit > 0 else np.nan
        estimated_stop_risk = self._estimated_stop_loss(primary) / p.equity_before_trade
        warmup_complete = bool(
            np.isfinite(getattr(p, "adx", np.nan)) and np.isfinite(getattr(p, "bb_width", np.nan))
        )

        row = {
            "pair_id": p.pair_id,
            "trade_id": f"{p.pair_id}-{row_kind}",
            "result_type": row_kind,
            "side": primary.side.value,
            "trade_direction": getattr(p, "trade_direction", primary.side.value),
            "strategy_profile_key": profile_key_value,
            "strategy_profile_run_mode": self.config.strategy_profile_run_mode,
            "applied_reward_risk_ratio": applied_rr,
            "stop_distance_units": stop_mult,
            "partial_tp_enabled": partial_tp,
            "tp1_r": tp1_r,
            "tp1_close_pct": tp1_pct,
            "tp2_r": tp2_r,
            "tp2_close_pct": 100.0 - tp1_pct if partial_tp else 0.0,
            "partial_sl_enabled": partial_sl,
            "sl1_distance_units": sl1_r,
            "sl1_close_pct": sl1_pct,
            "sl2_distance_units": sl2_r,
            "intrabar_partial_tp_ordering": "STOP_FIRST" if self.config.tie_policy == TiePolicy.PESSIMISTIC else "TP1_THEN_TP2_THEN_STOP",
            "trailing_profit_enabled": bool(primary.trailing_enabled),
            "trail_activation_r": profile.trailing_activation_r if profile and profile.trailing_enabled else None,
            "trail_distance_r": profile.trailing_distance_r if profile and profile.trailing_enabled else None,
            "break_even_enabled": bool(primary.be_enabled),
            "break_even_activation_r": profile.break_even_activation_r if profile and profile.break_even_enabled else None,
            "break_even_offset_r": primary.be_offset_r if primary.be_enabled else None,
            "profile_timeout_enabled": bool(getattr(p, "profile_timeout_enabled", False)),
            "profile_timeout_minutes": getattr(p, "profile_timeout_minutes", None),
            "profile_timeout_triggered": bool(getattr(p, "profile_timeout_triggered", False)),
            "profile_timeout_exit_time": getattr(p, "timeout_exit_time", None),
            "configured_account_risk_percentage": self.config.risk_per_leg,
            "estimated_all_in_stop_risk_percentage": estimated_stop_risk,
            "strategy_candle_open_time": p.strategy_candle_open_time,
            "strategy_entry_time": p.strategy_entry_time,
            "strategy_entry_price": p.strategy_entry_price,
            "entry_time": p.strategy_entry_time,
            "entry_price": primary.entry_price,
            "strategy_timeframe_minutes": self.config.strategy_timeframe_minutes,
            "intrabar_timeframe_minutes": self.config.intrabar_timeframe_minutes,
            "atr_period": self.config.atr_period,
            "atr_multiplier": self.config.atr_multiplier,
            "atr_at_entry": primary.atr_at_entry,
            "adx": getattr(p, "adx", np.nan),
            "plus_di": getattr(p, "plus_di", np.nan),
            "minus_di": getattr(p, "minus_di", np.nan),
            "di_spread": getattr(p, "di_spread", np.nan),
            "di_ratio": getattr(p, "di_ratio", np.nan),
            "di_spread_1": getattr(p, "di_spread_1", np.nan),
            "di_spread_3": getattr(p, "di_spread_3", np.nan),
            "di_spread_5": getattr(p, "di_spread_5", np.nan),
            "di_spread_entry_5bar_change": getattr(p, "di_spread_entry_5bar_change", np.nan),
            "bb_middle": getattr(p, "bb_middle", np.nan),
            "bb_upper": getattr(p, "bb_upper", np.nan),
            "bb_lower": getattr(p, "bb_lower", np.nan),
            "bb_width": getattr(p, "bb_width", np.nan),
            "bb_width_pct": getattr(p, "bb_width_pct", np.nan),
            "bb_width_1": getattr(p, "bb_width_1", np.nan),
            "bb_width_3": getattr(p, "bb_width_3", np.nan),
            "bb_width_5": getattr(p, "bb_width_5", np.nan),
            "bb_width_entry_5bar_change": getattr(p, "bb_width_entry_5bar_change", np.nan),
            "bb_width_entry_5bar_change_pct": getattr(p, "bb_width_entry_5bar_change_pct", np.nan),
            "indicator_warmup_complete": warmup_complete,
            "adx_available_at_entry": bool(np.isfinite(getattr(p, "adx", np.nan))),
            "bb_width_available_at_entry": bool(np.isfinite(getattr(p, "bb_width", np.nan))),
            "indicator_warmup_note": "Complete" if warmup_complete else "Indicator warm-up incomplete at entry; missing indicator values are expected until enough historical candles are available.",
            "daily_schedule_enabled": getattr(p, "daily_schedule_enabled", False),
            "scheduled_entry_time": getattr(p, "scheduled_entry_time", None),
            "scheduled_entry_timezone": getattr(p, "scheduled_entry_timezone", None),
            "scheduled_entry_timestamp": getattr(p, "scheduled_entry_timestamp", None),
            "actual_entry_timestamp": getattr(p, "actual_entry_timestamp", p.strategy_entry_time),
            "entry_schedule_status": getattr(p, "entry_schedule_status", None),
            "entry_filter_passed": getattr(p, "entry_filter_passed", True),
            "entry_filter_reason": getattr(p, "entry_filter_reason", "Strategy profile passed"),
            "distance_unit_price": primary.distance_unit,
            "trade_r_price_distance": primary.risk,
            "equity_before_trade": p.equity_before_trade,
            "combined_entry_notional": entry_notional,
            "combined_effective_leverage": entry_notional / p.equity_before_trade,
            "leverage_capped": p.leverage_capped,
            "pair_gross_pnl": gross,
            "pair_total_fees": fees,
            "pair_net_pnl": net,
            "pair_price_r": sum(pos.price_r for pos in positions),
            "pair_gross_account_r": gross / risk_base,
            "pair_fee_account_r": fees / risk_base,
            "pair_net_account_r": net / risk_base,
            "pair_gross_r": gross / risk_base,
            "pair_fee_r": fees / risk_base,
            "pair_net_r": net / risk_base,
            "pair_leg_gross_r_sum": sum(pos.gross_r for pos in positions),
            "pair_leg_net_r_sum": sum(pos.net_r for pos in positions),
            "expected_gross_winning_pair_pnl": expected_profit,
            "estimated_round_trip_fees": estimated_fees,
            "fees_as_percentage_of_expected_winning_profit": fee_pct,
            "equity_after_trade": p.equity_after_trade,
            "exit_time": exit_t,
            "holding_minutes": hold.total_seconds()/60,
            "holding_hours": hold.total_seconds()/3600,
            "holding_bars": max(0, (exit_t-pd.Timestamp(p.strategy_entry_time))/self.entry_delta),
            "holding_time": hold,
            "ambiguous_intrabar": any(pos.ambiguous for pos in positions),
            "ambiguous_candle": any(pos.ambiguous for pos in positions),
            "missing_intrabar_data": any(pos.missing_intrabar_data for pos in positions),
            "di_direction_selection_enabled": self.config.enable_di_direction_selection,
            "di_sizing_direction": getattr(p, "di_sizing_direction", None),
            "sizing_direction": getattr(p, "sizing_direction", None),
            "directional_di": getattr(p, "directional_di", np.nan),
            "opposing_di": getattr(p, "opposing_di", np.nan),
            "plus_di_change": getattr(p, "plus_di_change", np.nan),
            "minus_di_change": getattr(p, "minus_di_change", np.nan),
            "directional_di_change": getattr(p, "directional_di_change", np.nan),
            "opposing_di_change": getattr(p, "opposing_di_change", np.nan),
            "di_spread_change": getattr(p, "di_spread_change", np.nan),
            "di_pressure_state": getattr(p, "di_pressure_state", "UNKNOWN"),
            "di_pressure_lookback": getattr(p, "di_pressure_lookback", self.config.di_pressure_lookback),
            "mean_reversion_enabled": getattr(p, "mean_reversion_enabled", self.config.enable_mean_reversion_analysis),
            "mean_reversion_period": getattr(p, "mean_reversion_period", self.config.mean_reversion_period),
            "mean_price": getattr(p, "mean_price", np.nan),
            "mean_distance_atr": getattr(p, "mean_distance_atr", np.nan),
            "mean_distance_atr_previous": getattr(p, "mean_distance_atr_previous", np.nan),
            "mean_distance_change_atr": getattr(p, "mean_distance_change_atr", np.nan),
            "mean_reversion_state": getattr(p, "mean_reversion_state", "UNKNOWN"),
            "mean_reversion_motion": getattr(p, "mean_reversion_motion", "UNKNOWN"),
            "mean_reversion_alignment": getattr(p, "mean_reversion_alignment", "UNKNOWN"),
            "mean_reversion_di_alignment": getattr(p, "mean_reversion_di_alignment", "UNKNOWN"),
            "mean_reversion_trade_alignment": getattr(p, "mean_reversion_trade_alignment", "UNKNOWN"),
            "mean_reversion_strength": getattr(p, "mean_reversion_strength", -1),
            "mean_reversion_strength_label": getattr(p, "mean_reversion_strength_label", "UNKNOWN"),
            "market_regime_method": getattr(p, "market_regime_method", self.config.market_regime_method),
            "market_regime": getattr(p, "market_regime", None),
            "bull_regime_lookback_days": self.config.bull_regime_lookback_days,
            "bull_regime_return_threshold": self.config.bull_regime_return_threshold,
            "market_regime_return": getattr(p, "market_regime_return", np.nan),
            "bull_regime": getattr(p, "bull_regime", False),
            "entry_atr_pct": getattr(p, "entry_atr_pct", np.nan),
            "entry_rsi": getattr(p, "entry_rsi", np.nan),
            "entry_close_location": getattr(p, "entry_close_location", np.nan),
            "directional_momentum_return_at_entry": getattr(p, "directional_momentum_return", np.nan),
        }
        scheduled = getattr(p, "scheduled_entry_timestamp", None)
        actual = getattr(p, "actual_entry_timestamp", p.strategy_entry_time)
        row["entry_delay_minutes"] = (pd.Timestamp(actual)-pd.Timestamp(scheduled)).total_seconds()/60 if scheduled is not None else 0
        prefix = primary.side.value.lower()
        row.update(self._pos_cols(prefix, primary))
        row.update(self._partial_sl_cols(prefix, primary))
        return row

    def results_frame(self):
        rows=[]
        for p in self.completed_pairs:
            for row_kind,positions in self._result_rows_for_pair(p):
                rows.append(self._build_result_row(p,row_kind,positions))
        frame=pd.DataFrame(rows)
        if not frame.empty:
            entry = pd.to_datetime(frame["entry_time"], utc=True)
            frame["exit_time_before_entry"] = pd.to_datetime(frame["exit_time"], utc=True) < entry
            trailing_before_activation = pd.Series(False, index=frame.index)
            activated_after_entry = pd.Series(False, index=frame.index)
            for prefix in ("long", "short"):
                reason_col=f"{prefix}_exit_reason"; exit_col=f"{prefix}_exit_time"; activation_col=f"{prefix}_trailing_activation_time"
                if not {reason_col, exit_col, activation_col}.issubset(frame.columns):
                    continue
                leg_exit=pd.to_datetime(frame[exit_col], utc=True); activation=pd.to_datetime(frame[activation_col], utc=True)
                trailing_before_activation |= frame[reason_col].eq(ExitReason.TRAILING_STOP.value) & activation.notna() & (leg_exit < activation)
                activated_after_entry |= activation.notna() & (activation > entry)
            frame["trailing_exit_before_activation"] = trailing_before_activation
            frame["zero_holding_after_trailing_activation"] = activated_after_entry & frame["holding_minutes"].eq(0)
            frame["timestamp_validation_failed"] = frame[["exit_time_before_entry", "trailing_exit_before_activation", "zero_holding_after_trailing_activation"]].any(axis=1)
            frame["signals_evaluated"] = self.signals_evaluated
            frame["signals_skipped_by_adx"] = sum(("ADX unavailable" in str(x.get("entry_filter_reason", x.get("adx_filter_reason", "")))) or str(x.get("entry_filter_reason", x.get("adx_filter_reason", ""))).startswith("ADX ") for x in self.skipped_signals)
            frame["signals_skipped_by_filters"] = len(self.skipped_signals)
            frame["signals_traded"] = len(frame)
        frame.attrs["skipped_signals"] = self.skipped_signals
        frame.attrs["skipped_daily_entries"] = self.skipped_daily_entries
        frame.attrs["daily_schedule_stats"] = {"scheduled_entry_opportunities": self.daily_entry_opportunities, "trades_opened_on_schedule": self.daily_entries_on_schedule, "scheduled_entries_opened_next_available": self.daily_entries_next_available}
        return frame

    def telemetry_frame(self):
        from crypto_strategy_lab.telemetry import TELEMETRY_COLUMNS
        return pd.DataFrame(self.telemetry_rows, columns=TELEMETRY_COLUMNS)

    def _num(self, arr, i):
        value = arr[i]
        return float(value) if np.isfinite(value) else np.nan

    def _unrealized(self, pos, close):
        if not pos.is_open:
            return 0.0
        active_quantity=pos.remaining_quantity if (pos.partial_tp_enabled or pos.partial_sl_enabled) else pos.quantity
        gross = (close - pos.entry_price) * active_quantity if pos.side == Side.LONG else (pos.entry_price - close) * active_quantity
        return float(gross - pos.fees)

    def _record_active_telemetry(self, i):
        if not self.config.enable_trade_telemetry:
            return
        for pair in self.active_pairs:
            self._record_pair_telemetry(pair, i)

    def _record_pair_telemetry(self, pair, i):
        if not self.config.enable_trade_telemetry:
            return
        ts = pd.Timestamp(self.times[i]) + self.entry_delta
        entry = pd.Timestamp(pair.strategy_entry_time)
        if ts < entry or not pair.is_open:
            return
        elapsed = (ts - entry).total_seconds() / 60
        if elapsed < 0 or elapsed % self.config.telemetry_interval_minutes != 0:
            return
        if self.telemetry_rows and self.telemetry_rows[-1].get("pair_id") == pair.pair_id and pd.Timestamp(self.telemetry_rows[-1].get("timestamp")) == ts:
            return
        close = float(self.close[i]); high = float(self.high[i]); low = float(self.low[i])
        long_open = pair.long.is_open if pair.long is not None else False; short_open = pair.short.is_open if pair.short is not None else False
        long_pnl = self._unrealized(pair.long, close) if pair.long is not None else 0.0; short_pnl = self._unrealized(pair.short, close) if pair.short is not None else 0.0
        def distances(pos, is_long):
            if not pos.is_open:
                return (np.nan, np.nan, np.nan, np.nan)
            sl_d = close - pos.sl if is_long else pos.sl - close
            if pos.r_step_trailing_enabled and pos.r_step_maximum_r == 0:
                tp_d = np.nan
            else:
                tp_d = pos.tp - close if is_long else close - pos.tp
            return (float(sl_d), float(tp_d), float(sl_d / pos.risk) if pos.risk else np.nan, float(tp_d / pos.risk) if pos.risk else np.nan)
        lsl, ltp, lslr, ltpr = distances(pair.long, True) if pair.long is not None else (np.nan, np.nan, np.nan, np.nan); ssl, stp, sslr, stpr = distances(pair.short, False) if pair.short is not None else (np.nan, np.nan, np.nan, np.nan)
        row={"pair_id":pair.pair_id,"long_leg_id":f"{pair.pair_id}_LONG","short_leg_id":f"{pair.pair_id}_SHORT","timestamp":ts,"elapsed_minutes":elapsed,"elapsed_strategy_bars":int(elapsed / self.config.strategy_timeframe_minutes),"close":close,"high":high,"low":low,"atr":self._num(self.atr_values,i),"adx":self._num(self.adx_values,i),"plus_di":self._num(self.plus_di_values,i),"minus_di":self._num(self.minus_di_values,i),"di_spread":self._num(self.di_spread,i),"di_ratio":self._num(self.di_ratio,i),"bb_middle":self._num(self.bb_middle,i),"bb_upper":self._num(self.bb_upper,i),"bb_lower":self._num(self.bb_lower,i),"bb_width":self._num(self.bb_width,i),"bb_width_pct":self._num(self.bb_width_pct,i),"long_is_open":long_open,"short_is_open":short_open,"long_unrealized_pnl":long_pnl,"short_unrealized_pnl":short_pnl,"pair_unrealized_pnl":long_pnl+short_pnl,"long_distance_to_sl":lsl,"long_distance_to_tp":ltp,"short_distance_to_sl":ssl,"short_distance_to_tp":stp,"long_distance_to_sl_r":lslr,"long_distance_to_tp_r":ltpr,"short_distance_to_sl_r":sslr,"short_distance_to_tp_r":stpr,"long_current_sl":pair.long.sl if pair.long is not None and pair.long.is_open else np.nan,"short_current_sl":pair.short.sl if pair.short is not None and pair.short.is_open else np.nan,"long_tp":(np.nan if pair.long is not None and pair.long.r_step_trailing_enabled and pair.long.r_step_maximum_r==0 else (pair.long.tp if pair.long is not None else np.nan)),"short_tp":pair.short.tp if pair.short is not None else np.nan}
        for prefix, pos, is_long in (("long", pair.long, True), ("short", pair.short, False)):
            enabled = bool(pos and pos.trailing_enabled)
            active = bool(pos and pos.trailing_active)
            row.update({f"{prefix}_trailing_enabled":enabled, f"{prefix}_trailing_active":active, f"{prefix}_trailing_activation_price":pos.trailing_activation_price if enabled else np.nan, f"{prefix}_current_trailing_stop":pos.trailing_stop if active else np.nan, f"{prefix}_current_active_stop":pos.sl if pos and pos.is_open else np.nan, f"{prefix}_{'highest' if is_long else 'lowest'}_price_since_entry":pos.favourable_price if enabled else np.nan, f"{prefix}_distance_to_activation_r":((pos.trailing_activation_price-close) if is_long else (close-pos.trailing_activation_price))/pos.risk if enabled and pos.risk else np.nan, f"{prefix}_distance_to_trailing_stop_r":((close-pos.trailing_stop) if is_long else (pos.trailing_stop-close))/pos.risk if active and pos.risk else np.nan, f"{prefix}_unrealized_profit_r":((close-pos.entry_price) if is_long else (pos.entry_price-close))/pos.risk if pos and pos.is_open and pos.risk else np.nan, f"{prefix}_original_quantity":pos.original_quantity if pos and pos.partial_tp_enabled else np.nan, f"{prefix}_remaining_quantity":pos.remaining_quantity if pos and pos.partial_tp_enabled else np.nan, f"{prefix}_tp1_hit":bool(pos and pos.tp1_hit), f"{prefix}_tp2_hit":bool(pos and pos.tp2_hit), f"{prefix}_tp1_price":pos.tp1_price if pos and pos.partial_tp_enabled else np.nan, f"{prefix}_tp2_price":pos.tp2_price if pos and pos.partial_tp_enabled else np.nan, f"{prefix}_realized_pnl":pos.realized_pnl-pos.entry_fee if pos and pos.partial_tp_enabled else 0.0, f"{prefix}_total_current_pnl":(pos.realized_pnl-pos.entry_fee+self._unrealized(pos,close)+pos.fees if pos and pos.partial_tp_enabled else row.get(f"{prefix}_unrealized_pnl",0.0))})
        self.telemetry_rows.append(row)

    def _estimated_stop_loss(self,pos):
        if pos.partial_sl_enabled:
            first=pos.sl1_quantity; remainder=pos.original_quantity-first
            sl1_exit=pos.sl1_price*(1-self.config.slippage if pos.side==Side.LONG else 1+self.config.slippage)
            sl2_exit=pos.sl2_price*(1-self.config.slippage if pos.side==Side.LONG else 1+self.config.slippage)
            gross=((pos.entry_price-sl1_exit)*first+(pos.entry_price-sl2_exit)*remainder) if pos.side==Side.LONG else ((sl1_exit-pos.entry_price)*first+(sl2_exit-pos.entry_price)*remainder)
            exit_rate=self.config.maker_fee if self.config.use_maker_exit else self.config.taker_fee
            return gross+pos.entry_fee+(sl1_exit*first+sl2_exit*remainder)*exit_rate
        if pos.side==Side.LONG:
            stop_exit=pos.sl*(1-self.config.slippage); gross=(pos.entry_price-stop_exit)*pos.quantity
        else:
            stop_exit=pos.sl*(1+self.config.slippage); gross=(stop_exit-pos.entry_price)*pos.quantity
        exit_rate=self.config.maker_fee if self.config.use_maker_exit else self.config.taker_fee
        return gross + pos.entry_fee + stop_exit*pos.quantity*exit_rate
    def _partial_sl_cols(self,prefix,pos):
        return {
            f"{prefix}_partial_sl_enabled":pos.partial_sl_enabled,
            f"{prefix}_sl1_price":pos.sl1_price,
            f"{prefix}_sl2_price":pos.sl2_price,
            f"{prefix}_sl1_quantity":pos.sl1_quantity,
            f"{prefix}_sl1_hit":pos.sl1_hit,
            f"{prefix}_sl1_exit_time":pos.sl1_exit_time,
            f"{prefix}_sl1_exit_price":pos.sl1_exit_price,
            f"{prefix}_sl1_gross_pnl":pos.sl1_gross_pnl,
            f"{prefix}_sl1_fees":pos.sl1_fees,
            f"{prefix}_sl1_net_pnl":pos.sl1_net_pnl,
        }

    def _sr_event_labels(self,pos):
        """Entry-time S/R event classification derived from stored Position snapshot fields (no recalculation)."""
        if not self.config.enable_support_resistance_analysis:
            return None
        return self._classify_sr_event_labels(pos.sr_near_support,pos.sr_near_resistance,pos.sr_support_held,pos.sr_resistance_held,pos.sr_support_state,pos.sr_resistance_state)

    @staticmethod
    def _classify_sr_event_labels(near_support,near_resistance,support_held,resistance_held,support_state,resistance_state):
        labels=[]
        if near_support: labels.append("NEAR_SUPPORT")
        if near_resistance: labels.append("NEAR_RESISTANCE")
        if support_held: labels.append("SUPPORT_BOUNCE")
        if resistance_held: labels.append("RESISTANCE_REJECTION")
        if str(resistance_state)=="RESISTANCE_BROKEN": labels.append("RESISTANCE_BREAKOUT")
        if str(support_state)=="SUPPORT_BROKEN": labels.append("SUPPORT_BREAKDOWN")
        if not labels: labels.append("NO_NEARBY_SR")
        return labels

    def _pos_cols(self,prefix,pos):
        est_stop_loss=self._estimated_stop_loss(pos)
        cols={f"{prefix}_original_quantity":pos.original_quantity if pos.partial_tp_enabled else pos.quantity,f"{prefix}_remaining_quantity":pos.remaining_quantity if pos.partial_tp_enabled else 0.0,f"{prefix}_tp1_quantity":pos.tp1_quantity if pos.partial_tp_enabled else None,f"{prefix}_tp2_quantity":pos.tp2_quantity if pos.partial_tp_enabled else None,f"{prefix}_tp1_price":pos.tp1_price,f"{prefix}_tp2_price":pos.tp2_price,f"{prefix}_sl_price":pos.original_sl,f"{prefix}_tp1_hit":pos.tp1_hit,f"{prefix}_tp1_exit_time":pos.tp1_exit_time,f"{prefix}_tp1_exit_price":pos.tp1_exit_price,f"{prefix}_tp1_gross_pnl":pos.tp1_gross_pnl,f"{prefix}_tp1_fees":pos.tp1_fees,f"{prefix}_tp1_net_pnl":pos.tp1_net_pnl,f"{prefix}_tp2_hit":pos.tp2_hit,f"{prefix}_tp2_exit_time":pos.tp2_exit_time,f"{prefix}_tp2_exit_price":pos.tp2_exit_price,f"{prefix}_tp2_gross_pnl":pos.tp2_gross_pnl,f"{prefix}_tp2_fees":pos.tp2_fees,f"{prefix}_tp2_net_pnl":pos.tp2_net_pnl,f"{prefix}_stop_exit_time":pos.stop_exit_time,f"{prefix}_stop_exit_price":pos.stop_exit_price,f"{prefix}_stop_exit_quantity":pos.stop_exit_quantity,f"{prefix}_stop_gross_pnl":pos.stop_gross_pnl,f"{prefix}_stop_fees":pos.stop_fees,f"{prefix}_stop_net_pnl":pos.stop_net_pnl,f"{prefix}_total_gross_pnl":pos.gross_pnl,f"{prefix}_total_net_pnl":pos.net_pnl,f"{prefix}_final_exit_reason":pos.final_exit_reason or (pos.exit_reason.value if pos.exit_reason else None),f"{prefix}_trade_r_price_distance":pos.risk,f"{prefix}_distance_unit_price":pos.distance_unit,f"{prefix}_trailing_enabled":pos.trailing_enabled,f"{prefix}_trailing_activated":pos.trailing_active,f"{prefix}_trailing_activation_time":pos.trailing_activation_time,f"{prefix}_trailing_activation_price":pos.trailing_activation_price,f"{prefix}_{'highest' if pos.side==Side.LONG else 'lowest'}_favourable_price":pos.favourable_price if pos.trailing_active else None,f"{prefix}_final_trailing_stop":pos.trailing_stop,f"{prefix}_final_active_stop":pos.final_active_stop,f"{prefix}_trailing_exit_price":pos.trailing_exit_price,f"{prefix}_trailing_profit_r":pos.trailing_profit_r,f"{prefix}_entry_price":pos.entry_price,f"{prefix}_quantity":pos.quantity,f"{prefix}_uncapped_quantity":pos.uncapped_quantity,f"{prefix}_entry_notional":pos.entry_notional,f"{prefix}_effective_leverage":pos.effective_leverage,f"{prefix}_risk_amount":pos.risk_amount,f"{prefix}_configured_account_risk_percentage":self.config.risk_per_leg,f"{prefix}_estimated_all_in_stop_risk_percentage":est_stop_loss/pos.risk_amount*self.config.risk_per_leg if pos.risk_amount else 0,f"{prefix}_original_sl":pos.original_sl,f"{prefix}_current_sl":pos.sl,f"{prefix}_sl":pos.sl,f"{prefix}_tp":(None if pos.r_step_trailing_enabled and pos.r_step_maximum_r==0 else pos.tp),f"{prefix}_be_enabled":pos.be_enabled,f"{prefix}_be_triggered":pos.be_triggered,f"{prefix}_be_trigger_time":pos.be_trigger_time,f"{prefix}_be_triggered_by_side":pos.be_triggered_by_side.value if pos.be_triggered_by_side else None,f"{prefix}_be_mode":pos.be_mode,f"{prefix}_be_offset_r":pos.be_offset_r,f"{prefix}_be_stop_price":pos.be_stop_price,f"{prefix}_be_exit_reason":pos.be_exit_reason.value if pos.be_exit_reason else None,f"{prefix}_be_same_candle_ambiguous":pos.be_same_candle_ambiguous,f"{prefix}_exit_time":pos.exit_time,f"{prefix}_exit_price":pos.exit_price,f"{prefix}_exit_reason":pos.exit_reason.value if pos.exit_reason else None,f"{prefix}_exit_source":pos.exit_source.value if pos.exit_source else None,f"{prefix}_fallback_reason":pos.fallback_reason,f"{prefix}_entry_fee":pos.entry_fee,f"{prefix}_exit_fee":pos.exit_fee,f"{prefix}_total_fees":pos.fees,f"{prefix}_fees":pos.fees,f"{prefix}_gross_pnl":pos.gross_pnl,f"{prefix}_net_pnl":pos.net_pnl,f"{prefix}_price_r":pos.price_r,f"{prefix}_account_r":pos.net_r,f"{prefix}_gross_r":pos.gross_r,f"{prefix}_net_r":pos.net_r}
        cols.update({f"{prefix}_atr_checkpoint_enabled":pos.atr_checkpoint_extension_enabled or pos.atr_checkpoint_count>0,f"{prefix}_atr_checkpoint_count":pos.atr_checkpoint_count,f"{prefix}_atr_checkpoint_pass_count":pos.atr_checkpoint_pass_count,f"{prefix}_atr_checkpoint_fail_count":pos.atr_checkpoint_fail_count,f"{prefix}_atr_checkpoint_last_time":pos.atr_checkpoint_last_time,f"{prefix}_atr_checkpoint_last_r":pos.atr_checkpoint_last_r,f"{prefix}_atr_checkpoint_last_di_spread":pos.atr_checkpoint_last_di_spread,f"{prefix}_atr_checkpoint_last_bb_width":pos.atr_checkpoint_last_bb_width,f"{prefix}_atr_checkpoint_last_passed":pos.atr_checkpoint_last_passed,f"{prefix}_atr_checkpoint_initial_tp":pos.atr_checkpoint_initial_tp,f"{prefix}_atr_checkpoint_final_tp_r":pos.atr_checkpoint_final_tp_r,f"{prefix}_atr_checkpoint_profit_lock_r":pos.atr_checkpoint_profit_lock_r})
        cols.update({f"{prefix}_r_step_trailing_enabled":pos.r_step_trailing_enabled,f"{prefix}_r_step_trailing_active":pos.r_step_trailing_active,f"{prefix}_r_step_checkpoint_count":pos.r_step_checkpoint_count,f"{prefix}_r_step_last_checkpoint_r":pos.r_step_last_checkpoint_r,f"{prefix}_r_step_last_checkpoint_time":pos.r_step_last_checkpoint_time,f"{prefix}_r_step_locked_r":pos.r_step_locked_r,f"{prefix}_r_step_initial_tp":pos.r_step_initial_tp,f"{prefix}_r_step_activation_partial_taken":pos.r_step_activation_partial_taken,f"{prefix}_r_step_activation_close_pct":pos.r_step_activation_close_pct,f"{prefix}_r_step_activation_quantity":pos.r_step_activation_quantity,f"{prefix}_r_step_runner_quantity":pos.r_step_runner_quantity})
        cols.update({f"{prefix}_sr_nearest_support":pos.sr_nearest_support,f"{prefix}_sr_nearest_resistance":pos.sr_nearest_resistance,f"{prefix}_sr_support_distance_atr":pos.sr_support_distance_atr,f"{prefix}_sr_resistance_distance_atr":pos.sr_resistance_distance_atr,f"{prefix}_sr_support_distance_price":pos.sr_support_distance_price,f"{prefix}_sr_resistance_distance_price":pos.sr_resistance_distance_price,f"{prefix}_sr_near_support":pos.sr_near_support,f"{prefix}_sr_near_resistance":pos.sr_near_resistance,f"{prefix}_sr_inside_support_zone":pos.sr_inside_support_zone,f"{prefix}_sr_inside_resistance_zone":pos.sr_inside_resistance_zone,f"{prefix}_sr_location":pos.sr_location,f"{prefix}_sr_trade_location_rating":pos.sr_trade_location_rating,f"{prefix}_sr_room_in_direction_atr":pos.sr_room_in_direction_atr})
        cols.update({f"{prefix}_sr_support_state":pos.sr_support_state, f"{prefix}_sr_resistance_state":pos.sr_resistance_state, f"{prefix}_sr_support_tested":pos.sr_support_tested, f"{prefix}_sr_resistance_tested":pos.sr_resistance_tested, f"{prefix}_sr_support_held":pos.sr_support_held, f"{prefix}_sr_resistance_held":pos.sr_resistance_held, f"{prefix}_sr_support_rejection_atr":pos.sr_support_rejection_atr, f"{prefix}_sr_resistance_rejection_atr":pos.sr_resistance_rejection_atr, f"{prefix}_sr_support_test_count":pos.sr_support_test_count, f"{prefix}_sr_resistance_test_count":pos.sr_resistance_test_count, f"{prefix}_sr_bars_since_support_test":pos.sr_bars_since_support_test, f"{prefix}_sr_bars_since_resistance_test":pos.sr_bars_since_resistance_test, f"{prefix}_sr_support_last_test_index":pos.sr_support_last_test_index, f"{prefix}_sr_resistance_last_test_index":pos.sr_resistance_last_test_index, f"{prefix}_sr_support_last_test_time":pos.sr_support_last_test_time, f"{prefix}_sr_resistance_last_test_time":pos.sr_resistance_last_test_time, f"{prefix}_sr_confirmation_rating":pos.sr_confirmation_rating})
        cols.update({f"{prefix}_sr_support_zone_low":pos.sr_support_zone_low, f"{prefix}_sr_support_zone_high":pos.sr_support_zone_high, f"{prefix}_sr_resistance_zone_low":pos.sr_resistance_zone_low, f"{prefix}_sr_resistance_zone_high":pos.sr_resistance_zone_high, f"{prefix}_sr_level_price":pos.sr_level_price, f"{prefix}_sr_zone_low":pos.sr_zone_low, f"{prefix}_sr_zone_high":pos.sr_zone_high})
        sr_event_labels=self._sr_event_labels(pos)
        cols.update({
            f"{prefix}_sr_support_bounce":bool(sr_event_labels and "SUPPORT_BOUNCE" in sr_event_labels),
            f"{prefix}_sr_resistance_rejection":bool(sr_event_labels and "RESISTANCE_REJECTION" in sr_event_labels),
            f"{prefix}_sr_resistance_breakout":bool(sr_event_labels and "RESISTANCE_BREAKOUT" in sr_event_labels),
            f"{prefix}_sr_support_breakdown":bool(sr_event_labels and "SUPPORT_BREAKDOWN" in sr_event_labels),
            f"{prefix}_sr_context":"|".join(sr_event_labels) if sr_event_labels else None,
        })
        return cols
