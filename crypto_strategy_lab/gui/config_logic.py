"""Pure helpers for the current GUI configuration contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from crypto_strategy_lab.config import (
    BacktestConfig,
    DailyEntryMissedPolicy,
    EntryMode,
    IntrabarMissingPolicy,
    RiskMode,
    TiePolicy,
)
from crypto_strategy_lab.strategy_profiles import default_profiles, normalize_profiles, profiles_to_dict

CONFIG_VERSION = 2

DEFAULT_GUI_CONFIG: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "market_symbol": "XRPUSDT",
    "strategy_profile_run_mode": "COMBINED_SHARED_CAPITAL",
    "strategy_profiles": profiles_to_dict(default_profiles()),
    "input_csv": "C:/CryptoBots/Binance Market Data/futures/usdm/BTCUSDT_15m.csv",
    "intrabar_csv": "C:/CryptoBots/Binance Market Data/futures/usdm/BTCUSDT_1m.csv",
    "output_dir": "output",
    "run_name": "",
    "entry_mode": "WAIT_UNTIL_CLOSED",
    "entry_interval": 1,
    "max_active_pairs": 1,
    "tie_policy": "PESSIMISTIC",
    "risk_mode": "ATR",
    "atr_period": 14,
    "atr_multiplier": 1.0,
    "adx_period": 14,
    "bb_period": 20,
    "bb_stddevs": 2.0,
    "percent_r": 0.002,
    "fixed_r": 100.0,
    "initial_equity": 1000.0,
    "risk_per_leg": 0.01,
    "maker_fee": 0.0002,
    "taker_fee": 0.0005,
    "use_maker_entry": False,
    "use_maker_exit": False,
    "slippage": 0.0005,
    "strategy_timeframe_minutes": 15,
    "intrabar_timeframe_minutes": 1,
    "use_intrabar_data": True,
    "trading_start_date": None,
    "trading_end_date": None,
    "max_effective_leverage_per_leg": "3.0",
    "max_combined_effective_leverage": "5.0",
    "intrabar_missing_policy": "WARN_AND_USE_15M",
    "zero_cost_comparison": False,
    "enable_di_direction_selection": True,
    "enable_di_pressure_analysis": True,
    "di_pressure_lookback": 3,
    "enable_mean_reversion_analysis": True,
    "mean_reversion_period": 20,
    "enable_support_resistance_analysis": False,
    "sr_pivot_left": 5,
    "sr_pivot_right": 5,
    "sr_lookback_bars": 200,
    "sr_zone_width_atr": 0.5,
    "sr_near_distance_atr": 0.75,
    "enable_sr_hold_confirmation": False,
    "sr_hold_confirmation_bars": 3,
    "sr_hold_confirmation_atr": 0.25,
    "sr_break_tolerance_atr": 0.25,
    "sr_break_basis": "CLOSE",
    "sr_filter_mode": "ANALYSIS_ONLY",
    "sr_long_avoid_near_resistance": False,
    "sr_long_require_near_support": False,
    "sr_long_block_broken_support": False,
    "sr_long_min_room_to_resistance_atr": 0.0,
    "sr_short_avoid_near_support": False,
    "sr_short_require_near_resistance": False,
    "sr_short_block_broken_resistance": False,
    "sr_short_min_room_to_support_atr": 0.0,
    "market_regime_method": "BTC_STRUCTURAL",
    "structural_regime_sma_days": 200,
    "structural_regime_slope_lookback_days": 30,
    "structural_regime_benchmark_csv": None,
    "bull_regime_lookback_days": 90,
    "bull_regime_return_threshold": 0.20,
    "enable_daily_entry_schedule": False,
    "daily_entry_time": "00:00",
    "daily_entry_timezone": "UTC",
    "daily_entry_missed_policy": "SKIP_DAY",
    "analysis_level": "STANDARD",
    "enable_trade_telemetry": False,
    "save_full_telemetry_csv": False,
    "save_trade_journey_summary": False,
    "save_trade_journey_charts": False,
    "telemetry_interval_minutes": 15,
    "enable_indicator_lifecycle_analysis": False,
    "lifecycle_phases": 4,
    "lifecycle_early_checkpoints": [15, 30, 60],
    "lifecycle_minimum_bucket_sample": 20,
    "create_lifecycle_charts": False,
    "lifecycle_flat_pattern_threshold_pct": 5.0,
    "save_feature_analysis_reports": False,
    "save_indicator_analysis_reports": True,
    "create_standard_charts": True,
}


def default_gui_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_GUI_CONFIG)


def parse_percentage(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("percentage is required")
    if text.endswith("%"):
        return float(text[:-1].strip()) / 100.0
    return float(text)


def format_percentage(value: float, decimals: int = 4) -> str:
    text = f"{value * 100:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{text}%"


def theoretical_break_even(sl_mult: float, tp_mult: float) -> float:
    """Display helper retained for historical report math; not a config input."""
    winning = tp_mult - sl_mult
    losing_abs = abs(-2 * sl_mult)
    denom = winning + losing_abs
    return losing_abs / denom if denom > 0 else 1.0


def _merged_current(values: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(values) - set(DEFAULT_GUI_CONFIG))
    if unknown:
        raise ValueError(f"Unknown/retired configuration settings: {', '.join(unknown)}")
    merged = {**default_gui_config(), **values}
    if int(merged.get("config_version", -1)) != CONFIG_VERSION:
        raise ValueError(f"Configuration version {CONFIG_VERSION} is required. Recreate the configuration in the current GUI.")
    merged["strategy_profiles"] = profiles_to_dict(normalize_profiles(merged["strategy_profiles"]))
    return merged


def validate_config_values(values: dict[str, Any], require_paths: bool = True) -> list[str]:
    try:
        values = _merged_current(values)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    if require_paths:
        strategy_path = values.get("input_csv")
        if not Path(strategy_path or "").is_file():
            errors.append("Strategy CSV must exist.")
        intrabar_path = values.get("intrabar_csv")
        if values.get("use_intrabar_data") and intrabar_path and not Path(intrabar_path).is_file():
            errors.append("Intrabar CSV must exist when enabled.")
        if not str(values.get("output_dir", "")).strip():
            errors.append("Output folder is required.")
    positive = (
        ("initial_equity", "Starting equity"), ("risk_per_leg", "Risk per trade"),
        ("atr_period", "ATR period"), ("atr_multiplier", "ATR multiplier"),
        ("adx_period", "ADX period"), ("bb_period", "BB period"), ("bb_stddevs", "BB standard deviations"),
        ("percent_r", "Percentage R"), ("fixed_r", "Fixed R"),
        ("strategy_timeframe_minutes", "Strategy timeframe"), ("intrabar_timeframe_minutes", "Intrabar timeframe"),
        ("entry_interval", "Entry interval"), ("max_active_pairs", "Maximum active pairs"),
    )
    for key, label in positive:
        try:
            if float(values[key]) <= 0:
                errors.append(f"{label} must be > 0.")
        except (TypeError, ValueError, KeyError):
            errors.append(f"{label} must be numeric.")
    try:
        if float(values["risk_per_leg"]) >= 1:
            errors.append("Risk per trade must be < 100%.")
    except (TypeError, ValueError):
        pass
    if values["entry_mode"] not in (EntryMode.WAIT_UNTIL_CLOSED.value, EntryMode.EVERY_N_CANDLES.value):
        errors.append("Invalid entry mode.")
    if values["risk_mode"] not in [e.value for e in RiskMode]:
        errors.append("Invalid risk mode.")
    if values["tie_policy"] not in (TiePolicy.PESSIMISTIC.value, TiePolicy.OPTIMISTIC.value):
        errors.append("Invalid tie policy.")
    if values["intrabar_missing_policy"] not in [e.value for e in IntrabarMissingPolicy]:
        errors.append("Invalid intrabar missing-data policy.")
    if values["daily_entry_missed_policy"] not in [e.value for e in DailyEntryMissedPolicy]:
        errors.append("Invalid daily entry missed policy.")
    for key, label in (("maker_fee", "Maker fee"), ("taker_fee", "Taker fee"), ("slippage", "Slippage")):
        try:
            if float(values[key]) < 0:
                errors.append(f"{label} must be >= 0.")
        except (TypeError, ValueError):
            errors.append(f"{label} must be numeric.")
    try:
        strategy = int(values["strategy_timeframe_minutes"]); intrabar = int(values["intrabar_timeframe_minutes"])
        if values["use_intrabar_data"] and intrabar >= strategy:
            errors.append("Intrabar timeframe must be smaller than strategy timeframe.")
    except (TypeError, ValueError):
        pass
    if values["strategy_profile_run_mode"] not in ("ISOLATED_PROFILES", "COMBINED_SHARED_CAPITAL", "BOTH"):
        errors.append("Invalid Strategy Profile test mode.")
    if values["market_regime_method"] not in ("ASSET_RETURN", "BTC_STRUCTURAL", "ASSET_STRUCTURAL"):
        errors.append("Invalid market regime method.")
    if values["sr_filter_mode"] not in ("ANALYSIS_ONLY", "APPLY_ENTRY_RULES"):
        errors.append("Invalid support/resistance usage mode.")
    if str(values["sr_break_basis"]).upper() not in ("CLOSE", "WICK"):
        errors.append("Invalid support/resistance break basis.")
    try:
        if not 1 <= int(values["di_pressure_lookback"]) <= 100:
            errors.append("DI pressure lookback must be between 1 and 100.")
    except (TypeError, ValueError):
        errors.append("DI pressure lookback must be a whole number.")
    try:
        if int(values["mean_reversion_period"]) <= 0:
            errors.append("Mean reversion period must be greater than zero.")
    except (TypeError, ValueError):
        errors.append("Mean reversion period must be a whole number.")
    return errors


def build_backtest_config(values: dict[str, Any], require_paths: bool = True) -> BacktestConfig:
    merged = _merged_current(values)
    errors = validate_config_values(merged, require_paths=require_paths)
    if errors:
        raise ValueError("\n".join(errors))
    return BacktestConfig(
        input_csv=Path(merged["input_csv"]),
        intrabar_csv=Path(merged["intrabar_csv"]) if merged.get("intrabar_csv") else None,
        output_dir=Path(merged["output_dir"]),
        strategy_profile_run_mode=str(merged["strategy_profile_run_mode"]),
        strategy_profiles=merged["strategy_profiles"],
        entry_mode=EntryMode(merged["entry_mode"]),
        entry_interval=int(merged["entry_interval"]),
        enable_di_direction_selection=bool(merged["enable_di_direction_selection"]),
        enable_di_pressure_analysis=bool(merged["enable_di_pressure_analysis"]),
        di_pressure_lookback=int(merged["di_pressure_lookback"]),
        enable_mean_reversion_analysis=bool(merged["enable_mean_reversion_analysis"]),
        mean_reversion_period=int(merged["mean_reversion_period"]),
        enable_support_resistance_analysis=bool(merged["enable_support_resistance_analysis"]),
        sr_pivot_left=int(merged["sr_pivot_left"]), sr_pivot_right=int(merged["sr_pivot_right"]),
        sr_lookback_bars=int(merged["sr_lookback_bars"]), sr_zone_width_atr=float(merged["sr_zone_width_atr"]),
        sr_near_distance_atr=float(merged["sr_near_distance_atr"]),
        enable_sr_hold_confirmation=bool(merged["enable_sr_hold_confirmation"]),
        sr_hold_confirmation_bars=int(merged["sr_hold_confirmation_bars"]),
        sr_hold_confirmation_atr=float(merged["sr_hold_confirmation_atr"]),
        sr_break_tolerance_atr=float(merged["sr_break_tolerance_atr"]), sr_break_basis=str(merged["sr_break_basis"]).upper(),
        sr_filter_mode=str(merged["sr_filter_mode"]),
        sr_long_avoid_near_resistance=bool(merged["sr_long_avoid_near_resistance"]),
        sr_long_require_near_support=bool(merged["sr_long_require_near_support"]),
        sr_long_block_broken_support=bool(merged["sr_long_block_broken_support"]),
        sr_long_min_room_to_resistance_atr=float(merged["sr_long_min_room_to_resistance_atr"]),
        sr_short_avoid_near_support=bool(merged["sr_short_avoid_near_support"]),
        sr_short_require_near_resistance=bool(merged["sr_short_require_near_resistance"]),
        sr_short_block_broken_resistance=bool(merged["sr_short_block_broken_resistance"]),
        sr_short_min_room_to_support_atr=float(merged["sr_short_min_room_to_support_atr"]),
        market_regime_method=str(merged["market_regime_method"]),
        structural_regime_sma_days=int(merged["structural_regime_sma_days"]),
        structural_regime_slope_lookback_days=int(merged["structural_regime_slope_lookback_days"]),
        structural_regime_benchmark_csv=Path(merged["structural_regime_benchmark_csv"]) if merged.get("structural_regime_benchmark_csv") else None,
        bull_regime_lookback_days=int(merged["bull_regime_lookback_days"]),
        bull_regime_return_threshold=float(merged["bull_regime_return_threshold"]),
        enable_daily_entry_schedule=bool(merged["enable_daily_entry_schedule"]),
        daily_entry_time=str(merged["daily_entry_time"]), daily_entry_timezone=str(merged["daily_entry_timezone"]),
        daily_entry_missed_policy=DailyEntryMissedPolicy(merged["daily_entry_missed_policy"]),
        max_active_pairs=int(merged["max_active_pairs"]), tie_policy=TiePolicy(merged["tie_policy"]),
        risk_mode=RiskMode(merged["risk_mode"]), atr_period=int(merged["atr_period"]), atr_multiplier=float(merged["atr_multiplier"]),
        adx_period=int(merged["adx_period"]), bb_period=int(merged["bb_period"]), bb_stddevs=float(merged["bb_stddevs"]),
        percent_r=float(merged["percent_r"]), fixed_r=float(merged["fixed_r"]), initial_equity=float(merged["initial_equity"]),
        risk_per_leg=float(merged["risk_per_leg"]), maker_fee=float(merged["maker_fee"]), taker_fee=float(merged["taker_fee"]),
        use_maker_entry=bool(merged["use_maker_entry"]), use_maker_exit=bool(merged["use_maker_exit"]), slippage=float(merged["slippage"]),
        strategy_timeframe_minutes=int(merged["strategy_timeframe_minutes"]), intrabar_timeframe_minutes=int(merged["intrabar_timeframe_minutes"]),
        use_intrabar_data=bool(merged["use_intrabar_data"]), trading_start_date=merged.get("trading_start_date"), trading_end_date=merged.get("trading_end_date"),
        max_effective_leverage_per_leg=float(merged["max_effective_leverage_per_leg"]) if merged.get("max_effective_leverage_per_leg") not in (None, "") else None,
        max_combined_effective_leverage=float(merged["max_combined_effective_leverage"]) if merged.get("max_combined_effective_leverage") not in (None, "") else None,
        intrabar_missing_policy=IntrabarMissingPolicy(merged["intrabar_missing_policy"]), zero_cost_comparison=bool(merged["zero_cost_comparison"]),
        run_name=str(merged.get("run_name", "")),
        enable_trade_telemetry=bool(merged["enable_trade_telemetry"]), save_full_telemetry_csv=bool(merged["save_full_telemetry_csv"]),
        save_trade_journey_summary=bool(merged["save_trade_journey_summary"]), save_trade_journey_charts=bool(merged["save_trade_journey_charts"]),
        telemetry_interval_minutes=int(merged["telemetry_interval_minutes"]),
        enable_indicator_lifecycle_analysis=bool(merged["enable_indicator_lifecycle_analysis"]), lifecycle_phases=int(merged["lifecycle_phases"]),
        lifecycle_early_checkpoints=tuple(int(v) for v in merged["lifecycle_early_checkpoints"]),
        lifecycle_minimum_bucket_sample=int(merged["lifecycle_minimum_bucket_sample"]), create_lifecycle_charts=bool(merged["create_lifecycle_charts"]),
        lifecycle_flat_pattern_threshold_pct=float(merged["lifecycle_flat_pattern_threshold_pct"]),
        save_feature_analysis_reports=bool(merged["save_feature_analysis_reports"]),
        save_indicator_analysis_reports=bool(merged["save_indicator_analysis_reports"]),
        create_standard_charts=bool(merged["create_standard_charts"]),
    )


def canonical_config_values(values: dict[str, Any]) -> dict[str, Any]:
    merged = _merged_current(values)
    return {key: merged[key] for key in DEFAULT_GUI_CONFIG}


def save_config_json(path: str | Path, values: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(canonical_config_values(values), indent=2, default=str), encoding="utf-8")


def load_config_json(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Configuration JSON must contain an object.")
    return _merged_current(loaded)
