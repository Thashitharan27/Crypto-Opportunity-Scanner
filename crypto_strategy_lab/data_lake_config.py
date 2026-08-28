"""Strict composition-oriented configuration for native Data Lake research runs.

The authoritative Data Lake file format is the nested v3 ResearchRunConfig.
The existing GUI still emits the mature flat v2 contract; a bounded adapter is
kept here for that real consumer until the later GUI redesign. The native CLI,
benchmark, cache identity, and ResearchRunner never accept v2 through
``load_data_lake_config``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_VERSION = 3
GUI_COMPAT_CONFIG_VERSION = 2
PROFILE_KEYS = (
    "bull_long", "bull_short", "bear_long", "bear_short", "sideways_long", "sideways_short"
)


@dataclass(frozen=True)
class DataConfig:
    strategy_timeframe_minutes: int = 15
    intrabar_timeframe_minutes: int = 1
    use_intrabar_data: bool = True
    intrabar_missing_policy: str = "ERROR"


@dataclass(frozen=True)
class FeatureConfig:
    atr_period: int = 14
    adx_period: int = 14
    di_pressure_lookback: int = 3
    bb_period: int = 20
    bb_stddevs: float = 2.0
    mean_reversion_period: int = 20
    mean_reversion_mean_type: str = "SMA"
    mean_reversion_bb_stddevs: float = 2.0
    mean_reversion_rsi_period: int = 14
    mean_reversion_rsi_oversold: float = 30.0
    mean_reversion_rsi_overbought: float = 70.0
    mean_reversion_require_reentry: bool = True
    mean_reversion_track_atr_distance: bool = True
    mean_reversion_track_motion: bool = True
    enable_support_resistance_analysis: bool = False
    sr_timeframe_minutes: int = 0
    sr_pivot_left: int = 5
    sr_pivot_right: int = 5
    sr_lookback_bars: int = 200
    sr_zone_width_atr: float = 0.5
    sr_near_distance_atr: float = 0.75
    enable_sr_hold_confirmation: bool = False
    sr_hold_confirmation_bars: int = 3
    sr_hold_confirmation_atr: float = 0.25
    sr_break_tolerance_atr: float = 0.25
    sr_break_basis: str = "CLOSE"
    market_regime_method: str = "BTC_STRUCTURAL"
    structural_regime_sma_days: int = 200
    structural_regime_slope_lookback_days: int = 30
    bull_regime_lookback_days: int = 90
    bull_regime_return_threshold: float = 0.20
    trade_flow_enabled: bool = False
    trade_flow_source: str = "AGG_TRADES"
    trade_flow_base_interval: str = "1m"
    trade_flow_windows: tuple[str, ...] = ("1m", "5m", "15m", "1h")
    large_trade_quote_threshold: float | None = None
    order_book_enabled: bool = False
    order_book_base_interval: str = "1m"
    book_ticker_max_age_seconds: float = 5.0
    book_depth_max_age_seconds: float = 90.0
    taker_flow_interval: str = "5m"
    oi_zscore_window_days: float = 7.0
    oi_zscore_min_samples: int = 20
    funding_zscore_window_days: float = 7.0
    funding_zscore_min_samples: int = 6
    funding_extreme_zscore: float = 2.0
    basis_zscore_window_days: float = 7.0

    def registry_parameters(self, *, strategy_timeframe_minutes: int | None = None) -> dict[str, dict[str, Any]]:
        """Authoritative registered-feature parameters owned by FeatureConfig."""
        directional = {
            "atr_period": int(self.atr_period),
            "adx_period": int(self.adx_period),
            "di_pressure_lookback": int(self.di_pressure_lookback),
        }
        context = {
            "bb_period": int(self.bb_period),
            "bb_stddevs": float(self.bb_stddevs),
            "mean_reversion_period": int(self.mean_reversion_period),
            "mean_reversion_mean_type": str(self.mean_reversion_mean_type).upper(),
            "mean_reversion_bb_stddevs": float(self.mean_reversion_bb_stddevs),
            "mean_reversion_rsi_period": int(self.mean_reversion_rsi_period),
            "mean_reversion_rsi_oversold": float(self.mean_reversion_rsi_oversold),
            "mean_reversion_rsi_overbought": float(self.mean_reversion_rsi_overbought),
            "mean_reversion_require_reentry": bool(self.mean_reversion_require_reentry),
        }
        result = {
            "core_directional": directional,
            "production_market_context": context,
        }
        result["futures_positioning"] = {
            "oi_zscore_window_days": float(self.oi_zscore_window_days),
            "oi_zscore_min_samples": int(self.oi_zscore_min_samples),
        }
        result["funding_context"] = {
            "funding_zscore_window_days": float(self.funding_zscore_window_days),
            "funding_zscore_min_samples": int(self.funding_zscore_min_samples),
            "funding_extreme_zscore": float(self.funding_extreme_zscore),
        }
        result["taker_flow_context"] = {"taker_flow_interval": str(self.taker_flow_interval)}
        result["basis_context"] = {"basis_zscore_window_days": float(self.basis_zscore_window_days)}
        if self.trade_flow_enabled:
            source = str(self.trade_flow_source).upper()
            if source not in {"AGG_TRADES", "TRADES"}:
                raise ValueError("trade_flow_source must be AGG_TRADES or TRADES")
            if self.trade_flow_base_interval != "1m":
                raise ValueError("trade_flow_base_interval must be 1m")
            if self.large_trade_quote_threshold is not None and self.large_trade_quote_threshold <= 0:
                raise ValueError("large_trade_quote_threshold must be positive or null")
            result["trade_flow_context"] = {
                "trade_flow_source": source,
                "trade_flow_windows": tuple(self.trade_flow_windows),
            }
        if self.order_book_enabled:
            if self.order_book_base_interval != "1m":
                raise ValueError("order_book_base_interval must be 1m")
            if self.book_ticker_max_age_seconds < 0 or self.book_depth_max_age_seconds < 0:
                raise ValueError("order-book maximum ages must be non-negative")
            result["order_book_context"] = {
                "book_ticker_max_age_seconds": float(self.book_ticker_max_age_seconds),
                "book_depth_max_age_seconds": float(self.book_depth_max_age_seconds),
            }
        if self.enable_support_resistance_analysis:
            if strategy_timeframe_minutes is None:
                raise ValueError("strategy timeframe is required for S/R feature parameters")
            effective_sr = int(self.sr_timeframe_minutes or strategy_timeframe_minutes)
            result["support_resistance"] = {
                "atr_period": int(self.atr_period),
                "sr_timeframe_minutes": effective_sr,
                "sr_pivot_left": int(self.sr_pivot_left),
                "sr_pivot_right": int(self.sr_pivot_right),
                "sr_lookback_bars": int(self.sr_lookback_bars),
                "sr_zone_width_atr": float(self.sr_zone_width_atr),
                "sr_near_distance_atr": float(self.sr_near_distance_atr),
                "enable_sr_hold_confirmation": bool(self.enable_sr_hold_confirmation),
                "sr_hold_confirmation_bars": int(self.sr_hold_confirmation_bars),
                "sr_hold_confirmation_atr": float(self.sr_hold_confirmation_atr),
                "sr_break_tolerance_atr": float(self.sr_break_tolerance_atr),
                "sr_break_basis": str(self.sr_break_basis).upper(),
            }
        return result


@dataclass(frozen=True)
class StrategyProfileConfig:
    enabled: bool = True
    flip_direction: bool = False
    entry_rules: tuple = ()
    flip_rule_match_mode: str = "ANY"
    reject_rule_match_mode: str = "ANY"
    rsi_period: int = 14
    momentum_lookback_hours: int = 24


@dataclass(frozen=True)
class ExecutionProfileConfig:
    reward_risk_ratio: float = 1.0
    risk_multiplier: float = 1.0
    stop_loss_multiple: float = 2.0
    partial_stop_enabled: bool = False
    sl1_r: float = 0.5
    sl1_close_pct: float = 50.0
    sl2_r: float = 2.0
    partial_profit_enabled: bool = False
    tp1_r: float = 1.0
    tp1_close_pct: float = 50.0
    tp2_r: float = 2.0
    trailing_enabled: bool = False
    trailing_activation_r: float = 3.0
    trailing_distance_r: float = 1.0
    break_even_enabled: bool = False
    break_even_activation_r: float = 1.0
    break_even_offset_r: float = 0.0
    timeout_enabled: bool = False
    timeout_minutes: int = 480
    r_step_trailing_enabled: bool = False
    r_step_activation_r: float = 2.0
    r_step_distance_r: float = 2.0
    r_step_size_r: float = 1.0
    r_step_maximum_r: float = 0.0
    r_step_activation_close_pct: float = 0.0
    atr_checkpoint_tp_extension_enabled: bool = False
    atr_checkpoint_di_spread_minimum: float = 30.0
    atr_checkpoint_bb_width_minimum: float = 0.03
    atr_checkpoint_profit_lock_start: float = 3.0
    atr_checkpoint_profit_lock_distance: float = 1.0


def _strategy_profiles() -> dict[str, StrategyProfileConfig]:
    return {key: StrategyProfileConfig() for key in PROFILE_KEYS}


def _execution_profiles() -> dict[str, ExecutionProfileConfig]:
    return {key: ExecutionProfileConfig() for key in PROFILE_KEYS}


@dataclass(frozen=True)
class StrategyConfig:
    profiles: Mapping[str, StrategyProfileConfig] = field(default_factory=_strategy_profiles)
    strategy_profile_run_mode: str = "COMBINED_SHARED_CAPITAL"
    entry_mode: str = "WAIT_UNTIL_CLOSED"
    entry_interval: int = 1
    enable_di_direction_selection: bool = True
    enable_di_pressure_analysis: bool = True
    di_pressure_allow_expanding: bool = True
    di_pressure_allow_contracting: bool = True
    di_pressure_allow_mixed: bool = True
    enable_mean_reversion_analysis: bool = True
    sr_filter_mode: str = "ANALYSIS_ONLY"
    sr_long_avoid_near_resistance: bool = False
    sr_long_require_near_support: bool = False
    sr_long_block_broken_support: bool = False
    sr_long_min_room_to_resistance_atr: float = 0.0
    sr_short_avoid_near_support: bool = False
    sr_short_require_near_resistance: bool = False
    sr_short_block_broken_resistance: bool = False
    sr_short_min_room_to_support_atr: float = 0.0
    enable_daily_entry_schedule: bool = False
    daily_entry_time: str = "00:00"
    daily_entry_timezone: str = "UTC"
    daily_entry_missed_policy: str = "SKIP_DAY"


@dataclass(frozen=True)
class ExecutionConfig:
    profiles: Mapping[str, ExecutionProfileConfig] = field(default_factory=_execution_profiles)
    initial_equity: float = 1000.0
    risk_mode: str = "ATR"
    fixed_r: float = 100.0
    percent_r: float = 0.002
    atr_multiplier: float = 1.0
    risk_per_leg: float = 0.01
    max_effective_leverage_per_leg: float | None = 3.0
    max_combined_effective_leverage: float | None = 5.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    use_maker_entry: bool = False
    use_maker_exit: bool = False
    slippage: float = 0.0005
    tie_policy: str = "PESSIMISTIC"
    max_active_pairs: int = 1
    zero_cost_comparison: bool = False
    sr_take_profit_mode: str = "FIXED_R"
    sr_take_profit_maximum_r: float = 3.0
    sr_take_profit_minimum_r: float = 1.5
    sr_take_profit_buffer_r: float = 0.20
    sr_take_profit_no_level_policy: str = "USE_FIXED_TP"


@dataclass(frozen=True)
class ReportingConfig:
    run_name: str = ""
    output_dir: str = "output/data_lake_v2"
    analysis_level: str = "STANDARD"
    research_sampling_mode: str = "PORTFOLIO"
    research_sampling_interval_candles: int = 1
    enable_trade_telemetry: bool = False
    save_full_telemetry_csv: bool = False
    save_trade_journey_summary: bool = False
    save_trade_journey_charts: bool = False
    telemetry_interval_minutes: int = 15
    enable_indicator_lifecycle_analysis: bool = False
    lifecycle_phases: int = 4
    lifecycle_early_checkpoints: tuple = (15, 30, 60)
    lifecycle_minimum_bucket_sample: int = 20
    create_lifecycle_charts: bool = False
    lifecycle_flat_pattern_threshold_pct: float = 5.0
    save_feature_analysis_reports: bool = False
    save_indicator_analysis_reports: bool = True
    create_standard_charts: bool = True


@dataclass(frozen=True)
class ResearchRunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    config_version: int = CONFIG_VERSION

    def validate(self, request=None) -> None:
        data, features, strategy, execution, reporting = (
            self.data, self.features, self.strategy, self.execution, self.reporting
        )
        if data.strategy_timeframe_minutes <= 0 or data.intrabar_timeframe_minutes <= 0:
            raise ValueError("timeframes must be positive")
        if data.use_intrabar_data and data.intrabar_timeframe_minutes >= data.strategy_timeframe_minutes:
            raise ValueError("intrabar timeframe must be smaller than strategy timeframe")
        if data.intrabar_missing_policy not in {"ERROR", "WARN_AND_USE_15M", "WARN_AND_CONTINUE"}:
            raise ValueError("invalid intrabar missing policy")
        if request is not None:
            if request.strategy_interval != f"{data.strategy_timeframe_minutes}m":
                raise ValueError("DataRequest strategy interval disagrees with DataConfig")
            expected_intrabar = f"{data.intrabar_timeframe_minutes}m" if data.use_intrabar_data else None
            if request.intrabar_interval != expected_intrabar:
                raise ValueError("DataRequest intrabar interval disagrees with DataConfig")
        for value, label in (
            (features.atr_period, "ATR period"), (features.adx_period, "ADX period"),
            (features.di_pressure_lookback, "DI pressure lookback"),
            (features.bb_period, "BB period"), (features.mean_reversion_period, "MR period"),
            (features.mean_reversion_rsi_period, "MR RSI period"),
        ):
            if int(value) <= 0:
                raise ValueError(f"{label} must be positive")
        if features.bb_stddevs <= 0 or features.mean_reversion_bb_stddevs <= 0:
            raise ValueError("standard deviations must be positive")
        if not 0 <= features.mean_reversion_rsi_oversold < features.mean_reversion_rsi_overbought <= 100:
            raise ValueError("MR RSI thresholds are invalid")
        if features.market_regime_method not in {"ASSET_RETURN", "BTC_STRUCTURAL", "ASSET_STRUCTURAL"}:
            raise ValueError("invalid market regime method")
        if features.trade_flow_source not in {"AGG_TRADES", "TRADES"}:
            raise ValueError("trade_flow_source must be AGG_TRADES or TRADES")
        if features.trade_flow_base_interval != "1m":
            raise ValueError("trade_flow_base_interval must be 1m")
        if not features.trade_flow_windows or not set(features.trade_flow_windows) <= {"1m", "5m", "15m", "1h"}:
            raise ValueError("invalid trade_flow_windows")
        if features.large_trade_quote_threshold is not None and features.large_trade_quote_threshold <= 0:
            raise ValueError("large_trade_quote_threshold must be positive or null")
        if features.order_book_base_interval != "1m":
            raise ValueError("order_book_base_interval must be 1m")
        if features.book_ticker_max_age_seconds < 0 or features.book_depth_max_age_seconds < 0:
            raise ValueError("order-book maximum ages must be non-negative")
        if features.bull_regime_lookback_days <= 0 or features.bull_regime_return_threshold <= -1:
            raise ValueError("asset-return regime settings are invalid")
        if features.sr_break_basis not in {"CLOSE", "WICK"}:
            raise ValueError("invalid S/R break basis")
        if features.sr_timeframe_minutes:
            if features.sr_timeframe_minutes < data.strategy_timeframe_minutes:
                raise ValueError("S/R timeframe cannot be lower than strategy timeframe")
            if features.sr_timeframe_minutes % data.strategy_timeframe_minutes:
                raise ValueError("S/R timeframe must be an integer multiple of strategy timeframe")
        if set(strategy.profiles) != set(PROFILE_KEYS) or set(execution.profiles) != set(PROFILE_KEYS):
            raise ValueError("strategy and execution profile keys must match the current profile contract")
        if strategy.strategy_profile_run_mode not in {"ISOLATED_PROFILES", "COMBINED_SHARED_CAPITAL", "BOTH"}:
            raise ValueError("invalid strategy profile run mode")
        if strategy.entry_mode not in {"WAIT_UNTIL_CLOSED", "EVERY_N_CANDLES"} or strategy.entry_interval <= 0:
            raise ValueError("invalid entry cadence")
        if strategy.sr_filter_mode not in {"ANALYSIS_ONLY", "APPLY_ENTRY_RULES"}:
            raise ValueError("invalid S/R strategy policy")
        if strategy.enable_di_pressure_analysis and not any((
            strategy.di_pressure_allow_expanding,
            strategy.di_pressure_allow_contracting,
            strategy.di_pressure_allow_mixed,
        )):
            raise ValueError("at least one DI pressure state must be allowed")
        for key, profile in strategy.profiles.items():
            if profile.rsi_period <= 0 or profile.momentum_lookback_hours <= 0:
                raise ValueError(f"{key}: profile feature periods must be positive")
            if profile.rsi_period != features.mean_reversion_rsi_period:
                raise ValueError(f"{key}: profile RSI period must match prepared MR RSI period")
        if execution.initial_equity <= 0 or execution.fixed_r <= 0 or execution.percent_r <= 0:
            raise ValueError("execution equity/risk settings must be positive")
        if not 0 < execution.risk_per_leg < 1:
            raise ValueError("risk per leg must be between 0 and 1")
        if execution.risk_mode not in {"FIXED", "PERCENT", "ATR"}:
            raise ValueError("invalid risk mode")
        if execution.tie_policy not in {"PESSIMISTIC", "OPTIMISTIC", "INTRABAR"}:
            raise ValueError("invalid tie policy")
        if execution.max_active_pairs <= 0:
            raise ValueError("max active pairs must be positive")
        if min(execution.maker_fee, execution.taker_fee, execution.slippage) < 0:
            raise ValueError("fees/slippage must be non-negative")
        if reporting.research_sampling_mode not in {
            "PORTFOLIO", "EVERY_VIABLE_ENTRY", "FIXED_INTERVAL", "EPISODE_FIRST"
        }:
            raise ValueError("invalid research sampling mode")
        if reporting.research_sampling_interval_candles <= 0:
            raise ValueError("research sampling interval must be positive")
        if reporting.enable_trade_telemetry:
            if reporting.telemetry_interval_minutes <= 0:
                raise ValueError("telemetry interval must be positive")
            if reporting.telemetry_interval_minutes % data.strategy_timeframe_minutes:
                raise ValueError("telemetry interval must be a multiple of strategy timeframe")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict(cls, raw, label):
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} settings: " + ", ".join(unknown))
    values = dict(raw)
    if cls is StrategyProfileConfig and isinstance(values.get("entry_rules"), list):
        values["entry_rules"] = tuple(values["entry_rules"])
    if cls is ReportingConfig and isinstance(values.get("lifecycle_early_checkpoints"), list):
        values["lifecycle_early_checkpoints"] = tuple(values["lifecycle_early_checkpoints"])
    if cls is FeatureConfig and isinstance(values.get("trade_flow_windows"), list):
        values["trade_flow_windows"] = tuple(values["trade_flow_windows"])
    return cls(**values)


def _profiles(raw, strategy: bool):
    cls = StrategyProfileConfig if strategy else ExecutionProfileConfig
    if raw is None:
        return _strategy_profiles() if strategy else _execution_profiles()
    if not isinstance(raw, dict):
        raise ValueError("profiles must be an object")
    unknown = sorted(set(raw) - set(PROFILE_KEYS))
    if unknown:
        raise ValueError("unknown profile keys: " + ", ".join(unknown))
    return {key: _strict(cls, raw.get(key, {}), f"profiles.{key}") for key in PROFILE_KEYS}


def _parse_research_run_config(values: dict[str, Any]) -> ResearchRunConfig:
    if not isinstance(values, dict):
        raise ValueError("Data Lake configuration JSON must contain an object")
    allowed = {"config_version", "data", "features", "strategy", "execution", "reporting"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("Unknown Data Lake configuration sections: " + ", ".join(unknown))
    if values.get("config_version") != CONFIG_VERSION:
        raise ValueError("Data Lake configuration version 3 is required.")
    sraw, eraw = dict(values.get("strategy", {})), dict(values.get("execution", {}))
    strategy_profiles = _profiles(sraw.pop("profiles", None), True)
    execution_profiles = _profiles(eraw.pop("profiles", None), False)
    parsed_strategy = _strict(StrategyConfig, sraw, "strategy")
    parsed_execution = _strict(ExecutionConfig, eraw, "execution")
    strategy_values = {k: v for k, v in parsed_strategy.__dict__.items() if k != "profiles"}
    execution_values = {k: v for k, v in parsed_execution.__dict__.items() if k != "profiles"}
    result = ResearchRunConfig(
        data=_strict(DataConfig, values.get("data", {}), "data"),
        features=_strict(FeatureConfig, values.get("features", {}), "features"),
        strategy=StrategyConfig(profiles=strategy_profiles, **strategy_values),
        execution=ExecutionConfig(profiles=execution_profiles, **execution_values),
        reporting=_strict(ReportingConfig, values.get("reporting", {}), "reporting"),
    )
    result.validate()
    return result


def default_data_lake_config() -> dict[str, Any]:
    return ResearchRunConfig().to_dict()


def load_data_lake_config(path: str | Path) -> ResearchRunConfig:
    """Load only the authoritative nested v3 Data Lake contract."""
    return _parse_research_run_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


# ---------------------------------------------------------------------------
# Bounded GUI v2 adapter. This is a real legacy consumer, not an accepted
# native Data Lake file format. Remove it when the GUI is redesigned.
# ---------------------------------------------------------------------------
_GUI_EXCLUDED_FIELDS = {"input_csv", "intrabar_csv", "output_dir", "structural_regime_benchmark_csv"}


def _gui_defaults() -> dict[str, Any]:
    from crypto_strategy_lab.gui.enhanced_config import enhanced_default_gui_config
    return enhanced_default_gui_config()


DATA_LAKE_CONFIG_FIELDS = frozenset(_gui_defaults()) - _GUI_EXCLUDED_FIELDS


def _normalize_gui_v2(values: dict[str, Any]) -> dict[str, Any]:
    from crypto_strategy_lab.strategy_profiles import normalize_profiles, profiles_to_dict
    if not isinstance(values, dict):
        raise ValueError("Data Lake GUI configuration must contain an object")
    unknown = sorted(set(values) - DATA_LAKE_CONFIG_FIELDS)
    if unknown:
        raise ValueError("Unknown Data Lake GUI settings: " + ", ".join(unknown))
    merged = {key: value for key, value in _gui_defaults().items() if key in DATA_LAKE_CONFIG_FIELDS}
    merged.update(values)
    if int(merged.get("config_version", -1)) != GUI_COMPAT_CONFIG_VERSION:
        raise ValueError("Data Lake GUI compatibility configuration version 2 is required")
    merged["strategy_profiles"] = profiles_to_dict(normalize_profiles(merged["strategy_profiles"]))
    return merged


def normalize_data_lake_config(values: dict[str, Any]):
    """Normalize v3 natively, or v2 only for the existing GUI compatibility path."""
    if isinstance(values, dict) and values.get("config_version") == GUI_COMPAT_CONFIG_VERSION:
        return _normalize_gui_v2(values)
    return _parse_research_run_config(values)


def build_data_lake_backtest_config(values: dict[str, Any]):
    """Build v3 native config, or the bounded v2 GUI engine adapter."""
    if isinstance(values, dict) and values.get("config_version") == GUI_COMPAT_CONFIG_VERSION:
        from crypto_strategy_lab.gui.enhanced_config import (
            build_enhanced_backtest_config,
            enhanced_default_gui_config,
        )
        cleaned = _normalize_gui_v2(values)
        bridge = enhanced_default_gui_config()
        bridge.update(cleaned)
        bridge["input_csv"] = "__DATA_LAKE_STRATEGY__"
        bridge["intrabar_csv"] = None
        bridge["output_dir"] = "output"
        bridge["structural_regime_benchmark_csv"] = None
        return build_enhanced_backtest_config(bridge, require_paths=False)
    return _parse_research_run_config(values)