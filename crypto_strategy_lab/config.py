"""Current configuration contract for Crypto Strategy Lab."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from crypto_strategy_lab.strategy_profiles import StrategyProfile, default_profiles, normalize_profiles


class RiskMode(str, Enum):
    FIXED = "FIXED"; PERCENT = "PERCENT"; ATR = "ATR"
class EntryMode(str, Enum):
    WAIT_UNTIL_CLOSED = "WAIT_UNTIL_CLOSED"; EVERY_N_CANDLES = "EVERY_N_CANDLES"
class TiePolicy(str, Enum):
    PESSIMISTIC = "PESSIMISTIC"; OPTIMISTIC = "OPTIMISTIC"; INTRABAR = "INTRABAR"
class IntrabarMissingPolicy(str, Enum):
    ERROR = "ERROR"; WARN_AND_USE_15M = "WARN_AND_USE_15M"; WARN_AND_CONTINUE = "WARN_AND_CONTINUE"
class DailyEntryMissedPolicy(str, Enum):
    SKIP_DAY = "SKIP_DAY"; NEXT_AVAILABLE_CANDLE = "NEXT_AVAILABLE_CANDLE"


@dataclass(frozen=True)
class BacktestConfig:
    """Only settings belonging to the current GUI/Strategy Profile contract.

    Retired global strategy switches do not exist in this contract. Old JSON or
    direct constructor arguments therefore cannot reactivate removed behavior.
    """

    input_csv: Path = Path(r"C:\CryptoBots\Binance Market Data\futures\usdm\BTCUSDT_15m.csv")
    intrabar_csv: Optional[Path] = Path(r"C:\CryptoBots\Binance Market Data\futures\usdm\BTCUSDT_1m.csv")
    output_dir: Path = Path("output")
    timestamp_unit: Optional[str] = "ms"
    strategy_timeframe_minutes: int = 15
    intrabar_timeframe_minutes: int = 1
    use_intrabar_data: bool = True
    data_start_date: Optional[str] = None
    trading_start_date: Optional[str] = None
    trading_end_date: Optional[str] = None
    max_effective_leverage_per_leg: Optional[float] = None
    max_combined_effective_leverage: Optional[float] = None
    intrabar_missing_policy: IntrabarMissingPolicy = IntrabarMissingPolicy.WARN_AND_USE_15M
    zero_cost_comparison: bool = False

    strategy_profile_run_mode: str = "COMBINED_SHARED_CAPITAL"
    strategy_profiles: dict[str, StrategyProfile] = field(default_factory=default_profiles)

    initial_equity: float = 1000.0
    risk_mode: RiskMode = RiskMode.ATR
    fixed_r: float = 100.0
    percent_r: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 1.0
    risk_per_leg: float = 0.005
    adx_period: int = 14
    bb_period: int = 20
    bb_stddevs: float = 2.0

    entry_mode: EntryMode = EntryMode.WAIT_UNTIL_CLOSED
    entry_interval: int = 1
    enable_di_direction_selection: bool = True
    enable_di_pressure_analysis: bool = True
    di_pressure_lookback: int = 3
    enable_mean_reversion_analysis: bool = True
    mean_reversion_period: int = 20

    enable_support_resistance_analysis: bool = False
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
    sr_filter_mode: str = "ANALYSIS_ONLY"
    sr_long_avoid_near_resistance: bool = False
    sr_long_require_near_support: bool = False
    sr_long_block_broken_support: bool = False
    sr_long_min_room_to_resistance_atr: float = 0.0
    sr_short_avoid_near_support: bool = False
    sr_short_require_near_resistance: bool = False
    sr_short_block_broken_resistance: bool = False
    sr_short_min_room_to_support_atr: float = 0.0

    market_regime_method: str = "ASSET_RETURN"
    structural_regime_sma_days: int = 200
    structural_regime_slope_lookback_days: int = 30
    structural_regime_benchmark_csv: Optional[Path] = None
    bull_regime_lookback_days: int = 90
    bull_regime_return_threshold: float = 0.20

    enable_daily_entry_schedule: bool = False
    daily_entry_time: str = "00:00"
    daily_entry_timezone: str = "UTC"
    daily_entry_missed_policy: DailyEntryMissedPolicy = DailyEntryMissedPolicy.SKIP_DAY

    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    use_maker_entry: bool = False
    use_maker_exit: bool = False
    slippage: float = 0.0001
    tie_policy: TiePolicy = TiePolicy.PESSIMISTIC
    max_active_pairs: int = 1
    run_name: str = ""
    output_run_dir: Optional[Path] = None

    enable_trade_telemetry: bool = True
    save_full_telemetry_csv: bool = True
    save_trade_journey_summary: bool = True
    save_trade_journey_charts: bool = True
    telemetry_interval_minutes: int = 15
    enable_indicator_lifecycle_analysis: bool = True
    lifecycle_phases: int = 4
    lifecycle_early_checkpoints: tuple[int, ...] = (15, 30, 60)
    lifecycle_minimum_bucket_sample: int = 20
    create_lifecycle_charts: bool = True
    lifecycle_flat_pattern_threshold_pct: float = 5.0
    save_feature_analysis_reports: bool = True
    save_indicator_analysis_reports: bool = True
    create_standard_charts: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_profiles", normalize_profiles(self.strategy_profiles))
        if self.strategy_profile_run_mode not in ("ISOLATED_PROFILES", "COMBINED_SHARED_CAPITAL", "BOTH"):
            raise ValueError("invalid strategy_profile_run_mode")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.strategy_timeframe_minutes <= 0 or self.intrabar_timeframe_minutes <= 0:
            raise ValueError("timeframes must be positive")
        if self.use_intrabar_data and self.intrabar_timeframe_minutes >= self.strategy_timeframe_minutes:
            raise ValueError("intrabar timeframe must be less than strategy timeframe")
        if not 0 < self.risk_per_leg < 1:
            raise ValueError("risk_per_leg must be between 0 and 1")
        if self.atr_period <= 0 or self.atr_multiplier <= 0:
            raise ValueError("ATR settings must be positive")
        if self.adx_period <= 0 or self.bb_period <= 0 or self.bb_stddevs <= 0:
            raise ValueError("indicator periods/settings must be positive")
        if self.entry_interval <= 0 or self.max_active_pairs <= 0:
            raise ValueError("entry_interval and max_active_pairs must be positive")
        if self.fixed_r <= 0 or self.percent_r <= 0:
            raise ValueError("risk distances must be positive")
        if self.maker_fee < 0 or self.taker_fee < 0 or self.slippage < 0:
            raise ValueError("fees and slippage must be non-negative")
        if self.max_effective_leverage_per_leg is not None and self.max_effective_leverage_per_leg <= 0:
            raise ValueError("max leverage per trade must be positive")
        if self.max_combined_effective_leverage is not None and self.max_combined_effective_leverage <= 0:
            raise ValueError("max portfolio leverage must be positive")
        if not 1 <= self.di_pressure_lookback <= 100:
            raise ValueError("di_pressure_lookback must be between 1 and 100")
        if self.mean_reversion_period <= 0:
            raise ValueError("mean_reversion_period must be positive")
        if self.market_regime_method not in ("ASSET_RETURN", "BTC_STRUCTURAL", "ASSET_STRUCTURAL"):
            raise ValueError("invalid market_regime_method")
        if self.structural_regime_sma_days < 2 or self.structural_regime_slope_lookback_days < 1:
            raise ValueError("structural regime periods are invalid")
        if self.bull_regime_lookback_days <= 0 or self.bull_regime_return_threshold <= -1:
            raise ValueError("asset-return regime settings are invalid")
        if self.sr_pivot_left < 1 or self.sr_pivot_right < 1 or self.sr_lookback_bars < 1:
            raise ValueError("support/resistance lookbacks must be positive")
        if self.sr_zone_width_atr < 0 or self.sr_near_distance_atr < 0 or self.sr_hold_confirmation_atr < 0 or self.sr_break_tolerance_atr < 0:
            raise ValueError("support/resistance ATR distances must be non-negative")
        if self.sr_filter_mode not in ("ANALYSIS_ONLY", "APPLY_ENTRY_RULES"):
            raise ValueError("invalid sr_filter_mode")
        if self.sr_break_basis not in ("CLOSE", "WICK"):
            raise ValueError("invalid sr_break_basis")
        if isinstance(self.intrabar_missing_policy, str):
            object.__setattr__(self, "intrabar_missing_policy", IntrabarMissingPolicy(self.intrabar_missing_policy))
        if isinstance(self.risk_mode, str):
            object.__setattr__(self, "risk_mode", RiskMode(self.risk_mode))
        if isinstance(self.entry_mode, str):
            object.__setattr__(self, "entry_mode", EntryMode(self.entry_mode))
        if isinstance(self.tie_policy, str):
            object.__setattr__(self, "tie_policy", TiePolicy(self.tie_policy))
        if isinstance(self.daily_entry_missed_policy, str):
            object.__setattr__(self, "daily_entry_missed_policy", DailyEntryMissedPolicy(self.daily_entry_missed_policy))
        try:
            hh, mm = [int(part) for part in str(self.daily_entry_time).split(":", 1)]
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError
        except Exception as exc:
            raise ValueError("daily_entry_time must be HH:MM in 24-hour time") from exc
        if (hh * 60 + mm) % self.strategy_timeframe_minutes != 0:
            raise ValueError("daily_entry_time must align to the strategy timeframe")
        try:
            ZoneInfo(self.daily_entry_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("daily_entry_timezone must be a valid IANA timezone") from exc
        if self.telemetry_interval_minutes <= 0:
            raise ValueError("telemetry interval must be > 0")
        if self.lifecycle_phases != 4:
            raise ValueError("lifecycle_phases must currently be 4")
        if self.lifecycle_minimum_bucket_sample <= 0 or self.lifecycle_flat_pattern_threshold_pct < 0:
            raise ValueError("lifecycle settings are invalid")
        if isinstance(self.lifecycle_early_checkpoints, list):
            object.__setattr__(self, "lifecycle_early_checkpoints", tuple(int(v) for v in self.lifecycle_early_checkpoints))
        if any(v <= 0 for v in self.lifecycle_early_checkpoints):
            raise ValueError("lifecycle early checkpoints must be positive")
        if self.telemetry_interval_minutes % self.strategy_timeframe_minutes != 0:
            raise ValueError("telemetry interval must be a multiple of the strategy timeframe")
