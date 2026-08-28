"""Extended configuration for research analytics and optional entry filters."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.config_logic import (
    DEFAULT_GUI_CONFIG,
    build_backtest_config,
    canonical_config_values,
    default_gui_config,
)


DI_PRESSURE_FILTER_DEFAULTS: dict[str, Any] = {
    "di_pressure_allow_expanding": True,
    "di_pressure_allow_contracting": True,
    "di_pressure_allow_mixed": True,
}

MEAN_REVERSION_V2_DEFAULTS: dict[str, Any] = {
    "mean_reversion_mean_type": "SMA",
    "mean_reversion_bb_stddevs": 2.0,
    "mean_reversion_rsi_period": 14,
    "mean_reversion_rsi_oversold": 30.0,
    "mean_reversion_rsi_overbought": 70.0,
    "mean_reversion_require_reentry": True,
    "mean_reversion_track_atr_distance": True,
    "mean_reversion_track_motion": True,
}

SR_HTF_DEFAULTS: dict[str, Any] = {
    # 0 means use the strategy timeframe exactly as before.
    "sr_timeframe_minutes": 0,
}

SR_DYNAMIC_TP_DEFAULTS: dict[str, Any] = {
    "sr_take_profit_mode": "FIXED_R",
    "sr_take_profit_maximum_r": 3.0,
    "sr_take_profit_minimum_r": 1.5,
    "sr_take_profit_buffer_r": 0.20,
    "sr_take_profit_no_level_policy": "USE_FIXED_TP",
}

ENHANCED_DEFAULTS: dict[str, Any] = {
    **DI_PRESSURE_FILTER_DEFAULTS,
    **MEAN_REVERSION_V2_DEFAULTS,
    **SR_HTF_DEFAULTS,
    **SR_DYNAMIC_TP_DEFAULTS,
}


@dataclass(frozen=True)
class EnhancedBacktestConfig(BacktestConfig):
    """BacktestConfig plus DI-pressure, MR-v2, and advanced S/R settings."""

    di_pressure_allow_expanding: bool = True
    di_pressure_allow_contracting: bool = True
    di_pressure_allow_mixed: bool = True
    mean_reversion_mean_type: str = "SMA"
    mean_reversion_bb_stddevs: float = 2.0
    mean_reversion_rsi_period: int = 14
    mean_reversion_rsi_oversold: float = 30.0
    mean_reversion_rsi_overbought: float = 70.0
    mean_reversion_require_reentry: bool = True
    mean_reversion_track_atr_distance: bool = True
    mean_reversion_track_motion: bool = True
    sr_timeframe_minutes: int = 0
    sr_take_profit_mode: str = "FIXED_R"
    sr_take_profit_maximum_r: float = 3.0
    sr_take_profit_minimum_r: float = 1.5
    sr_take_profit_buffer_r: float = 0.20
    sr_take_profit_no_level_policy: str = "USE_FIXED_TP"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.enable_di_pressure_analysis and not any(
            (
                self.di_pressure_allow_expanding,
                self.di_pressure_allow_contracting,
                self.di_pressure_allow_mixed,
            )
        ):
            raise ValueError("At least one DI pressure state must be allowed when DI pressure analysis is enabled")

        mean_type = str(self.mean_reversion_mean_type).upper()
        object.__setattr__(self, "mean_reversion_mean_type", mean_type)
        if mean_type not in ("SMA", "EMA"):
            raise ValueError("mean_reversion_mean_type must be SMA or EMA")
        if self.mean_reversion_bb_stddevs <= 0:
            raise ValueError("mean_reversion_bb_stddevs must be positive")
        if self.mean_reversion_rsi_period <= 0:
            raise ValueError("mean_reversion_rsi_period must be positive")
        oversold = float(self.mean_reversion_rsi_oversold)
        overbought = float(self.mean_reversion_rsi_overbought)
        if not 0 <= oversold < overbought <= 100:
            raise ValueError("RSI thresholds must satisfy 0 <= oversold < overbought <= 100")

        sr_tf = int(self.sr_timeframe_minutes)
        object.__setattr__(self, "sr_timeframe_minutes", sr_tf)
        if sr_tf < 0:
            raise ValueError("sr_timeframe_minutes cannot be negative")
        if sr_tf:
            if sr_tf < self.strategy_timeframe_minutes:
                raise ValueError("S/R timeframe cannot be lower than the strategy timeframe")
            if sr_tf % self.strategy_timeframe_minutes != 0:
                raise ValueError("S/R timeframe must be an integer multiple of the strategy timeframe")

        tp_mode = str(self.sr_take_profit_mode).upper()
        no_level_policy = str(self.sr_take_profit_no_level_policy).upper()
        object.__setattr__(self, "sr_take_profit_mode", tp_mode)
        object.__setattr__(self, "sr_take_profit_no_level_policy", no_level_policy)
        if tp_mode not in ("FIXED_R", "SR_CAPPED_R"):
            raise ValueError("sr_take_profit_mode must be FIXED_R or SR_CAPPED_R")
        if no_level_policy not in ("USE_FIXED_TP", "REJECT_TRADE"):
            raise ValueError("sr_take_profit_no_level_policy must be USE_FIXED_TP or REJECT_TRADE")
        if self.sr_take_profit_maximum_r <= 0 or self.sr_take_profit_minimum_r <= 0:
            raise ValueError("S/R take-profit R values must be positive")
        if self.sr_take_profit_minimum_r > self.sr_take_profit_maximum_r:
            raise ValueError("S/R minimum TP cannot exceed maximum TP")
        if self.sr_take_profit_buffer_r < 0:
            raise ValueError("S/R take-profit buffer cannot be negative")
        if tp_mode == "SR_CAPPED_R" and not self.enable_support_resistance_analysis:
            raise ValueError("S/R analysis must be enabled when S/R-capped take profit is selected")
        if tp_mode == "SR_CAPPED_R":
            enabled_profiles = [p for p in self.strategy_profiles.values() if p.enabled]
            if any(getattr(p, "partial_profit_enabled", False) for p in enabled_profiles):
                raise ValueError("S/R-capped take profit is not compatible with partial take-profit profiles")


def enhanced_default_gui_config() -> dict[str, Any]:
    return {**default_gui_config(), **ENHANCED_DEFAULTS}


def _split_values(values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = set(DEFAULT_GUI_CONFIG) | set(ENHANCED_DEFAULTS)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown/retired configuration settings: {', '.join(unknown)}")
    base = {key: value for key, value in values.items() if key in DEFAULT_GUI_CONFIG}
    extras = {**ENHANCED_DEFAULTS, **{key: values[key] for key in ENHANCED_DEFAULTS if key in values}}
    return base, extras


def build_enhanced_backtest_config(values: dict[str, Any], require_paths: bool = True) -> EnhancedBacktestConfig:
    """Build the normal config, then preserve it exactly while adding enhanced fields."""
    base_values, extras = _split_values({**enhanced_default_gui_config(), **values})
    base = build_backtest_config(base_values, require_paths=require_paths)
    base_kwargs = {field.name: getattr(base, field.name) for field in fields(BacktestConfig)}
    return EnhancedBacktestConfig(**base_kwargs, **extras)


def save_enhanced_config_json(path: str | Path, values: dict[str, Any]) -> None:
    base_values, extras = _split_values({**enhanced_default_gui_config(), **values})
    payload = canonical_config_values(base_values)
    payload.update(extras)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_enhanced_config_json(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Configuration JSON must contain an object.")
    base_values, extras = _split_values({**enhanced_default_gui_config(), **loaded})
    from crypto_strategy_lab.gui.config_logic import _merged_current

    merged = _merged_current(base_values)
    merged.update(extras)
    return merged