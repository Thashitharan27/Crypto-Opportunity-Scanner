"""Current Strategy Profile schema and strict serialization helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

PROFILE_KEYS = ("bull_long", "bull_short", "bear_long", "bear_short", "sideways_long", "sideways_short")
RULE_INDICATORS = (
    "DI_SPREAD",
    "DIRECTIONAL_DI",
    "DI_PRESSURE_STATE",
    "DI_SPREAD_CHANGE",
    "DIRECTIONAL_DI_CHANGE",
    "OPPOSING_DI_CHANGE",
    "ADX",
    "ADX_CHANGE",
    "ATR_PCT",
    "RSI",
    "BB_WIDTH",
    "CLOSE_LOCATION",
    "MOMENTUM",
    "VWAP_DISTANCE",
    "SR_NEAR_SUPPORT",
    "SR_NEAR_RESISTANCE",
    "SR_INSIDE_SUPPORT_ZONE",
    "SR_INSIDE_RESISTANCE_ZONE",
    "SR_SUPPORT_STATE",
    "SR_RESISTANCE_STATE",
    "SR_SUPPORT_HELD",
    "SR_RESISTANCE_HELD",
    "SR_TRADE_LOCATION_RATING",
    "SR_ROOM_IN_DIRECTION_ATR",
    "SR_SUPPORT_DISTANCE_ATR",
    "SR_RESISTANCE_DISTANCE_ATR",
    "SR_SUPPORT_REJECTION_ATR",
    "SR_RESISTANCE_REJECTION_ATR",
    # Lightweight causal futures research. These blocks are prepared automatically
    # when validated local Binance coverage exists, so they can safely participate
    # in explicit Entry/Veto rules without introducing a second filtering path.
    "OI_CHANGE_PCT_5M",
    "OI_CHANGE_PCT_1H",
    "OI_CHANGE_PCT_24H",
    "OI_ZSCORE_7D",
    "PRICE_CHANGE_PCT_1H",
    "OI_VS_PRICE_STATE_1H",
    "TOP_TRADER_ACCOUNT_BIAS",
    "TOP_TRADER_POSITION_BIAS",
    "GLOBAL_LONG_SHORT_ACCOUNT_BIAS",
    "TAKER_LONG_SHORT_VOLUME_BIAS",
    "FUNDING_RATE_BPS",
    "FUNDING_BIAS",
    "FUNDING_24H_SUM_BPS",
    "FUNDING_CHANGE_BPS",
    "FUNDING_3_EVENT_MEAN_BPS",
    "FUNDING_ZSCORE_7D",
    "FUNDING_EXTREME_POSITIVE",
    "FUNDING_EXTREME_NEGATIVE",
    "MARK_INDEX_BASIS_BPS",
    "MARK_INDEX_BASIS_STATE",
    "MARK_INDEX_BASIS_ZSCORE_7D",
    "TRADE_MARK_BASIS_BPS",
    "TRADE_INDEX_BASIS_BPS",
    "PREMIUM_INDEX_ZSCORE_7D",
    "TAKER_BUY_SELL_RATIO",
    "TAKER_DELTA_PCT",
    "TAKER_DELTA_PCT_15M",
    "TAKER_DELTA_PCT_1H",
    "TAKER_FLOW_PERSISTENCE",
)


@dataclass(frozen=True)
class StrategyProfile:
    enabled: bool = False
    flip_direction: bool = False
    entry_rules: tuple = ()
    flip_rule_match_mode: str = "ANY"
    reject_rule_match_mode: str = "ANY"
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
    rsi_period: int = 14
    momentum_lookback_hours: int = 24
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

    def validate(self, key: str = "profile") -> None:
        if self.reward_risk_ratio <= 0 or self.risk_multiplier <= 0:
            raise ValueError(f"{key}: reward/risk and risk multiplier must be positive")
        if self.stop_loss_multiple <= 0:
            raise ValueError(f"{key}: stop-loss multiple must be positive")
        if self.sl1_r <= 0 or self.sl2_r <= self.sl1_r:
            raise ValueError(f"{key}: SL2 must be greater than SL1")
        if self.tp1_r <= 0 or self.tp2_r <= self.tp1_r:
            raise ValueError(f"{key}: TP2 must be greater than TP1")
        if not 0 < self.sl1_close_pct < 100 or not 0 < self.tp1_close_pct < 100:
            raise ValueError(f"{key}: partial close percentages must be between 0 and 100")
        if self.flip_rule_match_mode not in ("ANY", "ALL") or self.reject_rule_match_mode not in ("ANY", "ALL"):
            raise ValueError(f"{key}: rule match modes must be ANY or ALL")
        for number, rule in enumerate(self.entry_rules, 1):
            if not isinstance(rule, dict):
                raise ValueError(f"{key}: entry rule {number} must be an object")
            if rule.get("action") not in ("FLIP", "REJECT"):
                raise ValueError(f"{key}: entry rule {number} has an invalid action")
            if rule.get("condition", "INSIDE") not in ("INSIDE", "OUTSIDE"):
                raise ValueError(f"{key}: entry rule {number} has an invalid condition")
            if rule.get("indicator") not in RULE_INDICATORS:
                raise ValueError(f"{key}: entry rule {number} has an invalid indicator")
            if float(rule.get("minimum", 0)) > float(rule.get("maximum", 0)):
                raise ValueError(f"{key}: entry rule {number} minimum must not exceed maximum")
        if self.rsi_period < 1 or self.momentum_lookback_hours < 1 or self.timeout_minutes < 1:
            raise ValueError(f"{key}: periods must be positive")
        if self.break_even_activation_r <= 0:
            raise ValueError(f"{key}: break-even activation must be positive")
        if self.break_even_offset_r < 0:
            raise ValueError(f"{key}: break-even offset cannot be negative")
        if self.trailing_activation_r <= 0 or self.trailing_distance_r <= 0:
            raise ValueError(f"{key}: trailing-stop values must be positive")
        if self.r_step_activation_r <= 0 or self.r_step_distance_r <= 0 or self.r_step_size_r <= 0:
            raise ValueError(f"{key}: R-step distances must be positive")
        if self.r_step_maximum_r < 0:
            raise ValueError(f"{key}: R-step maximum cannot be negative")
        if not 0 <= self.r_step_activation_close_pct < 100:
            raise ValueError(f"{key}: R-step activation close must be from 0% up to, but not including, 100%")
        if self.atr_checkpoint_di_spread_minimum < 0 or self.atr_checkpoint_bb_width_minimum < 0:
            raise ValueError(f"{key}: checkpoint thresholds cannot be negative")
        if self.atr_checkpoint_profit_lock_start <= 0 or self.atr_checkpoint_profit_lock_distance <= 0:
            raise ValueError(f"{key}: checkpoint profit-lock values must be positive")
        if self.r_step_trailing_enabled and self.trailing_enabled:
            raise ValueError(f"{key}: choose either R-step staircase or trailing stop")
        if self.r_step_trailing_enabled and self.partial_profit_enabled:
            raise ValueError(f"{key}: R-step staircase cannot be combined with partial take-profit")
        if self.r_step_trailing_enabled and self.atr_checkpoint_tp_extension_enabled:
            raise ValueError(f"{key}: choose either R-step staircase or ATR checkpoint extension")


def default_profiles() -> dict[str, StrategyProfile]:
    return {key: StrategyProfile(enabled=True) for key in PROFILE_KEYS}


def profiles_to_dict(profiles: dict[str, StrategyProfile]) -> dict[str, dict[str, Any]]:
    return {key: asdict(profiles.get(key, StrategyProfile())) for key in PROFILE_KEYS}


def normalize_profiles(value: Any) -> dict[str, StrategyProfile]:
    """Normalize only the current schema; legacy profile keys are rejected."""
    if value is None:
        return default_profiles()
    if not isinstance(value, dict):
        raise ValueError("strategy_profiles must be an object")
    unknown_profiles = sorted(set(value) - set(PROFILE_KEYS))
    if unknown_profiles:
        raise ValueError(f"unknown strategy profile keys: {', '.join(unknown_profiles)}")
    allowed = {field.name for field in fields(StrategyProfile)}
    result: dict[str, StrategyProfile] = {}
    for key in PROFILE_KEYS:
        raw = value.get(key, {})
        if isinstance(raw, StrategyProfile):
            profile = raw
        else:
            if not isinstance(raw, dict):
                raise ValueError(f"{key}: profile must be an object")
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise ValueError(f"{key}: unknown profile settings: {', '.join(unknown)}")
            current = dict(raw)
            current["entry_rules"] = tuple(current.get("entry_rules", ()))
            profile = StrategyProfile(**current)
        profile.validate(key)
        result[key] = profile
    return result


def profile_key(regime: str, direction: str) -> str:
    return f"{regime.lower()}_{direction.lower()}"
