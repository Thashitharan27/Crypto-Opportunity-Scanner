"""Presentation metadata for the native v3 research configuration.

This module deliberately contains no research logic.  Values are converted only
at the widget boundary; the dataclasses continue to store their canonical units.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy


@dataclass(frozen=True)
class FieldPresentation:
    label: str
    help: str = ""
    unit: str = ""
    scale: float = 1.0
    decimals: int = 4

    def display(self, value: float) -> float:
        return value * self.scale

    def native(self, value: float) -> float:
        return value / self.scale


PERCENT_FIELDS = {"risk_per_leg", "maker_fee", "taker_fee", "slippage", "percent_r"}
FIELDS = {
    "initial_equity": FieldPresentation("Starting Equity", unit="$", decimals=2),
    "risk_per_leg": FieldPresentation("Base Risk Per Trade", "Fraction of equity risked at a full stop.", "%", 100, 4),
    "maker_fee": FieldPresentation("Maker Fee", unit="%", scale=100, decimals=4),
    "taker_fee": FieldPresentation("Taker Fee", unit="%", scale=100, decimals=4),
    "slippage": FieldPresentation("Slippage", unit="%", scale=100, decimals=4),
    "percent_r": FieldPresentation("Price Distance", unit="%", scale=100, decimals=4),
    "reward_risk_ratio": FieldPresentation("Profit Target", unit=" R", decimals=2),
    "stop_loss_multiple": FieldPresentation("Stop Distance", unit=" distance units", decimals=2),
    "risk_multiplier": FieldPresentation("Risk Multiplier", unit="x", decimals=2),
    "timeout_minutes": FieldPresentation("Maximum Holding Time", unit=" min", decimals=0),
    "strategy_profile_run_mode": FieldPresentation("Profile Test Mode"),
    "tie_policy": FieldPresentation("Same-bar Resolution"),
}

ENUM_LABELS = {
    "strategy_profile_run_mode": {"COMBINED_SHARED_CAPITAL": "Combined — Shared Account", "ISOLATED_PROFILES": "Each Profile Independently", "BOTH": "Combined + Independent Comparison"},
    "tie_policy": {"PESSIMISTIC": "Conservative — Stop First", "OPTIMISTIC": "Optimistic — Target First", "INTRABAR": "Resolve Using Intrabar Data"},
    "market_regime_method": {"BTC_STRUCTURAL": "BTC Structural Trend", "ASSET_STRUCTURAL": "Selected Asset Structural Trend", "ASSET_RETURN": "Selected Asset Trailing Return"},
    "sr_filter_mode": {"ANALYSIS_ONLY": "Analysis Only — Do Not Block Trades", "APPLY_ENTRY_RULES": "Use S/R Entry Filters"},
    "risk_mode": {"ATR": "ATR Volatility", "PERCENT": "Percent of Price", "FIXED": "Fixed Price Distance"},
    "trade_flow_source": {"AGG_TRADES": "Aggregate Trades", "TRADES": "Trades"},
    "flip_rule_match_mode": {"ANY": "Any Rule (OR)", "ALL": "All Rules (AND)"},
    "reject_rule_match_mode": {"ANY": "Any Rule (OR)", "ALL": "All Rules (AND)"},
    "sr_take_profit_mode": {"FIXED_R": "Fixed R Target", "SR_CAPPED_R": "Cap Target at Support/Resistance"},
    "sr_take_profit_no_level_policy": {"USE_FIXED_TP": "Use Fixed Target When No Level Exists", "REJECT_TRADE": "Reject Trade When No Valid Level Exists"},
}

PROFILE_LABELS = {"bull_long": "Bull Long", "bull_short": "Bull Short", "bear_long": "Bear Long", "bear_short": "Bear Short", "sideways_long": "Sideways Long", "sideways_short": "Sideways Short"}


def metadata(name: str) -> FieldPresentation:
    return FIELDS.get(name, FieldPresentation(name.replace("_", " ").title()))


def display_percentage(native: float, decimals: int = 2) -> str:
    return f"{native * 100:.{decimals}f}%"


def parse_percentage(display: str | float) -> float:
    return float(str(display).strip().removesuffix("%").strip()) / 100


REPORT_PRESETS = {
    "QUICK": dict(analysis_level="QUICK", enable_trade_telemetry=False, save_full_telemetry_csv=False,
        save_trade_journey_summary=False, save_trade_journey_charts=False,
        enable_indicator_lifecycle_analysis=False, create_lifecycle_charts=False,
        save_feature_analysis_reports=False, save_indicator_analysis_reports=False, create_standard_charts=False),
    "STANDARD": dict(analysis_level="STANDARD", enable_trade_telemetry=False, save_full_telemetry_csv=False,
        save_trade_journey_summary=False, save_trade_journey_charts=False,
        enable_indicator_lifecycle_analysis=False, create_lifecycle_charts=False,
        save_feature_analysis_reports=False, save_indicator_analysis_reports=True, create_standard_charts=True),
    "DEEP_RESEARCH": dict(analysis_level="DEEP", enable_trade_telemetry=True, save_full_telemetry_csv=True,
        save_trade_journey_summary=True, save_trade_journey_charts=True,
        enable_indicator_lifecycle_analysis=True, create_lifecycle_charts=True,
        save_feature_analysis_reports=True, save_indicator_analysis_reports=True, create_standard_charts=True),
}


def apply_report_preset(reporting, preset: str):
    """Return a new ReportingConfig using an explicit deterministic mapping."""
    return replace(reporting, **REPORT_PRESETS[preset])


def clone_profile_pair(strategy, execution):
    """Copy profile values without sharing mutable rule payloads."""
    return deepcopy(strategy), deepcopy(execution)
