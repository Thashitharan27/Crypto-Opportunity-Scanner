"""Authoritative conservative warm-up projection for native research requests."""
from __future__ import annotations

from datetime import timedelta
import math


WARMUP_POLICY_VERSION = "NATIVE_CONFIG_LOOKBACK_V1"


def strategy_warmup_period(config) -> timedelta:
    """Cover configured engine lookbacks without changing feature semantics."""
    enabled = [p for p in config.strategy_profiles.values() if getattr(p, "enabled", True)]
    profile_rsi = max((p.rsi_period for p in enabled), default=14)
    momentum_hours = max((p.momentum_lookback_hours for p in enabled), default=24)
    strategy_minutes = int(config.strategy_timeframe_minutes)
    base_bars = max(
        int(config.atr_period) + 5,
        int(config.adx_period) * 2 + 5,
        int(config.di_pressure_lookback) + 6,
        int(config.bb_period) + 5,
        int(config.mean_reversion_period) + 5,
        int(getattr(config, "mean_reversion_rsi_period", profile_rsi)) + 5,
        int(profile_rsi) + 5,
        30,
    )
    minutes = max(base_bars * strategy_minutes, momentum_hours * 60, 22 * 1440)
    if config.market_regime_method == "ASSET_RETURN":
        minutes = max(minutes, (int(config.bull_regime_lookback_days) + 2) * 1440)
    if config.enable_support_resistance_analysis:
        sr_minutes = int(config.sr_timeframe_minutes or strategy_minutes)
        sr_bars = (
            int(config.sr_lookback_bars) + int(config.sr_pivot_left) +
            int(config.sr_pivot_right) +
            (int(config.sr_hold_confirmation_bars) if config.enable_sr_hold_confirmation else 0) + 5
        )
        minutes = max(minutes, sr_bars * sr_minutes)
    # Two calendar days preserve the mature GUI/native boundary's safety margin.
    return timedelta(days=math.ceil(minutes / 1440) + 2)


def validation_warmup_bars(run_config, registry) -> int:
    """Combine configured native lookbacks with registry dependency warm-up."""
    from .research_adapters import native_simulator_config
    native = native_simulator_config(
        run_config.data, run_config.features, run_config.strategy, run_config.execution
    )
    configured = strategy_warmup_period(native)
    strategy_minutes = int(run_config.data.strategy_timeframe_minutes)
    configured_bars = math.ceil(configured.total_seconds() / 60 / strategy_minutes)
    registry_bars = registry.effective_warmup(registry.names())
    return max(configured_bars, int(registry_bars))
