import pandas as pd
import pytest

from crypto_strategy_lab.config import BacktestConfig, RiskMode, TiePolicy
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.strategy_profiles import StrategyProfile, default_profiles


def candles(*bars):
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=15 * i),
                "open": 100,
                "close": 100,
                "high": high,
                "low": low,
                "volume": 1,
            }
            for i, (high, low) in enumerate(bars)
        ]
    )


def open_long(*bars, profile=None, tie_policy=TiePolicy.PESSIMISTIC):
    profiles = default_profiles()
    profiles["sideways_long"] = profile or StrategyProfile(
        enabled=True,
        partial_stop_enabled=True,
        sl1_r=0.5,
        sl1_close_pct=50,
        sl2_r=2,
        reward_risk_ratio=2,
    )
    cfg = BacktestConfig(
        risk_mode=RiskMode.FIXED,
        fixed_r=10,
        atr_period=1,
        use_intrabar_data=False,
        enable_trade_telemetry=False,
        strategy_profiles=profiles,
        maker_fee=0,
        taker_fee=0,
        slippage=0,
        tie_policy=tie_policy,
    )
    engine = BacktestEngine(candles(*bars), cfg)
    engine.market_regime_values[:] = "SIDEWAYS"
    engine.plus_di_values[:] = 50
    engine.minus_di_values[:] = 10
    engine.di_spread[:] = 40
    engine._open_pair(0)
    pair = engine.active_pairs[0]
    assert pair.long is not None and pair.short is None
    return engine, pair.long


def weighted_price_r(position):
    return position.gross_pnl / (position.risk * position.original_quantity)


def test_sl1_then_sl2_closes_current_profile_position_at_weighted_loss():
    engine, position = open_long((100, 100), (100, 95), (100, 80))
    assert engine._scan_exit(position, 1)
    assert position.sl1_hit and position.is_open
    assert engine._scan_exit(position, 2)
    assert position.final_exit_reason == "SL1_THEN_SL2"
    assert weighted_price_r(position) == pytest.approx(-0.625)


def test_sl1_then_profile_target_closes_remainder():
    engine, position = open_long((100, 100), (100, 95), (140, 100))
    assert engine._scan_exit(position, 1)
    assert position.sl1_hit and position.is_open
    assert engine._scan_exit(position, 2)
    assert position.final_exit_reason == "SL1_THEN_TP"
    assert weighted_price_r(position) == pytest.approx(0.875)


def test_profile_partial_stop_requires_ordered_levels():
    profile = StrategyProfile(partial_stop_enabled=True, sl1_r=2, sl2_r=2)
    with pytest.raises(ValueError, match="SL2 must be greater than SL1"):
        profile.validate("sideways_long")


def test_profile_partial_stop_plan_owns_stop_levels_and_quantities():
    engine, position = open_long((100, 100), (100, 100))
    assert position.partial_sl_enabled
    assert position.sl1_price == pytest.approx(95)
    assert position.sl2_price == pytest.approx(80)
    assert position.sl == pytest.approx(80)
    assert position.sl1_quantity == pytest.approx(position.original_quantity * 0.5)


def test_pessimistic_same_bar_target_and_sl2_resolves_to_losses_first():
    engine, position = open_long((100, 100), (140, 80), tie_policy=TiePolicy.PESSIMISTIC)
    assert engine._scan_exit(position, 1)
    assert position.final_exit_reason == "SL1_THEN_SL2"


def test_optimistic_same_bar_target_and_sl2_resolves_to_target_first():
    engine, position = open_long((100, 100), (140, 80), tie_policy=TiePolicy.OPTIMISTIC)
    assert engine._scan_exit(position, 1)
    assert position.final_exit_reason == "TP"
    assert not position.sl1_hit
