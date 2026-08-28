import pandas as pd
import pytest

from crypto_strategy_lab.config import BacktestConfig, RiskMode
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.strategy_profiles import StrategyProfile, default_profiles


def candles():
    return pd.DataFrame([
        {"timestamp":pd.Timestamp("2024-01-01T00:00:00Z"),"open":100,"high":100,"low":100,"close":100,"volume":1},
        {"timestamp":pd.Timestamp("2024-01-01T00:15:00Z"),"open":100,"high":100,"low":100,"close":100,"volume":1},
    ])


def engine_with_profile(profile):
    profiles=default_profiles(); profiles["sideways_long"]=profile
    cfg=BacktestConfig(risk_mode=RiskMode.FIXED,fixed_r=10,use_intrabar_data=False,enable_trade_telemetry=False,strategy_profiles=profiles,maker_fee=0,taker_fee=0,slippage=0)
    engine=BacktestEngine(candles(),cfg)
    engine.market_regime_values[:]="SIDEWAYS"; engine.plus_di_values[:]=50; engine.minus_di_values[:]=10; engine.di_spread[:]=40
    return engine


def test_trade_r_is_full_initial_stop_distance_for_base_exit_and_break_even():
    profile=StrategyProfile(enabled=True,stop_loss_multiple=2,reward_risk_ratio=1,break_even_enabled=True,break_even_activation_r=1,break_even_offset_r=0)
    engine=engine_with_profile(profile); engine._open_pair(0); pos=engine.active_pairs[0].long
    assert pos is not None
    assert pos.distance_unit==pytest.approx(10)
    assert pos.risk==pytest.approx(20)
    assert pos.sl==pytest.approx(80)
    assert pos.tp==pytest.approx(120)
    assert not engine._maybe_activate_break_even(pos,119.9,100,pd.Timestamp("2024-01-01T00:15:00Z"))
    assert engine._maybe_activate_break_even(pos,120,100,pd.Timestamp("2024-01-01T00:15:00Z"))
    assert pos.sl==pytest.approx(100)


def test_partial_profit_and_trailing_use_full_trade_r():
    profile=StrategyProfile(enabled=True,stop_loss_multiple=2,partial_profit_enabled=True,tp1_r=1,tp1_close_pct=50,tp2_r=2,trailing_enabled=True,trailing_activation_r=1,trailing_distance_r=.5)
    engine=engine_with_profile(profile); engine._open_pair(0); pos=engine.active_pairs[0].long
    assert pos.tp1_price==pytest.approx(120)
    assert pos.tp2_price==pytest.approx(140)
    assert pos.trailing_activation_price==pytest.approx(120)
    assert pos.trailing_distance_r==pytest.approx(.5)


def test_distance_unit_checkpoint_extension_remains_distance_unit_based():
    profile=StrategyProfile(enabled=True,stop_loss_multiple=2,reward_risk_ratio=1,atr_checkpoint_tp_extension_enabled=True,atr_checkpoint_di_spread_minimum=0,atr_checkpoint_bb_width_minimum=0)
    engine=engine_with_profile(profile); engine.bb_width[:]=1; engine._open_pair(0); pos=engine.active_pairs[0].long
    assert pos.risk==pytest.approx(20)
    assert pos.distance_unit==pytest.approx(10)
    engine._apply_atr_checkpoint_extensions(pos,110,100,pd.Timestamp("2024-01-01T00:15:00Z"))
    assert pos.tp==pytest.approx(130)
