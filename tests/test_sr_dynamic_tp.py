from types import SimpleNamespace

import numpy as np
import pytest

from crypto_strategy_lab.gui.enhanced_config import (
    build_enhanced_backtest_config,
    enhanced_default_gui_config,
)
from crypto_strategy_lab.sr_dynamic_tp_engine import SRDynamicTPBacktestEngine


def test_dynamic_tp_defaults_preserve_fixed_r_baseline():
    values = enhanced_default_gui_config()
    assert values["sr_take_profit_mode"] == "FIXED_R"
    assert values["sr_take_profit_maximum_r"] == 3.0
    assert values["sr_take_profit_minimum_r"] == 1.5
    assert values["sr_take_profit_buffer_r"] == 0.20
    assert values["sr_take_profit_no_level_policy"] == "USE_FIXED_TP"


def test_dynamic_tp_config_validates_minimum_not_above_maximum():
    values = enhanced_default_gui_config()
    values.update(
        {
            "sr_take_profit_mode": "SR_CAPPED_R",
            "enable_support_resistance_analysis": True,
            "sr_take_profit_minimum_r": 3.5,
            "sr_take_profit_maximum_r": 3.0,
        }
    )
    with pytest.raises(ValueError, match="minimum TP cannot exceed maximum TP"):
        build_enhanced_backtest_config(values, require_paths=False)


def _fake_engine(resistance=102.0, support=98.0):
    engine = object.__new__(SRDynamicTPBacktestEngine)
    engine.config = SimpleNamespace(
        sr_take_profit_mode="SR_CAPPED_R",
        sr_take_profit_maximum_r=3.0,
        sr_take_profit_minimum_r=1.5,
        sr_take_profit_buffer_r=0.20,
        sr_take_profit_no_level_policy="USE_FIXED_TP",
        enable_daily_entry_schedule=False,
        slippage=0.0,
    )
    engine.risk = np.array([1.0])
    engine.close = np.array([100.0])
    engine.open = np.array([100.0])
    profile = SimpleNamespace(partial_stop_enabled=False, stop_loss_multiple=1.0)
    engine._profile_context = lambda _i: ("BULL", "LONG", "BULL_LONG", profile)
    engine._effective_trade_direction = lambda _i: "LONG"
    engine._analyze_support_resistance = lambda _i, _direction: SimpleNamespace(
        nearest_resistance_price=resistance,
        nearest_support_price=support,
    )
    return engine


def test_sr_capped_tp_allows_trade_when_room_after_buffer_meets_minimum():
    engine = _fake_engine(resistance=102.0)
    passed, reason = engine._sr_tp_filter_result(0)
    assert passed is True
    assert reason is None


def test_sr_capped_tp_rejects_trade_when_room_after_buffer_is_too_small():
    engine = _fake_engine(resistance=101.5)
    passed, reason = engine._sr_tp_filter_result(0)
    assert passed is False
    assert reason == "SR_TP_INSUFFICIENT_ROOM"


def test_sr_capped_tp_can_reject_when_no_opposing_level_exists():
    engine = _fake_engine(resistance=None)
    engine.config.sr_take_profit_no_level_policy = "REJECT_TRADE"
    passed, reason = engine._sr_tp_filter_result(0)
    assert passed is False
    assert reason == "SR_TP_NO_OPPOSING_LEVEL"
