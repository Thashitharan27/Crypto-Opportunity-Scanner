from dataclasses import asdict

import pytest

from crypto_strategy_lab.strategy_profiles import StrategyProfile, normalize_profiles
from crypto_strategy_lab.trade import Position, Side


def test_special_exit_management_is_profile_serializable():
    profile = StrategyProfile(
        enabled=True,
        r_step_trailing_enabled=True,
        r_step_activation_r=2.5,
        r_step_distance_r=1.25,
        r_step_size_r=0.5,
        r_step_maximum_r=8.0,
        r_step_activation_close_pct=20.0,
    )
    restored = normalize_profiles({"bull_long": asdict(profile)})["bull_long"]
    assert restored.r_step_trailing_enabled is True
    assert restored.r_step_activation_r == 2.5
    assert restored.r_step_activation_close_pct == 20.0


def test_checkpoint_extension_is_profile_specific():
    profiles = normalize_profiles({
        "bull_long": {"enabled": True, "atr_checkpoint_tp_extension_enabled": True, "atr_checkpoint_di_spread_minimum": 35.0},
        "bear_short": {"enabled": True},
    })
    assert profiles["bull_long"].atr_checkpoint_tp_extension_enabled is True
    assert profiles["bull_long"].atr_checkpoint_di_spread_minimum == 35.0
    assert profiles["bear_short"].atr_checkpoint_tp_extension_enabled is False


def test_conflicting_profile_exit_managers_are_rejected():
    with pytest.raises(ValueError):
        StrategyProfile(r_step_trailing_enabled=True, trailing_enabled=True).validate("bull_long")
    with pytest.raises(ValueError):
        StrategyProfile(r_step_trailing_enabled=True, atr_checkpoint_tp_extension_enabled=True).validate("bull_long")


def test_position_carries_profile_exit_parameters():
    pos = Position(Side.LONG, None, 0, 100.0, 1.0, 98.0, 105.0, 1.0, 1.0, 100.0, 1.0)
    pos.r_step_activation_r = 2.5
    pos.atr_checkpoint_di_spread_minimum = 40.0
    assert pos.r_step_activation_r == 2.5
    assert pos.atr_checkpoint_di_spread_minimum == 40.0
