import pandas as pd
import pytest

from crypto_strategy_lab.config import BacktestConfig, EntryMode, RiskMode
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.strategy_profiles import (
    PROFILE_KEYS,
    StrategyProfile,
    default_profiles,
    normalize_profiles,
    profiles_to_dict,
)


def rising_candles(count=160):
    timestamps = pd.date_range("2024-01-01", periods=count, freq="60min", tz="UTC")
    close = [100 + i * 0.5 for i in range(count)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [v + 1 for v in close],
            "low": [v - 1 for v in close],
            "close": close,
            "volume": 1,
        }
    )


def profile_config(**changes):
    profiles = default_profiles()
    profiles["bull_long"] = StrategyProfile(
        enabled=True,
        reward_risk_ratio=2.5,
        risk_multiplier=1.5,
        entry_rules=(
            {"action": "REJECT", "indicator": "DI_SPREAD", "condition": "OUTSIDE", "minimum": 0, "maximum": 1000},
        ),
    )
    base = dict(
        risk_mode=RiskMode.FIXED,
        fixed_r=2,
        entry_mode=EntryMode.WAIT_UNTIL_CLOSED,
        strategy_timeframe_minutes=60,
        telemetry_interval_minutes=60,
        initial_equity=1000,
        risk_per_leg=0.01,
        taker_fee=0,
        maker_fee=0,
        slippage=0,
        bull_regime_lookback_days=1,
        bull_regime_return_threshold=0.01,
        strategy_profiles=profiles,
        use_intrabar_data=False,
    )
    base.update(changes)
    return BacktestConfig(**base)


def test_profile_serialization_round_trip_preserves_all_six_profiles():
    profiles = default_profiles()
    profiles["bear_short"] = StrategyProfile(
        enabled=True,
        entry_rules=(
            {"action": "REJECT", "indicator": "ADX", "condition": "OUTSIDE", "minimum": 15, "maximum": 40},
        ),
    )
    restored = normalize_profiles(profiles_to_dict(profiles))
    assert tuple(restored) == tuple(PROFILE_KEYS)
    assert restored["bear_short"].enabled
    assert restored["bear_short"].entry_rules == (
        {"action": "REJECT", "indicator": "ADX", "condition": "OUTSIDE", "minimum": 15, "maximum": 40},
    )


def test_legacy_profile_fields_are_rejected_instead_of_migrated():
    raw = profiles_to_dict(default_profiles())
    raw["bull_long"]["adx_enabled"] = True
    with pytest.raises(ValueError, match="unknown profile settings: adx_enabled"):
        normalize_profiles(raw)


def test_profile_reward_risk_and_risk_multiplier_are_applied_to_trade():
    engine = BacktestEngine(rising_candles(), profile_config())
    i = len(engine.data) - 1
    passed, reason = engine._entry_filter_result(i)
    assert passed
    engine._open_pair(i, passed, reason)
    pair = engine.active_pairs[0]
    assert pair.strategy_profile_key == "bull_long"
    assert pair.di_applied_long_reward_risk_ratio == pytest.approx(2.5)
    assert pair.long is not None
    assert pair.short is None
    assert pair.long.risk_amount == pytest.approx(15)
    stop_distance = pair.long.entry_price - pair.long.sl
    target_distance = pair.long.tp - pair.long.entry_price
    assert target_distance == pytest.approx(2.5 * stop_distance)


def test_profile_can_flip_its_di_entry_direction():
    profiles = default_profiles()
    profiles["bull_long"] = StrategyProfile(enabled=True, flip_direction=True)
    engine = BacktestEngine(rising_candles(), profile_config(strategy_profiles=profiles))
    i = len(engine.data) - 1
    passed, reason = engine._entry_filter_result(i)
    assert passed

    engine._open_pair(i, passed, reason)

    pair = engine.active_pairs[0]
    assert pair.di_sizing_direction == "LONG"
    assert pair.sizing_direction == "SHORT"
    assert pair.long is None
    assert pair.short is not None


def test_profile_flip_rules_match_any_and_keep_nonmatches_normal():
    profiles = default_profiles()
    profiles["bull_long"] = StrategyProfile(
        enabled=True,
        entry_rules=(
            {"action": "FLIP", "indicator": "ADX", "condition": "INSIDE", "minimum": 10, "maximum": 20},
            {"action": "FLIP", "indicator": "CLOSE_LOCATION", "condition": "INSIDE", "minimum": 0.45, "maximum": 0.68},
            {"action": "FLIP", "indicator": "RSI", "condition": "INSIDE", "minimum": 30, "maximum": 40},
        ),
        flip_rule_match_mode="ANY",
    )

    matching = BacktestEngine(rising_candles(), profile_config(strategy_profiles=profiles))
    i = len(matching.data) - 1
    matching.adx_values[i] = 15
    passed, reason = matching._entry_filter_result(i)
    assert passed and "will be flipped" in reason
    matching._open_pair(i, passed, reason)
    assert matching.active_pairs[0].sizing_direction == "SHORT"

    normal = BacktestEngine(rising_candles(), profile_config(strategy_profiles=profiles))
    normal.adx_values[i] = 25
    normal.close_location_values[i] = 0.9
    passed, reason = normal._entry_filter_result(i)
    assert passed and "normal direction" in reason
    normal._open_pair(i, passed, reason)
    assert normal.active_pairs[0].sizing_direction == "LONG"


def test_reject_rules_take_precedence_over_flip_rules():
    profiles = default_profiles()
    profiles["bull_long"] = StrategyProfile(
        enabled=True,
        entry_rules=(
            {"action": "FLIP", "indicator": "ADX", "condition": "INSIDE", "minimum": 10, "maximum": 20},
            {"action": "REJECT", "indicator": "CLOSE_LOCATION", "condition": "INSIDE", "minimum": 0.4, "maximum": 0.6},
        ),
    )
    engine = BacktestEngine(rising_candles(), profile_config(strategy_profiles=profiles))
    i = len(engine.data) - 1
    engine.adx_values[i] = 15
    passed, reason = engine._entry_filter_result(i)
    assert not passed
    assert "rejected by entry rules" in reason


def test_disabled_profile_rejects_its_regime_direction():
    config = profile_config(strategy_profiles={key: StrategyProfile() for key in PROFILE_KEYS})
    engine = BacktestEngine(rising_candles(), config)
    passed, reason = engine._entry_filter_result(len(engine.data) - 1)
    assert not passed
    assert "bull_long is disabled" in reason


def test_profile_owns_stop_and_partial_profit_exit_plan():
    profiles = default_profiles()
    profiles["bull_long"] = StrategyProfile(
        enabled=True,
        reward_risk_ratio=9,
        risk_multiplier=1,
        stop_loss_multiple=1.5,
        partial_profit_enabled=True,
        tp1_r=1,
        tp1_close_pct=40,
        tp2_r=3,
        break_even_enabled=True,
        break_even_activation_r=1,
        break_even_offset_r=0,
    )
    engine = BacktestEngine(rising_candles(), profile_config(strategy_profiles=profiles))
    engine._open_pair(len(engine.data) - 1)
    pair = engine.active_pairs[0]
    position = pair.long
    assert position is not None and position.partial_tp_enabled
    assert position.entry_price - position.sl == pytest.approx(3)
    assert position.tp1_price - position.entry_price == pytest.approx(3)
    assert position.tp2_price - position.entry_price == pytest.approx(9)
    assert position.tp1_quantity / position.original_quantity == pytest.approx(0.40)
    assert position.profile_break_even_activation_r == pytest.approx(1)


def test_removed_post_tp1_stop_field_is_rejected():
    raw = profiles_to_dict(default_profiles())
    raw["bull_long"]["after_tp1_stop_mode"] = "MOVE_TO_ENTRY"
    with pytest.raises(ValueError, match="unknown profile settings: after_tp1_stop_mode"):
        normalize_profiles(raw)
