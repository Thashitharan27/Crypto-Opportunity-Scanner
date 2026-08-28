from __future__ import annotations

import pandas as pd
import pytest

from crypto_strategy_lab.trade import ExitReason, Position, Side, TradePair


def _position(side: Side) -> Position:
    return Position(
        side=side,
        entry_time=pd.Timestamp("2026-01-01T00:00:00Z"),
        entry_index=0,
        entry_price=100.0,
        risk=1.0,
        sl=99.0 if side == Side.LONG else 101.0,
        tp=102.0 if side == Side.LONG else 98.0,
        quantity=1.0,
        risk_amount=1.0,
        entry_notional=100.0,
        atr_at_entry=1.0,
    )


def _pair(long=None, short=None) -> TradePair:
    return TradePair(
        pair_id=1,
        long=long,
        short=short,
        equity_before_trade=1000.0,
        strategy_candle_open_time=pd.Timestamp("2026-01-01T00:00:00Z"),
        strategy_entry_time=pd.Timestamp("2026-01-01T00:15:00Z"),
        strategy_entry_price=100.0,
    )


def test_trade_pair_requires_exactly_one_position() -> None:
    long = _position(Side.LONG)
    short = _position(Side.SHORT)

    assert _pair(long=long).position is long
    assert _pair(short=short).position is short

    with pytest.raises(ValueError, match="exactly one position"):
        _pair()
    with pytest.raises(ValueError, match="simultaneous LONG\\+SHORT trades are retired"):
        _pair(long=long, short=short)


def test_positions_is_single_leg_compatibility_tuple() -> None:
    long = _position(Side.LONG)
    short = _position(Side.SHORT)

    long_pair = _pair(long=long)
    short_pair = _pair(short=short)
    assert long_pair.positions() == (long,)
    assert short_pair.positions() == (short,)
    assert long_pair.positions() is long_pair.positions()
    assert short_pair.positions() is short_pair.positions()


def test_is_open_tracks_the_single_position() -> None:
    long = _position(Side.LONG)
    pair = _pair(long=long)
    assert pair.is_open

    long.exit_time = pd.Timestamp("2026-01-01T00:20:00Z")
    assert not pair.is_open

    short = _position(Side.SHORT)
    pair = _pair(short=short)
    assert pair.is_open

    short.exit_time = pd.Timestamp("2026-01-01T00:20:00Z")
    assert not pair.is_open


def test_trade_result_timestamps_normalize_naive_values_to_utc() -> None:
    position = _position(Side.LONG)
    pair = TradePair(
        pair_id=1,
        long=position,
        short=None,
        equity_before_trade=1000.0,
        strategy_candle_open_time=pd.Timestamp("2026-01-01T00:00:00"),
        strategy_entry_time=pd.Timestamp("2026-01-01T00:15:00"),
        strategy_entry_price=100.0,
    )
    position.exit_time = pd.Timestamp("2026-01-01T00:20:00")

    assert str(pair.strategy_candle_open_time) == "2026-01-01 00:00:00+00:00"
    assert str(pair.strategy_entry_time) == "2026-01-01 00:15:00+00:00"
    assert str(position.exit_time) == "2026-01-01 00:20:00+00:00"


def test_runtime_leg_mutation_cannot_resurrect_dual_leg_state() -> None:
    pair = _pair(long=_position(Side.LONG))
    pair.short = _position(Side.SHORT)

    with pytest.raises(ValueError, match="simultaneous LONG\\+SHORT trades are retired"):
        _ = pair.position
    with pytest.raises(ValueError, match="simultaneous LONG\\+SHORT trades are retired"):
        pair.positions()


def test_retired_dual_leg_state_is_absent_from_models() -> None:
    pair = _pair(long=_position(Side.LONG))

    retired_pair_fields = (
        "pair_be_triggered",
        "remaining_leg_timeout_after_first_sl_started",
        "first_sl_side",
        "first_sl_time",
        "remaining_leg_timeout_deadline",
        "remaining_leg_timeout_triggered",
        "remaining_leg_timeout_exit_time",
        "remaining_leg_timeout_exit_side",
        "remaining_leg_timeout_checkpoint_count",
        "remaining_leg_timeout_extension_count",
        "remaining_leg_timeout_last_checkpoint_time",
        "remaining_leg_timeout_last_checkpoint_profit_r",
        "checkpoint_score_last_atr_pct",
        "checkpoint_score_last_directional_di",
        "checkpoint_score_last_bb_width_pct",
        "checkpoint_score_last_pass_count",
        "checkpoint_score_last_condition_count",
        "checkpoint_score_last_passed",
        "checkpoint_zero_score_streak",
        "checkpoint_zero_score_max_streak",
        "checkpoint_zero_score_last_time",
        "checkpoint_zero_score_confirmed_close",
        "first_sl_survivor_partial_taken",
        "first_sl_survivor_partial_side",
        "first_sl_survivor_partial_time",
        "first_sl_survivor_partial_pct",
        "first_sl_survivor_partial_quantity",
        "first_sl_survivor_partial_exit_price",
        "first_sl_survivor_partial_gross_pnl",
        "first_sl_survivor_partial_fee",
        "first_sl_survivor_partial_net_pnl",
        "checkpoint_reentry_gate_started",
        "checkpoint_reentry_gate_side",
        "checkpoint_reentry_gate_tp",
        "checkpoint_reentry_gate_sl",
        "checkpoint_reentry_gate_start_time",
        "checkpoint_reentry_gate_release_time",
        "checkpoint_reentry_gate_release_reason",
        "both_open_timeout_triggered",
    )
    for name in retired_pair_fields:
        assert not hasattr(pair, name), name

    assert not hasattr(ExitReason, "REMAINING_LEG_TIMEOUT_AFTER_FIRST_SL")
    assert not hasattr(ExitReason, "BOTH_OPEN_TIMEOUT")


def test_profile_timeout_uses_only_canonical_state_and_reason() -> None:
    pair = _pair(long=_position(Side.LONG))

    assert ExitReason.PROFILE_TIMEOUT.value == "PROFILE_TIMEOUT"
    assert pair.profile_timeout_triggered is False

    pair.profile_timeout_triggered = True

    assert pair.profile_timeout_triggered is True
