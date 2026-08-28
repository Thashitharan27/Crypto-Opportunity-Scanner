import pandas as pd
import pytest

from crypto_strategy_lab.config import BacktestConfig, RiskMode
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.strategy_profiles import StrategyProfile, default_profiles
from crypto_strategy_lab.trade import ExitReason, ExitSource, Position, Side


def engine(*, use_intrabar=False):
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [1] * 20,
        }
    )
    cfg = BacktestConfig(
        use_intrabar_data=use_intrabar,
        enable_trade_telemetry=False,
        risk_mode=RiskMode.FIXED,
        fixed_r=1,
        maker_fee=0,
        taker_fee=0,
        slippage=0,
    )
    return BacktestEngine(data, cfg)


def position(side=Side.LONG):
    p = Position(
        side,
        pd.Timestamp("2024-01-01", tz="UTC"),
        0,
        100,
        1,
        98 if side == Side.LONG else 102,
        103 if side == Side.LONG else 97,
        1,
        1,
        100,
        1,
        original_sl=98 if side == Side.LONG else 102,
    )
    p.trailing_enabled = True
    p.trailing_activation_price = 103 if side == Side.LONG else 97
    p.trailing_distance_r = 1
    p.favourable_price = 100
    return p


@pytest.mark.parametrize(
    "side,activation_bar,reversal_bar,expected",
    [
        (Side.LONG, (103, 102), (102.5, 102), 102),
        (Side.SHORT, (98, 97), (98, 97.5), 98),
    ],
)
def test_profile_trailing_activation_defers_new_stop_until_next_bar(side, activation_bar, reversal_bar, expected):
    e = engine()
    p = position(side)
    activation_time = p.entry_time + pd.Timedelta(minutes=15)
    exit_time = p.entry_time + pd.Timedelta(minutes=30)

    assert not e._maybe_exit_bar(p, 1, *activation_bar, activation_time, ExitSource.FALLBACK_15M)
    assert p.trailing_active
    assert p.trailing_activation_time == activation_time
    assert e._maybe_exit_bar(p, 2, *reversal_bar, exit_time, ExitSource.FALLBACK_15M)
    assert p.exit_reason == ExitReason.TRAILING_STOP
    assert p.exit_price == pytest.approx(expected)


@pytest.mark.parametrize(
    "side,first,second",
    [
        (Side.LONG, (105, 104.5), (104.5, 104.2)),
        (Side.SHORT, (95.5, 95), (95.8, 95.5)),
    ],
)
def test_trailing_stop_is_monotonic(side, first, second):
    e = engine()
    p = position(side)
    p.trailing_active = True

    assert not e._maybe_exit_bar(p, 1, *first, p.entry_time + pd.Timedelta(minutes=15), ExitSource.FALLBACK_15M)
    old = p.trailing_stop
    assert not e._maybe_exit_bar(p, 2, *second, p.entry_time + pd.Timedelta(minutes=30), ExitSource.FALLBACK_15M)
    assert p.trailing_stop >= old if side == Side.LONG else p.trailing_stop <= old


def test_fixed_tp_is_used_when_position_trailing_is_disabled():
    e = engine()
    p = position()
    p.trailing_enabled = False
    assert e._maybe_exit_bar(p, 1, 103, 99, p.entry_time + pd.Timedelta(minutes=15), ExitSource.FALLBACK_15M)
    assert p.exit_reason == ExitReason.TP


def test_strategy_profile_owns_trailing_activation_and_distance():
    profiles = default_profiles()
    profiles["sideways_long"] = StrategyProfile(
        enabled=True,
        trailing_enabled=True,
        trailing_activation_r=3,
        trailing_distance_r=0.75,
    )
    e = engine()
    e.config = BacktestConfig(
        use_intrabar_data=False,
        enable_trade_telemetry=False,
        risk_mode=RiskMode.FIXED,
        fixed_r=2,
        maker_fee=0,
        taker_fee=0,
        slippage=0,
        strategy_profiles=profiles,
    )
    e.market_regime_values[:] = "SIDEWAYS"
    e.plus_di_values[:] = 50
    e.minus_di_values[:] = 10
    e.di_spread[:] = 40
    e._open_pair(0)
    pair = e.active_pairs[0]
    assert pair.long is not None and pair.short is None
    assert pair.long.trailing_enabled
    assert pair.long.trailing_activation_price == pytest.approx(pair.long.entry_price + pair.long.risk * 3)
    assert pair.long.trailing_distance_r == pytest.approx(0.75)


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
@pytest.mark.parametrize("activation_offset,exit_offset", [(5, 10), (45, 75), (3 * 24 * 60, 4 * 24 * 60)])
def test_trailing_exit_reports_real_trigger_and_exit_timestamps(side, activation_offset, exit_offset):
    e = engine()
    p = position(side)
    activation = p.entry_time + pd.Timedelta(minutes=activation_offset)
    exit_time = p.entry_time + pd.Timedelta(minutes=exit_offset)
    favourable = (104, 102) if side == Side.LONG else (98, 96)
    reversal = (103, 102) if side == Side.LONG else (98, 97)

    assert not e._maybe_exit_bar(p, 1, *favourable, activation, ExitSource.INTRABAR)
    assert e._maybe_exit_bar(p, 2, *reversal, exit_time, ExitSource.INTRABAR)

    assert p.exit_reason == ExitReason.TRAILING_STOP
    assert p.trailing_activation_time == activation
    assert p.exit_time == exit_time
    assert p.exit_time >= p.trailing_activation_time


def test_intrabar_scan_does_not_replay_historical_candles():
    e = engine(use_intrabar=True)
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    e.intrabar_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=61, freq="1min"),
            "open": [100.0] * 61,
            "high": [101.0] * 61,
            "low": [99.0] * 61,
            "close": [100.0] * 61,
        }
    )
    p = position()
    p.entry_time = start + pd.Timedelta(minutes=15)
    e.intrabar_data.loc[31:45, ["high", "low"]] = [104.0, 104.0]
    e.intrabar_data.loc[46, ["high", "low"]] = [103.0, 102.0]

    assert not e._scan_exit(p, 2)
    assert p.trailing_activation_time == start + pd.Timedelta(minutes=31)
    assert e._scan_exit(p, 3)
    assert p.exit_time == start + pd.Timedelta(minutes=46)
