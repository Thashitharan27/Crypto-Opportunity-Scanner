from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pandas.testing as pdt

from crypto_strategy_lab.config import EntryMode, IntrabarMissingPolicy, RiskMode
from crypto_strategy_lab.data.intrabar_index import as_searchsorted_intrabar
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.gui.enhanced_config import EnhancedBacktestConfig
from crypto_strategy_lab.sr_dynamic_tp_engine import SRDynamicTPBacktestEngine
from crypto_strategy_lab.trade import ExitReason, Position, Side, TradePair


def _strategy_frame(periods: int = 12) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1.0,
        }
    )


def _intrabar_frame(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1.0,
        }
    )


def _config() -> EnhancedBacktestConfig:
    return EnhancedBacktestConfig(
        strategy_timeframe_minutes=15,
        intrabar_timeframe_minutes=1,
        telemetry_interval_minutes=15,
        use_intrabar_data=True,
        intrabar_missing_policy=IntrabarMissingPolicy.ERROR,
        risk_mode=RiskMode.FIXED,
        fixed_r=1.0,
        atr_period=2,
        adx_period=2,
        bb_period=2,
        mean_reversion_period=2,
        entry_mode=EntryMode.EVERY_N_CANDLES,
        max_active_pairs=1,
        maker_fee=0.0,
        taker_fee=0.0,
        slippage=0.0,
        enable_trade_telemetry=False,
        market_regime_method="ASSET_RETURN",
        bull_regime_lookback_days=1,
        enable_support_resistance_analysis=False,
    )


def _force_long_sideways(engine) -> None:
    engine.market_regime_values[:] = "SIDEWAYS"
    engine.plus_di_values[:] = 50.0
    engine.minus_di_values[:] = 10.0
    engine.di_spread[:] = 40.0
    engine.di_ratio[:] = 5.0


def test_profile_timeout_is_the_only_timeout_exit_reason() -> None:
    assert ExitReason.PROFILE_TIMEOUT.value == "PROFILE_TIMEOUT"
    assert not hasattr(ExitReason, "BOTH_OPEN_TIMEOUT")


def test_data_lake_pair_exit_wrapper_uses_single_position_directly() -> None:
    intrabar = _intrabar_frame(60)
    intrabar.loc[20, "high"] = 102.0
    engine = DataLakeProductionBacktestEngine(
        _strategy_frame(),
        _config(),
        as_searchsorted_intrabar(intrabar),
    )

    entry_time = pd.Timestamp("2026-01-01T00:15:00Z")
    position = Position(
        side=Side.LONG,
        entry_time=entry_time,
        entry_index=0,
        entry_price=100.0,
        risk=1.0,
        sl=99.0,
        tp=101.0,
        quantity=1.0,
        risk_amount=1.0,
        entry_notional=100.0,
        atr_at_entry=1.0,
    )
    pair = TradePair(
        pair_id=1,
        long=position,
        short=None,
        equity_before_trade=1000.0,
        strategy_candle_open_time=pd.Timestamp("2026-01-01T00:00:00Z"),
        strategy_entry_time=entry_time,
        strategy_entry_price=100.0,
    )

    def retired_iterator_must_not_run():
        raise AssertionError("Data Lake exit scan should use pair.position, not pair.positions()")

    pair.positions = retired_iterator_must_not_run  # type: ignore[method-assign]
    engine._scan_pair_exit(pair, 1)

    assert not position.is_open
    assert position.exit_time == pd.Timestamp("2026-01-01T00:20:00Z")
    assert position.exit_reason is not None
    assert position.exit_reason.value == "TP"


def test_data_lake_single_position_timeout_matches_mature_engine() -> None:
    plain = _intrabar_frame()
    base = _config()
    timeout_profiles = {
        key: replace(
            profile,
            timeout_enabled=True,
            timeout_minutes=20,
            reward_risk_ratio=20.0,
            stop_loss_multiple=20.0,
        )
        for key, profile in base.strategy_profiles.items()
    }
    config = replace(base, strategy_profiles=timeout_profiles)

    mature = SRDynamicTPBacktestEngine(_strategy_frame(), config, plain.copy())
    data_lake = DataLakeProductionBacktestEngine(
        _strategy_frame(),
        config,
        as_searchsorted_intrabar(plain.copy()),
    )
    _force_long_sideways(mature)
    _force_long_sideways(data_lake)

    expected = mature.run().reset_index(drop=True)
    actual = data_lake.run().reset_index(drop=True)

    assert not expected.empty
    columns = [
        "side",
        "strategy_candle_open_time",
        "strategy_entry_time",
        "entry_time",
        "exit_time",
        "long_exit_reason",
        "long_exit_source",
        "profile_timeout_triggered",
        "profile_timeout_minutes",
        "profile_timeout_exit_time",
        "pair_net_pnl",
        "pair_net_r",
        "equity_after_trade",
    ]
    pdt.assert_frame_equal(actual[columns], expected[columns], check_dtype=False)

    timed_out = actual["profile_timeout_triggered"].astype(bool)
    assert timed_out.any()
    assert actual.loc[timed_out, "long_exit_reason"].eq("PROFILE_TIMEOUT").all()
    elapsed = pd.to_datetime(actual.loc[timed_out, "exit_time"], utc=True) - pd.to_datetime(
        actual.loc[timed_out, "strategy_entry_time"], utc=True
    )
    assert elapsed.eq(pd.Timedelta(minutes=20)).all()
