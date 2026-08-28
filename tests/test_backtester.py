from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.atr import atr
from crypto_strategy_lab.config import (
    BacktestConfig,
    DailyEntryMissedPolicy,
    EntryMode,
    IntrabarMissingPolicy,
    RiskMode,
    TiePolicy,
)
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.loader import load_ohlcv_csv
from crypto_strategy_lab.statistics import equity_curve, summarize
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, StrategyProfile
from crypto_strategy_lab.trade import ExitSource, Position, Side


def candles(rows):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=15 * i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1] * len(rows),
        }
    )


def one_minute(start, rows):
    return pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1] * len(rows),
        }
    )


def profile_set(**changes):
    profile = StrategyProfile(enabled=True, **changes)
    return {key: profile for key in PROFILE_KEYS}


def cfg(**kw):
    base = dict(
        risk_mode=RiskMode.FIXED,
        fixed_r=10,
        initial_equity=1000,
        risk_per_leg=0.01,
        taker_fee=0,
        maker_fee=0,
        slippage=0,
        entry_mode=EntryMode.WAIT_UNTIL_CLOSED,
        use_intrabar_data=False,
        enable_trade_telemetry=False,
        strategy_profiles=profile_set(stop_loss_multiple=1, reward_risk_ratio=1),
    )
    base.update(kw)
    return BacktestConfig(**base)


def prepared_engine(data, config=None, intrabar=None, *, direction="LONG", regime="SIDEWAYS"):
    engine = BacktestEngine(data, config or cfg(), intrabar)
    engine.market_regime_values[:] = regime
    if direction == "LONG":
        engine.plus_di_values[:] = 50
        engine.minus_di_values[:] = 10
    else:
        engine.plus_di_values[:] = 10
        engine.minus_di_values[:] = 50
    engine.di_spread[:] = 40
    return engine


def test_loading_binance_open_time_ms_extra_columns_gap(capsys, tmp_path):
    path = tmp_path / "b.csv"
    path.write_text(
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1704067200000,1,2,0.5,1.5,10,x,x,x,x,x,x\n"
        "1704067200000,1,2,0.5,1.5,10,x,x,x,x,x,x\n"
        "1704069000000,2,3,1,2.5,11,x,x,x,x,x,x\n"
    )
    df = load_ohlcv_csv(str(path), "ms")
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert "missing_15m_candles=1" in capsys.readouterr().out


def test_wilder_atr():
    high = np.array([10, 12, 13, 14, 15.0], float)
    low = np.array([9, 10, 11, 12, 13.0], float)
    close = np.array([9.5, 11, 12, 13, 14.0], float)
    out = atr(high, low, close, 3)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(11 / 6)
    assert out[3] == pytest.approx(((11 / 6) * 2 + 2) / 3)


def test_profile_rsi_uses_only_completed_candles_and_has_warmup():
    close = np.arange(100.0, 140.0)
    engine = BacktestEngine(candles([(v, v, v, v) for v in close]), cfg())
    values = engine.profile_rsi_values[14]
    assert np.isnan(values[10])
    assert values[20] == pytest.approx(100.0)


def test_current_engine_opens_only_di_selected_profile_side():
    data = candles([(100, 100, 100, 100), (100, 111, 99, 100)])
    long_row = prepared_engine(data).run().iloc[0]
    assert long_row.side == "LONG"
    assert long_row.trade_direction == "LONG"
    assert "short_quantity" not in long_row.index

    short_row = prepared_engine(data, direction="SHORT").run().iloc[0]
    assert short_row.side == "SHORT"
    assert short_row.trade_direction == "SHORT"
    assert "long_quantity" not in short_row.index


def test_position_sizing_fees_and_net_r_for_selected_side():
    df = candles([(100, 101, 99, 100), (100, 111, 99, 100)])
    trades = prepared_engine(df, cfg(taker_fee=0.001, tie_policy=TiePolicy.OPTIMISTIC)).run()
    row = trades.iloc[0]
    assert row.long_quantity == pytest.approx(1)
    assert row.long_risk_amount == pytest.approx(10)
    assert row.long_entry_notional == pytest.approx(100)
    assert row.long_fees == pytest.approx(0.21)
    assert row.long_net_r == pytest.approx((10 - 0.21) / 10)
    assert row.pair_net_r < row.pair_gross_r


def test_same_candle_tie_policy_applies_to_selected_side():
    data = candles([(100, 100, 100, 100), (100, 111, 89, 100)])
    pess = prepared_engine(data, cfg(tie_policy=TiePolicy.PESSIMISTIC)).run().iloc[0]
    opt = prepared_engine(data, cfg(tie_policy=TiePolicy.OPTIMISTIC)).run().iloc[0]
    assert pess.ambiguous_candle and opt.ambiguous_candle
    assert pess.long_exit_reason == "SL"
    assert opt.long_exit_reason == "TP"


def test_end_of_data_closure_does_not_use_entry_candle_high_low():
    df = candles([(100, 111, 89, 100)])
    row = prepared_engine(df).run().iloc[0]
    assert row.long_exit_reason == "END_OF_DATA"
    assert row.holding_bars == 0


def test_strategy_entry_time_is_candle_close_and_uses_close_price():
    df = candles([(100, 1000, 1, 100), (100, 111, 99, 100)])
    row = prepared_engine(df).run().iloc[0]
    assert row.strategy_candle_open_time == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert row.strategy_entry_time == pd.Timestamp("2024-01-01 00:15", tz="UTC")
    assert row.strategy_entry_price == 100


def test_equity_and_drawdown_reconcile_with_single_side_results():
    data = candles(
        [
            (100, 100, 100, 100),
            (100, 111, 100, 100),
            (100, 100, 89, 100),
            (100, 111, 100, 100),
        ]
    )
    trades = prepared_engine(data, cfg(entry_mode=EntryMode.EVERY_N_CANDLES, max_active_pairs=10)).run()
    summary = summarize(trades, 1000)
    curve = equity_curve(trades, 1000)
    assert summary["ending_equity"] == pytest.approx(trades.equity_after_trade.iloc[-1])
    assert trades.equity_after_trade.iloc[-1] == pytest.approx(1000 + trades.pair_net_pnl.sum())
    assert "maximum_drawdown" in summary
    assert not curve.empty


def test_atr_uses_strategy_candles_not_intrabar_volatility():
    rows = [(100 + i, 102 + i, 99 + i, 101 + i) for i in range(20)]
    strat = candles(rows)
    start = pd.Timestamp("2024-01-01 00:15", tz="UTC")
    quiet = one_minute(start, [(100, 100, 100, 100)] * 300)
    wild = one_minute(start, [(100, 10000, 1, 100)] * 300)
    config = cfg(
        risk_mode=RiskMode.ATR,
        atr_period=14,
        entry_mode=EntryMode.EVERY_N_CANDLES,
        use_intrabar_data=True,
    )
    a = prepared_engine(strat, config, quiet).run().atr_at_entry.iloc[0]
    b = prepared_engine(strat, config, wild).run().atr_at_entry.iloc[0]
    assert a == pytest.approx(b)
    expected = atr(strat.high.to_numpy(float), strat.low.to_numpy(float), strat.close.to_numpy(float), 14)
    assert a == pytest.approx(expected[np.flatnonzero(np.isfinite(expected))[0]])


def test_intrabar_resolves_selected_side_and_source():
    strat = candles([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    intra = one_minute(pd.Timestamp("2024-01-01 00:15", tz="UTC"), [(100, 100, 100, 100)] + [(100, 111, 100, 100)] * 15)
    row = prepared_engine(strat, cfg(use_intrabar_data=True), intra).run().iloc[0]
    assert row.long_exit_reason == "TP"
    assert row.long_exit_source == "1M_INTRABAR"
    assert pd.Timestamp(row.long_exit_time) >= pd.Timestamp(row.entry_time)


def test_leverage_cap_and_full_notional_fees():
    df = candles([(100, 100, 100, 100), (100, 111, 100, 100)])
    row = prepared_engine(df, cfg(taker_fee=0.001, max_effective_leverage_per_leg=0.05)).run().iloc[0]
    assert row.leverage_capped
    assert row.long_quantity < row.long_uncapped_quantity
    assert row.long_effective_leverage == pytest.approx(0.05)
    assert row.long_entry_fee == pytest.approx(row.long_entry_notional * 0.001)


def test_warmup_candles_do_not_trade_before_requested_start():
    df = candles([(100, 100, 100, 100)] * 20 + [(100, 111, 100, 100)])
    config = cfg(trading_start_date="2024-01-01 04:00", risk_mode=RiskMode.ATR, atr_period=14)
    engine = prepared_engine(df, config)
    trades = engine.run()
    assert engine.warmup_candle_count == 16
    assert (trades.strategy_entry_time >= pd.Timestamp("2024-01-01 04:00", tz="UTC")).all()


def test_missing_intrabar_fallback_policy_records_flag():
    strat = candles([(100, 100, 100, 100), (100, 111, 100, 100)])
    intra = one_minute(
        pd.Timestamp("2024-01-01 00:16", tz="UTC"),
        [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)],
    )
    intra.loc[2, "timestamp"] = pd.Timestamp("2024-01-01 00:20", tz="UTC")
    row = prepared_engine(strat, cfg(use_intrabar_data=True), intra).run().iloc[0]
    assert row.missing_intrabar_data
    assert row.long_exit_source == "15M_FALLBACK"


def test_missing_intrabar_error_policy_stops_run():
    strat = candles([(100, 100, 100, 100), (100, 111, 100, 100)])
    intra = one_minute(
        pd.Timestamp("2024-01-01 00:16", tz="UTC"),
        [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)],
    )
    intra.loc[2, "timestamp"] = pd.Timestamp("2024-01-01 00:20", tz="UTC")
    engine = prepared_engine(
        strat,
        cfg(use_intrabar_data=True, intrabar_missing_policy=IntrabarMissingPolicy.ERROR),
        intra,
    )
    with pytest.raises(ValueError, match="Missing 1-minute intrabar candles"):
        engine.run()


def test_missing_intrabar_continue_policy_does_not_use_strategy_fallback():
    strat = candles([(100, 100, 100, 100), (100, 111, 100, 100)])
    intra = one_minute(
        pd.Timestamp("2024-01-01 00:16", tz="UTC"),
        [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)],
    )
    intra.loc[2, "timestamp"] = pd.Timestamp("2024-01-01 00:20", tz="UTC")
    row = prepared_engine(
        strat,
        cfg(use_intrabar_data=True, intrabar_missing_policy=IntrabarMissingPolicy.WARN_AND_CONTINUE),
        intra,
    ).run().iloc[0]
    assert row.missing_intrabar_data
    assert row.long_fallback_reason is None


def test_atr_checkpoint_extends_profile_position_and_locks_profit():
    df = candles([(100, 100, 100, 100)] * 4)
    engine = prepared_engine(
        df,
        cfg(
            strategy_profiles=profile_set(
                atr_checkpoint_tp_extension_enabled=True,
                atr_checkpoint_di_spread_minimum=30,
                atr_checkpoint_bb_width_minimum=0.03,
            )
        ),
    )
    engine.plus_di_values[:] = 50
    engine.minus_di_values[:] = 10
    engine.bb_width[:] = 0.04
    pos = Position(Side.LONG, df.timestamp.iloc[0], 0, 100, 10, 80, 120, 1, 10, 100, 10)
    pos.original_sl = 80
    pos.atr_checkpoint_extension_enabled = True
    pos.atr_checkpoint_initial_tp = 120
    pos.atr_checkpoint_final_tp_r = 2

    engine._apply_atr_checkpoint_extensions(pos, 130, 100, df.timestamp.iloc[2])

    assert pos.atr_checkpoint_pass_count == 3
    assert pos.tp == pytest.approx(150)
    assert pos.sl == pytest.approx(120)
    assert pos.atr_checkpoint_profit_lock_r == pytest.approx(2)


def test_r_step_staircase_advances_stop_and_ignores_fixed_tp():
    df = candles([(100, 100, 100, 100)] * 5)
    engine = prepared_engine(
        df,
        cfg(
            strategy_profiles=profile_set(
                r_step_trailing_enabled=True,
                r_step_activation_r=2,
                r_step_distance_r=2,
                r_step_size_r=1,
                r_step_maximum_r=0,
            )
        ),
    )
    pos = Position(Side.LONG, df.timestamp.iloc[0], 0, 100, 10, 90, 120, 1, 10, 100, 10)
    pos.original_sl = 90
    pos.r_step_trailing_enabled = True
    pos.r_step_activation_r = 2
    pos.r_step_distance_r = 2
    pos.r_step_size_r = 1
    pos.r_step_next_checkpoint_r = 2
    pos.r_step_maximum_r = 0
    pos.r_step_initial_tp = 120

    assert not engine._maybe_r_step_trailing_exit(pos, 1, 120, 101, df.timestamp.iloc[1], ExitSource.FALLBACK_15M)
    assert pos.sl == pytest.approx(100)
    assert not engine._maybe_r_step_trailing_exit(pos, 2, 130, 111, df.timestamp.iloc[2], ExitSource.FALLBACK_15M)
    assert pos.sl == pytest.approx(110)
    assert engine._maybe_r_step_trailing_exit(pos, 3, 115, 109, df.timestamp.iloc[3], ExitSource.FALLBACK_15M)
    assert pos.exit_reason.value == "R_STEP_TRAILING_STOP"
    assert pos.exit_price == pytest.approx(110)


def test_adx_matches_reference_values():
    from crypto_strategy_lab.adx import adx

    rows = [(100 + i, 102 + i + (i % 3), 99 + i - (i % 2), 101 + i + ((i % 4) - 1) * 0.3) for i in range(40)]
    df = candles(rows)
    out, plus, minus = adx(df.high.to_numpy(float), df.low.to_numpy(float), df.close.to_numpy(float), 14)
    assert out[-1] == pytest.approx(100.0, abs=1e-5)
    assert plus[-1] == pytest.approx(28.365647, abs=1e-5)
    assert minus[-1] == pytest.approx(0.0, abs=1e-5)


def test_bollinger_and_di_telemetry_columns_are_current_and_unique():
    df = candles([(100 + i, 102 + i + (i % 3), 99 + i - (i % 2), 101 + i) for i in range(45)])
    trades = prepared_engine(df, cfg(entry_mode=EntryMode.EVERY_N_CANDLES, max_active_pairs=50)).run()
    assert {
        "bb_width_entry_5bar_change",
        "bb_width_entry_5bar_change_pct",
        "di_spread",
        "di_ratio",
        "di_spread_entry_5bar_change",
        "indicator_warmup_complete",
        "adx_available_at_entry",
        "bb_width_available_at_entry",
        "indicator_warmup_note",
    }.issubset(trades.columns)
    assert not trades.columns.duplicated().any()


def scheduled_candles(start="2023-12-31 23:45", periods=24 * 4 + 5, price=100):
    ts = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=periods, freq="15min")
    return pd.DataFrame(
        {"timestamp": ts, "open": price, "high": price, "low": price, "close": price, "volume": 1}
    )


def test_daily_schedule_timezone_and_alignment():
    df = scheduled_candles()
    config = cfg(
        enable_daily_entry_schedule=True,
        daily_entry_time="19:00",
        daily_entry_timezone="America/New_York",
        daily_entry_missed_policy=DailyEntryMissedPolicy.SKIP_DAY,
    )
    trades = prepared_engine(df, config).run()
    assert trades.entry_time.iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    with pytest.raises(ValueError, match="align"):
        cfg(enable_daily_entry_schedule=True, daily_entry_time="00:07")


def test_trade_telemetry_tracks_only_current_selected_side():
    df = candles([(100, 100, 100, 100)] * 6)
    engine = prepared_engine(df, cfg(enable_trade_telemetry=True))
    engine.run()
    telemetry = engine.telemetry_frame()
    assert not telemetry.empty
    assert telemetry.long_is_open.any()
    assert not telemetry.short_is_open.any()
    assert telemetry.elapsed_strategy_bars.min() == 0


def test_indicator_reports_work_with_large_trade_attrs_detached():
    from crypto_strategy_lab.statistics import adx_analysis, bb_width_analysis, di_spread_analysis

    trades = pd.DataFrame(
        {
            "adx": [20.0, 40.0],
            "bb_width_pct": [2.0, 8.0],
            "di_spread": [10.0, 35.0],
            "pair_net_pnl": [-1.0, 1.0],
            "holding_minutes": [60.0, 120.0],
            "holding_hours": [1.0, 2.0],
            "pair_net_r": [-1.0, 1.0],
            "side": ["LONG", "SHORT"],
        }
    )
    detached = [{"large": "x" * 10000}]
    trades.attrs["skipped_signals"] = detached
    try:
        assert int(adx_analysis(trades)["Trades"].sum()) == 2
        assert int(bb_width_analysis(trades)["Trades"].sum()) == 2
        assert int(di_spread_analysis(trades)["Trades"].sum()) == 2
    finally:
        trades.attrs["skipped_signals"] = detached


def test_plot_frequency_aliases_fall_back_for_older_pandas(monkeypatch):
    from crypto_strategy_lab import plots

    returns = pd.DataFrame(
        {"exit_time": [pd.Timestamp("2024-01-31")], "pair_net_pnl": [100.0], "pair_net_r": [1.0]}
    )
    original = pd.Series.resample

    def legacy_resample(series, rule, *args, **kwargs):
        if rule in ("ME", "YE"):
            raise ValueError("unsupported alias")
        return original(series, rule, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "resample", legacy_resample)
    assert plots._month_year_frequencies(returns) == ("M", "Y")


def test_save_plots_accepts_single_side_csv_style_exit_time_strings(tmp_path):
    from crypto_strategy_lab.plots import save_plots

    trades = pd.DataFrame(
        {
            "pair_net_r": [1.0],
            "holding_hours": [2.0],
            "pair_net_pnl": [100.0],
            "exit_time": ["2024-01-31 00:00:00+00:00"],
        }
    )
    equity = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(["2024-01-31 00:00:00+00:00"]),
            "equity": [1100.0],
            "drawdown": [0.0],
        }
    )
    warnings = save_plots(trades, equity, tmp_path)
    assert isinstance(warnings, list)
