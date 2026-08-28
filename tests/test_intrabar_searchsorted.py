from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from crypto_strategy_lab.config import BacktestConfig, EntryMode, IntrabarMissingPolicy, RiskMode
from crypto_strategy_lab.data.intrabar_index import SearchsortedIntrabarFrame, as_searchsorted_intrabar
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.gui.enhanced_config import EnhancedBacktestConfig
from crypto_strategy_lab.sr_dynamic_tp_engine import SRDynamicTPBacktestEngine


def intrabar_frame(periods: int = 180) -> pd.DataFrame:
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


def strategy_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="15min")
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


def test_searchsorted_frame_matches_boolean_timestamp_ranges() -> None:
    plain = intrabar_frame()
    indexed = as_searchsorted_intrabar(plain)
    assert isinstance(indexed, SearchsortedIntrabarFrame)
    assert indexed.intrabar_index_mode == "searchsorted"
    assert indexed.intrabar_iteration_mode == "array_window"

    for start, end in (
        ("2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"),
        ("2026-01-01T00:15:00Z", "2026-01-01T01:00:00Z"),
        ("2026-01-01T01:07:00Z", "2026-01-01T01:53:00Z"),
        ("2026-01-01T02:59:00Z", "2026-01-01T03:30:00Z"),
    ):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        expected = plain[(plain.timestamp >= start_ts) & (plain.timestamp < end_ts)].reset_index(drop=True)
        actual = indexed[(indexed.timestamp >= start_ts) & (indexed.timestamp < end_ts)].reset_index(drop=True)
        assert type(actual) is pd.DataFrame
        pdt.assert_frame_equal(actual, expected)

    assert indexed.timestamp.max() == plain.timestamp.max()
    assert indexed.timestamp.min() == plain.timestamp.min()


def test_sorted_timestamp_endpoints_do_not_scan_series(monkeypatch) -> None:
    plain = intrabar_frame(30)
    indexed = as_searchsorted_intrabar(plain)
    expected_min = plain.timestamp.min()
    expected_max = plain.timestamp.max()

    def fail_if_series_is_requested(_self):
        raise AssertionError("sorted timestamp endpoint should use the existing DatetimeIndex")

    monkeypatch.setattr(SearchsortedIntrabarFrame, "_timestamp_series", fail_if_series_is_requested)
    assert indexed.timestamp.min() == expected_min
    assert indexed.timestamp.max() == expected_max


def test_fast_intrabar_window_matches_pandas_rows_and_gap_detection() -> None:
    plain = intrabar_frame(40)
    plain.loc[7, "high"] = 111.25
    plain.loc[11, "low"] = 88.75
    indexed = as_searchsorted_intrabar(plain)
    start = pd.Timestamp("2026-01-01T00:05:00Z")
    end = pd.Timestamp("2026-01-01T00:18:00Z")

    expected = plain[(plain.timestamp >= start) & (plain.timestamp < end)]
    window = indexed.fast_window(start, end)
    assert window is not None
    assert not window.empty
    assert window.first_timestamp == expected.timestamp.iloc[0]
    expected_rows = [
        (idx, row.timestamp, float(row.open), float(row.high), float(row.low))
        for idx, row in expected.iterrows()
    ]
    assert list(window.rows()) == expected_rows
    assert window.gap_pairs(pd.Timedelta(minutes=1)) == ()

    missing = plain.drop(index=10).reset_index(drop=True)
    missing_indexed = as_searchsorted_intrabar(missing)
    missing_window = missing_indexed.fast_window(start, end)
    assert missing_window is not None
    assert missing_window.gap_pairs(pd.Timedelta(minutes=1)) == (
        (
            pd.Timestamp("2026-01-01T00:09:00Z"),
            pd.Timestamp("2026-01-01T00:11:00Z"),
        ),
    )


def test_fast_intrabar_rows_preserve_utc_without_indexing_datetime_index(monkeypatch) -> None:
    indexed = as_searchsorted_intrabar(intrabar_frame(10))
    start = pd.Timestamp("2026-01-01T00:02:00Z")
    end = pd.Timestamp("2026-01-01T00:05:00Z")
    window = indexed.fast_window(start, end)
    assert window is not None

    def fail_on_datetime_index_getitem(_self, _key):
        raise AssertionError("hot row iteration must not index the DatetimeIndex")

    monkeypatch.setattr(pd.DatetimeIndex, "__getitem__", fail_on_datetime_index_getitem)
    rows = list(window.rows())

    assert [row[1] for row in rows] == [
        pd.Timestamp("2026-01-01T00:02:00Z"),
        pd.Timestamp("2026-01-01T00:03:00Z"),
        pd.Timestamp("2026-01-01T00:04:00Z"),
    ]
    assert all(type(row[1]) is pd.Timestamp for row in rows)
    assert all(row[1].tzinfo is not None and str(row[1].tzinfo) == "UTC" for row in rows)
    assert [(row[2], row[3], row[4]) for row in rows] == [
        (100.0, 100.5, 99.5),
        (100.0, 100.5, 99.5),
        (100.0, 100.5, 99.5),
    ]


def test_intrabar_window_tuple_iteration_matches_pandas_iterrows() -> None:
    plain = intrabar_frame(30)
    plain.loc[7, "high"] = 111.25
    plain.loc[11, "low"] = 88.75
    indexed = as_searchsorted_intrabar(plain)
    start = pd.Timestamp("2026-01-01T00:05:00Z")
    end = pd.Timestamp("2026-01-01T00:18:00Z")

    expected = plain[(plain.timestamp >= start) & (plain.timestamp < end)]
    actual = indexed[(indexed.timestamp >= start) & (indexed.timestamp < end)]

    expected_rows = [
        (idx, row.timestamp, row.open, row.high, row.low, row.close, row.volume)
        for idx, row in expected.iterrows()
    ]
    actual_rows = []
    for idx, row in actual.iterrows():
        assert not isinstance(row, pd.Series)
        actual_rows.append(
            (idx, row.timestamp, row.open, row.high, row.low, row.close, row.volume)
        )

    assert actual_rows == expected_rows


def test_engine_reset_index_preserves_searchsorted_acceleration() -> None:
    indexed = as_searchsorted_intrabar(intrabar_frame())
    reset = indexed.reset_index(drop=True)
    assert isinstance(reset, SearchsortedIntrabarFrame)
    assert reset.intrabar_index_mode == "searchsorted"
    assert reset.intrabar_iteration_mode == "array_window"
    start = pd.Timestamp("2026-01-01T00:30:00Z")
    end = pd.Timestamp("2026-01-01T00:45:00Z")
    assert len(reset[(reset.timestamp >= start) & (reset.timestamp < end)]) == 15
    window = reset.fast_window(start, end)
    assert window is not None
    assert len(list(window.rows())) == 15


def test_unsorted_intrabar_falls_back_without_changing_rows() -> None:
    plain = intrabar_frame(10).iloc[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]].reset_index(drop=True)
    indexed = as_searchsorted_intrabar(plain)
    assert isinstance(indexed, SearchsortedIntrabarFrame)
    assert indexed.intrabar_index_mode == "boolean_fallback"
    assert indexed.fast_window(
        pd.Timestamp("2026-01-01T00:01:00Z"),
        pd.Timestamp("2026-01-01T00:04:00Z"),
    ) is None
    start = pd.Timestamp("2026-01-01T00:01:00Z")
    end = pd.Timestamp("2026-01-01T00:04:00Z")
    expected = plain[(plain.timestamp >= start) & (plain.timestamp < end)]
    actual = indexed[(indexed.timestamp >= start) & (indexed.timestamp < end)]
    pdt.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert indexed.timestamp.min() == plain.timestamp.min()
    assert indexed.timestamp.max() == plain.timestamp.max()


def _engine(intrabar: pd.DataFrame) -> BacktestEngine:
    config = BacktestConfig(
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
    )
    engine = BacktestEngine(strategy_frame(), config, intrabar)
    engine.market_regime_values[:] = "SIDEWAYS"
    engine.plus_di_values[:] = 50.0
    engine.minus_di_values[:] = 10.0
    engine.di_spread[:] = 40.0
    engine.di_ratio[:] = 5.0
    return engine


def _production_config() -> EnhancedBacktestConfig:
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


def _force_direction(engine) -> None:
    engine.market_regime_values[:] = "SIDEWAYS"
    engine.plus_di_values[:] = 50.0
    engine.minus_di_values[:] = 10.0
    engine.di_spread[:] = 40.0
    engine.di_ratio[:] = 5.0


def test_searchsorted_wrapper_preserves_intrabar_trade_results() -> None:
    plain = intrabar_frame()
    for minute in (20, 50, 80, 110):
        plain.loc[minute, "high"] = 103.0

    legacy = _engine(plain.copy()).run().reset_index(drop=True)
    indexed = _engine(as_searchsorted_intrabar(plain.copy())).run().reset_index(drop=True)

    assert not legacy.empty
    columns = [
        "side",
        "strategy_candle_open_time",
        "strategy_entry_time",
        "entry_time",
        "exit_time",
        "long_exit_reason",
        "long_exit_source",
        "pair_net_pnl",
        "pair_net_r",
        "equity_after_trade",
    ]
    pdt.assert_frame_equal(indexed[columns], legacy[columns], check_dtype=False)


def test_data_lake_array_window_preserves_production_trade_results() -> None:
    plain = intrabar_frame()
    for minute in (20, 50, 80, 110):
        plain.loc[minute, "high"] = 103.0

    cfg = _production_config()
    legacy_engine = SRDynamicTPBacktestEngine(strategy_frame(), cfg, plain.copy())
    fast_engine = DataLakeProductionBacktestEngine(
        strategy_frame(),
        cfg,
        as_searchsorted_intrabar(plain.copy()),
    )
    _force_direction(legacy_engine)
    _force_direction(fast_engine)

    legacy = legacy_engine.run().reset_index(drop=True)
    fast = fast_engine.run().reset_index(drop=True)
    assert not legacy.empty
    columns = [
        "side",
        "strategy_candle_open_time",
        "strategy_entry_time",
        "entry_time",
        "exit_time",
        "long_exit_reason",
        "long_exit_source",
        "pair_net_pnl",
        "pair_net_r",
        "equity_after_trade",
    ]
    pdt.assert_frame_equal(fast[columns], legacy[columns], check_dtype=False)
