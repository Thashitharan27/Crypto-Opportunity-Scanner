from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

import crypto_strategy_lab.engine as engine_module
from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data_lake_engine import DataLakeBacktestEngine
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.features.context import MarketContextFeatureProvider
from crypto_strategy_lab.features.technical import CoreDirectionalFeatureProvider


def canonical_klines(n: int = 100) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    base = 100.0 + np.linspace(0, 15, n) + np.sin(np.arange(n) / 3.5) * 5.0
    open_ = base + np.sin(np.arange(n) / 7.0) * 0.6
    close = base + np.cos(np.arange(n) / 5.0) * 0.9
    high = np.maximum(open_, close) + 1.7
    low = np.minimum(open_, close) - 1.7
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + pd.Timedelta(hours=4),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 50 + np.arange(n) % 13,
            "source_fingerprint": "market-context-source",
        }
    )


def request_for(frame: pd.DataFrame) -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(hours=4)).to_pydatetime(),
        strategy_interval="4h",
    )


def legacy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": frame.period_start,
            "open": frame.open,
            "high": frame.high,
            "low": frame.low,
            "close": frame.close,
            "volume": frame.volume,
        }
    )


def config() -> BacktestConfig:
    return BacktestConfig(
        strategy_timeframe_minutes=240,
        intrabar_timeframe_minutes=1,
        telemetry_interval_minutes=240,
        use_intrabar_data=False,
        atr_period=7,
        adx_period=6,
        di_pressure_lookback=3,
        bb_period=12,
        bb_stddevs=2.25,
        mean_reversion_period=10,
        market_regime_method="ASSET_RETURN",
    )


def prepared(frame: pd.DataFrame):
    req = request_for(frame)
    cfg = config()
    directional_provider = CoreDirectionalFeatureProvider()
    directional = directional_provider.compute(
        req,
        {DatasetKind.KLINES: frame},
        {
            "atr_period": cfg.atr_period,
            "adx_period": cfg.adx_period,
            "di_pressure_lookback": cfg.di_pressure_lookback,
        },
    )
    directional.attrs["feature_cache_key"] = "directional-test-key"
    context_provider = MarketContextFeatureProvider()
    context = context_provider.compute(
        req,
        {DatasetKind.KLINES: frame},
        {
            "bb_period": cfg.bb_period,
            "bb_stddevs": cfg.bb_stddevs,
            "mean_reversion_period": cfg.mean_reversion_period,
        },
        {"core_directional": directional},
    )
    return directional, context


def test_market_context_matches_existing_engine_arrays() -> None:
    frame = canonical_klines()
    data = legacy_frame(frame)
    cfg = config()
    directional, context = prepared(frame)
    legacy = BacktestEngine(data, cfg)
    lake = DataLakeBacktestEngine(
        data,
        cfg,
        technical_features=directional,
        context_features=context,
    )

    for attribute in (
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_width_pct",
        "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
        "bb_width_change_pct", "mean_reversion_mean", "mean_reversion_distance_atr",
        "mean_reversion_distance_atr_previous", "session_vwap", "close_location_values",
    ):
        np.testing.assert_allclose(
            getattr(lake, attribute), getattr(legacy, attribute), equal_nan=True
        )
    assert lake.context_feature_source == "market_context@1"

    for i in range(len(data)):
        for di_direction, trade_direction in (("LONG", "LONG"), ("SHORT", "LONG")):
            left = legacy._mean_reversion_snapshot(i, di_direction, trade_direction)
            right = lake._mean_reversion_snapshot(i, di_direction, trade_direction)
            assert left["mean_reversion_state"] == right["mean_reversion_state"]
            assert left["mean_reversion_motion"] == right["mean_reversion_motion"]
            assert left["mean_reversion_strength_label"] == right["mean_reversion_strength_label"]
            assert left["mean_reversion_trade_alignment"] == right["mean_reversion_trade_alignment"]


def test_future_mutation_cannot_change_past_market_context() -> None:
    frame = canonical_klines()
    _, before = prepared(frame)
    cutoff = 55
    changed = frame.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close", "volume"]] *= 2.5
    _, after = prepared(changed)
    columns = [
        "bb_width_pct", "bb_width_change", "mean_reversion_mean",
        "mean_reversion_distance_atr", "mean_reversion_state", "mean_reversion_motion",
        "session_vwap", "close_location",
    ]
    pdt.assert_frame_equal(
        before.loc[:cutoff, columns].reset_index(drop=True),
        after.loc[:cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_prepared_context_does_not_execute_legacy_bb_or_mean_math(monkeypatch) -> None:
    frame = canonical_klines()
    data = legacy_frame(frame)
    cfg = config()
    directional, context = prepared(frame)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy stateless context calculation was executed")

    monkeypatch.setattr(engine_module, "bollinger_bands", forbidden)
    monkeypatch.setattr(engine_module, "ema", forbidden)
    monkeypatch.setattr(engine_module, "distance_from_mean_atr", forbidden)
    engine = DataLakeBacktestEngine(
        data,
        cfg,
        technical_features=directional,
        context_features=context,
    )
    assert engine.context_feature_source == "market_context@1"
