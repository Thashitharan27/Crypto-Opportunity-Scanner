from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

import crypto_strategy_lab.engine as engine_module
from crypto_strategy_lab.adx import adx
from crypto_strategy_lab.atr import atr
from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data_lake_engine import DataLakeBacktestEngine
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.features.technical import prepare_core_directional_features


def canonical_klines(n: int = 90, interval: str = "4h") -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq=interval, tz="UTC")
    base = 100.0 + np.linspace(0, 18, n) + np.sin(np.arange(n) / 3.0) * 4.0
    open_ = base + np.sin(np.arange(n) / 5.0) * 0.5
    close = base + np.cos(np.arange(n) / 4.0) * 0.7
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    delta = pd.Timedelta(interval)
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + delta,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.linspace(10, 20, n),
        }
    )


def request_for(frame: pd.DataFrame) -> DataRequest:
    interval = "4h"
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(interval)).to_pydatetime(),
        strategy_interval=interval,
    )


def legacy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": frame["period_start"],
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "volume": frame["volume"],
        }
    )


def prepared(frame: pd.DataFrame, atr_period=7, adx_period=6, lookback=3):
    return prepare_core_directional_features(
        request_for(frame),
        frame,
        atr_period=atr_period,
        adx_period=adx_period,
        di_pressure_lookback=lookback,
    )


def four_hour_config() -> BacktestConfig:
    return BacktestConfig(
        strategy_timeframe_minutes=240,
        intrabar_timeframe_minutes=1,
        telemetry_interval_minutes=240,
        use_intrabar_data=False,
        atr_period=7,
        adx_period=6,
        di_pressure_lookback=3,
        market_regime_method="ASSET_RETURN",
    )


def test_provider_matches_existing_atr_adx_dmi_math() -> None:
    frame = canonical_klines()
    features = prepared(frame)
    high = frame.high.to_numpy(float)
    low = frame.low.to_numpy(float)
    close = frame.close.to_numpy(float)
    expected_atr = atr(high, low, close, 7)
    expected_adx, expected_plus, expected_minus = adx(high, low, close, 6)

    np.testing.assert_allclose(features.atr, expected_atr, equal_nan=True)
    np.testing.assert_allclose(features.adx, expected_adx, equal_nan=True)
    np.testing.assert_allclose(features.plus_di, expected_plus, equal_nan=True)
    np.testing.assert_allclose(features.minus_di, expected_minus, equal_nan=True)
    assert features.attrs["feature_name"] == "core_directional"
    assert features.attrs["feature_version"] == "1"


def test_future_source_mutation_cannot_change_past_features() -> None:
    frame = canonical_klines()
    before = prepared(frame)
    cutoff = 50

    changed = frame.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 3.0
    after = prepared(changed)

    columns = [
        "atr",
        "adx",
        "plus_di",
        "minus_di",
        "di_spread",
        "di_ratio",
        "plus_di_change",
        "minus_di_change",
        "di_pressure_spread_change",
        "long_di_pressure_state",
        "short_di_pressure_state",
    ]
    pdt.assert_frame_equal(
        before.loc[:cutoff, columns].reset_index(drop=True),
        after.loc[:cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_data_lake_engine_reuses_prepared_features_and_preserves_di_pressure() -> None:
    frame = canonical_klines()
    data = legacy_frame(frame)
    config = four_hour_config()
    features = prepared(frame)
    legacy = BacktestEngine(data, config)
    lake = DataLakeBacktestEngine(data, config, technical_features=features)

    assert lake.technical_feature_source == "core_directional@1"
    np.testing.assert_allclose(lake.atr_values, legacy.atr_values, equal_nan=True)
    np.testing.assert_allclose(lake.adx_values, legacy.adx_values, equal_nan=True)
    np.testing.assert_allclose(lake.plus_di_values, legacy.plus_di_values, equal_nan=True)
    np.testing.assert_allclose(lake.minus_di_values, legacy.minus_di_values, equal_nan=True)
    np.testing.assert_allclose(lake.di_spread_change, legacy.di_spread_change, equal_nan=True)

    for i in range(len(data)):
        for direction in ("LONG", "SHORT"):
            left = legacy._di_pressure_snapshot(i, direction)
            right = lake._di_pressure_snapshot(i, direction)
            assert left["di_pressure_state"] == right["di_pressure_state"]
            for key in (
                "plus_di_change",
                "minus_di_change",
                "directional_di_change",
                "opposing_di_change",
                "di_spread_change",
            ):
                a, b = left[key], right[key]
                assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b)


def test_prepared_path_does_not_execute_legacy_atr_or_adx(monkeypatch) -> None:
    frame = canonical_klines()
    data = legacy_frame(frame)
    config = four_hour_config()
    features = prepared(frame)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy indicator calculation was executed")

    monkeypatch.setattr(engine_module, "atr", forbidden)
    monkeypatch.setattr(engine_module, "adx", forbidden)
    engine = DataLakeBacktestEngine(data, config, technical_features=features)
    assert engine.technical_feature_source == "core_directional@1"
