from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

import crypto_strategy_lab.engine as engine_module
import crypto_strategy_lab.enhanced_engine as enhanced_engine_module
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.features.production_context import ProductionContextFeatureProvider
from crypto_strategy_lab.features.technical import CoreDirectionalFeatureProvider
from crypto_strategy_lab.gui.enhanced_config import EnhancedBacktestConfig
from crypto_strategy_lab.sr_dynamic_tp_engine import SRDynamicTPBacktestEngine


def canonical_klines(n: int = 150) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    base = 120.0 + np.linspace(0, 8, n) + np.sin(np.arange(n) / 4.0) * 7.0
    open_ = base + np.sin(np.arange(n) / 6.0) * 0.8
    close = base + np.cos(np.arange(n) / 5.0) * 1.1
    high = np.maximum(open_, close) + 2.2
    low = np.minimum(open_, close) - 2.2
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + pd.Timedelta(hours=4),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + np.arange(n) % 19,
            "source_fingerprint": "production-context-source",
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


def config() -> EnhancedBacktestConfig:
    return EnhancedBacktestConfig(
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
        mean_reversion_mean_type="SMA",
        mean_reversion_bb_stddevs=1.75,
        mean_reversion_rsi_period=8,
        mean_reversion_rsi_oversold=35.0,
        mean_reversion_rsi_overbought=65.0,
        market_regime_method="ASSET_RETURN",
        enable_support_resistance_analysis=False,
    )


def prepared(frame: pd.DataFrame):
    cfg = config()
    request = request_for(frame)
    directional = CoreDirectionalFeatureProvider().compute(
        request,
        {DatasetKind.KLINES: frame},
        {
            "atr_period": cfg.atr_period,
            "adx_period": cfg.adx_period,
            "di_pressure_lookback": cfg.di_pressure_lookback,
        },
    )
    directional.attrs["feature_cache_key"] = "production-context-directional"
    context = ProductionContextFeatureProvider().compute(
        request,
        {DatasetKind.KLINES: frame},
        {
            "bb_period": cfg.bb_period,
            "bb_stddevs": cfg.bb_stddevs,
            "mean_reversion_period": cfg.mean_reversion_period,
            "mean_reversion_mean_type": cfg.mean_reversion_mean_type,
            "mean_reversion_bb_stddevs": cfg.mean_reversion_bb_stddevs,
            "mean_reversion_rsi_period": cfg.mean_reversion_rsi_period,
            "mean_reversion_rsi_oversold": cfg.mean_reversion_rsi_oversold,
            "mean_reversion_rsi_overbought": cfg.mean_reversion_rsi_overbought,
            "mean_reversion_require_reentry": cfg.mean_reversion_require_reentry,
        },
        {"core_directional": directional},
    )
    return directional, context


def test_production_context_matches_mature_enhanced_engine_arrays_and_snapshots() -> None:
    frame = canonical_klines()
    data = legacy_frame(frame)
    cfg = config()
    directional, context = prepared(frame)

    legacy = SRDynamicTPBacktestEngine(data, cfg)
    lake = DataLakeProductionBacktestEngine(
        data,
        cfg,
        technical_features=directional,
        context_features=context,
    )

    numeric_attributes = (
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_width_pct",
        "bb_width_1",
        "bb_width_3",
        "bb_width_5",
        "bb_width_change",
        "bb_width_change_pct",
        "session_vwap",
        "close_location_values",
        "mean_reversion_mean",
        "mean_reversion_distance_atr",
        "mean_reversion_distance_atr_previous",
        "mean_reversion_sigma",
        "mean_reversion_bb_upper",
        "mean_reversion_bb_lower",
        "mean_reversion_bb_zscore",
        "mean_reversion_rsi_values",
    )
    for attribute in numeric_attributes:
        np.testing.assert_allclose(
            getattr(lake, attribute),
            getattr(legacy, attribute),
            equal_nan=True,
        )
    np.testing.assert_array_equal(
        lake.mean_reversion_long_reentry,
        legacy.mean_reversion_long_reentry,
    )
    np.testing.assert_array_equal(
        lake.mean_reversion_short_reentry,
        legacy.mean_reversion_short_reentry,
    )
    assert lake.context_feature_source == "production_market_context@2"

    snapshot_fields = (
        "mean_price",
        "mean_distance_atr",
        "mean_reversion_state",
        "mean_reversion_motion",
        "mean_reversion_bb_upper",
        "mean_reversion_bb_lower",
        "mean_reversion_bb_zscore",
        "mean_reversion_bb_location",
        "mean_reversion_rsi",
        "mean_reversion_rsi_state",
        "mean_reversion_long_reentry",
        "mean_reversion_short_reentry",
        "mean_reversion_signal",
        "mean_reversion_signal_direction",
        "bb_reentry",
        "mr_signal",
        "mr_signal_direction",
    )
    prepared_snapshot_columns = {
        "mean_reversion_bb_location": "mean_reversion_bb_location",
        "mean_reversion_rsi_state": "mean_reversion_rsi_state",
        "mean_reversion_signal": "mean_reversion_signal",
        "mean_reversion_signal_direction": "mean_reversion_signal_direction",
        "bb_reentry": "bb_reentry",
        "mr_signal": "mr_signal",
        "mr_signal_direction": "mr_signal_direction",
    }
    for index in (45, 80, 120):
        for di_direction, trade_direction in (("LONG", "LONG"), ("SHORT", "LONG")):
            left = legacy._mean_reversion_snapshot(index, di_direction, trade_direction)
            right = lake._mean_reversion_snapshot(index, di_direction, trade_direction)
            for field in snapshot_fields:
                a = left[field]
                b = right[field]
                if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
                    if pd.isna(a) and pd.isna(b):
                        continue
                    assert np.isclose(float(a), float(b)), (index, field, a, b)
                else:
                    assert a == b, (index, field, a, b)
            for field, column in prepared_snapshot_columns.items():
                assert context.iloc[index][column] == left[field], (index, field)


def test_future_mutation_cannot_change_past_production_context() -> None:
    frame = canonical_klines()
    _, before = prepared(frame)
    cutoff = 85
    changed = frame.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close", "volume"]] *= 1.9
    _, after = prepared(changed)
    columns = [
        "bb_width_pct",
        "mean_reversion_mean",
        "mean_reversion_distance_atr",
        "mean_reversion_sigma",
        "mean_reversion_bb_zscore",
        "mean_reversion_rsi",
        "mean_reversion_long_reentry",
        "mean_reversion_short_reentry",
        "mean_reversion_bb_location",
        "mean_reversion_rsi_state",
        "mean_reversion_signal",
        "mean_reversion_signal_direction",
        "session_vwap",
        "close_location",
    ]
    pdt.assert_frame_equal(
        before.loc[:cutoff, columns].reset_index(drop=True),
        after.loc[:cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_prepared_production_context_skips_legacy_bb_and_mr_v2_math(monkeypatch) -> None:
    frame = canonical_klines()
    data = legacy_frame(frame)
    cfg = config()
    directional, context = prepared(frame)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy production context calculation was executed")

    # DataLakeBacktestEngine replaces these base-engine functions temporarily.
    monkeypatch.setattr(engine_module, "bollinger_bands", forbidden)
    monkeypatch.setattr(engine_module, "ema", forbidden)
    monkeypatch.setattr(engine_module, "distance_from_mean_atr", forbidden)

    # DataLakeProductionBacktestEngine must likewise replace all enhanced MR-v2
    # functions during legacy construction. Profile RSI lives in engine_module and
    # is intentionally not part of this feature migration.
    monkeypatch.setattr(enhanced_engine_module, "moving_mean", forbidden)
    monkeypatch.setattr(enhanced_engine_module, "distance_from_mean_atr", forbidden)
    monkeypatch.setattr(enhanced_engine_module, "bollinger_envelope", forbidden)
    monkeypatch.setattr(enhanced_engine_module, "bb_zscore", forbidden)
    monkeypatch.setattr(enhanced_engine_module, "rsi", forbidden)
    monkeypatch.setattr(enhanced_engine_module, "bollinger_reentry_flags", forbidden)

    engine = DataLakeProductionBacktestEngine(
        data,
        cfg,
        technical_features=directional,
        context_features=context,
    )
    assert engine.context_feature_source == "production_market_context@2"