from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pandas.testing as pdt

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.features.support_resistance import (
    PreparedSupportResistanceContextReader,
    SupportResistanceFeatureProvider,
)
from crypto_strategy_lab.features.technical import CoreDirectionalFeatureProvider
from crypto_strategy_lab.gui.enhanced_config import EnhancedBacktestConfig
from crypto_strategy_lab.sr_dynamic_tp_engine import SRDynamicTPBacktestEngine
from crypto_strategy_lab.support_resistance import SRContext, SupportResistanceDetector


def canonical_klines(n: int = 120) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    wave = np.sin(np.arange(n) / 3.0) * 8.0 + np.sin(np.arange(n) / 9.0) * 3.0
    base = 120.0 + np.linspace(0, 7, n) + wave
    open_ = base + np.sin(np.arange(n) / 4.0) * 0.7
    close = base + np.cos(np.arange(n) / 5.0) * 0.9
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + pd.Timedelta(hours=4),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + np.arange(n) % 17,
            "source_fingerprint": "sr-feature-source",
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
        market_regime_method="ASSET_RETURN",
        enable_support_resistance_analysis=True,
        sr_timeframe_minutes=0,
        sr_pivot_left=3,
        sr_pivot_right=2,
        sr_lookback_bars=50,
        sr_zone_width_atr=0.4,
        sr_near_distance_atr=0.8,
        enable_sr_hold_confirmation=True,
        sr_hold_confirmation_bars=3,
        sr_hold_confirmation_atr=0.2,
        sr_break_tolerance_atr=0.25,
        sr_break_basis="CLOSE",
    )


def directional_for(frame: pd.DataFrame, cfg: EnhancedBacktestConfig):
    req = request_for(frame)
    directional = CoreDirectionalFeatureProvider().compute(
        req,
        {DatasetKind.KLINES: frame},
        {
            "atr_period": cfg.atr_period,
            "adx_period": cfg.adx_period,
            "di_pressure_lookback": cfg.di_pressure_lookback,
        },
    )
    directional.attrs["feature_cache_key"] = "directional-sr-test"
    return directional


def prepared(frame: pd.DataFrame, cfg: EnhancedBacktestConfig | None = None):
    cfg = cfg or config()
    req = request_for(frame)
    directional = directional_for(frame, cfg)
    parameters = {
        "atr_period": cfg.atr_period,
        "sr_timeframe_minutes": int(cfg.sr_timeframe_minutes or cfg.strategy_timeframe_minutes),
        "sr_pivot_left": cfg.sr_pivot_left,
        "sr_pivot_right": cfg.sr_pivot_right,
        "sr_lookback_bars": cfg.sr_lookback_bars,
        "sr_zone_width_atr": cfg.sr_zone_width_atr,
        "sr_near_distance_atr": cfg.sr_near_distance_atr,
        "enable_sr_hold_confirmation": cfg.enable_sr_hold_confirmation,
        "sr_hold_confirmation_bars": cfg.sr_hold_confirmation_bars,
        "sr_hold_confirmation_atr": cfg.sr_hold_confirmation_atr,
        "sr_break_tolerance_atr": cfg.sr_break_tolerance_atr,
        "sr_break_basis": cfg.sr_break_basis,
    }
    sr = SupportResistanceFeatureProvider().compute(
        req,
        {DatasetKind.KLINES: frame},
        parameters,
        {"core_directional": directional},
    )
    return directional, sr, parameters


def assert_context_equal(left: SRContext, right: SRContext) -> None:
    for field in fields(SRContext):
        a = getattr(left, field.name)
        b = getattr(right, field.name)
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            if pd.isna(a) and pd.isna(b):
                continue
            assert np.isclose(float(a), float(b)), field.name
        else:
            assert a == b, field.name


def test_cached_reader_reconstructs_exact_detector_context() -> None:
    frame = canonical_klines()
    directional, sr, parameters = prepared(frame)
    reader = PreparedSupportResistanceContextReader(sr)
    detector = SupportResistanceDetector(
        pivot_left=parameters["sr_pivot_left"],
        pivot_right=parameters["sr_pivot_right"],
        lookback_bars=parameters["sr_lookback_bars"],
        zone_width_atr=parameters["sr_zone_width_atr"],
        near_distance_atr=parameters["sr_near_distance_atr"],
        enable_hold_confirmation=parameters["enable_sr_hold_confirmation"],
        hold_confirmation_bars=parameters["sr_hold_confirmation_bars"],
        hold_confirmation_atr=parameters["sr_hold_confirmation_atr"],
        break_tolerance_atr=parameters["sr_break_tolerance_atr"],
        break_basis=parameters["sr_break_basis"],
    )
    open_ = frame.open.to_numpy(float)
    high = frame.high.to_numpy(float)
    low = frame.low.to_numpy(float)
    close = frame.close.to_numpy(float)
    atr = directional.atr.to_numpy(float)

    for i in range(len(frame)):
        for direction in ("LONG", "SHORT"):
            direct = detector.analyze_price_location(i, open_, high, low, close, atr, direction)
            cached = reader.analyze_price_location(i, open_, high, low, close, atr, direction)
            assert_context_equal(direct, cached)


def test_future_mutation_cannot_change_past_support_resistance() -> None:
    frame = canonical_klines()
    _, before, _ = prepared(frame)
    cutoff = 70
    changed = frame.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 1.8
    _, after, _ = prepared(changed)
    columns = [column for column in before.columns if column not in {"timestamp", "available_at"}]
    pdt.assert_frame_equal(
        before.loc[:cutoff, columns].reset_index(drop=True),
        after.loc[:cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_production_engine_uses_cached_same_timeframe_sr_without_losing_enhanced_engine() -> None:
    frame = canonical_klines()
    directional, sr, _ = prepared(frame)
    engine = DataLakeProductionBacktestEngine(
        legacy_frame(frame),
        config(),
        technical_features=directional,
        support_resistance_features=sr,
    )

    assert isinstance(engine, SRDynamicTPBacktestEngine)
    assert isinstance(engine.sr_detector, PreparedSupportResistanceContextReader)
    assert engine.support_resistance_feature_source == "support_resistance@2"
    assert engine.sr_uses_higher_timeframe is False

    index = 75
    expected = PreparedSupportResistanceContextReader(sr).analyze_price_location(
        index, None, None, None, None, None, "LONG"
    )
    actual = engine._analyze_support_resistance(index, "LONG")
    assert_context_equal(expected, actual)


def test_production_engine_preserves_higher_timeframe_sr_path_for_legacy_callers() -> None:
    frame = canonical_klines()
    cfg = replace(config(), sr_timeframe_minutes=480)
    directional = directional_for(frame, cfg)
    engine = DataLakeProductionBacktestEngine(
        legacy_frame(frame),
        cfg,
        technical_features=directional,
        support_resistance_features=None,
    )
    assert engine.sr_uses_higher_timeframe is True
    assert engine.support_resistance_feature_source == "higher_timeframe_engine"


def test_prepared_higher_timeframe_sr_matches_mature_engine_and_is_causal() -> None:
    frame = canonical_klines(180)
    cfg = replace(config(), sr_timeframe_minutes=480)
    directional, sr, _ = prepared(frame, cfg)
    reader = PreparedSupportResistanceContextReader(sr)
    mature = DataLakeProductionBacktestEngine(
        legacy_frame(frame), cfg, technical_features=directional,
        support_resistance_features=None,
    )

    completed = pd.to_datetime(sr["sr_completed_candle_time"], utc=True)
    available = pd.to_datetime(sr["available_at"], utc=True)
    assert bool(completed.notna().any())
    assert bool((completed.dropna() <= available.loc[completed.dropna().index]).all())

    for i in range(len(frame)):
        for direction in ("LONG", "SHORT"):
            expected = mature._analyze_support_resistance(i, direction)
            cached = reader.analyze_price_location(i, None, None, None, None, None, direction)
            assert_context_equal(expected, cached)


def test_future_mutation_cannot_change_past_higher_timeframe_sr() -> None:
    frame = canonical_klines(180)
    cfg = replace(config(), sr_timeframe_minutes=480)
    _, before, _ = prepared(frame, cfg)
    cutoff = 100
    changed = frame.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 2.0
    _, after, _ = prepared(changed, cfg)
    columns = [column for column in before.columns if column not in {"timestamp", "available_at"}]
    pdt.assert_frame_equal(
        before.loc[:cutoff, columns].reset_index(drop=True),
        after.loc[:cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )