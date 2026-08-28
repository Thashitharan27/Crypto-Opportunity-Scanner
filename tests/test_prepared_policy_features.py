from dataclasses import replace

import numpy as np
import pandas as pd

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.features.market_regime import causal_trailing_return
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from test_prepared_backtest import valid_kwargs
from crypto_strategy_lab.prepared_backtest import PreparedBacktestFrame


def test_trailing_return_is_prefix_causal():
    times = pd.date_range("2025-01-01", periods=20, freq="4h", tz="UTC")
    close = np.arange(20, dtype=float) + 100
    baseline = causal_trailing_return(times, close, pd.Timedelta(hours=12))
    changed = close.copy()
    changed[15:] *= 100
    candidate = causal_trailing_return(times, changed, pd.Timedelta(hours=12))
    np.testing.assert_allclose(candidate[:15], baseline[:15], equal_nan=True)


def test_native_runtime_reads_prepared_feature_families(monkeypatch):
    prepared = PreparedBacktestFrame(**valid_kwargs(30))
    config = replace(
        BacktestConfig(), strategy_timeframe_minutes=240,
        telemetry_interval_minutes=240,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime feature calculator called")

    for name in ("_trailing_return_array", "_trailing_return_hours_array", "_market_regime_array"):
        monkeypatch.setattr(DataLakeProductionBacktestEngine, name, forbidden)
    engine = DataLakeProductionBacktestEngine.from_prepared(prepared, None, config)
    assert engine.di_spread is prepared.di_spread
    assert engine.mean_reversion_state is prepared.mean_reversion_state
    assert engine.market_regime_values is prepared.market_regime
    assert engine.profile_momentum_values[24] is prepared.momentum_returns_by_hours[24]


def test_native_mr_snapshot_uses_prepared_market_classifications(monkeypatch):
    prepared = PreparedBacktestFrame(**valid_kwargs(30))
    config = replace(
        BacktestConfig(), strategy_timeframe_minutes=240,
        telemetry_interval_minutes=240,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native runtime recalculated an MR market classification")

    for name in ("classify_bb_location", "classify_rsi_state", "classify_signal", "signal_direction"):
        monkeypatch.setattr(f"crypto_strategy_lab.enhanced_engine.{name}", forbidden)
    engine = DataLakeProductionBacktestEngine.from_prepared(prepared, None, config)
    snapshot = engine._mean_reversion_snapshot(5, "LONG", "LONG")
    assert snapshot["mean_reversion_bb_location"] == prepared.mean_reversion_bb_location[5]
    assert snapshot["mean_reversion_rsi_state"] == prepared.mean_reversion_rsi_state[5]
    assert snapshot["mean_reversion_signal"] == prepared.mean_reversion_signal[5]
    assert snapshot["mean_reversion_signal_direction"] == prepared.mean_reversion_signal_direction[5]


def test_native_di_snapshot_preserves_unknown_direction_semantics():
    prepared = PreparedBacktestFrame(**valid_kwargs(30))
    config = replace(
        BacktestConfig(), strategy_timeframe_minutes=240,
        telemetry_interval_minutes=240,
    )
    engine = DataLakeProductionBacktestEngine.from_prepared(prepared, None, config)
    snapshot = engine._di_pressure_snapshot(5, None)
    assert np.isnan(snapshot["plus_di_change"])
    assert np.isnan(snapshot["minus_di_change"])
    assert np.isnan(snapshot["di_spread_change"])
    assert snapshot["di_pressure_state"] == "UNKNOWN"