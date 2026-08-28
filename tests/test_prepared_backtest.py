from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.prepared_backtest import (
    IntrabarExecutionData, PreparedBacktestFrame, ResearchContext,
    from_data_lake_bundle,
)
from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.prepared_cache import PreparedRunCache


def valid_kwargs(n=3):
    timestamp = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    floats = {
        name: np.arange(n, dtype=float) + 1
        for name in (
            "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
            "plus_di", "minus_di", "bb_width", "bb_width_pct", "session_vwap",
            "close_location", "mean_reversion_mean", "mean_reversion_distance_atr",
            "mean_reversion_distance_atr_previous", "mean_reversion_sigma",
            "mean_reversion_bb_upper", "mean_reversion_bb_lower",
            "mean_reversion_bb_zscore", "mean_reversion_rsi",
            "di_spread", "di_spread_1", "di_spread_3", "di_spread_5",
            "di_spread_change", "di_ratio", "plus_di_change", "minus_di_change",
            "di_pressure_spread_change", "long_directional_di_change",
            "long_opposing_di_change", "short_directional_di_change",
            "short_opposing_di_change", "bb_middle", "bb_upper", "bb_lower",
            "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
            "bb_width_change_pct", "mean_reversion_distance_change_atr",
        )
    }
    return dict(
        timestamp=timestamp, strategy_interval=pd.Timedelta(hours=4), **floats,
        mean_reversion_long_reentry=np.zeros(n, dtype=bool),
        mean_reversion_short_reentry=np.ones(n, dtype=bool),
        long_di_pressure_state=np.full(n, "EXPANDING", dtype=object),
        short_di_pressure_state=np.full(n, "CONTRACTING", dtype=object),
        mean_reversion_state=np.full(n, "ABOVE_MEAN", dtype=object),
        mean_reversion_motion=np.full(n, "AWAY_FROM_MEAN", dtype=object),
        mean_reversion_strength=np.ones(n, dtype=int),
        mean_reversion_strength_label=np.full(n, "MODERATE", dtype=object),
        mean_reversion_bb_location=np.full(n, "ABOVE_MIDDLE", dtype=object),
        mean_reversion_rsi_state=np.full(n, "NEUTRAL", dtype=object),
        mean_reversion_reentry_confirmation=np.full(n, "SHORT", dtype=object),
        mean_reversion_signal=np.full(n, "STRONG_SHORT", dtype=object),
        mean_reversion_signal_direction=np.full(n, "SHORT", dtype=object),
        mean_reversion_setup_strength=np.full(n, "STRONG", dtype=object),
        bb_reentry=np.full(n, "SHORT", dtype=object),
        mr_signal=np.full(n, "CONFIRMED", dtype=object),
        mr_signal_direction=np.full(n, "SHORT", dtype=object),
        bull_regime_return=np.zeros(n),
        market_regime=np.full(n, "SIDEWAYS", dtype=object),
        momentum_returns_by_hours={24: np.zeros(n)},
        decision_available_at=timestamp + np.timedelta64(4, "h"),
    )


def test_prepared_run_cache_persistent_round_trip_and_corruption(tmp_path):
    cache = PreparedRunCache(tmp_path)
    key = cache.identity(request_identity="slice", feature_identities={"di": "v2"},
                         canonical_identities={"klines": "canonical-v1"}, prepared_inputs={})
    original = PreparedBacktestFrame(**valid_kwargs())
    cache.store(key, original, provenance={"request_identity": "slice"})
    loaded = cache.load(key)
    assert loaded is not None
    assert np.array_equal(loaded.timestamp, original.timestamp)
    assert not loaded.close.flags.writeable
    cache.paths(key)[0].write_bytes(b"broken")
    assert cache.load(key) is None


def test_prepared_run_identity_is_dependency_aware(tmp_path):
    cache = PreparedRunCache(tmp_path)
    common = dict(request_identity="slice", canonical_identities={"klines": "k1"}, prepared_inputs={})
    first = cache.identity(feature_identities={"di": "d1", "funding": "f1"}, **common)
    assert first == cache.identity(feature_identities={"di": "d1", "funding": "f1"}, **common)
    assert first != cache.identity(feature_identities={"di": "d2", "funding": "f1"}, **common)
    # Execution/reporting fields never enter this explicit prepared input contract.
    assert first == cache.identity(feature_identities={"di": "d1", "funding": "f1"}, **common)


def data_lake_bundle():
    times = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
    strategy = pd.DataFrame({
        "period_start": times, "available_at": times + pd.Timedelta(hours=4),
        "open": 1.0, "high": 2.0, "low": .5,
        "close": 1.5, "volume": 10.0,
    })
    technical = pd.DataFrame({
        "timestamp": times, "available_at": times + pd.Timedelta(hours=4),
        "atr": 1.0, "atr_pct": .1, "adx": 20.0, "plus_di": 15.0, "minus_di": 10.0,
        **{name: 1.0 for name in (
            "di_spread", "di_spread_1", "di_spread_3", "di_spread_5",
            "di_spread_change", "di_ratio", "plus_di_change", "minus_di_change",
            "di_pressure_spread_change", "long_directional_di_change",
            "long_opposing_di_change", "short_directional_di_change",
            "short_opposing_di_change")},
        "long_di_pressure_state": "EXPANDING", "short_di_pressure_state": "CONTRACTING",
    })
    context_names = (
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_width_pct",
        "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
        "bb_width_change_pct", "session_vwap", "close_location",
        "mean_reversion_mean", "mean_reversion_distance_atr",
        "mean_reversion_distance_atr_previous", "mean_reversion_sigma",
        "mean_reversion_bb_upper", "mean_reversion_bb_lower", "mean_reversion_bb_zscore",
        "mean_reversion_rsi", "mean_reversion_distance_change_atr",
    )
    context = pd.DataFrame({
        "timestamp": times, "available_at": times + pd.Timedelta(hours=4),
        **{name: 1.0 for name in context_names},
        "mean_reversion_long_reentry": False,
        "mean_reversion_short_reentry": True,
        "mean_reversion_state": "ABOVE_MEAN",
        "mean_reversion_motion": "AWAY_FROM_MEAN",
        "mean_reversion_strength": 1,
        "mean_reversion_strength_label": "MODERATE",
        "mean_reversion_bb_location": "ABOVE_MIDDLE",
        "mean_reversion_rsi_state": "NEUTRAL",
        "mean_reversion_reentry_confirmation": "SHORT",
        "mean_reversion_signal": "STRONG_SHORT",
        "mean_reversion_signal_direction": "SHORT",
        "mean_reversion_setup_strength": "STRONG",
        "bb_reentry": "SHORT",
        "mr_signal": "CONFIRMED",
        "mr_signal_direction": "SHORT",
    })
    intrabar_times = pd.date_range(times[0], periods=720, freq="1min", tz="UTC")
    intrabar = pd.DataFrame({
        "period_start": intrabar_times, "open": 1.0, "high": 2.0, "low": .5,
    })
    research = pd.DataFrame({
        "timestamp": times, "available_at": times + pd.Timedelta(hours=4), "funding": .01,
    })
    return SimpleNamespace(
        strategy=strategy, technical_features=technical, context_features=context,
        research_features={"funding": research}, intrabar=intrabar,
        request=SimpleNamespace(strategy_interval="4h", intrabar_interval="1m"),
    )


def test_required_fields_are_explicit():
    kwargs = valid_kwargs()
    del kwargs["atr"]
    with pytest.raises(TypeError, match="atr"):
        PreparedBacktestFrame(**kwargs)


def test_production_runtime_constructs_natively_from_prepared_arrays(monkeypatch):
    prepared = PreparedBacktestFrame(**valid_kwargs(30))
    intrabar_times = pd.date_range("2025-01-01", periods=60, freq="1min", tz="UTC")
    intrabar = IntrabarExecutionData(
        intrabar_times, pd.Timedelta(minutes=1),
        np.ones(60), np.ones(60) * 2, np.ones(60) * .5,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy constructor was invoked")

    monkeypatch.setattr("crypto_strategy_lab.data_lake_engine.DataLakeBacktestEngine.__init__", forbidden)
    config = replace(BacktestConfig(), strategy_timeframe_minutes=240, intrabar_timeframe_minutes=1, telemetry_interval_minutes=240)
    engine = DataLakeProductionBacktestEngine.from_prepared(prepared, intrabar, config)

    assert engine.prepared_frame is prepared
    assert engine.intrabar_data is intrabar
    assert engine.data is None
    assert engine.open is prepared.open
    window = engine.intrabar_data.fast_window(intrabar_times[5], intrabar_times[8])
    assert [row[0] for row in window.rows()] == [5, 6, 7]


def test_array_lengths_must_match():
    kwargs = valid_kwargs()
    kwargs["adx"] = np.ones(2)
    with pytest.raises(ValueError, match="adx length"):
        PreparedBacktestFrame(**kwargs)


@pytest.mark.parametrize("timestamps", [
    np.array(["2025-01-01T04:00", "2025-01-01T00:00"], dtype="datetime64[m]"),
    np.array(["2025-01-01T00:00", "2025-01-01T00:00"], dtype="datetime64[m]"),
])
def test_timestamps_are_strictly_ordered_and_unique(timestamps):
    kwargs = valid_kwargs(2)
    kwargs["timestamp"] = timestamps
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        PreparedBacktestFrame(**kwargs)


def test_dtype_and_missing_execution_values_fail_loudly():
    kwargs = valid_kwargs()
    kwargs["atr"] = np.array(["not", "numeric", "values"])
    with pytest.raises(TypeError, match="atr must have a numeric dtype"):
        PreparedBacktestFrame(**kwargs)
    kwargs = valid_kwargs()
    kwargs["close"][1] = np.nan
    with pytest.raises(ValueError, match="close contains missing"):
        PreparedBacktestFrame(**kwargs)


def test_optional_research_is_read_only_and_aligned():
    kwargs = valid_kwargs()
    block = ResearchContext("flow", kwargs["decision_available_at"], {"imbalance": np.ones(3)})
    frame = PreparedBacktestFrame(**kwargs, research=(block,))
    assert frame.research[0].values["imbalance"].flags.writeable is False
    bad = ResearchContext("flow", kwargs["decision_available_at"][:2], {"imbalance": np.ones(2)})
    with pytest.raises(ValueError, match="not aligned"):
        PreparedBacktestFrame(**kwargs, research=(bad,))


def test_causal_feature_rules():
    kwargs = valid_kwargs()
    kwargs["decision_available_at"][0] = kwargs["timestamp"][0] - np.timedelta64(1, "m")
    with pytest.raises(ValueError, match="before their strategy candle opens"):
        PreparedBacktestFrame(**kwargs)
    kwargs = valid_kwargs()
    kwargs["decision_available_at"][0] += np.timedelta64(1, "m")
    with pytest.raises(ValueError, match="unavailable at strategy candle completion"):
        PreparedBacktestFrame(**kwargs)


def test_intrabar_strategy_alignment_and_interval():
    frame = PreparedBacktestFrame(**valid_kwargs())
    times = pd.date_range("2025-01-01", periods=720, freq="1min", tz="UTC").to_numpy(dtype="datetime64[ns]")
    intrabar = IntrabarExecutionData(times, pd.Timedelta(minutes=1), np.ones(720), np.ones(720), np.ones(720))
    intrabar.validate_compatible(frame)
    shifted = IntrabarExecutionData(times + np.timedelta64(30, "s"), pd.Timedelta(minutes=1), np.ones(720), np.ones(720), np.ones(720))
    with pytest.raises(ValueError, match="not aligned"):
        shifted.validate_compatible(frame)
    same_interval = IntrabarExecutionData(frame.timestamp, pd.Timedelta(hours=4), np.ones(3), np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="must be smaller"):
        same_interval.validate_compatible(frame)


def test_intrabar_partial_coverage_is_compatible_when_it_overlaps():
    frame = PreparedBacktestFrame(**valid_kwargs())
    partial_start = pd.date_range(
        "2025-01-01T04:00:00Z", periods=120, freq="1min"
    ).to_numpy(dtype="datetime64[ns]")
    partial_end = pd.date_range(
        "2025-01-01T00:00:00Z", periods=120, freq="1min"
    ).to_numpy(dtype="datetime64[ns]")
    for times in (partial_start, partial_end):
        data = IntrabarExecutionData(
            times, pd.Timedelta(minutes=1), np.ones(len(times)), np.ones(len(times)), np.ones(len(times))
        )
        data.validate_compatible(frame)

    after = pd.date_range(
        "2025-01-01T12:00:00Z", periods=10, freq="1min"
    ).to_numpy(dtype="datetime64[ns]")
    data = IntrabarExecutionData(
        after, pd.Timedelta(minutes=1), np.ones(10), np.ones(10), np.ones(10)
    )
    with pytest.raises(ValueError, match="does not overlap"):
        data.validate_compatible(frame)


def test_constructs_from_current_data_lake_bundle_path():
    prepared, execution = from_data_lake_bundle(data_lake_bundle())
    assert len(prepared) == 3
    assert np.array_equal(
        prepared.timestamp,
        data_lake_bundle().strategy["period_start"].to_numpy(dtype="datetime64[ns]"),
    )
    assert np.array_equal(prepared.open, np.ones(3))
    assert prepared.research[0].name == "funding"
    assert execution is not None and len(execution.timestamp) == 720


def test_bundle_requires_real_volume_instead_of_fabricating_it():
    bundle = data_lake_bundle()
    bundle.strategy = bundle.strategy.drop(columns=["volume"])
    with pytest.raises(ValueError, match="strategy data is missing required columns.*volume"):
        from_data_lake_bundle(bundle)


@pytest.mark.parametrize("block", ["technical", "context", "research"])
def test_bundle_feature_timestamps_must_exactly_match_strategy(block):
    bundle = data_lake_bundle()
    if block == "technical":
        bundle.technical_features = bundle.technical_features.copy()
        bundle.technical_features["timestamp"] += pd.Timedelta(minutes=1)
        expected = "technical features timestamps are not exactly aligned"
    elif block == "context":
        bundle.context_features = bundle.context_features.copy()
        bundle.context_features["timestamp"] += pd.Timedelta(minutes=1)
        expected = "context features timestamps are not exactly aligned"
    else:
        shifted = bundle.research_features["funding"].copy()
        shifted["timestamp"] += pd.Timedelta(minutes=1)
        bundle.research_features = {"funding": shifted}
        expected = "research feature funding timestamps are not exactly aligned"

    with pytest.raises(ValueError, match=expected):
        from_data_lake_bundle(bundle)
