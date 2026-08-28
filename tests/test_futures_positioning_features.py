from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.features.futures_positioning import (
    FUTURES_POSITIONING_FEATURE_NAME,
    FuturesPositioningFeatureProvider,
    futures_positioning_price_resource,
)
from feature_causality_harness import CausalityCase, assert_future_mutation_invariant


def klines(n: int = 6) -> pd.DataFrame:
    starts = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="4h")
    close = 100.0 + np.arange(n) * 2.0
    return pd.DataFrame(
        {
            "period_start": starts,
            "available_at": starts + pd.Timedelta(hours=4),
            "close": close,
            "source_fingerprint": "kline-source-a",
        }
    )


def metrics() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2026-01-01T03:55:00Z",
            "2026-01-01T04:05:00Z",  # future relative to first candle close
            "2026-01-01T07:55:00Z",
            "2026-01-01T11:55:00Z",
            "2026-01-01T15:55:00Z",
            "2026-01-01T19:55:00Z",
            "2026-01-01T23:55:00Z",
        ],
        utc=True,
    )
    oi = [100.0, 999.0, 110.0, 105.0, 120.0, 126.0, 121.0]
    return pd.DataFrame(
        {
            "available_at": times,
            "open_interest": oi,
            "open_interest_value": np.asarray(oi) * 1000.0,
            "top_trader_account_long_short_ratio": [1.1, 9.9, 1.2, 0.9, 1.3, 1.4, 1.1],
            "top_trader_position_long_short_ratio": [1.05, 9.8, 1.1, 0.95, 1.2, 1.3, 1.0],
            "global_long_short_account_ratio": [0.95, 9.7, 1.0, 1.05, 1.1, 0.9, 0.85],
            "taker_long_short_volume_ratio": [1.2, 9.6, 0.8, 1.3, 1.4, 0.7, 0.9],
            "source_fingerprint": "metrics-source-a",
        }
    )


def request(frame: pd.DataFrame) -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(hours=4)).to_pydatetime(),
        strategy_interval="4h",
        datasets=(DatasetKind.KLINES, DatasetKind.FUTURES_METRICS),
    )


def prepared(kline_frame=None, metrics_frame=None, price_frame=None) -> pd.DataFrame:
    kline_frame = klines() if kline_frame is None else kline_frame
    metrics_frame = metrics() if metrics_frame is None else metrics_frame
    datasets: dict[object, pd.DataFrame] = {
        DatasetKind.KLINES: kline_frame,
        DatasetKind.FUTURES_METRICS: metrics_frame,
    }
    if price_frame is not None:
        datasets[futures_positioning_price_resource()] = price_frame
    return FuturesPositioningFeatureProvider().compute(
        request(kline_frame), datasets, {}
    )


def test_positioning_uses_latest_snapshot_available_at_candle_close() -> None:
    result = prepared()

    assert result.loc[0, "metrics_source_available_at"] == pd.Timestamp("2026-01-01T03:55:00Z")
    assert result.loc[0, "open_interest"] == 100.0
    assert result.loc[0, "metrics_age_seconds"] == 300.0
    assert result.loc[1, "metrics_source_available_at"] == pd.Timestamp("2026-01-01T07:55:00Z")
    assert result.loc[1, "open_interest"] == 110.0
    assert np.isclose(result.loc[1, "open_interest_change_1bar_pct"], 0.10)
    assert result.loc[1, "price_oi_state"] == "PRICE_UP_OI_UP"
    assert np.isclose(result.loc[0, "taker_long_short_volume_bias"], 0.2)
    assert bool((result["metrics_source_available_at"] <= result["available_at"]).all())
    # Without a true 1h source the provider refuses to relabel 4h strategy prices.
    assert result["price_change_pct_1h"].isna().all()
    assert set(result["oi_vs_price_state_1h"]) == {"UNKNOWN"}


def test_oi_horizons_and_price_state_are_source_native_not_strategy_bar_lags() -> None:
    strategy = klines(1)
    strategy.loc[:, "close"] = 999.0  # deliberately unrelated to the 1h price source
    metric_times = pd.to_datetime(
        ["2026-01-01T03:00:00Z", "2026-01-01T03:55:00Z", "2026-01-01T04:00:00Z"],
        utc=True,
    )
    metric_frame = pd.DataFrame(
        {
            "available_at": metric_times,
            "open_interest": [100.0, 110.0, 120.0],
            "open_interest_value": [1000.0, 1100.0, 1200.0],
        }
    )
    price_frame = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                ["2026-01-01T03:00:00Z", "2026-01-01T04:00:00Z"], utc=True
            ),
            "close": [100.0, 110.0],
        }
    )
    out = prepared(strategy, metric_frame, price_frame)
    assert out.loc[0, "oi_change_5m"] == 10.0
    assert out.loc[0, "oi_change_pct_5m"] == pytest.approx(10.0 / 110.0)
    assert out.loc[0, "oi_change_1h"] == 20.0
    assert out.loc[0, "oi_change_pct_1h"] == pytest.approx(0.20)
    assert out.loc[0, "price_change_pct_1h"] == pytest.approx(0.10)
    assert out.loc[0, "oi_vs_price_state_1h"] == "PRICE_UP_OI_UP"


def test_future_metrics_mutation_cannot_change_past_positioning() -> None:
    source = metrics()
    before = prepared(metrics_frame=source)
    cutoff_time = pd.Timestamp("2026-01-01T12:00:00Z")

    changed = source.copy()
    changed.loc[changed.available_at > cutoff_time, "open_interest"] *= 100.0
    changed.loc[changed.available_at > cutoff_time, "taker_long_short_volume_ratio"] = 99.0
    after = prepared(metrics_frame=changed)

    past_rows = before.available_at <= cutoff_time
    columns = [
        "open_interest",
        "open_interest_change_1bar_pct",
        "open_interest_change_3bar_pct",
        "taker_long_short_volume_ratio",
        "taker_long_short_volume_bias",
        "price_oi_state",
    ]
    pdt.assert_frame_equal(
        before.loc[past_rows, columns].reset_index(drop=True),
        after.loc[past_rows, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_future_one_hour_price_mutation_cannot_change_past_oi_price_state() -> None:
    strategy = klines(4)
    price_times = pd.date_range("2025-12-31T23:00:00Z", periods=18, freq="1h")
    price = pd.DataFrame(
        {"available_at": price_times, "close": 100.0 + np.arange(len(price_times))}
    )
    before = prepared(strategy, metrics(), price)
    cutoff = pd.Timestamp("2026-01-01T08:00:00Z")
    changed = price.copy()
    changed.loc[changed.available_at > cutoff, "close"] *= 10.0
    after = prepared(strategy, metrics(), changed)
    mask = before.available_at <= cutoff
    pdt.assert_frame_equal(
        before.loc[mask, ["price_change_pct_1h", "oi_vs_price_state_1h"]].reset_index(drop=True),
        after.loc[mask, ["price_change_pct_1h", "oi_vs_price_state_1h"]].reset_index(drop=True),
        check_dtype=False,
    )


def test_positioning_all_sources_participate_in_generic_causality_harness() -> None:
    strategy = klines(6)
    metric_frame = metrics()
    resource = futures_positioning_price_resource()
    price = pd.DataFrame(
        {
            "available_at": pd.date_range("2025-12-31T23:00:00Z", periods=26, freq="1h"),
            "close": 100.0 + np.arange(26),
        }
    )

    def mutate_metrics(frame: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
        frame.loc[mask, "open_interest"] *= 3.0

    def mutate_price(frame: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
        frame.loc[mask, "close"] *= 2.0

    case = CausalityCase(
        feature_name=FUTURES_POSITIONING_FEATURE_NAME,
        registry_factory=lambda _: production_feature_registry(),
        request=request(strategy),
        datasets={
            DatasetKind.KLINES: strategy,
            DatasetKind.FUTURES_METRICS: metric_frame,
            resource: price,
        },
        parameters={FUTURES_POSITIONING_FEATURE_NAME: {}},
        future_mutators={
            DatasetKind.FUTURES_METRICS: mutate_metrics,
            resource: mutate_price,
        },
    )
    assert_future_mutation_invariant(case)


def test_missing_optional_metrics_columns_remain_nan_without_breaking_alignment() -> None:
    sparse = metrics()[["available_at", "open_interest", "source_fingerprint"]].copy()
    result = prepared(metrics_frame=sparse)
    assert result["open_interest"].notna().all()
    assert result["top_trader_account_long_short_ratio"].isna().all()
    assert result["global_long_short_account_bias"].isna().all()
