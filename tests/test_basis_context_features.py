from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features.basis import BasisContextFeatureProvider


def _klines(n: int = 12, freq: str = "1h") -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    delta = pd.Timedelta(freq)
    trade = 100.0 + np.arange(n) * 0.5
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + delta,
            "close": trade,
            "source_fingerprint": "trade-source",
        }
    )


def _reference(frame: pd.DataFrame, prices: np.ndarray, fingerprint: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_start": frame["period_start"],
            "available_at": frame["available_at"],
            "close": prices,
            "source_fingerprint": fingerprint,
        }
    )


def _request(frame: pd.DataFrame, interval: str = "1h") -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(interval)).to_pydatetime(),
        strategy_interval=interval,
    )


def _prepared(frame: pd.DataFrame, *, mutate_after: int | None = None) -> pd.DataFrame:
    n = len(frame)
    index_price = 99.5 + np.arange(n) * 0.5
    mark_price = index_price * 1.001
    premium = np.full(n, 0.001)
    if mutate_after is not None:
        mark_price[mutate_after + 1 :] *= 1.2
        index_price[mutate_after + 1 :] *= 0.8
        premium[mutate_after + 1 :] = 0.2
    return BasisContextFeatureProvider().compute(
        _request(frame),
        {
            DatasetKind.KLINES: frame,
            DatasetKind.MARK_PRICE_KLINES: _reference(frame, mark_price, "mark-source"),
            DatasetKind.INDEX_PRICE_KLINES: _reference(frame, index_price, "index-source"),
            DatasetKind.PREMIUM_INDEX_KLINES: _reference(frame, premium, "premium-source"),
        },
        {},
    )


def test_basis_context_calculates_mark_index_and_trade_basis() -> None:
    frame = _klines()
    out = _prepared(frame)

    assert np.allclose(out["mark_index_basis"], 0.001)
    assert np.allclose(out["mark_index_basis_bps"], 10.0)
    assert set(out["mark_index_basis_state"]) == {"POSITIVE"}
    assert np.allclose(out["premium_index_close"], 0.001)
    assert (out["mark_age_seconds"] == 0).all()
    assert (out["index_age_seconds"] == 0).all()
    assert (out["premium_age_seconds"] == 0).all()
    assert (out["mark_source_available_at"] <= out["available_at"]).all()
    assert (out["index_source_available_at"] <= out["available_at"]).all()

    expected_trade_mark = (
        frame.loc[0, "close"] - out.loc[0, "mark_price"]
    ) / out.loc[0, "mark_price"]
    expected_trade_index = (
        frame.loc[0, "close"] - out.loc[0, "index_price"]
    ) / out.loc[0, "index_price"]
    assert out.loc[0, "trade_mark_basis"] == expected_trade_mark
    assert out.loc[0, "trade_index_basis"] == expected_trade_index


def test_basis_change_is_native_reference_change_not_strategy_sampled_change() -> None:
    strategy = _klines(2, "4h")
    reference_available = pd.date_range(
        "2026-01-01T01:00:00Z", periods=8, freq="1h"
    )
    index = pd.DataFrame(
        {"available_at": reference_available, "close": np.full(8, 100.0)}
    )
    # Basis increases by exactly 1 percentage point each native 1h observation.
    mark = pd.DataFrame(
        {
            "available_at": reference_available,
            "close": 100.0 * (1.0 + np.arange(1, 9) / 100.0),
        }
    )
    out = BasisContextFeatureProvider().compute(
        _request(strategy, "4h"),
        {
            DatasetKind.KLINES: strategy,
            DatasetKind.MARK_PRICE_KLINES: mark,
            DatasetKind.INDEX_PRICE_KLINES: index,
        },
        {"basis_zscore_window_days": 7.0, "basis_zscore_min_samples": 2},
    )
    assert out.loc[0, "mark_index_basis"] == pytest.approx(0.04)
    assert out.loc[1, "mark_index_basis"] == pytest.approx(0.08)
    # At 08:00 the native previous mark candle is 07:00 (7%), so change is 1%,
    # not the 4% difference from the previous 4h strategy sample.
    assert out.loc[1, "mark_index_basis_change"] == pytest.approx(0.01)
    assert np.isfinite(out.loc[1, "mark_index_basis_zscore_7d"])


def test_future_reference_mutation_cannot_change_past_basis() -> None:
    frame = _klines()
    cutoff = 6
    before = _prepared(frame)
    after = _prepared(frame, mutate_after=cutoff)
    columns = [
        "mark_price",
        "index_price",
        "premium_index_close",
        "mark_index_basis",
        "mark_index_basis_bps",
        "mark_index_basis_state",
        "trade_mark_basis",
        "trade_index_basis",
        "mark_index_basis_change",
        "mark_index_basis_zscore_7d",
        "premium_index_change",
        "premium_index_zscore_7d",
    ]
    pdt.assert_frame_equal(
        before.loc[:cutoff, columns].reset_index(drop=True),
        after.loc[:cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_basis_context_allows_missing_premium_archive() -> None:
    frame = _klines()
    n = len(frame)
    index_price = 99.5 + np.arange(n) * 0.5
    mark_price = index_price * 1.001
    out = BasisContextFeatureProvider().compute(
        _request(frame),
        {
            DatasetKind.KLINES: frame,
            DatasetKind.MARK_PRICE_KLINES: _reference(frame, mark_price, "mark-source"),
            DatasetKind.INDEX_PRICE_KLINES: _reference(frame, index_price, "index-source"),
        },
        {},
    )
    assert out["premium_source_available_at"].isna().all()
    assert out["premium_index_close"].isna().all()
    assert out["premium_age_seconds"].isna().all()
    assert out["premium_index_change"].isna().all()
    assert out["premium_index_zscore_7d"].isna().all()
