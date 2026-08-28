import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data import DataRequest, DatasetKind
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.features.order_book import (
    book_depth_resource,
    book_ticker_resource,
)
from feature_causality_harness import CausalityCase, assert_future_mutation_invariant


def _klines():
    starts = pd.date_range("2026-01-01", periods=180, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=1),
            "available_at": starts + pd.Timedelta(minutes=1),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
        }
    )


def _ticker():
    starts = pd.date_range("2026-01-01", periods=180, freq="1min", tz="UTC")
    x = np.arange(len(starts), dtype=float)
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=1),
            "available_at": starts + pd.Timedelta(minutes=1),
            "source_event_at": starts + pd.Timedelta(seconds=59),
            "best_bid_price": 100.0 + x * 0.001,
            "best_bid_qty": 4.0 + (x % 3),
            "best_ask_price": 101.0 + x * 0.001,
            "best_ask_qty": 2.0 + (x % 2),
            "book_ticker_observed": True,
            "book_ticker_covered": True,
            "book_ticker_locked": False,
        }
    )


def _depth():
    starts = pd.date_range("2026-01-01", periods=180, freq="1min", tz="UTC")
    x = np.arange(len(starts), dtype=float)
    data = {
        "period_start": starts,
        "period_end": starts + pd.Timedelta(minutes=1),
        "available_at": starts + pd.Timedelta(minutes=1),
        "source_event_at": starts + pd.Timedelta(seconds=30),
        "book_depth_observed": True,
        "book_depth_covered": True,
        "book_depth_snapshot_complete": True,
    }
    for band in range(1, 6):
        data[f"book_bid_depth_{band}pct"] = 10.0 + band + x * 0.01
        data[f"book_ask_depth_{band}pct"] = 8.0 + band + x * 0.005
        data[f"book_bid_notional_{band}pct"] = 1000.0 + band * 10 + x
        data[f"book_ask_notional_{band}pct"] = 900.0 + band * 10 + x
    return pd.DataFrame(data)


def _mutate_future_ticker(frame, cutoff):
    mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
    assert mask.any()
    frame.loc[mask, "best_bid_price"] -= 20
    frame.loc[mask, "best_ask_price"] += 20
    frame.loc[mask, "best_bid_qty"] *= 7


def _mutate_future_depth(frame, cutoff):
    mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
    assert mask.any()
    numeric_band_columns = [
        column
        for column in frame
        if column.startswith("book_bid_depth_")
        or column.startswith("book_ask_depth_")
        or column.startswith("book_bid_notional_")
        or column.startswith("book_ask_notional_")
    ]
    assert numeric_band_columns
    frame.loc[mask, numeric_band_columns] *= 9


@pytest.mark.parametrize("source", ["ticker", "depth"])
def test_order_book_context_uses_generic_available_at_causality_harness(source):
    klines = _klines()
    if source == "ticker":
        resource = book_ticker_resource()
        compact = _ticker()
        mutate = _mutate_future_ticker
    else:
        resource = book_depth_resource()
        compact = _depth()
        mutate = _mutate_future_depth
    request = DataRequest(
        "BTCUSDT",
        klines.period_start.iloc[0].to_pydatetime(),
        klines.period_end.iloc[-1].to_pydatetime(),
        "1m",
    )
    case = CausalityCase(
        feature_name="order_book_context",
        registry_factory=lambda _: production_feature_registry(),
        request=request,
        datasets={DatasetKind.KLINES: klines, resource: compact},
        parameters={
            "order_book_context": {
                "book_ticker_max_age_seconds": 120.0,
                "book_depth_max_age_seconds": 120.0,
            }
        },
        future_mutators={resource: mutate},
    )
    assert_future_mutation_invariant(case)
