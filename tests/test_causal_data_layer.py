from __future__ import annotations

import pandas as pd
import pytest

from crypto_strategy_lab.data.alignment import assert_causal_availability, causal_asof_join
from crypto_strategy_lab.data.resampling import resample_complete_ohlcv


def test_causal_asof_join_never_uses_future_feature() -> None:
    decisions = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "decision_time": pd.to_datetime(
                ["2026-01-01T00:04:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:09:00Z"], utc=True
            ),
        }
    )
    features = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "available_at": pd.to_datetime(["2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z"], utc=True),
            "oi_change": [1.0, 99.0],
        }
    )
    joined = causal_asof_join(decisions, features, by="symbol")
    assert pd.isna(joined.iloc[0]["oi_change"])
    assert joined.iloc[1]["oi_change"] == 1.0
    assert joined.iloc[2]["oi_change"] == 1.0
    assert_causal_availability(joined)


def test_assert_causal_availability_rejects_future_row() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-01T00:05:00Z")],
            "available_at": [pd.Timestamp("2026-01-01T00:06:00Z")],
        }
    )
    with pytest.raises(AssertionError):
        assert_causal_availability(frame)


def test_resample_emits_only_complete_higher_timeframe_bars() -> None:
    starts = pd.date_range("2026-01-01T00:00:00Z", periods=7, freq="1min")
    frame = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=1),
            "available_at": starts + pd.Timedelta(minutes=1),
            "open": range(100, 107),
            "high": range(101, 108),
            "low": range(99, 106),
            "close": range(100, 107),
            "volume": 1.0,
        }
    )
    result = resample_complete_ohlcv(frame, source_interval="1m", target_interval="5m")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["period_start"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert row["period_end"] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert row["available_at"] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert row["source_bars"] == 5
    assert row["high"] == 105
    assert row["low"] == 99
