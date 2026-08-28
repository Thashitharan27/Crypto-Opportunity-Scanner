from __future__ import annotations

import json

import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features.funding import FundingContextFeatureProvider


def test_microsecond_backing_dtype_keeps_funding_math_in_nanoseconds() -> None:
    starts = pd.Series(
        pd.array(
            ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            dtype="datetime64[us, UTC]",
        )
    )
    klines = pd.DataFrame(
        {
            "period_start": starts,
            "available_at": starts + pd.Timedelta(days=1),
        }
    )
    event_times = pd.Series(
        pd.array(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T08:00:00Z",
                "2026-01-01T16:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-02T08:00:00Z",
                "2026-01-02T16:00:00Z",
                "2026-01-03T00:00:00Z",
            ],
            dtype="datetime64[us, UTC]",
        )
    )
    funding = pd.DataFrame(
        {
            "available_at": event_times,
            "funding_rate": [0.0001, 0.0002, -0.0001, 0.0003, 0.0004, 0.0005, 0.0006],
            "funding_interval_hours": 8.0,
        }
    )
    request = DataRequest(
        symbol="BTCUSDT",
        start=pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2026-01-03T00:00:00Z").to_pydatetime(),
        strategy_interval="1d",
        datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE),
    )

    result = FundingContextFeatureProvider().compute(
        request,
        {DatasetKind.KLINES: klines, DatasetKind.FUNDING_RATE: funding},
        {},
    )

    # Three 8-hour settlements belong in a 24-hour trailing window, not ~3000.
    assert result["funding_24h_count"].tolist() == [3, 3]

    first = json.loads(result.loc[0, "funding_settlements_json"])
    assert [pd.Timestamp(item[0], unit="ns", tz="UTC") for item in first] == [
        pd.Timestamp("2026-01-01T08:00:00Z"),
        pd.Timestamp("2026-01-01T16:00:00Z"),
        pd.Timestamp("2026-01-02T00:00:00Z"),
    ]

    # At the source settlement itself, the next known schedule is one full 8h interval away.
    assert result.loc[0, "time_to_next_funding"] == 28800.0
