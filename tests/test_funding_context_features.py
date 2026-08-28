from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features.funding import FundingContextFeatureProvider


def klines(n: int = 8) -> pd.DataFrame:
    starts = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "period_start": starts,
            "available_at": starts + pd.Timedelta(hours=4),
            "source_fingerprint": "funding-kline-source",
        }
    )


def funding_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                [
                    "2025-12-31T16:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T04:05:00Z",  # must not leak into 04:00 decision
                    "2026-01-01T08:00:00Z",
                    "2026-01-01T16:00:00Z",
                    "2026-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "funding_rate": [0.0001, 0.0002, 0.0099, -0.0001, 0.0003, 0.0004],
            "funding_interval_hours": [8, 8, 8, 8, 8, 8],
            "source_fingerprint": "funding-event-source",
        }
    )


def request(frame: pd.DataFrame) -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(hours=4)).to_pydatetime(),
        strategy_interval="4h",
        datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE),
    )


def prepared(kline_frame=None, funding_frame=None, parameters=None) -> pd.DataFrame:
    kline_frame = klines() if kline_frame is None else kline_frame
    funding_frame = funding_events() if funding_frame is None else funding_frame
    return FundingContextFeatureProvider().compute(
        request(kline_frame),
        {
            DatasetKind.KLINES: kline_frame,
            DatasetKind.FUNDING_RATE: funding_frame,
        },
        parameters or {},
    )


def test_latest_funding_event_is_causally_aligned() -> None:
    result = prepared()

    assert result.loc[0, "funding_source_available_at"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert np.isclose(result.loc[0, "funding_rate"], 0.0002)
    assert np.isclose(result.loc[0, "funding_rate_bps"], 2.0)
    assert result.loc[0, "funding_bias"] == "POSITIVE"
    assert np.isclose(result.loc[0, "funding_age_hours"], 4.0)
    assert result.loc[0, "time_to_next_funding"] == pytest.approx(4 * 3600)

    assert result.loc[1, "funding_source_available_at"] == pd.Timestamp("2026-01-01T08:00:00Z")
    assert np.isclose(result.loc[1, "funding_rate"], -0.0001)
    assert result.loc[1, "funding_bias"] == "NEGATIVE"
    assert result.loc[1, "funding_event_changed"]
    assert result.loc[1, "time_to_next_funding"] == pytest.approx(8 * 3600)
    assert bool((result["funding_source_available_at"] <= result["available_at"]).all())


def test_funding_previous_change_and_three_event_mean_use_event_timeline() -> None:
    result = prepared()
    row = result.loc[1]  # 08:00 strategy decision, latest event is exactly 08:00
    assert row["funding_previous"] == pytest.approx(0.0099)
    assert row["funding_change"] == pytest.approx(-0.0100)
    assert row["funding_3_event_mean"] == pytest.approx((0.0002 + 0.0099 - 0.0001) / 3)
    # The early decision has insufficient 7d sample history: unknown is nullable,
    # not silently converted to False / "not extreme".
    assert pd.isna(result.loc[0, "funding_extreme_positive"])
    assert pd.isna(result.loc[0, "funding_extreme_negative"])


def test_trailing_24h_funding_uses_only_events_known_by_decision_time() -> None:
    result = prepared()

    assert result.loc[0, "funding_24h_count"] == 2
    assert np.isclose(result.loc[0, "funding_24h_sum"], 0.0003)
    assert result.loc[1, "funding_24h_count"] == 4
    assert np.isclose(result.loc[1, "funding_24h_sum"], 0.0101)


def test_time_to_next_funding_rolls_known_cadence_without_reading_a_future_event() -> None:
    strategy = klines(5)  # final decision at 20:00
    events = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                ["2025-12-31T16:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z"],
                utc=True,
            ),
            "funding_rate": [0.0001, 0.0002, -0.0001],
            "funding_interval_hours": [8.0, 8.0, 8.0],
        }
    )
    out = prepared(strategy, events)
    final = out.iloc[-1]
    assert final["funding_source_available_at"] == pd.Timestamp("2026-01-01T08:00:00Z")
    # 16:00 is a missed expected event; based only on the already-known 8h cadence,
    # the next scheduled boundary after 20:00 is 00:00 -> four hours away.
    assert final["time_to_next_funding"] == pytest.approx(4 * 3600)


def test_funding_interval_can_be_inferred_only_from_past_published_events() -> None:
    strategy = klines(5)
    events = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                ["2025-12-31T16:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", "2026-01-01T16:00:00Z"],
                utc=True,
            ),
            "funding_rate": [0.0001, 0.0002, -0.0001, 0.0003],
        }
    )
    out = prepared(strategy, events)
    final = out.iloc[-1]
    assert final["funding_interval_hours"] == pytest.approx(8.0)
    assert final["time_to_next_funding"] == pytest.approx(4 * 3600)


def test_future_funding_mutation_cannot_change_past_context() -> None:
    source = funding_events()
    before = prepared(funding_frame=source)
    cutoff = pd.Timestamp("2026-01-01T12:00:00Z")

    changed = source.copy()
    changed.loc[changed.available_at > cutoff, "funding_rate"] = 0.5
    after = prepared(funding_frame=changed)

    mask = before.available_at <= cutoff
    columns = [
        "funding_rate",
        "funding_previous",
        "funding_change",
        "funding_3_event_mean",
        "funding_rate_bps",
        "funding_bias",
        "funding_24h_sum",
        "funding_24h_count",
        "time_to_next_funding",
    ]
    pdt.assert_frame_equal(
        before.loc[mask, columns].reset_index(drop=True),
        after.loc[mask, columns].reset_index(drop=True),
        check_dtype=False,
    )
