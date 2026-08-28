from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from crypto_strategy_lab.data import (
    DataQualityCache,
    DataQualityStatus,
    DataRequest,
    DatasetKind,
    classify_archive_overlap,
    validate_dataset,
)


def request(interval="1h"):
    return DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 4, tzinfo=timezone.utc),
        strategy_interval=interval,
    )


def candles():
    r = request()
    starts = pd.date_range(r.start, r.end, freq="1h", inclusive="left")
    frame = pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta("1h"),
            "available_at": starts + pd.Timedelta("1h"),
            "open": 2.0,
            "high": 3.0,
            "low": 1.0,
            "close": 2.0,
            "volume": 1.0,
            "symbol": r.symbol,
            "exchange": r.exchange,
            "market": r.market.value,
            "dataset": "klines",
            "interval": "1h",
        }
    )
    frame.attrs["canonical_source_identity"] = "source-a"
    return frame


def metrics():
    r = request()
    times = pd.to_datetime([r.start, r.start + pd.Timedelta("2h")], utc=True)
    frame = pd.DataFrame(
        {
            "event_time": times,
            "available_at": times,
            "symbol": r.symbol,
            "exchange": r.exchange,
            "market": r.market.value,
            "dataset": "metrics",
            "open_interest": [100.0, 110.0],
            "open_interest_value": [1000.0, 1100.0],
            "top_trader_account_long_short_ratio": [2.5, 1.2],
            "top_trader_position_long_short_ratio": [1.8, 1.1],
            "global_long_short_account_ratio": [1.4, 0.9],
            "taker_long_short_volume_ratio": [3.0, 1.0],
        }
    )
    frame.attrs["canonical_source_identity"] = "metrics-a"
    return frame


def _overlap_kline_row(*, volume=200_000.0, taker_base=132_462.5, close=0.6036, source="monthly"):
    start = pd.Timestamp("2023-11-30T12:35:00Z")
    return pd.DataFrame(
        {
            "period_start": [start],
            "period_end": [start + pd.Timedelta(minutes=1)],
            "event_time": [start],
            "available_at": [start + pd.Timedelta(minutes=1)],
            "open": [0.6038],
            "high": [0.6040],
            "low": [0.6034],
            "close": [close],
            "volume": [volume],
            "quote_volume": [200_298.79368],
            "trade_count": [470],
            "taker_buy_base_volume": [taker_base],
            "taker_buy_quote_volume": [79_957.28937],
            "symbol": ["XRPUSDT"],
            "exchange": ["binance"],
            "market": ["futures_um"],
            "dataset": ["klines"],
            "interval": ["1m"],
            "source_archive": [f"{source}.zip"],
            "source_fingerprint": [source],
        }
    )


def test_complete_fixed_cadence_is_ok_and_internal_gap_is_summarized():
    frame = candles()
    assert validate_dataset(
        frame, request(), DatasetKind.KLINES, interval="1h"
    ).status is DataQualityStatus.OK

    report = validate_dataset(
        frame.drop(index=[1, 2]), request(), DatasetKind.KLINES, interval="1h"
    )
    issue = next(item for item in report.issues if item.code == "MISSING_INTERNAL_INTERVAL")
    assert issue.count == 2
    assert issue.details["ranges"][0]["missing_count"] == 2
    assert report.status is DataQualityStatus.ERROR


@pytest.mark.parametrize(
    ("drop_index", "issue_code"),
    [(0, "LEADING_COVERAGE_GAP"), (3, "TRAILING_COVERAGE_GAP")],
)
def test_fixed_cadence_detects_edge_coverage_gaps(drop_index, issue_code):
    report = validate_dataset(
        candles().drop(index=[drop_index]), request(), DatasetKind.KLINES, interval="1h"
    )
    assert any(item.code == issue_code for item in report.issues)
    assert report.status is DataQualityStatus.ERROR


def test_off_grid_and_duplicate_timestamps_are_rejected():
    off_grid = candles()
    off_grid.loc[1, "period_start"] += pd.Timedelta(minutes=30)
    off_grid.loc[1, "period_end"] += pd.Timedelta(minutes=30)
    off_grid.loc[1, "available_at"] += pd.Timedelta(minutes=30)
    report = validate_dataset(off_grid, request(), DatasetKind.KLINES, interval="1h")
    assert any(item.code == "OFF_GRID_TIMESTAMP" for item in report.issues)

    duplicate = candles()
    duplicate.loc[1, ["period_start", "period_end", "available_at"]] = duplicate.loc[
        0, ["period_start", "period_end", "available_at"]
    ].to_numpy()
    report = validate_dataset(duplicate, request(), DatasetKind.KLINES, interval="1h")
    assert any(item.code == "DUPLICATE_LOGICAL_KEY" for item in report.issues)


def test_malformed_timestamp_and_missing_volume_are_rejected():
    malformed = candles()
    malformed["period_start"] = malformed["period_start"].astype(object)
    malformed.loc[1, "period_start"] = "not-a-time"
    report = validate_dataset(malformed, request(), DatasetKind.KLINES, interval="1h")
    assert any(item.code == "MALFORMED_TIMESTAMP" for item in report.issues)

    missing_volume = candles().drop(columns=["volume"])
    report = validate_dataset(missing_volume, request(), DatasetKind.KLINES, interval="1h")
    issue = next(item for item in report.issues if item.code == "MISSING_REQUIRED_COLUMN")
    assert "volume" in issue.details["columns"]


def test_kline_domain_rules_reject_negative_volume_and_impossible_ohlc():
    negative = candles()
    negative.loc[1, "volume"] = -1.0
    report = validate_dataset(negative, request(), DatasetKind.KLINES, interval="1h")
    assert any(
        item.code == "INVALID_DOMAIN_VALUE" and item.details.get("column") == "volume"
        for item in report.issues
    )

    invalid = candles()
    invalid.loc[1, "low"] = 4.0
    report = validate_dataset(invalid, request(), DatasetKind.KLINES, interval="1h")
    assert any(item.code == "INVALID_OHLC" for item in report.issues)


def test_event_streams_never_receive_regular_grid_issues():
    r = request()
    times = pd.to_datetime([r.start, r.start + pd.Timedelta("3h")], utc=True)
    funding = pd.DataFrame(
        {
            "event_time": times,
            "available_at": times,
            "funding_rate": [0.0001, -0.0002],
            "funding_interval_hours": [8.0, 8.0],
            "symbol": r.symbol,
            "exchange": r.exchange,
            "market": r.market.value,
            "dataset": "funding_rate",
        }
    )
    funding.attrs["canonical_source_identity"] = "funding-a"
    report = validate_dataset(funding, r, DatasetKind.FUNDING_RATE, interval="1m")
    assert not any("GAP" in item.code or "GRID" in item.code for item in report.issues)

    agg = pd.DataFrame(
        {
            "event_time": times,
            "available_at": times,
            "agg_trade_id": [1, 2],
            "price": [100.0, 101.0],
            "quantity": [1.0, 2.0],
            "symbol": r.symbol,
            "exchange": r.exchange,
            "market": r.market.value,
            "dataset": "agg_trades",
        }
    )
    agg.attrs["canonical_source_identity"] = "agg-a"
    report = validate_dataset(agg, r, DatasetKind.AGG_TRADES, interval="1m")
    assert not any("GAP" in item.code or "GRID" in item.code for item in report.issues)


def test_real_metrics_ratios_above_one_are_valid_but_negative_values_are_not():
    frame = metrics()
    report = validate_dataset(frame, request(), DatasetKind.FUTURES_METRICS)
    assert not any(item.code == "INVALID_DOMAIN_VALUE" for item in report.issues)

    frame.loc[0, "top_trader_account_long_short_ratio"] = -1.0
    report = validate_dataset(frame, request(), DatasetKind.FUTURES_METRICS)
    assert any(
        item.code == "INVALID_DOMAIN_VALUE"
        and item.details.get("column") == "top_trader_account_long_short_ratio"
        for item in report.issues
    )


def test_invalid_funding_interval_and_agg_trade_quantity_are_rejected():
    r = request()
    event = pd.to_datetime([r.start], utc=True)
    funding = pd.DataFrame(
        {
            "event_time": event,
            "available_at": event,
            "funding_rate": [0.0001],
            "funding_interval_hours": [0.0],
            "symbol": r.symbol,
            "exchange": r.exchange,
            "market": r.market.value,
            "dataset": "funding_rate",
        }
    )
    funding.attrs["canonical_source_identity"] = "funding-b"
    assert any(
        item.code == "INVALID_DOMAIN_VALUE"
        and item.details.get("column") == "funding_interval_hours"
        for item in validate_dataset(funding, r, DatasetKind.FUNDING_RATE).issues
    )

    agg = pd.DataFrame(
        {
            "event_time": event,
            "available_at": event,
            "agg_trade_id": [1],
            "price": [100.0],
            "quantity": [-1.0],
            "symbol": r.symbol,
            "exchange": r.exchange,
            "market": r.market.value,
            "dataset": "agg_trades",
        }
    )
    agg.attrs["canonical_source_identity"] = "agg-b"
    assert any(
        item.code == "INVALID_DOMAIN_VALUE" and item.details.get("column") == "quantity"
        for item in validate_dataset(agg, r, DatasetKind.AGG_TRADES).issues
    )


def test_archive_overlap_ignores_provenance_but_detects_market_conflicts():
    first = pd.DataFrame(
        {
            "period_start": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "close": [2.0],
            "source_archive": ["daily.zip"],
            "source_fingerprint": ["daily"],
        }
    )
    second = first.copy()
    second["source_archive"] = "monthly.zip"
    second["source_fingerprint"] = "monthly"
    issues = classify_archive_overlap([first, second], ["period_start"])
    assert issues[-1].code == "IDENTICAL_ARCHIVE_OVERLAP"

    conflicting = second.copy()
    conflicting.loc[0, "close"] = 3.0
    issues = classify_archive_overlap([first, conflicting], ["period_start"])
    assert issues[-1].code == "CONFLICTING_ARCHIVE_OVERLAP"
    assert issues[-1].severity is DataQualityStatus.ERROR


def test_archive_overlap_allows_valid_last_source_to_override_invalid_kline_row():
    bad_monthly = _overlap_kline_row(
        volume=91_695.7,
        taker_base=132_462.5,
        source="monthly",
    )
    good_daily = _overlap_kline_row(
        volume=200_000.0,
        taker_base=132_462.5,
        source="daily",
    )

    issues = classify_archive_overlap([bad_monthly, good_daily], ["period_start"])
    codes = {issue.code for issue in issues}

    assert "INVALID_SOURCE_ROW_OVERRIDDEN" in codes
    assert "CONFLICTING_ARCHIVE_OVERLAP" not in codes
    repaired = next(issue for issue in issues if issue.code == "INVALID_SOURCE_ROW_OVERRIDDEN")
    assert repaired.severity is DataQualityStatus.WARN
    assert repaired.count == 1


def test_archive_overlap_still_blocks_two_valid_sources_that_disagree():
    monthly = _overlap_kline_row(close=0.6036, source="monthly")
    daily = _overlap_kline_row(close=0.6037, source="daily")

    issues = classify_archive_overlap([monthly, daily], ["period_start"])
    conflict = next(issue for issue in issues if issue.code == "CONFLICTING_ARCHIVE_OVERLAP")

    assert conflict.severity is DataQualityStatus.ERROR
    assert conflict.count == 1
    assert not any(issue.code == "INVALID_SOURCE_ROW_OVERRIDDEN" for issue in issues)


def test_quality_cache_is_keyed_by_source_identity_and_does_not_cache_missing(tmp_path):
    cache = DataQualityCache(tmp_path)
    frame = candles()
    first = cache.get_or_validate(frame, request(), DatasetKind.KLINES, interval="1h")
    assert first.cache_hit is False
    second = cache.get_or_validate(frame, request(), DatasetKind.KLINES, interval="1h")
    assert second.cache_hit is True

    changed = candles()
    changed.attrs["canonical_source_identity"] = "source-b"
    third = cache.get_or_validate(changed, request(), DatasetKind.KLINES, interval="1h")
    assert third.cache_hit is False

    missing_first = cache.get_or_validate(None, request(), DatasetKind.FUNDING_RATE, required=False)
    missing_second = cache.get_or_validate(None, request(), DatasetKind.FUNDING_RATE, required=False)
    assert missing_first.cache_hit is False
    assert missing_second.cache_hit is False
    assert missing_second.status is DataQualityStatus.MISSING
