from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data.binance.events import (
    BookDepthArchiveAdapter,
    BookTickerArchiveAdapter,
)
from crypto_strategy_lab.data.order_book import OrderBookSnapshotStore
from crypto_strategy_lab.data.schemas import ArchiveRecord, DatasetKind, MarketKind
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.features.order_book import (
    OrderBookContextFeatureProvider,
    book_depth_resource,
    book_ticker_resource,
)
from crypto_strategy_lab.features.trade_flow import trade_flow_resource

UTC = timezone.utc


def _record(
    dataset,
    *,
    path=Path("source.zip"),
    raw_root=None,
    start=datetime(2026, 1, 1, tzinfo=UTC),
    end=datetime(2026, 1, 2, tzinfo=UTC),
    fingerprint="fixture",
):
    return ArchiveRecord(
        raw_root=Path(raw_root) if raw_root is not None else Path("."),
        path=Path(path),
        market=MarketKind.FUTURES_UM,
        dataset=dataset,
        symbol="BTCUSDT",
        interval=None,
        frequency="daily",
        period_start=start,
        period_end=end,
        size_bytes=Path(path).stat().st_size if Path(path).exists() else 1,
        mtime_ns=Path(path).stat().st_mtime_ns if Path(path).exists() else 1,
        fingerprint=fingerprint,
    )


def _request(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 2, tzinfo=UTC)):
    return DataRequest(
        "BTCUSDT",
        start,
        end,
        "4h",
        market=MarketKind.FUTURES_UM,
    )


def _strategy(decision="2026-01-01 12:00:00Z"):
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime([decision], utc=True),
            "available_at": pd.to_datetime([decision], utc=True),
        }
    )


def _ticker_events(times=None):
    times = times or ["2026-01-01 11:59:59.999Z", "2026-01-01 12:00:00Z"]
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime(times, utc=True, format="mixed"),
            "update_id": [1, 2],
            "best_bid_price": [100.0, 50.0],
            "best_bid_qty": [4.0, 1.0],
            "best_ask_price": [102.0, 60.0],
            "best_ask_qty": [1.0, 1.0],
        }
    )


def _complete_depth_events(timestamp="2026-01-01 11:59:59Z"):
    percentages = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime([timestamp] * len(percentages), utc=True),
            "percentage": percentages,
            "depth": [10, 9, 8, 7, 4, 1, 5, 6, 7, 8],
            "notional": [1000, 900, 800, 700, 400, 100, 500, 600, 700, 800],
        }
    )


def _write_ticker(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _write_depth(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,percentage,depth,notional\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_book_ticker_adapter_sorts_out_of_order_and_deduplicates_identical(tmp_path):
    path = tmp_path / "ticker.csv"
    _write_ticker(
        path,
        [
            "2,101,2,102,3,1767225601000,1767225601000",
            "1,100,4,101,5,1767225600000,1767225600000",
            "1,100,4,101,5,1767225600000,1767225600000",
        ],
    )
    frame = BookTickerArchiveAdapter().read(
        _record(DatasetKind.BOOK_TICKER, path=path, raw_root=tmp_path)
    )
    assert frame.update_id.tolist() == [1, 2]
    assert frame.event_time.is_monotonic_increasing
    assert frame.available_at.equals(frame.event_time)


def test_book_ticker_adapter_rejects_conflicting_duplicate_update_id(tmp_path):
    path = tmp_path / "ticker.csv"
    _write_ticker(
        path,
        [
            "1,100,4,101,5,1767225600000,1767225600000",
            "1,99,4,101,5,1767225600000,1767225600000",
        ],
    )
    with pytest.raises(ValueError, match="Conflicting duplicate bookTicker"):
        BookTickerArchiveAdapter().read(
            _record(DatasetKind.BOOK_TICKER, path=path, raw_root=tmp_path)
        )


@pytest.mark.parametrize(
    "epoch, expected",
    [
        (1767225599999, pd.Timestamp("2025-12-31 23:59:59.999Z")),
        (1767225599999000, pd.Timestamp("2025-12-31 23:59:59.999Z")),
    ],
)
def test_book_ticker_adapter_supports_ms_and_us_archives(tmp_path, epoch, expected):
    path = tmp_path / f"ticker-{epoch}.csv"
    _write_ticker(path, [f"1,100,4,101,5,{epoch},{epoch}"])
    frame = BookTickerArchiveAdapter().read(
        _record(DatasetKind.BOOK_TICKER, path=path, raw_root=tmp_path)
    )
    assert frame.event_time.iloc[0] == expected


def test_book_depth_adapter_preserves_percentage_bands_and_rejects_conflicts(tmp_path):
    path = tmp_path / "depth.csv"
    _write_depth(
        path,
        [
            "2026-01-01 12:00:00,-1,4,400",
            "2026-01-01 12:00:00,1,1,100",
        ],
    )
    frame = BookDepthArchiveAdapter().read(
        _record(DatasetKind.BOOK_DEPTH, path=path, raw_root=tmp_path)
    )
    assert frame.percentage.tolist() == [-1.0, 1.0]
    assert "update_id" not in frame

    _write_depth(
        path,
        [
            "2026-01-01 12:00:00,-1,4,400",
            "2026-01-01 12:00:00,-1,5,500",
        ],
    )
    with pytest.raises(ValueError, match="Conflicting duplicate bookDepth"):
        BookDepthArchiveAdapter().read(
            _record(DatasetKind.BOOK_DEPTH, path=path, raw_root=tmp_path)
        )


def test_ticker_snapshot_is_causal_and_formulas_are_exact():
    events = _ticker_events(
        ["2026-01-01 12:00:00Z", "2026-01-01 11:59:59.999Z"]
    )
    events["update_id"] = [2, 1]
    events["best_bid_price"] = [50, 100]
    events["best_bid_qty"] = [1, 4]
    events["best_ask_price"] = [60, 102]
    events["best_ask_qty"] = [1, 1]
    compact = OrderBookSnapshotStore._compact_ticker(
        _record(DatasetKind.BOOK_TICKER), events
    )
    assert list(compact["available_at"]) == list(
        pd.to_datetime(["2026-01-01 12:00Z", "2026-01-01 12:01Z"], utc=True)
    )
    result = OrderBookContextFeatureProvider().compute(
        None,
        {DatasetKind.KLINES: _strategy(), book_ticker_resource(): compact},
        {"book_ticker_max_age_seconds": 5, "book_depth_max_age_seconds": 90},
    )
    row = result.iloc[0]
    assert row.book_best_bid_price == 100
    assert row.book_spread == 2
    assert row.book_midprice == 101
    assert row.book_spread_bps == pytest.approx(2 / 101 * 10000)
    assert row.book_imbalance_l1 == pytest.approx(3 / 5)
    assert row.book_microprice == pytest.approx((102 * 4 + 100) / 5)
    assert row.book_microprice_offset_bps == pytest.approx(
        (row.book_microprice - 101) / 101 * 10000
    )


def test_depth_partial_snapshot_retains_nan_and_band_formulas():
    events = _complete_depth_events().iloc[:-1].copy()
    compact = OrderBookSnapshotStore._compact_depth(
        _record(DatasetKind.BOOK_DEPTH), events
    )
    assert not bool(compact.iloc[0].book_depth_snapshot_complete)
    assert np.isnan(compact.iloc[0].book_ask_depth_5pct)
    result = OrderBookContextFeatureProvider().compute(
        None,
        {DatasetKind.KLINES: _strategy(), book_depth_resource(): compact},
        {"book_ticker_max_age_seconds": 5, "book_depth_max_age_seconds": 90},
    )
    assert result.iloc[0].book_bid_depth_1pct == 4
    assert result.iloc[0].book_ask_depth_1pct == 1
    assert result.iloc[0].book_depth_imbalance_1pct == pytest.approx(3 / 5)
    assert result.iloc[0].book_depth_ratio_1pct == 4
    assert np.isnan(result.iloc[0].book_ask_depth_5pct)


def test_stale_ticker_and_depth_values_are_masked():
    ticker_events = _ticker_events(["2026-01-01 11:59:40Z", "2026-01-01 12:00:00Z"])
    ticker = OrderBookSnapshotStore._compact_ticker(
        _record(DatasetKind.BOOK_TICKER), ticker_events
    ).iloc[:1]
    depth = OrderBookSnapshotStore._compact_depth(
        _record(DatasetKind.BOOK_DEPTH),
        _complete_depth_events("2026-01-01 11:58:00Z"),
    )
    result = OrderBookContextFeatureProvider().compute(
        None,
        {
            DatasetKind.KLINES: _strategy(),
            book_ticker_resource(): ticker,
            book_depth_resource(): depth,
        },
        {"book_ticker_max_age_seconds": 5, "book_depth_max_age_seconds": 90},
    )
    row = result.iloc[0]
    assert bool(row.book_ticker_stale)
    assert np.isnan(row.book_spread)
    assert bool(row.book_depth_stale)
    assert np.isnan(row.book_bid_depth_1pct)
    assert pd.isna(row.book_depth_snapshot_complete)


def test_provider_rejects_raw_order_book_events():
    raw_ticker = OrderBookSnapshotStore._compact_ticker(
        _record(DatasetKind.BOOK_TICKER), _ticker_events()
    )
    raw_ticker["update_id"] = 1
    with pytest.raises(ValueError, match="raw order-book event columns"):
        OrderBookContextFeatureProvider().compute(
            None,
            {DatasetKind.KLINES: _strategy(), book_ticker_resource(): raw_ticker},
            {"book_ticker_max_age_seconds": 5, "book_depth_max_age_seconds": 90},
        )


def test_provider_normalizes_microsecond_snapshot_against_nanosecond_strategy():
    compact = OrderBookSnapshotStore._compact_ticker(
        _record(DatasetKind.BOOK_TICKER), _ticker_events()
    )
    for column in ("period_start", "period_end", "available_at", "source_event_at"):
        compact[column] = compact[column].astype("datetime64[us, UTC]")
    strategy = _strategy()
    assert str(compact.available_at.dtype) == "datetime64[us, UTC]"
    assert str(strategy.available_at.dtype) == "datetime64[ns, UTC]"
    result = OrderBookContextFeatureProvider().compute(
        None,
        {DatasetKind.KLINES: strategy, book_ticker_resource(): compact},
        {"book_ticker_max_age_seconds": 5, "book_depth_max_age_seconds": 90},
    )
    assert str(result.available_at.dtype) == "datetime64[ns, UTC]"
    assert result.iloc[0].book_best_bid_price == 100


def test_snapshot_cache_is_partition_local_and_never_calls_multi_year_load_dataset(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    day1 = raw / "bookTicker" / "BTCUSDT" / "daily" / "BTCUSDT-bookTicker-2026-01-01.csv"
    day2 = raw / "bookTicker" / "BTCUSDT" / "daily" / "BTCUSDT-bookTicker-2026-01-02.csv"
    _write_ticker(day1, ["1,100,4,101,5,1767225600000,1767225600000"])
    _write_ticker(day2, ["2,101,4,102,5,1767312000000,1767312000000"])
    rec1 = _record(
        DatasetKind.BOOK_TICKER,
        path=day1,
        raw_root=raw,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        fingerprint="day1",
    )
    rec2 = _record(
        DatasetKind.BOOK_TICKER,
        path=day2,
        raw_root=raw,
        start=datetime(2026, 1, 2, tzinfo=UTC),
        end=datetime(2026, 1, 3, tzinfo=UTC),
        fingerprint="day2",
    )
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [rec1, rec2])

    def forbidden_load(*args, **kwargs):
        raise AssertionError("order-book snapshot path must not call load_dataset")

    monkeypatch.setattr(store, "load_dataset", forbidden_load)
    snapshots = OrderBookSnapshotStore(store)
    req = _request(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)
    )
    first = snapshots.load(req, DatasetKind.BOOK_TICKER)
    assert first.partitions_built == 2 and first.partitions_reused == 0
    assert first.source_event_count == 2
    second = snapshots.load(req, DatasetKind.BOOK_TICKER)
    assert second.cache_hit
    assert second.partitions_built == 0 and second.partitions_reused == 2
    assert second.source_event_count == 2

    rec2_changed = _record(
        DatasetKind.BOOK_TICKER,
        path=day2,
        raw_root=raw,
        start=datetime(2026, 1, 2, tzinfo=UTC),
        end=datetime(2026, 1, 3, tzinfo=UTC),
        fingerprint="day2-v2",
    )
    store.catalog.sync_root(raw, [rec1, rec2_changed])
    third = snapshots.load(req, DatasetKind.BOOK_TICKER)
    assert third.partitions_built == 1
    assert third.partitions_reused == 1
    assert third.source_identity != second.source_identity


def test_order_book_quality_is_bounded_and_warm_cache_is_metadata_only(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    path = raw / "bookTicker" / "BTCUSDT" / "daily" / "BTCUSDT-bookTicker-2026-01-01.csv"
    _write_ticker(
        path,
        [
            "1,100,4,101,5,1767225600000,1767225600000",
            "2,100,4,101,5,1767225660000,1767225660000",
        ],
    )
    rec = _record(
        DatasetKind.BOOK_TICKER,
        path=path,
        raw_root=raw,
        fingerprint="ticker-day",
    )
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [rec])

    def forbidden_load(*args, **kwargs):
        raise AssertionError("book quality must not concatenate through load_dataset")

    monkeypatch.setattr(store, "load_dataset", forbidden_load)
    req = _request()
    first = store.data_quality_report(req, DatasetKind.BOOK_TICKER, required=False)
    assert first.status.value == "OK"
    assert not first.cache_hit

    # Warm quality must not reopen the source adapter either.
    monkeypatch.setattr(
        store._adapter_for(DatasetKind.BOOK_TICKER),
        "read",
        lambda *_: (_ for _ in ()).throw(AssertionError("warm quality reopened source")),
    )
    second = store.data_quality_report(req, DatasetKind.BOOK_TICKER, required=False)
    assert second.cache_hit
    assert second.row_count == first.row_count


def test_order_book_feature_identity_ignores_unrelated_auxiliary_resources():
    registry = production_feature_registry()
    params = {
        "order_book_context": {
            "book_ticker_max_age_seconds": 5.0,
            "book_depth_max_age_seconds": 90.0,
        }
    }
    resolved = registry.resolve(["order_book_context"], params)[0]
    req = _request()
    ticker_resource = book_ticker_resource()
    unrelated = trade_flow_resource(DatasetKind.AGG_TRADES)
    base = {
        DatasetKind.KLINES: "kline",
        ticker_resource: "book-A",
        unrelated: "trade-A",
    }
    identity = registry.identity(resolved, req, base, {})
    changed_unrelated = dict(base)
    changed_unrelated[unrelated] = "trade-B"
    assert registry.identity(resolved, req, changed_unrelated, {}) == identity
    changed_book = dict(base)
    changed_book[ticker_resource] = "book-B"
    assert registry.identity(resolved, req, changed_book, {}) != identity
