from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data import DataRequest, DatasetKind, MarketKind, MarketDataStore
from crypto_strategy_lab.data.binance.trades import AggTradesArchiveAdapter, TradesArchiveAdapter
from crypto_strategy_lab.data.schemas import ArchiveRecord
from crypto_strategy_lab.data.trade_aggregates import TradeAggregateStore
from crypto_strategy_lab.features.trade_flow import (
    TradeFlowContextFeatureProvider,
    trade_flow_resource,
)


UTC = timezone.utc


def record(
    path: Path,
    dataset=DatasetKind.TRADES,
    *,
    start=datetime(2026, 1, 1, tzinfo=UTC),
    end=datetime(2026, 1, 2, tzinfo=UTC),
    fingerprint="source",
    raw_root: Path | None = None,
):
    return ArchiveRecord(
        raw_root or path.parent,
        path,
        MarketKind.FUTURES_UM,
        dataset,
        "BTCUSDT",
        None,
        "daily",
        start,
        end,
        path.stat().st_size,
        path.stat().st_mtime_ns,
        fingerprint,
        "binance",
    )


def write_trades(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "trade_id,price,quantity,quote_quantity,transact_time,is_buyer_maker\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def request(start, end):
    return DataRequest(
        "BTCUSDT",
        start,
        end,
        "4h",
        market=MarketKind.FUTURES_UM,
    )


def aggregate_fixture(starts: pd.DatetimeIndex, *, covered=True) -> pd.DataFrame:
    n = len(starts)
    covered_values = np.full(n, covered, dtype=bool)
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=1),
            "available_at": starts + pd.Timedelta(minutes=1),
            "trade_flow_source_covered": covered_values,
            "underlying_trade_count": np.ones(n),
            "source_event_count": np.ones(n),
            "base_volume": np.ones(n),
            "quote_volume": np.full(n, 100.0),
            "aggressive_buy_base_volume": np.ones(n),
            "aggressive_sell_base_volume": np.zeros(n),
            "aggressive_buy_quote_volume": np.full(n, 100.0),
            "aggressive_sell_quote_volume": np.zeros(n),
            "trade_delta_base": np.ones(n),
            "trade_delta_quote": np.full(n, 100.0),
            "weighted_price_sum": np.full(n, 100.0),
            "large_source_event_count": np.full(n, np.nan),
            "large_source_event_quote_volume": np.full(n, np.nan),
            "large_buy_quote_volume": np.full(n, np.nan),
            "large_sell_quote_volume": np.full(n, np.nan),
            "median_source_event_size": np.ones(n),
            "last_event_at": starts + pd.Timedelta(seconds=30),
        }
    )


def test_raw_trades_adapter_direction_and_microsecond_timestamps(tmp_path):
    path = tmp_path / "trades-us.csv"
    write_trades(
        path,
        [
            "1,100,2,200,1767225599999000,false",
            "2,101,1,101,1767225600000000,true",
        ],
    )
    frame = TradesArchiveAdapter().read(record(path))
    assert frame.taker_side.tolist() == ["BUY", "SELL"]
    assert frame.available_at.equals(frame.event_time)
    assert frame.event_time.iloc[0] == pd.Timestamp("2025-12-31 23:59:59.999Z")
    assert frame.event_time.iloc[1] == pd.Timestamp("2026-01-01 00:00:00Z")


def test_raw_trades_adapter_millisecond_timestamps(tmp_path):
    path = tmp_path / "trades-ms.csv"
    write_trades(
        path,
        [
            "1,100,2,200,1767225599999,false",
            "2,101,1,101,1767225600000,true",
        ],
    )
    frame = TradesArchiveAdapter().read(record(path))
    assert frame.taker_side.tolist() == ["BUY", "SELL"]
    assert frame.event_time.iloc[0] == pd.Timestamp("2025-12-31 23:59:59.999Z")
    assert frame.event_time.iloc[1] == pd.Timestamp("2026-01-01 00:00:00Z")


def test_raw_trades_adapter_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "trades.csv"
    write_trades(
        path,
        [
            "1,100,1,100,1767225600000,false",
            "1,101,1,101,1767225601000,true",
        ],
    )
    with pytest.raises(ValueError, match="Duplicate Binance trade IDs"):
        TradesArchiveAdapter().read(record(path))


def test_agg_trades_adapter_preserves_underlying_ids(tmp_path):
    path = tmp_path / "agg.csv"
    path.write_text(
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "7,100,3,10,12,1767225600000,false\n",
        encoding="utf-8",
    )
    frame = AggTradesArchiveAdapter().read(record(path, DatasetKind.AGG_TRADES))
    assert frame.agg_trade_id.tolist() == [7]
    assert frame.first_trade_id.tolist() == [10]
    assert frame.last_trade_id.tolist() == [12]
    assert frame.taker_side.tolist() == ["BUY"]


def test_exact_aggregate_boundary_and_large_event_semantics(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text("x", encoding="utf-8")
    rec = record(path)
    events = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-01-01 11:59:59.999Z", "2026-01-01 12:00:00Z"],
                format="mixed",
            ),
            "trade_id": [1, 2],
            "price": [100.0, 110.0],
            "quantity": [2.0, 1.0],
            "quote_quantity": [200.0, 110.0],
            "is_buyer_maker": [False, True],
        }
    )
    out = TradeAggregateStore._aggregate(rec, events, 150.0)
    first = out.loc[out.period_start == pd.Timestamp("2026-01-01 11:59Z")].iloc[0]
    second = out.loc[out.period_start == pd.Timestamp("2026-01-01 12:00Z")].iloc[0]
    assert first.available_at == pd.Timestamp("2026-01-01 12:00Z")
    assert first.source_event_count == first.underlying_trade_count == 1
    assert first.aggressive_buy_base_volume == 2
    assert first.trade_delta_base == 2
    assert first.weighted_price_sum / first.base_volume == 100
    assert first.large_source_event_count == 1
    assert first.large_buy_quote_volume == 200
    assert second.aggressive_sell_base_volume == 1
    assert second.trade_delta_base == -1
    assert second.large_source_event_count == 0


def test_agg_trade_source_event_and_underlying_counts_differ(tmp_path):
    path = tmp_path / "agg.csv"
    path.write_text("x", encoding="utf-8")
    rec = record(path, DatasetKind.AGG_TRADES)
    events = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-01-01 12:00:10Z"]),
            "agg_trade_id": [1],
            "price": [100.0],
            "quantity": [3.0],
            "quote_quantity": [300.0],
            "first_trade_id": [10],
            "last_trade_id": [12],
            "is_buyer_maker": [False],
        }
    )
    out = TradeAggregateStore._aggregate(rec, events, None)
    row = out.loc[out.period_start == pd.Timestamp("2026-01-01 12:00Z")].iloc[0]
    assert row.source_event_count == 1
    assert row.underlying_trade_count == 3
    assert row.base_volume / row.underlying_trade_count == 1
    assert row.median_source_event_size == 3


def test_aggregate_cache_is_partition_local_and_never_calls_multi_year_load_dataset(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    day1 = raw / "day1.csv"
    day2 = raw / "day2.csv"
    write_trades(day1, ["1,100,1,100,1767225600000,false"])
    write_trades(day2, ["2,110,1,110,1767312000000,true"])
    rec1 = record(day1, start=datetime(2026, 1, 1, tzinfo=UTC),
                  end=datetime(2026, 1, 2, tzinfo=UTC), fingerprint="day1", raw_root=raw)
    rec2 = record(day2, start=datetime(2026, 1, 2, tzinfo=UTC),
                  end=datetime(2026, 1, 3, tzinfo=UTC), fingerprint="day2", raw_root=raw)
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [rec1, rec2])

    def forbidden_load(*args, **kwargs):
        raise AssertionError("trade aggregate path must not call MarketDataStore.load_dataset")

    monkeypatch.setattr(store, "load_dataset", forbidden_load)
    aggregates = TradeAggregateStore(store)
    req = request(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC))
    first = aggregates.load(req, DatasetKind.TRADES)
    assert first.partitions_built == 2 and first.partitions_reused == 0
    second = aggregates.load(req, DatasetKind.TRADES)
    assert second.cache_hit and second.partitions_built == 0 and second.partitions_reused == 2
    assert first.source_identity == second.source_identity

    # Change only the second immutable source identity. Day 1 must remain reusable.
    rec2_changed = record(day2, start=datetime(2026, 1, 2, tzinfo=UTC),
                          end=datetime(2026, 1, 3, tzinfo=UTC), fingerprint="day2-v2", raw_root=raw)
    store.catalog.sync_root(raw, [rec1, rec2_changed])
    third = aggregates.load(req, DatasetKind.TRADES)
    assert third.partitions_built == 1
    assert third.partitions_reused == 1
    assert third.source_identity != second.source_identity


def test_large_threshold_changes_aggregate_identity(tmp_path):
    raw = tmp_path / "raw"
    path = raw / "day.csv"
    write_trades(path, ["1,100,2,200,1767225600000,false"])
    rec = record(path, fingerprint="day", raw_root=raw)
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [rec])
    aggregates = TradeAggregateStore(store)
    req = request(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC))
    low = aggregates.load(req, DatasetKind.TRADES, large_trade_quote_threshold=150.0)
    high = aggregates.load(req, DatasetKind.TRADES, large_trade_quote_threshold=250.0)
    assert low.source_identity != high.source_identity
    assert low.partitions_built == high.partitions_built == 1


def test_covered_zero_minute_is_distinct_from_unavailable_minute(tmp_path):
    raw = tmp_path / "raw"
    path = raw / "day.csv"
    write_trades(path, ["1,100,1,100,1767225600000,false"])
    rec = record(path, fingerprint="day", raw_root=raw)
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [rec])
    result = TradeAggregateStore(store).load(
        request(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)),
        DatasetKind.TRADES,
    ).frame
    covered_empty = result.loc[result.period_start == pd.Timestamp("2026-01-01 00:10Z")].iloc[0]
    unavailable = result.loc[result.period_start == pd.Timestamp("2026-01-02 00:10Z")].iloc[0]
    assert bool(covered_empty.trade_flow_source_covered)
    assert covered_empty.source_event_count == 0
    assert covered_empty.base_volume == 0
    assert not bool(unavailable.trade_flow_source_covered)
    assert pd.isna(unavailable.source_event_count)
    assert pd.isna(unavailable.base_volume)


def test_provider_rejects_raw_event_resource():
    req = request(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC))
    klines = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2026-01-01 00:00Z"]),
            "available_at": pd.to_datetime(["2026-01-01 00:01Z"]),
        }
    )
    raw = pd.DataFrame({"trade_id": [1]})
    with pytest.raises(ValueError, match="raw trade-event"):
        TradeFlowContextFeatureProvider().compute(
            req,
            {DatasetKind.KLINES: klines, trade_flow_resource(DatasetKind.TRADES): raw},
            {"trade_flow_source": "TRADES", "trade_flow_windows": ("1m",)},
        )


def test_cvd_uses_utc_day_prefix_and_large_metrics_are_queryable(tmp_path):
    raw = tmp_path / "raw"
    path = raw / "day.csv"
    write_trades(
        path,
        [
            "1,100,1,100,1767261600000,false",  # 10:00 UTC
            "2,100,2,200,1767268800000,false",  # 12:00 UTC
        ],
    )
    rec = record(path, fingerprint="day", raw_root=raw)
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [rec])
    req = request(datetime(2026, 1, 1, 12, tzinfo=UTC), datetime(2026, 1, 1, 13, tzinfo=UTC))
    aggregate = TradeAggregateStore(store).load(
        req, DatasetKind.TRADES, large_trade_quote_threshold=150.0
    ).frame
    klines = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2026-01-01 12:00Z"]),
            "available_at": pd.to_datetime(["2026-01-01 12:01Z"]),
        }
    )
    output = TradeFlowContextFeatureProvider().compute(
        req,
        {DatasetKind.KLINES: klines, trade_flow_resource(DatasetKind.TRADES): aggregate},
        {"trade_flow_source": "TRADES", "trade_flow_windows": ("1m",)},
    )
    row = output.iloc[0]
    assert row.cvd_utc_day == pytest.approx(3.0)
    assert row.large_source_event_volume_share_1m == pytest.approx(1.0)
    assert row.large_buy_share_1m == pytest.approx(1.0)
    assert row.median_source_event_size_1m == pytest.approx(2.0)


@pytest.mark.parametrize("source", [DatasetKind.AGG_TRADES, DatasetKind.TRADES])
def test_future_raw_event_mutation_cannot_change_past_aggregate_or_feature(source, tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("x", encoding="utf-8")
    rec = record(
        path,
        source,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 1, 2, tzinfo=UTC),
    )
    starts = pd.to_datetime(
        ["2026-01-01 00:10Z", "2026-01-01 00:50Z", "2026-01-01 01:10Z", "2026-01-01 01:50Z"]
    )
    events = pd.DataFrame(
        {
            "event_time": starts,
            "price": [100.0, 101.0, 102.0, 103.0],
            "quantity": [1.0, 1.0, 1.0, 1.0],
            "quote_quantity": [100.0, 101.0, 102.0, 103.0],
            "is_buyer_maker": [False, True, False, True],
        }
    )
    if source is DatasetKind.AGG_TRADES:
        events["agg_trade_id"] = [1, 2, 3, 4]
        events["first_trade_id"] = [10, 20, 30, 40]
        events["last_trade_id"] = [11, 21, 31, 41]
    else:
        events["trade_id"] = [1, 2, 3, 4]

    before_aggregate = TradeAggregateStore._aggregate(rec, events, None)
    cutoff = pd.Timestamp("2026-01-01 01:00Z")
    changed_events = events.copy(deep=True)
    future = changed_events.event_time > cutoff
    changed_events.loc[future, "quantity"] *= 9
    changed_events.loc[future, "quote_quantity"] *= 9
    changed_events.loc[future, "is_buyer_maker"] = ~changed_events.loc[
        future, "is_buyer_maker"
    ].astype(bool)
    after_aggregate = TradeAggregateStore._aggregate(rec, changed_events, None)
    past = before_aggregate.available_at <= cutoff
    pd.testing.assert_frame_equal(
        before_aggregate.loc[past].reset_index(drop=True),
        after_aggregate.loc[past].reset_index(drop=True),
    )

    kline_starts = pd.date_range("2026-01-01 00:00Z", periods=120, freq="1min")
    klines = pd.DataFrame(
        {
            "period_start": kline_starts,
            "available_at": kline_starts + pd.Timedelta(minutes=1),
        }
    )
    req = DataRequest(
        "BTCUSDT",
        kline_starts[0].to_pydatetime(),
        (kline_starts[-1] + pd.Timedelta(minutes=1)).to_pydatetime(),
        "1m",
    )
    provider = TradeFlowContextFeatureProvider()
    resource = trade_flow_resource(source)
    params = {"trade_flow_source": source.name, "trade_flow_windows": ("1m", "5m", "1h")}
    before = provider.compute(req, {DatasetKind.KLINES: klines, resource: before_aggregate}, params)
    after = provider.compute(req, {DatasetKind.KLINES: klines, resource: after_aggregate}, params)
    mask = before.available_at <= cutoff
    pd.testing.assert_frame_equal(
        before.loc[mask].reset_index(drop=True),
        after.loc[mask].reset_index(drop=True),
    )
