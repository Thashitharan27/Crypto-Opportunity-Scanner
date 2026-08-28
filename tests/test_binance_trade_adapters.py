from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from crypto_strategy_lab.data import DataRequest, DatasetKind, MarketDataStore
from crypto_strategy_lab.data.binance.discovery import discover_archives
from crypto_strategy_lab.data.binance.trades import AggTradesArchiveAdapter


UTC = timezone.utc


def _zip(path: Path, member: str, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return path


def test_agg_trades_adapter_normalizes_aggressor_side(tmp_path: Path) -> None:
    path = tmp_path / "raw/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-01-01.zip"
    _zip(
        path,
        "BTCUSDT-aggTrades-2026-01-01.csv",
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "10,100.5,2.0,20,21,1767225601000,false\n"
        "11,100.0,1.5,22,22,1767225602000,true\n",
    )
    records = discover_archives(tmp_path / "raw")
    assert len(records) == 1
    assert records[0].dataset == DatasetKind.AGG_TRADES

    frame = AggTradesArchiveAdapter().read(records[0])
    assert list(frame["taker_side"]) == ["BUY", "SELL"]
    assert list(frame["is_buyer_maker"]) == [False, True]
    assert frame.loc[0, "quote_quantity"] == 201.0
    assert frame.loc[0, "event_time"] == pd.Timestamp("2026-01-01T00:00:01Z")
    assert frame.loc[1, "available_at"] == pd.Timestamp("2026-01-01T00:00:02Z")


def test_agg_trades_adapter_rejects_duplicate_ids_within_one_archive(tmp_path: Path) -> None:
    path = tmp_path / "raw/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-01-01.zip"
    _zip(
        path,
        "BTCUSDT-aggTrades-2026-01-01.csv",
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "10,100.5,2.0,20,21,1767225601000,false\n"
        "10,100.6,1.0,22,22,1767225602000,true\n",
    )
    record = discover_archives(tmp_path / "raw")[0]
    with pytest.raises(ValueError, match="Duplicate Binance aggregate trade IDs"):
        AggTradesArchiveAdapter().read(record)


def test_store_preserves_distinct_agg_trades_with_same_event_timestamp(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = raw / "futures/usdm/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-01-01.zip"
    _zip(
        path,
        "BTCUSDT-aggTrades-2026-01-01.csv",
        # Two aggregate trades can share a millisecond; identity is agg_trade_id,
        # not event_time.
        "1,100,1,1,1,1767225601000,false\n"
        "2,101,2,2,3,1767225601000,true\n",
    )
    store = MarketDataStore(raw, tmp_path / "cache")
    assert store.refresh_catalog() == 1
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_interval="1h",
        datasets=(DatasetKind.AGG_TRADES,),
    )
    frame = store.load_dataset(request, DatasetKind.AGG_TRADES)
    assert len(frame) == 2
    assert list(frame["agg_trade_id"]) == [1, 2]
    assert frame["event_time"].nunique() == 1
    assert frame["quantity"].sum() == 3
    assert frame.loc[1, "first_trade_id"] == 2
    assert frame.loc[1, "last_trade_id"] == 3
