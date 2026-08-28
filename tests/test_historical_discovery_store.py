from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.store import MarketDataStore


def test_causal_load_filters_future_revision_before_deduplication(tmp_path: Path, monkeypatch):
    """A future correction cannot hide the candle version visible at decision time."""
    decision = datetime(2025, 1, 10, tzinfo=timezone.utc)
    start = decision - timedelta(days=1)
    visible = {
        "symbol": "BTCUSDT", "interval": "1d", "period_start": start,
        "period_end": decision, "available_at": decision,
        "open": 100, "high": 120, "low": 90, "close": 110, "quote_volume": 20_000_000,
    }
    future = {**visible, "available_at": decision + timedelta(seconds=1), "high": 999}
    store = MarketDataStore(tmp_path / "raw", tmp_path / "cache")
    record = object()
    monkeypatch.setattr(store.catalog, "records_for", lambda *args, **kwargs: [record])
    monkeypatch.setattr(store, "_ensure_canonical", lambda _: tmp_path / "rows.parquet")
    monkeypatch.setattr(
        store, "canonical_source_identity",
        lambda *args, **kwargs: type("Signature", (), {"cache_identity": lambda self: "source"})(),
    )

    class Relation:
        def df(self):
            return pd.DataFrame([visible, future])

    monkeypatch.setattr("duckdb.DuckDBPyConnection.read_parquet", lambda *args, **kwargs: Relation())
    request = DataRequest("BTCUSDT", start - timedelta(days=1), decision, "1d")
    frame = store.load_dataset(
        request, DatasetKind.KLINES, interval="1d", available_at_cutoff=decision
    )
    assert len(frame) == 1
    assert frame.iloc[0]["high"] == 120
    assert frame.iloc[0]["available_at"].to_pydatetime() == decision
