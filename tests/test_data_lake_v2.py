from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd

from crypto_strategy_lab.data import DataRequest, DatasetKind, MarketDataStore, MarketKind
from crypto_strategy_lab.data.binance.discovery import discover_archives
from crypto_strategy_lab.data.binance.klines import KlineArchiveAdapter
from crypto_strategy_lab.data.timing import canonical_available_at, interval_to_timedelta, normalize_binance_interval


UTC = timezone.utc


def _make_archive(root: Path) -> Path:
    directory = root / "data" / "futures" / "um" / "daily" / "klines" / "BTCUSDT" / "1m"
    directory.mkdir(parents=True)
    path = directory / "BTCUSDT-1m-2026-01-01.zip"
    rows = [
        "1767225600000,100,102,99,101,10,1767225659999,1000,4,6,600,0",
        "1767225660000,101,103,100,102,12,1767225719999,1200,5,7,700,0",
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-2026-01-01.csv", "\n".join(rows) + "\n")
    return path


def test_interval_and_availability_contract() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert interval_to_timedelta("15m").total_seconds() == 900
    assert normalize_binance_interval("60m") == "1h"
    assert normalize_binance_interval("240m") == "4h"
    assert normalize_binance_interval("1440m") == "1d"
    assert normalize_binance_interval("90m") == "90m"
    assert canonical_available_at(DatasetKind.KLINES, start, interval="1m") == datetime(
        2026, 1, 1, 0, 1, tzinfo=UTC
    )
    # Event datasets are not artificially shifted into the future.
    assert canonical_available_at(DatasetKind.FUNDING_RATE, start) == start


def test_discovery_infers_standard_binance_metadata(tmp_path: Path) -> None:
    archive_path = _make_archive(tmp_path)
    records = discover_archives(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.path == archive_path.resolve()
    assert record.market == MarketKind.FUTURES_UM
    assert record.dataset == DatasetKind.KLINES
    assert record.symbol == "BTCUSDT"
    assert record.interval == "1m"
    assert record.frequency == "daily"
    assert record.period_start == datetime(2026, 1, 1, tzinfo=UTC)
    assert record.period_end == datetime(2026, 1, 2, tzinfo=UTC)


def test_kline_adapter_exposes_completed_candle_available_at(tmp_path: Path) -> None:
    _make_archive(tmp_path)
    record = discover_archives(tmp_path)[0]
    frame = KlineArchiveAdapter().read(record)
    assert len(frame) == 2
    assert frame.loc[0, "period_start"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert frame.loc[0, "available_at"] == pd.Timestamp("2026-01-01T00:01:00Z")
    assert frame.loc[1, "available_at"] == pd.Timestamp("2026-01-01T00:02:00Z")


def test_market_data_store_catalogs_caches_and_loads_by_request(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _make_archive(raw_root)
    store = MarketDataStore(raw_root=raw_root, cache_root=tmp_path / "cache")
    assert store.refresh_catalog() == 1

    request = DataRequest(
        symbol="btcusdt",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        strategy_interval="1m",
    )
    frame = store.load_klines(request)
    assert len(frame) == 2
    assert request.symbol == "BTCUSDT"
    assert frame["source_fingerprint"].nunique() == 1
    assert list((tmp_path / "cache" / "market").rglob("*.parquet"))


def test_execution_kline_load_projects_filters_and_preserves_ohlcv(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _make_archive(raw_root)
    store = MarketDataStore(raw_root=raw_root, cache_root=tmp_path / "cache")
    store.refresh_catalog()
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        strategy_interval="1m",
    )

    projected = store.load_execution_klines(request, "1m")
    canonical = store.load_klines(request, "1m")

    assert list(projected.columns) == [
        "period_start", "open", "high", "low", "close", "volume"
    ]
    pd.testing.assert_frame_equal(
        projected,
        canonical.loc[:, projected.columns].reset_index(drop=True),
    )


def test_data_request_normalizes_gui_minute_intervals_for_archive_lookup() -> None:
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_interval="240m",
        intrabar_interval="60m",
    )
    assert request.strategy_interval == "4h"
    assert request.intrabar_interval == "1h"


def test_data_request_key_changes_when_the_data_slice_changes() -> None:
    common = dict(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        strategy_interval="1h",
    )
    first = DataRequest(end=datetime(2026, 2, 1, tzinfo=UTC), **common)
    second = DataRequest(end=datetime(2026, 3, 1, tzinfo=UTC), **common)
    assert first.cache_key() != second.cache_key()
