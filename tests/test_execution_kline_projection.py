from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd

from crypto_strategy_lab.data import DataRequest, MarketDataStore


UTC = timezone.utc


def _write_kline_archive(
    root: Path,
    *,
    frequency: str,
    archive_name: str,
    csv_name: str,
    rows: list[str],
) -> None:
    directory = root / "data" / "futures" / "um" / frequency / "klines" / "BTCUSDT" / "1m"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / archive_name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(csv_name, "\n".join(rows) + "\n")


def test_execution_kline_overlap_uses_canonical_archive_precedence(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    timestamp = 1767225600000
    close_time = 1767225659999

    _write_kline_archive(
        raw_root,
        frequency="daily",
        archive_name="BTCUSDT-1m-2026-01-01.zip",
        csv_name="BTCUSDT-1m-2026-01-01.csv",
        rows=[
            f"{timestamp},100,103,99,101,10,{close_time},1000,4,6,600,0",
        ],
    )
    _write_kline_archive(
        raw_root,
        frequency="monthly",
        archive_name="BTCUSDT-1m-2026-01.zip",
        csv_name="BTCUSDT-1m-2026-01.csv",
        rows=[
            f"{timestamp},200,203,199,201,20,{close_time},2000,8,12,1200,0",
        ],
    )

    store = MarketDataStore(raw_root=raw_root, cache_root=tmp_path / "cache")
    assert store.refresh_catalog() == 2
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        strategy_interval="1m",
    )

    canonical = store.load_klines(request, "1m")
    projected = store.load_execution_klines(request, "1m")

    assert len(canonical) == 1
    assert len(projected) == 1
    assert float(canonical.loc[0, "close"]) == 201.0
    expected = canonical.loc[:, ["period_start", "open", "high", "low", "close", "volume"]]
    pd.testing.assert_frame_equal(projected, expected.reset_index(drop=True))
