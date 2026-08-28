from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import zipfile

from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data.backtest_service import load_backtest_bundle


UTC = timezone.utc


def _write_zip(root: Path, interval: str, rows: list[str]) -> None:
    directory = root / "raw" / "futures" / "um" / "daily" / "klines" / "BTCUSDT" / interval
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"BTCUSDT-{interval}-2026-01-01.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"BTCUSDT-{interval}-2026-01-01.csv", "\n".join(rows) + "\n")


def _row(open_ms: int, close_ms: int, price: float) -> str:
    return f"{open_ms},{price},{price + 1},{price - 1},{price + 0.25},10,{close_ms},0,1,0,0,0"


def test_intrabar_start_avoids_loading_strategy_warmup_minutes(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    start_ms = 1767225600000  # 2026-01-01 00:00 UTC
    four_hour = [
        _row(start_ms, start_ms + 4 * 3600000 - 1, 100),
        _row(start_ms + 4 * 3600000, start_ms + 8 * 3600000 - 1, 101),
    ]
    one_minute = [
        _row(start_ms + minute * 60000, start_ms + (minute + 1) * 60000 - 1, 100 + minute / 1000)
        for minute in range(8 * 60)
    ]
    _write_zip(root, "4h", four_hour)
    _write_zip(root, "1m", one_minute)

    store = MarketDataStore(root, tmp_path / "cache")
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
        strategy_interval="4h",
        intrabar_interval="1m",
    )
    bundle = load_backtest_bundle(
        store,
        request,
        market_regime_method="ASSET_RETURN",
        refresh_catalog=True,
        intrabar_start=datetime(2026, 1, 1, 4, 0, tzinfo=UTC),
    )

    assert len(bundle.strategy) == 2
    assert bundle.intrabar is not None
    assert len(bundle.intrabar) == 4 * 60
    assert bundle.intrabar["period_start"].min() == datetime(2026, 1, 1, 4, 0, tzinfo=UTC)
