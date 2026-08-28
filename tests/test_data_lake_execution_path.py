from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data.backtest_service import load_backtest_bundle
from crypto_strategy_lab.data_lake_engine import DataLakeBacktestEngine
from crypto_strategy_lab.features.market_regime import structural_regime_values


UTC = timezone.utc


def _strategy_frame() -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="4h")
    close = pd.Series(range(100, 130), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": times,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
        }
    )


def _benchmark_frame() -> pd.DataFrame:
    times = pd.date_range("2025-12-28T00:00:00Z", periods=9 * 24, freq="1h")
    close = pd.Series(range(100, 100 + len(times)), dtype=float)
    return pd.DataFrame({"period_start": times, "close": close})


def test_structural_regime_does_not_change_when_future_benchmark_rows_change() -> None:
    benchmark = _benchmark_frame()
    strategy_times = pd.to_datetime(
        ["2026-01-02T12:00:00Z", "2026-01-04T12:00:00Z", "2026-01-06T12:00:00Z"],
        utc=True,
    )
    original = structural_regime_values(
        strategy_times,
        benchmark,
        sma_days=2,
        slope_lookback_days=1,
    )

    changed = benchmark.copy()
    cutoff = pd.Timestamp("2026-01-04T12:00:00Z")
    changed.loc[changed["period_start"] > cutoff, "close"] *= 0.01
    mutated = structural_regime_values(
        strategy_times,
        changed,
        sma_days=2,
        slope_lookback_days=1,
    )

    assert original[0] == mutated[0]
    assert original[1] == mutated[1]


def test_data_lake_engine_ignores_missing_benchmark_csv_when_frame_is_injected() -> None:
    config = replace(
        BacktestConfig(),
        strategy_timeframe_minutes=240,
        intrabar_timeframe_minutes=1,
        use_intrabar_data=False,
        telemetry_interval_minutes=240,
        enable_trade_telemetry=False,
        market_regime_method="BTC_STRUCTURAL",
        structural_regime_sma_days=2,
        structural_regime_slope_lookback_days=1,
        structural_regime_benchmark_csv=Path("Z:/this-file-does-not-exist.csv"),
    )
    engine = DataLakeBacktestEngine(
        _strategy_frame(),
        config,
        structural_benchmark=_benchmark_frame(),
    )

    assert len(engine.market_regime_values) == len(engine.data)
    assert any(
        value in {"BULL", "BEAR", "SIDEWAYS"}
        for value in engine.market_regime_values
        if value is not None
    )


def _write_kline_zip(root: Path, interval: str, rows: list[str], day: str = "2026-01-01") -> None:
    directory = root / "raw" / "futures" / "um" / "daily" / "klines" / "BTCUSDT" / interval
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"BTCUSDT-{interval}-{day}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"BTCUSDT-{interval}-{day}.csv", "\n".join(rows) + "\n")


def test_backtest_bundle_loads_structural_benchmark_from_store(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    four_hour_rows = [
        "1767225600000,100,103,99,102,10,1767239999999,0,1,0,0,0",
        "1767240000000,102,104,101,103,11,1767254399999,0,1,0,0,0",
    ]
    _write_kline_zip(root, "4h", four_hour_rows)

    # Structural preparation requests sma + slope + 7 days of causal warmup.
    # Give the catalog honest daily archive metadata for that whole window rather
    # than hiding old rows in a file named only for 2026-01-01.
    for day_offset in range(-10, 1):
        day = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=day_offset)
        start_ms = int(day.timestamp() * 1000)
        rows = []
        for hour in range(24):
            close = 100.0 + (day_offset + 10) * 0.5 + hour * 0.1
            open_ms = start_ms + hour * 3_600_000
            rows.append(
                f"{open_ms},{close - 0.5},{close + 1},{close - 1},{close},10,"
                f"{open_ms + 3_599_999},0,1,0,0,0"
            )
        _write_kline_zip(root, "1h", rows, day.date().isoformat())

    store = MarketDataStore(root, tmp_path / "cache")
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
        strategy_interval="4h",
    )
    bundle = load_backtest_bundle(
        store,
        request,
        market_regime_method="BTC_STRUCTURAL",
        structural_regime_sma_days=2,
        structural_regime_slope_lookback_days=1,
    )

    assert len(bundle.strategy) == 2
    assert "period_start" in bundle.strategy
    assert "timestamp" not in bundle.strategy
    assert bundle.strategy.attrs["canonical_source_identity"]
    assert bundle.intrabar is None
    assert bundle.structural_benchmark is not None
    assert len(bundle.structural_benchmark) == 10 * 24 + 8
    assert bundle.structural_benchmark_symbol == "BTCUSDT"
    assert bundle.structural_benchmark_interval == "1h"
