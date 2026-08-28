from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd

from crypto_strategy_lab.data import (
    DataQualityStatus,
    DataRequest,
    DatasetKind,
    MarketDataStore,
)
from crypto_strategy_lab.data.binance.discovery import discover_archives
from crypto_strategy_lab.data.binance.events import (
    FundingRateArchiveAdapter,
    FuturesMetricsArchiveAdapter,
)


UTC = timezone.utc


def _zip(path: Path, member: str, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return path


def test_metrics_adapter_normalizes_open_interest_and_positioning(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    )
    _zip(
        path,
        "BTCUSDT-metrics-2026-01-01.csv",
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2026-01-01 00:05:00,BTCUSDT,100,100000,0.8,1.1,0.9,1.2\n",
    )
    record = discover_archives(tmp_path)[0]
    frame = FuturesMetricsArchiveAdapter().read(record)
    assert frame.loc[0, "available_at"] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert frame.loc[0, "open_interest"] == 100
    assert frame.loc[0, "taker_long_short_volume_ratio"] == 1.2


def test_metrics_adapter_preserves_sparse_optional_ratios_as_unknown(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = (
        raw
        / "futures/usdm/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    )
    _zip(
        path,
        "BTCUSDT-metrics-2026-01-01.csv",
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2026-01-01 00:05:00,BTCUSDT,100,100000,,1.1,,1.2\n"
        "2026-01-01 00:10:00,BTCUSDT,101,101000,0.8,,0.9,1.1\n",
    )
    store = MarketDataStore(raw, tmp_path / "cache")
    assert store.refresh_catalog() == 1
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_interval="15m",
        datasets=(DatasetKind.FUTURES_METRICS,),
    )
    frame = store.load_dataset(request, DatasetKind.FUTURES_METRICS)
    assert pd.isna(frame.loc[0, "top_trader_account_long_short_ratio"])
    assert pd.isna(frame.loc[1, "top_trader_position_long_short_ratio"])

    report = store.data_quality_report(
        request,
        DatasetKind.FUTURES_METRICS,
        required=False,
        frame=frame,
    )
    assert report.status is DataQualityStatus.OK
    assert not any(issue.code == "INVALID_NUMERIC" for issue in report.issues)


def test_funding_adapter_accepts_binance_vision_schema(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-01.zip"
    )
    _zip(
        path,
        "BTCUSDT-fundingRate-2026-01.csv",
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1767225600000,8,0.0001\n",
    )
    records = discover_archives(tmp_path)
    assert records[0].dataset == DatasetKind.FUNDING_RATE
    frame = FundingRateArchiveAdapter().read(records[0])
    assert frame.loc[0, "event_time"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert frame.loc[0, "available_at"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert frame.loc[0, "funding_rate"] == 0.0001
    assert frame.loc[0, "funding_interval_hours"] == 8


def test_store_loads_metrics_without_filename_or_interval(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = (
        raw
        / "futures/usdm/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    )
    _zip(
        path,
        "BTCUSDT-metrics-2026-01-01.csv",
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2026-01-01 00:05:00,BTCUSDT,100,100000,0.8,1.1,0.9,1.2\n",
    )
    store = MarketDataStore(raw, tmp_path / "cache")
    assert store.refresh_catalog() == 1
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_interval="15m",
        datasets=(DatasetKind.FUTURES_METRICS,),
    )
    frame = store.load_dataset(request, DatasetKind.FUTURES_METRICS)
    assert len(frame) == 1
    assert frame.loc[0, "open_interest_value"] == 100000
