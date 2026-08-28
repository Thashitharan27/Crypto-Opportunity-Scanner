from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from crypto_strategy_lab.data import DataRequest
from crypto_strategy_lab.data.cache import CacheLayout
from crypto_strategy_lab.data.backtest_service import _cached_catalog_feature
from crypto_strategy_lab.data.schemas import ArchiveRecord, DatasetKind, MarketKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.features.cache import FeatureFrameCache
from crypto_strategy_lab.features.futures_positioning import FuturesPositioningFeatureProvider
from crypto_strategy_lab.features.technical import CoreDirectionalFeatureProvider


UTC = timezone.utc


def source_frame() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC")
    close = 100 + np.linspace(0, 10, len(times)) + np.sin(np.arange(len(times)))
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + pd.Timedelta(hours=1),
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "source_fingerprint": "archive-fingerprint-a",
        }
    )


def request() -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 3, tzinfo=UTC),
        strategy_interval="1h",
    )


def test_feature_cache_roundtrip_and_invalidation(tmp_path: Path) -> None:
    provider = CoreDirectionalFeatureProvider()
    source = source_frame()
    params = {"atr_period": 7, "adx_period": 6, "di_pressure_lookback": 3}
    req = request()
    frame = provider.compute(req, {DatasetKind.KLINES: source}, params)
    cache = FeatureFrameCache(tmp_path)
    key = cache.key(provider.definition, req, params, source)

    assert cache.load(provider.definition, req, key) is None
    cache.store(provider.definition, req, key, frame)
    loaded = cache.load(provider.definition, req, key)
    assert loaded is not None
    assert loaded.attrs["feature_cache_hit"] is True
    assert loaded.attrs["feature_cache_key"] == key
    pdt.assert_frame_equal(loaded, frame, check_dtype=False)

    changed_params = {**params, "adx_period": 7}
    assert cache.key(provider.definition, req, changed_params, source) != key

    changed_source = source.copy()
    changed_source["source_fingerprint"] = "archive-fingerprint-b"
    assert cache.key(provider.definition, req, params, changed_source) != key


def test_feature_cache_key_tracks_additional_dataset_sources(tmp_path: Path) -> None:
    provider = CoreDirectionalFeatureProvider()
    source = source_frame()
    req = request()
    params = {"atr_period": 7, "adx_period": 6, "di_pressure_lookback": 3}
    cache = FeatureFrameCache(tmp_path)

    metrics_a = pd.DataFrame({"source_fingerprint": ["metrics-a"]})
    metrics_b = pd.DataFrame({"source_fingerprint": ["metrics-b"]})
    key_a = cache.key(
        provider.definition,
        req,
        params,
        source,
        additional_sources=(metrics_a,),
    )
    key_b = cache.key(
        provider.definition,
        req,
        params,
        source,
        additional_sources=(metrics_b,),
    )
    assert key_a != key_b


def _archive(*, fingerprint: str = "source-a", size_bytes: int = 100) -> ArchiveRecord:
    return ArchiveRecord(
        raw_root=Path("/lake"),
        path=Path("/lake/monthly/metrics.zip"),
        market=MarketKind.FUTURES_UM,
        dataset=DatasetKind.FUTURES_METRICS,
        symbol="BTCUSDT",
        interval=None,
        frequency="monthly",
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        size_bytes=size_bytes,
        mtime_ns=123,
        fingerprint=fingerprint,
    )


def test_catalog_source_signature_is_stable_and_tracks_relevant_metadata() -> None:
    first = SourceSignature.from_records(DatasetKind.FUTURES_METRICS, [_archive()])
    identical = SourceSignature.from_records(DatasetKind.FUTURES_METRICS, [_archive()])
    changed = SourceSignature.from_records(
        DatasetKind.FUTURES_METRICS, [_archive(size_bytes=101)]
    )

    assert first == identical
    assert first != changed


def test_metadata_cache_key_tracks_version_parameters_and_dependencies(tmp_path: Path) -> None:
    cache = FeatureFrameCache(tmp_path)
    provider = FuturesPositioningFeatureProvider()
    signature = SourceSignature.from_records(DatasetKind.FUTURES_METRICS, [_archive()])
    baseline = cache.key_from_signatures(
        provider.definition, request(), {"window": 1}, [signature], dependency_keys=["dep-a"]
    )

    assert cache.key_from_signatures(
        replace(provider.definition, version="2"), request(), {"window": 1}, [signature],
        dependency_keys=["dep-a"],
    ) != baseline
    assert cache.key_from_signatures(
        provider.definition, request(), {"window": 2}, [signature], dependency_keys=["dep-a"]
    ) != baseline
    assert cache.key_from_signatures(
        provider.definition, request(), {"window": 1}, [signature], dependency_keys=["dep-b"]
    ) != baseline


class _CatalogFeatureStore:
    def __init__(self, root: Path, metrics: pd.DataFrame) -> None:
        self.cache = CacheLayout(root)
        self.cache.ensure()
        self.metrics = metrics
        self.loads = 0

    def source_signature(self, _request, dataset, *, interval=None):
        del interval
        fingerprint = "klines-a" if dataset == DatasetKind.KLINES else "metrics-a"
        return SourceSignature(dataset, fingerprint, 1)

    def load_dataset(self, _request, dataset, *, interval=None):
        del interval
        assert dataset == DatasetKind.FUTURES_METRICS
        self.loads += 1
        return self.metrics.copy()


def test_positioning_cache_hit_does_not_materialize_metrics_but_miss_does(tmp_path: Path) -> None:
    from tests.test_futures_positioning_features import klines, metrics, request as positioning_request

    canonical = klines()
    req = positioning_request(canonical)
    store = _CatalogFeatureStore(tmp_path, metrics())
    provider = FuturesPositioningFeatureProvider()

    missed = _cached_catalog_feature(
        store, req, canonical, DatasetKind.FUTURES_METRICS, provider
    )
    assert missed is not None
    assert missed.attrs["feature_cache_hit"] is False
    assert store.loads == 1

    hit = _cached_catalog_feature(
        store, req, canonical, DatasetKind.FUTURES_METRICS, provider
    )
    assert hit is not None
    assert hit.attrs["feature_cache_hit"] is True
    assert store.loads == 1
    pdt.assert_frame_equal(hit, missed, check_dtype=False)
