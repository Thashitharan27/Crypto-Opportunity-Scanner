from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd

from crypto_strategy_lab.data import DataRequest, DatasetKind, MarketDataStore
from crypto_strategy_lab.data.cache import CANONICAL_CACHE_FORMAT_VERSION
from crypto_strategy_lab.features import FeatureDefinition, FeatureRegistry, OutputField
from crypto_strategy_lab.prepared_backtest import PreparedBacktestFrame
from crypto_strategy_lab.prepared_cache import (
    PreparedRunCache,
    prepared_policy_inputs,
)


UTC = timezone.utc


def _make_archive(root: Path) -> None:
    directory = root / "data" / "futures" / "um" / "daily" / "klines" / "BTCUSDT" / "1m"
    directory.mkdir(parents=True)
    path = directory / "BTCUSDT-1m-2026-01-01.zip"
    rows = [
        "1767225600000,100,102,99,101,10,1767225659999,1000,4,6,600,0",
        "1767225660000,101,103,100,102,12,1767225719999,1200,5,7,700,0",
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-2026-01-01.csv", "\n".join(rows) + "\n")


def _request(*, intrabar_interval: str | None = None) -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        strategy_interval="1m",
        intrabar_interval=intrabar_interval,
    )


def test_l1_reuses_same_contract_and_normalizer_version_invalidates(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _make_archive(raw_root)
    store = MarketDataStore(raw_root, tmp_path / "cache")
    store.refresh_catalog()
    request = _request()

    first = store.load_klines(request)
    first_identity = first.attrs["canonical_source_identity"]
    assert store.canonical_cache_events == {"hit": 0, "miss": 1}

    second = store.load_klines(request)
    assert second.attrs["canonical_source_identity"] == first_identity
    assert store.canonical_cache_events == {"hit": 1, "miss": 1}

    store._adapters[DatasetKind.KLINES].normalizer_version = 2
    third = store.load_klines(request)
    assert third.attrs["canonical_source_identity"] != first_identity
    assert store.canonical_cache_events == {"hit": 1, "miss": 2}
    assert len(list((tmp_path / "cache" / "market").rglob("*.parquet"))) == 2


def test_l1_bad_manifest_format_is_a_miss_and_rebuilt(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _make_archive(raw_root)
    store = MarketDataStore(raw_root, tmp_path / "cache")
    store.refresh_catalog()
    request = _request()
    store.load_klines(request)

    manifests = list((tmp_path / "cache" / "market").rglob("*.json"))
    assert len(manifests) == 1
    manifest = manifests[0]
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["cache_format_version"] = 999
    manifest.write_text(json.dumps(metadata), encoding="utf-8")

    before = dict(store.canonical_cache_events)
    store.load_klines(request)
    assert store.canonical_cache_events["miss"] == before["miss"] + 1
    repaired = json.loads(manifest.read_text(encoding="utf-8"))
    assert repaired["cache_format_version"] == CANONICAL_CACHE_FORMAT_VERSION


class _Provider:
    def __init__(self, name, *, version="1", datasets=(DatasetKind.KLINES,), dependencies=()):
        self.definition = FeatureDefinition(
            name=name,
            version=version,
            required_datasets=datasets,
            required_features=dependencies,
            output_schema={"value": OutputField("numeric", False)},
        )

    def compute(self, request, datasets, parameters, feature_frames=None):  # pragma: no cover
        raise NotImplementedError


def _feature_identities(*, core_version="1", funding_version="1", kline="k1", funding="f1"):
    registry = FeatureRegistry()
    for provider in (
        _Provider("core_directional", version=core_version),
        _Provider("production_market_context", dependencies=("core_directional",)),
        _Provider(
            "funding_context",
            version=funding_version,
            datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE),
        ),
    ):
        registry.register(provider)

    source_ids = {
        DatasetKind.KLINES: kline,
        DatasetKind.FUNDING_RATE: funding,
    }
    identities: dict[str, str] = {}
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 2, 1, tzinfo=UTC),
        strategy_interval="4h",
        intrabar_interval="1m",
    )
    for resolved in registry.resolve(["production_market_context", "funding_context"]):
        identities[resolved.definition.name] = registry.identity(
            resolved, request, source_ids, identities
        )
    return identities, request


def _l3_identity(feature_identities: dict[str, str], request: DataRequest, *, canonical="k1") -> str:
    cache = PreparedRunCache(Path("unused"))
    return cache.identity(
        request_identity=request.feature_scope_key(),
        feature_identities=feature_identities,
        canonical_identities={"strategy_ohlcv": canonical},
        prepared_inputs={"policy_market_definition": "policy-v1"},
    )


def test_dependency_aware_invalidation_matrix_is_selective() -> None:
    base_features, request = _feature_identities()
    base_l3 = _l3_identity(base_features, request)

    core_changed, _ = _feature_identities(core_version="2")
    assert core_changed["core_directional"] != base_features["core_directional"]
    assert core_changed["production_market_context"] != base_features["production_market_context"]
    assert core_changed["funding_context"] == base_features["funding_context"]
    assert _l3_identity(core_changed, request) != base_l3

    funding_changed, _ = _feature_identities(funding_version="2")
    assert funding_changed["core_directional"] == base_features["core_directional"]
    assert funding_changed["production_market_context"] == base_features["production_market_context"]
    assert funding_changed["funding_context"] != base_features["funding_context"]
    assert _l3_identity(funding_changed, request) != base_l3

    funding_source_changed, _ = _feature_identities(funding="f2")
    assert funding_source_changed["core_directional"] == base_features["core_directional"]
    assert funding_source_changed["production_market_context"] == base_features["production_market_context"]
    assert funding_source_changed["funding_context"] != base_features["funding_context"]

    strategy_source_changed, _ = _feature_identities(kline="k2")
    assert all(strategy_source_changed[name] != base_features[name] for name in base_features)
    assert _l3_identity(strategy_source_changed, request, canonical="k2") != base_l3

    request_5m = DataRequest(
        symbol=request.symbol,
        start=request.start,
        end=request.end,
        strategy_interval=request.strategy_interval,
        intrabar_interval="5m",
        datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE),
    )
    assert request_5m.feature_scope_key() == request.feature_scope_key()
    assert _l3_identity(base_features, request_5m) == base_l3


def _prepared_kwargs(n: int = 3) -> dict[str, object]:
    timestamp = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC").to_numpy(
        dtype="datetime64[ns]"
    )
    float_names = (
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "plus_di", "minus_di", "bb_width", "bb_width_pct", "session_vwap",
        "close_location", "mean_reversion_mean", "mean_reversion_distance_atr",
        "mean_reversion_distance_atr_previous", "mean_reversion_sigma",
        "mean_reversion_bb_upper", "mean_reversion_bb_lower", "mean_reversion_bb_zscore",
        "mean_reversion_rsi", "di_spread", "di_spread_1", "di_spread_3", "di_spread_5",
        "di_spread_change", "di_ratio", "plus_di_change", "minus_di_change",
        "di_pressure_spread_change", "long_directional_di_change", "long_opposing_di_change",
        "short_directional_di_change", "short_opposing_di_change", "bb_middle", "bb_upper",
        "bb_lower", "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
        "bb_width_change_pct", "mean_reversion_distance_change_atr",
    )
    values: dict[str, object] = {
        name: np.arange(n, dtype=float) + 1 for name in float_names
    }
    values.update(
        timestamp=timestamp,
        strategy_interval=pd.Timedelta(hours=4),
        mean_reversion_long_reentry=np.zeros(n, dtype=bool),
        mean_reversion_short_reentry=np.ones(n, dtype=bool),
        long_di_pressure_state=np.full(n, "EXPANDING", dtype=object),
        short_di_pressure_state=np.full(n, "CONTRACTING", dtype=object),
        mean_reversion_state=np.full(n, "ABOVE_MEAN", dtype=object),
        mean_reversion_motion=np.full(n, "AWAY_FROM_MEAN", dtype=object),
        mean_reversion_strength=np.ones(n, dtype=int),
        mean_reversion_strength_label=np.full(n, "MODERATE", dtype=object),
        mean_reversion_bb_location=np.full(n, "ABOVE_MIDDLE", dtype=object),
        mean_reversion_rsi_state=np.full(n, "NEUTRAL", dtype=object),
        mean_reversion_reentry_confirmation=np.full(n, "SHORT", dtype=object),
        mean_reversion_signal=np.full(n, "STRONG_SHORT", dtype=object),
        mean_reversion_signal_direction=np.full(n, "SHORT", dtype=object),
        mean_reversion_setup_strength=np.full(n, "STRONG", dtype=object),
        bb_reentry=np.full(n, "SHORT", dtype=object),
        mr_signal=np.full(n, "CONFIRMED", dtype=object),
        mr_signal_direction=np.full(n, "SHORT", dtype=object),
        bull_regime_return=np.zeros(n),
        market_regime=np.full(n, "SIDEWAYS", dtype=object),
        momentum_returns_by_hours={24: np.zeros(n)},
        decision_available_at=timestamp + np.timedelta64(4, "h"),
    )
    return values


def test_valid_l3_hit_bypasses_prepared_builder(tmp_path: Path) -> None:
    cache = PreparedRunCache(tmp_path)
    key = cache.identity(
        request_identity="slice",
        feature_identities={"core": "c1"},
        canonical_identities={"strategy": "k1"},
        prepared_inputs={"policy_market_definition": "p1"},
    )
    original = PreparedBacktestFrame(**_prepared_kwargs())
    built, hit = cache.get_or_build(key, lambda: original, provenance={})
    assert built is original
    assert hit is False

    def forbidden():
        raise AssertionError("prepared builder must not execute on an L3 hit")

    loaded, hit = cache.get_or_build(key, forbidden, provenance={})
    assert hit is True
    assert np.array_equal(loaded.close, original.close)
    assert not loaded.close.flags.writeable


def _policy_config(**extra):
    values = dict(
        market_regime_method="ASSET_RETURN",
        bull_regime_lookback_days=90,
        bull_regime_return_threshold=0.20,
        structural_regime_sma_days=200,
        structural_regime_slope_lookback_days=30,
        strategy_profiles={"default": SimpleNamespace(momentum_lookback_hours=24)},
    )
    values.update(extra)
    return SimpleNamespace(**values)


def test_execution_and_reporting_fields_do_not_enter_prepared_identity() -> None:
    first = _policy_config(stop_loss_multiple=2.0, fees_bps=4.0, output_dir="a")
    second = _policy_config(stop_loss_multiple=6.0, fees_bps=20.0, output_dir="b")
    assert prepared_policy_inputs(first) == prepared_policy_inputs(second)

    changed_prepared_input = _policy_config(bull_regime_return_threshold=0.25)
    assert prepared_policy_inputs(first) != prepared_policy_inputs(changed_prepared_input)


def test_prepared_contract_version_changes_l3_identity(tmp_path: Path) -> None:
    kwargs = dict(
        request_identity="slice",
        feature_identities={"core": "c1"},
        canonical_identities={"strategy": "k1"},
        prepared_inputs={},
    )
    assert PreparedRunCache(tmp_path, contract_version=1).identity(**kwargs) != PreparedRunCache(
        tmp_path, contract_version=2
    ).identity(**kwargs)
