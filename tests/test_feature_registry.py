from __future__ import annotations

import pytest
from datetime import datetime, timezone

from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features import FeatureDefinition, FeatureRegistry
from crypto_strategy_lab.data.query import DataRequest


class _Provider:
    definition = FeatureDefinition(
        name="open_interest",
        version="1",
        required_datasets=(DatasetKind.FUTURES_METRICS, DatasetKind.KLINES),
        output_columns=("oi_change_1h",),
        warmup_bars=12,
    )

    def compute(self, request, datasets, parameters):  # pragma: no cover - contract fixture
        raise NotImplementedError


class _CoreProvider:
    definition = FeatureDefinition(
        name="core",
        version="2",
        required_datasets=(DatasetKind.KLINES,),
        output_columns=("atr",),
    )


class _ChildProvider:
    definition = FeatureDefinition(
        name="child",
        version="1",
        required_datasets=(DatasetKind.FUNDING_RATE,),
        required_features=("core",),
        output_columns=("funding_atr",),
    )


def test_feature_registry_resolves_dependencies_and_versions() -> None:
    registry = FeatureRegistry()
    registry.register(_Provider())
    assert registry.names() == ("open_interest",)
    assert registry.required_datasets(["open_interest"]) == (
        DatasetKind.KLINES,
        DatasetKind.FUTURES_METRICS,
    )
    first_hash = registry.definition_hash(["open_interest"])
    assert len(first_hash) == 64


def test_feature_registry_orders_feature_dependencies_before_dependants() -> None:
    registry = FeatureRegistry()
    registry.register(_ChildProvider())
    registry.register(_CoreProvider())
    assert registry.dependency_order(["child"]) == ("core", "child")
    assert registry.required_datasets(["child"]) == (
        DatasetKind.FUNDING_RATE,
        DatasetKind.KLINES,
    )
    child_hash = registry.definition_hash(["child"])
    core_hash = registry.definition_hash(["core"])
    assert child_hash != core_hash


def test_feature_registry_rejects_dependency_cycles() -> None:
    class _A:
        definition = FeatureDefinition(
            name="a", version="1", required_datasets=(DatasetKind.KLINES,),
            required_features=("b",), output_columns=("a",),
        )

    class _B:
        definition = FeatureDefinition(
            name="b", version="1", required_datasets=(DatasetKind.KLINES,),
            required_features=("a",), output_columns=("b",),
        )

    registry = FeatureRegistry()
    registry.register(_A())
    registry.register(_B())
    with pytest.raises(ValueError, match="cycle"):
        registry.dependency_order(["a"])


def test_feature_identity_excludes_intrabar_and_propagates_dependencies():
    registry = FeatureRegistry(); registry.register(_CoreProvider()); registry.register(_ChildProvider())
    base = dict(symbol="BTCUSDT", start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 2, 1, tzinfo=timezone.utc), strategy_interval="4h")
    one = DataRequest(**base, intrabar_interval="1m")
    five = DataRequest(**base, intrabar_interval="5m", datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE))
    core = registry.resolve(["core"])[0]
    sources = {DatasetKind.KLINES: "canonical-kline"}
    assert registry.identity(core, one, sources, {}) == registry.identity(core, five, sources, {})
    child = registry.resolve(["child"])[1]
    child_sources = {DatasetKind.FUNDING_RATE: "canonical-funding"}
    assert registry.identity(child, one, child_sources, {"core": "v2"}) != registry.identity(
        child, one, child_sources, {"core": "v3"})
