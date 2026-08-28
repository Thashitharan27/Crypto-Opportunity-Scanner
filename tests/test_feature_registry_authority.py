from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features import (
    FeatureDefinition, FeatureRegistry, OutputField, ParameterDefinition,
    production_feature_registry,
)
from crypto_strategy_lab.features.market_regime import (
    POLICY_MARKET_FEATURE_NAME,
    prepare_policy_market_features,
)
from crypto_strategy_lab.features.technical import CORE_DIRECTIONAL_FEATURE_NAME


REQUEST = DataRequest(
    symbol="BTCUSDT", start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    end=datetime(2026, 1, 2, tzinfo=timezone.utc), strategy_interval="4h",
)


class Provider:
    def __init__(
        self,
        name,
        *,
        version="1",
        dependencies=(),
        warmup=0,
        parameters=None,
        optional_datasets=(),
    ):
        self.calls = 0
        self.definition = FeatureDefinition(
            name=name, version=version, required_datasets=(DatasetKind.KLINES,),
            required_features=dependencies, optional_datasets=optional_datasets,
            warmup_bars=warmup, parameters=parameters or {},
            output_schema={"value": OutputField("numeric", False)},
        )

    def compute(self, request, datasets, parameters, feature_frames=None):
        self.calls += 1
        return pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
            "available_at": pd.to_datetime(["2026-01-01 04:00"], utc=True),
            "value": [float(parameters.get("period", 1))],
        })


class InvalidFrameCache:
    def __init__(self) -> None:
        self.stored = None

    def load(self, definition, request, identity):
        return pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
            "available_at": pd.to_datetime(["2026-01-01 04:00"], utc=True),
        })

    def store(self, definition, request, identity, frame):
        self.stored = frame.copy()


def test_registration_rejects_duplicates_and_missing_dependencies() -> None:
    registry = FeatureRegistry()
    registry.register(Provider("a"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Provider("a"))
    registry = FeatureRegistry()
    registry.register(Provider("root", dependencies=("missing",)))
    with pytest.raises(KeyError, match="Unknown feature"):
        registry.dependency_order(["root"])


def test_parameter_contract_defaults_normalizes_and_rejects_unknown() -> None:
    provider = Provider("a", parameters={"period": ParameterDefinition(int, 14)})
    definition = provider.definition
    assert definition.normalize_parameters() == {"period": 14}
    assert definition.normalize_parameters({"period": "20"}) == {"period": 20}
    with pytest.raises(ValueError, match="Unknown parameters"):
        definition.normalize_parameters({"typo": 1})


def test_boolean_parameter_strings_are_canonical_not_python_truthiness() -> None:
    definition = Provider(
        "a", parameters={"enabled": ParameterDefinition(bool, True)}
    ).definition
    assert definition.normalize_parameters({"enabled": "false"}) == {"enabled": False}
    assert definition.normalize_parameters({"enabled": "TRUE"}) == {"enabled": True}
    with pytest.raises(ValueError, match="invalid parameter value"):
        definition.normalize_parameters({"enabled": "not-a-bool"})


def test_required_parameter_and_output_schema_are_enforced() -> None:
    definition = Provider(
        "a", parameters={"period": ParameterDefinition(int)}
    ).definition
    with pytest.raises(ValueError, match="Missing required"):
        definition.normalize_parameters()
    with pytest.raises(ValueError, match="missing columns"):
        definition.validate_output(pd.DataFrame({"value": [1.0]}))
    bad = pd.DataFrame({"timestamp": ["not-datetime"], "available_at": ["no"], "value": ["x"]})
    with pytest.raises(ValueError, match="must be"):
        definition.validate_output(bad)


def test_invalid_cached_schema_is_recomputed_and_replaced() -> None:
    registry = FeatureRegistry()
    provider = Provider("a")
    registry.register(provider)
    cache = InvalidFrameCache()
    frames = registry.execute(
        ["a"],
        REQUEST,
        {DatasetKind.KLINES: pd.DataFrame()},
        cache=cache,
        source_identities={DatasetKind.KLINES: "source"},
    )
    assert provider.calls == 1
    assert list(frames["a"]["value"]) == [1.0]
    assert cache.stored is not None
    assert list(cache.stored["value"]) == [1.0]


def test_graph_is_deterministic_deduplicated_and_propagates_warmup() -> None:
    registry = FeatureRegistry()
    leaf, left, right, root = (
        Provider("leaf", warmup=2), Provider("left", dependencies=("leaf",), warmup=3),
        Provider("right", dependencies=("leaf",), warmup=4),
        Provider("root", dependencies=("right", "left"), warmup=5),
    )
    for provider in (root, right, leaf, left):
        registry.register(provider)
    assert registry.dependency_order(["root"]) == ("leaf", "left", "right", "root")
    assert registry.effective_warmup(["root"]) == 11
    registry.execute(
        ["root"], REQUEST, {DatasetKind.KLINES: pd.DataFrame()},
        source_identities={DatasetKind.KLINES: "source"},
    )
    assert leaf.calls == left.calls == right.calls == root.calls == 1


def test_identity_propagates_only_related_definition_and_parameters() -> None:
    def identities(child_version="1", period=14, unrelated_version="1"):
        registry = FeatureRegistry()
        child = Provider("child", version=child_version, parameters={"period": ParameterDefinition(int, 14)})
        parent = Provider("parent", dependencies=("child",))
        unrelated = Provider("unrelated", version=unrelated_version)
        for provider in (child, parent, unrelated):
            registry.register(provider)
        resolved = registry.resolve(["parent", "unrelated"], {"child": {"period": period}})
        values = {}
        source = {DatasetKind.KLINES: "source"}
        for item in resolved:
            values[item.definition.name] = registry.identity(item, REQUEST, source, values)
        return values
    base = identities()
    changed_parameter = identities(period=20)
    assert base["child"] != changed_parameter["child"]
    assert base["parent"] != changed_parameter["parent"]
    assert base["unrelated"] == changed_parameter["unrelated"]
    changed_version = identities(child_version="2")
    assert base["parent"] != changed_version["parent"]


def test_optional_material_dataset_changes_identity_only_when_present() -> None:
    registry = FeatureRegistry()
    provider = Provider("a", optional_datasets=(DatasetKind.PREMIUM_INDEX_KLINES,))
    registry.register(provider)
    resolved = registry.resolve(["a"])[0]
    base_sources = {DatasetKind.KLINES: "kline-source"}
    without_optional = registry.identity(resolved, REQUEST, base_sources, {})
    with_optional = registry.identity(
        resolved,
        REQUEST,
        {**base_sources, DatasetKind.PREMIUM_INDEX_KLINES: "premium-v1"},
        {},
    )
    changed_optional = registry.identity(
        resolved,
        REQUEST,
        {**base_sources, DatasetKind.PREMIUM_INDEX_KLINES: "premium-v2"},
        {},
    )
    assert without_optional != with_optional
    assert with_optional != changed_optional


def test_policy_market_schema_expands_from_registered_momentum_parameters() -> None:
    definition = production_feature_registry().get(POLICY_MARKET_FEATURE_NAME).definition
    parameters = definition.normalize_parameters({"momentum_lookback_hours": [48, 12, 12]})
    schema = definition.schema_for(parameters)
    assert parameters["momentum_lookback_hours"] == (12, 48)
    assert {"momentum_return_12h", "momentum_return_48h"} <= set(schema)


def test_policy_market_compatibility_adapter_executes_through_registry(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    original = FeatureRegistry.execute

    def recording_execute(self, feature_names, *args, **kwargs):
        calls.append(tuple(feature_names))
        return original(self, feature_names, *args, **kwargs)

    monkeypatch.setattr(FeatureRegistry, "execute", recording_execute)
    times = pd.date_range("2026-01-01", periods=16, freq="4h", tz="UTC")
    close = np.linspace(100.0, 110.0, len(times))
    config = SimpleNamespace(
        strategy_timeframe_minutes=240,
        market_symbol="BTCUSDT",
        market_regime_method="ASSET_RETURN",
        bull_regime_lookback_days=1,
        bull_regime_return_threshold=0.01,
        structural_regime_sma_days=2,
        structural_regime_slope_lookback_days=1,
        strategy_profiles={"only": SimpleNamespace(momentum_lookback_hours=24)},
    )
    bull, regime, momentum = prepare_policy_market_features(times, close, config)
    assert calls == [(POLICY_MARKET_FEATURE_NAME,)]
    assert len(bull) == len(regime) == len(times)
    assert set(momentum) == {24}
    assert len(momentum[24]) == len(times)


def test_production_registry_exposes_core_authoritative_metadata() -> None:
    registry = production_feature_registry()
    expected = {"core_directional", "production_market_context", "policy_market_context",
                "support_resistance", "state_transition_daily", "funding_context",
                "basis_context", "futures_positioning", "trade_flow_context"}
    assert expected <= set(registry.names())
    core = registry.get(CORE_DIRECTIONAL_FEATURE_NAME).definition
    assert core.availability_rule == "current_completed_kline_available_at"
    assert set(core.parameters) == {"atr_period", "adx_period", "di_pressure_lookback"}
    daily = registry.get("state_transition_daily").definition
    assert daily.availability_rule == "daily_state_available_from_following_utc_midnight"
    context = registry.get("production_market_context").definition
    assert context.required_features == (CORE_DIRECTIONAL_FEATURE_NAME,)
    assert context.output_schema["mean_reversion_reentry_confirmation"].kind == "string"
    assert context.output_schema["close_location"].kind == "numeric"
    policy = registry.get(POLICY_MARKET_FEATURE_NAME).definition
    assert set(policy.parameters) == {
        "market_regime_method", "bull_regime_lookback_days",
        "bull_regime_return_threshold", "structural_regime_sma_days",
        "structural_regime_slope_lookback_days", "momentum_lookback_hours",
    }
    assert "structural_daily_state_available_following_utc_midnight" in policy.availability_rule
