from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data.backtest_service import _optional_futures_research_features
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.features.registry import FeatureRegistry
from crypto_strategy_lab.features.taker_flow import (
    TAKER_FLOW_CONTEXT_FEATURE_NAME,
    TakerFlowContextFeatureProvider,
    taker_flow_resource,
)
from feature_causality_harness import CausalityCase, assert_future_mutation_invariant


def _strategy(n: int = 8) -> pd.DataFrame:
    starts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "period_start": starts,
            "available_at": starts + pd.Timedelta(hours=4),
            "close": 100.0 + np.arange(n),
        }
    )


def _request(strategy: pd.DataFrame) -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=strategy.period_start.iloc[0].to_pydatetime(),
        end=(strategy.period_start.iloc[-1] + pd.Timedelta(hours=4)).to_pydatetime(),
        strategy_interval="4h",
    )


def _source(start: str, periods: int) -> pd.DataFrame:
    available = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    phase = np.arange(periods)
    buy = np.where(phase % 3 == 0, 7.0, 6.0)
    return pd.DataFrame(
        {
            "available_at": available,
            "volume": 10.0,
            "taker_buy_base_volume": buy,
        }
    )


def test_taker_flow_uses_distinct_native_timeline_and_rejects_invalid_volume():
    strategy = _strategy(2)
    source = _source("2026-01-01 03:05", 12)
    request = _request(strategy)
    provider = TakerFlowContextFeatureProvider()
    out = provider.compute(
        request,
        {DatasetKind.KLINES: strategy, taker_flow_resource("5m"): source},
        {},
    )
    assert len(out) == 2
    assert out.loc[0, "taker_source_available_at"] == source.available_at.iloc[-1]
    assert np.isfinite(out.loc[0, "taker_delta_1h"])

    bad = source.copy()
    bad.loc[0, "taker_buy_base_volume"] = 11
    with pytest.raises(ValueError, match="exceeds volume"):
        provider.compute(
            request,
            {DatasetKind.KLINES: strategy, taker_flow_resource("5m"): bad},
            {},
        )


def test_taker_elapsed_windows_do_not_double_count_left_boundary():
    strategy = _strategy(1)
    source = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                [
                    "2026-01-01T03:40:00Z",
                    "2026-01-01T03:45:00Z",
                    "2026-01-01T03:50:00Z",
                    "2026-01-01T03:55:00Z",
                    "2026-01-01T04:00:00Z",
                ],
                utc=True,
            ),
            "volume": 10.0,
            "taker_buy_base_volume": 6.0,
        }
    )
    out = TakerFlowContextFeatureProvider().compute(
        _request(strategy),
        {DatasetKind.KLINES: strategy, taker_flow_resource("5m"): source},
        {},
    )
    # (03:45, 04:00] contains exactly three completed 5m candles.
    assert out.loc[0, "taker_delta_15m"] == pytest.approx(6.0)
    assert 0.0 <= out.loc[0, "flow_persistence"] <= 1.0


def test_taker_partial_history_keeps_strategy_timeline_with_nan_before_coverage():
    strategy = _strategy(3)
    source = _source("2026-01-01 07:05", 12)
    out = TakerFlowContextFeatureProvider().compute(
        _request(strategy),
        {DatasetKind.KLINES: strategy, taker_flow_resource("5m"): source},
        {},
    )
    assert len(out) == len(strategy)
    assert pd.isna(out.loc[0, "taker_delta"])
    assert out.loc[1:, "taker_delta"].notna().any()


def test_taker_auxiliary_source_identity_invalidates_only_its_feature_key():
    registry = production_feature_registry()
    request = _request(_strategy(2))
    params = {TAKER_FLOW_CONTEXT_FEATURE_NAME: {"taker_flow_interval": "5m"}}
    resolved = registry.resolve([TAKER_FLOW_CONTEXT_FEATURE_NAME], params)[0]
    resource = taker_flow_resource("5m")
    key_a = registry.identity(
        resolved,
        request,
        {DatasetKind.KLINES: "strategy-a", resource: "taker-a"},
        {},
    )
    key_b = registry.identity(
        resolved,
        request,
        {DatasetKind.KLINES: "strategy-a", resource: "taker-b"},
        {},
    )
    assert key_a != key_b


class _Signature:
    def __init__(self, value: str):
        self.value = value

    def cache_identity(self) -> str:
        return self.value


class _FakeStore:
    def __init__(self, root, source: pd.DataFrame):
        self.cache = SimpleNamespace(root=root)
        self.source = source
        self.loads = 0

    def source_signature(self, request, dataset, *, interval=None):
        return _Signature(
            f"{getattr(dataset, 'value', dataset)}:{interval}:{request.start.isoformat()}"
        )

    def load_dataset(self, request, dataset, *, interval=None):
        assert dataset is DatasetKind.KLINES
        assert interval == "5m"
        self.loads += 1
        return self.source.copy()


def test_authoritative_optional_research_path_attaches_taker_flow_and_reuses_l2(tmp_path):
    strategy = _strategy(3)
    request = _request(strategy)
    source = _source("2025-12-31 23:05", 110)
    store = _FakeStore(tmp_path, source)
    kwargs = dict(
        registry=production_feature_registry(),
        usable_datasets=set(),
        feature_parameters={
            TAKER_FLOW_CONTEXT_FEATURE_NAME: {"taker_flow_interval": "5m"}
        },
        positioning_price_usable=False,
        taker_flow_usable=True,
    )
    first = _optional_futures_research_features(
        store, request, strategy, **kwargs
    )
    second = _optional_futures_research_features(
        store, request, strategy, **kwargs
    )
    assert TAKER_FLOW_CONTEXT_FEATURE_NAME in first
    assert len(first[TAKER_FLOW_CONTEXT_FEATURE_NAME]) == len(strategy)
    assert second[TAKER_FLOW_CONTEXT_FEATURE_NAME].attrs["feature_cache_hit"] is True
    assert store.loads == 1


def test_taker_flow_participates_in_generic_future_mutation_causality_family():
    strategy = _strategy(10)
    request = _request(strategy)
    source = _source("2025-12-31 23:05", 520)
    resource = taker_flow_resource("5m")

    def mutate(frame: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
        frame.loc[mask, "taker_buy_base_volume"] = 9.0

    case = CausalityCase(
        feature_name=TAKER_FLOW_CONTEXT_FEATURE_NAME,
        registry_factory=lambda _: production_feature_registry(),
        request=request,
        datasets={DatasetKind.KLINES: strategy, resource: source},
        parameters={
            TAKER_FLOW_CONTEXT_FEATURE_NAME: {"taker_flow_interval": "5m"}
        },
        future_mutators={resource: mutate},
    )
    assert_future_mutation_invariant(case)
