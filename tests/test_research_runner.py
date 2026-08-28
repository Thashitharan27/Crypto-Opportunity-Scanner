from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from crypto_strategy_lab.data import DataRequest
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.prepared_cache import prepared_policy_inputs
from crypto_strategy_lab.research_adapters import prepared_policy_config
from crypto_strategy_lab.research_runner import ResearchRunner
import crypto_strategy_lab.research_runner as runner_module


class _Prepared:
    def __len__(self):
        return 2


class _Cache:
    def __init__(self):
        self.calls = 0

    def get_or_build(self, key, builder, *, provenance):
        self.calls += 1
        assert key == "prepared-key"
        assert provenance == {"source": "test"}
        return builder(), False


class _Strategy:
    def __init__(self):
        self.bound = None

    def bind(self, prepared, config):
        self.bound = (prepared, config)
        return "bound-policy"


class _Simulator:
    def __init__(self):
        self.call = None
        self.last_engine_init_seconds = 0.25
        self.last_simulation_seconds = 0.5

    def run(
        self,
        prepared,
        intrabar,
        strategy,
        execution_config,
        *,
        data_config,
        feature_config,
    ):
        self.call = (
            prepared,
            intrabar,
            strategy,
            execution_config,
            data_config,
            feature_config,
        )
        return pd.DataFrame({"result": [1]})


class _Reporter:
    def __init__(self):
        self.calls = []

    def report(self, result, context):
        self.calls.append((result, context))


class _Store:
    canonical_cache_events = {"hit": 10, "miss": 0}


def _frame(name: str) -> pd.DataFrame:
    frame = pd.DataFrame({"x": [1, 2]})
    frame.attrs.update(feature_cache_hit=True, feature_cache_key=f"{name}-key")
    return frame


def _request() -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        strategy_interval="15m",
        intrabar_interval="1m",
    )


def test_runner_uses_injected_composition_and_single_feature_config(monkeypatch):
    config = ResearchRunConfig()
    request = _request()
    registry = object()
    cache = _Cache()
    strategy = _Strategy()
    simulator = _Simulator()
    reporter = _Reporter()
    prepared = _Prepared()
    bundle = type(
        "Bundle",
        (),
        {
            "request": request,
            "strategy": pd.DataFrame(index=range(2)),
            "intrabar": pd.DataFrame(index=range(3)),
            "technical_features": _frame("directional"),
            "context_features": _frame("context"),
            "support_resistance_features": None,
            "state_transition_daily_features": _frame("daily"),
            "research_features": {},
        },
    )()
    observed = {}

    def fake_load(store, incoming_request, **kwargs):
        observed.update(store=store, request=incoming_request, kwargs=kwargs)
        return bundle

    monkeypatch.setattr(runner_module, "load_backtest_bundle", fake_load)
    monkeypatch.setattr(
        runner_module,
        "bundle_prepared_identity",
        lambda supplied_cache, supplied_bundle, policy, **kwargs: (
            "prepared-key",
            {"source": "test"},
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "from_data_lake_bundle",
        lambda supplied_bundle, policy: (prepared, None),
    )
    monkeypatch.setattr(runner_module, "intrabar_from_data_lake_bundle", lambda bundle: None)

    runner = ResearchRunner(_Store(), registry, cache, strategy, simulator, (reporter,))
    result = runner.run(request, config, refresh_catalog=False)

    assert observed["store"] is runner.data_store
    assert observed["request"] is request
    assert observed["kwargs"]["feature_registry"] is registry
    assert observed["kwargs"]["feature_config"] is config.features
    assert "atr_period" not in observed["kwargs"]
    assert cache.calls == 1
    assert strategy.bound == (prepared, config.strategy)
    assert simulator.call[2] == "bound-policy"
    assert simulator.call[3] is config.execution
    assert simulator.call[4] is config.data
    assert simulator.call[5] is config.features
    assert len(reporter.calls) == 1
    assert result.feature_cache_metadata["state_transition_daily"]["cache_hit"] is True
    assert result.stage_timings["engine_init"] == 0.25
    assert result.stage_timings["simulation"] == 0.5


def test_reporting_and_execution_do_not_enter_prepared_policy_identity():
    base = ResearchRunConfig()
    execution_changed = replace(
        base,
        execution=replace(base.execution, maker_fee=0.9, slippage=0.1),
    )
    reporting_changed = replace(
        base,
        reporting=replace(base.reporting, output_dir="elsewhere", create_standard_charts=False),
    )

    class Registry:
        def definition_hash(self, names):
            assert names == ["policy_market_context"]
            return "policy-definition"

    registry = Registry()
    base_inputs = prepared_policy_inputs(
        prepared_policy_config(base), feature_registry=registry
    )
    assert prepared_policy_inputs(
        prepared_policy_config(execution_changed), feature_registry=registry
    ) == base_inputs
    assert prepared_policy_inputs(
        prepared_policy_config(reporting_changed), feature_registry=registry
    ) == base_inputs


def test_runner_and_native_adapters_add_no_inheritance_chain():
    from crypto_strategy_lab.research_adapters import NativeSimulator, NativeStrategyPolicy

    assert ResearchRunner.__bases__ == (object,)
    assert NativeSimulator.__bases__ == (object,)
    assert NativeStrategyPolicy.__bases__ == (object,)
