from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from crypto_strategy_lab.data.binance.selective_acquisition import (
    DATA_HUB_DATASET_KEYS, AcquisitionState, BackendAcquisitionResult,
    data_hub_dataset_key,
)
from crypto_strategy_lab.data.quality import (
    DataQualityIssue, DataQualityStatus, DatasetQualityReport,
)
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import ArchiveRecord, DatasetKind, MarketKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.data.store import MarketDataStore
from crypto_strategy_lab.data_lake_config import FeatureConfig
from crypto_strategy_lab.rich_data_acquisition import (
    FeatureReadiness, RequirementRequiredness, RichDataAcquisitionConfig,
    SelectiveRichDataAcquirer, resolve_rich_data_requirements,
)

UTC = timezone.utc
START = datetime(2024, 2, 10, tzinfo=UTC)
END = datetime(2024, 2, 12, tzinfo=UTC)


def candidate(symbol="BTCUSDT", rank=1, interval="4h"):
    request = DataRequest(symbol, START, END, interval)
    return SimpleNamespace(symbol=symbol, final_rank=rank,
                           strategy_data_request=request,
                           strategy_source_identity=f"strategy:{symbol}")


def candidate_set(*items, exclusions=()):
    return SimpleNamespace(candidates=tuple(items), exclusions=tuple(exclusions))


def plan(features=(), params=None, interval="4h", intrabar=None):
    return resolve_rich_data_requirements(
        candidate_set(candidate(interval=interval)),
        RichDataAcquisitionConfig(tuple(features), params, intrabar),
    )


def test_empty_and_final_candidates_only():
    assert not plan().requirements
    final = candidate_set(candidate("AUSDT", 2), candidate("BUSDT", 1),
                          exclusions=(SimpleNamespace(symbol="REJECTED"),))
    result = resolve_rich_data_requirements(final, RichDataAcquisitionConfig(("funding_context",)))
    assert {r.symbol for r in result.requirements} == {"AUSDT", "BUSDT"}
    assert [r.symbol for r in result.requirements] == ["BUSDT", "AUSDT"]


def test_feature_dependency_mapping_and_warmups():
    funding = plan(("funding_context",), {"funding_context": {"funding_zscore_window_days": 3}}).requirements
    assert [(r.dataset, r.start) for r in funding] == [(DatasetKind.FUNDING_RATE, START.replace(day=7))]
    basis = plan(("basis_context",)).requirements
    assert {(r.dataset, r.interval, r.requiredness) for r in basis} == {
        (DatasetKind.MARK_PRICE_KLINES, "4h", RequirementRequiredness.REQUIRED_FOR_FEATURE),
        (DatasetKind.INDEX_PRICE_KLINES, "4h", RequirementRequiredness.REQUIRED_FOR_FEATURE),
        (DatasetKind.PREMIUM_INDEX_KLINES, "4h", RequirementRequiredness.OPTIONAL_FOR_FEATURE),
    }
    positioning = plan(("futures_positioning",)).requirements
    assert {(r.dataset, r.interval) for r in positioning} == {(DatasetKind.FUTURES_METRICS, None), (DatasetKind.KLINES, "1h")}
    assert [r.dataset for r in plan(("futures_positioning",), interval="1h").requirements] == [DatasetKind.FUTURES_METRICS]


def test_parameterized_sources_intrabar_and_no_parallel_dataset():
    taker = plan(("taker_flow_context",)).requirements
    assert [(r.dataset, r.interval) for r in taker] == [(DatasetKind.KLINES, "5m")]
    assert not plan(("taker_flow_context",), interval="5m").requirements
    for source, expected in (("AGG_TRADES", DatasetKind.AGG_TRADES), ("TRADES", DatasetKind.TRADES)):
        reqs = plan(("trade_flow_context",), {"trade_flow_context": {"trade_flow_source": source}}).requirements
        assert [r.dataset for r in reqs] == [expected]
    assert [(r.dataset, r.interval) for r in plan(intrabar="60m").requirements] == [(DatasetKind.KLINES, "1h")]


def test_coalescing_preserves_features_and_reasons():
    combined = plan(("futures_positioning", "taker_flow_context"),
                    {"taker_flow_context": {"taker_flow_interval": "1h"}})
    klines = [r for r in combined.requirements if r.dataset is DatasetKind.KLINES]
    assert len(klines) == 1
    assert klines[0].feature_names == ("futures_positioning", "taker_flow_context")
    assert len(klines[0].reasons) == 2


def test_coalesced_source_preserves_each_features_requiredness():
    config = RichDataAcquisitionConfig(
        ("futures_positioning", "taker_flow_context"),
        {"taker_flow_context": {"taker_flow_interval": "1h"}},
    )
    store = Store({(DatasetKind.FUTURES_METRICS, None): DataQualityStatus.OK})
    result = SelectiveRichDataAcquirer(
        store, Backend(AcquisitionState.MISSING), config
    ).acquire(candidate_set(candidate()))
    assert {item.feature_name: item.readiness for item in result.symbols[0].features} == {
        "futures_positioning": FeatureReadiness.DEGRADED,
        "taker_flow_context": FeatureReadiness.UNAVAILABLE,
    }


def report(req, status, *, corrupt=False):
    issues = ()
    if status is not DataQualityStatus.OK:
        code = "SCHEMA_INVALID" if corrupt else "DATASET_MISSING"
        issues = (DataQualityIssue(code, DataQualityStatus.ERROR, code),)
    return DatasetQualityReport(req.dataset.value, req.symbol, req.interval,
        req.requiredness is RequirementRequiredness.REQUIRED_FOR_FEATURE,
        req.start.isoformat(), req.end.isoformat(), None, None, None, None, 0,
        None, status, issues)


class Store:
    def __init__(self, states, post_states=None):
        self.states, self.post_states = states, post_states or states
        self.calls, self.refreshes = [], 0
    def refresh_catalog(self): self.refreshes += 1
    def data_quality_report(self, request, dataset, *, interval=None, required=True):
        key = (dataset, interval)
        state = self.states.get(key, DataQualityStatus.MISSING) if self.refreshes <= 1 else self.post_states.get(key, DataQualityStatus.MISSING)
        fake = SimpleNamespace(dataset=dataset, symbol=request.symbol, interval=interval,
            requiredness=RequirementRequiredness.REQUIRED_FOR_FEATURE if required else RequirementRequiredness.OPTIONAL_FOR_FEATURE,
            start=request.start, end=request.end)
        return report(fake, state[0], corrupt=state[1]) if isinstance(state, tuple) else report(fake, state)
    def source_signature(self, request, dataset, *, interval=None):
        self.calls.append((request.symbol, dataset, interval))
        return SourceSignature(dataset, f"{request.symbol}-{dataset.value}", 1)
    event_archive_quality_report = data_quality_report


class Backend:
    def __init__(self, outcome=AcquisitionState.ACQUIRED): self.requests, self.outcome = [], outcome
    def acquire_archive(self, request, *, cancelled=None):
        self.requests.append(request)
        return BackendAcquisitionResult(self.outcome)


def acquire(features, states, post=None):
    store, backend = Store(states, post), Backend()
    result = SelectiveRichDataAcquirer(store, backend, RichDataAcquisitionConfig(tuple(features), max_workers=2)).acquire(candidate_set(candidate()))
    return result, store, backend


def test_reuse_missing_range_post_validation_and_source_identity():
    result, store, backend = acquire(("funding_context",), {(DatasetKind.FUNDING_RATE, None): DataQualityStatus.OK})
    item = result.symbols[0].datasets[0]
    assert item.state is AcquisitionState.REUSED and not backend.requests
    assert item.source_identity == item.source_signature.cache_identity()
    result, _, backend = acquire(("funding_context",), {}, {})
    item = result.symbols[0].datasets[0]
    assert backend.requests[0].missing_ranges[0].start == item.requirement.start
    assert item.state is AcquisitionState.MISSING


def test_data_hub_missing_is_revalidated_after_successful_daily_fallback():
    store = Store({}, {(DatasetKind.FUNDING_RATE, None): DataQualityStatus.OK})
    result = SelectiveRichDataAcquirer(
        store, Backend(AcquisitionState.MISSING),
        RichDataAcquisitionConfig(("funding_context",)),
    ).acquire(candidate_set(candidate()))
    item = result.symbols[0].datasets[0]
    assert item.state is AcquisitionState.ACQUIRED
    assert item.source_signature is not None


def test_post_download_validation_exception_is_isolated_as_quality_failed():
    class RaisingPostStore(Store):
        def data_quality_report(self, *args, **kwargs):
            if self.refreshes > 1:
                raise ValueError("invalid downloaded archive")
            return super().data_quality_report(*args, **kwargs)
    result = SelectiveRichDataAcquirer(
        RaisingPostStore({}), Backend(),
        RichDataAcquisitionConfig(("funding_context",)),
    ).acquire(candidate_set(candidate()))
    assert result.symbols[0].datasets[0].state is AcquisitionState.QUALITY_FAILED


def test_corruption_is_not_downloaded_and_required_candidate_remains():
    result, _, backend = acquire(("funding_context",), {(DatasetKind.FUNDING_RATE, None): (DataQualityStatus.ERROR, True)})
    assert result.symbols[0].datasets[0].state is AcquisitionState.QUALITY_FAILED
    assert result.symbols[0].features[0].readiness is FeatureReadiness.UNAVAILABLE
    assert result.symbols[0].symbol == "BTCUSDT" and not backend.requests


@pytest.mark.parametrize("ticker,depth,expected", [
    (True, True, FeatureReadiness.READY), (True, False, FeatureReadiness.DEGRADED),
    (False, True, FeatureReadiness.DEGRADED), (False, False, FeatureReadiness.UNAVAILABLE),
])
def test_order_book_any_of_readiness(ticker, depth, expected):
    states = {}
    if ticker: states[(DatasetKind.BOOK_TICKER, None)] = DataQualityStatus.OK
    if depth: states[(DatasetKind.BOOK_DEPTH, None)] = DataQualityStatus.OK
    result, _, _ = acquire(("order_book_context",), states)
    assert result.symbols[0].features[0].readiness is expected


def test_basis_optional_degradation_is_visible():
    states = {(DatasetKind.MARK_PRICE_KLINES, "4h"): DataQualityStatus.OK,
              (DatasetKind.INDEX_PRICE_KLINES, "4h"): DataQualityStatus.OK}
    result, _, _ = acquire(("basis_context",), states)
    readiness = result.symbols[0].features[0]
    assert readiness.readiness is FeatureReadiness.DEGRADED
    assert any(r.requirement.dataset is DatasetKind.PREMIUM_INDEX_KLINES and r.state is AcquisitionState.MISSING for r in readiness.datasets)


def test_data_hub_mapping_is_total_and_exact():
    expected = {DatasetKind.KLINES: "klines", DatasetKind.FUTURES_METRICS: "metrics",
        DatasetKind.FUNDING_RATE: "fundingRate", DatasetKind.MARK_PRICE_KLINES: "markPriceKlines",
        DatasetKind.INDEX_PRICE_KLINES: "indexPriceKlines", DatasetKind.PREMIUM_INDEX_KLINES: "premiumIndexKlines",
        DatasetKind.AGG_TRADES: "aggTrades", DatasetKind.TRADES: "trades",
        DatasetKind.BOOK_DEPTH: "bookDepth", DatasetKind.BOOK_TICKER: "bookTicker"}
    assert DATA_HUB_DATASET_KEYS == expected
    assert {kind: data_hub_dataset_key(kind) for kind in DatasetKind} == expected


def test_event_readiness_fake_never_loads_frames():
    result, store, _ = acquire(("trade_flow_context",), {(DatasetKind.AGG_TRADES, None): DataQualityStatus.OK})
    assert result.symbols[0].features[0].readiness is FeatureReadiness.READY
    assert not hasattr(store, "load_dataset")


def test_real_event_readiness_is_partition_bounded_and_never_calls_full_load(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    path = raw / "trade.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "trade_id,price,quantity,quote_quantity,transact_time,is_buyer_maker\n"
        "1,100,1,100,1707523200000,false\n",
        encoding="utf-8",
    )
    record = ArchiveRecord(
        raw, path, MarketKind.FUTURES_UM, DatasetKind.TRADES, "BTCUSDT", None,
        "daily", START, END, path.stat().st_size, path.stat().st_mtime_ns,
        "trade-source",
    )
    store = MarketDataStore(raw, tmp_path / "cache")
    store.catalog.sync_root(raw, [record])
    # Keep the contract test independent of DuckDB's optional timezone module;
    # the real store, adapter, validator, and source identity remain exercised.
    monkeypatch.setattr(store.catalog, "records_for", lambda *args, **kwargs: [record])
    monkeypatch.setattr(store, "refresh_catalog", lambda: 1)
    monkeypatch.setattr(
        store, "load_dataset",
        lambda *args, **kwargs: pytest.fail("full-range load_dataset invoked"),
    )
    result = SelectiveRichDataAcquirer(
        store, Backend(),
        RichDataAcquisitionConfig(
            ("trade_flow_context",),
            {"trade_flow_context": {"trade_flow_source": "TRADES"}},
        ),
    ).acquire(candidate_set(candidate()))
    assert result.symbols[0].datasets[0].state is AcquisitionState.REUSED


def test_complete_feature_config_parameters_are_filtered_to_enabled_features():
    parameters = FeatureConfig().registry_parameters()
    result = plan(("funding_context",), parameters)
    assert [item.dataset for item in result.requirements] == [DatasetKind.FUNDING_RATE]
    with pytest.raises(ValueError, match="unknown features"):
        plan(("funding_context",), {**parameters, "not_registered": {}})


def test_cancellation_marks_unstarted_requirements_explicitly():
    store, backend = Store({}), Backend()
    result = SelectiveRichDataAcquirer(
        store, backend, RichDataAcquisitionConfig(("basis_context",), max_workers=2)
    ).acquire(candidate_set(candidate()), cancelled=lambda: True)
    assert result.symbols[0].datasets
    assert {item.state for item in result.symbols[0].datasets} == {AcquisitionState.CANCELLED}
    assert not backend.requests


def test_cancellation_does_not_rewrite_completed_backend_failure():
    cancellation = [False]

    class CompletingBackend(Backend):
        def acquire_archive(self, request, *, cancelled=None):
            cancellation[0] = True
            return BackendAcquisitionResult(AcquisitionState.DOWNLOAD_FAILED)

    result = SelectiveRichDataAcquirer(
        Store({}), CompletingBackend(),
        RichDataAcquisitionConfig(("funding_context",)),
    ).acquire(candidate_set(candidate()), cancelled=lambda: cancellation[0])
    assert result.symbols[0].datasets[0].state is AcquisitionState.DOWNLOAD_FAILED
