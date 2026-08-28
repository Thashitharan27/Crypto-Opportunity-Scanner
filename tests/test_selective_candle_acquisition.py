from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import random
import threading
import time

import pandas as pd

from crypto_strategy_lab.data.binance.historical_discovery import (
    DiscoveryDecisionTime, HistoricalDiscoveryConfig, HistoricalDiscoveryResult,
    HistoricalSnapshotRow, HistoricalUniverseSnapshot,
)
from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionBackendConfigurationError, AcquisitionState,
    BackendAcquisitionResult, BinanceDataHubBackend, CandleAcquisitionRequest,
    SelectiveCandleAcquirer,
    SelectiveCandleAcquisitionConfig, shortlist_from_historical,
    shortlist_from_live,
)
from crypto_strategy_lab.data.binance.universe import DiscoveryRow
from crypto_strategy_lab.data.quality import validate_dataset
from crypto_strategy_lab.data.quality import MissingCoverageRange
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=6)


def live_rows(count=3):
    now = START
    return [
        DiscoveryRow(f"S{i:03}USDT", True, (), timedelta(days=100), Decimal(1),
                     Decimal(1), Decimal(2), Decimal(1), Decimal(1), Decimal(1),
                     i + 1, now)
        for i in range(count)
    ]


def frame(symbol, missing=(), corrupt=False):
    starts = [value for value in pd.date_range(START, END, freq="1h", inclusive="left")
              if value.to_pydatetime() not in missing]
    result = pd.DataFrame({
        "period_start": starts, "period_end": [x + pd.Timedelta("1h") for x in starts],
        "available_at": [x + pd.Timedelta("1h") for x in starts],
        "open": 2.0, "high": 3.0, "low": 1.0, "close": 2.0,
        "volume": -1.0 if corrupt else 1.0, "symbol": symbol,
        "exchange": "binance", "market": "futures_um", "dataset": "klines",
        "interval": "1h",
    })
    result.attrs["canonical_source_identity"] = f"identity-{symbol}"
    return result


class FakeStore:
    def __init__(self, frames=None):
        self.frames = dict(frames or {})
        self.versions = {symbol: 1 for symbol in self.frames}
        self.refreshes = 0

    def refresh_catalog(self):
        self.refreshes += 1

    def data_quality_report(self, request, dataset, *, interval):
        value = self.frames.get(request.symbol)
        if value is not None:
            value.attrs["canonical_source_identity"] = self.source_signature(
                request, dataset, interval=interval).cache_identity()
        return validate_dataset(value, request, dataset, interval=interval)

    def load_dataset(self, request, dataset, *, interval):
        return self.frames[request.symbol].copy()

    def source_signature(self, request, dataset, *, interval):
        return SourceSignature(dataset, f"{request.symbol}-{self.versions[request.symbol]}", 1)


class FakeBackend:
    def __init__(self, store, *, populate=True, delays=False, stop_event=None):
        self.store, self.populate, self.delays, self.stop_event = store, populate, delays, stop_event
        self.calls = []
        self.active = self.maximum_active = 0
        self.lock = threading.Lock()

    def acquire(self, request, *, cancelled=None):
        with self.lock:
            self.calls.append(request)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        if self.stop_event:
            self.stop_event.set()
        if self.delays:
            time.sleep(random.uniform(.002, .02))
        if self.populate:
            symbol = request.data_request.symbol
            self.store.frames[symbol] = frame(symbol)
            self.store.versions[symbol] = self.store.versions.get(symbol, 0) + 1
        with self.lock:
            self.active -= 1
        return BackendAcquisitionResult(True)


def acquire(rows, store, backend, *, size=25, workers=4, cancelled=None):
    service = SelectiveCandleAcquirer(
        store, backend, SelectiveCandleAcquisitionConfig(size, "60m", workers)
    )
    return service.acquire(rows, START, END, cancelled=cancelled)


def test_300_symbol_universe_acquires_only_top_25_and_ranking_controls_shortlist():
    store = FakeStore()
    backend = FakeBackend(store)
    result = acquire(list(reversed(live_rows(300))), store, backend)
    assert len(backend.calls) == 25
    assert [call.data_request.symbol for call in backend.calls] == [f"S{i:03}USDT" for i in range(25)]
    assert [item.rank for item in result.symbols] == list(range(1, 26))


def test_complete_local_data_is_reused_with_canonical_identity():
    store = FakeStore({"S000USDT": frame("S000USDT")})
    backend = FakeBackend(store)
    item = acquire(live_rows(1), store, backend).symbols[0]
    assert not backend.calls
    assert item.state is AcquisitionState.REUSED
    assert item.row_count == 6 and item.source_signature.cache_identity().startswith("canonical-v2")


def test_only_structured_missing_ranges_are_acquired_and_rerun_reuses():
    missing = (START + timedelta(hours=2), START + timedelta(hours=3))
    store = FakeStore({"S000USDT": frame("S000USDT", missing)})
    backend = FakeBackend(store)
    first = acquire(live_rows(1), store, backend).symbols[0]
    assert first.state is AcquisitionState.ACQUIRED
    assert [(x.start, x.end) for x in backend.calls[0].missing_ranges] == [
        (START + timedelta(hours=2), START + timedelta(hours=4))
    ]
    second = acquire(live_rows(1), store, backend).symbols[0]
    assert second.state is AcquisitionState.REUSED and len(backend.calls) == 1


def test_downloader_success_is_not_success_without_post_validation():
    store = FakeStore()
    item = acquire(live_rows(1), store, FakeBackend(store, populate=False)).symbols[0]
    assert item.state is AcquisitionState.MISSING


def test_corrupt_existing_data_is_quality_failed_without_download():
    store = FakeStore({"S000USDT": frame("S000USDT", corrupt=True)})
    backend = FakeBackend(store)
    item = acquire(live_rows(1), store, backend).symbols[0]
    assert item.state is AcquisitionState.QUALITY_FAILED and not backend.calls


def test_source_identity_changes_when_source_partitions_change():
    store = FakeStore({"S000USDT": frame("S000USDT")})
    backend = FakeBackend(store)
    first = acquire(live_rows(1), store, backend).symbols[0].source_signature
    store.versions["S000USDT"] += 1
    second = acquire(live_rows(1), store, backend).symbols[0].source_signature
    assert first.cache_identity() != second.cache_identity()


def test_symbol_concurrency_is_bounded_and_results_remain_rank_ordered():
    store = FakeStore()
    backend = FakeBackend(store, delays=True)
    result = acquire(live_rows(20), store, backend, size=20, workers=3)
    assert backend.maximum_active <= 3
    assert [item.rank for item in result.symbols] == list(range(1, 21))


def test_cancellation_stops_new_submission_and_records_unattempted_symbols():
    event = threading.Event()
    store = FakeStore()
    backend = FakeBackend(store, stop_event=event)
    result = acquire(live_rows(5), store, backend, size=5, workers=1,
                     cancelled=event.is_set)
    assert len(backend.calls) == 1
    assert [item.state for item in result.symbols] == [AcquisitionState.CANCELLED] * 5
    assert result.symbols[1].detail == "not attempted"


def test_live_and_historical_discovery_adapters_are_compatible():
    live = live_rows(3)
    assert [x.symbol for x in shortlist_from_live(live, 2)] == ["S000USDT", "S001USDT"]
    historical_rows = tuple(
        HistoricalSnapshotRow(row.symbol, START, START + timedelta(days=1), END,
                              Decimal(1), Decimal(2), Decimal(1), Decimal(1), Decimal(1),
                              Decimal(1), Decimal(1), Decimal(1), True, (), row.preliminary_rank, "source")
        for row in live
    )
    decision = DiscoveryDecisionTime(END)
    historical = HistoricalDiscoveryResult(
        decision, HistoricalUniverseSnapshot(decision, historical_rows), historical_rows,
        (), HistoricalDiscoveryConfig(minimum_quote_volume=Decimal(0)),
    )
    assert [x.symbol for x in shortlist_from_historical(historical, 2)] == ["S000USDT", "S001USDT"]
    store = FakeStore()
    assert len(acquire(historical, store, FakeBackend(store), size=2).symbols) == 2


def test_data_hub_bridge_is_explicitly_configured_and_preserves_half_open_dates(monkeypatch, tmp_path):
    calls = []
    module = type("Hub", (), {"download_archive_library": staticmethod(
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"downloaded": 1}
    )})
    monkeypatch.setattr("importlib.import_module", lambda name: module)
    request = DataRequest("BTCUSDT", START, END, "1h")
    outcome = BinanceDataHubBackend(tmp_path, module="configured.hub").acquire(
        CandleAcquisitionRequest(request, (MissingCoverageRange(START, END),))
    )
    assert outcome.succeeded
    args, kwargs = calls[0]
    assert args[:3] == (["BTCUSDT"], ["klines"], ["1h"])
    assert kwargs["start_date"] == START.date()
    assert kwargs["end_date"] == (END - timedelta(microseconds=1)).date()
    assert kwargs["workers"] == 1


def test_unresolved_data_hub_has_clear_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(ImportError()))
    backend = BinanceDataHubBackend(tmp_path, module="missing.hub")
    try:
        backend._downloader()
    except AcquisitionBackendConfigurationError as exc:
        assert "configure" in str(exc)
    else:
        raise AssertionError("missing Data Hub must not silently fall back")
