from __future__ import annotations

from types import SimpleNamespace

from tools.data_lake_preparation_profile import _instrument_preparation, _summarize_events


def test_summarize_events_groups_canonical_stage_timings() -> None:
    events = [
        {"category": "dataset_load", "name": "klines:4h", "seconds": 1.25},
        {"category": "dataset_load", "name": "klines:1m", "seconds": 2.50},
        {"category": "research_feature_cache", "name": "funding_context", "seconds": 0.15},
    ]

    summary = _summarize_events(events, 5.0)

    assert summary["category_seconds"] == {
        "dataset_load": 3.75,
        "research_feature_cache": 0.15,
    }
    assert summary["category_calls"] == {
        "dataset_load": 2,
        "research_feature_cache": 1,
    }
    assert abs(summary["instrumented_seconds"] - 3.9) < 1e-12
    assert abs(summary["unattributed_seconds"] - 1.1) < 1e-12


def test_summarize_events_never_reports_negative_unattributed_time() -> None:
    events = [
        {"category": "dataset_load", "name": "klines:1m", "seconds": 1.01},
    ]

    summary = _summarize_events(events, 1.0)

    assert summary["instrumented_seconds"] == 1.01
    assert summary["unattributed_seconds"] == 0.0


def test_instrumentation_no_longer_depends_on_deleted_bridge_symbols() -> None:
    store = SimpleNamespace(
        load_dataset=lambda *args, **kwargs: None,
        load_execution_klines=lambda *args, **kwargs: None,
    )
    with _instrument_preparation(store, []):
        assert callable(store.load_dataset)
        assert callable(store.load_execution_klines)
