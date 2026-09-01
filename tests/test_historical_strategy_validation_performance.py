from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

import crypto_strategy_lab.historical_strategy_validation as validation_module
from crypto_strategy_lab.data.quality import DataQualityStatus
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.historical_strategy_validation import (
    EVERY_VIABLE_ENTRY,
    HistoricalStrategyValidator,
    SymbolResearchResult,
)
from crypto_strategy_lab.validation_data_preflight import ValidationDataPreparer


def _candidates():
    return pd.DataFrame({
        "decision_timestamp": pd.to_datetime(["2025-01-01T00:00Z"]),
        "final_rank": [1],
        "symbol": ["SOLUSDT"],
        "scan_run_id": ["scan"],
    })


def _timeout_disabled_config():
    base = ResearchRunConfig()
    execution = replace(base.execution, profiles={
        key: replace(profile, timeout_enabled=False)
        for key, profile in base.execution.profiles.items()
    })
    return replace(base, execution=execution)


def _empty_standard():
    return pd.DataFrame(columns=["entry_time", "pair_net_r", "side", "pair_id"])


def _empty_viable():
    return pd.DataFrame(columns=["entry_time", "pair_net_r", "side", "research_sample_id"])


def test_fast_path_stops_after_resolved_entry_horizon_without_latest_tail(tmp_path):
    calls = []
    preflight_scopes = []

    def execute(symbol, start, end, config):
        calls.append(pd.Timestamp(end))
        viable = _empty_viable()
        return SymbolResearchResult(
            "probe", _empty_standard(), viable, viable.copy(), end,
            tmp_path / "probe", start, end,
        )

    def latest(_symbol):
        raise AssertionError("latest coverage must not be consulted for resolved NO_ENTRY windows")

    def preflight(*args, coverage_scope, **kwargs):
        preflight_scopes.append(coverage_scope)
        return []

    result = HistoricalStrategyValidator(
        execute,
        warmup_bars=lambda _: 0,
        latest_available=latest,
        preflight=preflight,
        defer_outcome_tail_until_needed=True,
        enforce_stable_code_provenance=False,
    ).run(
        _candidates(), _timeout_disabled_config(), config_path=tmp_path / "unused",
        publish=False,
    )

    mandatory_end = pd.Timestamp("2025-01-02T00:00Z")
    assert calls == [mandatory_end]
    assert preflight_scopes == ["MANDATORY_ENTRY"]
    assert set(result.outcomes.result) == {"NO_ENTRY"}
    native = result.manifest["native_research_runs_by_symbol"]["SOLUSDT"]
    assert native["entry_probe_run_id"] == "probe"
    assert native["tail_extension_performed"] is False
    assert native["outcome_tail_status"] == "NOT_REQUIRED_CANDIDATE_WINDOWS_RESOLVED"
    assert result.manifest["resolution_policy"]["defer_outcome_tail_until_needed"] is True
    assert "performance_timings" in result.manifest


def test_fast_path_extends_only_when_candidate_outcome_is_unresolved(tmp_path):
    calls = []
    preflight_scopes = []
    latest = pd.Timestamp("2025-02-01T00:00Z")
    mandatory_end = pd.Timestamp("2025-01-02T00:00Z")

    def execute(symbol, start, end, config):
        end = pd.Timestamp(end)
        calls.append(end)
        if end == mandatory_end:
            censored = pd.DataFrame({
                "entry_time": pd.to_datetime(["2025-01-01T01:00Z"]),
                "pair_net_r": [None],
                "side": ["LONG"],
                "research_sample_id": ["sample-1"],
            })
            return SymbolResearchResult(
                "probe", _empty_standard(), _empty_viable(), censored,
                end.to_pydatetime(), tmp_path / "probe", start, end.to_pydatetime(),
            )
        resolved = pd.DataFrame({
            "entry_time": pd.to_datetime(["2025-01-01T01:00Z"]),
            "pair_net_r": [1.0],
            "side": ["LONG"],
            "research_sample_id": ["sample-1"],
        })
        return SymbolResearchResult(
            "extended", _empty_standard(), resolved, _empty_viable(),
            end.to_pydatetime(), tmp_path / "extended", start, end.to_pydatetime(),
        )

    def preflight(*args, coverage_scope, **kwargs):
        preflight_scopes.append(coverage_scope)
        return []

    result = HistoricalStrategyValidator(
        execute,
        warmup_bars=lambda _: 0,
        latest_available=lambda _symbol: latest,
        preflight=preflight,
        common_available_end=lambda symbol, start, end, config: end,
        defer_outcome_tail_until_needed=True,
        enforce_stable_code_provenance=False,
    ).run(
        _candidates(), _timeout_disabled_config(), config_path=tmp_path / "unused",
        publish=False,
    )

    assert calls == [mandatory_end, latest]
    assert preflight_scopes == ["MANDATORY_ENTRY", "OUTCOME_TAIL"]
    eve = result.outcomes[result.outcomes.population.eq(EVERY_VIABLE_ENTRY)].iloc[0]
    assert eve.result == "WIN" and eve.net_r == 1.0
    native = result.manifest["native_research_runs_by_symbol"]["SOLUSDT"]
    assert native["entry_probe_run_id"] == "probe"
    assert native["tail_extension_run_id"] == "extended"
    assert native["tail_extension_performed"] is True
    assert native["actual_native_run_end"] == latest.isoformat()


def test_validation_preparer_refreshes_catalog_once_until_acquisition():
    class Report:
        status = DataQualityStatus.OK
        def missing_coverage_ranges(self): return ()
        def has_non_missing_errors(self): return False

    class Store:
        def __init__(self):
            self.refreshes = 0
            self.raw_root = "raw"
        def refresh_catalog(self): self.refreshes += 1
        def data_quality_report(self, *args, **kwargs): return Report()
        def source_signature(self, *args, **kwargs):
            return type("Source", (), {"cache_identity": lambda self: "source"})()

    class Backend:
        def acquire_archive(self, *args, **kwargs):
            raise AssertionError("all test data is already reusable")

    base = ResearchRunConfig()
    config = replace(base,
        data=replace(base.data, strategy_timeframe_minutes=1440, use_intrabar_data=False),
        features=replace(base.features, market_regime_method="ASSET_RETURN"),
    )
    store = Store(); preparer = ValidationDataPreparer(store, Backend())
    preparer.prepare("SOLUSDT", datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, tzinfo=timezone.utc), config)
    preparer.prepare("SOLUSDT", datetime(2025, 1, 2, tzinfo=timezone.utc),
        datetime(2025, 1, 3, tzinfo=timezone.utc), config)
    assert store.refreshes == 1


def test_validation_aborts_if_git_head_changes_mid_job(tmp_path, monkeypatch):
    commits = iter(["commit-a", "commit-a", "commit-b"])
    monkeypatch.setattr(validation_module, "_commit", lambda: next(commits))

    def execute(symbol, start, end, config):
        viable = _empty_viable()
        return SymbolResearchResult(
            "probe", _empty_standard(), viable, viable.copy(), end,
            tmp_path / "probe", start, end,
        )

    validator = HistoricalStrategyValidator(
        execute,
        warmup_bars=lambda _: 0,
        defer_outcome_tail_until_needed=True,
        enforce_stable_code_provenance=True,
    )
    with pytest.raises(RuntimeError, match="code provenance changed"):
        validator.run(_candidates(), _timeout_disabled_config(),
            config_path=tmp_path / "unused", publish=False)
