from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    OpportunityScanCancelled,
)
from crypto_strategy_lab.paper_scanner import (
    CycleStatus,
    LatestNativeStrategyEvaluator,
    PaperScannerConfig,
    PaperScannerRunner,
    StrategyEvaluation,
)
from crypto_strategy_lab.paper_scanner_state import (
    PaperScannerState,
    PaperScannerStateError,
    PaperScannerStateStore,
)
from crypto_strategy_lab.rule_native_engine import (
    RuleAwareDataLakeProductionBacktestEngine,
)
from crypto_strategy_lab.scanner_operations import (
    DiskMonitor, DiskMonitoringConfig, ScannerOperationalConfig,
)


UTC = timezone.utc
DECISION = datetime(2026, 1, 2, 14, 37, tzinfo=UTC)


class Scanner:
    def __init__(self, completed=None, failures=0):
        self.completed, self.failures, self.calls = completed, failures, 0

    def run(self, request, cancelled):
        self.calls += 1
        assert request.mode == "LIVE" and not cancelled()
        if self.calls <= self.failures:
            raise ConnectionError("temporary public API failure")
        return self.completed


class Evaluator:
    def __init__(
        self,
        candle=datetime(2026, 1, 2, 13, 0, tzinfo=UTC),
        available=datetime(2026, 1, 2, 14, 0, tzinfo=UTC),
        accepted=True,
    ):
        self.candle, self.available, self.accepted, self.calls = (
            candle,
            available,
            accepted,
            [],
        )

    def evaluate(self, candidate, decision, stale_limit):
        self.calls.append((candidate, decision))
        return StrategyEvaluation(
            self.accepted,
            self.candle,
            self.available,
            "BULL_LONG",
            "LONG",
            100.0,
            "native result",
        )


def completed(decision=DECISION, run_id="run-1"):
    final = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "final_rank": 1,
                "discovery_rank": 2,
                "strategy_interval": "1h",
                "strategy_source_identity": "source",
                "opportunity_model_name": "model",
                "opportunity_model_rank": 3,
                "opportunity_score": 0.75,
            }
        ]
    )
    return SimpleNamespace(
        manifest={"run_id": run_id},
        summary={"decision_timestamp": decision.isoformat()},
        final=final,
        preliminary=pd.DataFrame(),
        readiness=pd.DataFrame(),
    )


def runner(
    tmp_path,
    *,
    scan=None,
    evaluator=None,
    now=DECISION,
    retries=0,
    stale_market=timedelta(minutes=5),
    stale_strategy=timedelta(hours=2),
    sleeps=None,
    max_history=10_000,
    retention=timedelta(days=365),
    operational=None,
    monotonic=None,
):
    config = PaperScannerConfig(
        timedelta(seconds=1),
        stale_market,
        stale_strategy,
        retries,
        timedelta(seconds=2),
        tmp_path / "state.json",
        tmp_path / "audit.jsonl",
        max_history,
        retention,
        operational or ScannerOperationalConfig(),
    )
    scan = scan or Scanner(completed())
    return PaperScannerRunner(
        config,
        scan,
        lambda: SimpleNamespace(mode="LIVE"),
        evaluator or Evaluator(),
        clock=lambda: now,
        sleeper=(sleeps.append if sleeps is not None else lambda _: None),
        monotonic=monotonic,
    ), scan


def events(path):
    return [json.loads(line)["event_type"] for line in path.read_text().splitlines()]


def audit_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def health_config(tmp_path, *, disk=None):
    return ScannerOperationalConfig(
        health_path=tmp_path / "health.json",
        disk=disk or DiskMonitoringConfig(),
    )


def health(tmp_path):
    return json.loads((tmp_path / "health.json").read_text())


def test_fresh_verified_scan_emits_paper_entry_and_duplicate_survives_restart(tmp_path):
    app, scan = runner(tmp_path)
    first = app.run_once()
    assert scan.calls == 1
    assert first.status is CycleStatus.COMPLETED
    assert (
        first.final_candidates,
        first.candidates_evaluated,
        first.strategy_signals,
        first.new_paper_entries,
    ) == (1, 1, 1, 1)
    assert app.state.paper_entries[0]["record_type"] == "PAPER_ENTRY"
    assert (
        app.state.paper_entries[0]["scanner_decision_timestamp"] == DECISION.isoformat()
    )
    assert (
        json.loads((tmp_path / "state.json").read_text())["last_completed_cycle"][
            "status"
        ]
        == "COMPLETED"
    )
    next_decision = DECISION + timedelta(minutes=1)
    reloaded, _ = runner(
        tmp_path,
        scan=Scanner(completed(decision=next_decision, run_id="run-2")),
        now=next_decision,
    )
    second = reloaded.run_once()
    assert second.duplicate_signals_suppressed == 1
    assert second.new_paper_entries == 0
    assert len(reloaded.state.paper_entries) == 1
    assert {
        "SIGNAL_PENDING",
        "SIGNAL_EMITTED",
        "PAPER_ENTRY_RECORDED",
        "DUPLICATE_SUPPRESSED",
        "CANDIDATE_ACTIVATED",
        "CANDIDATE_SET_APPLIED",
    } <= set(events(tmp_path / "audit.jsonl"))


def test_lifecycle_commit_failure_keeps_prior_durable_membership(tmp_path):
    app, _ = runner(tmp_path, operational=health_config(tmp_path))
    assert app.run_once().status is CycleStatus.COMPLETED
    durable_before = (tmp_path / "state.json").read_bytes()
    state_before = list(app.state.candidate_lifecycle)

    def fail(_state):
        raise OSError("disk full")

    app.store.save = fail
    result = app.run_once()
    assert result.status is CycleStatus.FAILED
    assert app.state.candidate_lifecycle == state_before
    assert (tmp_path / "state.json").read_bytes() == durable_before
    assert app.consecutive_failed_cycles == 1
    assert health(tmp_path)["status"] == "UNHEALTHY"
    assert health(tmp_path)["last_error_category"] == "LIFECYCLE_PERSIST_FAILED"
    assert "total_cycle_duration_ms" in result.metrics
    terminal = audit_rows(tmp_path / "audit.jsonl")[-1]
    assert (terminal["event_type"], terminal["detail"]) == (
        "CYCLE_COMPLETED", "FAILED"
    )


def test_out_of_order_paper_scan_fails_without_changing_durable_state(tmp_path):
    scanner = Scanner(completed(decision=DECISION, run_id="newer"))
    app, _ = runner(
        tmp_path, scan=scanner, operational=health_config(tmp_path)
    )
    assert app.run_once().status is CycleStatus.COMPLETED
    durable_before = (tmp_path / "state.json").read_bytes()
    lifecycle_before = list(app.state.candidate_lifecycle)

    scanner.completed = completed(
        decision=DECISION - timedelta(minutes=1), run_id="older"
    )
    rejected = app.run_once()
    assert rejected.status is CycleStatus.FAILED
    assert app.state.candidate_lifecycle == lifecycle_before
    assert (tmp_path / "state.json").read_bytes() == durable_before
    assert health(tmp_path)["status"] == "UNHEALTHY"
    assert health(tmp_path)["last_error_category"] == "LIFECYCLE_REJECTED"
    assert "lifecycle_duration_ms" in rejected.metrics
    assert "total_cycle_duration_ms" in rejected.metrics
    assert "CANDIDATE_SET_REJECTED" in events(tmp_path / "audit.jsonl")
    terminal = audit_rows(tmp_path / "audit.jsonl")[-1]
    assert (terminal["event_type"], terminal["detail"]) == (
        "CYCLE_COMPLETED", "FAILED"
    )

    reloaded = PaperScannerStateStore(tmp_path / "state.json").load()
    assert reloaded.candidate_lifecycle == lifecycle_before


def test_empty_cursor_survives_restart_and_rejects_older_nonempty_scan(tmp_path):
    empty = completed(decision=DECISION, run_id="empty-newer")
    empty.final = empty.final.iloc[0:0]
    app, _ = runner(tmp_path, scan=Scanner(empty))
    assert app.run_once().status is CycleStatus.COMPLETED
    assert not app.state.candidate_lifecycle
    assert app.state.lifecycle_cursor.decision_timestamp == DECISION.isoformat()
    durable_before = (tmp_path / "state.json").read_bytes()

    older_decision = DECISION - timedelta(minutes=1)
    restarted, _ = runner(
        tmp_path,
        scan=Scanner(completed(decision=older_decision, run_id="older-present")),
        now=DECISION,
    )
    assert restarted.state.lifecycle_cursor == app.state.lifecycle_cursor
    assert restarted.run_once().status is CycleStatus.FAILED
    assert (tmp_path / "state.json").read_bytes() == durable_before
    assert not PaperScannerStateStore(tmp_path / "state.json").load().candidate_lifecycle


def test_default_lifecycle_preserves_task10_candidate_and_signal_parity(tmp_path):
    scan_result = completed()
    scan_result.final = pd.DataFrame([
        {**scan_result.final.iloc[0].to_dict(), "symbol": "ETHUSDT", "final_rank": 2},
        {**scan_result.final.iloc[0].to_dict(), "symbol": "BTCUSDT", "final_rank": 1},
    ])
    evaluator = Evaluator()
    app, _ = runner(tmp_path, scan=Scanner(scan_result), evaluator=evaluator)
    result = app.run_once()
    legacy_rows = scan_result.final.sort_values("final_rank").to_dict("records")
    assert [call[0] for call in evaluator.calls] == legacy_rows
    assert result.strategy_signals == result.new_paper_entries == len(legacy_rows)
    assert [entry["symbol"] for entry in app.state.paper_entries] == [
        row["symbol"] for row in legacy_rows
    ]


def test_new_completed_signal_candle_may_emit_again(tmp_path):
    app, _ = runner(tmp_path)
    app.run_once()
    app.evaluator.candle = datetime(2026, 1, 2, 14, 0, tzinfo=UTC)
    assert app.run_once().new_paper_entries == 1


def test_stale_discovery_and_strategy_do_not_emit_or_evaluate_in_wrong_order(tmp_path):
    stale_app, _ = runner(tmp_path / "market", now=DECISION + timedelta(minutes=6))
    assert stale_app.run_once().status is CycleStatus.STALE_DISCOVERY
    assert not stale_app.evaluator.calls
    strategy_app, _ = runner(
        tmp_path / "strategy",
        evaluator=Evaluator(available=DECISION - timedelta(hours=3)),
    )
    result = strategy_app.run_once()
    assert (result.stale_candidates, result.new_paper_entries) == (1, 0)
    assert result.status is CycleStatus.STALE_STRATEGY_DATA
    assert "STALE_STRATEGY_DATA" in events(tmp_path / "strategy" / "audit.jsonl")


def test_off_grid_native_adapter_selects_latest_available_completed_row():
    prepared = SimpleNamespace(
        timestamp=np.array(
            ["2026-01-02T13:00", "2026-01-02T14:00"], dtype="datetime64[ns]"
        ),
        decision_available_at=np.array(
            ["2026-01-02T14:00", "2026-01-02T14:00"], dtype="datetime64[ns]"
        ),
        strategy_interval=pd.Timedelta(hours=1),
        research=(),
        close=np.array([10.0, 20.0]),
    )

    class Engine:
        def required_entry_research_features(self, index):
            return frozenset()

        def evaluate_prepared_entry(self, index):
            assert index == 0
            return SimpleNamespace(
                accepted=True,
                strategy_profile_key="BULL_LONG",
                side="LONG",
                reference_price=10.0,
                detail="passed",
            )

    result = LatestNativeStrategyEvaluator(
        lambda candidate: (prepared, Engine())
    ).evaluate({}, DECISION, timedelta(hours=2))
    assert result.signal_candle_timestamp == datetime(2026, 1, 2, 13, 0, tzinfo=UTC)
    assert result.decision_available_at == datetime(2026, 1, 2, 14, 0, tzinfo=UTC)


def test_native_adapter_excludes_row_with_post_decision_research_value():
    prepared = SimpleNamespace(
        timestamp=np.array(
            ["2026-01-02T12:00", "2026-01-02T13:00"], dtype="datetime64[ns]"
        ),
        decision_available_at=np.array(
            ["2026-01-02T13:00", "2026-01-02T14:00"], dtype="datetime64[ns]"
        ),
        strategy_interval=pd.Timedelta(hours=1),
        research=(
            SimpleNamespace(
                name="required",
                available_at=np.array(
                    ["2026-01-02T13:00", "2026-01-02T14:45"],
                    dtype="datetime64[ns]",
                ),
            ),
        ),
        close=np.array([10.0, 20.0]),
    )

    class Engine:
        def required_entry_research_features(self, index):
            return frozenset({"required"})

        def evaluate_prepared_entry(self, index):
            assert index == 0
            return SimpleNamespace(
                accepted=True,
                strategy_profile_key="BULL_LONG",
                side="LONG",
                reference_price=10.0,
                detail="passed",
            )

    result = LatestNativeStrategyEvaluator(
        lambda candidate: (prepared, Engine())
    ).evaluate({}, DECISION, timedelta(hours=2))
    assert result.signal_candle_timestamp == datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    assert result.decision_available_at == datetime(2026, 1, 2, 13, 0, tzinfo=UTC)


def test_native_adapter_uses_candle_close_as_minimum_availability():
    prepared = SimpleNamespace(
        timestamp=np.array(["2026-01-02T13:00"], dtype="datetime64[ns]"),
        decision_available_at=np.array(["2026-01-02T13:00"], dtype="datetime64[ns]"),
        strategy_interval=pd.Timedelta(hours=1),
        research=(),
        close=np.array([10.0]),
    )

    class Engine:
        def required_entry_research_features(self, index):
            return frozenset()

        def evaluate_prepared_entry(self, index):
            return SimpleNamespace(
                accepted=True,
                strategy_profile_key="BULL_LONG",
                side="LONG",
                reference_price=10.0,
                detail="passed",
            )

    result = LatestNativeStrategyEvaluator(
        lambda candidate: (prepared, Engine())
    ).evaluate({}, DECISION, timedelta(hours=1))
    assert result.decision_available_at == datetime(2026, 1, 2, 14, 0, tzinfo=UTC)


def test_native_adapter_ignores_unrelated_late_research_block():
    prepared = SimpleNamespace(
        timestamp=np.array(["2026-01-02T13:00"], dtype="datetime64[ns]"),
        decision_available_at=np.array(["2026-01-02T14:00"], dtype="datetime64[ns]"),
        strategy_interval=pd.Timedelta(hours=1),
        research=(
            SimpleNamespace(
                name="unused",
                available_at=np.array(["2026-01-02T14:45"], dtype="datetime64[ns]"),
            ),
        ),
        close=np.array([10.0]),
    )

    class Engine:
        def required_entry_research_features(self, index):
            return frozenset()

        def evaluate_prepared_entry(self, index):
            return SimpleNamespace(
                accepted=True,
                strategy_profile_key="BULL_LONG",
                side="LONG",
                reference_price=10.0,
                detail="passed",
            )

    result = LatestNativeStrategyEvaluator(
        lambda candidate: (prepared, Engine())
    ).evaluate({}, DECISION, timedelta(hours=1))
    assert result.accepted
    assert result.decision_available_at == datetime(2026, 1, 2, 14, 0, tzinfo=UTC)


def test_no_completed_causal_row_is_audited_as_stale_not_runtime_error(tmp_path):
    class NoCausalEvaluator:
        def evaluate(self, candidate, decision, stale_limit):
            from crypto_strategy_lab.paper_scanner import StaleStrategyDataError

            raise StaleStrategyDataError("no completed causal strategy candle")

    app, _ = runner(tmp_path, evaluator=NoCausalEvaluator())
    result = app.run_once()
    assert result.status is CycleStatus.STALE_STRATEGY_DATA
    assert (result.stale_candidates, result.new_paper_entries) == (1, 0)
    assert "STALE_STRATEGY_DATA" in events(tmp_path / "audit.jsonl")


def test_strategy_integrity_error_fails_cycle_without_killing_runner(tmp_path):
    class BrokenEvaluator:
        broken = True

        def evaluate(self, candidate, decision, stale_limit):
            if self.broken:
                raise ValueError("prepared research is misaligned")
            return Evaluator().evaluate(candidate, decision, stale_limit)

    evaluator = BrokenEvaluator()
    app, _ = runner(tmp_path, evaluator=evaluator)
    assert app.run_once().status is CycleStatus.FAILED
    assert not app.state.paper_entries
    evaluator.broken = False
    assert app.run_once().new_paper_entries == 1
    assert "STRATEGY_EVALUATION_FAILED" in events(tmp_path / "audit.jsonl")


def test_signal_state_failure_does_not_install_uncommitted_duplicate_key(tmp_path):
    app, _ = runner(tmp_path)
    original_save = app.store.save
    failed = False

    def fail_once(state):
        nonlocal failed
        if state.paper_entries and not failed:
            failed = True
            raise OSError("disk unavailable")
        original_save(state)

    app.store.save = fail_once
    result = app.run_once()
    assert result.status is CycleStatus.FAILED
    assert not app.state.emitted_signal_ids
    assert not app.state.paper_entries
    # The lifecycle snapshot was already committed atomically; failed signal
    # persistence leaves that prior durable state intact without an identity.
    durable = json.loads((tmp_path / "state.json").read_text())
    assert not durable["emitted_signal_ids"] and durable["candidate_lifecycle"]
    assert "STATE_PERSIST_FAILED" in events(tmp_path / "audit.jsonl")

    # The failed in-memory attempt does not suppress a later durable emission.
    assert app.run_once().new_paper_entries == 1
    assert len(app.state.paper_entries) == 1


def test_production_factory_composes_scanner_data_lake_and_native_engine(
    tmp_path, monkeypatch
):
    import crypto_strategy_lab.paper_scanner_production as production

    scanner = Scanner(completed())
    scanner_factory_calls = []
    monkeypatch.setattr(
        production,
        "create_opportunity_scanner_service",
        lambda raw, cache, output, **options: (
            scanner_factory_calls.append((raw, cache, output, options)) or scanner
        ),
    )
    bundle = object()
    prepared = SimpleNamespace(strategy_interval=pd.Timedelta(hours=1))
    load_calls = []
    monkeypatch.setattr(
        production,
        "load_backtest_bundle",
        lambda store, request, **options: (
            load_calls.append((store, request, options)) or bundle
        ),
    )
    monkeypatch.setattr(
        production,
        "from_data_lake_bundle",
        lambda actual, config: (prepared, None),
    )
    native_engine = object()
    engine_calls = []
    monkeypatch.setattr(
        RuleAwareDataLakeProductionBacktestEngine,
        "from_prepared",
        classmethod(
            lambda cls, frame, intrabar, config: (
                engine_calls.append((frame, intrabar, config)) or native_engine
            )
        ),
    )
    paper_config = PaperScannerConfig(
        timedelta(minutes=1),
        timedelta(minutes=2),
        timedelta(hours=2),
        0,
        timedelta(0),
        tmp_path / "paper-state.json",
        tmp_path / "paper-audit.jsonl",
        operational=health_config(
            tmp_path,
            disk=DiskMonitoringConfig(
                warning_free_percent=None, critical_free_percent=None,
            ),
        ),
    )
    strategy_config = BacktestConfig(
        strategy_timeframe_minutes=60, telemetry_interval_minutes=60
    )

    runner = production.create_production_paper_scanner(
        market_data_root=tmp_path / "raw",
        cache_root=tmp_path / "cache",
        scan_output_root=tmp_path / "scans",
        paper_config=paper_config,
        scan_request_factory=lambda: SimpleNamespace(mode="LIVE"),
        strategy_config=strategy_config,
        clock=lambda: DECISION,
    )

    assert isinstance(runner, PaperScannerRunner)
    assert isinstance(runner.evaluator, LatestNativeStrategyEvaluator)
    assert runner.scanner is scanner
    assert runner.config.state_path == tmp_path / "paper-state.json"
    assert runner.config.audit_log_path == tmp_path / "paper-audit.jsonl"
    assert len(scanner_factory_calls) == 1
    raw, cache, output, options = scanner_factory_calls[0]
    assert (raw, cache, output) == (
        tmp_path / "raw", tmp_path / "cache", tmp_path / "scans"
    )
    assert options["live_client"].base_url == "https://fapi.binance.com"
    assert runner.config.operational.disk.paths == {
        "paper_state": tmp_path,
        "paper_audit": tmp_path,
        "opportunity_scans": tmp_path / "scans",
        "data_lake_raw": tmp_path / "raw",
        "cache": tmp_path / "cache",
    }
    assert runner.config.operational.disk.cache_path == tmp_path / "cache"
    candidate = {
        "symbol": "BTCUSDT",
        "strategy_interval": "1h",
        "strategy_request_start": "2026-01-01T00:00:00+00:00",
        "strategy_request_end": "2026-01-02T00:00:00+00:00",
        "strategy_request_market": "futures_um",
        "strategy_request_exchange": "binance",
    }
    assert runner.evaluator.engine_builder(candidate) == (prepared, native_engine)
    assert load_calls[0][1].symbol == "BTCUSDT"
    assert load_calls[0][1].strategy_interval == "1h"
    assert engine_calls == [(prepared, None, strategy_config)]
    assert not any(
        name in vars(runner)
        for name in ("broker", "order_client", "auth", "live_orders")
    )
    options["live_client"].telemetry({
        "endpoint": "/fapi/v1/ticker/24hr", "http_status": 429,
        "retry_after_seconds": "4", "used_weight": "120",
    })
    scanner.completed.final = scanner.completed.final.iloc[0:0]
    result = runner.run_once()
    assert result.metrics["binance_public_api"]["rate_limit_pressure"] is True
    api_rows = [row for row in audit_rows(tmp_path / "paper-audit.jsonl")
                if row["event_type"] == "BINANCE_API_RESPONSE"]
    assert api_rows[-1]["fields"]["used_weight"] == "120"
    assert health(tmp_path)["metrics"]["binance_public_api"]["latest"]["http_status"] == 429
    assert health(tmp_path)["status"] == "DEGRADED"


def test_retry_is_bounded_and_failed_runner_can_run_next_cycle(tmp_path):
    sleeps = []
    app, scan = runner(
        tmp_path, scan=Scanner(completed(), failures=2), retries=1, sleeps=sleeps
    )
    assert app.run_once().status is CycleStatus.FAILED
    assert scan.calls == 2 and sleeps == [2.0]
    scan.failures = 0
    assert app.run_once().new_paper_entries == 1
    assert {"SCAN_RETRY", "SCAN_FAILED"} <= set(events(tmp_path / "audit.jsonl"))


def test_runner_health_clean_retry_partial_and_recovery_transitions(tmp_path):
    scan_result = completed()
    scan_result.preliminary = pd.DataFrame([{
        "symbol": "BTCUSDT", "acquisition_state": "REUSED",
        "row_count": 10, "acquisition_ranges": "[]",
    }])
    app, scan = runner(
        tmp_path, scan=Scanner(scan_result, failures=1), retries=1,
        operational=health_config(tmp_path), sleeps=[],
    )
    assert app.run_once().status is CycleStatus.COMPLETED
    assert health(tmp_path)["status"] == "DEGRADED"
    assert app.consecutive_failed_cycles == 0

    scan_result.preliminary.loc[0, "acquisition_state"] = "DOWNLOAD_FAILED"
    scan_result.readiness = pd.DataFrame([{
        "symbol": "BTCUSDT", "dataset": "funding", "interval": "8h",
        "requested_start": "s", "requested_end": "e",
        "acquisition_state": "MISSING", "feature_name": "context",
        "feature_readiness": "DEGRADED",
    }])
    assert app.run_once().status is CycleStatus.COMPLETED
    current = health(tmp_path)
    assert current["status"] == "DEGRADED"
    assert current["metrics"]["strategy_candle_acquisition"]["download_failed"] == 1
    assert current["metrics"]["rich_feature_readiness"]["degraded"] == 1

    scan_result.preliminary.loc[0, "acquisition_state"] = "REUSED"
    scan_result.readiness = pd.DataFrame()
    assert app.run_once().status is CycleStatus.COMPLETED
    assert health(tmp_path)["status"] == "HEALTHY"
    assert app.consecutive_failed_cycles == 0


@pytest.mark.parametrize("free,expected", [(15, "DEGRADED"), (5, "UNHEALTHY")])
def test_runner_health_observes_disk_warning_and_critical(tmp_path, free, expected):
    disk = DiskMonitoringConfig(
        paths={"paper": tmp_path}, sample_every_cycles=1,
        warning_free_percent=20, critical_free_percent=10,
    )
    app, _ = runner(tmp_path, operational=health_config(tmp_path, disk=disk))
    app.disk_monitor.usage = lambda _path: SimpleNamespace(
        total=100, used=100 - free, free=free
    )
    assert app.run_once().status is CycleStatus.COMPLETED
    assert health(tmp_path)["status"] == expected


def test_disk_monitor_failure_degrades_success_without_state_or_scheduler_crash(tmp_path):
    disk = DiskMonitoringConfig(
        paths={"paper": tmp_path}, sample_every_cycles=1,
    )
    app, _ = runner(tmp_path, operational=health_config(tmp_path, disk=disk))
    app.disk_monitor.usage = lambda path: (_ for _ in ()).throw(
        OSError(5, "filesystem unavailable", str(path))
    )
    result = app.run_once()
    durable = PaperScannerStateStore(tmp_path / "state.json").load()
    assert result.status is CycleStatus.COMPLETED
    assert len(durable.paper_entries) == 1
    assert durable.candidate_lifecycle == app.state.candidate_lifecycle
    current = health(tmp_path)
    assert current["status"] == "DEGRADED"
    assert current["metrics"]["disk_monitor_error"] == {
        "error_type": "OSError", "operation": "disk_cache_sample",
        "path": str(tmp_path),
    }
    assert "DISK_MONITOR_FAILED" in events(tmp_path / "audit.jsonl")

    class Stop:
        def is_set(self): return False
        def wait(self, _seconds): return True

    app.run_forever(Stop())
    assert "RUNTIME_CYCLE_CRASH" not in events(tmp_path / "audit.jsonl")


def test_different_device_critical_cache_disk_makes_runner_unhealthy(tmp_path):
    first, second = tmp_path / "raw", tmp_path / "cache"
    second.mkdir()
    (second / "entry.bin").write_bytes(b"cache-data")
    disk = DiskMonitoringConfig(
        paths={"data_lake_raw": first, "cache": second}, cache_path=second,
        sample_every_cycles=1,
        warning_free_percent=20, critical_free_percent=10,
    )
    app, _ = runner(tmp_path, operational=health_config(tmp_path, disk=disk))
    devices = {first: 1, second: 2}
    usage_calls = []
    app.disk_monitor = DiskMonitor(
        disk,
        usage=lambda path: (
            usage_calls.append(Path(path)),
            SimpleNamespace(total=100, used=(50 if Path(path) == first else 95),
                            free=(50 if Path(path) == first else 5)),
        )[1],
        filesystem_stat=lambda path: SimpleNamespace(st_dev=devices[Path(path)]),
    )
    assert app.run_once().status is CycleStatus.COMPLETED
    assert usage_calls == [first, second]
    current = health(tmp_path)
    assert current["disk"]["data_lake_raw"]["level"] == "OK"
    assert current["disk"]["cache"]["level"] == "CRITICAL"
    assert current["disk"]["cache_usage"]["size_bytes"] == 10
    assert current["status"] == "UNHEALTHY"


def test_cancelled_cycle_is_not_success_and_preserves_failure_history(tmp_path):
    app, _ = runner(tmp_path, operational=health_config(tmp_path))
    app.consecutive_failed_cycles = 2
    app.last_error_category = "PRIOR_FAILURE"
    result = app.run_once(cancelled=lambda: True)
    current = health(tmp_path)
    assert result.status is CycleStatus.CANCELLED
    assert current["status"] == "DEGRADED"
    assert current["last_successful_scan_at"] is None
    assert current["consecutive_failed_cycles"] == 2
    assert current["last_error_category"] == "PRIOR_FAILURE"


def test_monotonic_cycle_metrics_are_present_without_wall_clock_elapsed(tmp_path):
    class Monotonic:
        value = 0.0
        def __call__(self):
            self.value += 0.25
            return self.value

    app, _ = runner(
        tmp_path, operational=health_config(tmp_path), monotonic=Monotonic()
    )
    metrics = app.run_once().metrics
    assert all(metrics[name] >= 0 for name in (
        "pipeline_duration_ms", "lifecycle_duration_ms",
        "strategy_evaluation_duration_ms", "total_cycle_duration_ms",
    ))


def test_stale_discovery_and_stale_only_strategy_are_unhealthy(tmp_path):
    stale_app, _ = runner(
        tmp_path / "discovery", now=DECISION + timedelta(minutes=6),
        operational=health_config(tmp_path / "discovery"),
    )
    assert stale_app.run_once().status is CycleStatus.STALE_DISCOVERY
    assert health(tmp_path / "discovery")["status"] == "UNHEALTHY"

    strategy_app, _ = runner(
        tmp_path / "strategy",
        evaluator=Evaluator(available=DECISION - timedelta(hours=3)),
        operational=health_config(tmp_path / "strategy"),
    )
    assert strategy_app.run_once().status is CycleStatus.STALE_STRATEGY_DATA
    assert health(tmp_path / "strategy")["status"] == "UNHEALTHY"


def test_scheduler_recovers_unexpected_cycle_then_later_cycle_succeeds(tmp_path):
    app, _ = runner(tmp_path, operational=health_config(tmp_path))
    real_run_once = app.run_once
    calls = 0

    def crashing_once(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("cycle crash")
        return real_run_once(*args)

    app.run_once = crashing_once

    class Stop:
        waits = []
        def is_set(self): return False
        def wait(self, seconds):
            self.waits.append(seconds)
            return len(self.waits) == 2

    stop = Stop()
    app.run_forever(stop)
    assert calls == 2
    assert stop.waits == [30.0, 1.0]
    assert "RUNTIME_CYCLE_CRASH" in events(tmp_path / "audit.jsonl")
    assert app.consecutive_failed_cycles == 0


def test_state_is_versioned_atomic_and_corruption_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    store = PaperScannerStateStore(path)
    replaced = []
    import crypto_strategy_lab.run_manifest as manifest

    original = manifest.os.replace
    monkeypatch.setattr(
        manifest.os,
        "replace",
        lambda source, target: (
            replaced.append((source, target)),
            original(source, target),
        )[1],
    )
    store.save(
        PaperScannerState(
            ["one"],
            [
                {
                    "record_type": "PAPER_ENTRY",
                    "signal_id": "one",
                    "signal_candle_timestamp": DECISION.isoformat(),
                }
            ],
        )
    )
    assert replaced and json.loads(path.read_text())["version"] == 2
    path.write_text("not json")
    with pytest.raises(PaperScannerStateError):
        store.load()


@pytest.mark.parametrize(
    "ids,entries",
    [
        (["one"], []),
        (
            [],
            [
                {
                    "record_type": "PAPER_ENTRY",
                    "signal_id": "one",
                    "signal_candle_timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        ),
        (
            ["one"],
            [
                {
                    "record_type": "PAPER_ENTRY",
                    "signal_id": "two",
                    "signal_candle_timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        ),
    ],
)
def test_state_rejects_ledger_duplicate_index_mismatch(tmp_path, ids, entries):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "emitted_signal_ids": ids,
                "paper_entries": entries,
                "last_completed_cycle": None,
                "last_successful_scan_run_id": None,
            }
        )
    )
    with pytest.raises(PaperScannerStateError, match="corrupt"):
        PaperScannerStateStore(path).load()


@pytest.mark.parametrize("timestamp", ["not-a-time", "2026-01-01T00:00:00"])
def test_state_rejects_invalid_or_naive_signal_timestamp(tmp_path, timestamp):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "emitted_signal_ids": ["one"],
                "paper_entries": [
                    {
                        "record_type": "PAPER_ENTRY",
                        "signal_id": "one",
                        "signal_candle_timestamp": timestamp,
                    }
                ],
                "last_completed_cycle": None,
                "last_successful_scan_run_id": None,
            }
        )
    )
    with pytest.raises(PaperScannerStateError, match="corrupt"):
        PaperScannerStateStore(path).load()


def test_scheduler_waits_and_stops_without_order_api(tmp_path):
    app, scan = runner(tmp_path)

    class Stop:
        def __init__(self):
            self.waits = []

        def is_set(self):
            return False

        def wait(self, seconds):
            self.waits.append(seconds)
            return True

    stop = Stop()
    app.run_forever(stop)
    assert scan.calls == 1 and stop.waits == [1.0]
    # The PAPER runner's constructor has no broker/order dependency or mode switch.
    assert not any("order" in name.lower() for name in vars(app))
    assert {"RUNTIME_STARTED", "RUNTIME_STOPPED"} <= set(
        events(tmp_path / "audit.jsonl")
    )


def test_scheduler_propagates_stop_callback_into_active_scan(tmp_path):
    stop = Event()

    class CancellingScanner:
        calls = 0

        def run(self, request, cancelled):
            self.calls += 1
            stop.set()
            assert cancelled()
            raise OpportunityScanCancelled("stopped")

    app, scanner = runner(tmp_path, scan=CancellingScanner())
    app.run_forever(stop)
    assert scanner.calls == 1
    assert app.state.last_completed_cycle["status"] == CycleStatus.CANCELLED.value
    assert "SCAN_CANCELLED" in events(tmp_path / "audit.jsonl")


def test_scheduler_stop_interrupts_retry_backoff(tmp_path):
    stop = Event()

    class FailingScanner:
        calls = 0

        def run(self, request, cancelled):
            self.calls += 1
            stop.set()
            raise ConnectionError("temporary failure")

    app, scanner = runner(tmp_path, scan=FailingScanner(), retries=3)
    app.run_forever(stop)
    assert scanner.calls == 1
    assert app.state.last_completed_cycle["status"] == CycleStatus.CANCELLED.value
    audit_events = events(tmp_path / "audit.jsonl")
    assert "SCAN_RETRY" in audit_events
    assert "SCAN_CANCELLED" in audit_events


def test_non_entry_is_audited_without_ledger_record(tmp_path):
    app, _ = runner(tmp_path, evaluator=Evaluator(accepted=False))
    assert app.run_once().new_paper_entries == 0
    assert "NO_STRATEGY_ENTRY" in events(tmp_path / "audit.jsonl")


def test_history_bound_fails_closed_without_evicting_live_duplicate_key(tmp_path):
    app, _ = runner(tmp_path, max_history=1)
    assert app.run_once().new_paper_entries == 1
    original_id = app.state.emitted_signal_ids[0]
    app.evaluator.candle = datetime(2026, 1, 2, 14, 0, tzinfo=UTC)
    result = app.run_once()
    assert result.status is CycleStatus.FAILED
    assert app.state.emitted_signal_ids == [original_id]
    assert len(app.state.paper_entries) == 1
    assert "STATE_CAPACITY_REACHED" in events(tmp_path / "audit.jsonl")


def test_retention_can_only_prune_keys_older_than_stale_window(tmp_path):
    with pytest.raises(ValueError, match="must exceed strategy staleness"):
        PaperScannerConfig(
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=2),
            0,
            timedelta(0),
            tmp_path / "state",
            tmp_path / "audit",
            signal_history_retention=timedelta(hours=2),
        )
