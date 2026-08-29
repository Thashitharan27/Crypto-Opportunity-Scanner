from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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
):
    config = PaperScannerConfig(
        timedelta(seconds=1),
        stale_market,
        stale_strategy,
        retries,
        timedelta(seconds=2),
        tmp_path / "state.json",
        tmp_path / "audit.jsonl",
    )
    scan = scan or Scanner(completed())
    return PaperScannerRunner(
        config,
        scan,
        lambda: SimpleNamespace(mode="LIVE"),
        evaluator or Evaluator(),
        clock=lambda: now,
        sleeper=(sleeps.append if sleeps is not None else lambda _: None),
    ), scan


def events(path):
    return [json.loads(line)["event_type"] for line in path.read_text().splitlines()]


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
    reloaded, _ = runner(tmp_path, scan=Scanner(completed(run_id="run-2")))
    second = reloaded.run_once()
    assert second.duplicate_signals_suppressed == 1
    assert second.new_paper_entries == 0
    assert len(reloaded.state.paper_entries) == 1
    assert {
        "SIGNAL_PENDING",
        "SIGNAL_EMITTED",
        "PAPER_ENTRY_RECORDED",
        "DUPLICATE_SUPPRESSED",
    } <= set(events(tmp_path / "audit.jsonl"))


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
    assert "STALE_STRATEGY_DATA" in events(tmp_path / "strategy" / "audit.jsonl")


def test_off_grid_native_adapter_selects_latest_available_completed_row():
    prepared = SimpleNamespace(
        timestamp=np.array(
            ["2026-01-02T13:00", "2026-01-02T14:00"], dtype="datetime64[ns]"
        ),
        decision_available_at=np.array(
            ["2026-01-02T14:00", "2026-01-02T15:00"], dtype="datetime64[ns]"
        ),
        close=np.array([10.0, 20.0]),
    )

    class Engine:
        def _strategy_profile_filter_result(self, index):
            assert index == 0
            return True, "passed"

        def _profile_context(self, index):
            return "BULL", "LONG", "BULL_LONG", object()

    result = LatestNativeStrategyEvaluator(
        lambda candidate: (prepared, Engine())
    ).evaluate({}, DECISION, timedelta(hours=2))
    assert result.signal_candle_timestamp == datetime(2026, 1, 2, 13, 0, tzinfo=UTC)
    assert result.decision_available_at == datetime(2026, 1, 2, 14, 0, tzinfo=UTC)


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
    store.save(PaperScannerState(["one"], [{"signal_id": "one"}]))
    assert replaced and json.loads(path.read_text())["version"] == 1
    path.write_text("not json")
    with pytest.raises(PaperScannerStateError):
        store.load()


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


def test_non_entry_is_audited_without_ledger_record(tmp_path):
    app, _ = runner(tmp_path, evaluator=Evaluator(accepted=False))
    assert app.run_once().new_paper_entries == 0
    assert "NO_STRATEGY_ENTRY" in events(tmp_path / "audit.jsonl")
