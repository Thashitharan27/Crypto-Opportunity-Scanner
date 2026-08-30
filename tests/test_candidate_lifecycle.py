from datetime import datetime, timedelta, timezone
import json

import pytest

from crypto_strategy_lab.candidate_lifecycle import (
    CandidateLifecyclePolicy,
    HistoricalCandidateSnapshot,
    LifecycleStatus,
    TransitionType,
    apply_candidate_lifecycle,
    replay_candidate_lifecycle,
)
from crypto_strategy_lab.paper_scanner_state import (
    PaperScannerState,
    PaperScannerStateError,
    PaperScannerStateStore,
)


UTC = timezone.utc
T1 = datetime(2026, 1, 1, 10, tzinfo=UTC)


def row(symbol, final, discovery=None):
    return {"symbol": symbol, "final_rank": final, "discovery_rank": discovery,
            "strategy_interval": "1h", "opaque_task10_value": symbol.lower()}


def apply(state=(), rows=(), when=T1, run="run-1", cursor=None):
    return apply_candidate_lifecycle(
        state, cursor, rows, when, run, CandidateLifecyclePolicy()
    )


def kinds(result):
    return [(item.symbol, item.transition) for item in result.transitions]


def test_rolling_immediate_lifecycle_activation_retention_removal_reactivation():
    first = apply(rows=[row("BTCUSDT", 1, 2)])
    assert kinds(first) == [("BTCUSDT", TransitionType.ACTIVATED)]
    assert first.state[0].status is LifecycleStatus.ACTIVE
    second = apply(first.state, [row("BTCUSDT", 1, 2)], T1 + timedelta(hours=1), "run-2", first.cursor)
    assert not second.transitions
    assert second.state[0].first_seen_decision_timestamp == T1.isoformat()
    removed = apply(second.state, [], T1 + timedelta(hours=2), "run-3", second.cursor)
    assert kinds(removed) == [("BTCUSDT", TransitionType.REMOVED)]
    assert removed.state[0].removed_timestamp == (T1 + timedelta(hours=2)).isoformat()
    again = apply(removed.state, [row("BTCUSDT", 1, 2)], T1 + timedelta(hours=3), "run-4", removed.cursor)
    assert kinds(again) == [("BTCUSDT", TransitionType.ACTIVATED)]
    assert again.state[0].activated_timestamp == (T1 + timedelta(hours=3)).isoformat()
    assert again.state[0].first_seen_decision_timestamp == T1.isoformat()


def test_rank_tracking_no_false_transition_and_deterministic_symbol_order():
    first = apply(rows=[row("ETHUSDT", 2, 8), row("BTCUSDT", 1, 4)])
    assert [item["symbol"] for item in first.active_rows] == ["BTCUSDT", "ETHUSDT"]
    changed = apply(
        first.state,
        [row("ETHUSDT", 1, 7), row("BTCUSDT", 2, 4), row("ADAUSDT", 3, 9)],
        T1 + timedelta(hours=1), "run-2",
        first.cursor,
    )
    assert kinds(changed) == [
        ("ADAUSDT", TransitionType.ACTIVATED),
        ("BTCUSDT", TransitionType.RANK_CHANGED),
        ("ETHUSDT", TransitionType.RANK_CHANGED),
    ]
    btc = next(item for item in changed.state if item.symbol == "BTCUSDT")
    assert (btc.previous_final_rank, btc.final_rank) == (1, 2)
    unchanged = apply(changed.state, list(changed.active_rows), T1 + timedelta(hours=2), "run-3", changed.cursor)
    assert not unchanged.transitions


def test_discovery_rank_only_change_has_unambiguous_transition():
    first = apply(rows=[row("BTCUSDT", 1, 4)])
    changed = apply(
        first.state, [row("BTCUSDT", 1, 7)], T1 + timedelta(hours=1), "run-2", first.cursor
    )
    transition = changed.transitions[0]
    assert transition.transition is TransitionType.RANK_CHANGED
    assert (transition.previous_final_rank, transition.final_rank) == (1, 1)
    assert (transition.previous_discovery_rank, transition.discovery_rank) == (4, 7)


def test_shared_engine_rejects_older_decision_for_active_and_removed_state():
    active = apply(rows=[row("BTCUSDT", 1)], when=T1 + timedelta(hours=1), run="run-2")
    with pytest.raises(ValueError, match="later than prior"):
        apply(active.state, [row("BTCUSDT", 1)], T1, "run-1", active.cursor)

    removed = apply(active.state, [], T1 + timedelta(hours=2), "run-3", active.cursor)
    with pytest.raises(ValueError, match="later than prior"):
        apply(removed.state, [row("BTCUSDT", 1)], T1, "run-1", removed.cursor)


def test_exact_same_scan_replay_is_idempotent_but_equal_other_run_is_rejected():
    first = apply(rows=[row("BTCUSDT", 1, 2)])
    assert apply(first.state, [row("BTCUSDT", 1, 2)], cursor=first.cursor) == first.__class__(
        first.state, first.cursor, (), first.active_rows
    )
    with pytest.raises(ValueError, match="later than prior"):
        apply(first.state, [row("BTCUSDT", 1, 2)], run="different-run", cursor=first.cursor)


def test_empty_snapshot_cursor_blocks_older_scan_and_exact_empty_replay_is_idempotent():
    empty = apply(rows=(), when=T1 + timedelta(hours=3), run="empty-3")
    assert empty.state == () and empty.transitions == ()
    assert empty.cursor.decision_timestamp == (T1 + timedelta(hours=3)).isoformat()
    with pytest.raises(ValueError, match="later than prior"):
        apply((), [row("BTCUSDT", 1)], T1 + timedelta(hours=2), "older", empty.cursor)
    replay = apply((), (), T1 + timedelta(hours=3), "empty-3", empty.cursor)
    assert replay.cursor == empty.cursor and replay.transitions == ()
    with pytest.raises(ValueError, match="later than prior"):
        apply((), (), T1 + timedelta(hours=3), "other-run", empty.cursor)


def test_cursor_advances_across_consecutive_empty_snapshots():
    first = apply(rows=(), when=T1, run="empty-1")
    second = apply((), (), T1 + timedelta(hours=1), "empty-2", first.cursor)
    third = apply((), (), T1 + timedelta(hours=2), "empty-3", second.cursor)
    assert [result.cursor.decision_timestamp for result in (first, second, third)] == [
        T1.isoformat(),
        (T1 + timedelta(hours=1)).isoformat(),
        (T1 + timedelta(hours=2)).isoformat(),
    ]
    assert not first.state and not second.state and not third.state


def test_empty_no_transition_after_removal_advances_cursor_and_blocks_intermediate_scan():
    active = apply(rows=[row("BTCUSDT", 1)], when=T1, run="active")
    removed = apply(
        active.state, (), T1 + timedelta(hours=1), "removed", active.cursor
    )
    later_empty = apply(
        removed.state, (), T1 + timedelta(hours=3), "empty-3", removed.cursor
    )
    assert not later_empty.transitions
    assert later_empty.state == removed.state
    with pytest.raises(ValueError, match="later than prior"):
        apply(
            later_empty.state, [row("BTCUSDT", 1)],
            T1 + timedelta(hours=2), "older-present", later_empty.cursor,
        )


def test_empty_snapshot_removes_all_in_symbol_order_and_function_is_pure():
    first = apply(rows=[row("ETHUSDT", 2), row("BTCUSDT", 1)])
    before = tuple(first.state)
    one = apply(before, [], T1 + timedelta(hours=1), "run-2", first.cursor)
    two = apply(before, [], T1 + timedelta(hours=1), "run-2", first.cursor)
    assert one == two
    assert [item.symbol for item in one.transitions] == ["BTCUSDT", "ETHUSDT"]
    assert before == first.state


def test_default_policy_identity_is_canonical_and_rejects_speculative_policy():
    policy = CandidateLifecyclePolicy()
    assert json.loads(policy.canonical_json())["mode"] == "ROLLING"
    assert policy.identity == CandidateLifecyclePolicy().identity
    with pytest.raises(ValueError, match="unsupported"):
        apply_candidate_lifecycle((), None, (), T1, "run", CandidateLifecyclePolicy(promotion_observations=2))


def test_historical_replay_uses_shared_results_and_rejects_noncausal_order():
    snapshots = (
        HistoricalCandidateSnapshot(T1, "one", (row("BTCUSDT", 1),)),
        HistoricalCandidateSnapshot(T1 + timedelta(hours=1), "two", (row("ETHUSDT", 1),)),
    )
    historical = replay_candidate_lifecycle(snapshots)
    direct_one = apply(rows=snapshots[0].final_candidates, run="one")
    direct_two = apply(direct_one.state, snapshots[1].final_candidates,
                       T1 + timedelta(hours=1), "two", direct_one.cursor)
    assert historical == (direct_one, direct_two)
    with pytest.raises(ValueError, match="later than prior"):
        replay_candidate_lifecycle(reversed(snapshots))
    with pytest.raises(ValueError, match="timezone-aware"):
        replay_candidate_lifecycle((HistoricalCandidateSnapshot(T1.replace(tzinfo=None), "x", ()),))


def test_historical_empty_snapshots_match_direct_shared_cursor_results():
    snapshots = (
        HistoricalCandidateSnapshot(T1, "empty-1", ()),
        HistoricalCandidateSnapshot(T1 + timedelta(hours=1), "empty-2", ()),
        HistoricalCandidateSnapshot(
            T1 + timedelta(hours=2), "present-3", (row("BTCUSDT", 1),)
        ),
    )
    historical = replay_candidate_lifecycle(snapshots)
    first = apply(rows=(), when=T1, run="empty-1")
    second = apply((), (), T1 + timedelta(hours=1), "empty-2", first.cursor)
    third = apply(
        (), snapshots[2].final_candidates, T1 + timedelta(hours=2),
        "present-3", second.cursor,
    )
    assert historical == (first, second, third)


def test_v1_state_migrates_duplicate_ledger_and_cycle_without_inventing_membership(tmp_path):
    path = tmp_path / "state.json"
    entry = {"record_type": "PAPER_ENTRY", "signal_id": "same",
             "signal_candle_timestamp": T1.isoformat(), "extra": "preserved"}
    path.write_text(json.dumps({
        "version": 1, "emitted_signal_ids": ["same"], "paper_entries": [entry],
        "last_completed_cycle": {"cycle_id": "old"},
        "last_successful_scan_run_id": "old-run",
    }))
    state = PaperScannerStateStore(path).load()
    assert state.emitted_signal_ids == ["same"] and state.paper_entries == [entry]
    assert state.last_completed_cycle == {"cycle_id": "old"}
    assert state.last_successful_scan_run_id == "old-run"
    assert state.candidate_lifecycle == []
    assert state.lifecycle_cursor is None
    PaperScannerStateStore(path).save(state)
    assert json.loads(path.read_text())["version"] == 2


def test_v2_restart_reload_preserves_lifecycle_and_policy_identity(tmp_path):
    path = tmp_path / "state.json"
    applied = apply(rows=[row("BTCUSDT", 1, 4)])
    original = PaperScannerState(
        candidate_lifecycle=list(applied.state), lifecycle_cursor=applied.cursor
    )
    store = PaperScannerStateStore(path)
    store.save(original)
    reloaded = store.load()
    assert reloaded.candidate_lifecycle == list(applied.state)
    assert reloaded.lifecycle_cursor == applied.cursor
    assert reloaded.lifecycle_policy.identity == original.lifecycle_policy.identity


@pytest.mark.parametrize("corruption", ["missing", "earlier", "identity"])
def test_v2_rejects_cursor_membership_disagreement(tmp_path, corruption):
    path = tmp_path / "state.json"
    applied = apply(rows=[row("BTCUSDT", 1, 4)], when=T1 + timedelta(hours=1))
    store = PaperScannerStateStore(path)
    store.save(PaperScannerState(
        candidate_lifecycle=list(applied.state), lifecycle_cursor=applied.cursor
    ))
    value = json.loads(path.read_text())
    if corruption == "missing":
        value["lifecycle_cursor"] = None
    elif corruption == "earlier":
        value["lifecycle_cursor"]["decision_timestamp"] = (
            T1 - timedelta(hours=1)
        ).isoformat()
    else:
        value["lifecycle_cursor"]["snapshot_identity"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(PaperScannerStateError, match="corrupt"):
        store.load()


@pytest.mark.parametrize("payload", ["not-json", '{"version":99}'])
def test_corrupt_and_unsupported_state_fail_closed(tmp_path, payload):
    path = tmp_path / "state.json"
    path.write_text(payload)
    with pytest.raises(PaperScannerStateError):
        PaperScannerStateStore(path).load()


def test_active_rows_are_exact_task10_rows_not_reconstructed():
    rows = [row("ETHUSDT", 2, 6), row("BTCUSDT", 1, 3)]
    expected = sorted(rows, key=lambda item: item["final_rank"])
    assert list(apply(rows=rows).active_rows) == expected
