"""Deterministic candidate membership policy, transitions, and replay adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


class LifecycleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REMOVED = "REMOVED"


class TransitionType(str, Enum):
    ACTIVATED = "ACTIVATED"
    REMOVED = "REMOVED"
    RANK_CHANGED = "RANK_CHANGED"


@dataclass(frozen=True, slots=True)
class CandidateLifecyclePolicy:
    """Versioned policy contract.  Only this Task 11 v1 contract is executable."""

    contract: str = "candidate-lifecycle"
    version: int = 1
    mode: str = "ROLLING"
    refresh: str = "EVERY_COMPLETED_SCAN"
    promotion_observations: int = 1
    removal_missing_observations: int = 1
    removal_cooldown: str = "NONE"
    rank_hysteresis: str = "NONE"
    active_position_retention: str = "NOT_IMPLEMENTED"

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def identity(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def validate_executable(self) -> None:
        if self != CandidateLifecyclePolicy():
            raise ValueError("unsupported candidate lifecycle policy")


DEFAULT_LIFECYCLE_POLICY = CandidateLifecyclePolicy()


@dataclass(frozen=True, slots=True)
class CandidateLifecycleRecord:
    symbol: str
    status: LifecycleStatus
    first_seen_decision_timestamp: str
    last_seen_decision_timestamp: str
    activated_timestamp: str
    removed_timestamp: str | None
    discovery_rank: int | None
    final_rank: int
    previous_final_rank: int | None
    last_scanner_run_id: str

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateLifecycleRecord":
        record = cls(
            symbol=str(value["symbol"]),
            status=LifecycleStatus(value["status"]),
            first_seen_decision_timestamp=str(value["first_seen_decision_timestamp"]),
            last_seen_decision_timestamp=str(value["last_seen_decision_timestamp"]),
            activated_timestamp=str(value["activated_timestamp"]),
            removed_timestamp=value.get("removed_timestamp"),
            discovery_rank=_rank(value.get("discovery_rank"), optional=True),
            final_rank=_rank(value["final_rank"]),
            previous_final_rank=_rank(value.get("previous_final_rank"), optional=True),
            last_scanner_run_id=str(value["last_scanner_run_id"]),
        )
        if not record.symbol or not record.last_scanner_run_id:
            raise ValueError("candidate lifecycle identity is invalid")
        first = _parse_timestamp(record.first_seen_decision_timestamp)
        last = _parse_timestamp(record.last_seen_decision_timestamp)
        activated = _parse_timestamp(record.activated_timestamp)
        removed = (
            _parse_timestamp(record.removed_timestamp)
            if record.removed_timestamp is not None else None
        )
        if last < first or activated < first:
            raise ValueError("candidate lifecycle timestamps are inconsistent")
        if record.status is LifecycleStatus.ACTIVE and removed is not None:
            raise ValueError("active candidate cannot have a removed timestamp")
        if record.status is LifecycleStatus.REMOVED and (
            removed is None or removed < last
        ):
            raise ValueError("removed candidate timestamp is invalid")
        return record


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    transition: TransitionType
    symbol: str
    decision_timestamp: str
    scanner_run_id: str
    previous_final_rank: int | None
    final_rank: int | None
    previous_discovery_rank: int | None
    discovery_rank: int | None


@dataclass(frozen=True, slots=True)
class LifecycleCursor:
    """Durable identity of the latest successfully applied completed snapshot."""

    decision_timestamp: str
    scanner_run_id: str
    snapshot_identity: str

    def __post_init__(self) -> None:
        _parse_timestamp(self.decision_timestamp)
        if not self.scanner_run_id or not self.snapshot_identity.startswith("sha256:"):
            raise ValueError("candidate lifecycle cursor is invalid")

    def serializable(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LifecycleCursor":
        return cls(
            str(value["decision_timestamp"]),
            str(value["scanner_run_id"]),
            str(value["snapshot_identity"]),
        )


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    state: tuple[CandidateLifecycleRecord, ...]
    cursor: LifecycleCursor
    transitions: tuple[CandidateTransition, ...]
    active_rows: tuple[dict[str, Any], ...]


def apply_candidate_lifecycle(
    previous_state: Sequence[CandidateLifecycleRecord],
    previous_cursor: LifecycleCursor | None,
    current_snapshot: Iterable[Mapping[str, Any]],
    decision_timestamp: datetime,
    scanner_run_id: str,
    policy: CandidateLifecyclePolicy = DEFAULT_LIFECYCLE_POLICY,
) -> LifecycleResult:
    """Pure v1 transition: snapshot membership is active membership."""
    policy.validate_executable()
    decision_time = _aware(decision_timestamp)
    decision = decision_time.isoformat()
    if not scanner_run_id:
        raise ValueError("scanner_run_id is required")
    previous = {record.symbol: record for record in previous_state}
    if len(previous) != len(previous_state):
        raise ValueError("previous lifecycle symbols must be unique")
    rows = [dict(row) for row in current_snapshot]
    rows.sort(key=lambda row: (_rank(row["final_rank"]), str(row["symbol"]).upper()))
    symbols = [str(row["symbol"]).upper() for row in rows]
    if len(symbols) != len(set(symbols)):
        raise ValueError("candidate snapshot symbols must be unique")

    snapshot_identity = _snapshot_identity(rows)
    if previous_cursor is not None:
        authoritative_time = _parse_timestamp(previous_cursor.decision_timestamp)
    else:
        authoritative_time = None
    if authoritative_time is not None and decision_time <= authoritative_time:
        if (
            decision_time == authoritative_time
            and previous_cursor.scanner_run_id == scanner_run_id
            and previous_cursor.snapshot_identity == snapshot_identity
        ):
            return LifecycleResult(previous_state, previous_cursor, (), tuple(rows))
        raise ValueError(
            "candidate lifecycle decision must be later than prior lifecycle state"
        )
    cursor = LifecycleCursor(decision, scanner_run_id, snapshot_identity)

    records: dict[str, CandidateLifecycleRecord] = dict(previous)
    transitions: list[CandidateTransition] = []
    for row, symbol in zip(rows, symbols):
        final_rank = _rank(row["final_rank"])
        discovery_rank = _rank(row.get("discovery_rank"), optional=True)
        old = previous.get(symbol)
        activated = old is None or old.status is LifecycleStatus.REMOVED
        if activated:
            record = CandidateLifecycleRecord(
                symbol, LifecycleStatus.ACTIVE,
                old.first_seen_decision_timestamp if old else decision,
                decision, decision, None,
                discovery_rank, final_rank, old.final_rank if old else None, scanner_run_id,
            )
            transitions.append(CandidateTransition(
                TransitionType.ACTIVATED, symbol, decision, scanner_run_id,
                old.final_rank if old else None, final_rank,
                old.discovery_rank if old else None, discovery_rank,
            ))
        else:
            rank_changed = (old.final_rank, old.discovery_rank) != (final_rank, discovery_rank)
            record = CandidateLifecycleRecord(
                symbol, LifecycleStatus.ACTIVE, old.first_seen_decision_timestamp,
                decision, old.activated_timestamp, None, discovery_rank, final_rank,
                old.final_rank if rank_changed else old.previous_final_rank, scanner_run_id,
            )
            if rank_changed:
                transitions.append(CandidateTransition(
                    TransitionType.RANK_CHANGED, symbol, decision, scanner_run_id,
                    old.final_rank, final_rank,
                    old.discovery_rank, discovery_rank,
                ))
        records[symbol] = record

    current = set(symbols)
    for symbol in sorted(previous):
        old = previous[symbol]
        if old.status is LifecycleStatus.ACTIVE and symbol not in current:
            records[symbol] = CandidateLifecycleRecord(
                symbol, LifecycleStatus.REMOVED, old.first_seen_decision_timestamp,
                old.last_seen_decision_timestamp, old.activated_timestamp, decision,
                old.discovery_rank, old.final_rank, old.previous_final_rank, scanner_run_id,
            )
            transitions.append(CandidateTransition(
                TransitionType.REMOVED, symbol, decision, scanner_run_id,
                old.final_rank, None,
                old.discovery_rank, None,
            ))
    transitions.sort(key=lambda item: (item.symbol, item.transition.value))
    return LifecycleResult(
        tuple(records[symbol] for symbol in sorted(records)),
        cursor,
        tuple(transitions),
        tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class HistoricalCandidateSnapshot:
    decision_timestamp: datetime
    scanner_run_id: str
    final_candidates: tuple[Mapping[str, Any], ...]


def replay_candidate_lifecycle(
    snapshots: Iterable[HistoricalCandidateSnapshot],
    policy: CandidateLifecyclePolicy = DEFAULT_LIFECYCLE_POLICY,
) -> tuple[LifecycleResult, ...]:
    """Causal Historical adapter over the exact Paper transition function."""
    state: tuple[CandidateLifecycleRecord, ...] = ()
    cursor: LifecycleCursor | None = None
    results: list[LifecycleResult] = []
    for snapshot in snapshots:
        current_time = _aware(snapshot.decision_timestamp)
        result = apply_candidate_lifecycle(
            state, cursor, snapshot.final_candidates, current_time,
            snapshot.scanner_run_id, policy
        )
        results.append(result)
        state, cursor = result.state, result.cursor
    return tuple(results)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("decision timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _snapshot_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    membership = [
        {
            "symbol": str(row["symbol"]).upper(),
            "final_rank": _rank(row["final_rank"]),
            "discovery_rank": _rank(row.get("discovery_rank"), optional=True),
        }
        for row in rows
    ]
    canonical = json.dumps(membership, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _rank(value: Any, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError("candidate rank must be an integer")
    rank = int(value)
    if rank < 1 or rank != value:
        raise ValueError("candidate rank must be a positive integer")
    return rank
