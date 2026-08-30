# Task 11 — candidate refresh and lifecycle

## Declared policy contract

`CandidateLifecyclePolicy` is a canonical-JSON, SHA-256-identified contract. The
only executable contract in v1 is `candidate-lifecycle` version 1: `ROLLING`,
refreshed on every completed scan, promotion after one qualifying observation,
removal after one missing observation, no removal cooldown, no rank hysteresis,
and no active-position retention. The production scheduler remains the Task 10
configurable interval (one hour is recommended); cadence is not lifecycle state.
Unsupported policy combinations fail closed rather than silently acquiring
speculative semantics.

## Transition model

The pure `apply_candidate_lifecycle` boundary accepts prior records, the complete
current final-candidate snapshot, an explicit timezone-aware decision timestamp,
an explicit scanner run ID, and the policy. It returns new records, ordered
`ACTIVATED`, `REMOVED`, or `RANK_CHANGED` facts, and active rows. Symbols and ties
are ordered deterministically. An unchanged active symbol emits no transition.
Records retain first/last-seen and activation/removal times, latest discovery and
final rank, previous final rank, status, and originating run.

The shared boundary rejects decisions older than its latest prior lifecycle
timestamp. Equal timestamps are also rejected unless the run ID, active
membership, and both ranks prove an exact replay of the same scan; that replay is
an explicit idempotent no-op. Reactivation preserves the symbol's original
first-seen timestamp while updating its latest activation timestamp. Rank-change
transitions and audit details carry both discovery and final before/after ranks.

Under v1, active rows are the current snapshot rows sorted by the existing final
rank (with symbol only as a deterministic tie-break), so the native Task 10
strategy evaluator receives the same rows and signal identity/duplicate behavior
is unchanged.

## Atomic persistence and migration

Paper state schema v2 extends the single Task 10 atomic JSON document with the
policy identity plus canonical config and all candidate lifecycle records. A
lifecycle snapshot is atomically installed before evaluation, and every later
paper-ledger replacement carries that same lifecycle state. There is no second
membership store. Audit transition rows are written only after the lifecycle
commit; `CANDIDATE_SET_APPLIED` summarizes each applied set.

A valid Task 10 v1 document migrates in memory without changing its emitted-ID
index, paper-entry ledger, last cycle, or last successful scan ID. Candidate
membership starts empty because v1 did not persist a candidate snapshot; the
first fresh completed scan establishes it deterministically. The next save emits
v2. Invalid ledgers, corrupt JSON, unsupported versions or policy identities fail
closed. Existing temp-file, fsync, and replace persistence remains authoritative.

## Historical parity boundary

`replay_candidate_lifecycle` validates strictly increasing, timezone-aware
decision times and unique run IDs, then calls the same
`apply_candidate_lifecycle` function used by Paper for each snapshot. It has no
clock and cannot inspect future snapshots while applying the current one. The
shared function independently enforces the monotonic rule, so Paper cannot move
durable lifecycle time backward either.

## Intentionally deferred

Fixed daily sets, alternate refresh cadence, multi-observation promotion,
delayed removal/cooldown, rank buffers, position pinning, trade exits, operational
monitoring, dashboards, and any broker/live-order path are not implemented.
