# Final candidate boundary (Task 5)

The scanner-to-Lab handoff is deliberately narrow:

```text
Discovery
   ↓
Acquisition
   ↓
Optional precomputed opportunity ranking
   ↓
FinalCandidate
   ↓
DataRequest
   ↓
Existing Crypto Strategy Lab strategy/research evaluation
```

**A `FinalCandidate` is permission to evaluate a symbol, not permission to
trade it.** It contains no direction, entry, exit, position, stop, target, or
portfolio semantics and the boundary does not run discovery, downloads,
features, scoring, backtests, or strategy code.

## Selection and integrity

`FinalCandidateBoundaryConfig` defaults to normalized `1h`, no candidate cap,
and no opportunity model. In that default mode, supplied Task 4 output is
ignored, opportunity fields remain `None`, and ordering is discovery rank then
symbol. There is no automatic model promotion.

Callers may explicitly select one exact Task 4 model name/version. Only its
already-computed, scorable rows participate; ordering uses model rank,
discovery rank, then symbol. A score must match the candidate's decision time,
normalized interval, discovery rank, and the canonical Task 3 candle source
identity. There is no discovery-order fallback for a missing or invalid score.
The candidate retains both the actual Task 3 `SourceSignature` and its identity.

The maximum candidate count is applied only after deterministic ordering and
final ranks are assigned from one. Input ordering cannot affect serialized
output. Every omitted discovery row receives a machine-readable exclusion,
including acquisition readiness/source failures, interval and score mismatches,
unscorable or absent scores, and candidate-limit truncation. Ambiguous duplicate
or contradictory contracts fail fast.

## Discovery evidence and handoff

Live rows retain quote volume, signed and absolute 24-hour change, 24-hour
range, spread, and listing age; unavailable reference-period fields are `None`.
Their discovery timestamp is the Task 1 timestamp and their discovery source
identity is `None`. Historical rows use the Task 2 causal decision time, retain
the completed-day period/availability, range, signed/absolute change and quote
volume, and separately preserve the historical 1D discovery source identity.
Historical spread and listing age remain `None` rather than being fabricated.

Only Task 3 `REUSED` and `ACQUIRED` states cross the boundary. The downstream
interface is the existing canonical `DataRequest`, reconstructed from Task 3's
verified symbol, requested bounds, and normalized strategy interval with the
`KLINES` dataset and Binance Futures UM market. `FinalCandidateSet.strategy_requests()`
returns those requests in final candidate order.

