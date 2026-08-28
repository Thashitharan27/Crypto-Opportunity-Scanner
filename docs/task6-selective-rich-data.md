# Task 6 — selective rich-data acquisition

Task 6 is a data-readiness boundary, not a feature or trading engine:

```text
FinalCandidateSet
     ↓
explicitly enabled research features
     ↓
deterministic requirement resolver
     ↓
dataset / interval acquisition plan
     ↓
reuse Data Lake catalog + quality
     ↓ missing ranges only
Binance Data Hub public archives
     ↓
Data Lake refresh, revalidation, SourceSignature
     ↓
per-feature READY / DEGRADED / UNAVAILABLE
```

Only `FinalCandidateSet.candidates` is read. Exclusions, discovery rows, and the
preliminary shortlist cannot become requirements. Registry presence is not
enablement: an empty feature set and disabled intrabar interval produce an empty
plan.

| Feature | Required source | Optional source |
| --- | --- | --- |
| `funding_context` | `FUNDING_RATE` | — |
| `futures_positioning` | `FUTURES_METRICS` | 1h `KLINES` |
| `basis_context` | mark + index price klines | premium-index klines |
| `taker_flow_context` | parameterized `KLINES` | — |
| `trade_flow_context` | `AGG_TRADES` **or** `TRADES` | — |
| `order_book_context` | at least ticker **or** depth | second book source |

Funding, basis, and positioning retain their provider warmups. Taker-flow uses
the configured interval (5m by default), and trade-flow uses its configured raw
event family (aggregate trades by default). Strategy-interval klines are reused
from Task 5; optional intrabar klines are acquired only when explicitly set.

Requirements with the same symbol, dataset, and interval are coalesced while
retaining every sorted feature, reason, and that feature's own required/optional
role. Thus one physical source can remain optional for one feature and required
for another. Acquisition checks quality first,
downloads only structured missing ranges with bounded outer concurrency and one
Data Hub worker, then refreshes and revalidates. Structural failure is reported
as `QUALITY_FAILED` and is never treated as a repair request. Raw event frames
are not loaded merely to establish readiness; canonical catalog quality and
`SourceSignature.cache_identity()` remain authoritative.

Both Data Hub `ACQUIRED` and `MISSING` outcomes are revalidated: a missing
monthly archive can coexist with a successful daily fallback. Raw trade-event
readiness validates one existing Data Lake partition at a time, so Task 6 never
concatenates an entire `AGG_TRADES` or `TRADES` request merely to establish
archive readiness.

Missing rich data never rejects or removes a final candidate. It makes a
required feature `UNAVAILABLE`, an optional enrichment `DEGRADED`, or—in the
order-book any-of case—`DEGRADED` when one of the two sources is available.
Current-day/unpublished public archives remain visibly missing; Task 6 does not
add REST or websocket collection.
