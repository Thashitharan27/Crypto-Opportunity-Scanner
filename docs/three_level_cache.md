# Three-level content-addressed cache

Task 9 completes the local persistent hierarchy. Raw Binance archives remain
read-only and are not cache entries.

| Level | Content identity | Persistence | Corruption behavior |
|---|---|---|---|
| L1 canonical | exchange/market/dataset/symbol/interval, raw fingerprint, adapter class, dataset normalizer version, canonical schema version | Parquet + JSON manifest | miss and normalize from raw |
| L2 feature | feature format/name/version/schema, normalized parameters, feature slice, material canonical identities, dependency feature identities | Parquet + JSON metadata | miss, validate, and execute the registered provider |
| L3 prepared run | prepared format/contract, strategy slice, all physically included L2 keys, strategy/structural canonical identities, prepared policy-feature inputs | Parquet + JSON manifest | miss, rebuild normally, validate, atomically replace |

Identities flow downstream; no child lists or cache scans are used. Disposable
old entries can remain unreachable.

## Prepared policy boundary

The frame physically contains market-regime and profile momentum arrays. Thus
L3 directly includes regime method, return lookback/threshold, structural SMA
and slope lookbacks, and the unique profile momentum lookbacks. Feature
parameters enter through L2 keys. Research features (including funding when
available) are physically stored, so their L2 keys enter L3 as well.

L3 intentionally excludes intrabar interval/content and all simulator-only or
non-data settings: take-profit, stop-loss, trailing, break-even, fees,
slippage, partial exits, shared-equity/portfolio execution, output directory,
UI state, and report formatting/version. Intrabar is projected independently
from L1 on every run and validated against the loaded prepared frame.

All manifests are written after their Parquet payload. Missing, mismatched,
incomplete, unreadable, or validation-failing entries are misses. Loaded L3
objects are constructed through `PreparedBacktestFrame`, preserving causality,
alignment, type, and read-only-array validation.

Eviction, garbage collection, remote caches, and compatibility migrations for
obsolete disposable entries remain intentionally deferred.
