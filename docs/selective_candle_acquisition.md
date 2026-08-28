# Task 3: selective candle acquisition

Task 3 is the boundary **discovery → ranked shortlist → required strategy
candles**. It does not score opportunities or evaluate a strategy. Live Task 1
rows use `preliminary_rank`; historical Task 2 candidates use `rank`. In both
cases ineligible rows are excluded, ties are stabilized by symbol, duplicate
symbols are removed, and `shortlist_size` (default 25) is a hard upper bound.

## Reuse and missing-range planning

Every shortlist symbol becomes the existing half-open `DataRequest` for
`KLINES`, `FUTURES_UM`, and the normalized strategy interval. The Data Lake
catalog is refreshed and its authoritative quality report is checked first.
Complete valid data is loaded and returned as `REUSED`; no backend call occurs.

`DatasetQualityReport.missing_coverage_ranges()` converts only structured
coverage issue codes and fields into typed `[start, end)` ranges. Fixed-grid
issue ranges record the first and last missing timestamps, so one interval is
added to the latter. Catalog edge fields and wholly absent datasets are handled
directly, and touching/overlapping ranges are coalesced. Human-readable issue
messages are never parsed.

Structural errors (invalid OHLC/numerics, duplicates, malformed or off-grid
timestamps, identity mismatch, and conflicting source overlap) are not missing
coverage. Existing corrupt immutable archives therefore produce
`QUALITY_FAILED`; Task 3 does not overwrite or repair them.

## Data Hub boundary

`CandleAcquisitionBackend` is the only acquisition dependency. Tests use an
in-memory fake. `BinanceDataHubBackend` dynamically imports a configurable
module and optional project path, then calls its existing
`download_archive_library`; this repository contains no Binance download or raw
cache implementation. An unresolved package/function raises a clear backend
configuration error. Half-open timestamp gaps are translated to the Data Hub's
inclusive archive dates. The bridge treats reported missing, failed, or
cancelled work as unsuccessful rather than trusting a generic return value.

## Verification, provenance, and execution policy

Downloader success is provisional. Task 3 refreshes the catalog, reruns the
same Data Lake validation, loads through `MarketDataStore`, and obtains the
final `MarketDataStore.source_signature()`. Only then is the state `ACQUIRED`.
Results retain rank, interval, research bounds, attempted ranges, row count,
quality status, and the canonical `SourceSignature` and are always returned in
shortlist rank order.

Symbol calls use a bounded thread pool (`max_workers`, 1–32). The Data Hub
bridge forces one internal worker, preventing multiplied concurrency. Failures
are isolated per symbol. Work is submitted incrementally, with cooperative
cancellation checked before submission, before/in the backend, and after each
future completes. Threads are not killed; unscheduled symbols are deterministically
reported `CANCELLED`/not attempted.

## Known limitations

Task 3 only acquires strategy-timeframe USD-M kline archives. It does not repair
corrupt raw partitions, provide intrabar/rich datasets, score candidates, or
guarantee that Data Hub has published a recent archive. Real Data Hub integration
requires its package/project path to be configured in the deployment; tests do
not require it or network access.
