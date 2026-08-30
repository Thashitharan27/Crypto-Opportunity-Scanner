# Task 12 — operational hardening

The continuous scanner remains **PAPER / research only**. It has no authenticated
exchange, broker, order-placement, position, leverage, or TP/SL path.

## Policy and retry behaviour

`ScannerOperationalConfig` declares the health path, capped retry backoff,
crash-recovery delay, and `DiskMonitoringConfig`. Existing `api_retry_count` and
`retry_backoff` settings remain compatible. Retries are bounded and deterministic:
connection interruptions, timeouts, URL errors, HTTP 408/429 and temporary
500/502/503/504 responses retry; validation, causality, configuration and local
state failures do not. Backoff doubles up to the configured cap. A valid 429
`Retry-After` is honored up to that same cap. HTTP 418 is deliberately not retried.
Cooperative cancellation interrupts retry and recovery waits.

HTTP 429 additionally uses the explicit positive `rate_limit_min_backoff`, even
when the normal retry base is zero or `Retry-After` is missing/malformed. The
validated retry cap is positive and cannot be lower than this minimum, preventing
rate-limit retries from becoming a zero-delay loop.

The injectable public Binance transport reports endpoint, status,
`Retry-After`, and `X-MBX-USED-WEIGHT-1M` when supplied. Headers are telemetry,
not hard-coded exchange-limit truth; no credentials are used.

## Safety, partial sources, and logs

Stale discovery still stops before lifecycle/strategy evaluation. Stale strategy
rows remain unusable, and completed-candle causality is unchanged. Core discovery
failure fails/retries the cycle; selective candle and rich-data outcomes retain
their existing `AcquisitionState` and `READY/DEGRADED/UNAVAILABLE` meanings.

The append-only JSONL audit retains `timestamp`, `event_type`, `cycle_id`,
`scan_run_id`, `symbol`, `signal_id`, and `detail`. Operational rows add
`severity`, `component`, and a deterministic `fields` object containing numeric
ages, durations, attempts, delays, HTTP status, counts, and health status. Never
put credentials or environment dumps in log details or fields.

## Health and metrics

When `health_path` is configured, a schema-version 1 latest-status JSON document
is atomically replaced. Publication is best-effort: failure is audited and can
never roll back or reset PAPER/lifecycle state.

Top-level fields are `schema_version`, `generated_at`, `runtime_started_at`,
`status`, cycle/run identifiers and status, success/failure timestamps,
`consecutive_failed_cycles`, `last_error_category`, `metrics`, and `disk`.
Statuses are:

* `STARTING`: durable state loaded and validated;
* `HEALTHY`: latest cycle completed without an operational warning;
* `DEGRADED`: valid completion with retry, partial Task 3/6 results, stale
  candidates, or disk warning; cancellation is also reported as degraded but
  does not reset or increment failure history and is not recorded as success;
* `UNHEALTHY`: failed/stale-discovery/stale-only cycle, runtime crash, lifecycle
  rejection/persistence failure, or critical disk state;
* `STOPPED`: scheduler exited.

Metrics include monotonic pipeline, lifecycle, strategy-evaluation, and total
cycle milliseconds; numeric discovery age and stale limits; fresh/stale/active
candidate counts; and retry delay/count. `aggregate_acquisition_metrics` reports
Task 3 requested/state/row/actual-gap counts and Task 6 dataset/readiness counts
directly from the existing enums. Task 6 dataset requirements are deduplicated
by request identity and feature readiness by `(symbol, feature_name)`; conflicting
denormalized rows fail closed.

## Crash and storage boundaries

`run_forever` catches a cycle-scoped unexpected exception, audits it, publishes
`UNHEALTHY`, waits the configured cancellable recovery delay, and permits another
cycle. Constructor-time corrupt/unsupported state or incompatible lifecycle
policy still fails closed. Recovery never deletes signal IDs, entries, lifecycle
state, cursors, Data Lake data, or caches.

Disk observation reports total/free bytes and free percentage for configured
paths, deduplicates volume calls using the nearest existing path's filesystem
device identity, classifies explicit warning/critical thresholds, and samples on
a configured cadence. Cache size walking has a file bound and tolerates entries
that disappear during concurrent cache replacement. It is strictly read-only and
performs no cleanup or repair.

Filesystem capacity remains under each configured path name (including
`disk.cache`), while directory-walk usage is published separately as
`disk.cache_usage`. Cache sizing therefore cannot hide the cache filesystem's
`WARNING` or `CRITICAL` capacity level.

Disk/cache observation is best-effort. An `OSError` produces a structured
`DISK_MONITOR_FAILED` audit event and `disk_monitor_error` health metric; it does
not unwind an otherwise completed PAPER cycle or enter crash recovery. A valid
cycle with unavailable disk telemetry is `DEGRADED`, never falsely `HEALTHY`.

The production PAPER composition automatically supplies the known Paper state,
audit, OPPORTUNITY_SCAN output, raw Data Lake, and cache paths when no explicit
disk mapping is configured. Explicit operator mappings are never overwritten.

## Windows launch and troubleshooting

Double-click `Crypto Opportunity Scanner.vbs` for a hidden-window launch. It sets
the project root as working directory and directly invokes the quoted local
`.venv\Scripts\pythonw.exe`. Use `Debug Launcher.bat` for a visible console; it
uses local `python.exe`, preserves a non-zero exit code, and pauses on failure.

If health is `UNHEALTHY`, inspect the latest structured audit rows, verify core
public-source availability, timestamps and free space, then correct the cause.
Do not delete authoritative state. Health publication failure affects
observability only. Directory-size sampling is bounded (and therefore may be a
lower bound); Binance headers are optional; no external monitoring server is
included.
