# Phase 7D: Data Lake simulation hotspot report

## Scope and reference contract

Phase 7D is profiling-only. No simulator, preparation, feature, or trading-rule
code was changed. The intended reference capture is BTCUSDT, 4-hour strategy
candles, 1-minute intrabars, and the half-open period 2024-01-01 through
2026-08-01.

The required parity gate is:

| Measure | Required value |
| --- | ---: |
| Strategy rows | 5,658 |
| Intrabar rows | 1,357,920 |
| Trades | 504 |
| Trade fingerprint | `83cfd17605b544554aed57ec1c851854a327116c2bb75b3e2500c9cd4ebce0f5` |
| Warm simulation baseline | approximately 9.18 seconds |

## Capture status

A valid hotspot timing table cannot be produced from this checkout because the
reference market-data archive and its warmed feature cache are not present in
the execution environment. The repository contains the smoke configuration,
but no BTCUSDT kline archive or prepared cache. Consequently, running the
simulation profiler cannot reach (or verify) the row counts, trade count, and
fingerprint above.

No timings have been estimated, extrapolated from tests, or invented. In
particular, unit-test or synthetic-data timings would not be reported as BTC
reference measurements: position lifetime and enabled exit-management features
determine both the intrabar call counts and the relative cost of exit helpers.

## Reproducible simulation-only capture

On a machine containing the validated archive, first warm the feature cache,
then run:

```bash
python tools/data_lake_profile.py \
  --config config/data_lake/BTCUSDT_4H_SMOKE.json \
  --raw-root <validated-raw-root> \
  --cache-root <warmed-cache-root> \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2026-08-01 \
  --top 60 \
  --output <capture.json>
```

The profiler constructs data and the engine before enabling `cProfile`; only
`DataLakeProductionBacktestEngine.run()` is inside the measured region. Reject
the capture unless all four parity-gate values match. Keep the approximately
9.18-second uninstrumented warm benchmark as the percentage denominator;
`cProfile` adds observer overhead, and cumulative percentages may overlap and
therefore must not be summed.

## Candidate call paths to classify after capture

These are inspection targets, **not measured hotspots**. They are listed to
make review of the reference capture deterministic without claiming results
that this environment cannot support.

| Call path | Potential expense | Semantics-preserving optimization candidate | Complexity / risk | Benefit hypothesis |
| --- | --- | --- | --- | --- |
| `run` → `_update_positions_to_strategy_index` → `_scan_position_exit` | Executes once per strategy candle while a position is active and sets up each intrabar window. Timestamp normalization and completeness checks are repeated. | Cache immutable interval/end-bound values or use already-normalized timestamp scalars, only if a reference profile attributes meaningful self time here. | Medium; timestamp boundary changes could introduce look-ahead or omit a bar. | Low to medium unless timestamp construction is a leading self-time entry. |
| `_scan_position_exit` → `fast_window` / `gap_pairs` / `rows` | Traverses every 1-minute candle while a trade is open. Even array-backed iteration pays Python generator and tuple-unpacking overhead per bar. | Consider a semantics-identical indexed loop or compiled scan only after separately validating every exit mode and same-bar policy. | High; this is the causal execution boundary. | Potentially high if iteration machinery has material self time. |
| `_scan_position_exit` → `_maybe_timeout_position_at` | Called for every visited intrabar even when timeout is disabled; enabled timeouts also construct pandas timestamps and timedeltas. | Hoist immutable timeout state per position and retain exactly the current comparison/fill ordering. | Medium to high; timeout ordering relative to bar exits is semantic. | Low when disabled; potentially medium if pandas conversion dominates. |
| `_scan_position_exit` → `_maybe_exit_bar` and exit-management helpers | Applies TP/SL ambiguity, break-even, partial stop/profit, R-step trailing, ATR checkpoint extension, and trailing logic per visited bar. | Specialize dispatch by the position's immutable feature flags while keeping the current helper order and fill rules. | Very high; the branch order encodes trading behavior. | Potentially high, but only if cumulative and self timings identify repeated disabled-feature dispatch as meaningful. |
| `run` → `_record_active_telemetry` / `_record_pair_telemetry` | Materializes telemetry dictionaries and many Python attributes while trades are active or opened. | Delay only representation work that is proven not to feed execution, preserving the exact recorded values and timestamps. | Medium; telemetry is an output contract and may be used by reports. | Medium if dictionary construction appears prominently by self time. |
| `run` → `results_frame` → `_build_result_row` | Builds a very wide result dictionary/DataFrame for closed trades and research fields. | Predeclare columns or batch columnar values without changing formatting, nulls, or row order. | Low to medium; fingerprint columns and report schema must remain identical. | Usually low at 504 trades unless pandas construction is unexpectedly prominent. |

## Required measured table

Once a valid capture exists, replace this section with rows for functions that
have meaningful cumulative **or** self time. Each row must include function and
caller path, cumulative seconds, self seconds, total/primitive call counts,
percentage of 9.18 seconds, expense explanation, proposed optimization,
complexity/risk, and likely benefit. Until the parity-gated capture can be run,
those numeric fields are intentionally not asserted.

Any later optimization must separately prove unchanged candle/entry timing,
no-look-ahead behavior, sizing, TP/SL and same-bar ambiguity, break-even,
trailing, partial exits, timeout, fees/slippage, and shared-equity semantics.
