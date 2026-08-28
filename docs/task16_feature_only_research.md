# Task 16: feature-only research

## Two deliberately separate paths

The original authoritative path remains Binance archives → `MarketDataStore` →
feature registry → `PreparedBacktestFrame` → strategy → simulator → completed
trades. Task 16 does not change its trading or execution behavior. At the end of
that path, the artifact reporter serializes the *existing* completed trades and
prepared frame.

The new path is run directory → `ResearchQueryService` → DuckDB → compact result.
It has no imports or references to the market-data store, feature registry,
prepared cache, runner, strategy, or simulator. Query specifications therefore
cannot invalidate L1/L2/L3 or the execution fingerprint. Raw data may be deleted
after the completed run.

## Versioned artifact

Contract `feature_research_v1`, schema version `1`, publishes atomically as:

```text
run_dir/
  trade_list.csv
  run_manifest.json
  research/
    trades.parquet
    feature_context.parquet
    research_manifest.json
```

Parquet is authoritative for Task-16 queries; CSV remains the human-readable
report. Temporary Parquets are row-count validated before atomic rename. The
research manifest records request intervals, prepared and feature identities,
counts, fingerprint, write duration, and artifact sizes. A small pointer is
added to the existing run manifest. Missing legacy artifacts fail with `run does
not contain Task-16 research artifacts`; they are never reconstructed.

`trades.parquet` preserves completed-trade execution outcomes and causal signal
metadata without recalculating an outcome. `feature_context.parquet` contains
one row per prepared strategy row. It includes the prepared scalar arrays,
research-block scalar arrays, each block's `<block>_feature_available_at`, and
deterministically flattened momentum mappings (`momentum_return_4h`, etc.). A
duplicate flattened name is an error rather than an `_x`/`_y` rename. It contains
neither raw provider events nor intrabar execution arrays.

## Exact causal relationship and validation

The only trade/context join is:

```sql
trades.research_signal_index = feature_context.strategy_index
```

No timestamp approximation, as-of join, forward fill, or entry-time matching is
used. Every open validates an exact row count and unique contiguous index,
monotonic candle opens, valid decision availability, exact signal-open and
availability timestamps, and equality of every non-null comparable field shared
by a trade and its context. Missing rows or causal/parity mismatches are explicit
corruption errors. There is no fallback and no raw-data regeneration.

## Query contract

Any persisted scalar can be a direct dimension. `directional_di` and
`opposing_di` are descriptive trade projections based on `side` and neutral
`plus_di`/`minus_di`; they are not strategy inputs. Structured filters support
comparisons plus `IS NULL` and `IS NOT NULL`; `year` projects from entry time.
Unavailable requested columns produce an explicit error.

Default outcomes use completed `pair_net_r` and `pair_net_pnl`:

* win: R > 0; loss: R < 0; breakeven: R = 0;
* win rate: wins / trades;
* net/average R and net/average PnL.

Numeric bucket boundaries are strictly increasing and use left-closed,
right-open intervals: `[0,5)`, `[5,10)`, and so on. Values below the first edge
and at/above the last edge receive explicit open-ended buckets. SQL NULL and NaN
receive `MISSING`; direct dimensions also retain a visible `MISSING` group.
Consequently an unfiltered grouping's trade counts sum to all completed trades.
Coverage/source fields remain ordinary queryable dimensions.

DuckDB scans and joins the two Parquets and materializes only the final grouped
DataFrame. It does not use `pandas.read_parquet`, a pandas merge, or a pandas
group-by.

## API and CLI examples

```python
from crypto_strategy_lab.feature_research import ResearchQueryService

with ResearchQueryService(run_dir) as research:
    result = research.query({
        "dimensions": [
            {"column": "directional_di", "alias": "di_bucket",
             "boundaries": [0, 5, 10, 15, 20, 25, 30]},
            {"column": "price_oi_state"},
        ],
        "filters": [{"column": "side", "operator": "=", "value": "LONG"}],
    })
```

```console
python tools/research_query.py --run-dir runs/BTCUSDT_4h_... --spec query.json
python tools/research_query.py --run-dir runs/BTCUSDT_4h_... --spec query.json --output result.csv
```

Only a run directory, query JSON, and optional output are accepted—never a raw
root, cache root, or strategy configuration. This service boundary is intended
for later exposure by **Task 17 — Reporting, run provenance, and MCP
modernization**; Task 16 does not modify MCP or reporting provenance broadly.
