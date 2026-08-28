# Task 12: data quality and causality

## Data-quality architecture

```text
archive/catalog
    ↓
canonical data
    ↓
DatasetValidationContract
    ↓
DataQualityReport
    ↓
validated feature input
```

`crypto_strategy_lab.data.quality` owns deterministic typed reports, dataset
contracts, domain checks, coverage checks, and a disposable JSON cache. `OK`
means complete and valid. `WARN` means a non-fatal limitation (including an
optional partial source or identical archive overlap), `MISSING` means an
optional source was unavailable, and `ERROR` means required data is missing or
invalid. Required errors abort before simulation; warning metadata never enters
trading policy.

Fixed-cadence contracts build an expected grid and summarize leading, internal,
and trailing gaps rather than emitting one issue per timestamp. Event contracts
(funding, aggTrades, and raw trades) validate ordering, logical keys,
availability, and values without inventing bar gaps. Candle availability cannot
precede `period_end`; event availability cannot precede `event_time`.

Overlap must be inspected before canonical last-source-wins resolution.
`classify_archive_overlap` reports identical rows as warnings and conflicting
values at one logical key as errors; raw archives remain immutable.

Quality cache keys contain validator contract version, canonical source identity,
dataset/interval, requiredness, symbol, and request bounds. JSON writes are
atomic and corrupt entries miss. The quality key is deliberately absent from L2
and L3 identities, so validator-only changes do not invalidate trading output.

Native v3 configuration is fail-closed (`intrabar_missing_policy=ERROR`). A
lower-resolution execution fallback exists only when explicitly configured by a
compatibility consumer. Explicit `agg_trade_flow` requests fail if aggTrades are
absent. Auto-attached futures contexts remain optional, but their absence is a
typed `MISSING` dataset report; no zero or neutral frame is synthesized.

`BacktestDataBundle.data_quality` carries the completed report and
`ResearchRunResult.data_quality` surfaces the same object to reporters. The CSV
manifest reporter writes both the manifest entry and `data_quality.json` without
rescanning raw data. The validator CLI is a presentation layer over the same
library validator, and benchmark records contain compact status fields.

## Causality-test architecture

```text
source fixtures
    ↓
FeatureRegistry.execute(target)
    ↓
choose cutoff by output available_at
    ↓
mutate only source rows available after cutoff
    ↓
recreate registry and recompute with cache=None
    ↓
compare every already-available target output
```

`tests/feature_causality_harness.py` defines `CausalityCase` and the single
`assert_future_mutation_invariant` experiment. It compares timestamps,
`available_at`, and all declared outputs with pandas' NaN/None-aware equality.
Each material source has an independent mutator, and a fresh registry plus
disabled cache prevents false passes. Dependency features are exercised through
the registry graph rather than direct helper calls.

A future provider opts in by adding a small case containing its registry
factory, request, canonical fixture frames, parameters, and one future-only
mutator per material source. Structural-regime cases inject their benchmark as
fixture context and mutate benchmark observations after the cutoff. Existing
provider-specific tests remain valuable for pivot confirmation, funding age,
daily-state availability, and trailing-window semantics.

Order-book dataset enum values remain architectural placeholders. Task 12 adds
neither ingestion nor fake quality/causality cases because no canonical adapter
or provider exists; a future provider can use these contracts once that source
path is real.
