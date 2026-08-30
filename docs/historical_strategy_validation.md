# Historical Strategy Validation

A scanner selection is **not** itself a trade. Historical Strategy Validation
connects frozen, completed Historical Single/Range Final Candidates to the
existing Strategy Lab path:

> frozen scanner candidates → Data Lake → ResearchRunner → NativeStrategyPolicy
> → NativeSimulator / native Every Viable Entry sampling → outcome attachment →
> rank and win-rate research summaries

The validator neither repeats discovery nor ranks or scores candidates. It runs
research once per unique symbol over one continuous interval (including the
production-registry warmup), then attaches entries to each candidate's explicit,
half-open `[decision, decision + horizon)` window. The default horizon is `24h`.

## Populations and denominators

`STANDARD_SINGLE_SYMBOL` is the ordinary authoritative simulation for one
symbol, including within-symbol holding/suppression. Runs for different symbols
are independent; this is **not a combined portfolio simulation**.

`EVERY_VIABLE_ENTRY` is the existing overlapping-entry resilience research
population. It preserves native entry/veto and execution semantics while
removing portfolio suppression. Portfolio equity, drawdown and compounded return
are therefore intentionally not reported for it.

Candidate conversion counts candidate windows. Thus one native trade may make
two overlapping candidate windows valid. Unique-trade metrics deduplicate that
trade by `(research run ID, symbol, pair_id)` for Standard or `(research run ID,
symbol, research_sample_id)` for Every Viable Entry. This makes conversion and
unique-trade win rate complementary rather than competing figures.

Completed positive R is a win, negative R a loss, and exactly zero R a neutral.
Resolved win rate is `wins / (wins + losses)`; the separately reported all-
completed win share includes neutrals in its denominator.

Insufficient coverage through the horizon is `UNRESOLVED`, never No Entry or a
loss. Native `END_OF_DATA` trades are censored and likewise unresolved. No
candles or movement outcomes are fabricated.

## Configuration, cancellation, and output

Version 1 applies **one selected strict v3 ResearchRunConfig to every candidate
symbol**. Pair-specific mappings and parameter optimization are out of scope.
StrategyConfig and ExecutionConfig are unchanged; only symbol/time requests and
validation reporting are varied. Existing Data Lake and feature/prepared caches
remain authoritative.

Cancellation is cooperative between symbols: it prevents the next symbol from
starting and preserves completed native runs, but does not terminate a running
simulator unsafely. Unexpected native failure stops validation and no COMPLETE
package is published.

Completed packages live under `output/opportunity_validation/<validation_run_id>`
with candidate outcomes, overall/rank/Top-K/year tables, hashes, row counts,
scanner/config provenance, native run IDs, and a completion marker. Scanner run
directories remain immutable. This feature is historical research only and has
no exchange-key, order, account, leverage, or live-trading path.
