# Deep opportunity scoring (Task 4)

Discovery identifies **active symbols**; this research layer ranks which of the
already-acquired symbols exhibit unusually large or interesting movement. It
never chooses LONG/SHORT, trade eligibility, entries, stops, or targets.

## Causal features and models

Every call has a decision time. Candles are clipped first and the latest feature
whose `available_at <= decision_time` is selected. `CoreDirectionalFeatureProvider`
is authoritative for ATR, ATR %, ADX and absolute DI spread/pressure inputs;
`PolicyMarketFeatureProvider` supplies signed causal 24-hour momentum and regime.
Regime is context only and momentum is ranked by absolute magnitude while its
sign is retained. `OpportunityActivityFeatureProvider` v1 adds only 24-bar
realized volatility, recent range %, current-range expansion over the older
preceding 24-bar median (`shift(24).rolling(24)`) using the historical
`1e-9` floor, and the latest/prior 24-bar **quote-volume** ratio. Its effective
warmup is 49 bars, and absent quote volume remains missing.

`legacy_volatility_v1` is explicitly limited to 1h strategy candles and weights
those measures plus reused ATR % by
30/25/20/10/15 percent. It faithfully assigns ordinal `rank/(N-1)` percentiles,
including distinct ranks for ties, after discovery-rank/symbol ordering.
`balanced_activity_v1` uses 20/15/15/10/10/15/15 percent for realized volatility,
ATR %, range, expansion, volume, ADX and DI spread. `momentum_activity_v1` uses
20/15/15/10/15/10/15 percent for the first five, ADX and absolute 24-hour
momentum. Alternatives use average-rank, tie-aware percentiles; final ties use
score descending, discovery rank ascending, then symbol. Missing inputs make a
row explicitly `UNSCORABLE`; none are imputed.

## Evaluation and reports

The default 24-hour evaluator uses candle opens in the half-open interval
`[decision_time, decision_time + 24h)`. It emits no outcome unless every candle
on the strategy interval's fixed grid is present, so truncated and internally
gapped windows cannot bias comparisons. It
reports future high-low range divided by decision close; the larger absolute
high or low excursion; absolute final-close return; and maximum excursion
normalized by decision-time ATR. **Future movement is used only to evaluate a
score after that score has been frozen.** Comparison includes discovery order,
all three models, and the untruncated `no_second_stage` control at top 5/10/20,
with counts, coverage/rejection, means, medians, 1/2 ATR reach rates, discovery
lift, year strata and regime strata. Deterministic writers emit score CSV,
comparison CSV and JSON research content while deliberately not implementing a
Task 7 publication/run manifest.

Limitations: these fixed baselines are neither fitted nor production-promoted;
the legacy formulas preserve 1h semantics, structural regime depends on its
existing provider configuration, insufficient acquired history reduces
coverage, and direction-neutral excursions do not represent executable trades.
