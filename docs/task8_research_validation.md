# Task 8 research validation

Task 8 is an aggregation and reporting boundary. It does not run discovery,
rerank candidates, calculate indicators, evaluate entries, calculate trades, or
derive market regimes. Scanner facts come from completed historical Task 7
`OPPORTUNITY_SCAN` manifests and their hash-verified catalog artifacts. Strategy
facts are supplied by existing Strategy Lab research outputs and joined on the
natural key `(decision_timestamp, symbol)`.

The normalization adapter consumes native `entry_time`, `pair_net_r` (falling
back to native `pair_net_pnl`), and `market_regime` trade fields plus Task 4
`OpportunityOutcome` rows. An outcome marks that key as entry-observed; a valid
entry means at least one authoritative trade entry in the half-open interval
from the decision through the explicit upstream evaluation horizon.

## Metric definitions

* **Candidate-to-entry conversion** is the number of authoritative valid-entry
  observations divided by the candidates for which the research output supplies
  a valid-entry observation. Missing research observations are visible through
  the sample counts and are not treated as failed entries.
* **Setup frequency** is the same per-candidate valid-entry frequency, named
  explicitly so that it can be compared across the eligible universe,
  preliminary shortlist, and final candidate stages.
* **Opportunity capture** is direction-neutral: the sum of Task 4's canonical
  `forward_max_abs_excursion_pct` for selected candidates divided by the sum of
  that movement for eligible baseline observations in the same year/regime or
  overall stratum. It is not defined by winning trades.
* **Expectancy** is the arithmetic mean of the authoritative Strategy Lab trade
  expectancy value over completed trades. Task 8 does not derive PnL or trade
  outcomes.
* **Rank correlation** is Spearman's ordinal correlation, calculated with average
  ranks over pairwise-complete observations. Sample count accompanies every
  association; constant or fewer-than-two observations produce a null result.

The immutable configuration contains only Top-K presentation values, a
deterministic rank-bucket size, and a minimum-sample warning level. The warning
never filters observations. Evaluation horizons and movement construction remain
owned by the upstream historical forward-opportunity evaluator.

## Causality and interpretation

The Task 7 loader rejects live scans and scanner facts whose `available_at` is
after the decision boundary. The research join only counts native entries inside
the explicit post-decision horizon; earlier and later trades cannot affect that
key. Future outcomes may only be attached after selection; no outcome column
participates in scanner selection or ranking.

Every output reports sample counts. Win rate is secondary to conversion,
movement, opportunity capture, and expectancy. Top-K rows use `rank <= K`, so a
run with fewer than K candidates reports only ranks actually present. Component
and model correlations are descriptive redundancy/association diagnostics and do
not select a winner or tune production thresholds.
The summary also records deterministic decision timestamps, Task 7 scan run IDs,
scanner source identities, research run/source identities, and horizons.
