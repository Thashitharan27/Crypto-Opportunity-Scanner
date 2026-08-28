# Bayesian Trade Probability Research

Crypto Strategy Lab attaches a causal empirical-Bayes probability layer to every completed native run.

## What it does

The existing strategy still decides which trades to enter and the simulator still owns all entry, fill, stop, target, break-even, trailing and timeout semantics. Bayesian research runs **after** those decisions and does not change them.

For every historical entry the research layer estimates:

- probability that a LONG trade is profitable;
- probability that a SHORT trade is profitable;
- probability assigned to the side the strategy actually traded;
- shrunk expected net R for LONG and SHORT;
- the amount and breadth of historical evidence supporting each estimate;
- a 90% diagnostic interval and LOW / MEDIUM / HIGH confidence label;
- the strongest positive and negative evidence families;
- optional probability-gate simulations at 55%, 60%, 65% and 70%.

The columns are written into the normal trade artifacts, including `trade_list.csv` and the authoritative trade Parquet, so they are also available through the existing run/MCP analysis path.

## No-lookahead rule

An outcome is allowed to train the Bayesian model only after that trade has exited.

For an entry at time `T`, the model may use a historical trade only when that historical trade's exit time is at or before `T`. A trade that is still open at `T` cannot contribute its eventual win/loss or R to the estimate.

This matters when trades overlap or hold for many candles. Sorting rows only by entry time is not enough; the implementation keeps outcomes pending until their exit event is causally available.

## Model v2

Model identifier: `BAYES_EVIDENCE_V2`.

V1 required one exact coarse context match. That protected against naive indicator multiplication, but it left many trades without enough context and could create unstable high-probability tails when an exact bucket had only a small number of examples.

V2 keeps the causal Beta-Bernoulli shrinkage but factorizes evidence into **bounded evidence families**. Each family learns its own historical lift relative to the side baseline, and correlated observations inside the same family are averaged and capped before families are combined.

The current evidence families are:

1. **Core direction** — market regime, DI direction agreement/counter state, directional-DI bucket, DI spread, DI pressure, DI spread change, plus higher-timeframe DI direction labels when those columns are present.
2. **Trend & volatility** — ADX, ATR %, Bollinger width and Bollinger-width change.
3. **Momentum & mean reversion** — MR state/motion/direction, MR Bollinger location/z-score, MR RSI, distance from mean, RSI, momentum and VWAP distance when present.
4. **Support & resistance** — side-specific trade-location rating, room in direction, near/inside support/resistance, hold/state information and side-specific distance to support/resistance.
5. **Open interest & positioning** — Price/OI state, OI changes, OI z-score, 1h price change and long/short positioning biases.
6. **Funding & basis** — funding bias/rate/24h/change/z-score/extremes, mark-index basis state/rate/z-score and premium z-score.
7. **Taker flow** — buy/sell ratio, 15m/1h taker delta, flow persistence and taker positioning bias.
8. **Microstructure** — optional detailed trade-flow and order-book evidence such as trade delta, intensity, CVD, large-trade shares, spread, L1 imbalance, microprice offset and depth imbalance.

Optional families contribute only when their fields exist on the trade row. Stale or unobserved order-book snapshots and uncovered trade-flow data are ignored rather than converted into artificial zero evidence.

### Why evidence is grouped

Many research fields are correlated. For example, directional DI, DI spread and DI pressure all describe related parts of the same directional state. Funding rate, funding bias and funding z-score are also related.

V2 therefore does **not** multiply their probabilities as if they were independent. Instead:

- each evidence token is shrunk toward the side prior;
- tiny samples receive an additional reliability taper;
- correlated tokens are averaged inside their family;
- each family's log-odds contribution is capped;
- the total combined contribution is capped;
- diagnostic sample size uses median family support, not the sum of correlated observations.

This is intentionally conservative. More indicators should improve information coverage, not manufacture false confidence.

### Hierarchical side prior

LONG and SHORT remain separately scored, but each side baseline is now shrunk toward the all-trade baseline before contextual evidence is added. This prevents a short early streak on one side from becoming an unjustified 80–90% prior.

## Confidence and readiness

`bayes_context_ready` requires both:

- at least 20 historical supporting samples by default; and
- at least 3 evidence families with historical support.

The confidence label also considers evidence breadth and total side history. HIGH confidence therefore requires substantially more support than merely crossing a probability threshold.

The 90% interval is deliberately conservative: its effective sample size is based on median family support rather than adding sample counts across correlated evidence families.

## Main trade columns

Core probability fields:

- `bayes_long_probability`
- `bayes_short_probability`
- `bayes_actual_side_probability`
- `bayes_actual_side_prior_probability`
- `bayes_probability_edge_long_minus_short`
- `bayes_expected_net_r_long`
- `bayes_expected_net_r_short`
- `bayes_long_context_samples`
- `bayes_short_context_samples`
- `bayes_actual_context_samples`
- `bayes_actual_side_confidence`
- `bayes_long_lower_90` / `bayes_long_upper_90`
- `bayes_short_lower_90` / `bayes_short_upper_90`
- `bayes_context_ready`
- `bayes_recommended_direction`
- `bayes_recommendation_matches_actual`

Evidence diagnostics:

- `bayes_long_evidence_families`
- `bayes_short_evidence_families`
- `bayes_actual_evidence_families`
- `bayes_actual_evidence_items`
- `bayes_top_positive_evidence`
- `bayes_top_negative_evidence`
- `bayes_actual_core_direction_log_odds_lift`
- `bayes_actual_trend_volatility_log_odds_lift`
- `bayes_actual_momentum_mean_reversion_log_odds_lift`
- `bayes_actual_support_resistance_log_odds_lift`
- `bayes_actual_oi_positioning_log_odds_lift`
- `bayes_actual_funding_basis_log_odds_lift`
- `bayes_actual_taker_flow_log_odds_lift`
- `bayes_actual_microstructure_log_odds_lift`

Probability-gate research simulations:

- `bayes_take_55_min20`
- `bayes_take_60_min20`
- `bayes_take_65_min20`
- `bayes_take_70_min20`

The `bayes_take_*` fields are **research simulations**, not entry rules. For example, `bayes_take_60_min20=true` means the actual trade would have passed a 60% posterior threshold with the required causal historical sample support and evidence breadth.

## Interpreting LONG vs SHORT

The model estimates LONG and SHORT independently from trades that were actually observed on those sides. A SHORT probability shown beside a LONG trade is useful comparative research evidence, but it is not proof of what would have happened if that exact historical trade had been reversed.

A later experiment can run a dedicated direction-selection simulation after the probability model proves that it separates strong and weak trades out of sample. Probability research remains separate from strategy execution for now.

## Recommended validation workflow

1. Run minimally filtered BTC/ETH baselines over long periods so the model sees broad market states.
2. Compare raw performance with the 55/60/65/70 probability-gated subsets.
3. Review `bayes_actual_evidence_families`, confidence and the 90% interval, not only the point probability.
4. Break results down by `bayes_top_positive_evidence`, `bayes_top_negative_evidence` and the per-family lift columns to see which data actually adds value.
5. Compare Net R / average R as well as probability of profit.
6. Validate the useful threshold on a later out-of-sample period before allowing Bayes to influence actual direction or entry rules.
