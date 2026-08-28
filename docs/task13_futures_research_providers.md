# Task 13: compact futures research providers

These fields are **numeric research facts**, not trading signals. They are not consulted by entry,
direction, sizing, TP/SL, or execution code. Every provider first computes on its native source
timeline and only then performs a backward `available_at <= decision_time` join. Partial optional
history therefore yields a full strategy-row result with nullable values before coverage and through
gaps; it is never replaced by zero. Dataset coverage remains the responsibility of the Task 12
quality report (`OK`, `WARN/PARTIAL`, or `MISSING/ERROR` according to selection policy).

## Futures positioning

**Source.** Binance futures-metrics observations (normally 5m), using their publication
`available_at`. `open_interest`, `open_interest_value`, and the top-trader account, top-trader
position, global-account, and taker-volume long/short ratios are the authoritative raw fields.

For each source observation at T, `oi_change_H = OI[T] - OI[P]`, where P is the last observation at
or before T-H; percentage change divides by OI[P]. H is 5m, 1h, or 24h, and absent/zero priors yield
NaN. `oi_zscore_7d` uses a trailing, right-inclusive seven elapsed-day window (default minimum 20
finite observations, population standard deviation); insufficient or zero-variance history is NaN.

`price_change_pct_1h` is deliberately **not** calculated from strategy rows. A distinct auxiliary
`FeatureDataResource(KLINES, "1h", "futures_positioning")` supplies completed 1h prices. The provider
calculates the elapsed 1h price change on that native source and then backward-aligns it to the
strategy decision. If the genuine 1h source is unavailable, `price_change_pct_1h` stays NaN and
`oi_vs_price_state_1h` stays `UNKNOWN` rather than relabeling a 4h strategy-bar return as 1h.
Together the true 1h price change and `oi_change_pct_1h` describe `oi_vs_price_state_1h` as
`PRICE_UP_OI_UP`, `PRICE_UP_OI_DOWN`, `PRICE_DOWN_OI_UP`, `PRICE_DOWN_OI_DOWN`,
`FLAT_OR_MIXED`, or `UNKNOWN`. The numeric inputs always accompany the label.

Pipeline: futures metrics native timeline -> elapsed OI changes/z-score + auxiliary 1h price native
change -> causal strategy alignment. The strategy, metrics and auxiliary 1h source identities plus
OI parameters form the L2 identity.

## Compact taker flow

**Source.** A distinct `FeatureDataResource(KLINES, interval, "taker_flow")`, default 5m. This
prevents the auxiliary research klines from overwriting either strategy or execution klines and
makes interval/role part of source identity. Candles enter only at their canonical `available_at`.

`buy = taker_buy_base_volume`; `sell = volume - buy`; `ratio = buy/sell`; `delta = buy-sell`; and
`delta_pct = delta/volume`. Zero denominators produce NaN. Buy greater than volume beyond the
configured relative tolerance is an integrity error, never silently clamped; only a tiny negative
sell remainder within that tolerance is normalized to zero.

`taker_delta_15m` and `_1h` sum native deltas over elapsed **(T-H, T]** windows. Thus on complete 5m
data a 15m value contains exactly three completed 5m candles, not four boundary-inclusive candles.
Their percentages divide summed delta by summed volume. `flow_acceleration` is current 15m delta
minus the previous completed source observation's 15m delta. `flow_persistence` is calculated with
vectorized trailing-hour sign counts: it is the fraction of completed trailing-hour source intervals
whose delta sign equals the current one-hour aggregate sign (minimum two observations; zero aggregate
is NaN).

Pipeline: auxiliary klines -> native buy/sell and elapsed rolling facts -> causal strategy alignment.
The auxiliary source identity, interval, and tolerance form L2 identity. Warm L2 lookup is resolved
from catalog/source identity before re-materializing the auxiliary 5m frame.

## Funding context

**Source.** Published funding events, not repeated strategy rows. Raw/provenance fields include rate,
rate bps, known interval, source availability, and age. On the event timeline, previous/change use
the prior event; the three-event mean requires three events; 24h sum/count use completed events in
an elapsed **(T-24h, T]** window; and the 7d z-score uses a causal event window (defaults: six samples,
population standard deviation, NaN for insufficient/constant history). Extreme positive/negative
flags compare the z-score with the configurable absolute threshold (default 2.0) and remain nullable
when the z-score itself is unavailable.

`time_to_next_funding` never reads the next archive row. It uses the latest published event and the
funding interval already known at that time. If the current event does not contain an explicit
interval, the interval may be inferred only from that event and a **previously published** event.
When a decision is more than one known interval after the last event, the known cadence is rolled
forward to the next scheduled boundary rather than producing a negative duration. If no interval is
causally knowable, the value is NaN.

Funding source identity, window/minimum, and extreme threshold form L2 identity.

## Basis context

**Sources.** Trade, mark, index, and optional Binance premium-index klines. Mark, index, and premium
retain independent source availability and age. Raw prices produce mark-index, trade-mark, and
trade-index relative bases and bps.

Mark-index derivatives are computed **before strategy sampling**: index is backward-aligned to each
completed mark-price source observation, then `mark_index_basis_change` and the trailing z-score are
calculated on that native reference timeline. Premium change/z-score likewise use the native premium
timeline. Only those completed source-native facts are then aligned to strategy decisions. This means
a 4h strategy cannot turn a native 1h reference change into a mislabeled 4h-sampled change.

Missing premium data stays NaN; mark-index basis is not relabeled as Binance premium data. Each
source is independently checked against future attachment. Source identities and z-score parameters
form L2 identity; exact L2 identities naturally feed prepared/L3 identity. Provider columns are
embedded by the existing research feature path and can be joined to causal trade-entry rows for
generic DI, MR, regime, OI, funding, positioning, and flow grouping.

## Partial coverage and quality

Compact futures datasets are descriptive optional research sources by default. Partial valid source
history remains queryable: the provider emits the full strategy decision timeline with NaN/UNKNOWN
outside available history while Task 12 continues to report `WARN`/partial coverage. Missing data is
never replaced by zeros or neutral synthetic facts. Source-integrity failures (invalid values,
conflicting duplicates, non-causal timestamps, etc.) remain unusable even when the research source is
optional.

The authoritative path quality-checks futures metrics, funding, mark/index/premium, the auxiliary 1h
positioning-price source, and the configured taker-flow kline interval. The Task 12 generic
future-mutation harness also supports `FeatureDataResource` keys so auxiliary research timelines can
be mutated independently with caches disabled.

## Trading boundary

All Task 13 outputs are research/provenance facts. They may be grouped later as, for example,
`DI bucket × OI state`, `MR × OI`, `regime × taker imbalance`, or `DI state × funding`. Task 13 does
not change entry eligibility or turn any descriptive state into a trading signal.
