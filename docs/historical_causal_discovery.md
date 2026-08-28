# Historical causal discovery (Task 2)

Historical discovery answers: **at this exact decision instant, using only data
already available, which explicitly supplied symbols would have been shortlisted?**
`DiscoveryDecisionTime` rejects naive datetimes and normalizes aware values to UTC.
Every selected observation must satisfy both `period_end <= decision_time` and
`available_at <= decision_time`. Equality is intentional: a candle that becomes
available exactly at the cutoff can be selected. A later observation cannot be used,
even when its `period_start` precedes the cutoff.
The Data Lake applies this cutoff before logical-row de-duplication, so a correction
published after the decision cannot hide the candle version that was visible then.

## Completed-day contract

The Task 2 baseline loads `DatasetKind.KLINES` at Binance interval `1d` through a
`DataRequest` and `MarketDataStore`. For every requested symbol it selects the latest
fully completed, causally available candle and computes:

* range percent: `(high - low) / close * 100`;
* price-change percent: `(close - open) / open * 100`;
* absolute price-change percent; and
* quote volume.

Invalid, non-finite, or nonsensical OHLC/volume values reject the symbol. Missing
source coverage, missing schema, and absence of a usable completed candle are also
explicit rejected snapshot states rather than silently ranked symbols. No spread is
inferred from OHLC.
The request covers all possible Binance history from the Unix epoch through the
decision time; there is no hidden freshness window that could discard an older but
still latest available candle.

Eligible rows use this deterministic completed-day ranking:

1. completed-day range percent, descending;
2. absolute completed-day price change percent, descending;
3. completed-day quote volume, descending; and
4. symbol, ascending.

The symbol key is the final deterministic tie-breaker. The caller-supplied symbol
collection defines the historical universe; replay never consults today's live
Binance universe.

## Live and historical discovery are intentionally different

Live Task 1 uses rolling exchange-wide 24-hour ticker data plus current book spread
and exchange metadata. Task 2 initially uses the latest causally completed 1D Data
Lake observation. These inputs and rankings are not numerically equivalent, and the
historical contract does not claim that they are. Historical replay calls neither
`exchangeInfo`, `/ticker/24hr`, `/ticker/bookTicker`, nor any other live discovery
endpoint.

## Provenance and limitations

Each symbol retains its exact `DataRequest` and the canonical `SourceSignature`
returned by `MarketDataStore.source_signature()`. Snapshot rows carry that signature's
canonical cache identity. Thus discovery reuses the Data Lake partition fingerprints
and provenance contract rather than creating a scanner-specific cache or source hash.

Task 2 does not reconstruct rolling 24-hour values, hourly aggregates, or historical
book spreads. It does not acquire strategy candles or apply trading indicators.
Official immutable historical listing metadata is not currently a Data Lake contract,
so listing-age enforcement is explicitly disabled and requesting it fails. The first
candle observed is deliberately **not** treated as an official listing date.
