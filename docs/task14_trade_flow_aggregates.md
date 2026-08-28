# Task 14: trade-flow aggregates

## Architecture

```text
Binance aggTrades / raw trades
  -> bounded archive processing
  -> content-addressed 1m aggregate cache
  -> FeatureDataResource
  -> FeatureRegistry
  -> Prepared research values
  -> strategy-aligned research and trade rows
```

`TradeAggregateStore` reads one immutable archive, validates it, reduces it and
releases its event frame before opening the next archive. Its disposable files
live at `cache/trade_aggregates/<agg_trades|trades>/<symbol>/1m/<identity>.parquet`
with a JSON manifest. Writes use a validated temporary Parquet followed by an
atomic rename. Identity includes dataset, market, symbol, source fingerprint,
adapter contract, interval, threshold and `TRADE_AGGREGATE_SCHEMA_VERSION`.
Consequently changing one daily archive invalidates only its partition.

## Semantics

Each bucket is `[period_start, period_end)` and is available only at
`period_end`. An event at exactly 12:00 belongs to 12:00--12:01, whereas an
event at 11:59:59.999 belongs to 11:59--12:00. Both millisecond and microsecond
epochs use `timestamp_series`; no event-time truncation or tick rule is used.

`is_buyer_maker=false` is an aggressive buy and `true` is an aggressive sell.
Base and quote buy/sell volume, their deltas, `sum(price*quantity)`, and VWAP are
kept in explicit units. For aggTrades, `source_event_count` counts aggregate
rows while `underlying_trade_count` sums `last_trade_id-first_trade_id+1`. For
raw trades both counts are the individual row count. An aggTrade's median
source-event size is **not** an underlying individual-trade median.

The optional positive quote threshold classifies source events. It therefore
means individual large trades for `TRADES`, but large aggregate events for
`AGG_TRADES`; null disables these metrics. It is never described as a universal
whale threshold.

Additive 5m, 15m and 1h windows sum completed 1m buckets. Trade intensity is
underlying trades per minute. Intensity change is current 5m intensity divided
by the immediately prior, non-overlapping 60m intensity minus one; missing or
zero reference is NaN. `cvd_utc_day` cumulatively sums base delta from 00:00 UTC
through the latest completed bucket, and `cvd_1h` is a trailing 60-minute sum.

Covered empty minutes contain zero counts and volumes. Minutes outside archive
coverage remain unavailable/NaN and have `trade_source_covered=false`; absence
is never fabricated as zero. Partial coverage is visible in data quality, while
no requested source coverage is an error. `AGG_TRADES` and `TRADES` never fall
back to each other.

Cold runs build aggregate partitions and then L2/L3. Warm runs validate compact
manifests and read aggregate Parquet, without reopening raw event archives.
Only the compact aggregate resource enters `trade_flow_context`. Raw IDs,
buyer-maker arrays and raw event tables cannot enter PreparedBacktestFrame,
simulator state or reporting, which also preserves no-look-ahead behavior.
