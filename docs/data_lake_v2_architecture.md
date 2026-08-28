# Data Lake v2 architecture

## Purpose

Crypto Strategy Lab is a research and backtesting application. The separate Binance Market Data collector owns the immutable raw archive at `C:\CryptoBots\Binance Market Data`. The lab must never mutate, reorganize, or duplicate that collector as an application concern.

The target pipeline is:

`Binance ZIP archive -> catalog -> canonical cache -> causal alignment -> feature store -> simulator -> research/reporting`

The first migration rule is simple: **only the data layer may touch the Binance archive filesystem**. Strategy code, feature code, the simulator, reports, and the GUI consume typed requests and canonical frames.

## Physical storage

- Raw source: external read-only Binance Market Data directory.
- `cache/catalog/catalog.duckdb`: local archive index.
- `cache/market/.../*.parquet`: normalized immutable archive cache.
- Future `cache/features/.../*.parquet`: derived feature cache.
- `output/` or `outputs/`: run artifacts only.

Caches are disposable and reproducible. Raw Binance files are not.

## Canonical identity

Every source archive is cataloged with at least:

- exchange
- market
- dataset
- symbol
- interval where applicable
- archive frequency (daily/monthly/unknown)
- covered period start/end
- path
- size and mtime
- stat fingerprint

A `DataRequest` asks for market data using those identities plus an inclusive start and exclusive end. It contains no filename.

## Availability and no-look-ahead contract

Every canonical record has a first-class `available_at` timestamp. At decision time `T`, research and strategy code may only use records where:

`available_at <= T`

The convention is:

- regular, mark-price, index-price and premium-index klines become available only when the candle is complete;
- interval aggregates such as futures metrics become available at the end of their reporting interval;
- funding is available at the funding event timestamp;
- trades, aggregate trades, book ticker and book-depth events are available at their event/update timestamp;
- a derived feature is available no earlier than the latest dependency used to calculate it.

All intervals use half-open windows `[start, end)`. Resampling must use only complete source bars. As-of joins always select the latest value whose `available_at` is not later than the decision timestamp.

## Dataset registry

The data layer recognizes these Binance dataset families from the start:

- `klines`
- `metrics` (futures metrics / open interest / long-short / taker ratios)
- `funding_rate`
- `mark_price_klines`
- `index_price_klines`
- `premium_index_klines`
- `agg_trades`
- `trades`
- `book_depth`
- `book_ticker`

Adapters normalize exchange-specific files into canonical tables. Adapters do not calculate strategy features.

## Cache model

### Level 1 - canonical market cache

Each raw archive is normalized once and written to deterministic Parquet keyed by its source stat fingerprint. If the source archive changes, a new cache file is produced. DuckDB is used so Parquet does not require a separate PyArrow dependency.

### Level 2 - feature cache (next stage)

Features are keyed by source snapshot, feature version, symbol, interval, date range and parameters. Changing TP, fees, or another exit-only setting must not recalculate DI/MR/regime/OI features.

### Level 3 - prepared backtest frame (next stage)

A prepared frame contains the strategy timeline plus exactly the features required by the strategy/research request. Repeated execution-only experiments can reuse this frame.

## Feature-store contract (next stage)

Feature providers will declare:

- feature name and version;
- required source datasets;
- parameters;
- warm-up requirement;
- output schema;
- causal availability rule.

Planned feature groups are price/trend, DI, mean reversion, volatility, regime, S/R, OI, funding, derivatives positioning, taker flow, trade flow and order-book research.

## Simulator contract

The current event-driven execution semantics are retained while the migration is in progress. The simulator remains responsible for positions, risk, intrabar exits, TP/SL ambiguity, partial exits, break-even, trailing, timeouts, fees and portfolio state.

Indicator calculation and direct file access will move out of the simulator. Intrabar lookup will later use pre-indexed/searchsorted integer ranges and NumPy slices instead of a DataFrame boolean scan for each strategy candle.

## GUI target

The future Data page selects market, symbol, period, strategy interval, execution interval and datasets. It shows catalog coverage instead of Binance CSV browse fields. Research feature groups are enabled independently. A Data Status page reports gaps and dataset coverage.

The current GUI remains untouched until the new data path reproduces a validated baseline.

## Reporting/provenance target

Runs will eventually emit a `run_manifest.json` containing code commit, data snapshot/fingerprints, feature versions/parameters, strategy hash and execution hash. Parquet becomes the efficient internal format while CSV/Excel remain convenience outputs. The read-only MCP reporting interface remains.

## Migration sequence

1. Contracts, dataset registry, availability semantics.
2. Archive discovery/catalog and kline ZIP adapter.
3. Canonical Parquet market cache and `MarketDataStore`.
4. Baseline parity using the new store while leaving the old loader available.
5. Causal alignment/resampling framework.
6. Price/DI/MR/regime/volatility/SR feature store.
7. Remove feature calculation from the simulator and optimize intrabar indexing.
8. OI, taker, funding, long-short and premium/mark/index research.
9. Aggregate-trade and raw-trade features.
10. Order-book research.
11. GUI redesign.
12. Reporting/MCP modernization.
13. Retire the CSV-first loader, Binance combiner workflow and GUI/engine monkey-patching after parity.

## Migration safety gates

Before replacing the old path, automated golden tests must prove unchanged execution behavior for normal TP/SL, ambiguous bars, intrabar fills, partial exits, break-even, trailing, timeout and dynamic S/R target behavior.

Causality tests must include a generic invariant: modifying any source row strictly after decision time `T` must not change any feature value available at `T`.
