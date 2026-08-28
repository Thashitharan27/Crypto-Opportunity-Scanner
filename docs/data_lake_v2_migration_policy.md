# Data Lake v2 migration policy

Crypto Strategy Lab is taking a clean-break migration to the Binance Data Lake architecture.

## Supported forward path

`Binance raw daily/monthly archives -> MarketDataStore -> canonical cache -> causal features -> simulator -> reports`

The raw Binance archive tree is read-only. Disposable catalog/Parquet/feature caches live outside the raw tree.

## Explicit non-goals

The migration does **not** need to preserve:

- old combined files such as `BTCUSDT_4h.csv`, `BTCUSDT_1h.csv`, or `BTCUSDT_1m.csv`;
- old output-run configuration snapshots;
- retired GUI/configuration fields;
- exact historical run outputs from the pre-Data-Lake architecture;
- filename-based benchmark discovery inside the simulator.

Existing compatibility/parity utilities may remain temporarily as diagnostics, but they are not release gates and should not shape the new design.

## Engine boundary

The execution simulator must not discover or read market-data files. The Data Lake execution path prepares strategy candles, intrabar candles, benchmark data, and eventually all feature columns before simulation starts.

`DataLakeBacktestEngine` is the migration adapter while the monolithic simulator is decomposed. Its structural regime is injected from a prepared benchmark frame and does not use `structural_regime_benchmark_csv`.

## Feature migration order

1. structural/asset market regime
2. ATR / ADX / DI base features
3. DI direction and pressure state
4. mean reversion
5. volatility state
6. support/resistance
7. futures metrics / open interest
8. taker flow
9. funding and long/short ratios
10. mark/index/premium basis
11. trade and order-book datasets

Every migrated feature must retain explicit availability timing and no-look-ahead tests.
