# Native prepared simulator boundary

## Constructor trace and ownership

The legacy production path is `DataLakeProductionBacktestEngine -> DataLakeBacktestEngine -> SRDynamicTPBacktestEngine -> EnhancedBacktestEngine -> BacktestEngine`. Its constructors formerly rebuilt an OHLCV DataFrame view, installed temporary indicator shims, calculated ATR/ADX/Bollinger/mean-reversion values, initialized strategy-policy arrays, and then initialized mutable execution state before `run()`.

The native entry point is `DataLakeProductionBacktestEngine.from_prepared(prepared, intrabar, config, ...)`. It deliberately invokes none of those constructors. The simulator-facing market inputs are immutable `PreparedBacktestFrame` arrays and optional `IntrabarExecutionData` arrays. Structural regime labels are an optional, aligned strategy-policy input because their benchmark is intentionally outside the simulator. `BacktestConfig` remains the bounded configuration adapter for this migration; a project-wide configuration split is out of scope.

## Prepared versus runtime values

OHLCV, timestamps, ATR/ATR%, ADX/DI, Bollinger width, session VWAP, close location, mean-reversion values and re-entry flags are assigned directly by reference from `PreparedBacktestFrame`. Intrabar scans use `searchsorted` windows over `IntrabarExecutionData`; they do not filter or construct a pandas frame.

Lagged width/DI telemetry, DI ratios, trailing asset returns, risk units, profile momentum, and market-regime labels are strategy-policy state derived only from config plus prepared arrays. Portfolio equity, active/completed positions, pair IDs, schedules, timeout state, missing-data state, telemetry, fees, slippage, TP/SL, break-even, partial exits and trailing state remain mutable execution-runtime state.

No loader, benchmark loader, feature provider, indicator function, legacy bridge, canonical-to-legacy conversion, or constructor monkey-patch is used by the native path. Pandas remains in timestamp normalization and final result reporting, not as its market-data input interface.

## Reused execution logic

The established `run()` event loop and entry, sizing, portfolio, exit, ambiguity, timeout, fee/slippage, partial-exit, break-even, trailing, closure, telemetry, and result-row methods are reused unchanged. The only newly extracted execution primitive is the immutable array-backed intrabar window. Data Lake preparation consumes canonical frames directly; standalone CSV constructors remain available only to non-Data-Lake callers.
