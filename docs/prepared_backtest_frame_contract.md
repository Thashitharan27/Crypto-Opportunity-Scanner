# PreparedBacktestFrame contract inspection

## Existing production boundary

`DataLakeProductionBacktestEngine(data, config, intrabar_data, ...)` still accepts a
legacy OHLCV DataFrame plus separate technical, production-context, S/R, research,
and structural-benchmark DataFrames. Construction converts OHLCV to arrays, injects
prepared indicator arrays through temporary function shims, and initializes mutable
portfolio, scheduling, telemetry, risk, regime, profile, and execution state. This is
the legacy bridge, not the native target interface.

## Values read by simulation

The strategy timeline reads candle timestamp and OHLC for scheduling, entry price,
bar exits and market structure; volume for session VWAP; ATR for sizing, stops,
targets and S/R distances; ADX and +/-DI for eligibility/direction; ATR percentage,
Bollinger width, session VWAP and close location for profile filters; and the prepared
mean-reversion envelope, RSI, re-entry flags, mean/distance values for enhanced entry
management. Regime and risk arrays also affect eligibility and sizing, but they are
configuration-derived outputs and therefore belong in a later native decision-policy
contract rather than being copied from today's loader schema.

Construction currently creates OHLCV/time arrays; ATR/ADX/DI, Bollinger, lag/change,
regime, ATR-percent, close-location, risk, VWAP, mean-reversion, per-profile RSI and
momentum arrays; S/R readers; intrabar indexes; and mutable run state. The new frame
contains only externally prepared row values, never mutable simulation state.

## Classification

**A — execution-critical:** timestamp, OHLCV and ATR. These determine scheduling,
prices, bar-path exits, sizing, stop/target distances, and price-location analysis.
Volume is therefore required by the bounded Data Lake adapter; it is never fabricated
when absent.

**B — strategy-decision context:** ATR%, ADX, +/-DI, Bollinger width/width%, session
VWAP, close location, and the mean-reversion mean/distance/envelope/RSI/re-entry
values. The simulator may interpret them but must not generate them.
`decision_available_at` records their causal boundary.

**C — reporting/research only:** named `ResearchContext` blocks. They are immutable,
aligned and causally validated, but structurally separated so adding research does
not widen or silently influence the executable core.

The bounded adapter requires technical, production-context, and research feature
frames to match the strategy timestamps exactly row-for-row after UTC normalization.
Equal length alone is not sufficient; shifted, reordered, missing, or duplicate rows
are rejected before values are attached to the native contract.

Intrabar timestamp/open/high/low belong exclusively to `IntrabarExecutionData`.
Its interval, time-grid alignment and strict timestamp ordering are validated against
the strategy frame. Compatibility requires overlap with the strategy execution period,
not complete coverage of it, because the current Data Lake path intentionally supports
an `intrabar_start` later than the strategy start and missing-data policy remains a
simulator concern.

## Intentional exclusions and remaining legacy dependencies

Raw feature DataFrames, column attrs/cache keys/provider versions, arbitrary feature
maps, `available_at` columns per legacy block, benchmark candles, S/R object snapshots,
DI diagnostic lag/change columns, Bollinger bands not used by current profile logic,
research metadata, loader/catalog identity, config, and mutable engine state are not
core contract fields. They are either reporting diagnostics, generator provenance,
configuration-specific derived state, or legacy artifacts to remove rather than
perpetuate. Entry-long/short flags and regime/state enums are also not fabricated:
today the mature strategy derives those decisions from config plus prepared values.

`from_data_lake_bundle` is the only bounded adapter. Production execution is not
routed through the new contract in this task; the legacy DataFrame bridge, structural
benchmark/regime generation, S/R provider, risk/profile-derived arrays, and existing
intrabar wrapper remain for the native-consumption task.
