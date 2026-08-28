# Task 11 — Research runner and configuration composition

## Before

```
CLI / benchmark / GUI
    ↓
manual orchestration
    ↓
giant EnhancedBacktestConfig
    ↓
bundle preparation
    ↓
native engine
```

Each tool chose data, executed features, built L3, constructed an engine, and
formatted output. `load_backtest_bundle` also constructed its own registry.

## After

```
composition root
    ↓
ResearchRunner
    ├── DataStore
    ├── FeatureRegistry
    ├── PreparedRunCache
    ├── Strategy
    ├── Simulator
    └── Reporters

ResearchRunConfig
    ├── DataConfig
    ├── FeatureConfig
    ├── StrategyConfig
    ├── ExecutionConfig
    └── ReportingConfig
```

The CLI and benchmark are composition roots. The runner validates, loads L1,
executes the injected registry once, resolves L3 through the injected cache,
binds policy, invokes the simulator, and finally passes immutable run artifacts
to reporters. A preparation exception prevents simulation; a simulation
exception prevents reporting. The runner imports no indicator provider.

## Field ownership

* **Data** owns strategy/intrabar timeframes, whether intrabar is used, and the
  missing-intrabar policy. Symbol, exchange, market, and dates remain in
  `DataRequest`; roots remain constructor arguments to stores/caches.
* **Features** owns ATR/ADX/DI, Bollinger and mean-reversion calculation inputs,
  S/R detector and hold/break inputs, structural/return regime calculation,
  and optional research feature selection.
* **Strategy** owns entry cadence, profile selection/enabled state, entry rules
  and match modes, DI/MR/S/R allow/reject interpretation, and schedules.
  `StrategyProfileConfig` contains only these pre-entry facts/requirements.
* **Execution** owns equity/risk/leverage, fees, order type, slippage/tie policy,
  capacity, dynamic S/R TP policy, and `ExecutionProfileConfig`. The latter owns
  RR/stops, partial exits, trailing, break-even, timeout, R-step, and checkpoint
  mechanics. `enabled` exists only in the strategy profile.
* **Reporting** owns the run/output name, telemetry/lifecycle exports, analysis
  reports, charts, and formatting toggles.

Aggregate validation checks request/timeframe agreement, intrabar ordering, and
that every enabled policy profile has an execution profile before preparation.
The strict JSON v3 loader rejects unknown sections, component fields, profile
keys, aliases, and old flat v2 documents.

## Cache identity

| Component | L2 | L3 | simulation |
|---|---|---|---|
| Data/request | source-dependent | yes | yes |
| Feature | affected feature/dependants | yes | yes |
| pure Strategy policy | no | no | yes |
| profile RSI/momentum materialized facts | provider/L3 inputs | yes | yes |
| Execution | no | no | yes |
| Reporting | no | no | no |

L3 continues to hash canonical and L2 identities plus only values physically
materialized in `PreparedBacktestFrame`; it never hashes the aggregate config.
Consequently fee, slippage, RR, stop, trailing, break-even, output directory,
chart, and formatter changes reuse L2 and L3. Feature periods produce affected
L2 identities and therefore a downstream L3 identity.

## Boundaries retained

`NativeSimulator` is a plain adapter around the proven
`DataLakeProductionBacktestEngine.from_prepared` implementation. Historical
engine inheritance remains an internal simulator detail; no new subclass was
added. Rewriting that mature event loop would risk fill/TP/SL parity and is not
part of this architecture change.

The native adapter performs one bounded in-memory conversion to the existing
engine's input object. It is not the Data Lake schema and is never serialized.
Standalone CSV/CSE, the existing GUI worker, portfolio/state-transition tools,
and legacy report workflows remain consumers of `BacktestConfig` or
`EnhancedBacktestConfig`. GUI layout and widgets are unchanged; migrating that
compatibility boundary can proceed independently without blocking the validated
CLI/golden path.

`CsvManifestReporter` preserves `trade_list.csv` and provenance manifest output.
The benchmark supplies no reporters, so reporting cannot request or recompute
features. Profiling is available from result stage timings (`data_features`,
`prepared_cache`, `simulation`, and `reporting`).
