# Task 7 — Feature / simulator separation audit

## Scope

Task 7 audits and removes market-feature calculation from the **native Data Lake production/reference path** introduced by Tasks 5–6. It does not delete old DataFrame engine constructors or redesign FeatureRegistry/cache architecture.

The boundary is:

1. **feature generation** — causal market-derived values prepared before simulation;
2. **strategy policy** — interpretation of prepared values for direction/profile/risk/target decisions;
3. **execution runtime** — mutable trade/portfolio state and fill/exit mechanics;
4. **reporting/research** — descriptive outputs derived from prepared values and completed trades.

## Pre-migration reachable native inventory

Before this task the native `from_prepared` path still derived several market features at construction or snapshot time:

- DI spread lags/change/ratio and pressure changes/states;
- Bollinger width lags/change;
- asset return regime, structural regime alignment and configured momentum-return windows;
- MR state/motion/strength and additional MR-v2 classifications through inherited engine snapshot code;
- higher-timeframe support/resistance through the mature engine detector rather than a prepared provider;
- state-transition daily regime/volatility state by rebuilding it after simulation.

Risk-unit arithmetic, profile selection, direction-dependent alignment, dynamic-target selection, and entry allow/reject are strategy policy rather than market-feature generation. Position lifecycle, intrabar fills, TP/SL, partial exits, break-even, trailing, timeout, fees/slippage and equity are execution state.

## Migration by family

### Directional / DI

`CoreDirectionalFeatureProvider` is the market-feature authority for:

- ATR / ATR percent;
- ADX, +DI and -DI;
- DI spread and lagged spreads;
- DI spread change and DI ratio;
- +DI/-DI changes;
- LONG and SHORT directional/opposing changes;
- LONG and SHORT pressure states.

`PreparedBacktestFrame` carries these arrays explicitly. The native runtime only selects the side implied by strategy policy. Unknown/no-direction telemetry preserves the previous NaN/UNKNOWN behavior.

### Mean Reversion v2

`ProductionContextFeatureProvider` is the market-feature authority for the complete currently used MR-v2 context:

- mean and ATR-normalized distance;
- previous distance and distance change;
- state, motion, strength and strength label;
- sigma, MR Bollinger envelope and z-score;
- Bollinger location;
- RSI and RSI state;
- LONG/SHORT re-entry flags and re-entry confirmation;
- MR signal, signal direction and setup strength;
- compatibility reporting values `bb_reentry`, `mr_signal`, and `mr_signal_direction`.

The provider version is bumped to invalidate older cached frames that do not contain the complete schema. The native runtime consumes these prepared classifications directly. Only alignment against the strategy-selected DI/trade direction remains policy.

### Regime and momentum

`features.market_regime` prepares:

- causal trailing asset returns;
- asset-return regime labels;
- structural benchmark regime labels with completed-daily availability;
- configured momentum-return windows.

Benchmark loading remains in the Data Lake preparation layer. The native simulator neither reads benchmark files nor resamples benchmark prices.

### Support / resistance

`SupportResistanceFeatureProvider` prepares both LONG and SHORT context for same-timeframe and higher-timeframe S/R.

For higher-timeframe S/R it reuses the mature causal resampling/detector semantics before simulation:

- only complete higher-timeframe candles are eligible;
- the latest completed HTF candle is selected for each strategy row;
- pivot/right-side confirmation, zones, tests, hold/break/rejection state and room are prepared;
- the current strategy close is used only as the evaluation price, matching the mature HTF path;
- `sr_completed_candle_time` records the exact HTF candle used and is validated not to exceed feature availability.

Native execution consumes the prepared row through `PreparedSupportResistanceContextReader`; it does not rebuild pivots or an HTF detector.

### Volatility

Currently used production volatility inputs are provider-owned:

- ATR / ATR percent from the directional provider;
- Bollinger width and its lags/changes from the production-context provider;
- daily state-transition volatility from the cached daily research provider.

Risk/sizing decisions based on those prepared values remain policy.

### State-transition research

`StateTransitionDailyFeatureProvider` prepares and caches the existing daily research state before simulation. It preserves the existing state-transition algorithm and adds explicit `available_at = following UTC midnight` semantics.

The Data Lake reporting path now calls `generate_prepared_state_transition_reports(...)`, which consumes the prepared daily frame and completed trades. It does not call `daily_state_frame` or reload/reconstruct market state after simulation. State-transition research remains descriptive and is not an entry filter.

## PreparedBacktestFrame contract changes

The contract was widened only with named production fields required by the current native strategy/runtime:

- complete DI/pressure arrays;
- Bollinger/volatility lags and changes;
- complete MR-v2 market classifications;
- prepared regime labels and bull-regime returns;
- configured momentum-return arrays keyed by lookback hours.

Research-only futures/S/R blocks remain separated as `ResearchContext`. No arbitrary-column escape hatch was added.

## Causality and parity tests

The focused tests cover:

- DI provider parity and LONG/SHORT interpretation;
- preservation of UNKNOWN/no-direction DI telemetry semantics;
- MR-v2 provider parity against mature engine snapshots;
- prevention of native MR market-classification calls;
- prefix/future-mutation causality for directional, MR and S/R features;
- same-timeframe S/R reader parity;
- higher-timeframe S/R parity against the mature engine and completed-candle causality;
- structural regime completed-daily availability;
- state-transition daily availability and future-mutation causality;
- prepared state-transition reporting with post-simulation `daily_state_frame` recomputation explicitly forbidden;
- native construction with inherited return/regime feature calculators forbidden.

GitHub CI is required to remain green after amendments. The real BTC archive golden run remains the final execution acceptance gate.

## Final reachable native audit

### A. Strategy policy

The native path still performs, intentionally:

- DI side/direction selection from prepared +DI/-DI;
- profile selection and threshold/rule evaluation;
- direction-dependent DI/MR alignment;
- risk-mode sizing arithmetic from prepared values;
- S/R entry/dynamic-target decisions from prepared S/R context;
- daily scheduling and portfolio admission decisions.

### B. Execution state

The native path still owns:

- active/completed position state;
- entry, quantity and leverage application;
- intrabar search/iteration;
- TP/SL and same-bar ambiguity;
- partial exits;
- break-even;
- trailing / checkpoint state;
- timeout;
- fees and slippage;
- equity and portfolio constraints;
- force-close/end-of-data behavior.

### C. Reporting / telemetry

The native path still builds:

- trade/result rows;
- prepared-value snapshots;
- policy/alignment labels;
- skipped-signal reasons;
- signal-time research freezing;
- progress/diagnostic output.

### D. Market feature calculations

**NONE for currently used production market features on `DataLakeProductionBacktestEngine.from_prepared`.**

Direction-dependent interpretation of prepared features is policy, not feature generation.

## Remaining legacy/non-native feature code

Legacy code intentionally remains for later cleanup:

- raw-DataFrame constructors in `engine.py`, `enhanced_engine.py`, `data_lake_engine.py`, and related classes still calculate indicators for legacy/non-native callers and tests;
- the compatibility Data Lake constructor still uses temporary feature-injection shims;
- the legacy higher-timeframe S/R engine path remains available to non-native callers;
- the current GUI worker still binds the mature constructor-based engine for its existing reporting pipeline, although its Data Lake feature inputs and state-transition reporting are prepared/cached;
- At Task 7 completion, `legacy_bridge.py` remained untouched; Task 10 subsequently retired it after the native boundary was proven.

These are not the canonical native benchmark/reference path and are scheduled for later migration/cleanup rather than being hidden behind new compatibility wrappers.

## Golden and performance status

Codex does not have the validated local Binance archive, so no real-data golden result is claimed in this document.

Final local validation command on Windows:

```bat
python tools\data_lake_benchmark.py ^
  --config "config\data_lake\BTCUSDT_4H_SMOKE.json" ^
  --raw-root "C:\CryptoBots\Binance Market Data" ^
  --symbol BTCUSDT ^
  --start 2024-01-01 ^
  --end 2026-08-01 ^
  --iterations 2
```

Required gate:

- strategy rows: 5,658;
- intrabar rows: 1,357,920;
- trades: 504;
- fingerprint: `83cfd17605b544554aed57ec1c851854a327116c2bb75b3e2500c9cd4ebce0f5`.

Performance is recorded only from the same exact-parity run; it is secondary to correctness for Task 7.

## Next roadmap task

**Task 8 — Make FeatureRegistry authoritative.**
