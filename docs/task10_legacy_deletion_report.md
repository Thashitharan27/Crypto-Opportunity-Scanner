# Task 10 legacy deletion report

## Authoritative architecture

The production/reference flow is now **Binance archives → canonical Data Lake
frames → L2 provider-owned features → L3 `PreparedBacktestFrame` → native
simulator**. Strategy and execution frames retain `period_start`, canonical
availability fields, identity columns, and `canonical_source_identity` until the
bounded array projection. No native module converts them to the standalone CSV
loader's `timestamp` DataFrame contract.

## Pre-deletion reference inventory

| Candidate | Classification before deletion | Traced consumers / conclusion |
|---|---|---|
| `data/legacy_bridge.py` conversion and parity helpers | C — test/migration tool only, plus A — bridge used by active preparation | The conversion was called by `data/backtest_service.py`; remaining calls were the parity and validation migration tools and bridge tests. Native preparation was changed first, leaving no production consumer. |
| `data/backtest_service.py` | A — active native Data Lake path | Used by the run, benchmark/profile tools and GUI worker. Retained; its strategy and execution loads now remain canonical. |
| `prepared_backtest.py` / `prepared_cache.py` | A — active native Data Lake path | L3 construction/cache path. Retained and changed only at the canonical projection boundary. |
| `data_lake_engine.py`, `data_lake_production_engine.py`, inherited engines | A — active execution semantics and strategy policy; F — some inherited construction compatibility | Native `from_prepared` still reuses proven simulation semantics. No broad inheritance redesign was safe in Task 10. |
| `loader.py`, `load_ohlcv_csv()`, CSV config fields | B — active non-Data-Lake path | Standalone CLI/desktop/CSE and older CSV workflows still call them. Native architecture tests now prohibit them in Data Lake modules. |
| `data_lake_config.py` inert legacy config fields | F — compatibility-only | Strict Data Lake input excludes these keys, but the shared `BacktestConfig` construction still needs placeholders. Separation belongs to Task 11. |
| `tools/data_lake_backtest_parity.py` | C — migration tool only | Referenced only by its dedicated tests and CI help/compile smoke. The golden native benchmark supersedes it. |
| `tools/data_lake_validate.py` legacy comparison | C — migration behavior inside a useful native diagnostic | Rewritten to validate canonical schema, coverage, availability/causality, identity, and provenance directly. |
| `tools/combine_binance_data.py` | C — manual migration tool only | Referenced only by its own tests and README instructions. Immutable archive ingestion supersedes it. |
| GUI/CLI Data Lake entry points | A — active native path | Retained. Data requests supply symbols; run/benchmark manifests do not infer Data Lake identity from CSV filenames. |
| old documentation and workflow smoke entries | D — documentation only | Updated or removed where they instructed users/CI to preserve deleted migration tools. |
| dual-leg and timeout names, state-transition research, generic filename inference | B/F or active strategy research | Not deleted merely by name: callers remain outside this bounded migration removal, or code implements active execution/research semantics. |

Engine code was classified during tracing as: execution event-loop/exit/position
state (A, retained), strategy policy (B, retained), reporting/telemetry (C,
retained), Data-Lake feature initialization (D, bypassed by `from_prepared` but
still shared with non-native constructors), and dead migration adapters (E,
deleted only where listed below). State-transition reporting consumes the
prepared daily feature block and has no native post-simulation market reload.

## Removed items

| Removed item | Why it was safe | Replacement | Tests proving replacement |
|---|---|---|---|
| `crypto_strategy_lab/data/legacy_bridge.py` (`canonical_to_legacy_ohlcv`, store-frame loader, frame/trade parity records and comparators) | After changing both native projections, every remaining reference was migration-only. | Direct canonical projection in `from_data_lake_bundle()` and the golden native benchmark. | `test_prepared_backtest.py`, `test_data_lake_execution_path.py`, `test_data_lake_native_architecture.py` |
| `_legacy_from_canonical()` and `_legacy_klines()` | They existed solely to manufacture the retired six-column shape. | `MarketDataStore.load_klines()` output is stored directly in `BacktestDataBundle`. | Native architecture and execution-path tests. |
| Legacy-shaped intrabar wrapper in bundle loading | Prepared/native execution uses immutable searchsorted arrays and does not need DataFrame mask compatibility. | `load_execution_klines()` → canonical `period_start` projection → `IntrabarExecutionData`. | `test_prepared_backtest.py`, `test_data_lake_intrabar_window.py` |
| `tools/data_lake_backtest_parity.py` and `tests/test_data_lake_parity_tool.py` | Only dedicated tests and CI smoke referenced it; it compared two legacy-shaped engine inputs. | Stable golden fingerprint gate in `tools/data_lake_benchmark.py`. | Benchmark tests and architecture deletion assertion. |
| Legacy CSV options/comparison in `tools/data_lake_validate.py` | CSV parity was migration-only and imported both retired bridge and loader. | Canonical catalog/schema/coverage/causality/source validation. | Validator help smoke plus canonical validator tests. |
| `tools/combine_binance_data.py` and its test | Only self-test and obsolete README manual workflow consumed it. | `MarketDataStore` directly catalogs immutable Binance ZIP archives. | Data Lake v2 archive adapter/store tests. |
| Bridge test and CI parity/bridge smoke entries | Their assertions forced deleted architecture to remain. | Tests of canonical values/provenance and structural dependency guards. | `test_data_lake_native_architecture.py`, updated prepared/execution tests. |

## Remaining legacy/non-native items

* **Generic CSV loader and CSV config fields.** Standalone desktop/CLI and CSE
  workflows remain real consumers. Deleting them would break supported
  non-Binance data. Config and runner separation is planned for Task 11.
* **Shared `BacktestConfig` placeholders in strict Data Lake config loading.** They
  are inert on the Data Lake runtime but required to instantiate the shared
  configuration class. Removing them requires Task 11's config separation.
* **Inherited legacy execution classes and constructors.** The native engine
  bypasses DataFrame feature initialization through `from_prepared`, while still
  deliberately inheriting stable strategy/execution/reporting behavior. A
  composition redesign is Task 11, not this deletion.
* **Filename symbol inference in standalone CSV mode.** Its consumers are old
  CSV/CSE runs. Data Lake requests and metadata provide symbol identity, so it is
  isolated from the protected native modules.
* **Searchsorted DataFrame compatibility utility.** Non-native constructors and
  focused compatibility tests still consume it; native prepared intrabar data
  uses `IntrabarExecutionData.fast_window()` directly.

## Acceptance notes

The BTC archive is not included in the repository, so the 2024–2026 golden and
warm-cache/performance measurement must be run on the local archive before
merge. The fingerprint is not changed. A cold/warm cache contract change was
not required: L3 identity already uses the canonical source identity and the
prepared arrays have the same values and timeline.
