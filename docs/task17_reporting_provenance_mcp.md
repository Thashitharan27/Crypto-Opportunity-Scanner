# Task 17: reporting, provenance, and MCP

## Completed-run contract and lifecycle

Modern runs use `crypto_strategy_lab_run_v1`, version `1`. A UUID-based `run_id`,
UTC start time, and Git code state are captured by the reporter's `begin` hook
before the data loader runs. Artifacts are written and semantically validated
first; the `COMPLETED` `run_manifest.json` is atomically renamed into place last.
A partial directory is therefore not a completed run and is ignored by MCP.

The current v3 layout is intentionally small:

```text
<symbol>_<strategy interval>_<run id>/
  run_manifest.json
  trade_list.csv
  summary.json
  data_quality.json
  backtest_report.xlsx
  artifacts/
    trades.parquet
    feature_context.parquet
    signals.parquet
  provenance/
    source_archives.parquet
```

There is one authoritative trade Parquet. The Task-16 query service resolves
both trade and feature-context Parquets from the canonical artifact catalog; no
duplicated research manifest is written for modern runs. Fixed indicator,
S/R, journey, lifecycle, and other question-specific reports are not part of the
completed-run contract. Those questions are answered on demand from canonical
artifacts so the stored output does not grow with historical research workflows.

## Reproducibility and configuration identities

The manifest records the commit, code-dirty flag, provenance status,
Python/platform, configuration contract, and `requirements.txt` digest. Clean
code state is `REPRODUCIBLE`. Tracked changes, untracked source-like files, or
unavailable Git state are `PARTIAL`. Generated output/cache/query files do not
make code provenance dirty merely because they are untracked. Dirty code state
records a tracked-diff digest and names (never contents) of untracked source-like
files. Environment variables, `.env`, credentials, and arbitrary file contents
are never recorded.

The complete native v3 data, feature, strategy, execution, and reporting config
is embedded using deterministic JSON. Strategy identity contains only strategy
config. Execution identity contains execution config plus strategy/intrabar
resolution and missing-intrabar semantics that can affect fills. Reporting
settings and paths are excluded from both. Separate data and feature-config
hashes make their changes explicit and reporting remains downstream of L1/L2/L3
cache identity.

## Source, feature, and prepared provenance

`DataCatalog.records_for` passively records the catalog rows selected by the
real load. Reporting deduplicates and canonically sorts those metadata records;
it does not open or hash Binance archives. The selected table includes exchange,
market, dataset, symbol, interval, frequency, periods, size, mtime, raw
fingerprint, and the **same canonical partition identity used by L1**, including
the adapter canonical contract/version. Its canonical rows define the
selected-catalog digest, so unrelated catalog rows cannot affect a run. Compact
per-dataset signatures remain in the manifest; exact rows are compressed in
`source_archives.parquet`.

Materialized feature provenance includes explicit provider version, normalized
parameters, dependency names, source dataset roles, cache key, and rows. The
feature cache key remains the authoritative dependency/resource identity and the
exact contributing archives are independently preserved by the selected-source
snapshot. Prepared provenance records the prepared cache key, contract/version,
and row count. None of this reporting metadata participates in prepared-cache
identity.

## Artifact semantics and integrity

Every catalog entry declares path, format, schema version, rows where relevant,
bytes, and SHA-256. Publication validates Task-16 strategy-index/timestamp/value
parity and the semantic completed-trade fingerprint before the completion
manifest can exist. It also checks CSV, Parquet, and summary trade counts.
`summary.json` uses the existing statistics helper and adds an explicit `net_pnl`
derived from ending minus initial equity. `backtest_report.xlsx` is the compact
human review surface: Dashboard, Monthly, Yearly, Market Regime, and Direction ×
Regime. Machine research uses the canonical Parquets rather than the human
formats.

`feature_context.parquet` stores the causal prepared feature/research state for
every strategy row, including decision-availability timestamps. This is the
primary surface for on-demand DI, ADX, volatility, funding, open-interest,
positioning, taker-flow, S/R, and similar research.

`signals.parquet` contains observations produced by the **same simulation run**.
Accepted decisions come from completed trade rows and therefore retain the exact
`research_signal_index`, candle-open time, and decision-availability time.
Rejected decisions come from the engine's existing decision-point
`skipped_signals` records and are attached only by an exact strategy-candle key
to the already-bound `PreparedBacktestFrame`; failure to find that exact row is
an error. There is no nearest/asof match and no strategy re-evaluation. Signal
publication verifies its causal timestamps against `feature_context.parquet` and
requires one `ENTER` signal for every completed trade.

Execution telemetry is not part of the current native v3 completed-run contract.
Legacy telemetry/journey/lifecycle implementations may remain outside this
contract while retired code is cleaned separately, but standard runs neither
publish an empty telemetry artifact nor expose a telemetry MCP query.

Readers verify the catalog hash before opening immutable artifacts. Missing,
changed, symlinked, escaping, or uncataloged artifacts produce an integrity
error; they are never regenerated from market data.

## Read-only MCP

MCP discovers direct-child runs solely from valid completed manifests and orders
them by manifest start time. It exposes `list_runs`, `latest_run`,
`get_run_manifest`, `list_run_files`, `read_report`, `query_trades`,
`query_signals`, `research_aggregate`, and `compare_runs`. Research aggregation
delegates to the Task-16 `ResearchQueryService`.

Parquet relations are registered internally in DuckDB. User SQL remains one
read-only statement, returns at most 5,000 rows, and rejects mutation/DDL,
extensions, pragmas, attachment, copies, external read functions, quoted file
relations, and semicolons. Absolute paths, traversal, indirect symlink escape,
and files outside the artifact catalog are rejected. MCP imports no runner,
simulator, feature registry, or market-data service and performs no writes.

Comparisons expose statistics and code/source/feature/strategy/execution hashes,
plus explicit equality flags against the first run. Dirty-code equality includes
the tracked diff and untracked source-path identity, not only the Git commit.
Incomplete and legacy directories are ignored; corrupt completed runs fail rather
than falling back to filename/config/mtime heuristics.
