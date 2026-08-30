# Crypto Strategy Lab

Desktop research and backtesting application for directional cryptocurrency strategies using Binance OHLCV data.

## Filtered direction flip

The DI Direction Selection panel includes **Flip direction after filters pass
(Long ↔ Short)**. Filters are evaluated against the original DI-selected
direction. When the signal passes, the backtester opens only the opposite side.

Example: enable DI-direction selection with Preferred Side Only execution, set
the ADX filter to `ADX <= Maximum` with a maximum of `10`, then enable the flip.
An eligible DI Long becomes a Short; an eligible DI Short becomes a Long.

Each strategy profile has one **Entry Rules** table. Every row chooses an
action (Flip or Reject), indicator, Inside/Outside condition, and range. Flip
and Reject groups each have an Any (OR) / All (AND) selector. Reject rules are
evaluated first, followed by Flip rules; signals matching neither trade in the
normal DI direction. Use **Add rule** or **Remove selected** to manage any
number of rules.

## DI Direction & Pressure

DI direction selects Long when +DI is above -DI and Short when -DI is above
+DI. The optional pressure analysis records whether the selected directional DI
is expanding, contracting, or mixed over a configurable lookback. This is
record-only telemetry and never filters a trade.

DI spread acceptance remains configured separately in **Strategy Profiles →
Rules → DI Spread**. The indicator analysis workbook includes direction,
regime, pressure-state, and DI-spread-change performance tables.

## Start the application

On Windows, double-click the normal no-console launcher:

```text
Crypto Opportunity Scanner.vbs
```

It starts `.venv\Scripts\pythonw.exe` from the project directory, including when
the directory path contains spaces. Startup errors are displayed in a GUI dialog.
`Debug Launcher.bat` is available for troubleshooting only; it intentionally opens
a console so Python diagnostics remain visible.

Or launch it from a terminal:

```powershell
.\.venv\Scripts\python.exe app.py
```

For a new environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt pytest
python app.py
```

## Main workflow

1. In **Backtest Setup**, choose the Binance pair and strategy timeframe.
2. Enable intrabar data when lower-timeframe exit ordering is required.
3. Use the separate `C:\CryptoBots\Binance Data Hub` application to create or update shared candle files.
4. Configure Bull Long, Bull Short, Bear Long, Bear Short, Sideways Long, and Sideways Short under **Strategy Profiles**.
5. Save the configuration and run the backtest.
6. Review the Summary, Trades, Charts, and generated output folder.

Crypto Strategy Lab reads Binance USD-M Futures candles from `C:\CryptoBots\Binance Market Data\futures\usdm`. Downloading is handled by the separate Binance Data Hub so it can continue in the background and serve every bot. The older Spot cache is kept isolated and is not selected.

## Portfolio workflow

The **Portfolio** tab accepts a dynamic list of assets. Each enabled asset uses its own saved strategy configuration while sharing one account.

- Base risk is configured per asset.
- Maximum Total Portfolio Risk is a hard entry limit and defaults to 5%.
- Entries that would exceed that limit are blocked and reported.
- Portfolio output includes accepted trades, blocked candidates, realized equity, mark-to-market equity, monthly/yearly results, and component contribution.

## Project structure

```text
.github/                  GitHub Actions workflows
.vscode/                  Repository-local editor settings
config/                   Current saved strategy configurations
crypto_strategy_lab/      Installable application package
  gui/                    Interface, profile editor, and background workers
  config.py               Backtest configuration model
  engine.py               Trade simulation engine
  portfolio.py            Shared-equity portfolio replay
  strategy_profiles.py    Six market-regime strategy profiles
  loader.py               OHLCV loading and validation
  output_manager.py       Reports and run-folder management
docs/                     Implementation and design documentation
mcp_server/               Read-only report MCP server
tests/                    Automated regression suite
tools/                    Offline import and maintenance utilities
Crypto Opportunity Scanner.vbs   Normal no-console Windows launcher
Debug Launcher.bat        Console launcher for debugging only
app.py                    Desktop application entry point
cli.py                    Optional command-line backtest entry point
requirements.txt          Python dependencies
```

Local market data is stored outside the repository workflow and generated backtest results use `output/`, which is ignored by Git.

## Supported CSV format

The loader requires timestamp, open, high, low, close, and volume. It accepts these timestamp headers:

```text
timestamp, open_time, time, datetime, date
```

Raw Binance files with additional columns are supported. Extra fields such as close time, quote volume, trade count, and taker volume are ignored.

Use **Validate Data** before a run to check timeframe, coverage, duplicates, invalid candles, and missing candles.

## Offline utilities

The Binance Data Lake reads immutable exchange archives directly. Other maintenance
utilities are kept in `tools/` and should be run as modules from the repository root.

## Tests

```powershell
$env:QT_QPA_PLATFORM="offscreen"
.\.venv\Scripts\python.exe -m pytest tests -q
```

Market data, generated reports, virtual environments, and Python caches are intentionally excluded from version control.

## GitHub Integration

The desktop **GitHub** tab replaces the old pull and push command files. **Check
Status** fetches remote metadata and reports the branch, changed files, and
ahead/behind counts. **Pull Latest** uses a fast-forward-only pull. **Review
Changes** displays working-tree and staged diffs, and **Commit & Push** requires
the user to select each path and enter a commit message.

The workflow is intentionally conservative: pulling requires a clean working
tree, diverged branches and merge conflicts require manual resolution, and no
force push, automatic stash, hard reset, or repository clean is performed.
Ignored files cannot be added and common secret or credential filenames are
blocked. Authentication is delegated to the user's existing Git configuration,
SSH agent, or Git Credential Manager; the application does not store GitHub
passwords or personal access tokens. Pull and commit/push are disabled while a
backtest or portfolio calculation is running. After an update, restart Crypto
Strategy Lab manually to load the downloaded source.

## Local MCP Server

A local, read-only MCP server lets an MCP client inspect existing backtest reports. Install the dependencies with:

```powershell
pip install -r requirements.txt
```

By default the server exposes only this project's `output` directory at
`http://127.0.0.1:8765/mcp`. Set `CRYPTO_STRATEGY_LAB_OUTPUT_DIR` to select a
different existing output root, or `CRYPTO_STRATEGY_LAB_MCP_PORT` to change the
local port. Paths are resolved beneath that root and the server cannot edit
files, run backtests or commands, or execute mutating SQL.

The supported tools are `list_runs`, `latest_run`, `list_run_files`,
`read_report`, `query_trades`, and `compare_runs`. They support saved CSV, XLSX,
JSON, and TXT reports; DuckDB access is restricted to read-only queries over a
run's `trade_list.csv`. Connecting ChatGPT or another client (and configuring a
supported MCP tunnel when needed) is a separate step. Do not expose the local
server directly to the public internet.

## ChatGPT Integration

The desktop application's **ChatGPT** tab can run both the existing read-only
MCP server and an OpenAI Secure Tunnel without separate command windows:

1. Create the OpenAI tunnel externally once and download `tunnel-client.exe`.
2. Open **Crypto Strategy Lab → ChatGPT**.
3. Browse to the tunnel client executable (its location is not fixed).
4. Paste the tunnel ID supplied when the tunnel was created.
5. Choose **Set / Change API Key** and securely save the tunnel runtime API key.
6. Select **Start ChatGPT Connection**. The local MCP server becomes ready
   before the secure tunnel is started.
7. Enable the Crypto Strategy Lab plugin in ChatGPT.

The runtime API key is stored through `keyring` in Windows Credential Manager;
it is never saved in application settings, configuration files, command-line
arguments, or connection logs. The tunnel path, tunnel ID, auto-start choice,
and local port are non-secret settings. **Test Configuration** performs local
checks only and does not make a model request. The MCP endpoint remains bound to
`127.0.0.1` and retains its read-only report-access security model.

Use **Open Logs** for bounded, redacted MCP/tunnel diagnostics. Processes
started by the GUI are stopped in tunnel-then-server order. Because Qt-owned
child processes cannot be reliably detached on every supported platform, the
application offers the safe choices **Stop and Exit** or **Cancel** at shutdown
rather than risking broken orphan processes.

## Opportunity Scanner historical execution

### Historical Single

**Single timestamp** reconstructs one causal decision point at the exact UTC timestamp supplied. It continues to use the native Task 1–7 scan and immutable `OPPORTUNITY_SCAN` publication path.

### Historical Range

**Date/time range** generates `start`, `start + cadence`, and subsequent exact UTC decision timestamps through the inclusive end, then runs the same single historical scan sequentially for each timestamp. **A range replay is not one giant backtest. It is a sequence of immutable historical `OPPORTUNITY_SCAN` experiments.** Completed publications are retained if a later scan is cancelled or fails.

Runtime depends heavily on range length, preliminary shortlist size, selected rich features, existing Data Lake coverage, and how much missing data must be acquired. The GUI displays exact range completion and monotonic elapsed time; estimated remaining time appears only after completed scans provide timing evidence.
