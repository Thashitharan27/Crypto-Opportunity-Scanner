import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from crypto_strategy_lab.run_manifest import atomic_json, file_sha256
from mcp_server.server import BacktestReports, create_server


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.register("frame", frame)
        escaped = str(path).replace("'", "''")
        con.execute(f"COPY frame TO '{escaped}' (FORMAT PARQUET)")


def _artifact(path: Path, run: Path, fmt: str, rows=None):
    return {
        "path": path.relative_to(run).as_posix(),
        "format": fmt,
        "schema_version": 1,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _make_run(
    root: Path,
    name: str,
    *,
    started: str,
    symbol: str = "BTCUSDT",
    commit: str = "abc123",
    dirty: bool = False,
    source: str = "source-a",
    strategy_hash: str = "strategy-a",
    execution_hash: str = "execution-a",
    feature_hash: str = "feature-a",
    ending_equity: float = 1100.0,
) -> Path:
    run = root / name
    artifacts = run / "artifacts"
    provenance = run / "provenance"
    notes = run / "notes"
    run.mkdir()
    artifacts.mkdir()
    provenance.mkdir()
    notes.mkdir()

    trades = pd.DataFrame(
        {
            "side": ["LONG", "SHORT"],
            "pnl": [10.0, -4.0],
        }
    )
    signals = pd.DataFrame(
        {
            "signal_id": ["enter-1", "reject-1"],
            "decision": ["ENTER", "REJECT"],
        }
    )
    feature_context = pd.DataFrame(
        {
            "strategy_index": [0, 1],
            "plus_di": [31.0, 18.0],
            "minus_di": [12.0, 27.0],
            "adx": [24.0, 35.0],
        }
    )
    source_archives = pd.DataFrame(
        {
            "dataset": ["klines", "metrics"],
            "interval": ["1h", None],
            "archive_count": [31, 1],
        }
    )
    trades_path = artifacts / "trades.parquet"
    signals_path = artifacts / "signals.parquet"
    context_path = artifacts / "feature_context.parquet"
    source_path = provenance / "source_archives.parquet"
    _write_parquet(trades_path, trades)
    _write_parquet(signals_path, signals)
    _write_parquet(context_path, feature_context)
    _write_parquet(source_path, source_archives)

    trade_csv = run / "trade_list.csv"
    trades.to_csv(trade_csv, index=False)
    summary_path = run / "summary.json"
    atomic_json(
        summary_path,
        {
            "total_trades": 2,
            "wins": 1,
            "losses": 1,
            "win_rate": 0.5,
            "ending_equity": ending_equity,
            "total_net_r": 0.5,
            "average_net_r": 0.25,
        },
    )
    (notes / "diagnostic.txt").write_text("completed run note\n", encoding="utf-8")

    manifest = {
        "run_manifest_contract": "crypto_strategy_lab_run_v1",
        "run_manifest_version": 1,
        "run_status": "COMPLETED",
        "run_id": name,
        "run_started_at": started,
        "run_completed_at": started,
        "code_commit": commit,
        "code_dirty": dirty,
        "reproducibility_status": "PARTIAL" if dirty else "REPRODUCIBLE",
        "request": {
            "symbol": symbol,
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-02-01T00:00:00+00:00",
            "requested_strategy_interval": "4h",
            "requested_intrabar_interval": "1m",
            "effective_intrabar_interval": "1m",
        },
        "config": {"execution": {"initial_equity": 1000.0}},
        "hashes": {
            "strategy_hash": strategy_hash,
            "execution_hash": execution_hash,
            "feature_config_hash": feature_hash,
            "data_config_hash": "data-a",
        },
        "catalog": {"catalog_snapshot_digest": source},
        "features": {"core_directional": {"provider_version": 1}},
        "execution_result": {"completed_trade_rows": 2},
        "artifacts": {
            "trades": _artifact(trades_path, run, "parquet", 2),
            "signals": _artifact(signals_path, run, "parquet", 2),
            "feature_context": _artifact(context_path, run, "parquet", 2),
            "source_archives": _artifact(source_path, run, "parquet", 2),
            "trade_csv": _artifact(trade_csv, run, "csv", 2),
            "summary": _artifact(summary_path, run, "json"),
        },
        "research": {
            "artifact_contract": "feature_research_v1",
            "artifact_version": 1,
        },
    }
    if dirty:
        manifest["tracked_diff_sha256"] = f"diff-{name}"
        manifest["untracked_source_paths"] = []
    atomic_json(run / "run_manifest.json", manifest)
    return run


@pytest.fixture
def reports(tmp_path: Path) -> BacktestReports:
    _make_run(tmp_path, "run-one", started="2026-01-01T00:00:00+00:00")
    _make_run(
        tmp_path,
        "run-two",
        started="2026-01-02T00:00:00+00:00",
        symbol="ETHUSDT",
        source="source-b",
        execution_hash="execution-b",
        ending_equity=1025.0,
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "summary.json").write_text("{}", encoding="utf-8")
    return BacktestReports(tmp_path)


def test_paths_are_confined(reports: BacktestReports, tmp_path: Path):
    with pytest.raises(ValueError, match="traversal"):
        reports.resolve_run("../outside")
    with pytest.raises(ValueError, match="Absolute"):
        reports.resolve_run(str(tmp_path.resolve()))
    assert reports.resolve_run("run-one").name == "run-one"


def test_completed_run_listing_latest_and_reads(reports: BacktestReports):
    runs = reports.list_runs()
    assert [item["folder_name"] for item in runs] == ["run-two", "run-one"]
    assert reports.latest_run()["folder_name"] == "run-two"

    listed = reports.list_run_files("run-one")
    by_name = {item["filename"]: item for item in listed}
    assert "run_manifest.json" in by_name
    assert "artifacts/feature_context.parquet" in by_name
    assert "provenance/source_archives.parquet" in by_name
    assert "notes/diagnostic.txt" in by_name
    assert by_name["artifacts/feature_context.parquet"]["registered_artifact"] is True
    assert by_name["notes/diagnostic.txt"]["registered_artifact"] is False

    csv_report = reports.read_report("run-one", "trade_list.csv", limit=1)
    assert csv_report["columns"] == ["side", "pnl"]
    assert csv_report["truncated"] and csv_report["row_count"] == 2
    assert reports.read_report("run-one", "summary.json")["data"]["wins"] == 1
    assert reports.read_run_file("run-one", "run_manifest.json")["data"]["run_id"] == "run-one"
    assert "completed run note" in reports.read_run_file("run-one", "notes/diagnostic.txt")["text"]
    assert reports.get_run_manifest("run-one")["run_id"] == "run-one"


def test_parquet_preview_and_queries_cover_all_current_run_parquets(reports: BacktestReports):
    preview = reports.read_run_file("run-one", "artifacts/feature_context.parquet", limit=1)
    assert preview["format"] == "parquet"
    assert "plus_di" in preview["columns"]
    assert preview["row_count"] == 2

    selected = reports.query_trades(
        "run-one", "SELECT side, pnl FROM trades ORDER BY pnl DESC"
    )
    assert selected["rows"][0] == ["LONG", 10.0]
    assert reports.query_signals(
        "run-one", "SELECT count(*) FROM signals WHERE decision='ENTER'"
    )["rows"] == [[1]]
    assert reports.query_feature_context(
        "run-one",
        "SELECT strategy_index, plus_di, minus_di, adx FROM feature_context ORDER BY strategy_index",
    )["rows"][0] == [0, 31.0, 12.0, 24.0]
    assert reports.query_parquet(
        "run-one",
        "provenance/source_archives.parquet",
        "SELECT dataset, archive_count FROM data ORDER BY dataset",
    )["row_count"] == 2
    assert not hasattr(reports, "query_telemetry")


def test_run_file_and_sql_safety(reports: BacktestReports, tmp_path: Path):
    with pytest.raises(ValueError, match="traversal"):
        reports.read_run_file("run-one", "../run-two/summary.json")
    with pytest.raises(ValueError, match="Absolute"):
        reports.query_parquet("run-one", str(tmp_path / "outside.parquet"), "SELECT * FROM data")
    with pytest.raises(ValueError, match=".parquet"):
        reports.query_parquet("run-one", "summary.json", "SELECT * FROM data")

    forbidden = [
        "DELETE FROM trades",
        "SELECT * FROM trades; SELECT 1",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('/tmp/x.parquet')",
        "SELECT * FROM read_blob('/etc/passwd')",
        "ATTACH 'x.db' AS x",
        "COPY trades TO '/tmp/x.csv'",
        "PRAGMA version",
        "INSTALL httpfs",
        "LOAD httpfs",
    ]
    for sql in forbidden:
        with pytest.raises(ValueError):
            reports.query_trades("run-one", sql)


def test_symlinked_run_file_is_rejected(reports: BacktestReports, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = reports.resolve_run("run-one") / "notes" / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")
    with pytest.raises(ValueError, match="Symlinked"):
        reports.read_run_file("run-one", "notes/link.txt")
    assert "notes/link.txt" not in {item["filename"] for item in reports.list_run_files("run-one")}


def test_compare_uses_manifest_provenance_and_derives_net_pnl(reports: BacktestReports):
    compared = reports.compare_runs(["run-one", "run-two"])
    assert compared[0]["symbol"] == "BTCUSDT"
    assert compared[0]["total_trades"] == 2
    assert compared[0]["net_pnl"] == 100.0
    assert compared[1]["net_pnl"] == 25.0
    assert compared[1]["provenance_vs_first"]["same_sources"] is False
    assert compared[1]["provenance_vs_first"]["same_execution"] is False


def test_dirty_code_comparison_includes_diff_identity(tmp_path: Path):
    _make_run(tmp_path, "dirty-a", started="2026-01-01T00:00:00+00:00", dirty=True)
    _make_run(tmp_path, "dirty-b", started="2026-01-02T00:00:00+00:00", dirty=True)
    reports = BacktestReports(tmp_path)
    compared = reports.compare_runs(["dirty-a", "dirty-b"])
    assert compared[1]["provenance_vs_first"]["same_code"] is False


def test_create_server_registers_complete_read_only_tools(reports: BacktestReports):
    from mcp.server import MCPServer

    server = create_server(reports)
    assert isinstance(server, MCPServer)
    assert set(server._tool_manager._tools) == {
        "list_runs",
        "latest_run",
        "get_run_manifest",
        "list_run_files",
        "read_report",
        "read_run_file",
        "query_trades",
        "query_signals",
        "query_feature_context",
        "query_parquet",
        "research_aggregate",
        "compare_runs",
    }


def test_tampered_registered_parquet_is_rejected(reports: BacktestReports):
    path = reports.resolve_run("run-one") / "artifacts" / "trades.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="integrity"):
        reports.query_trades("run-one", "SELECT count(*) FROM trades")
    with pytest.raises(Exception, match="integrity"):
        reports.read_run_file("run-one", "artifacts/trades.parquet")


def test_package_does_not_eagerly_import_server_module():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mcp_server, sys; print('mcp_server.server' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
