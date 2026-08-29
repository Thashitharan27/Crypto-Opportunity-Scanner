"""Downstream-only publication of a canonical completed Data Lake run."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import duckdb
import pandas as pd

from .feature_research import (
    FEATURE_RESEARCH_ARTIFACT_CONTRACT,
    FEATURE_RESEARCH_ARTIFACT_VERSION,
    _trade_fingerprint,
    _write_parquet_atomic,
    write_research_artifacts,
)
from .report_workbooks import (
    build_backtest_workbook,
    build_performance_breakdowns,
    build_periodic_breakdown,
)
from .run_manifest import (
    CATALOG_SNAPSHOT_CONTRACT,
    PREPARED_CACHE_CONTRACT,
    RUN_MANIFEST_CONTRACT,
    RUN_MANIFEST_VERSION,
    atomic_json,
    canonical_sha256,
    capture_code_provenance,
    config_hashes,
    config_snapshot,
    catalog_entry,
    new_run_identity,
    runtime_provenance,
    selected_source_snapshot,
)
from .statistics import summarize


def _catalog_entry(
    path: Path,
    run_dir: Path,
    fmt: str,
    rows: int | None = None,
    schema_version: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    return catalog_entry(path, run_dir, fmt, rows, schema_version, **extra)


def _validate_research_artifacts(
    trades_path: Path,
    context_path: Path,
    research: dict[str, Any],
) -> None:
    """Run Task-16 semantic validation before any COMPLETED marker is published."""
    if (
        research.get("artifact_contract") != FEATURE_RESEARCH_ARTIFACT_CONTRACT
        or research.get("artifact_version") != FEATURE_RESEARCH_ARTIFACT_VERSION
    ):
        raise ValueError("incompatible feature research artifact contract")
    with duckdb.connect() as con:
        trade_columns = [
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(trades_path)]
            ).fetchall()
        ]
        context_columns = [
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(context_path)]
            ).fetchall()
        ]
        trade_rows = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(trades_path)]
        ).fetchone()[0]
        context_rows = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(context_path)]
        ).fetchone()[0]
        if int(trade_rows) != int(research.get("trade_row_count", -1)):
            raise ValueError("research trade row count mismatch")
        if int(context_rows) != int(research.get("feature_context_row_count", -1)):
            raise ValueError("research feature-context row count mismatch")
        if trade_columns != list(research.get("trade_columns", ())):
            raise ValueError("research trade schema mismatch")
        if context_columns != list(research.get("feature_context_columns", ())):
            raise ValueError("research feature-context schema mismatch")

        interval = pd.Timedelta(research["request"]["strategy_interval"])
        bad_context = con.execute(
            """
            WITH q AS (
              SELECT strategy_index, strategy_candle_open_time, decision_available_at,
                     lag(strategy_index) OVER (ORDER BY strategy_index) prior_i,
                     lag(strategy_candle_open_time) OVER (ORDER BY strategy_index) prior_t
              FROM read_parquet(?)
            )
            SELECT count(*) FROM q
            WHERE strategy_index < 0
               OR decision_available_at < strategy_candle_open_time
               OR decision_available_at > strategy_candle_open_time + ?
               OR (prior_i IS NULL AND strategy_index <> 0)
               OR (prior_i IS NOT NULL AND strategy_index <> prior_i + 1)
               OR (prior_t IS NOT NULL AND strategy_candle_open_time <= prior_t)
            """,
            [str(context_path), interval.to_pytimedelta()],
        ).fetchone()[0]
        if bad_context:
            raise ValueError("research feature-context timeline is invalid")

        mismatch = con.execute(
            """
            SELECT count(*)
            FROM read_parquet(?) t
            LEFT JOIN read_parquet(?) c
              ON t.research_signal_index=c.strategy_index
            WHERE c.strategy_index IS NULL
               OR t.research_signal_candle_open_time
                    IS DISTINCT FROM c.strategy_candle_open_time
               OR t.research_signal_available_at
                    IS DISTINCT FROM c.decision_available_at
            """,
            [str(trades_path), str(context_path)],
        ).fetchone()[0]
        if mismatch:
            raise ValueError("research trade/context causal attachment is invalid")

        parity_columns = research.get("trade_context_parity_columns", [])
        if not isinstance(parity_columns, list):
            raise ValueError("research parity column contract is invalid")
        available_trade = set(trade_columns)
        available_context = set(context_columns)
        for column in parity_columns:
            if column not in available_trade or column not in available_context:
                raise ValueError(f"research parity column unavailable: {column}")
            quoted = '"' + str(column).replace('"', '""') + '"'
            bad = con.execute(
                f"""
                SELECT count(*)
                FROM read_parquet(?) t
                JOIN read_parquet(?) c
                  ON t.research_signal_index=c.strategy_index
                WHERE t.{quoted} IS NOT NULL
                  AND c.{quoted} IS NOT NULL
                  AND t.{quoted} IS DISTINCT FROM c.{quoted}
                """,
                [str(trades_path), str(context_path)],
            ).fetchone()[0]
            if bad:
                raise ValueError(f"research value parity mismatch: {column}")

        nonfinite = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE NOT isfinite(pair_net_r)",
            [str(trades_path)],
        ).fetchone()[0]
        if nonfinite:
            raise ValueError("research trades contain non-finite pair_net_r")
        persisted = con.execute(
            "SELECT * FROM read_parquet(?)", [str(trades_path)]
        ).fetchdf()
    if _trade_fingerprint(persisted) != research.get("trade_fingerprint"):
        raise ValueError("research semantic trade fingerprint mismatch")


def _validate_signal_artifact(
    signals_path: Path, context_path: Path, trade_rows: int
) -> None:
    required = {
        "signal_id",
        "strategy_index",
        "candle_open_time",
        "decision_available_at",
        "side",
        "decision",
        "reason_code",
    }
    with duckdb.connect() as con:
        columns = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(signals_path)]
            ).fetchall()
        }
        missing = required - columns
        if missing:
            raise ValueError(f"signals artifact missing required columns: {sorted(missing)}")
        bad = con.execute(
            """
            SELECT count(*)
            FROM read_parquet(?) s
            LEFT JOIN read_parquet(?) c ON s.strategy_index=c.strategy_index
            WHERE c.strategy_index IS NULL
               OR s.candle_open_time IS DISTINCT FROM c.strategy_candle_open_time
               OR s.decision_available_at IS DISTINCT FROM c.decision_available_at
               OR upper(cast(s.decision AS VARCHAR)) NOT IN ('ENTER','REJECT')
            """,
            [str(signals_path), str(context_path)],
        ).fetchone()[0]
        if bad:
            raise ValueError("signals artifact causal attachment is invalid")
        entered = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE upper(cast(decision AS VARCHAR))='ENTER'",
            [str(signals_path)],
        ).fetchone()[0]
        if int(entered) != int(trade_rows):
            raise ValueError("entered signal count does not match completed trades")


def _build_human_workbook(
    summary: dict[str, Any],
    report_config: Any,
    run_dir: Path,
    trades: pd.DataFrame,
) -> Path:
    """Build the compact human report from completed modern trades only."""
    monthly = build_periodic_breakdown(trades, "ME")
    yearly = build_periodic_breakdown(trades, "YE")
    market_regime, direction_regime = build_performance_breakdowns(trades)
    return build_backtest_workbook(
        summary,
        report_config,
        run_dir,
        monthly,
        yearly,
        market_regime,
        direction_regime,
    )


class CsvManifestReporter:
    """Publish the minimal canonical run set, validate it, then publish the manifest."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.run_id: str | None = None
        self.run_started_at: str | None = None
        self.code_provenance: dict[str, Any] | None = None

    def begin(self, request, config) -> None:
        if self.run_id is not None:
            raise RuntimeError("reporter instances publish exactly one run")
        self.run_id, self.run_started_at = new_run_identity()
        self.code_provenance = capture_code_provenance()

    def report(self, result, context):
        started = time.perf_counter()
        if self.run_id is None:
            self.begin(result.request, context.config)
        self.output_root.mkdir(parents=True, exist_ok=True)
        run_dir = (
            self.output_root
            / f"{result.request.symbol}_{result.request.strategy_interval}_{self.run_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir()
        provenance_dir = run_dir / "provenance"
        provenance_dir.mkdir()

        # Task 16 owns the one authoritative trade/context Parquets.
        research = write_research_artifacts(
            run_dir, result, context, authoritative_layout=True
        )
        trades_path = artifacts_dir / "trades.parquet"
        context_path = artifacts_dir / "feature_context.parquet"
        _validate_research_artifacts(trades_path, context_path, research)

        trade_csv = run_dir / "trade_list.csv"
        result.trades.to_csv(trade_csv, index=False)

        signals = getattr(result, "signals", None)
        if signals is None:
            raise ValueError(
                "native run did not expose original-run signal provenance"
            )
        signals = pd.DataFrame(signals)
        signals_path = artifacts_dir / "signals.parquet"
        _write_parquet_atomic(signals, signals_path)
        _validate_signal_artifact(signals_path, context_path, len(result.trades))

        source_rows, source_digest = selected_source_snapshot(
            context.selected_source_records
        )
        source_frame = pd.DataFrame(source_rows)
        if source_frame.empty:
            source_frame = pd.DataFrame(
                {
                    name: pd.Series(dtype="string")
                    for name in (
                        "exchange",
                        "market",
                        "dataset",
                        "symbol",
                        "interval",
                        "frequency",
                        "period_start",
                        "period_end",
                        "raw_archive_fingerprint",
                        "canonical_partition_identity",
                    )
                }
            )
            source_frame["size_bytes"] = pd.Series(dtype="int64")
            source_frame["mtime_ns"] = pd.Series(dtype="int64")
        source_path = provenance_dir / "source_archives.parquet"
        _write_parquet_atomic(source_frame, source_path)

        initial_equity = float(context.config.execution.initial_equity)
        try:
            summary = summarize(result.trades, initial_equity)
        except (KeyError, TypeError):
            pnl = pd.to_numeric(
                result.trades.get("pair_net_pnl", pd.Series(dtype=float)),
                errors="coerce",
            )
            r = pd.to_numeric(
                result.trades.get("pair_net_r", pd.Series(dtype=float)),
                errors="coerce",
            )
            summary = {
                "total_trades": len(result.trades),
                "wins": int((pnl > 0).sum()),
                "losses": int((pnl < 0).sum()),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
                "total_net_r": float(r.sum()),
                "average_net_r": float(r.mean()) if len(r) else 0.0,
                "ending_equity": initial_equity + float(pnl.sum()),
            }
        summary["net_pnl"] = (
            float(summary.get("ending_equity", initial_equity)) - initial_equity
        )
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        quality_path = run_dir / "data_quality.json"
        atomic_json(
            quality_path,
            result.data_quality.to_dict()
            if result.data_quality
            else {"status": "NOT_AVAILABLE"},
        )

        request = result.request
        report_config = SimpleNamespace(
            run_name=context.config.reporting.run_name or self.run_id,
            input_csv=f"{request.symbol}.csv",
            strategy_timeframe_minutes=context.config.data.strategy_timeframe_minutes,
            intrabar_timeframe_minutes=context.config.data.intrabar_timeframe_minutes,
            use_intrabar_data=context.config.data.use_intrabar_data,
            initial_equity=initial_equity,
        )
        workbook = _build_human_workbook(
            summary,
            report_config,
            run_dir,
            result.trades,
        )

        with duckdb.connect() as con:
            parquet_count = con.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(trades_path)]
            ).fetchone()[0]
            csv_count = con.execute(
                "SELECT count(*) FROM read_csv_auto(?, header=true)",
                [str(trade_csv)],
            ).fetchone()[0]
        expected_count = len(result.trades)
        if (
            int(parquet_count) != expected_count
            or int(csv_count) != expected_count
            or int(summary.get("total_trades", -1)) != expected_count
        ):
            raise ValueError("trade CSV/Parquet/summary parity validation failed")

        effective_intrabar = getattr(
            context.bundle, "intrabar_interval", request.intrabar_interval
        )
        hashes = config_hashes(context.config, effective_intrabar)
        artifacts = {
            "trades": _catalog_entry(
                trades_path, run_dir, "parquet", expected_count
            ),
            "feature_context": _catalog_entry(
                context_path, run_dir, "parquet", len(context.prepared)
            ),
            "signals": _catalog_entry(
                signals_path,
                run_dir,
                "parquet",
                len(signals),
                collection_status="COLLECTED",
            ),
            "source_archives": _catalog_entry(
                source_path, run_dir, "parquet", len(source_frame)
            ),
            "trade_csv": _catalog_entry(
                trade_csv, run_dir, "csv", expected_count
            ),
            "summary": _catalog_entry(summary_path, run_dir, "json", None),
            "workbook": _catalog_entry(workbook, run_dir, "xlsx", None),
            "data_quality": _catalog_entry(
                quality_path, run_dir, "json", None
            ),
        }

        grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
        for row in source_rows:
            grouped.setdefault((row["dataset"], row["interval"]), []).append(row)
        source_summaries = [
            {
                "dataset": key[0],
                "interval": key[1],
                "partition_count": len(rows),
                "source_signature": canonical_sha256(rows),
                "identity_version": 1,
            }
            for key, rows in sorted(grouped.items(), key=str)
        ]
        manifest = {
            "run_manifest_contract": RUN_MANIFEST_CONTRACT,
            "run_manifest_version": RUN_MANIFEST_VERSION,
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
            "run_completed_at": datetime.now(timezone.utc).isoformat(),
            "run_status": "COMPLETED",
            **(self.code_provenance or {}),
            "runtime": runtime_provenance(),
            "request": {
                "symbol": request.symbol,
                "market": getattr(
                    getattr(request, "market", None),
                    "value",
                    getattr(request, "market", None),
                ),
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "requested_strategy_interval": request.strategy_interval,
                "requested_intrabar_interval": request.intrabar_interval,
                "effective_intrabar_interval": effective_intrabar,
            },
            "config": config_snapshot(context.config),
            "hashes": hashes,
            "catalog": {
                "catalog_contract_version": CATALOG_SNAPSHOT_CONTRACT,
                "catalog_snapshot_digest": source_digest,
                "selected_archive_count": len(source_rows),
                "source_archives_artifact": "provenance/source_archives.parquet",
                "source_archives_sha256": artifacts["source_archives"]["sha256"],
                "datasets": source_summaries,
            },
            "features": result.feature_cache_metadata,
            "prepared": {
                "prepared_cache_key": result.prepared_cache_key,
                "prepared_cache_contract": PREPARED_CACHE_CONTRACT,
                "prepared_cache_version": 1,
                "prepared_rows": result.prepared_rows,
            },
            "execution_result": {
                "strategy_rows": result.strategy_rows,
                "intrabar_rows": result.intrabar_rows,
                "completed_trade_rows": expected_count,
                "trade_fingerprint": research["trade_fingerprint"],
                "trade_fingerprint_contract": research[
                    "trade_fingerprint_contract"
                ],
                "stage_timings": {
                    **result.stage_timings,
                    "reporting": time.perf_counter() - started,
                },
            },
            "artifacts": artifacts,
            "research": research,
        }
        # Completion marker is deliberately the final write and atomic rename.
        atomic_json(run_dir / "run_manifest.json", manifest)
        object.__setattr__(result, "output_dir", run_dir)
