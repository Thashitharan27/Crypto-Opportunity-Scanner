"""Publication of strategy-aware research-sampling artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

from crypto_strategy_lab.feature_research import _write_parquet_atomic
from crypto_strategy_lab.prepared_backtest import intrabar_from_data_lake_bundle
from crypto_strategy_lab.research_adapters import native_simulator_config
from crypto_strategy_lab.research_reporting import _catalog_entry
from crypto_strategy_lab.research_sampling import (
    RESEARCH_SAMPLING_MODES,
    build_context_breakdown,
    build_episode_table,
    build_sampling_summary,
    generate_strategy_research_samples,
)
from crypto_strategy_lab.run_manifest import atomic_json


def research_sampling_enabled(reporting_config) -> bool:
    return str(getattr(reporting_config, "research_sampling_mode", "PORTFOLIO")).upper() != "PORTFOLIO"


def _stable_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if not samples.empty:
        return samples
    return pd.DataFrame({
        "research_sample_id": pd.Series(dtype="string"),
        "research_sampling_version": pd.Series(dtype="string"),
        "research_sampling_mode": pd.Series(dtype="string"),
        "research_episode_id": pd.Series(dtype="string"),
        "research_episode_entry_number": pd.Series(dtype="int64"),
        "research_episode_viable_entries": pd.Series(dtype="int64"),
        "research_signal_index": pd.Series(dtype="int64"),
        "strategy_profile_key": pd.Series(dtype="string"),
        "side": pd.Series(dtype="string"),
        "entry_time": pd.Series(dtype="datetime64[ns]"),
        "exit_time": pd.Series(dtype="datetime64[ns]"),
        "pair_net_r": pd.Series(dtype="float64"),
        "pair_net_pnl": pd.Series(dtype="float64"),
    })


def _validate_samples(samples: pd.DataFrame) -> None:
    required = {
        "research_sample_id", "research_sampling_mode", "research_episode_id",
        "research_episode_entry_number", "research_signal_index", "side",
        "entry_time", "exit_time", "pair_net_r",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"research sampling artifact is missing columns: {missing}")
    if samples.empty:
        return
    if samples["research_sample_id"].astype(str).duplicated().any():
        raise ValueError("research sampling IDs are not unique")
    entries = pd.to_datetime(samples["entry_time"], utc=True)
    exits = pd.to_datetime(samples["exit_time"], utc=True)
    if (exits < entries).any():
        raise ValueError("research sampling artifact contains exit-before-entry rows")
    if set(samples["side"].astype(str).str.upper()) - {"LONG", "SHORT"}:
        raise ValueError("research sampling artifact contains an unsupported direction")


def _episode_reporting_context(
    episodes: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Add explicit calendar/regime cluster context and summary distributions."""
    result = episodes.copy()
    if result.empty:
        result["episode_start_year"] = pd.Series(dtype="Int64")
        result["market_regime"] = pd.Series(dtype="string")
        return result, {
            "episodes_by_year": {},
            "episodes_by_regime": {},
            "average_viable_entries_per_episode": None,
            "median_viable_entries_per_episode": None,
            "entry_to_episode_ratio": None,
            "bayesian_effective_cluster_units": 0,
        }

    starts = pd.to_datetime(result["episode_start_time"], utc=True, errors="coerce")
    result["episode_start_year"] = starts.dt.year.astype("Int64")

    if "market_regime" in samples.columns:
        regime_by_episode = (
            samples[["research_episode_id", "market_regime"]]
            .dropna(subset=["market_regime"])
            .drop_duplicates("research_episode_id", keep="first")
            .set_index("research_episode_id")["market_regime"]
        )
        result["market_regime"] = (
            result["research_episode_id"].map(regime_by_episode).astype("string")
        )
    else:
        result["market_regime"] = pd.Series(pd.NA, index=result.index, dtype="string")

    year_counts = result["episode_start_year"].dropna().value_counts().sort_index()
    regime_counts = result["market_regime"].dropna().value_counts().sort_index()
    episode_count = len(result)
    selected_entry_count = len(samples)
    return result, {
        "episodes_by_year": {
            str(int(year)): int(count) for year, count in year_counts.items()
        },
        "episodes_by_regime": {
            str(regime): int(count) for regime, count in regime_counts.items()
        },
        "average_viable_entries_per_episode": float(result["viable_entries"].mean()),
        "median_viable_entries_per_episode": float(result["viable_entries"].median()),
        "entry_to_episode_ratio": selected_entry_count / episode_count,
        "bayesian_effective_cluster_units": int(episode_count),
    }


def append_research_sampling_artifacts(result, context) -> None:
    """Append optional strategy-resilience artifacts to an already-completed run."""
    reporting = context.config.reporting
    if not research_sampling_enabled(reporting):
        return

    mode = str(reporting.research_sampling_mode).upper()
    if mode not in RESEARCH_SAMPLING_MODES:
        raise ValueError(f"unsupported research sampling mode: {mode}")
    interval = int(reporting.research_sampling_interval_candles)
    if interval <= 0:
        raise ValueError("research sampling interval must be positive")

    started = time.perf_counter()
    native_config = native_simulator_config(
        context.config.data,
        context.config.features,
        context.config.strategy,
        context.config.execution,
    )
    intrabar = intrabar_from_data_lake_bundle(context.bundle)
    samples = generate_strategy_research_samples(
        context.prepared,
        intrabar,
        native_config,
        mode=mode,
        interval_candles=interval,
    )
    metadata = dict(samples.attrs.get("research_sampling", {}))
    samples = _stable_samples(samples)
    _validate_samples(samples)
    episodes = build_episode_table(samples)
    episodes, episode_reporting = _episode_reporting_context(episodes, samples)
    context_breakdown = build_context_breakdown(samples)
    summary = build_sampling_summary(samples, metadata)
    summary.update(episode_reporting)
    elapsed = time.perf_counter() - started

    run_dir = Path(result.output_dir)
    artifacts_dir = run_dir / "artifacts"
    samples_path = artifacts_dir / "research_sampling_trades.parquet"
    episodes_path = artifacts_dir / "research_sampling_episodes.parquet"
    context_path = run_dir / "research_sampling_context.csv"
    summary_path = run_dir / "research_sampling_summary.json"

    _write_parquet_atomic(samples, samples_path)
    _write_parquet_atomic(episodes, episodes_path)
    context_breakdown.to_csv(context_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["research_sampling_trades"] = _catalog_entry(
        samples_path,
        run_dir,
        "parquet",
        len(samples),
        collection_status="COLLECTED",
        research_population="STRATEGY_VIABLE",
    )
    manifest["artifacts"]["research_sampling_episodes"] = _catalog_entry(
        episodes_path,
        run_dir,
        "parquet",
        len(episodes),
        collection_status="COLLECTED",
        research_population="STRATEGY_EPISODES",
    )
    manifest["artifacts"]["research_sampling_context"] = _catalog_entry(
        context_path,
        run_dir,
        "csv",
        len(context_breakdown),
        collection_status="COLLECTED",
        research_population="STRATEGY_VIABLE_BREAKDOWNS",
    )
    manifest["artifacts"]["research_sampling_summary"] = _catalog_entry(
        summary_path,
        run_dir,
        "json",
        1,
        collection_status="COLLECTED",
        research_population="STRATEGY_RESILIENCE_SUMMARY",
    )
    execution = manifest.setdefault("execution_result", {})
    execution["research_sampling_rows"] = len(samples)
    execution["research_sampling_episode_rows"] = len(episodes)
    execution.setdefault("stage_timings", {})["research_sampling"] = elapsed
    manifest.setdefault("research", {})["strategy_research_sampling"] = {
        **metadata,
        "artifact": "artifacts/research_sampling_trades.parquet",
        "episode_artifact": "artifacts/research_sampling_episodes.parquet",
        "context_report": "research_sampling_context.csv",
        "summary": "research_sampling_summary.json",
        "selected_rows": len(samples),
        "episode_rows": len(episodes),
        "portfolio_metrics_valid": False,
        "equity_curve_valid": False,
        "drawdown_valid": False,
        "compounded_return_valid": False,
        "bayesian_cluster_key": "research_episode_id",
        "bayesian_effective_cluster_units": episode_reporting[
            "bayesian_effective_cluster_units"
        ],
    }
    manifest["run_completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)