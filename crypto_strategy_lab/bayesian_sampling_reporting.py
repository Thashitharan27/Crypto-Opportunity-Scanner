"""Completed-run publication for optional Bayesian market-grid research samples."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

from crypto_strategy_lab.bayesian_sampling import generate_bayesian_research_samples
from crypto_strategy_lab.feature_research import _write_parquet_atomic
from crypto_strategy_lab.prepared_backtest import intrabar_from_data_lake_bundle
from crypto_strategy_lab.research_adapters import native_simulator_config
from crypto_strategy_lab.research_reporting import CsvManifestReporter, _catalog_entry
from crypto_strategy_lab.research_sampling_reporting import append_research_sampling_artifacts
from crypto_strategy_lab.run_manifest import atomic_json


_DEEP_LEVELS = {"DEEP", "DEEP_RESEARCH"}


def bayesian_sampling_enabled(reporting_config) -> bool:
    """Use the existing explicit Deep Research intent as the opt-in boundary."""
    return str(getattr(reporting_config, "analysis_level", "")).upper() in _DEEP_LEVELS


def _stable_empty_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if not samples.empty or len(samples.columns):
        return samples
    return pd.DataFrame(
        {
            "bayes_sample_id": pd.Series(dtype="string"),
            "bayes_sample_population": pd.Series(dtype="string"),
            "bayes_sample_version": pd.Series(dtype="string"),
            "bayes_sampling_side": pd.Series(dtype="string"),
            "bayes_sampling_interval_minutes": pd.Series(dtype="int64"),
            "bayes_sampling_strategy_timeframe_minutes": pd.Series(dtype="int64"),
            "side": pd.Series(dtype="string"),
            "research_signal_index": pd.Series(dtype="int64"),
            "entry_time": pd.Series(dtype="datetime64[ns]"),
            "exit_time": pd.Series(dtype="datetime64[ns]"),
            "pair_net_pnl": pd.Series(dtype="float64"),
            "pair_net_r": pd.Series(dtype="float64"),
            "bayes_long_probability": pd.Series(dtype="float64"),
            "bayes_short_probability": pd.Series(dtype="float64"),
        }
    )


def _validate_sampling_artifact(samples: pd.DataFrame) -> None:
    required = {
        "bayes_sample_id",
        "bayes_sample_population",
        "bayes_sampling_side",
        "bayes_sampling_interval_minutes",
        "bayes_sampling_strategy_timeframe_minutes",
        "side",
        "research_signal_index",
        "entry_time",
        "exit_time",
        "pair_net_pnl",
        "pair_net_r",
        "bayes_long_probability",
        "bayes_short_probability",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Bayes research sample artifact is missing columns: {missing}")
    if samples.empty:
        return
    if samples["bayes_sample_id"].astype(str).duplicated().any():
        raise ValueError("Bayes research sample IDs are not unique")
    entries = pd.to_datetime(samples["entry_time"], utc=True)
    exits = pd.to_datetime(samples["exit_time"], utc=True)
    if (exits < entries).any():
        raise ValueError("Bayes research sample artifact contains exit-before-entry rows")
    if set(samples["side"].astype(str).str.upper()) - {"LONG", "SHORT"}:
        raise ValueError("Bayes research sample artifact contains an unsupported direction")


class BayesianSamplingCsvManifestReporter(CsvManifestReporter):
    """Canonical reporter plus optional downstream research-sampling artifacts.

    Normal strategy trades are published exactly as before. Deep Research may add
    the direction-neutral market grid, while ReportingConfig.research_sampling_mode
    independently adds strategy-valid resilience samples with episode labels.
    """

    def report(self, result, context):
        if not bayesian_sampling_enabled(context.config.reporting):
            super().report(result, context)
            append_research_sampling_artifacts(result, context)
            return

        sampling_started = time.perf_counter()
        native_config = native_simulator_config(
            context.config.data,
            context.config.features,
            context.config.strategy,
            context.config.execution,
        )
        intrabar = intrabar_from_data_lake_bundle(context.bundle)
        samples = generate_bayesian_research_samples(
            context.prepared,
            intrabar,
            native_config,
            sampling_interval_minutes=None,
            directions=("LONG", "SHORT"),
        )
        sampling_seconds = time.perf_counter() - sampling_started
        sampling_metadata = dict(samples.attrs.get("bayesian_sampling", {}))
        samples = _stable_empty_samples(samples)
        _validate_sampling_artifact(samples)

        # Let the mature reporter publish and validate the normal strategy run
        # first. The atomic rewrites below add optional research artifacts without
        # changing the simulator result or the canonical strategy-trade artifact.
        super().report(result, context)
        run_dir = Path(result.output_dir)
        sample_path = run_dir / "artifacts" / "bayes_research_samples.parquet"
        _write_parquet_atomic(samples, sample_path)

        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["bayes_research_samples"] = _catalog_entry(
            sample_path,
            run_dir,
            "parquet",
            len(samples),
            collection_status="COLLECTED",
            research_population="MARKET_GRID",
        )
        execution = manifest.setdefault("execution_result", {})
        execution["bayes_research_sample_rows"] = len(samples)
        timings = execution.setdefault("stage_timings", {})
        timings["bayes_research_sampling"] = sampling_seconds
        research = manifest.setdefault("research", {})
        research["bayes_research_sampling"] = {
            "enabled": True,
            "artifact": "artifacts/bayes_research_samples.parquet",
            "rows": len(samples),
            "population": "MARKET_GRID",
            "directions": sampling_metadata.get("directions", ["LONG", "SHORT"]),
            "strategy_timeframe_minutes": sampling_metadata.get(
                "strategy_timeframe_minutes",
                int(context.config.data.strategy_timeframe_minutes),
            ),
            "sampling_interval_minutes": sampling_metadata.get(
                "sampling_interval_minutes",
                int(context.config.data.strategy_timeframe_minutes),
            ),
            "overlap_allowed": True,
            "independent_equity": True,
            "end_of_data_samples_censored": True,
            "entry_policy": "AFTER_COMPLETED_STRATEGY_CANDLE",
        }
        manifest["run_completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        append_research_sampling_artifacts(result, context)
