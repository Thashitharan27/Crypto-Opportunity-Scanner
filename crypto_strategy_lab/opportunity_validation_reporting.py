"""Deterministic artifact writer for Task 8 validation results."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .opportunity_validation import OpportunityValidationResult


def write_validation_reports(directory: Path, result: OpportunityValidationResult) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reports = {
        "validation_overall.csv": result.overall,
        "validation_by_rank.csv": result.by_rank,
        "validation_top_k.csv": result.top_k,
        "validation_rank_decay.csv": result.rank_decay,
        "validation_by_year.csv": result.by_year,
        "validation_by_regime.csv": result.by_regime,
        "validation_components.csv": result.components,
    }
    for name, frame in reports.items():
        frame.to_csv(directory / name, index=False, lineterminator="\n")
    observations = result.observations
    def values(column: str) -> list[str]:
        if column not in observations:
            return []
        return sorted(set(observations[column].dropna().astype(str)))
    research_sources: set[str] = set()
    for encoded in values("research_source_identities"):
        research_sources.update(json.loads(encoded))
    summary = {
        "contract": "opportunity_validation_v1",
        "config": asdict(result.config),
        "natural_key": ["decision_timestamp", "symbol"],
        "definitions": {
            "candidate_to_entry_conversion": "valid entries divided by candidates with an authoritative entry observation",
            "setup_frequency": "same per-candidate valid-entry frequency; retained explicitly for selection-stage comparison",
            "opportunity_capture": "sum of direction-neutral canonical absolute movement for selected observations divided by that of eligible baseline observations in the same stratum",
            "expectancy": "arithmetic mean of authoritative Strategy Lab trade_expectancy_value over completed trades",
            "rank_correlation": "Spearman ordinal correlation using average ranks and pairwise complete observations",
        },
        "observation_count": len(result.observations),
        "decision_timestamps": values("decision_timestamp"),
        "scan_run_ids": values("scan_run_id"),
        "scanner_source_identities": values("scanner_source_identity"),
        "research_run_ids": values("research_run_id"),
        "research_source_identities": sorted(research_sources),
        "evaluation_horizons": values("evaluation_horizon"),
        "associations": result.associations.where(pd.notna(result.associations), None).to_dict("records"),
    }
    (directory / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")
