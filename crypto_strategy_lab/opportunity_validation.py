"""Task 8 causal validation of frozen opportunity scans against research results.

This module deliberately does not calculate indicators, signals, regimes, or trade
outcomes.  It joins Task 7 scanner facts to already-produced Strategy Lab facts on
the natural ``(decision_timestamp, symbol)`` key and only aggregates them.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .run_manifest import artifact_path, load_completed_manifest


@dataclass(frozen=True, slots=True)
class OpportunityValidationConfig:
    """Immutable research presentation choices (never selection thresholds)."""

    top_ks: tuple[int, ...] = (5, 10, 20, 30)
    rank_bucket_size: int = 5
    minimum_sample_warning: int = 30

    def __post_init__(self) -> None:
        values = tuple(sorted(set(self.top_ks)))
        if not values or any(value <= 0 for value in values):
            raise ValueError("top_ks must contain positive integers")
        if self.rank_bucket_size <= 0 or self.minimum_sample_warning <= 0:
            raise ValueError("bucket and warning sizes must be positive")
        object.__setattr__(self, "top_ks", values)


@dataclass(frozen=True, slots=True)
class OpportunityValidationResult:
    config: OpportunityValidationConfig
    observations: pd.DataFrame
    overall: pd.DataFrame
    by_rank: pd.DataFrame
    top_k: pd.DataFrame
    rank_decay: pd.DataFrame
    by_year: pd.DataFrame
    by_regime: pd.DataFrame
    components: pd.DataFrame
    associations: pd.DataFrame


def load_scanner_observations(run_directories: Iterable[Path]) -> pd.DataFrame:
    """Load only catalog-verified Task 7 OPPORTUNITY_SCAN artifacts.

    One scanner row is emitted per universe member and decision. Score/component
    values remain the frozen Task 4 values; this loader never reranks candidates.
    """
    output: list[pd.DataFrame] = []
    for directory in sorted(map(Path, run_directories), key=lambda p: p.as_posix()):
        manifest = load_completed_manifest(directory)
        scan = manifest.get("opportunity_scan", {})
        if manifest.get("run_type") != "OPPORTUNITY_SCAN" or scan.get("discovery_mode") != "HISTORICAL":
            raise ValueError("validation requires historical OPPORTUNITY_SCAN runs")
        decision = pd.Timestamp(scan["decision_timestamp"])
        if decision.tzinfo is None:
            raise ValueError("decision timestamp must be timezone-aware")
        universe = pd.read_csv(artifact_path(directory, manifest, "universe_snapshot"))
        preliminary = pd.read_csv(artifact_path(directory, manifest, "preliminary_candidates"))
        final = pd.read_csv(artifact_path(directory, manifest, "final_candidates"))
        universe["symbol"] = universe["symbol"].astype(str).str.upper()
        frame = universe[["symbol", "eligible", "discovery_rank", "reference_available_at", "discovery_source_identity"]].copy()
        if frame.eligible.dtype == object:
            normalized = frame.eligible.astype(str).str.lower().map({"true": True, "false": False})
            if normalized.isna().any():
                raise ValueError("universe artifact has invalid eligible values")
            frame["eligible"] = normalized
        frame["decision_timestamp"] = decision
        frame["scan_run_id"] = manifest["run_id"]
        frame["scanner_source_identity"] = frame["discovery_source_identity"]
        frame["preliminary"] = frame.symbol.isin(set(preliminary.symbol.astype(str).str.upper()))
        if not preliminary.empty and "strategy_source_identity" in preliminary:
            strategy_sources = preliminary.assign(
                symbol=preliminary.symbol.astype(str).str.upper()
            ).set_index("symbol")["strategy_source_identity"]
            frame["scanner_source_identity"] = frame.symbol.map(strategy_sources).combine_first(
                frame.scanner_source_identity
            )
        finals = final.assign(symbol=final.symbol.astype(str).str.upper()).set_index("symbol")
        frame["final"] = frame.symbol.isin(finals.index)
        frame["final_rank"] = frame.symbol.map(finals.get("final_rank", pd.Series(dtype=float)))
        if "opportunity_scores" in manifest["artifacts"]:
            scores = pd.read_csv(artifact_path(directory, manifest, "opportunity_scores"))
            if not scores.empty:
                scores["symbol"] = scores.symbol.astype(str).str.upper()
                for row in scores.itertuples(index=False):
                    mask = frame.symbol.eq(row.symbol)
                    prefix = f"score__{row.model_name}_v{row.model_version}"
                    frame.loc[mask, prefix] = row.score
                    frame.loc[mask, f"rank__{row.model_name}_v{row.model_version}"] = row.model_rank
                    raw = json.loads(row.raw_components) if isinstance(row.raw_components, str) else {}
                    for name, value in raw.items():
                        frame.loc[mask, f"component__{name}"] = value
        available = pd.to_datetime(frame.reference_available_at, utc=True, errors="coerce")
        if (available.dropna() > decision).any():
            raise ValueError("scanner artifact contains information available after its decision")
        output.append(frame.drop(columns=["discovery_source_identity", "reference_available_at"]))
    if not output:
        return pd.DataFrame()
    combined = pd.concat(output, ignore_index=True)
    if combined.duplicated(["decision_timestamp", "symbol"]).any():
        raise ValueError("duplicate natural research key across scanner runs")
    return combined.sort_values(["decision_timestamp", "symbol"], kind="stable").reset_index(drop=True)


def join_research_results(scanner: pd.DataFrame, research: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative research outcomes without recomputing strategy logic.

    Required research columns are ``decision_timestamp``, ``symbol``,
    ``valid_entry``, Task 4's canonical ``forward_max_abs_excursion_pct``,
    ``trade_expectancy_value`` and ``market_regime``. A null expectancy value
    means no completed trade.
    """
    required = {"decision_timestamp", "symbol", "valid_entry", "forward_max_abs_excursion_pct",
                "trade_expectancy_value", "market_regime"}
    missing = required - set(research.columns)
    if missing:
        raise ValueError(f"research results missing authoritative columns: {sorted(missing)}")
    left, right = scanner.copy(), research.copy()
    for frame in (left, right):
        frame["decision_timestamp"] = pd.to_datetime(frame.decision_timestamp, utc=True)
        frame["symbol"] = frame.symbol.astype(str).str.upper()
    if right.duplicated(["decision_timestamp", "symbol"]).any():
        raise ValueError("duplicate natural research key in research results")
    if "entry_timestamp" in right:
        entry = pd.to_datetime(right.entry_timestamp, utc=True, errors="coerce")
        if (entry.notna() & (entry < right.decision_timestamp)).any():
            raise ValueError("strategy entry predates scan decision")
    if "outcome_available_at" in right:
        available = pd.to_datetime(right.outcome_available_at, utc=True, errors="coerce")
        if (available.notna() & (available < right.decision_timestamp)).any():
            raise ValueError("research outcome predates scan decision")
    joined = left.merge(right, on=["decision_timestamp", "symbol"], how="left", validate="one_to_one")
    joined["absolute_movement"] = joined.pop("forward_max_abs_excursion_pct")
    joined["year"] = joined.decision_timestamp.dt.year
    joined["valid_entry"] = joined.valid_entry.astype("boolean")
    joined["absolute_movement"] = pd.to_numeric(joined.absolute_movement, errors="coerce")
    joined["trade_expectancy_value"] = pd.to_numeric(joined.trade_expectancy_value, errors="coerce")
    if "trade_won" not in joined:
        joined["trade_won"] = pd.NA
    return joined.sort_values(["decision_timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[int, float | None]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 2 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return len(pair), None
    return len(pair), float(pair.x.rank(method="average").corr(pair.y.rank(method="average")))


def _metrics(selected: pd.DataFrame, baseline: pd.DataFrame, minimum: int) -> dict[str, object]:
    observed = selected[selected.absolute_movement.notna()]
    entries = selected.valid_entry.dropna()
    trades = selected[selected.trade_expectancy_value.notna()]
    denominator = baseline.groupby("decision_timestamp").absolute_movement.sum(min_count=1).sum(min_count=1)
    numerator = observed.absolute_movement.sum(min_count=1)
    capture = None if pd.isna(denominator) or denominator == 0 or pd.isna(numerator) else float(numerator / denominator)
    return {
        "sample_count": int(len(selected)), "movement_sample_count": int(len(observed)),
        "valid_entry_count": int((entries == True).sum()),  # noqa: E712
        "candidate_to_entry_conversion": float(entries.mean()) if len(entries) else None,
        "setup_frequency": float(entries.mean()) if len(entries) else None,
        "mean_absolute_movement": float(observed.absolute_movement.mean()) if len(observed) else None,
        "median_absolute_movement": float(observed.absolute_movement.median()) if len(observed) else None,
        "opportunity_capture": capture, "trade_count": int(len(trades)),
        "expectancy": float(trades.trade_expectancy_value.mean()) if len(trades) else None,
        "win_rate": float(trades.trade_won.astype("boolean").mean()) if len(trades) and trades.trade_won.notna().any() else None,
        "small_sample_warning": bool(len(selected) < minimum),
    }


def validate_opportunities(observations: pd.DataFrame, config: OpportunityValidationConfig = OpportunityValidationConfig()) -> OpportunityValidationResult:
    """Aggregate existing scanner/research facts deterministically."""
    frame = observations.copy()
    if frame.empty:
        empty = pd.DataFrame()
        return OpportunityValidationResult(config, frame, empty, empty, empty, empty, empty, empty, empty, empty)
    required = {"decision_timestamp", "symbol", "eligible", "preliminary", "final", "final_rank",
                "valid_entry", "absolute_movement", "trade_expectancy_value", "market_regime", "year"}
    if required - set(frame):
        raise ValueError(f"validation observations missing columns: {sorted(required-set(frame))}")
    frame = frame.sort_values(["decision_timestamp", "symbol"], kind="stable").reset_index(drop=True)
    baseline = frame[frame.eligible.astype(bool)]
    stages = (("eligible_universe", baseline), ("preliminary_shortlist", frame[frame.preliminary.astype(bool)]),
              ("final_candidates", frame[frame.final.astype(bool)]))
    overall = pd.DataFrame([{"selection": name, **_metrics(group, baseline, config.minimum_sample_warning)} for name, group in stages])
    final = frame[frame.final.astype(bool) & frame.final_rank.notna()].copy()
    final["final_rank"] = pd.to_numeric(final.final_rank)
    by_rank = pd.DataFrame([{"rank": int(rank), **_metrics(group, baseline[baseline.decision_timestamp.isin(group.decision_timestamp)], config.minimum_sample_warning)}
                            for rank, group in final.groupby("final_rank", sort=True)])
    top_rows = []
    for k in config.top_ks:
        chosen = final[final.final_rank <= k]
        top_rows.append({"top_k": k, "max_rank_present": int(chosen.final_rank.max()) if len(chosen) else None,
                         **_metrics(chosen, baseline, config.minimum_sample_warning)})
    top_k = pd.DataFrame(top_rows)
    if len(final):
        starts = ((final.final_rank.astype(int)-1)//config.rank_bucket_size)*config.rank_bucket_size+1
        final["rank_bucket"] = starts.map(lambda n: f"{n}-{n+config.rank_bucket_size-1}")
    decay = pd.DataFrame([{"rank_bucket": bucket, **_metrics(group, baseline[baseline.decision_timestamp.isin(group.decision_timestamp)], config.minimum_sample_warning)}
                          for bucket, group in final.groupby("rank_bucket", sort=False)]) if len(final) else pd.DataFrame()
    def split(column: str, label: str) -> pd.DataFrame:
        rows=[]
        for value, stratum in frame.dropna(subset=[column]).groupby(column, sort=True):
            base = stratum[stratum.eligible.astype(bool)]
            for name, group in (("eligible_universe", base), ("preliminary_shortlist", stratum[stratum.preliminary.astype(bool)]), ("final_candidates", stratum[stratum.final.astype(bool)])):
                rows.append({label: value, "selection": name, **_metrics(group, base, config.minimum_sample_warning)})
        return pd.DataFrame(rows)
    by_year, by_regime = split("year", "year"), split("market_regime", "market_regime")
    associations=[]
    for outcome in ("absolute_movement", "valid_entry", "trade_expectancy_value"):
        n, rho = _spearman(final.final_rank, final[outcome])
        associations.append({"predictor":"final_rank", "outcome":outcome, "sample_count":n, "spearman_rho":rho})
    predictors=[c for c in frame if c.startswith("score__") or c.startswith("rank__")]
    for predictor in sorted(predictors):
        for outcome in ("absolute_movement", "valid_entry", "trade_expectancy_value"):
            n,rho=_spearman(frame[predictor],frame[outcome])
            associations.append({"predictor":predictor,"outcome":outcome,"sample_count":n,"spearman_rho":rho})
    component_cols=sorted(c for c in frame if c.startswith("component__"))
    component_rows=[]
    for i,left in enumerate(component_cols):
        for right in component_cols[i+1:]:
            n,rho=_spearman(frame[left],frame[right])
            component_rows.append({"component_a":left.removeprefix("component__"),"component_b":right.removeprefix("component__"),"sample_count":n,"spearman_rho":rho})
    return OpportunityValidationResult(config, frame, overall, by_rank, top_k, decay, by_year, by_regime,
                                       pd.DataFrame(component_rows), pd.DataFrame(associations))
