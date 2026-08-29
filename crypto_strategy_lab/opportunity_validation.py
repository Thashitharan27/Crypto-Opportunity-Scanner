"""Task 8 causal validation of frozen opportunity scans against research results.

This module deliberately does not calculate indicators, signals, regimes, or trade
outcomes.  It joins Task 7 scanner facts to already-produced Strategy Lab facts on
the natural ``(decision_timestamp, symbol)`` key and only aggregates them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

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


@dataclass(frozen=True, slots=True)
class StrategyResearchSource:
    """Existing Strategy Lab facts and their immutable run provenance.

    ``trades`` is the native research/backtest trade frame (``entry_time`` and
    ``pair_net_r``/``pair_net_pnl``). ``outcomes`` contains Task 4
    ``OpportunityOutcome`` rows and defines which scan keys were evaluated.
    """

    run_id: str
    evaluation_horizon: str
    trades: pd.DataFrame = field(repr=False, compare=False)
    outcomes: pd.DataFrame | Sequence[object] = field(repr=False, compare=False)
    source_identities: tuple[str, ...] = ()
    symbol: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("research run_id must not be empty")
        horizon = pd.Timedelta(self.evaluation_horizon)
        if horizon <= pd.Timedelta(0):
            raise ValueError("research evaluation_horizon must be positive")
        object.__setattr__(self, "evaluation_horizon", str(horizon))
        object.__setattr__(self, "source_identities", tuple(sorted(set(self.source_identities))))
        if self.symbol is not None:
            object.__setattr__(self, "symbol", self.symbol.strip().upper())


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
        base_columns = ["symbol", "eligible", "discovery_rank", "reference_available_at", "discovery_source_identity"]
        frame = universe[base_columns].copy()
        discovery_metrics = {
            "range_percent": "discovery__range_percent",
            "absolute_price_change_percent": "discovery__absolute_price_change_percent",
            "quote_volume": "discovery__quote_volume",
            "spread_percent": "discovery__spread_percent",
            "listing_age_seconds": "discovery__listing_age_seconds",
        }
        for source_name, output_name in discovery_metrics.items():
            if source_name in universe:
                frame[output_name] = pd.to_numeric(universe[source_name], errors="coerce")
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
                    if "market_regime" not in frame:
                        frame["market_regime"] = pd.NA
                    frame.loc[mask, "market_regime"] = row.market_regime
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


def join_research_results(scanner: pd.DataFrame, source: StrategyResearchSource) -> pd.DataFrame:
    """Normalize native Task 4 and Strategy Lab outputs onto scanner keys.

    A key is entry-observed when Task 4 supplied its canonical outcome. Entry is
    true iff at least one already-authoritative trade has ``entry_time`` in the
    half-open post-decision evaluation horizon. No signal or PnL is recomputed.
    """
    if isinstance(source.outcomes, pd.DataFrame):
        outcomes = source.outcomes.copy()
    else:
        outcomes = pd.DataFrame(
            asdict(row) if is_dataclass(row) else dict(row)
            for row in source.outcomes
        )
    required = {"decision_time", "symbol", "forward_max_abs_excursion_pct"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Task 4 outcomes missing columns: {sorted(missing)}")
    outcomes = outcomes.rename(columns={"decision_time": "decision_timestamp",
                                        "forward_max_abs_excursion_pct": "absolute_movement"})
    outcomes["decision_timestamp"] = pd.to_datetime(outcomes.decision_timestamp, utc=True)
    outcomes["symbol"] = outcomes.symbol.astype(str).str.upper()
    if outcomes.duplicated(["decision_timestamp", "symbol"]).any():
        raise ValueError("duplicate natural research key in Task 4 outcomes")

    trades = source.trades.copy()
    trade_required = {"entry_time", "market_regime"}
    if trade_required - set(trades):
        raise ValueError(f"native trades missing columns: {sorted(trade_required-set(trades))}")
    expectancy_column = "pair_net_r" if "pair_net_r" in trades else "pair_net_pnl" if "pair_net_pnl" in trades else None
    if expectancy_column is None:
        raise ValueError("native trades require pair_net_r or pair_net_pnl")
    if "symbol" not in trades:
        if source.symbol is None:
            raise ValueError("native single-symbol trades require source symbol provenance")
        trades["symbol"] = source.symbol
    trades["symbol"] = trades.symbol.astype(str).str.upper()
    trades["entry_time"] = pd.to_datetime(trades.entry_time, utc=True, errors="coerce")
    trades["_expectancy"] = pd.to_numeric(trades[expectancy_column], errors="coerce")

    normalized=[]
    horizon = pd.Timedelta(source.evaluation_horizon)
    for outcome in outcomes.sort_values(["decision_timestamp", "symbol"], kind="stable").itertuples(index=False):
        end = outcome.decision_timestamp + horizon
        selected = trades[(trades.symbol == outcome.symbol) & (trades.entry_time >= outcome.decision_timestamp) & (trades.entry_time < end)]
        completed = selected[selected._expectancy.notna()]
        regimes = sorted(set(selected.market_regime.dropna().astype(str)))
        normalized.append({"decision_timestamp": outcome.decision_timestamp, "symbol": outcome.symbol,
            "valid_entry": bool(len(selected)), "absolute_movement": outcome.absolute_movement,
            "trade_expectancy_value": completed._expectancy.mean() if len(completed) else None,
            "trade_won": completed._expectancy.gt(0).mean() if len(completed) else None,
            "completed_trade_count": len(completed),
            "trade_expectancy_sum": completed._expectancy.sum() if len(completed) else 0.0,
            "winning_trade_count": int(completed._expectancy.gt(0).sum()),
            "research_market_regime": regimes[0] if len(regimes)==1 else getattr(outcome, "market_regime", None),
            "research_run_id": source.run_id,
            "research_source_identities": json.dumps(source.source_identities),
            "evaluation_horizon": source.evaluation_horizon})
    right = pd.DataFrame(normalized)
    left = scanner.copy()
    left["decision_timestamp"] = pd.to_datetime(left.decision_timestamp, utc=True)
    left["symbol"] = left.symbol.astype(str).str.upper()
    joined = left.merge(right, on=["decision_timestamp", "symbol"], how="left", validate="one_to_one")
    if "market_regime" in joined:
        joined["market_regime"] = joined.research_market_regime.combine_first(joined.market_regime)
    else:
        joined["market_regime"] = joined.research_market_regime
    joined = joined.drop(columns="research_market_regime")
    joined["year"] = joined.decision_timestamp.dt.year
    joined["valid_entry"] = joined.valid_entry.astype("boolean")
    joined["absolute_movement"] = pd.to_numeric(joined.absolute_movement, errors="coerce")
    joined["trade_expectancy_value"] = pd.to_numeric(joined.trade_expectancy_value, errors="coerce")
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
    if "completed_trade_count" in selected:
        trade_count = int(pd.to_numeric(selected.completed_trade_count, errors="coerce").fillna(0).sum())
        expectancy = (float(pd.to_numeric(selected.trade_expectancy_sum, errors="coerce").fillna(0).sum()) / trade_count
                      if trade_count else None)
        win_rate = (float(pd.to_numeric(selected.winning_trade_count, errors="coerce").fillna(0).sum()) / trade_count
                    if trade_count else None)
    else:
        trade_count = len(trades)
        expectancy = float(trades.trade_expectancy_value.mean()) if len(trades) else None
        win_rate = (float(trades.trade_won.astype("boolean").mean())
                    if len(trades) and trades.trade_won.notna().any() else None)
    return {
        "sample_count": int(len(selected)), "entry_observation_count": int(len(entries)),
        "movement_sample_count": int(len(observed)),
        "valid_entry_count": int((entries == True).sum()),  # noqa: E712
        "candidate_to_entry_conversion": float(entries.mean()) if len(entries) else None,
        "setup_frequency": float(entries.mean()) if len(entries) else None,
        "mean_absolute_movement": float(observed.absolute_movement.mean()) if len(observed) else None,
        "median_absolute_movement": float(observed.absolute_movement.median()) if len(observed) else None,
        "opportunity_capture": capture, "trade_count": trade_count,
        "expectancy": expectancy, "win_rate": win_rate,
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
    component_cols=sorted(c for c in frame if c.startswith(("component__", "discovery__")))
    component_rows=[]
    for i,left in enumerate(component_cols):
        for right in component_cols[i+1:]:
            n,rho=_spearman(frame[left],frame[right])
            component_rows.append({"analysis_type":"REDUNDANCY", "component_a":left,"component_b":right,"outcome":None,"sample_count":n,"spearman_rho":rho})
    for component in component_cols:
        for outcome in ("absolute_movement", "valid_entry", "trade_expectancy_value"):
            n,rho=_spearman(frame[component],frame[outcome])
            component_rows.append({"analysis_type":"OUTCOME_ASSOCIATION", "component_a":component,"component_b":None,"outcome":outcome,"sample_count":n,"spearman_rho":rho})
    return OpportunityValidationResult(config, frame, overall, by_rank, top_k, decay, by_year, by_regime,
                                       pd.DataFrame(component_rows), pd.DataFrame(associations))
