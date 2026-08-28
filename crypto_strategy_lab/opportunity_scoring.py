"""Task 4: causal, direction-neutral opportunity ranking and research evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import csv, json, math
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

import pandas as pd

from crypto_strategy_lab.data.source_identity import SourceSignature


class ScoreStatus(str, Enum):
    SCORABLE = "SCORABLE"
    UNSCORABLE = "UNSCORABLE"


@dataclass(frozen=True, slots=True)
class OpportunityComponent:
    name: str
    weight: float


@dataclass(frozen=True, slots=True)
class OpportunityScoringModelDefinition:
    name: str
    version: str
    components: tuple[OpportunityComponent, ...]
    ordinal_percentiles: bool = False
    supported_intervals: tuple[str, ...] | None = None
    def __post_init__(self):
        if not math.isclose(sum(x.weight for x in self.components), 1.0):
            raise ValueError("component weights must total 1.0")


LEGACY_VOLATILITY_V1 = OpportunityScoringModelDefinition("legacy_volatility", "1", tuple(OpportunityComponent(*x) for x in (
    ("realized_volatility", .30), ("atr_pct", .25), ("recent_range_pct", .20), ("range_expansion", .10), ("volume_ratio", .15))), True, ("1h",))
BALANCED_ACTIVITY_V1 = OpportunityScoringModelDefinition("balanced_activity", "1", tuple(OpportunityComponent(*x) for x in (
    ("realized_volatility", .20), ("atr_pct", .15), ("recent_range_pct", .15), ("range_expansion", .10), ("volume_ratio", .10), ("adx", .15), ("di_spread", .15))))
MOMENTUM_ACTIVITY_V1 = OpportunityScoringModelDefinition("momentum_activity", "1", tuple(OpportunityComponent(*x) for x in (
    ("realized_volatility", .20), ("atr_pct", .15), ("recent_range_pct", .15), ("range_expansion", .10), ("volume_ratio", .15), ("adx", .10), ("abs_momentum_24h", .15))))
SCORING_MODELS = (LEGACY_VOLATILITY_V1, BALANCED_ACTIVITY_V1, MOMENTUM_ACTIVITY_V1)


@dataclass(frozen=True, slots=True)
class OpportunityScoringConfig:
    strategy_interval: str = "1h"
    models: tuple[OpportunityScoringModelDefinition, ...] = SCORING_MODELS

    def __post_init__(self):
        if not self.strategy_interval.strip():
            raise ValueError("strategy_interval must not be empty")


@dataclass(frozen=True, slots=True)
class OpportunityFeatureSnapshot:
    symbol: str; discovery_rank: int; decision_time: datetime; strategy_interval: str
    feature_timestamp: datetime; available_at: datetime; values: Mapping[str, object]
    source_signature: SourceSignature
    feature_versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OpportunityScoreRow:
    symbol: str; discovery_rank: int; decision_time: datetime; strategy_interval: str
    feature_timestamp: datetime; available_at: datetime; raw_components: Mapping[str, float | None]
    normalized_components: Mapping[str, float]; component_weights: Mapping[str, float]
    model_name: str; model_version: str; score: float | None; model_rank: int | None
    atr: float | None; atr_pct: float | None; adx: float | None; di_spread: float | None
    previous_di_spread: float | None; di_pressure_delta: float | None; di_pressure_state: str
    signed_momentum_24h: float | None; absolute_momentum_24h: float | None; market_regime: str | None
    source_identity: str; feature_versions: Mapping[str, str]; status: ScoreStatus; unscorable_reason: str | None


@dataclass(frozen=True, slots=True)
class OpportunityScoringResult:
    decision_time: datetime
    rows: tuple[OpportunityScoreRow, ...]


def latest_causal_snapshot(symbol: str, discovery_rank: int, decision_time: datetime, strategy_interval: str,
                           feature_frame: pd.DataFrame, source_signature: SourceSignature,
                           feature_versions: Mapping[str, str] | None = None) -> OpportunityFeatureSnapshot | None:
    """Select exactly the latest observation available at the explicit boundary."""
    boundary = pd.Timestamp(decision_time)
    boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    eligible = feature_frame[pd.to_datetime(feature_frame.available_at, utc=True) <= boundary].copy()
    if eligible.empty: return None
    eligible["_available"] = pd.to_datetime(eligible.available_at, utc=True)
    eligible["_timestamp"] = pd.to_datetime(eligible.timestamp, utc=True)
    row = eligible.sort_values(["_available", "_timestamp"], kind="stable").iloc[-1]
    ignored = {"timestamp", "available_at", "_available", "_timestamp"}
    return OpportunityFeatureSnapshot(symbol, discovery_rank, boundary.to_pydatetime(), strategy_interval,
        row._timestamp.to_pydatetime(), row._available.to_pydatetime(), {k: row[k] for k in eligible.columns if k not in ignored},
        source_signature, dict(feature_versions or {}))


def snapshot_from_registry_features(symbol: str, discovery_rank: int, decision_time: datetime,
                                    strategy_interval: str, feature_frames: Mapping[str, pd.DataFrame],
                                    source_signature: SourceSignature) -> OpportunityFeatureSnapshot | None:
    """Join authoritative provider outputs at their individual causal boundaries.

    Expected inputs are registry results for ``core_directional``,
    ``policy_market_context`` and ``opportunity_activity``.  This function does
    not calculate any technical indicator and therefore keeps provider
    ownership testable and explicit.
    """
    required=("core_directional","policy_market_context","opportunity_activity")
    if any(name not in feature_frames for name in required): return None
    boundary=pd.Timestamp(decision_time); boundary=boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    values={}; versions={}; chosen=[]
    for name in required:
        frame=feature_frames[name]
        eligible=frame[pd.to_datetime(frame.available_at,utc=True)<=boundary].copy()
        if eligible.empty:return None
        eligible["_a"]=pd.to_datetime(eligible.available_at,utc=True); eligible["_t"]=pd.to_datetime(eligible.timestamp,utc=True)
        row=eligible.sort_values(["_a","_t"],kind="stable").iloc[-1]; chosen.append(row)
        values.update({column:row[column] for column in frame.columns if column not in {"timestamp","available_at"}})
        versions[name]=str(frame.attrs.get("feature_version","unknown"))
    available=max(row._a for row in chosen); timestamp=max(row._t for row in chosen)
    return OpportunityFeatureSnapshot(symbol,discovery_rank,boundary.to_pydatetime(),strategy_interval,
        timestamp.to_pydatetime(),available.to_pydatetime(),values,source_signature,versions)


def _percentiles(items, ordinal):
    n=len(items)
    if n == 1: return {items[0][0]: 1.0}
    ordered=sorted(enumerate(items), key=lambda x: (x[1][1], x[0]))
    if ordinal: return {item[0]: rank/(n-1) for rank, (_, item) in enumerate(ordered)}
    groups={}
    for rank, (_, item) in enumerate(ordered): groups.setdefault(item[1], []).append(rank)
    return {key: mean(groups[value])/(n-1) for key, value in items}


def score_opportunities(snapshots: Sequence[OpportunityFeatureSnapshot], decision_time: datetime,
                        config: OpportunityScoringConfig = OpportunityScoringConfig()) -> OpportunityScoringResult:
    boundary=pd.Timestamp(decision_time); boundary=boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    current=[s for s in snapshots if pd.Timestamp(s.decision_time)==boundary and pd.Timestamp(s.available_at)<=boundary]
    current=sorted(current, key=lambda s:(s.discovery_rank,s.symbol))
    output=[]
    for model in config.models:
        valid=[]; missing_by_symbol={}
        for s in current:
            values=dict(s.values); momentum=values.get("momentum_return_24h")
            values["abs_momentum_24h"] = abs(float(momentum)) if _finite(momentum) else None
            missing=[c.name for c in model.components if not _finite(values.get(c.name))]
            if s.strategy_interval != config.strategy_interval:
                missing.insert(0, f"strategy_interval={s.strategy_interval!r} does not match config {config.strategy_interval!r}")
            elif model.supported_intervals and s.strategy_interval not in model.supported_intervals:
                missing.insert(0, f"model supports only {', '.join(model.supported_intervals)} strategy interval")
            if missing: missing_by_symbol[s.symbol]=missing
            else: valid.append((s,values))
        normalized={s.symbol:{} for s,_ in valid}
        for component in model.components:
            ranks=_percentiles([(s.symbol,float(v[component.name])) for s,v in valid], model.ordinal_percentiles)
            for symbol,value in ranks.items(): normalized[symbol][component.name]=value
        scored=[]
        for s,values in valid:
            score=sum(normalized[s.symbol][c.name]*c.weight for c in model.components); scored.append((s,values,score))
        scored.sort(key=lambda x:(-x[2],x[0].discovery_rank,x[0].symbol)); ranks={s.symbol:i+1 for i,(s,_,_) in enumerate(scored)}
        for s in current:
            match=next(((v,score) for ss,v,score in scored if ss.symbol==s.symbol),None)
            output.append(_row(s,model,match,normalized.get(s.symbol,{}),ranks.get(s.symbol),missing_by_symbol.get(s.symbol)))
    return OpportunityScoringResult(boundary.to_pydatetime(),tuple(output))


def _finite(value):
    try: return value is not None and math.isfinite(float(value))
    except (TypeError,ValueError): return False


def _row(s,m,match,norm,rank,missing):
    v=dict(s.values); momentum=v.get("momentum_return_24h"); spread=v.get("di_spread"); previous=v.get("di_spread_1")
    delta=float(spread)-float(previous) if _finite(spread) and _finite(previous) else None
    state="FLAT" if delta is not None and math.isclose(delta,0,abs_tol=1e-12) else "EXPANDING" if delta is not None and delta>0 else "CONTRACTING" if delta is not None else "UNKNOWN"
    # Preserve everything observed for research diagnostics. Missing components
    # are explicit None and are never converted to zero.
    raw={c.name:_num(v.get(c.name)) for c in m.components}
    if "abs_momentum_24h" in raw:
        raw["abs_momentum_24h"] = abs(float(momentum)) if _finite(momentum) else None
    return OpportunityScoreRow(s.symbol,s.discovery_rank,s.decision_time,s.strategy_interval,s.feature_timestamp,s.available_at,raw,norm,
        {c.name:c.weight for c in m.components},m.name,m.version,None if match is None else match[1],rank,
        _num(v.get("atr")),_num(v.get("atr_pct")),_num(v.get("adx")),_num(spread),_num(previous),delta,state,_num(momentum),abs(float(momentum)) if _finite(momentum) else None,
        None if v.get("market_regime") is None else str(v["market_regime"]),s.source_signature.cache_identity(),s.feature_versions,
        ScoreStatus.UNSCORABLE if missing else ScoreStatus.SCORABLE, "missing components: "+", ".join(missing) if missing else None)


def _num(v): return float(v) if _finite(v) else None


@dataclass(frozen=True, slots=True)
class OpportunityOutcome:
    symbol: str; decision_time: datetime; decision_close: float; forward_range_pct: float
    forward_max_abs_excursion_pct: float; forward_abs_close_return_pct: float; forward_max_abs_excursion_atr: float


class HistoricalOpportunityEvaluator:
    def __init__(self, horizon="24h"): self.horizon=pd.Timedelta(horizon)
    def evaluate(self,row: OpportunityScoreRow, candles: pd.DataFrame, decision_close: float) -> OpportunityOutcome | None:
        start=pd.Timestamp(row.decision_time); start=start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        interval=pd.Timedelta(row.strategy_interval)
        if interval <= pd.Timedelta(0) or self.horizon % interval:
            return None
        times=pd.to_datetime(candles.period_start,utc=True)
        # Candle-open convention: include the candle opening at the decision and
        # exclude the one opening at the horizon boundary.
        future=candles[(times>=start)&(times<start+self.horizon)].copy()
        future["_period_start"]=pd.to_datetime(future.period_start,utc=True)
        future=future.sort_values("_period_start",kind="stable").drop_duplicates("_period_start",keep="last")
        expected=pd.date_range(start,start+self.horizon,freq=interval,inclusive="left")
        if len(future)!=len(expected) or not pd.DatetimeIndex(future._period_start).equals(expected):
            return None
        if not row.atr or row.atr<=0:return None
        hi=float(pd.to_numeric(future.high).max()); lo=float(pd.to_numeric(future.low).min()); final=float(pd.to_numeric(future.close).iloc[-1])
        excursion=max(hi/decision_close-1,1-lo/decision_close)
        return OpportunityOutcome(row.symbol,row.decision_time,decision_close,(hi-lo)/decision_close,excursion,abs(final/decision_close-1),excursion*decision_close/row.atr)


def build_comparison(scores: Sequence[OpportunityScoreRow], outcomes: Mapping[tuple[str, datetime], OpportunityOutcome],
                     top_ks: Sequence[int] = (5, 10, 20)) -> tuple[dict, ...]:
    """Aggregate model/control performance, including year and regime strata.

    ``discovery_order`` and ``no_second_stage`` are derived from the same frozen
    rows, avoiding a second scoring path. The latter always uses the complete
    acquired shortlist and consequently emits one ``ALL`` K row.
    """
    unique={}
    for row in scores:
        unique.setdefault((row.symbol,row.decision_time),row)
    controls=[]
    for row in unique.values():
        controls.append(("discovery_order",row.discovery_rank,row,True))
        controls.append(("no_second_stage",row.discovery_rank,row,True))
    # Retain unscorable rows in the population. Their missing rank excludes them
    # from selection but not from rejection/coverage denominators.
    modeled=[(f"{r.model_name}_v{r.model_version}",r.model_rank,r,r.status==ScoreStatus.SCORABLE) for r in scores]
    result=[]
    for model in ("discovery_order","legacy_volatility_v1","balanced_activity_v1","momentum_activity_v1","no_second_stage"):
        pool=[x for x in controls+modeled if x[0]==model]
        decisions=sorted({x[2].decision_time for x in pool})
        ks=(None,) if model=="no_second_stage" else tuple(top_ks)
        for k in ks:
            selected=[]
            for decision in decisions:
                day=sorted((x for x in pool if x[2].decision_time==decision and x[3]),key=lambda x:(x[1],x[2].discovery_rank,x[2].symbol))
                selected.extend(day if k is None else day[:k])
            evaluated=[(row,outcomes.get((row.symbol,row.decision_time))) for _,_,row,_ in selected]
            evaluated=[x for x in evaluated if x[1] is not None]
            result.extend(_aggregate(model,"ALL","ALL",k,pool,selected,evaluated))
            years=sorted({x[2].decision_time.year for x in pool})
            regimes=sorted({x[2].market_regime for x in pool if x[2].market_regime})
            for year in years:
                result.extend(_aggregate(model,"YEAR",str(year),k,
                    [x for x in pool if x[2].decision_time.year==year],
                    [x for x in selected if x[2].decision_time.year==year],
                    [(r,o) for r,o in evaluated if r.decision_time.year==year]))
            for regime in regimes:
                result.extend(_aggregate(model,"REGIME",regime,k,
                    [x for x in pool if x[2].market_regime==regime],
                    [x for x in selected if x[2].market_regime==regime],
                    [(r,o) for r,o in evaluated if r.market_regime==regime]))
    baseline={(r["stratum_type"],r["stratum"],r["top_k"]):r for r in result if r["model"]=="discovery_order"}
    for row in result:
        base=baseline.get((row["stratum_type"],row["stratum"],row["top_k"]))
        current=row["mean_forward_range_pct"]
        baseline_value=None if not base else base["mean_forward_range_pct"]
        row["lift_vs_discovery_order_forward_range_pct"] = (
            None if current is None or baseline_value is None or baseline_value == 0
            else current/baseline_value-1
        )
    return tuple(result)


def _aggregate(model,stratum_type,stratum,k,pool,selected,evaluated):
    outs=[o for _,o in evaluated]
    vals=lambda name:[getattr(o,name) for o in outs]
    metric=lambda name,fn: fn(vals(name)) if outs else None
    population=len(pool)
    scorable=sum(x[3] for x in pool)
    decisions={x[2].decision_time for x in pool}
    return [{"model":model,"top_k":"ALL" if k is None else k,"stratum_type":stratum_type,"stratum":stratum,
        "decision_count":len(decisions),"candidate_count":population,"scorable_count":scorable,
        "selected_count":len(selected),"outcome_count":len(evaluated),
        "mean_forward_range_pct":metric("forward_range_pct",mean),"median_forward_range_pct":metric("forward_range_pct",median),
        "mean_forward_max_abs_excursion_pct":metric("forward_max_abs_excursion_pct",mean),"median_forward_max_abs_excursion_pct":metric("forward_max_abs_excursion_pct",median),
        "mean_forward_excursion_atr":metric("forward_max_abs_excursion_atr",mean),"median_forward_excursion_atr":metric("forward_max_abs_excursion_atr",median),
        "pct_reaching_1_atr":mean([o.forward_max_abs_excursion_atr>=1 for o in outs]) if outs else None,
        "pct_reaching_2_atr":mean([o.forward_max_abs_excursion_atr>=2 for o in outs]) if outs else None,
        "coverage":len(evaluated)/len(selected) if selected else 0.0,
        "unscorable_rate":1-scorable/population if population else 1.0}]


def write_opportunity_reports(directory: Path, scores: Sequence[OpportunityScoreRow], comparison: Sequence[Mapping], metadata: Mapping) -> None:
    """Write deterministic research artifacts; this is not a run publication system."""
    directory.mkdir(parents=True,exist_ok=True)
    def flat(row):
        d=asdict(row); d["status"]=row.status.value
        for key in ("raw_components","normalized_components","component_weights","feature_versions"): d[key]=json.dumps(d[key],sort_keys=True)
        return d
    score_rows=[flat(r) for r in sorted(scores,key=lambda r:(r.decision_time,r.model_name,r.model_rank or 10**9,r.symbol))]
    _csv(directory/"opportunity_score_rows.csv",score_rows)
    _csv(directory/"opportunity_model_comparison.csv",list(comparison))
    (directory/"opportunity_comparison.json").write_text(json.dumps({"metadata":metadata,"comparison":list(comparison)},sort_keys=True,indent=2,default=str)+"\n")


def _csv(path,rows):
    fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
