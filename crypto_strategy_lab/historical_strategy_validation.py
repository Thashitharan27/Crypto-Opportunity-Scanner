"""Automated validation of frozen historical scanner candidates.

This is an orchestration/reporting layer.  It deliberately delegates every
strategy decision and trade outcome to ResearchRunner and the existing research
sampling implementation; it only attaches their authoritative rows to candidate
windows and aggregates them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable, Mapping
from uuid import uuid4

import pandas as pd

from .data_lake_config import ResearchRunConfig, load_data_lake_config
from .research_adapters import native_simulator_config
from .research_sampling import generate_strategy_research_sampling_result
from .prepared_backtest import intrabar_from_data_lake_bundle
from .opportunity_validation import load_scanner_observations
from .run_manifest import file_sha256
from .data import DataRequest, DatasetKind, MarketDataStore, MarketKind
from .features import production_feature_registry
from .prepared_cache import PreparedRunCache
from .research_adapters import NativeSimulator, NativeStrategyPolicy
from .research_runner import ResearchRunner
from .bayesian_sampling_reporting import BayesianSamplingCsvManifestReporter
from .data.timing import (floor_fixed_candle_grid, normalize_binance_interval,
    normalize_native_fixed_candle_interval)
from .research_warmup import WARMUP_POLICY_VERSION, validation_warmup_bars
from .validation_data_preflight import ValidationDataPreparer

STANDARD_SINGLE_SYMBOL = "STANDARD_SINGLE_SYMBOL"
EVERY_VIABLE_ENTRY = "EVERY_VIABLE_ENTRY"
POPULATIONS = (STANDARD_SINGLE_SYMBOL, EVERY_VIABLE_ENTRY)


class ValidationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolResearchResult:
    """Rows emitted by one native symbol research invocation."""
    research_run_id: str
    standard_trades: pd.DataFrame
    viable_trades: pd.DataFrame
    viable_censored: pd.DataFrame
    available_through: datetime
    run_dir: Path
    request_start: datetime
    request_end: datetime
    source_identities: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyValidationResult:
    run_dir: Path | None
    outcomes: pd.DataFrame
    summary: pd.DataFrame
    by_rank: pd.DataFrame
    top_k: pd.DataFrame
    by_year: pd.DataFrame
    associations: pd.DataFrame
    data_readiness: pd.DataFrame
    manifest: Mapping[str, object]


class NativeResearchExecutor:
    """Adapter that obtains both populations from one ``ResearchRunner.run``.

    ``request_factory`` owns DataRequest construction (and therefore the normal
    Data Lake acquisition/quality contract). A temporary capture reporter sees
    the exact prepared/intrabar frames already built by ResearchRunner and calls
    the existing sampling engine; neither frame nor any strategy rule is rebuilt.
    """
    def __init__(self, runner_factory, request_factory):
        self.runner_factory, self.request_factory = runner_factory, request_factory

    def __call__(self, symbol, start, end, config):
        captured = {}
        runner = self.runner_factory()
        class CaptureReporter:
            def report(_self, result, context):
                native = native_simulator_config(
                    context.config.data, context.config.features,
                    context.config.strategy, context.config.execution,
                )
                captured["sampling"] = generate_strategy_research_sampling_result(
                    context.prepared, intrabar_from_data_lake_bundle(context.bundle),
                    native, mode=EVERY_VIABLE_ENTRY, interval_candles=1,
                )
                timestamps = pd.to_datetime(context.bundle.strategy["period_start"], utc=True)
                captured["available_through"] = timestamps.max() + pd.Timedelta(minutes=context.config.data.strategy_timeframe_minutes)
        original = runner.reporters
        runner.reporters = (*original, CaptureReporter())
        try:
            result = runner.run(self.request_factory(symbol,start,end,config),config)
        finally:
            runner.reporters = original
        run_dir=Path(result.output_dir)
        manifest=json.loads((run_dir/"run_manifest.json").read_text(encoding="utf-8"))
        sampling=captured["sampling"]; viable=sampling.resolved; censored=sampling.censored
        standard=result.trades
        if standard.empty:
            standard=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"])
        if viable.empty:
            viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        if censored.empty:
            censored=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        sources=tuple(x.get("source_signature","") for x in manifest.get("catalog",{}).get("datasets",[]) if x.get("source_signature"))
        request=manifest["request"]
        return SymbolResearchResult(manifest["run_id"],standard,viable,censored,
            captured["available_through"].to_pydatetime(),run_dir,
            pd.Timestamp(request["start"]).to_pydatetime(),pd.Timestamp(request["end"]).to_pydatetime(),sources)


def extract_final_candidates(run_directories) -> pd.DataFrame:
    """Read immutable Task-7 facts and retain final candidates without reranking."""
    frame = load_scanner_observations(run_directories)
    if frame.empty:
        return frame
    result = frame.loc[frame["final"].eq(True), [
        "decision_timestamp", "final_rank", "symbol", "scan_run_id",
        "scanner_source_identity",
    ]].copy()
    result["final_rank"] = pd.to_numeric(result.final_rank, errors="raise").astype(int)
    if result.duplicated(["decision_timestamp", "symbol"]).any():
        raise ValueError("duplicate candidate identity (decision_timestamp, symbol)")
    return result.sort_values(["decision_timestamp", "final_rank", "symbol"], kind="stable").reset_index(drop=True)


def load_validation_config(path: str | Path) -> ResearchRunConfig:
    """Load the single authoritative, strict v3 configuration."""
    return load_data_lake_config(path)


def build_validation_data_request(symbol,start,end,config,registry=None):
    """Production DataRequest projection; strategy/execution are untouched."""
    strategy=normalize_native_fixed_candle_interval(f"{config.data.strategy_timeframe_minutes}m")
    intrabar=(normalize_native_fixed_candle_interval(f"{config.data.intrabar_timeframe_minutes}m")
        if config.data.use_intrabar_data else None)
    start=floor_fixed_candle_grid(start,strategy)
    return DataRequest(symbol,start,end,strategy,intrabar,
        market=MarketKind.FUTURES_UM)


VALIDATION_REPORTING_OVERRIDES={
    "research_sampling_mode":"PORTFOLIO", "analysis_level":"STANDARD",
    "enable_trade_telemetry":False, "save_full_telemetry_csv":False,
    "save_trade_journey_summary":False, "save_trade_journey_charts":False,
    "enable_indicator_lifecycle_analysis":False, "create_lifecycle_charts":False,
    "save_feature_analysis_reports":False, "save_indicator_analysis_reports":False,
    "create_standard_charts":False,
}


def validation_execution_config(config: ResearchRunConfig) -> ResearchRunConfig:
    """Clone reporting only; strategy and execution objects remain authoritative."""
    return replace(config,reporting=replace(config.reporting,**VALIDATION_REPORTING_OVERRIDES))


def latest_strategy_coverage(rows,symbol,strategy_timeframe_minutes):
    """Resolve normalized native-kline coverage from catalog-like inventory rows."""
    interval=normalize_binance_interval(f"{int(strategy_timeframe_minutes)}m")
    selected=[]
    for row in rows:
        if (row.get("symbol")!=symbol or
            str(getattr(row.get("dataset"),"value",row.get("dataset")))!=DatasetKind.KLINES.value or
            row.get("last_period") is None): continue
        try: catalog_interval=normalize_binance_interval(str(row.get("interval")))
        except ValueError: continue
        if catalog_interval==interval: selected.append(row)
    if not selected: return None
    latest=max(pd.Timestamp(r["last_period"]) for r in selected)
    latest=latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")
    return latest.to_pydatetime()


def _trade_id(frame: pd.DataFrame, population: str, run_id: str) -> pd.Series:
    native = "research_sample_id" if population == EVERY_VIABLE_ENTRY else "pair_id"
    if native not in frame:
        raise ValueError(f"{population} native rows require authoritative {native}")
    return run_id + "|" + frame["symbol"].astype(str) + "|" + frame[native].astype(str)


def attach_candidate_trades(candidates: pd.DataFrame, trades: pd.DataFrame, *,
                            population: str, run_id: str, horizon: pd.Timedelta,
                            available_through, censored_trades: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach native entries using Task-8's half-open ``[decision, decision+horizon)`` rule."""
    rows, associations = [], []
    native = trades.copy()
    required = {"entry_time", "pair_net_r", "side"}
    if missing := required - set(native):
        raise ValueError(f"native trades missing columns: {sorted(missing)}")
    if "symbol" not in native:
        native["symbol"] = candidates.symbol.iloc[0]
    if not native.empty:
        native["symbol"] = native["symbol"].astype(str).str.upper()
        native["entry_time"] = pd.to_datetime(native.entry_time, utc=True, errors="raise")
        native["pair_net_r"] = pd.to_numeric(native.pair_net_r, errors="coerce")
        native["trade_identity"] = _trade_id(native, population, run_id)
    else:
        native["entry_time"] = pd.to_datetime(native["entry_time"], utc=True)
    censored_native = pd.DataFrame() if censored_trades is None else censored_trades.copy()
    if not censored_native.empty:
        if "symbol" not in censored_native: censored_native["symbol"] = candidates.symbol.iloc[0]
        censored_native["symbol"] = censored_native.symbol.astype(str).str.upper()
        censored_native["entry_time"] = pd.to_datetime(censored_native.entry_time, utc=True)
        censored_native["trade_identity"] = _trade_id(censored_native, population, run_id)
    coverage = pd.Timestamp(available_through)
    coverage = coverage.tz_localize("UTC") if coverage.tzinfo is None else coverage.tz_convert("UTC")
    for candidate in candidates.itertuples(index=False):
        decision = pd.Timestamp(candidate.decision_timestamp)
        end = decision + horizon
        complete = coverage >= end
        selected = native[(native.symbol == candidate.symbol) & (native.entry_time >= decision) & (native.entry_time < end)]
        external_censored = (censored_native[(censored_native.symbol == candidate.symbol) &
            (censored_native.entry_time >= decision) & (censored_native.entry_time < end)]
            if not censored_native.empty else censored_native)
        reason_columns = [c for c in ("exit_reason", "long_exit_reason", "short_exit_reason") if c in selected]
        censored = pd.Series(False, index=selected.index)
        for column in reason_columns:
            censored |= selected[column].astype(str).str.upper().eq("END_OF_DATA")
        internal_censored=selected.loc[censored].copy()
        selected_censored=pd.concat([internal_censored,external_censored],ignore_index=True)
        resolved = selected.loc[~censored & selected.pair_net_r.notna()]
        wins, losses = int(resolved.pair_net_r.gt(0).sum()), int(resolved.pair_net_r.lt(0).sum())
        neutrals = int(resolved.pair_net_r.eq(0).sum())
        unresolved = not complete or len(selected_censored) > 0 or (len(selected) > 0 and resolved.empty)
        if unresolved: result = "UNRESOLVED"
        elif selected.empty: result = "NO_ENTRY"
        elif len(resolved) > 1: result = "MULTIPLE"
        elif wins: result = "WIN"
        elif losses: result = "LOSS"
        else: result = "NEUTRAL"
        all_entries=pd.concat([selected,external_censored],ignore_index=True)
        sides = sorted(set(all_entries.side.dropna().astype(str).str.upper()))
        side = sides[0] if len(sides) == 1 else "MIXED" if sides else "—"
        base = {"decision_timestamp": decision, "final_rank": int(candidate.final_rank),
                "symbol": candidate.symbol, "scan_run_id": candidate.scan_run_id,
                "scanner_source_identity": getattr(candidate,"scanner_source_identity",None),
                "population": population, "research_run_id": run_id,
                "evaluation_horizon": str(horizon),
                "entry_horizon_status": "COMPLETE" if complete else "INSUFFICIENT_FUTURE_DATA",
                "future_data_status": "COMPLETE" if complete else "INSUFFICIENT_FUTURE_DATA",
                "outcome_resolution_status":"UNRESOLVED" if unresolved else "RESOLVED",
                "valid_entry": bool(len(all_entries)), "entry_count": len(all_entries),
                "completed_trade_count": len(resolved), "wins": wins, "losses": losses,
                "neutrals": neutrals, "win_rate_resolved": wins/(wins+losses) if wins+losses else None,
                "win_share_all_completed": wins/len(resolved) if len(resolved) else None,
                "average_r": resolved.pair_net_r.mean() if len(resolved) else None,
                "net_r": resolved.pair_net_r.sum() if len(resolved) else None,
                "first_entry_time": all_entries.entry_time.min() if len(all_entries) else None,
                "last_entry_time": all_entries.entry_time.max() if len(all_entries) else None,
                "side": side, "result": result}
        rows.append(base)
        for trade in resolved.itertuples(index=False):
            associations.append({**{k: base[k] for k in ("decision_timestamp", "final_rank", "symbol", "population","scan_run_id","scanner_source_identity")},
                                 "trade_identity": trade.trade_identity, "pair_net_r": trade.pair_net_r,
                                 "entry_time": trade.entry_time, "side": str(trade.side).upper(),
                                 "research_run_id":run_id,"status":"RESOLVED"})
        for trade in selected_censored.itertuples(index=False):
            associations.append({**{k: base[k] for k in ("decision_timestamp", "final_rank", "symbol", "population","scan_run_id","scanner_source_identity")},
                "research_run_id":run_id,"trade_identity":trade.trade_identity,"pair_net_r":None,
                "entry_time":trade.entry_time,"side":str(trade.side).upper(),"status":"CENSORED"})
    return pd.DataFrame(rows), pd.DataFrame(associations)


def _attach_symbol_result(candidates: pd.DataFrame, result: SymbolResearchResult,
                          horizon: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project one native symbol run onto both validation populations."""
    rows=[]; associations=[]
    for population,trades,censored in (
        (STANDARD_SINGLE_SYMBOL,result.standard_trades,None),
        (EVERY_VIABLE_ENTRY,result.viable_trades,result.viable_censored),
    ):
        window_rows,assoc=attach_candidate_trades(candidates,trades,population=population,
            run_id=result.research_run_id,horizon=horizon,available_through=result.available_through,
            censored_trades=censored)
        rows.append(window_rows); associations.append(assoc)
    combined_rows=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    combined_assoc=(pd.concat([x for x in associations if not x.empty],ignore_index=True)
        if any(not x.empty for x in associations) else pd.DataFrame())
    return combined_rows,combined_assoc


def _aggregate(windows: pd.DataFrame, associations: pd.DataFrame) -> dict:
    observed = windows[windows.entry_horizon_status.eq("COMPLETE")]
    resolved_associations = associations[associations.get("status", "RESOLVED").eq("RESOLVED")] if not associations.empty else associations
    unique = resolved_associations.drop_duplicates("trade_identity") if not resolved_associations.empty else resolved_associations
    wins = int(unique.pair_net_r.gt(0).sum()) if len(unique) else 0
    losses = int(unique.pair_net_r.lt(0).sum()) if len(unique) else 0
    neutrals = int(unique.pair_net_r.eq(0).sum()) if len(unique) else 0
    return {"candidate_observations": len(windows), "observed_candidate_windows": len(observed),
            "unresolved_candidate_windows": int(windows.outcome_resolution_status.eq("UNRESOLVED").sum()),
            "candidate_windows_with_entry": int(observed.valid_entry.sum()),
            "candidate_to_entry_conversion": observed.valid_entry.mean() if len(observed) else None,
            "unique_trade_count": len(unique), "unique_wins": wins, "unique_losses": losses,
            "unique_neutrals": neutrals,
            "resolved_unique_trade_win_rate": wins/(wins+losses) if wins+losses else None,
            "win_share_all_completed": wins/(wins+losses+neutrals) if wins+losses+neutrals else None,
            "average_r_per_unique_trade": unique.pair_net_r.mean() if len(unique) else None,
            "net_r_unique": unique.pair_net_r.sum() if len(unique) else None}


def aggregate_reports(outcomes: pd.DataFrame, associations: pd.DataFrame):
    def table(groups, key_names):
        result=[]
        for keys, windows in groups:
            keys=(keys,) if not isinstance(keys,tuple) else keys
            assoc = associations
            for name,value in zip(key_names,keys): assoc=assoc[assoc[name].eq(value)] if name in assoc else assoc.iloc[0:0]
            result.append({**dict(zip(key_names,keys)), **_aggregate(windows,assoc)})
        return pd.DataFrame(result)
    summary=table(outcomes.groupby("population",sort=False),["population"])
    by_rank=table(outcomes.groupby(["population","final_rank"],sort=True),["population","final_rank"])
    top=[]
    for population, frame in outcomes.groupby("population",sort=False):
        for k in (1,3,5,10):
            selected=frame[frame.final_rank.le(k)]
            if selected.empty: continue
            assoc=associations[(associations.population==population)&(associations.final_rank.le(k))]
            top.append({"population":population,"top_k":k,**_aggregate(selected,assoc)})
    yearly=outcomes.assign(year=outcomes.decision_timestamp.dt.year)
    associations=associations.assign(year=associations.decision_timestamp.dt.year) if not associations.empty else associations
    by_year=table(yearly.groupby(["population","year"],sort=True),["population","year"])
    return summary,by_rank,pd.DataFrame(top),by_year


class HistoricalStrategyValidator:
    """Sequential native validator with an optional causal two-stage fast path."""
    def __init__(self, executor: Callable, *, warmup_bars: Callable[[ResearchRunConfig], int],
                 output_root: Path = Path("output/opportunity_validation"), monotonic=time.monotonic,
                 latest_available: Callable[[str], datetime | None] = lambda _symbol:None,
                 preflight: Callable | None = None, common_available_end: Callable | None = None,
                 defer_outcome_tail_until_needed: bool = False,
                 enforce_stable_code_provenance: bool = True):
        self.executor, self.warmup_bars = executor, warmup_bars
        self.output_root, self.monotonic, self.latest_available = Path(output_root), monotonic, latest_available
        self.preflight=preflight
        self.common_available_end=common_available_end
        self.defer_outcome_tail_until_needed=bool(defer_outcome_tail_until_needed)
        self.enforce_stable_code_provenance=bool(enforce_stable_code_provenance)

    def run(self, candidates: pd.DataFrame, config: ResearchRunConfig, *, config_path: Path,
            evaluation_horizon="24h", cancelled=lambda:False, progress=lambda event:None,
            publish=True) -> StrategyValidationResult:
        config.validate(); horizon=pd.Timedelta(evaluation_horizon)
        if horizon <= pd.Timedelta(0): raise ValueError("entry evaluation horizon must be positive")
        started_timestamp=datetime.now(timezone.utc); validation_started=self.monotonic()
        code_commit_start=_commit()
        def assert_code_stable():
            current=_commit()
            if (self.enforce_stable_code_provenance and code_commit_start!="UNKNOWN" and
                current!="UNKNOWN" and current!=code_commit_start):
                raise RuntimeError("validation code provenance changed during the run; rerun from one pinned commit")
            return current
        canonical_config=json.dumps(config.to_dict(),sort_keys=True,separators=(",",":"),default=str).encode()
        config_hash=sha256(canonical_config).hexdigest()
        execution_config=validation_execution_config(config)
        required={"decision_timestamp","final_rank","symbol","scan_run_id"}
        if missing:=required-set(candidates): raise ValueError(f"candidates missing columns: {sorted(missing)}")
        candidates=candidates.copy(); candidates["decision_timestamp"]=pd.to_datetime(candidates.decision_timestamp,utc=True)
        candidates["symbol"]=candidates.symbol.astype(str).str.upper()
        if candidates.duplicated(["decision_timestamp","symbol"]).any(): raise ValueError("duplicate candidate identity")
        symbols=sorted(candidates.symbol.unique()); all_rows=[]; all_assoc=[]; readiness=[]; run_ids={}; native_runs={}; durations=[]
        phase_totals={"mandatory_preflight":0.0,"latest_coverage":0.0,"common_coverage":0.0,
            "outcome_tail_preflight":0.0,"entry_probe":0.0,"native_research":0.0,
            "outcome_extension":0.0}
        interval=pd.Timedelta(minutes=config.data.strategy_timeframe_minutes)
        warmup_bars=int(self.warmup_bars(config))
        profiles=[key for key,value in config.strategy.profiles.items() if value.enabled]
        finite=all(config.execution.profiles[key].timeout_enabled for key in profiles)
        tail=pd.Timedelta(minutes=max((config.execution.profiles[key].timeout_minutes for key in profiles),default=0)) if finite else None
        for index,symbol in enumerate(symbols,1):
            if cancelled(): raise ValidationCancelled(f"cancelled after {index-1} of {len(symbols)} symbols")
            symbol_started=self.monotonic(); timing={key:0.0 for key in phase_totals}
            subset=candidates[candidates.symbol.eq(symbol)]; start=subset.decision_timestamp.min()-warmup_bars*interval
            mandatory_end=subset.decision_timestamp.max()+horizon
            if mandatory_end <= start:
                raise ValueError(f"{symbol}: no usable market data after required warmup start")
            def preflight_progress(event,i=index,total=len(symbols),s=symbol):
                progress({"symbol_index":i,"symbol_total":total,"symbol":s,
                    "elapsed":sum(durations),"eta":None,**event})

            if self.defer_outcome_tail_until_needed:
                if self.preflight is not None:
                    began=self.monotonic()
                    readiness.extend(self.preflight(symbol,start.to_pydatetime(),mandatory_end.to_pydatetime(),
                        execution_config,coverage_scope="MANDATORY_ENTRY",cancelled=cancelled,progress=preflight_progress))
                    timing["mandatory_preflight"]+=self.monotonic()-began
                progress({"symbol_index":index,"symbol_total":len(symbols),"symbol":symbol,
                    "stage":"Probing candidate entry horizon","elapsed":sum(durations),
                    "eta":(sum(durations)/len(durations)*(len(symbols)-index+1)) if durations else None})
                assert_code_stable(); began=self.monotonic()
                probe: SymbolResearchResult=self.executor(symbol,start.to_pydatetime(),mandatory_end.to_pydatetime(),execution_config)
                timing["entry_probe"]+=self.monotonic()-began; timing["native_research"]+=timing["entry_probe"]
                assert_code_stable()
                final_result=probe; final_rows,final_assoc=_attach_symbol_result(subset,probe,horizon)
                needs_extension=bool(final_rows.outcome_resolution_status.eq("UNRESOLVED").any())
                desired_end=mandatory_end; end=mandatory_end
                tail_status="NOT_REQUIRED_CANDIDATE_WINDOWS_RESOLVED"
                extension_run_id=None
                if needs_extension:
                    began=self.monotonic(); latest=self.latest_available(symbol); timing["latest_coverage"]+=self.monotonic()-began
                    if tail is not None:
                        desired_end=mandatory_end+tail
                    else:
                        desired_end=max(mandatory_end,pd.Timestamp(latest)) if latest is not None else mandatory_end
                    began=self.monotonic()
                    common_end=(self.common_available_end(symbol,start.to_pydatetime(),desired_end.to_pydatetime(),execution_config)
                        if self.common_available_end is not None else desired_end.to_pydatetime())
                    timing["common_coverage"]+=self.monotonic()-began
                    common=pd.Timestamp(common_end) if common_end is not None else mandatory_end
                    end=min(desired_end,common)
                    if end < mandatory_end:
                        raise ValueError(f"{symbol}: required inputs end before mandatory entry-observation coverage")
                    if self.preflight is not None and end > mandatory_end:
                        began=self.monotonic()
                        readiness.extend(self.preflight(symbol,mandatory_end.to_pydatetime(),end.to_pydatetime(),
                            execution_config,coverage_scope="OUTCOME_TAIL",cancelled=cancelled,progress=preflight_progress))
                        timing["outcome_tail_preflight"]+=self.monotonic()-began
                    if end > mandatory_end:
                        progress({"symbol_index":index,"symbol_total":len(symbols),"symbol":symbol,
                            "stage":"Extending native research for unresolved candidate outcome",
                            "elapsed":sum(durations),"eta":None})
                        assert_code_stable(); began=self.monotonic()
                        final_result=self.executor(symbol,start.to_pydatetime(),end.to_pydatetime(),execution_config)
                        timing["outcome_extension"]+=self.monotonic()-began
                        timing["native_research"]+=timing["outcome_extension"]
                        assert_code_stable(); extension_run_id=final_result.research_run_id
                        final_rows,final_assoc=_attach_symbol_result(subset,final_result,horizon)
                    tail_status=("FULLY_RESOLVED_HORIZON_AVAILABLE" if end>=desired_end
                        else "RIGHT_CENSORED_BY_AVAILABLE_DATA")
                run_ids[symbol]=final_result.research_run_id
                native_runs[symbol]={"run_id":final_result.research_run_id,"run_dir":str(final_result.run_dir),
                    "entry_probe_run_id":probe.research_run_id,"tail_extension_run_id":extension_run_id,
                    "tail_extension_performed":bool(extension_run_id),
                    "request_start":pd.Timestamp(final_result.request_start).isoformat(),
                    "request_end":pd.Timestamp(final_result.request_end).isoformat(),
                    "available_through":pd.Timestamp(final_result.available_through).isoformat(),
                    "source_identities":list(final_result.source_identities),
                    "mandatory_entry_observation_end":mandatory_end.isoformat(),
                    "desired_outcome_resolution_end":desired_end.isoformat(),"actual_native_run_end":end.isoformat(),
                    "outcome_tail_status":tail_status,"timings_seconds":timing}
                all_rows.append(final_rows); all_assoc.append(final_assoc)
            else:
                began=self.monotonic(); latest=self.latest_available(symbol); timing["latest_coverage"]+=self.monotonic()-began
                if tail is not None:
                    desired_end=mandatory_end+tail
                else:
                    desired_end=max(mandatory_end,pd.Timestamp(latest)) if latest is not None else mandatory_end
                if self.preflight is not None:
                    began=self.monotonic()
                    readiness.extend(self.preflight(symbol,start.to_pydatetime(),mandatory_end.to_pydatetime(),
                        execution_config,coverage_scope="MANDATORY_ENTRY",cancelled=cancelled,progress=preflight_progress))
                    timing["mandatory_preflight"]+=self.monotonic()-began
                began=self.monotonic()
                common_end=(self.common_available_end(symbol,start.to_pydatetime(),desired_end.to_pydatetime(),execution_config)
                    if self.common_available_end is not None else desired_end.to_pydatetime())
                timing["common_coverage"]+=self.monotonic()-began
                common=pd.Timestamp(common_end) if common_end is not None else mandatory_end
                end=min(desired_end,common)
                if end < mandatory_end:
                    raise ValueError(f"{symbol}: required inputs end before mandatory entry-observation coverage")
                if self.preflight is not None and end > mandatory_end:
                    began=self.monotonic()
                    readiness.extend(self.preflight(symbol,mandatory_end.to_pydatetime(),end.to_pydatetime(),
                        execution_config,coverage_scope="OUTCOME_TAIL",cancelled=cancelled,progress=preflight_progress))
                    timing["outcome_tail_preflight"]+=self.monotonic()-began
                progress({"symbol_index":index,"symbol_total":len(symbols),"symbol":symbol,"stage":"Starting native research","elapsed":sum(durations),"eta":(sum(durations)/len(durations)*(len(symbols)-index+1)) if durations else None})
                assert_code_stable(); began=self.monotonic()
                final_result: SymbolResearchResult=self.executor(symbol,start.to_pydatetime(),end.to_pydatetime(),execution_config)
                timing["native_research"]+=self.monotonic()-began; assert_code_stable()
                run_ids[symbol]=final_result.research_run_id
                native_runs[symbol]={"run_id":final_result.research_run_id,"run_dir":str(final_result.run_dir),
                    "request_start":pd.Timestamp(final_result.request_start).isoformat(),"request_end":pd.Timestamp(final_result.request_end).isoformat(),
                    "available_through":pd.Timestamp(final_result.available_through).isoformat(),"source_identities":list(final_result.source_identities),
                    "mandatory_entry_observation_end":mandatory_end.isoformat(),
                    "desired_outcome_resolution_end":desired_end.isoformat(),"actual_native_run_end":end.isoformat(),
                    "outcome_tail_status":"FULLY_RESOLVED_HORIZON_AVAILABLE" if end>=desired_end else "RIGHT_CENSORED_BY_AVAILABLE_DATA",
                    "timings_seconds":timing}
                final_rows,final_assoc=_attach_symbol_result(subset,final_result,horizon)
                all_rows.append(final_rows); all_assoc.append(final_assoc)

            symbol_elapsed=self.monotonic()-symbol_started; timing["total"]=symbol_elapsed
            native_runs[symbol]["timings_seconds"]={**timing,"total":symbol_elapsed}
            durations.append(symbol_elapsed)
            for key in phase_totals: phase_totals[key]+=timing.get(key,0.0)
        outcomes=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
        associations=pd.concat([x for x in all_assoc if not x.empty],ignore_index=True) if any(not x.empty for x in all_assoc) else pd.DataFrame(columns=["population","final_rank","decision_timestamp","trade_identity","pair_net_r"])
        summary,by_rank,top_k,by_year=aggregate_reports(outcomes,associations)
        readiness_frame=pd.DataFrame(readiness)
        code_commit_end=assert_code_stable(); completed_timestamp=datetime.now(timezone.utc)
        total_elapsed=self.monotonic()-validation_started
        manifest={"validation_run_id":f"strategy-validation-{completed_timestamp:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
                  "started_timestamp":started_timestamp.isoformat(),"created_timestamp":completed_timestamp.isoformat(),"status":"COMPLETE",
                  "code_commit":code_commit_start,"code_commit_end":code_commit_end,"code_provenance_stable":True,
                  "scanner_run_ids":sorted(candidates.scan_run_id.unique()),
                  "scanner_decision_timestamps":sorted(x.isoformat() for x in candidates.decision_timestamp.unique()),
                  "strategy_config_snapshot":"strategy_config_snapshot.json","strategy_config_sha256":config_hash,
                  "authoritative_research_config_sha256":config_hash,
                  "validation_reporting_overrides":VALIDATION_REPORTING_OVERRIDES,
                  "validation_warmup_bars":warmup_bars,
                  "validation_warmup_policy_version":WARMUP_POLICY_VERSION,
                  "strategy_timeframe_minutes":config.data.strategy_timeframe_minutes,
                  "intrabar_timeframe_minutes":config.data.intrabar_timeframe_minutes if config.data.use_intrabar_data else None,
                  "evaluation_horizon":str(horizon),"unique_candidate_symbols":symbols,
                  "native_research_run_ids_by_symbol":run_ids,
                  "native_research_runs_by_symbol":native_runs,
                  "performance_timings":{"total_elapsed_seconds":total_elapsed,
                    "phase_totals_seconds":phase_totals,"symbol_total_seconds":dict(zip(symbols,durations))},
                  "resolution_policy":{"finite":finite,"timeout_tail":str(tail) if tail is not None else None,
                    "defer_outcome_tail_until_needed":self.defer_outcome_tail_until_needed,
                    "timeout_disabled_behavior":"LATEST_AVAILABLE_DATA_AND_NATIVE_END_OF_DATA_CENSORING" if not finite else None},
                  "scanner_candidate_sources":candidates[["scan_run_id","decision_timestamp","symbol","scanner_source_identity"]].to_dict("records") if "scanner_source_identity" in candidates else [],
                  "population_definitions":{STANDARD_SINGLE_SYMBOL:"independent normal native run per symbol (not a combined portfolio)",EVERY_VIABLE_ENTRY:"native overlapping resilience samples; no portfolio metrics"}}
        result=StrategyValidationResult(None,outcomes,summary,by_rank,top_k,by_year,associations,readiness_frame,manifest)
        if not publish: return result
        expected_commit=code_commit_start if self.enforce_stable_code_provenance else None
        return self._publish(result,canonical_config,expected_code_commit=expected_commit)

    def _publish(self,result,canonical_config,*,expected_code_commit=None):
        final=self.output_root/result.manifest["validation_run_id"]; temp=self.output_root/("."+final.name+".tmp")
        self.output_root.mkdir(parents=True,exist_ok=True); shutil.rmtree(temp,ignore_errors=True); temp.mkdir()
        frames={"candidate_trade_outcomes.csv":result.outcomes,"strategy_validation_summary.csv":result.summary,"strategy_validation_by_rank.csv":result.by_rank,"strategy_validation_top_k.csv":result.top_k,"strategy_validation_by_year.csv":result.by_year,"strategy_validation_trade_associations.csv":result.associations,"strategy_validation_data_readiness.csv":result.data_readiness}
        artifacts={}
        for name,frame in frames.items():
            path=temp/name; frame.to_csv(path,index=False); artifacts[name]={"sha256":file_sha256(path),"rows":len(frame)}
        snapshot=temp/"strategy_config_snapshot.json"; snapshot.write_bytes(canonical_config)
        artifacts[snapshot.name]={"sha256":file_sha256(snapshot),"rows":1}
        manifest={**result.manifest,"artifacts":artifacts}; (temp/"validation_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
        (temp/"COMPLETED").write_text(manifest["validation_run_id"]+"\n",encoding="utf-8")
        if expected_code_commit is not None:
            current=_commit()
            if (expected_code_commit!="UNKNOWN" and current!="UNKNOWN" and current!=expected_code_commit):
                shutil.rmtree(temp,ignore_errors=True)
                raise RuntimeError("validation code provenance changed during publication; rerun from one pinned commit")
        temp.rename(final)
        return StrategyValidationResult(final,result.outcomes,result.summary,result.by_rank,result.top_k,result.by_year,result.associations,result.data_readiness,manifest)


def _commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "UNKNOWN"


class HistoricalStrategyValidationService:
    """GUI-safe production application boundary; widgets know no engines."""
    def __init__(self, raw_root: Path, cache_root: Path, output_root: Path,backend=None):
        self.raw_root,self.cache_root,self.output_root=map(Path,(raw_root,cache_root,output_root))
        self.backend=backend

    def validate(self, run_dirs, config_path, horizon, cancelled, progress):
        config=load_validation_config(config_path)
        candidates=extract_final_candidates(run_dirs)
        store=MarketDataStore(self.raw_root,self.cache_root); registry=production_feature_registry()
        backend=self.backend or __import__("crypto_strategy_lab.data.binance.selective_acquisition",fromlist=["BinanceDataHubBackend"]).BinanceDataHubBackend(self.raw_root)
        native_root=self.output_root/"native_research"
        def runner_factory():
            return ResearchRunner(store,registry,PreparedRunCache(self.cache_root),
                NativeStrategyPolicy(),NativeSimulator(),(BayesianSamplingCsvManifestReporter(native_root),))
        def request(symbol,start,end,cfg):
            return build_validation_data_request(symbol,start,end,cfg,registry)
        def latest(symbol):
            return latest_strategy_coverage(
                store.catalog.inventory(store.raw_root,market=MarketKind.FUTURES_UM),
                symbol,config.data.strategy_timeframe_minutes,
            )
        executor=NativeResearchExecutor(runner_factory,request)
        preparer=ValidationDataPreparer(store,backend)
        validator=HistoricalStrategyValidator(executor,warmup_bars=lambda cfg:validation_warmup_bars(cfg,registry),
            output_root=self.output_root,latest_available=latest,preflight=preparer.prepare,
            common_available_end=preparer.common_available_end,
            defer_outcome_tail_until_needed=True,enforce_stable_code_provenance=True)
        current={}
        def outer(event): current.update(event); progress(event)
        store.progress_callback=lambda event:progress({**current,"native_stage":event.get("label",event.get("phase","—"))})
        return validator.run(candidates,config,config_path=Path(config_path),evaluation_horizon=horizon,
            cancelled=cancelled,progress=outer)


def create_historical_strategy_validation_service(raw_root,cache_root,output_root,backend=None):
    return HistoricalStrategyValidationService(raw_root,cache_root,output_root,backend)
