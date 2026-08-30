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
from .research_sampling import generate_strategy_research_samples
from .opportunity_validation import load_scanner_observations
from .run_manifest import file_sha256

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
    available_through: datetime


@dataclass(frozen=True)
class StrategyValidationResult:
    run_dir: Path | None
    outcomes: pd.DataFrame
    summary: pd.DataFrame
    by_rank: pd.DataFrame
    top_k: pd.DataFrame
    by_year: pd.DataFrame
    manifest: Mapping[str, object]


class NativeResearchExecutor:
    """Adapter that obtains both populations from one ``ResearchRunner.run``.

    ``request_factory`` owns DataRequest construction (and therefore the normal
    Data Lake acquisition/quality contract). A temporary capture reporter sees
    the exact prepared/intrabar frames already built by ResearchRunner and calls
    the existing sampling engine; neither frame nor any strategy rule is rebuilt.
    """
    def __init__(self, runner, request_factory, run_id_factory=None):
        self.runner, self.request_factory = runner, request_factory
        self.run_id_factory = run_id_factory or (lambda symbol: f"native-{symbol}-{uuid4().hex}")

    def __call__(self, symbol, start, end, config):
        captured = {}
        class CaptureReporter:
            def report(_self, result, context):
                native = native_simulator_config(
                    context.config.data, context.config.features,
                    context.config.strategy, context.config.execution,
                )
                captured["viable"] = generate_strategy_research_samples(
                    context.prepared,
                    # Preserve the exact intrabar frame selected by ResearchRunner.
                    __import__("crypto_strategy_lab.prepared_backtest", fromlist=["intrabar_from_data_lake_bundle"]).intrabar_from_data_lake_bundle(context.bundle),
                    native, mode=EVERY_VIABLE_ENTRY, interval_candles=1,
                )
        original = self.runner.reporters
        self.runner.reporters = (*original, CaptureReporter())
        try:
            result = self.runner.run(self.request_factory(symbol,start,end,config),config)
        finally:
            self.runner.reporters = original
        viable=captured.get("viable",pd.DataFrame())
        if viable.empty:
            viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        return SymbolResearchResult(self.run_id_factory(symbol),result.trades,viable,end)


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


def _trade_id(frame: pd.DataFrame, population: str, run_id: str) -> pd.Series:
    native = "research_sample_id" if population == EVERY_VIABLE_ENTRY else "pair_id"
    if native not in frame:
        raise ValueError(f"{population} native rows require authoritative {native}")
    return run_id + "|" + frame["symbol"].astype(str) + "|" + frame[native].astype(str)


def attach_candidate_trades(candidates: pd.DataFrame, trades: pd.DataFrame, *,
                            population: str, run_id: str, horizon: pd.Timedelta,
                            available_through) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    coverage = pd.Timestamp(available_through)
    coverage = coverage.tz_localize("UTC") if coverage.tzinfo is None else coverage.tz_convert("UTC")
    for candidate in candidates.itertuples(index=False):
        decision = pd.Timestamp(candidate.decision_timestamp)
        end = decision + horizon
        complete = coverage >= end
        selected = native[(native.symbol == candidate.symbol) & (native.entry_time >= decision) & (native.entry_time < end)]
        # Existing sampling removes END_OF_DATA. Be defensive for standard native rows.
        reason_columns = [c for c in ("exit_reason", "long_exit_reason", "short_exit_reason") if c in selected]
        censored = pd.Series(False, index=selected.index)
        for column in reason_columns:
            censored |= selected[column].astype(str).str.upper().eq("END_OF_DATA")
        resolved = selected.loc[~censored & selected.pair_net_r.notna()]
        wins, losses = int(resolved.pair_net_r.gt(0).sum()), int(resolved.pair_net_r.lt(0).sum())
        neutrals = int(resolved.pair_net_r.eq(0).sum())
        unresolved = not complete or (len(selected) > 0 and resolved.empty)
        if unresolved: result = "UNRESOLVED"
        elif selected.empty: result = "NO_ENTRY"
        elif len(resolved) > 1: result = "MULTIPLE"
        elif wins: result = "WIN"
        elif losses: result = "LOSS"
        else: result = "NEUTRAL"
        sides = sorted(set(selected.side.dropna().astype(str).str.upper()))
        side = sides[0] if len(sides) == 1 else "MIXED" if sides else "—"
        base = {"decision_timestamp": decision, "final_rank": int(candidate.final_rank),
                "symbol": candidate.symbol, "scan_run_id": candidate.scan_run_id,
                "population": population, "research_run_id": run_id,
                "evaluation_horizon": str(horizon),
                "future_data_status": "COMPLETE" if complete else "INSUFFICIENT_FUTURE_DATA",
                "valid_entry": bool(len(selected)), "entry_count": len(selected),
                "completed_trade_count": len(resolved), "wins": wins, "losses": losses,
                "neutrals": neutrals, "win_rate_resolved": wins/(wins+losses) if wins+losses else None,
                "win_share_all_completed": wins/len(resolved) if len(resolved) else None,
                "average_r": resolved.pair_net_r.mean() if len(resolved) else None,
                "net_r": resolved.pair_net_r.sum() if len(resolved) else None,
                "first_entry_time": selected.entry_time.min() if len(selected) else None,
                "last_entry_time": selected.entry_time.max() if len(selected) else None,
                "side": side, "result": result}
        rows.append(base)
        for trade in resolved.itertuples(index=False):
            associations.append({**{k: base[k] for k in ("decision_timestamp", "final_rank", "symbol", "population")},
                                 "trade_identity": trade.trade_identity, "pair_net_r": trade.pair_net_r,
                                 "entry_time": trade.entry_time, "side": str(trade.side).upper()})
    return pd.DataFrame(rows), pd.DataFrame(associations)


def _aggregate(windows: pd.DataFrame, associations: pd.DataFrame) -> dict:
    observed = windows[windows.future_data_status.eq("COMPLETE")]
    unique = associations.drop_duplicates("trade_identity") if not associations.empty else associations
    wins = int(unique.pair_net_r.gt(0).sum()) if len(unique) else 0
    losses = int(unique.pair_net_r.lt(0).sum()) if len(unique) else 0
    neutrals = int(unique.pair_net_r.eq(0).sum()) if len(unique) else 0
    return {"candidate_observations": len(windows), "observed_candidate_windows": len(observed),
            "unresolved_candidate_windows": len(windows)-len(observed),
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
    """Sequential one-native-run-per-symbol coordinator."""
    def __init__(self, executor: Callable, *, warmup_bars: Callable[[ResearchRunConfig], int],
                 output_root: Path = Path("output/opportunity_validation"), monotonic=time.monotonic):
        self.executor, self.warmup_bars = executor, warmup_bars
        self.output_root, self.monotonic = Path(output_root), monotonic

    def run(self, candidates: pd.DataFrame, config: ResearchRunConfig, *, config_path: Path,
            evaluation_horizon="24h", cancelled=lambda:False, progress=lambda event:None,
            publish=True) -> StrategyValidationResult:
        config.validate(); horizon=pd.Timedelta(evaluation_horizon)
        if horizon <= pd.Timedelta(0): raise ValueError("entry evaluation horizon must be positive")
        required={"decision_timestamp","final_rank","symbol","scan_run_id"}
        if missing:=required-set(candidates): raise ValueError(f"candidates missing columns: {sorted(missing)}")
        candidates=candidates.copy(); candidates["decision_timestamp"]=pd.to_datetime(candidates.decision_timestamp,utc=True)
        candidates["symbol"]=candidates.symbol.astype(str).str.upper()
        if candidates.duplicated(["decision_timestamp","symbol"]).any(): raise ValueError("duplicate candidate identity")
        symbols=sorted(candidates.symbol.unique()); all_rows=[]; all_assoc=[]; run_ids={}; durations=[]
        interval=pd.Timedelta(minutes=config.data.strategy_timeframe_minutes)
        for index,symbol in enumerate(symbols,1):
            if cancelled(): raise ValidationCancelled(f"cancelled after {index-1} of {len(symbols)} symbols")
            subset=candidates[candidates.symbol.eq(symbol)]; start=subset.decision_timestamp.min()-self.warmup_bars(config)*interval
            end=subset.decision_timestamp.max()+horizon
            progress({"symbol_index":index,"symbol_total":len(symbols),"symbol":symbol,"stage":"Starting native research","elapsed":sum(durations),"eta":(sum(durations)/len(durations)*(len(symbols)-index+1)) if durations else None})
            began=self.monotonic()
            result: SymbolResearchResult=self.executor(symbol,start.to_pydatetime(),end.to_pydatetime(),config)
            durations.append(self.monotonic()-began); run_ids[symbol]=result.research_run_id
            for population,trades in ((STANDARD_SINGLE_SYMBOL,result.standard_trades),(EVERY_VIABLE_ENTRY,result.viable_trades)):
                rows,assoc=attach_candidate_trades(subset,trades,population=population,run_id=result.research_run_id,horizon=horizon,available_through=result.available_through)
                all_rows.append(rows); all_assoc.append(assoc)
        outcomes=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
        associations=pd.concat(all_assoc,ignore_index=True) if any(not x.empty for x in all_assoc) else pd.DataFrame(columns=["population","final_rank","decision_timestamp","trade_identity","pair_net_r"])
        summary,by_rank,top_k,by_year=aggregate_reports(outcomes,associations)
        manifest={"validation_run_id":f"strategy-validation-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
                  "created_timestamp":datetime.now(timezone.utc).isoformat(),"status":"COMPLETE",
                  "code_commit":_commit(),"scanner_run_ids":sorted(candidates.scan_run_id.unique()),
                  "scanner_decision_timestamps":sorted(x.isoformat() for x in candidates.decision_timestamp.unique()),
                  "strategy_config_path":str(config_path),"strategy_config_sha256":file_sha256(Path(config_path)),
                  "strategy_timeframe_minutes":config.data.strategy_timeframe_minutes,
                  "intrabar_timeframe_minutes":config.data.intrabar_timeframe_minutes if config.data.use_intrabar_data else None,
                  "evaluation_horizon":str(horizon),"unique_candidate_symbols":symbols,
                  "native_research_run_ids_by_symbol":run_ids,
                  "population_definitions":{STANDARD_SINGLE_SYMBOL:"independent normal native run per symbol (not a combined portfolio)",EVERY_VIABLE_ENTRY:"native overlapping resilience samples; no portfolio metrics"}}
        result=StrategyValidationResult(None,outcomes,summary,by_rank,top_k,by_year,manifest)
        return self._publish(result) if publish else result

    def _publish(self,result):
        final=self.output_root/result.manifest["validation_run_id"]; temp=self.output_root/("."+final.name+".tmp")
        self.output_root.mkdir(parents=True,exist_ok=True); shutil.rmtree(temp,ignore_errors=True); temp.mkdir()
        frames={"candidate_trade_outcomes.csv":result.outcomes,"strategy_validation_summary.csv":result.summary,"strategy_validation_by_rank.csv":result.by_rank,"strategy_validation_top_k.csv":result.top_k,"strategy_validation_by_year.csv":result.by_year}
        artifacts={}
        for name,frame in frames.items():
            path=temp/name; frame.to_csv(path,index=False); artifacts[name]={"sha256":file_sha256(path),"rows":len(frame)}
        manifest={**result.manifest,"artifacts":artifacts}; (temp/"validation_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
        (temp/"COMPLETED").write_text(manifest["validation_run_id"]+"\n",encoding="utf-8"); temp.rename(final)
        return StrategyValidationResult(final,result.outcomes,result.summary,result.by_rank,result.top_k,result.by_year,manifest)


def _commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "UNKNOWN"
