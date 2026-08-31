from dataclasses import replace
from datetime import datetime,timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from crypto_strategy_lab.data.binance.selective_acquisition import AcquisitionState,BackendAcquisitionResult
from crypto_strategy_lab.data.quality import DataQualityStatus,MissingCoverageRange
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.validation_data_preflight import (ValidationDataPreparer,
    ValidationDataUnavailable,validation_data_requirements)
from crypto_strategy_lab.historical_strategy_validation import HistoricalStrategyValidator,SymbolResearchResult

START=datetime(2025,1,1,tzinfo=timezone.utc); END=datetime(2025,1,3,tzinfo=timezone.utc)

class Report:
    def __init__(self,ok,gaps=()): self.status=DataQualityStatus.OK if ok else DataQualityStatus.MISSING; self._gaps=tuple(gaps)
    def missing_coverage_ranges(self): return self._gaps
    def has_non_missing_errors(self): return False

class Store:
    def __init__(self,missing=()): self.missing=set(missing); self.refreshes=0
    def refresh_catalog(self): self.refreshes+=1
    def data_quality_report(self,request,dataset,interval=None,required=True):
        key=(request.symbol,dataset,interval)
        return Report(key not in self.missing,() if key not in self.missing else (MissingCoverageRange(request.start,request.end),))
    def source_signature(self,*args,**kwargs): return SimpleNamespace(cache_identity=lambda:"source")

class Backend:
    def __init__(self,store,complete=True): self.store,self.complete,self.calls=store,complete,[]
    def acquire_archive(self,request,cancelled=None):
        self.calls.append(request); key=(request.data_request.symbol,request.dataset,request.interval)
        if self.complete: self.store.missing.discard(key)
        return BackendAcquisitionResult(AcquisitionState.ACQUIRED)

def config(*,intrabar=True,method="ASSET_RETURN",indicator=None):
    base=ResearchRunConfig(); data=replace(base.data,strategy_timeframe_minutes=1440,
        intrabar_timeframe_minutes=1,use_intrabar_data=intrabar)
    features=replace(base.features,market_regime_method=method)
    strategy=base.strategy
    if indicator:
        first=next(iter(strategy.profiles)); strategy=replace(strategy,profiles={k:replace(v,
            enabled=k==first,entry_rules=({"action":"REJECT","indicator":indicator,"condition":"ABOVE","threshold":0},) if k==first else ()) for k,v in strategy.profiles.items()})
    return replace(base,data=data,features=features,strategy=strategy)

def test_1d_strategy_acquires_only_missing_1m_then_reuses_cache():
    missing={("SOLUSDT",DatasetKind.KLINES,"1m")}; store=Store(missing); backend=Backend(store)
    rows=ValidationDataPreparer(store,backend).prepare("SOLUSDT",START,END,config())
    assert len(backend.calls)==1 and backend.calls[0].interval=="1m"
    assert {r["required_role"] for r in rows}=={"STRATEGY","INTRABAR"}
    assert next(r for r in rows if r["required_role"]=="INTRABAR")["state"]=="ACQUIRED"
    backend.calls.clear(); ValidationDataPreparer(store,backend).prepare("SOLUSDT",START,END,config())
    assert backend.calls==[]

def test_intrabar_failure_is_blocking_and_disabled_intrabar_is_not_requested():
    store=Store({("SOLUSDT",DatasetKind.KLINES,"1m")}); backend=Backend(store,complete=False)
    with pytest.raises(ValidationDataUnavailable,match="1m klines"):
        ValidationDataPreparer(store,backend).prepare("SOLUSDT",START,END,config())
    assert all(r.role!="INTRABAR" for r in validation_data_requirements("SOLUSDT",START,END,config(intrabar=False)))

@pytest.mark.parametrize(("method","benchmark"),[("BTC_STRUCTURAL","BTCUSDT"),("ASSET_STRUCTURAL","SOLUSDT")])
def test_structural_benchmark_requirement_is_prepared(method,benchmark):
    requirements=validation_data_requirements("SOLUSDT",START,END,config(intrabar=False,method=method))
    row=next(r for r in requirements if r.role=="STRUCTURAL_BENCHMARK")
    assert row.request.symbol==benchmark and row.interval=="1h" and row.request.start<START

@pytest.mark.parametrize(("indicator","dataset"),[("FUNDING_BIAS",DatasetKind.FUNDING_RATE),("OI_CHANGE_PCT_1H",DatasetKind.FUTURES_METRICS)])
def test_required_rule_evidence_is_not_optional(indicator,dataset):
    requirements=validation_data_requirements("SOLUSDT",START,END,config(intrabar=False,indicator=indicator))
    assert any(r.role=="STRATEGY_CONTEXT" and r.dataset is dataset for r in requirements)

def test_oi_price_state_promotes_supporting_1h_klines_to_required():
    requirements=validation_data_requirements("SOLUSDT",START,END,
        config(intrabar=False,indicator="OI_VS_PRICE_STATE_1H"))
    assert any(r.role=="STRATEGY_CONTEXT" and r.dataset is DatasetKind.KLINES
        and r.interval=="1h" for r in requirements)

def test_optional_unused_futures_families_are_not_requested():
    requirements=validation_data_requirements("SOLUSDT",START,END,config(intrabar=False))
    assert not any(r.dataset in {DatasetKind.FUNDING_RATE,DatasetKind.FUTURES_METRICS} for r in requirements)

def test_1d_1m_preflight_completes_before_single_native_symbol_run(tmp_path):
    store=Store({("SOLUSDT",DatasetKind.KLINES,"1m")}); backend=Backend(store); native=[]
    def execute(symbol,start,end,used):
        native.append(symbol); standard=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"])
        viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        return SymbolResearchResult("run",standard,viable,viable.copy(),end,tmp_path/start.isoformat(),start,end)
    candidates=pd.DataFrame({"decision_timestamp":pd.to_datetime(["2025-01-02T00:00Z"]),
        "final_rank":[1],"symbol":["SOLUSDT"],"scan_run_id":["scan"]})
    preparer=ValidationDataPreparer(store,backend)
    HistoricalStrategyValidator(execute,warmup_bars=lambda _:0,preflight=preparer.prepare).run(
        candidates,config(),config_path=tmp_path/"config",publish=False)
    assert native==["SOLUSDT"] and len(backend.calls)==1 and backend.calls[0].interval=="1m"
