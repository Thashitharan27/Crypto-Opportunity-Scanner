"""Focused Task-9 GUI/application-boundary tests."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from crypto_strategy_lab.opportunity_scoring import SCORING_MODELS
from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionState, CandleAcquisitionResult, SymbolAcquisitionResult,
)
from crypto_strategy_lab.data.binance.universe import DiscoveryRow
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.rich_data_acquisition import (
    RichDataAcquisitionPlan, RichDataAcquisitionResult, SymbolRichDataResult,
)
from crypto_strategy_lab.run_manifest import (OPPORTUNITY_SCAN_ARTIFACT_CONTRACT,
    OPPORTUNITY_SCAN_ARTIFACT_VERSION, RUN_MANIFEST_CONTRACT, RUN_MANIFEST_VERSION,
    RunArtifactError, file_sha256)
from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    OpportunityScanResultReader, OpportunityScannerApplicationService,
    OpportunityScanCancelled, Task1To7OpportunityScanner, build_request,
    create_opportunity_scanner_service)


def test_gui_values_map_to_native_task_configs():
    model=SCORING_MODELS[1]
    request=build_request(mode="HISTORICAL",decision_time=datetime(2026,1,2,3,4,tzinfo=timezone.utc),
        minimum_listing_age_days=45,minimum_quote_volume=Decimal("123.5"),
        maximum_spread_percent=Decimal("0.125"),preliminary_size=17,final_size=6,
        strategy_interval="4h",model=model,enabled_features=("funding_context",))
    assert request.live_discovery.minimum_listing_age.days == 45
    assert request.live_discovery.minimum_quote_volume == request.historical_discovery.minimum_quote_volume == Decimal("123.5")
    assert request.live_discovery.maximum_spread_percent == Decimal("0.125")
    assert request.candle_acquisition.shortlist_size == 17
    assert request.final_candidates.max_candidates == 6
    assert request.scoring.models == (model,)
    assert request.final_candidates.opportunity_model.name == model.name
    assert request.candle_acquisition.strategy_interval == request.final_candidates.strategy_interval == "4h"
    assert request.rich_data.enabled_features == ("funding_context",)
    no_model=build_request(mode="LIVE",decision_time=None,minimum_listing_age_days=30,
        minimum_quote_volume=Decimal(0),maximum_spread_percent=Decimal(0),preliminary_size=1,
        final_size=1,strategy_interval="1h",model=None,enabled_features=())
    assert no_model.scoring is None and no_model.final_candidates.opportunity_model is None
    with pytest.raises(ValueError,match="supports only 1h"):
        build_request(mode="LIVE",decision_time=None,minimum_listing_age_days=30,
            minimum_quote_volume=Decimal(0),maximum_spread_percent=Decimal(1),
            preliminary_size=1,final_size=1,strategy_interval="4h",
            model=SCORING_MODELS[0],enabled_features=())


def _published_run(directory: Path):
    frames={"universe_snapshot":"symbol,eligible,range_percent\nBTCUSDT,True,2.5\n",
        "discovery_rejections":"symbol,eligible\n",
        "preliminary_candidates":"symbol,discovery_rank\nBTCUSDT,1\n",
        "final_candidates":"symbol,final_rank\nBTCUSDT,1\n",
        "final_candidate_exclusions":"symbol,reason,detail\n",
        "rich_data_readiness":"symbol,feature_readiness\nBTCUSDT,DEGRADED\n"}
    artifacts={}
    def entry(path,rows=0):
        return {"path":path.name,"format":path.suffix.lstrip("."),"schema_version":1,
            "rows":rows,"bytes":path.stat().st_size,"sha256":file_sha256(path)}
    for name,body in frames.items():
        path=directory/f"{name}.csv"; path.write_text(body); artifacts[name]=entry(path,max(0,len(body.splitlines())-1))
    summary=directory/"summary.json"; summary.write_text(json.dumps({"final_candidate_count":1}))
    artifacts["opportunity_summary"]=entry(summary)
    sources=directory/"source_identities.json"; sources.write_text('{"entries": []}')
    artifacts["source_identities"]=entry(sources)
    manifest={"run_manifest_contract":RUN_MANIFEST_CONTRACT,"run_manifest_version":RUN_MANIFEST_VERSION,
        "run_type":"OPPORTUNITY_SCAN","run_id":"scan-1","run_status":"COMPLETED",
        "opportunity_scan":{"artifact_contract":OPPORTUNITY_SCAN_ARTIFACT_CONTRACT,
            "artifact_version":OPPORTUNITY_SCAN_ARTIFACT_VERSION,"scan_timestamp":"2026-01-01T00:00:00+00:00",
            "decision_timestamp":"2026-01-01T00:00:00+00:00","discovery_mode":"LIVE",
            "discovery_contract":"binance_usdm_live_24h_v1","universe_definition":{},"counts":{},
            "source_identity_digest":"0"*64,"source_identities_artifact":"source_identities"},"artifacts":artifacts}
    (directory/"run_manifest.json").write_text(json.dumps(manifest)); return directory


def test_completed_scan_is_loaded_only_through_verified_catalog(tmp_path):
    result=OpportunityScanResultReader().read(_published_run(tmp_path))
    assert result.summary["final_candidate_count"] == 1
    assert result.preliminary.iloc[0]["range_percent"] == 2.5
    assert result.scores.empty and result.readiness.iloc[0]["feature_readiness"] == "DEGRADED"
    (tmp_path/"final_candidates.csv").write_text("tampered")
    with pytest.raises(RunArtifactError,match="integrity"):
        OpportunityScanResultReader().read(tmp_path)


def test_scored_preliminary_rows_join_published_model_rank_and_score(tmp_path):
    _published_run(tmp_path)
    scores=tmp_path/"opportunity_scores.csv"
    scores.write_text(
        "symbol,model_name,model_version,model_rank,score\n"
        "BTCUSDT,legacy_volatility,1,1,0.99\n"
        "BTCUSDT,balanced_activity,1,2,0.75\n"
    )
    manifest=json.loads((tmp_path/"run_manifest.json").read_text())
    manifest["artifacts"]["opportunity_scores"]={
        "path":scores.name,"format":"csv","schema_version":1,"rows":2,
        "bytes":scores.stat().st_size,"sha256":file_sha256(scores),
    }
    manifest["config"]={
        "scoring":{"models":[
            {"name":"legacy_volatility","version":"1"},
            {"name":"balanced_activity","version":"1"},
        ]},
        "final_candidates":{"opportunity_model":{
            "name":"balanced_activity","version":"1",
        }},
    }
    (tmp_path/"run_manifest.json").write_text(json.dumps(manifest))
    row=OpportunityScanResultReader().read(tmp_path).preliminary.iloc[0]
    assert row["model_rank"] == 2 and row["score"] == .75
    assert len(OpportunityScanResultReader().read(tmp_path).preliminary) == 1


def test_no_selected_model_does_not_merge_task4_comparison_scores(tmp_path):
    _published_run(tmp_path)
    scores=tmp_path/"opportunity_scores.csv"
    scores.write_text(
        "symbol,model_name,model_version,model_rank,score\n"
        "BTCUSDT,legacy_volatility,1,1,0.99\n"
        "BTCUSDT,balanced_activity,1,2,0.75\n"
    )
    manifest=json.loads((tmp_path/"run_manifest.json").read_text())
    manifest["artifacts"]["opportunity_scores"]={
        "path":scores.name,"format":"csv","schema_version":1,"rows":2,
        "bytes":scores.stat().st_size,"sha256":file_sha256(scores),
    }
    manifest["config"]={
        "scoring":{"models":[{"name":"balanced_activity","version":"1"}]},
        "final_candidates":{"opportunity_model":None},
    }
    (tmp_path/"run_manifest.json").write_text(json.dumps(manifest))
    preliminary=OpportunityScanResultReader().read(tmp_path).preliminary
    assert len(preliminary) == 1
    assert "model_rank" not in preliminary and "score" not in preliminary


def test_application_service_invokes_pipeline_once_then_reads_publication(tmp_path):
    calls=[]
    service=OpportunityScannerApplicationService(lambda request,cancelled:(calls.append(request) or _published_run(tmp_path)))
    request=build_request(mode="LIVE",decision_time=None,minimum_listing_age_days=1,
        minimum_quote_volume=Decimal(0),maximum_spread_percent=Decimal(1),preliminary_size=1,
        final_size=1,strategy_interval="1h",model=None,enabled_features=())
    assert service.run(request,lambda:False).manifest["run_id"] == "scan-1"
    assert calls == [request]


def test_normal_service_factory_installs_real_task_1_to_7_pipeline(tmp_path):
    service=create_opportunity_scanner_service(tmp_path/"raw",tmp_path/"cache",tmp_path/"runs")
    assert isinstance(service._run_once,Task1To7OpportunityScanner)


@pytest.mark.parametrize("mode",["LIVE","HISTORICAL"])
def test_off_grid_decision_aligns_task3_but_preserves_pipeline_boundary(
        monkeypatch,tmp_path,mode):
    import crypto_strategy_lab.gui.opportunity_scanner_controller as controller
    decision=datetime(2026,1,2,12,34,56,tzinfo=timezone.utc)
    calls=[]
    live=[SimpleNamespace(discovery_timestamp=decision)]
    historical=SimpleNamespace(decision_time=SimpleNamespace(value=decision))
    monkeypatch.setattr(controller,"scan_universe",lambda *a,**k:live)
    def discover(_store,symbols,boundary,*args,**kwargs):
        calls.append(("historical",boundary.value)); return historical
    monkeypatch.setattr(controller,"discover_historical_universe",discover)
    class CandleStage:
        def __init__(self,*args): pass
        def acquire(self,discovery,start,end,**kwargs):
            calls.append(("task3",start,end,discovery));
            return CandleAcquisitionResult(())
    class RichStage:
        def __init__(self,*args): pass
        def acquire(self,final,**kwargs): calls.append(("task6",final)); return "rich"
    monkeypatch.setattr(controller,"SelectiveCandleAcquirer",CandleStage)
    monkeypatch.setattr(controller,"SelectiveRichDataAcquirer",RichStage)
    monkeypatch.setattr(controller,"build_final_candidate_set",
        lambda discovery,candles,scoring,config:(calls.append(("task5",discovery)) or "final"))
    monkeypatch.setattr(controller,"publish_opportunity_scan",
        lambda root,package:(calls.append(("task7",package)) or tmp_path/"published"))
    store=SimpleNamespace(raw_root=tmp_path,catalog=SimpleNamespace(
        inventory=lambda *a,**k:[{"symbol":"BTCUSDT"}]))
    registry=SimpleNamespace(effective_warmup=lambda names:50)
    pipeline=Task1To7OpportunityScanner(store,object(),tmp_path,
        live_client=object(),registry=registry,now=lambda:decision)
    request=build_request(mode=mode,decision_time=decision if mode=="HISTORICAL" else None,
        minimum_listing_age_days=30,minimum_quote_volume=Decimal(0),
        maximum_spread_percent=Decimal(1),preliminary_size=1,final_size=1,
        strategy_interval="1h",model=None,enabled_features=())
    assert pipeline(request,lambda:False) == tmp_path/"published"
    task3=next(call for call in calls if call[0]=="task3")
    assert task3[2] == datetime(2026,1,2,12,0,tzinfo=timezone.utc)
    assert task3[1] == datetime(2025,12,31,9,0,tzinfo=timezone.utc)
    assert next(call for call in calls if call[0]=="task5")[1] is task3[3]
    if mode == "HISTORICAL":
        assert ("historical",decision) in calls
    assert any(call[0]=="task7" for call in calls)


def test_cancelled_facade_never_enters_task7_publication(monkeypatch,tmp_path):
    import crypto_strategy_lab.gui.opportunity_scanner_controller as controller
    decision=datetime(2026,1,2,12,tzinfo=timezone.utc); cancelled=[False]
    monkeypatch.setattr(controller,"scan_universe",lambda *a,**k:
        [SimpleNamespace(discovery_timestamp=decision)])
    class CandleStage:
        def __init__(self,*args): pass
        def acquire(self,*args,**kwargs): return CandleAcquisitionResult(())
    class RichStage:
        def __init__(self,*args): pass
        def acquire(self,*args,**kwargs): cancelled[0]=True; return "rich"
    monkeypatch.setattr(controller,"SelectiveCandleAcquirer",CandleStage)
    monkeypatch.setattr(controller,"SelectiveRichDataAcquirer",RichStage)
    monkeypatch.setattr(controller,"build_final_candidate_set",lambda *a:"final")
    published=[]
    monkeypatch.setattr(controller,"publish_opportunity_scan",
        lambda *a:(published.append(True) or tmp_path))
    pipeline=Task1To7OpportunityScanner(SimpleNamespace(),object(),tmp_path,
        live_client=object(),registry=SimpleNamespace(effective_warmup=lambda _:50),
        now=lambda:decision)
    request=build_request(mode="LIVE",decision_time=None,minimum_listing_age_days=1,
        minimum_quote_volume=Decimal(0),maximum_spread_percent=Decimal(1),
        preliminary_size=1,final_size=1,strategy_interval="1h",model=None,
        enabled_features=())
    with pytest.raises(OpportunityScanCancelled):
        pipeline(request,lambda:cancelled[0])
    assert published == []


def test_injected_facade_reaches_real_task7_publication(monkeypatch,tmp_path):
    import crypto_strategy_lab.gui.opportunity_scanner_controller as controller
    decision=datetime(2026,1,2,12,34,56,tzinfo=timezone.utc)
    discovery=DiscoveryRow("BTCUSDT",True,(),pd.Timedelta(days=100).to_pytimedelta(),
        Decimal("20000000"),Decimal("10"),Decimal("10.01"),Decimal("0.1"),
        Decimal("8"),Decimal("-3"),1,decision)
    signature=SourceSignature(DatasetKind.KLINES,"scanner-facade",1)
    class CandleStage:
        def __init__(self,*args): pass
        def acquire(self,_discovery,start,end,**kwargs):
            return CandleAcquisitionResult((SymbolAcquisitionResult(
                "BTCUSDT",1,AcquisitionState.REUSED,"1h",start,end,
                row_count=51,source_signature=signature),))
    class RichStage:
        def __init__(self,*args): pass
        def acquire(self,final,**kwargs):
            candidate=final.candidates[0]
            return RichDataAcquisitionResult(RichDataAcquisitionPlan((),()),(
                SymbolRichDataResult(candidate.symbol,candidate.final_rank,
                    candidate.strategy_source_identity,(),()),))
    monkeypatch.setattr(controller,"scan_universe",lambda *a,**k:[discovery])
    monkeypatch.setattr(controller,"SelectiveCandleAcquirer",CandleStage)
    monkeypatch.setattr(controller,"SelectiveRichDataAcquirer",RichStage)
    pipeline=Task1To7OpportunityScanner(SimpleNamespace(),object(),tmp_path,
        live_client=object(),registry=SimpleNamespace(effective_warmup=lambda _:50),
        now=lambda:decision)
    request=build_request(mode="LIVE",decision_time=None,minimum_listing_age_days=1,
        minimum_quote_volume=Decimal(0),maximum_spread_percent=Decimal(1),
        preliminary_size=1,final_size=1,strategy_interval="1h",model=None,
        enabled_features=())
    run_dir=pipeline(request,lambda:False)
    completed=OpportunityScanResultReader().read(run_dir)
    assert completed.manifest["run_type"] == "OPPORTUNITY_SCAN"
    assert completed.summary["decision_timestamp"] == decision.isoformat()
    assert completed.final.iloc[0]["symbol"] == "BTCUSDT"


def test_workspace_constructs_and_mode_controls_utc_timestamp():
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    widgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import OpportunityScannerWorkspace
    app=widgets.QApplication.instance() or widgets.QApplication([])
    workspace=OpportunityScannerWorkspace(object())
    try:
        assert not workspace.decision_time.isEnabled()
        workspace.mode.setCurrentIndex(1)
        assert workspace.decision_time.isEnabled()
        assert not workspace.listing_age.isEnabled()
        assert not workspace.spread.isEnabled()
        assert workspace.volume.isEnabled()
        assert workspace.request().decision_time.tzinfo is not None
        assert [workspace.model.itemData(i) for i in range(1,workspace.model.count())] == list(SCORING_MODELS)
    finally: workspace.close()


def test_validation_worker_retains_complete_range_and_runs_offscreen(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    widgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from PySide6.QtCore import QEventLoop, QTimer
    from crypto_strategy_lab.data_lake_config import ResearchRunConfig
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import OpportunityScannerWorkspace
    calls=[]
    class Validation:
        def validate(self,dirs,path,horizon,cancelled,progress):
            calls.append((dirs,path,horizon)); progress({"symbol_index":1,"symbol_total":1,"symbol":"BTCUSDT","native_stage":"Preparing data & research features","elapsed":0,"eta":None})
            return SimpleNamespace(run_dir=tmp_path/"validation",outcomes=pd.DataFrame(),summary=pd.DataFrame(),by_rank=pd.DataFrame(),top_k=pd.DataFrame())
    service=SimpleNamespace(validation_service=Validation())
    app=widgets.QApplication.instance() or widgets.QApplication([]); workspace=OpportunityScannerWorkspace(service)
    config=tmp_path/"config.json"; config.write_text(json.dumps(ResearchRunConfig().to_dict()))
    scan=lambda number:SimpleNamespace(run_dir=tmp_path/f"scan-{number}",manifest={"opportunity_scan":{"discovery_mode":"HISTORICAL"}},summary={},preliminary=pd.DataFrame(),final=pd.DataFrame({"symbol":["BTCUSDT"]}),readiness=pd.DataFrame())
    replay=SimpleNamespace(completed=(scan(1),scan(2)),last=scan(2),elapsed_seconds=2,decision_points=(datetime(2026,1,1,tzinfo=timezone.utc),datetime(2026,1,2,tzinfo=timezone.utc)))
    try:
        workspace.mode.setCurrentIndex(1); workspace._completed(replay)
        assert workspace._validation_scan_run_dirs==(tmp_path/"scan-1",tmp_path/"scan-2") and workspace.validate_button.isEnabled()
        workspace.validation_config.setText(str(config)); workspace.validate_button.click()
        loop=QEventLoop(); QTimer.singleShot(3000,loop.quit)
        while workspace._validation_thread is not None: QTimer.singleShot(20,loop.quit); loop.exec()
        assert calls and calls[0][0]==workspace._validation_scan_run_dirs
        assert not workspace.validation_cancel.isEnabled()
    finally: workspace.shutdown(); workspace.close()


def test_validation_live_switch_mutual_exclusion_and_elapsed_timer(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    widgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from PySide6.QtCore import QEventLoop, QTimer
    from threading import Event
    from crypto_strategy_lab.data_lake_config import ResearchRunConfig
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import OpportunityScannerWorkspace
    release=Event()
    class Validation:
        def validate(self,*args):
            progress=args[-1]; progress({"symbol_index":1,"symbol_total":1,"symbol":"BTCUSDT","native_stage":"Running strategy simulation","eta":None})
            release.wait(3)
            return SimpleNamespace(run_dir=tmp_path,outcomes=pd.DataFrame(),summary=pd.DataFrame(),by_rank=pd.DataFrame(),top_k=pd.DataFrame())
    workspace=OpportunityScannerWorkspace(SimpleNamespace(validation_service=Validation()))
    config=tmp_path/"config.json"; config.write_text(json.dumps(ResearchRunConfig().to_dict()))
    try:
        workspace.mode.setCurrentIndex(1); workspace._validation_scan_run_dirs=(tmp_path/"scan",)
        workspace.validation_config.setText(str(config)); workspace._sync_controls(); workspace.validate_button.click()
        app=widgets.QApplication.instance(); app.processEvents()
        assert not workspace.run_button.isEnabled() and not workspace.validate_button.isEnabled()
        workspace._validation_started-=2; workspace._update_validation_elapsed()
        assert "Elapsed: 00:00:02" in workspace.validation_progress.text()
        workspace.mode.setCurrentIndex(0); release.set()
        loop=QEventLoop(); QTimer.singleShot(3000,loop.quit)
        while workspace._validation_thread is not None: QTimer.singleShot(20,loop.quit); loop.exec()
        assert workspace.run_button.isEnabled() and not workspace.validate_button.isEnabled()
    finally: release.set(); workspace.shutdown(); workspace.close()


def test_workspace_uses_code_commit_and_has_shutdown_wait_contract():
    source=(Path(__file__).parents[1]/"crypto_strategy_lab/gui/opportunity_scanner_workspace.py").read_text(encoding="utf-8")
    assert "m.get('code_commit','—')" in source
    assert "def shutdown" in source and "thread.wait()" in source
    assert '"acquisition_state","quality_status","detail"' in source
    installer=(Path(__file__).parents[1]/"crypto_strategy_lab/gui/opportunity_scanner_install.py").read_text(encoding="utf-8")
    assert "application.aboutToQuit.connect(workspace.shutdown)" in installer
