"""Focused Task-9 GUI/application-boundary tests."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from crypto_strategy_lab.opportunity_scoring import SCORING_MODELS
from crypto_strategy_lab.run_manifest import (OPPORTUNITY_SCAN_ARTIFACT_CONTRACT,
    OPPORTUNITY_SCAN_ARTIFACT_VERSION, RUN_MANIFEST_CONTRACT, RUN_MANIFEST_VERSION,
    RunArtifactError, file_sha256)
from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    OpportunityScanResultReader, OpportunityScannerApplicationService,
    Task1To7OpportunityScanner, build_request, create_opportunity_scanner_service)


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


def test_workspace_uses_code_commit_and_has_shutdown_wait_contract():
    source=(Path(__file__).parents[1]/"crypto_strategy_lab/gui/opportunity_scanner_workspace.py").read_text()
    assert "m.get('code_commit','—')" in source
    assert "def closeEvent" in source and "thread.wait()" in source
