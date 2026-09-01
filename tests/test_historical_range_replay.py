"""Offline application-layer contracts for historical range replay."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    HISTORICAL_REPLAY_CADENCES, HistoricalRangeRunner, HistoricalReplayFailure,
    OpportunityScanCancelled, OpportunityScanProgress,
    OpportunityScannerApplicationService, build_request, historical_decision_points,
)

UTC = timezone.utc
T = datetime(2025, 2, 6, 0, 5, tzinfo=UTC)


def native_request(decision):
    return build_request(mode="HISTORICAL", decision_time=decision,
        minimum_listing_age_days=30, minimum_quote_volume=Decimal("10"),
        maximum_spread_percent=Decimal("1"), preliminary_size=12, final_size=4,
        strategy_interval="4h", model=None, enabled_features=("funding_context",))


@pytest.mark.parametrize("name,step", [("1h",timedelta(hours=1)),("4h",timedelta(hours=4)),("1d",timedelta(days=1))])
def test_exact_utc_inclusive_cadences(name, step):
    points=historical_decision_points(T,T+step*2,HISTORICAL_REPLAY_CADENCES[name])
    assert points == (T,T+step,T+step*2)


def test_range_validation_is_explicit():
    with pytest.raises(ValueError,match="end must be"):
        historical_decision_points(T,T-timedelta(seconds=1),timedelta(hours=1))
    with pytest.raises(ValueError,match="maximum is 2"):
        historical_decision_points(T,T+timedelta(hours=2),timedelta(hours=1),maximum=2)
    with pytest.raises(ValueError,match="timezone-aware UTC"):
        historical_decision_points(T.replace(tzinfo=None),T,timedelta(hours=1))
    with pytest.raises(ValueError,match="must be UTC"):
        historical_decision_points(T.astimezone(timezone(timedelta(hours=1))),T,timedelta(hours=1))
    with pytest.raises(ValueError,match="greater than zero"):
        historical_decision_points(T,T,timedelta(0))


class Clock:
    def __init__(self): self.value=0
    def __call__(self): return self.value


class Service:
    def __init__(self, clock, fail=None, cancel=None):
        self.clock=clock; self.calls=[]; self.active=0; self.max_active=0
        self.fail=fail; self.cancel=cancel
    def run_with_progress(self, request, cancelled, progress):
        self.active += 1; self.max_active=max(self.max_active,self.active)
        self.calls.append(request)
        progress(OpportunityScanProgress("discovery",1,message="Historical discovery"))
        self.clock.value += 10
        self.active -= 1
        if request.decision_time == self.fail: raise RuntimeError("offline failure")
        if self.cancel is not None and len(self.calls)==self.cancel: raise OpportunityScanCancelled()
        return request


def test_range_reuses_single_request_path_in_chronological_order_with_parity_and_eta():
    clock=Clock(); service=Service(clock); events=[]
    points=historical_decision_points(T,T+timedelta(hours=2),timedelta(hours=1))
    result=HistoricalRangeRunner(service,monotonic=clock).run(points,native_request,lambda:False,events.append)
    assert [r.decision_time for r in service.calls] == list(points)
    assert service.max_active == 1
    assert result.completed == tuple(service.calls)
    assert service.calls[1] == native_request(points[1])  # manual/range native parity
    complete=[event for event in events if event.stage=="scan_complete"]
    assert [event.completed_scans for event in complete] == [1,2,3]
    assert complete[0].average_scan_seconds == 10
    assert complete[0].eta_seconds == 20
    assert complete[-1].eta_seconds == 0
    assert events[0].eta_seconds is None


def test_failure_and_cancellation_stop_and_retain_completed_results():
    points=historical_decision_points(T,T+timedelta(hours=3),timedelta(hours=1))
    clock=Clock(); service=Service(clock,fail=points[2])
    with pytest.raises(HistoricalReplayFailure) as caught:
        HistoricalRangeRunner(service,monotonic=clock).run(points,native_request,lambda:False)
    assert len(caught.value.completed)==2 and len(service.calls)==3
    clock=Clock(); service=Service(clock,cancel=2)
    with pytest.raises(OpportunityScanCancelled):
        HistoricalRangeRunner(service,monotonic=clock).run(points,native_request,lambda:False)
    assert len(service.calls)==2


def test_two_argument_service_api_remains_compatible(tmp_path):
    calls=[]
    reader=type("Reader",(),{"read":lambda self,path:path})()
    service=OpportunityScannerApplicationService(lambda request,cancelled:(calls.append(request) or tmp_path),reader)
    request=native_request(T)
    assert service.run(request,lambda:False)==tmp_path
    assert calls == [request]


def test_range_falls_back_to_existing_two_argument_service():
    class LegacyService:
        def __init__(self): self.calls=[]
        def run(self, request, cancelled):
            self.calls.append(request)
            return request
    service=LegacyService(); clock=Clock()
    points=(T,T+timedelta(hours=1))
    result=HistoricalRangeRunner(service,monotonic=clock).run(
        points,native_request,lambda:False)
    assert [request.decision_time for request in service.calls] == list(points)
    assert result.completed == tuple(service.calls)


def test_production_stage_events_precede_each_operation_and_scoring_skip(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import crypto_strategy_lab.gui.opportunity_scanner_controller as controller
    from crypto_strategy_lab.data.binance.selective_acquisition import CandleAcquisitionResult

    trace=[]
    def event(progress): trace.append(("progress",progress.stage,progress.message))
    def discovery(*args,**kwargs): trace.append(("operation","historical_discovery")); return "discovery"
    class CandleStage:
        def __init__(self,*args): pass
        def acquire(self,*args,**kwargs):
            trace.append(("operation","candle_acquisition")); return CandleAcquisitionResult(())
    def final(*args): trace.append(("operation","final_candidates")); return "final"
    class RichStage:
        def __init__(self,*args): pass
        def acquire(self,*args,**kwargs): trace.append(("operation","rich_data")); return "rich"
    def publication(*args): trace.append(("operation","publication")); return tmp_path/"run"
    monkeypatch.setattr(controller,"discover_historical_universe",discovery)
    monkeypatch.setattr(controller,"SelectiveCandleAcquirer",CandleStage)
    monkeypatch.setattr(controller,"build_final_candidate_set",final)
    monkeypatch.setattr(controller,"SelectiveRichDataAcquirer",RichStage)
    monkeypatch.setattr(controller,"publish_opportunity_scan",publication)
    store=SimpleNamespace(raw_root=tmp_path,catalog=SimpleNamespace(inventory=lambda *a,**k:[]))
    pipeline=controller.Task1To7OpportunityScanner(store,object(),tmp_path,
        registry=SimpleNamespace(effective_warmup=lambda _:1),now=lambda:T)
    pipeline.run_with_progress(native_request(T),lambda:False,event)
    stages=[item[1] for item in trace if item[0]=="progress"]
    assert stages == ["historical_discovery","candle_acquisition","scoring",
                      "final_candidates","rich_data","publication"]
    assert "skipped" in next(item[2] for item in trace if item[:2]==("progress","scoring")).lower()
    for stage in ("historical_discovery","candle_acquisition","final_candidates","rich_data","publication"):
        assert trace.index(next(item for item in trace if item[:2]==("progress",stage))) < trace.index(("operation",stage))


def test_gui_rows_and_timestamp_precision_match_displayed_seconds():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    widgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from PySide6.QtCore import QDateTime, Qt
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import OpportunityScannerWorkspace
    app=widgets.QApplication.instance() or widgets.QApplication([])
    workspace=OpportunityScannerWorkspace(object())
    def row_visible(field):
        label=workspace._form.labelForField(field)
        return field.isVisible() and (label is None or label.isVisible())
    try:
        workspace.show(); app.processEvents()
        assert not row_visible(workspace.execution)
        assert not row_visible(workspace.decision_time)
        assert not row_visible(workspace.range_start)
        workspace.mode.setCurrentIndex(1); app.processEvents()
        assert row_visible(workspace.execution) and row_visible(workspace.decision_time)
        assert not row_visible(workspace.range_start)
        precise=QDateTime.fromMSecsSinceEpoch(1738800300789,Qt.UTC)
        workspace.decision_time.setDateTime(precise)
        single=workspace.request().decision_time
        assert single.microsecond == 0
        assert single.strftime("%Y-%m-%d %H:%M:%S UTC") == workspace.decision_time.text()
        workspace.execution.setCurrentIndex(1); app.processEvents()
        assert row_visible(workspace.execution) and not row_visible(workspace.decision_time)
        assert all(row_visible(field) for field in
                   (workspace.range_start,workspace.range_end,workspace.cadence,workspace.planned))
        workspace.range_start.setDateTime(precise)
        workspace.range_end.setDateTime(precise.addSecs(7200))
        points=workspace.decision_points()
        assert all(point.microsecond==0 for point in points)
        assert points[0].strftime("%Y-%m-%d %H:%M:%S UTC") == workspace.range_start.text()
        executed=__import__("dataclasses").replace(workspace.request(),decision_time=points[0])
        assert executed.decision_time == points[0]
        assert workspace.planned.text()=="Planned scans: 3"
        workspace.range_progress.show(); workspace.progress_text.setText("ETA: stale")
        workspace.execution.setCurrentIndex(0); app.processEvents()
        assert not workspace.range_progress.isVisible()
        assert "stale" not in workspace.progress_text.text()
    finally:
        workspace.close()


def test_historical_timing_defaults_recommendations_and_exact_replay():
    from crypto_strategy_lab.gui.opportunity_scanner_controller import (
        historical_range_defaults, historical_single_recommendation)
    now=datetime(2026,8,31,14,30,tzinfo=timezone.utc)
    start,end=historical_range_defaults(now)
    assert start==datetime(2026,8,30,0,1,tzinfo=timezone.utc)
    assert end==datetime(2026,8,30,23,59,59,tzinfo=timezone.utc)
    assert historical_single_recommendation(now,"1h")==datetime(2026,8,31,14,1,tzinfo=timezone.utc)
    assert historical_single_recommendation(now,"4h")==datetime(2026,8,31,12,1,tzinfo=timezone.utc)
    assert historical_single_recommendation(now,"1d")==datetime(2026,8,31,0,1,tzinfo=timezone.utc)
    midnight=datetime(2026,8,31,0,0,30,tzinfo=timezone.utc)
    assert historical_single_recommendation(midnight,"1d")==datetime(2026,8,30,0,1,tzinfo=timezone.utc)
    assert historical_single_recommendation(midnight,"1d") <= midnight
    custom=datetime(2026,8,30,6,37,12,tzinfo=timezone.utc)
    for name,step in HISTORICAL_REPLAY_CADENCES.items():
        points=historical_decision_points(custom,custom+step*2,step)
        assert points==(custom,custom+step,custom+step*2),name


def test_manual_single_edit_is_not_overwritten_by_timeframe_change():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    widgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from PySide6.QtCore import QDateTime
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import OpportunityScannerWorkspace
    app=widgets.QApplication.instance() or widgets.QApplication([])
    workspace=OpportunityScannerWorkspace(object())
    try:
        custom=QDateTime(datetime(2025,8,30,6,37,12,tzinfo=timezone.utc))
        workspace.decision_time.setDateTime(custom)
        workspace.timeframe.setCurrentText("4h")
        assert workspace._utc(workspace.decision_time)==datetime(2025,8,30,6,37,12,tzinfo=timezone.utc)
    finally:
        workspace.close()


def test_gui_progress_has_exact_counts_stages_and_time_based_eta():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    widgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from crypto_strategy_lab.gui.opportunity_scanner_workspace import OpportunityScannerWorkspace
    app=widgets.QApplication.instance() or widgets.QApplication([])
    workspace=OpportunityScannerWorkspace(object())
    try:
        workspace.range_progress.setRange(0,3)
        workspace.range_progress.setValue(0); workspace.range_progress.setFormat("0 / 3")
        assert workspace.range_progress.text()=="0 / 3"
        for stage in range(1,7):
            event=OpportunityScanProgress(f"stage_{stage}",stage,6,f"Stage {stage}",T,
                0,3,1,stage,None,None)
            workspace._progress(event)
            assert f"Stage: {stage}/6" in workspace.progress_text.text()
        assert "ETA: calculating" in workspace.progress_text.text()
        first=OpportunityScanProgress("scan_complete",6,6,"Scan complete",T,1,3,1,10,10,20)
        workspace._progress(first)
        assert workspace.range_progress.text()=="1 / 3"
        assert "ETA (estimate): 00:00:20" in workspace.progress_text.text()
        second=OpportunityScanProgress("scan_complete",6,6,"Scan complete",T,2,3,2,20,10,10)
        workspace._progress(second)
        assert workspace.range_progress.text()=="2 / 3"
        assert "ETA (estimate): 00:00:10" in workspace.progress_text.text()
        last=OpportunityScanProgress("scan_complete",6,6,"Scan complete",T,3,3,3,30,10,0)
        workspace._progress(last)
        assert workspace.range_progress.text()=="3 / 3"
        single=OpportunityScanProgress("candle_acquisition",2,6,"Selective candle acquisition",T,0,1,1,5)
        workspace._progress(single)
        assert "Stage: 2/6" in workspace.progress_text.text() and "ETA" not in workspace.progress_text.text()
    finally:
        workspace.close()
