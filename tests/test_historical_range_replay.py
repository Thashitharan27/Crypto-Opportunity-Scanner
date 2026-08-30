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
