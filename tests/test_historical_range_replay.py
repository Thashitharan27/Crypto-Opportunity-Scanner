"""Historical range replay boundary tests."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    HistoricalRangeRunner, build_request, historical_decision_points,
)


def _request(decision):
    return build_request(mode="HISTORICAL", decision_time=decision,
        minimum_listing_age_days=1, minimum_quote_volume=Decimal(0),
        maximum_spread_percent=Decimal(1), preliminary_size=1, final_size=1,
        strategy_interval="1h", model=None, enabled_features=())


def test_decision_points_are_inclusive_sequential_and_do_not_round():
    start=datetime(2026,1,1,0,0,0,437000,tzinfo=timezone.utc)
    points=historical_decision_points(start,start+timedelta(hours=2),timedelta(hours=1))
    assert points == tuple(start+timedelta(hours=i) for i in range(3))
    assert all(point.microsecond == 437000 for point in points)


def test_range_runner_falls_back_to_task10_run_and_reuses_single_requests():
    calls=[]
    class Service:
        def run(self,request,cancelled):
            calls.append(request); return request.decision_time
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    points=historical_decision_points(start,start+timedelta(hours=2),timedelta(hours=1))
    progress=[]
    result=HistoricalRangeRunner(Service()).run(_request(start),points,lambda:False,
                                                 lambda *event:progress.append(event))
    assert result == points
    assert tuple(call.decision_time for call in calls) == points
    assert [event[1] for event in progress] == [1,2,3]
