from datetime import datetime, timezone

import pytest

from crypto_strategy_lab.data.timing import floor_fixed_candle_grid


@pytest.mark.parametrize(("interval","expected"),[
    ("1s","2025-08-30T06:37:12+00:00"),
    ("1m","2025-08-30T06:37:00+00:00"),
    ("3m","2025-08-30T06:36:00+00:00"),
    ("5m","2025-08-30T06:35:00+00:00"),
    ("15m","2025-08-30T06:30:00+00:00"),
    ("30m","2025-08-30T06:30:00+00:00"),
    ("1h","2025-08-30T06:00:00+00:00"),
    ("2h","2025-08-30T06:00:00+00:00"),
    ("120m","2025-08-30T06:00:00+00:00"),
    ("4h","2025-08-30T04:00:00+00:00"),
    ("6h","2025-08-30T06:00:00+00:00"),
    ("8h","2025-08-30T00:00:00+00:00"),
    ("12h","2025-08-30T00:00:00+00:00"),
    ("1d","2025-08-30T00:00:00+00:00"),
    ("3d","2025-08-28T00:00:00+00:00"),
    ("1w","2025-08-25T00:00:00+00:00"),
])
def test_authoritative_fixed_grid_flooring(interval,expected):
    value=datetime(2025,8,30,6,37,12,tzinfo=timezone.utc)
    assert floor_fixed_candle_grid(value,interval).isoformat()==expected


def test_daily_floor_accepts_off_grid_decision_without_changing_it():
    decision=datetime(2025,8,30,1,1,1,tzinfo=timezone.utc)
    assert floor_fixed_candle_grid(decision,"1d")==datetime(2025,8,30,tzinfo=timezone.utc)
    assert decision==datetime(2025,8,30,1,1,1,tzinfo=timezone.utc)


@pytest.mark.parametrize("interval",["7m","1M","", "nonsense"])
def test_non_native_or_non_fixed_grids_are_rejected(interval):
    with pytest.raises(ValueError,match="Unsupported|must not be empty"):
        floor_fixed_candle_grid(datetime(2025,8,30,tzinfo=timezone.utc),interval)
