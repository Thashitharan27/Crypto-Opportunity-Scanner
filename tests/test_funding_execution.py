from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features.funding import FundingContextFeatureProvider
from crypto_strategy_lab.funding_execution import (
    FundingAwareRuleBacktestEngine,
    _extract_prepared_funding_events,
    _funding_block,
)
from crypto_strategy_lab.trade import Position, Side


def _daily_funding_frame() -> pd.DataFrame:
    starts = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1D")
    klines = pd.DataFrame(
        {
            "period_start": starts,
            "available_at": starts + pd.Timedelta(days=1),
        }
    )
    funding = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T08:00:00Z",
                    "2026-01-01T16:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T08:00:00Z",
                    "2026-01-02T16:00:00Z",
                ],
                utc=True,
            ),
            "funding_rate": [0.0001, 0.0002, -0.0001, 0.0003, 0.0004, 0.0005],
            "funding_interval_hours": 8.0,
        }
    )
    request = DataRequest(
        symbol="BTCUSDT",
        start=starts[0].to_pydatetime(),
        end=(starts[-1] + pd.Timedelta(days=1)).to_pydatetime(),
        strategy_interval="1d",
        datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE),
    )
    return FundingContextFeatureProvider().compute(
        request,
        {DatasetKind.KLINES: klines, DatasetKind.FUNDING_RATE: funding},
        {},
    )


def _prepared_from_funding_frame(frame: pd.DataFrame):
    values = {
        column: frame[column].to_numpy()
        for column in frame.columns
        if column not in {"timestamp", "available_at"}
    }
    block = SimpleNamespace(name="funding_context", values=values)
    return SimpleNamespace(
        timestamp=pd.to_datetime(frame["timestamp"], utc=True).to_numpy(dtype="datetime64[ns]"),
        decision_available_at=pd.to_datetime(frame["available_at"], utc=True).to_numpy(
            dtype="datetime64[ns]"
        ),
        research=(block,),
    )


def _closed_position(side: Side, *, quantity: float = 2.0) -> Position:
    pos = Position(
        side=side,
        entry_time=pd.Timestamp("2026-01-01T00:01:00Z"),
        entry_index=0,
        entry_price=100.0,
        risk=10.0,
        sl=90.0 if side == Side.LONG else 110.0,
        tp=120.0 if side == Side.LONG else 80.0,
        quantity=quantity,
        risk_amount=20.0,
        entry_notional=100.0 * quantity,
        atr_at_entry=5.0,
    )
    pos.exit_time = pd.Timestamp("2026-01-01T23:59:00Z")
    pos.net_pnl = 10.0
    pos.net_r = 0.5
    return pos


def _engine_with_events(rates) -> FundingAwareRuleBacktestEngine:
    engine = FundingAwareRuleBacktestEngine.__new__(FundingAwareRuleBacktestEngine)
    engine._funding_event_times = np.asarray(
        ["2026-01-01T08:00:00", "2026-01-01T16:00:00"],
        dtype="datetime64[ns]",
    )
    engine._funding_event_rates = np.asarray(rates, dtype=float)
    engine.intrabar_data = None
    engine.times = np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
    engine.open = np.asarray([100.0], dtype=float)
    return engine


def test_daily_context_preserves_every_settlement_inside_candle() -> None:
    result = _daily_funding_frame()

    first = json.loads(result.loc[0, "funding_settlements_json"])
    second = json.loads(result.loc[1, "funding_settlements_json"])

    # Lower-bound events are excluded, upper-bound events are included. This
    # yields every settlement exactly once even on 1D strategy candles.
    assert [pd.Timestamp(item[0], unit="ns", tz="UTC") for item in first] == [
        pd.Timestamp("2026-01-01T08:00:00Z"),
        pd.Timestamp("2026-01-01T16:00:00Z"),
        pd.Timestamp("2026-01-02T00:00:00Z"),
    ]
    assert [pd.Timestamp(item[0], unit="ns", tz="UTC") for item in second] == [
        pd.Timestamp("2026-01-02T08:00:00Z"),
        pd.Timestamp("2026-01-02T16:00:00Z"),
    ]


def test_prepared_contract_is_authoritative_settlement_transport() -> None:
    frame = _daily_funding_frame()
    prepared = _prepared_from_funding_frame(frame)
    engine = SimpleNamespace(research_features={})

    block = _funding_block(prepared, engine)
    times, rates = _extract_prepared_funding_events(prepared, block)

    assert block is prepared.research[0]
    assert len(times) == 5
    assert len(rates) == 5
    assert pd.Timestamp(times[0], tz="UTC") == pd.Timestamp("2026-01-01T08:00:00Z")
    assert pd.Timestamp(times[-1], tz="UTC") == pd.Timestamp("2026-01-02T16:00:00Z")


def test_missing_settlement_transport_cannot_silently_become_zero_funding() -> None:
    frame = _daily_funding_frame().drop(columns=["funding_settlements_json"])
    prepared = _prepared_from_funding_frame(frame)

    with pytest.raises(ValueError, match="no funding_settlements_json transport"):
        _extract_prepared_funding_events(prepared, prepared.research[0])


def test_empty_settlement_transport_cannot_silently_become_zero_funding() -> None:
    frame = _daily_funding_frame()
    frame["funding_settlements_json"] = "[]"
    prepared = _prepared_from_funding_frame(frame)

    with pytest.raises(ValueError, match="prepared settlement timeline is empty"):
        _extract_prepared_funding_events(prepared, prepared.research[0])


def test_absent_funding_context_is_allowed_to_have_no_settlements() -> None:
    prepared = SimpleNamespace(
        timestamp=np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
        decision_available_at=np.asarray(["2026-01-02T00:00:00"], dtype="datetime64[ns]"),
        research=(),
    )

    times, rates = _extract_prepared_funding_events(prepared, None)

    assert len(times) == 0
    assert len(rates) == 0


def test_positive_funding_is_paid_by_long_and_received_by_short() -> None:
    long_engine = _engine_with_events([0.001, 0.001])
    long = _closed_position(Side.LONG)
    long_engine._apply_funding_cashflow(long)

    assert long.funding_event_count == 2
    assert long.funding_paid == pytest.approx(0.4)
    assert long.funding_received == pytest.approx(0.0)
    assert long.funding_net_pnl == pytest.approx(-0.4)
    assert long.net_pnl == pytest.approx(9.6)
    assert long.net_r == pytest.approx(0.48)

    short_engine = _engine_with_events([0.001, 0.001])
    short = _closed_position(Side.SHORT)
    short_engine._apply_funding_cashflow(short)

    assert short.funding_paid == pytest.approx(0.0)
    assert short.funding_received == pytest.approx(0.4)
    assert short.funding_net_pnl == pytest.approx(0.4)
    assert short.net_pnl == pytest.approx(10.4)


def test_negative_funding_reverses_payer_and_receiver() -> None:
    long_engine = _engine_with_events([-0.001, -0.001])
    long = _closed_position(Side.LONG)
    long_engine._apply_funding_cashflow(long)
    assert long.funding_received == pytest.approx(0.4)
    assert long.funding_net_pnl == pytest.approx(0.4)

    short_engine = _engine_with_events([-0.001, -0.001])
    short = _closed_position(Side.SHORT)
    short_engine._apply_funding_cashflow(short)
    assert short.funding_paid == pytest.approx(0.4)
    assert short.funding_net_pnl == pytest.approx(-0.4)


def test_partial_exit_reduces_quantity_for_later_funding_events() -> None:
    engine = _engine_with_events([0.001, 0.001])
    pos = _closed_position(Side.LONG)
    pos.original_quantity = 2.0
    pos.tp1_hit = True
    pos.tp1_quantity = 1.0
    pos.tp1_exit_time = pd.Timestamp("2026-01-01T12:00:00Z")

    engine._apply_funding_cashflow(pos)

    # 08:00 charges 2 units; after the 12:00 partial exit, 16:00 charges 1.
    assert pos.funding_event_count == 2
    assert pos.funding_paid == pytest.approx(0.3)
    assert pos.funding_net_pnl == pytest.approx(-0.3)


def test_exact_entry_and_exit_boundary_settlements_are_not_charged() -> None:
    engine = _engine_with_events([0.001, 0.001])
    pos = _closed_position(Side.LONG)
    pos.entry_time = pd.Timestamp("2026-01-01T08:00:00Z")
    pos.exit_time = pd.Timestamp("2026-01-01T16:00:00Z")

    engine._apply_funding_cashflow(pos)

    assert pos.funding_event_count == 0
    assert pos.funding_net_pnl == pytest.approx(0.0)
