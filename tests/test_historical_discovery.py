from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import random

import pandas as pd
import pytest

from crypto_strategy_lab.data.binance.historical_discovery import (
    DiscoveryDecisionTime,
    HistoricalDiscoveryConfig,
    discover_historical_universe,
)
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.data.store import DataNotAvailableError


T = datetime(2025, 1, 10, tzinfo=timezone.utc)


def candle(
    symbol,
    start,
    *,
    available_at=None,
    open="100",
    high="120",
    low="90",
    close="110",
    volume="20000000",
):
    end = start + timedelta(days=1)
    return {
        "symbol": symbol,
        "period_start": start,
        "period_end": end,
        "available_at": available_at or end,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "quote_volume": volume,
    }


class Store:
    def __init__(self, rows, partitions=None):
        self.rows = rows
        self.partitions = partitions or {}
        self.calls = []

    def source_signature(self, request, dataset, *, interval=None):
        self.calls.append(("source_signature", request.symbol, dataset, interval))
        return SourceSignature.from_canonical_identities(
            DatasetKind.KLINES,
            self.partitions.get(request.symbol, [f"partition:{request.symbol}"]),
        )

    def load_dataset(
        self, request, dataset, *, interval=None, available_at_cutoff=None
    ):
        self.calls.append(("load_dataset", request.symbol, dataset, interval))
        values = [
            row
            for row in self.rows.get(request.symbol, [])
            if available_at_cutoff is None
            or pd.Timestamp(row["available_at"])
            <= pd.Timestamp(available_at_cutoff)
        ]
        return pd.DataFrame(
            values,
            columns=[
                "symbol",
                "period_start",
                "period_end",
                "available_at",
                "open",
                "high",
                "low",
                "close",
                "quote_volume",
            ],
        )


def ranks(result):
    return [(row.symbol, row.rank) for row in result.eligible_candidates]


def test_future_candles_cannot_change_selection_or_ranks():
    prior = T - timedelta(days=1)
    rows = {
        "AAAUSDT": [candle("AAAUSDT", prior - timedelta(days=1))],
        "BBBUSDT": [
            candle("BBBUSDT", prior - timedelta(days=1), high="115")
        ],
    }
    before = discover_historical_universe(Store(rows), rows, T)
    rows["AAAUSDT"].append(
        candle(
            "AAAUSDT",
            prior,
            available_at=T + timedelta(seconds=1),
            high="9999",
        )
    )
    rows["BBBUSDT"].append(candle("BBBUSDT", T, high="9999"))
    after = discover_historical_universe(Store(rows), reversed(rows), T)
    assert ranks(before) == ranks(after)
    assert [
        (r.symbol, r.period_start, r.range_percent) for r in before.snapshot.rows
    ] == [
        (r.symbol, r.period_start, r.range_percent) for r in after.snapshot.rows
    ]


def test_available_at_boundary_is_allowed_and_later_is_not():
    boundary = candle(
        "AAAUSDT", T - timedelta(days=1), available_at=T, high="130"
    )
    later = candle(
        "AAAUSDT",
        T - timedelta(days=2),
        available_at=T + timedelta(microseconds=1),
        high="999",
    )
    result = discover_historical_universe(
        Store({"AAAUSDT": [later, boundary]}), ["AAAUSDT"], T
    )
    assert result.snapshot.rows[0].period_start == T - timedelta(days=1)
    assert result.snapshot.rows[0].available_at == T


def test_latest_completed_available_observation_wins():
    values = [
        candle("AAAUSDT", T - timedelta(days=3), high="999"),
        candle("AAAUSDT", T - timedelta(days=2), high="125"),
    ]
    result = discover_historical_universe(
        Store({"AAAUSDT": values}), ["AAAUSDT"], T
    )
    assert result.snapshot.rows[0].period_start == T - timedelta(days=2)
    assert result.snapshot.rows[0].high == Decimal("125")


def test_latest_market_period_wins_over_later_delivery_time():
    older = candle(
        "AAAUSDT", T - timedelta(days=3), available_at=T, high="999"
    )
    newer = candle(
        "AAAUSDT",
        T - timedelta(days=2),
        available_at=T - timedelta(hours=12),
        high="125",
    )
    result = discover_historical_universe(
        Store({"AAAUSDT": [older, newer]}), ["AAAUSDT"], T
    )
    assert result.snapshot.rows[0].period_start == T - timedelta(days=2)
    assert result.snapshot.rows[0].high == Decimal("125")


def test_deterministic_under_symbol_and_row_shuffle():
    symbols = ["CCCUSDT", "AAAUSDT", "BBBUSDT"]
    rows = {
        symbol: [
            candle(symbol, T - timedelta(days=3)),
            candle(symbol, T - timedelta(days=2)),
        ]
        for symbol in symbols
    }
    expected = ranks(discover_historical_universe(Store(rows), symbols, T))
    random.Random(7).shuffle(symbols)
    for values in rows.values():
        random.Random(9).shuffle(values)
    assert ranks(discover_historical_universe(Store(rows), symbols, T)) == expected
    assert expected == [("AAAUSDT", 1), ("BBBUSDT", 2), ("CCCUSDT", 3)]


def test_utc_equivalence_and_naive_rejection():
    rows = {"AAAUSDT": [candle("AAAUSDT", T - timedelta(days=2))]}
    offset = timezone(timedelta(hours=5, minutes=30))
    equivalent = T.astimezone(offset)
    assert discover_historical_universe(
        Store(rows), rows, T
    ) == discover_historical_universe(Store(rows), rows, equivalent)
    with pytest.raises(ValueError, match="timezone-aware"):
        DiscoveryDecisionTime(datetime(2025, 1, 10))


def test_source_signature_is_canonical_and_partition_sensitive():
    rows = {"AAAUSDT": [candle("AAAUSDT", T - timedelta(days=2))]}
    first = discover_historical_universe(
        Store(rows, {"AAAUSDT": ["p1"]}), rows, T
    )
    repeated = discover_historical_universe(
        Store(rows, {"AAAUSDT": ["p1"]}), rows, T
    )
    changed = discover_historical_universe(
        Store(rows, {"AAAUSDT": ["p2"]}), rows, T
    )
    assert first.sources == repeated.sources
    assert (
        first.sources[0].signature.cache_identity()
        != changed.sources[0].signature.cache_identity()
    )
    assert (
        first.snapshot.rows[0].source_identity
        != changed.snapshot.rows[0].source_identity
    )


def test_missing_invalid_and_low_volume_are_auditable_and_fail_closed():
    invalid = candle("BADUSDT", T - timedelta(days=2), close="NaN")
    low = candle("LOWUSDT", T - timedelta(days=2), volume="1")
    result = discover_historical_universe(
        Store(
            {
                "BADUSDT": [invalid],
                "LOWUSDT": [low],
                "NONEUSDT": [],
            }
        ),
        ["NONEUSDT", "LOWUSDT", "BADUSDT"],
        T,
    )
    states = {
        row.symbol: row.rejection_reasons for row in result.snapshot.rows
    }
    assert states == {
        "BADUSDT": ("invalid_or_missing_numeric_value",),
        "LOWUSDT": ("quote_volume_below_minimum",),
        "NONEUSDT": ("no_completed_candle_available",),
    }
    assert result.eligible_candidates == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"high": "0"},
        {"low": "0"},
        {"open": "130", "high": "120"},
        {"close": "80", "low": "90"},
    ],
)
def test_invalid_ohlc_values_fail_closed(overrides):
    row = candle("BADUSDT", T - timedelta(days=2), **overrides)
    result = discover_historical_universe(
        Store({"BADUSDT": [row]}), ["BADUSDT"], T
    )
    snapshot = result.snapshot.rows[0]
    assert not snapshot.eligible
    assert snapshot.rejection_reasons == ("invalid_kline_values",)
    assert result.eligible_candidates == ()


def test_source_validation_failure_is_isolated_to_one_symbol():
    class InvalidSourceStore(Store):
        def load_dataset(
            self, request, dataset, *, interval=None, available_at_cutoff=None
        ):
            if request.symbol == "BADUSDT":
                raise ValueError("invalid canonical archive")
            return super().load_dataset(
                request,
                dataset,
                interval=interval,
                available_at_cutoff=available_at_cutoff,
            )

    rows = {"GOODUSDT": [candle("GOODUSDT", T - timedelta(days=2))]}
    result = discover_historical_universe(
        InvalidSourceStore(rows), ["BADUSDT", "GOODUSDT"], T
    )
    states = {
        row.symbol: row.rejection_reasons for row in result.snapshot.rows
    }
    assert states["BADUSDT"] == ("source_validation_failed",)
    assert ranks(result) == [("GOODUSDT", 1)]


def test_data_not_available_is_auditable_no_source_coverage():
    class NoCoverageStore(Store):
        def source_signature(self, request, dataset, *, interval=None):
            raise DataNotAvailableError("no coverage")

    result = discover_historical_universe(
        NoCoverageStore({}), ["NONEUSDT"], T
    )
    row = result.snapshot.rows[0]
    assert not row.eligible
    assert row.rejection_reasons == ("no_source_coverage",)
    assert result.sources[0].signature is None


def test_historical_path_uses_only_data_lake_contract_not_live_client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("live discovery endpoint was accessed")

    monkeypatch.setattr(
        "crypto_strategy_lab.data.binance.universe.BinanceUsdMDiscoveryClient._get",
        forbidden,
    )
    store = Store(
        {"AAAUSDT": [candle("AAAUSDT", T - timedelta(days=2))]}
    )
    result = discover_historical_universe(
        store,
        ["AAAUSDT"],
        T,
        HistoricalDiscoveryConfig(minimum_quote_volume=Decimal("0")),
    )
    assert ranks(result) == [("AAAUSDT", 1)]
    assert {call[0] for call in store.calls} == {
        "source_signature",
        "load_dataset",
    }
