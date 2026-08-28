"""Causal, completed-day symbol discovery backed only by the Data Lake."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable, Protocol

import pandas as pd

from ..query import DataRequest
from ..schemas import DatasetKind, MarketKind
from ..source_identity import SourceSignature
from ..store import DataNotAvailableError


@dataclass(frozen=True, slots=True)
class DiscoveryDecisionTime:
    """Exact, immutable causal cutoff, normalized to UTC."""

    value: datetime

    def __post_init__(self) -> None:
        if self.value.tzinfo is None or self.value.utcoffset() is None:
            raise ValueError("historical discovery decision time must be timezone-aware")
        object.__setattr__(self, "value", self.value.astimezone(timezone.utc))


@dataclass(frozen=True, slots=True)
class HistoricalDiscoveryConfig:
    """Task-2 completed-day eligibility and query-window configuration.

    Listing age is deliberately not enforced: the Data Lake currently has no
    immutable official listing-date dataset.  First-candle time is not a proxy.
    """

    minimum_quote_volume: Decimal = Decimal("10000000")
    listing_age_enforcement: bool = False

    def __post_init__(self) -> None:
        minimum = _decimal(self.minimum_quote_volume)
        if minimum is None or minimum < 0:
            raise ValueError("minimum_quote_volume must be finite and non-negative")
        if self.listing_age_enforcement:
            raise ValueError("historical listing-age metadata is unavailable")
        object.__setattr__(self, "minimum_quote_volume", minimum)


@dataclass(frozen=True, slots=True)
class HistoricalSnapshotRow:
    symbol: str
    period_start: datetime | None
    period_end: datetime | None
    available_at: datetime | None
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal | None
    quote_volume: Decimal | None
    range_percent: Decimal | None
    price_change_percent: Decimal | None
    absolute_price_change_percent: Decimal | None
    eligible: bool
    rejection_reasons: tuple[str, ...]
    rank: int | None
    source_identity: str | None


@dataclass(frozen=True, slots=True)
class HistoricalUniverseSnapshot:
    decision_time: DiscoveryDecisionTime
    rows: tuple[HistoricalSnapshotRow, ...]


@dataclass(frozen=True, slots=True)
class HistoricalSource:
    symbol: str
    request: DataRequest
    signature: SourceSignature | None


@dataclass(frozen=True, slots=True)
class HistoricalDiscoveryResult:
    decision_time: DiscoveryDecisionTime
    snapshot: HistoricalUniverseSnapshot
    eligible_candidates: tuple[HistoricalSnapshotRow, ...]
    sources: tuple[HistoricalSource, ...]
    config: HistoricalDiscoveryConfig
    contract: str = "binance_1d_completed_day_v1"


class HistoricalMarketDataStore(Protocol):
    def load_dataset(
        self, request: DataRequest, dataset: DatasetKind, *,
        interval: str | None = None, available_at_cutoff: datetime | None = None,
    ) -> pd.DataFrame: ...
    def source_signature(self, request: DataRequest, dataset: DatasetKind, *, interval: str | None = None) -> SourceSignature: ...


def _decimal(value: object) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _utc(value: object) -> datetime | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _missing(symbol: str, reason: str, identity: str | None = None) -> HistoricalSnapshotRow:
    return HistoricalSnapshotRow(symbol, None, None, None, None, None, None, None,
                                 None, None, None, None, False, (reason,), None, identity)


def discover_historical_universe(
    store: HistoricalMarketDataStore,
    symbols: Iterable[str],
    decision_time: DiscoveryDecisionTime | datetime,
    config: HistoricalDiscoveryConfig = HistoricalDiscoveryConfig(),
    *,
    market: MarketKind = MarketKind.FUTURES_UM,
) -> HistoricalDiscoveryResult:
    """Rank the latest causally available completed Binance 1D kline per symbol.

    The explicitly supplied symbols define the historical universe.  This path
    has no live-discovery client and cannot access Binance REST endpoints.
    """

    decision = decision_time if isinstance(decision_time, DiscoveryDecisionTime) else DiscoveryDecisionTime(decision_time)
    normalized = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    rows: list[HistoricalSnapshotRow] = []
    sources: list[HistoricalSource] = []
    for symbol in normalized:
        request = DataRequest(symbol=symbol, start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                              end=decision.value, strategy_interval="1d",
                              datasets=(DatasetKind.KLINES,), market=market)
        try:
            signature = store.source_signature(request, DatasetKind.KLINES, interval="1d")
            identity = signature.cache_identity()
            frame = store.load_dataset(
                request, DatasetKind.KLINES, interval="1d",
                available_at_cutoff=decision.value,
            )
        except DataNotAvailableError:
            sources.append(HistoricalSource(symbol, request, None))
            rows.append(_missing(symbol, "no_source_coverage"))
            continue
        sources.append(HistoricalSource(symbol, request, signature))
        required = {"period_start", "period_end", "available_at", "open", "high", "low", "close", "quote_volume"}
        if not required.issubset(frame.columns):
            rows.append(_missing(symbol, "invalid_kline_schema", identity))
            continue
        candidates = frame.copy()
        candidates["_period_start"] = pd.to_datetime(candidates["period_start"], utc=True, errors="coerce")
        candidates["_period_end"] = pd.to_datetime(candidates["period_end"], utc=True, errors="coerce")
        candidates["_available_at"] = pd.to_datetime(candidates["available_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp(decision.value)
        candidates = candidates.loc[
            candidates["_period_start"].notna()
            & candidates["_period_end"].notna()
            & candidates["_available_at"].notna()
            & (candidates["_period_end"] <= cutoff)
            & (candidates["_available_at"] <= cutoff)
        ]
        if candidates.empty:
            rows.append(_missing(symbol, "no_completed_candle_available", identity))
            continue
        # "Latest completed candle" is defined by market period, not by when a
        # delayed archive/correction happened to become available.
        selected = candidates.sort_values(
            ["_period_end", "_period_start", "_available_at"], kind="stable"
        ).iloc[-1]
        values = {name: _decimal(selected[name]) for name in ("open", "high", "low", "close", "quote_volume")}
        reasons: list[str] = []
        if any(value is None for value in values.values()):
            reasons.append("invalid_or_missing_numeric_value")
        range_percent = change = absolute_change = None
        if not reasons:
            opening, high, low, close, volume = (values[name] for name in ("open", "high", "low", "close", "quote_volume"))
            if opening <= 0 or close <= 0 or high < low or volume < 0:
                reasons.append("invalid_kline_values")
            else:
                range_percent = (high - low) / close * 100
                change = (close - opening) / opening * 100
                absolute_change = abs(change)
                if volume < config.minimum_quote_volume:
                    reasons.append("quote_volume_below_minimum")
        rows.append(HistoricalSnapshotRow(
            symbol, _utc(selected["_period_start"]), _utc(selected["_period_end"]),
            _utc(selected["_available_at"]), values["open"], values["high"], values["low"],
            values["close"], values["quote_volume"], range_percent, change, absolute_change,
            not reasons, tuple(reasons), None, identity,
        ))

    ranked = sorted((row for row in rows if row.eligible), key=lambda row: (
        -row.range_percent, -row.absolute_price_change_percent, -row.quote_volume, row.symbol
    ))
    ranks = {row.symbol: index for index, row in enumerate(ranked, 1)}
    final_rows = tuple(replace(row, rank=ranks.get(row.symbol)) for row in sorted(rows, key=lambda row: row.symbol))
    eligible = tuple(sorted((row for row in final_rows if row.rank is not None), key=lambda row: row.rank))
    snapshot = HistoricalUniverseSnapshot(decision, final_rows)
    return HistoricalDiscoveryResult(decision, snapshot, eligible, tuple(sources), config)
