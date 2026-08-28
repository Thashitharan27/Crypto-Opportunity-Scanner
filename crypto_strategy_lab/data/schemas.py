"""Typed identities shared by data-lake discovery, cataloging and loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class MarketKind(str, Enum):
    """Exchange market namespaces supported by the data layer."""

    FUTURES_UM = "futures_um"
    FUTURES_CM = "futures_cm"
    SPOT = "spot"


class DatasetKind(str, Enum):
    """Canonical Binance dataset families."""

    KLINES = "klines"
    FUTURES_METRICS = "metrics"
    FUNDING_RATE = "funding_rate"
    MARK_PRICE_KLINES = "mark_price_klines"
    INDEX_PRICE_KLINES = "index_price_klines"
    PREMIUM_INDEX_KLINES = "premium_index_klines"
    AGG_TRADES = "agg_trades"
    TRADES = "trades"
    BOOK_DEPTH = "book_depth"
    BOOK_TICKER = "book_ticker"


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """Metadata for one raw Binance archive or CSV file."""

    raw_root: Path
    path: Path
    market: MarketKind
    dataset: DatasetKind
    symbol: str
    interval: str | None
    frequency: str
    period_start: datetime | None
    period_end: datetime | None
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    exchange: str = "binance"


@dataclass(frozen=True, slots=True)
class Coverage:
    """Catalog coverage summary for a symbol/dataset/interval."""

    first_period: datetime | None
    last_period: datetime | None
    archive_count: int
