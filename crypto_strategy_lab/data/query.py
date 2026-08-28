"""Filename-free requests for market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json

from .schemas import DatasetKind, MarketKind
from .timing import ensure_utc, normalize_binance_interval


@dataclass(frozen=True, slots=True)
class DataRequest:
    """A reproducible request for a slice of the market-data lake.

    `start` is inclusive and `end` is exclusive. Filenames are deliberately not
    part of the request contract. Fixed kline intervals are normalized to the
    equivalent Binance-native archive name (for example 240m becomes 4h).
    """

    symbol: str
    start: datetime
    end: datetime
    strategy_interval: str
    intrabar_interval: str | None = None
    datasets: tuple[DatasetKind, ...] = (DatasetKind.KLINES,)
    market: MarketKind = MarketKind.FUTURES_UM
    exchange: str = "binance"

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")
        start = ensure_utc(self.start)
        end = ensure_utc(self.end)
        if start >= end:
            raise ValueError("DataRequest start must be before end")
        if not self.strategy_interval.strip():
            raise ValueError("strategy_interval must not be empty")
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "strategy_interval", normalize_binance_interval(self.strategy_interval))
        if self.intrabar_interval is not None:
            interval = self.intrabar_interval.strip()
            object.__setattr__(
                self,
                "intrabar_interval",
                normalize_binance_interval(interval) if interval else None,
            )
        object.__setattr__(self, "datasets", tuple(dict.fromkeys(self.datasets)))

    def cache_key(self) -> str:
        """Stable identity used by future prepared-frame/feature caches."""

        payload = {
            "exchange": self.exchange,
            "market": self.market.value,
            "symbol": self.symbol,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "strategy_interval": self.strategy_interval,
            "intrabar_interval": self.intrabar_interval,
            "datasets": sorted(item.value for item in self.datasets),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(raw).hexdigest()

    def feature_scope_key(self) -> str:
        """Identity of the strategy slice, excluding execution-only request fields."""
        payload = {"exchange": self.exchange, "market": self.market.value,
                   "symbol": self.symbol, "start": self.start.isoformat(),
                   "end": self.end.isoformat(),
                   "strategy_interval": self.strategy_interval}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
