"""Cheap, exchange-wide Binance USD-M futures universe discovery.

This module deliberately has no dependency on strategy or candle-data code.  The
preliminary order is lexicographic: quote volume (descending), 24h range percent
(descending), absolute 24h change percent (descending), spread percent
(ascending), and symbol (ascending).  The symbol key makes all ties stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.request import urlopen


DEFAULT_STABLECOIN_BASES = frozenset({
    "BUSD", "DAI", "FDUSD", "PAX", "TUSD", "USDC", "USDP", "USDS", "USDT",
})


class DiscoveryClient(Protocol):
    """The only public, exchange-wide resources needed by discovery."""

    def exchange_info(self) -> Mapping[str, Any]: ...
    def tickers_24h(self) -> Sequence[Mapping[str, Any]]: ...
    def book_tickers(self) -> Sequence[Mapping[str, Any]]: ...


class BinanceUsdMDiscoveryClient:
    """Minimal adapter for Binance's public USD-M Futures REST API."""

    def __init__(self, base_url: str = "https://fapi.binance.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> Any:
        with urlopen(f"{self.base_url}{path}", timeout=self.timeout) as response:
            return json.load(response)

    def exchange_info(self) -> Mapping[str, Any]:
        return self._get("/fapi/v1/exchangeInfo")

    def tickers_24h(self) -> Sequence[Mapping[str, Any]]:
        return self._get("/fapi/v1/ticker/24hr")

    def book_tickers(self) -> Sequence[Mapping[str, Any]]:
        return self._get("/fapi/v1/ticker/bookTicker")


@dataclass(frozen=True)
class DiscoveryConfig:
    minimum_listing_age: timedelta = timedelta(days=30)
    minimum_quote_volume: Decimal = Decimal("10000000")
    maximum_spread_percent: Decimal = Decimal("0.20")
    excluded_base_assets: frozenset[str] = field(default_factory=lambda: DEFAULT_STABLECOIN_BASES)

    def __post_init__(self) -> None:
        if self.minimum_listing_age < timedelta(0):
            raise ValueError("minimum_listing_age cannot be negative")
        if self.minimum_quote_volume < 0 or self.maximum_spread_percent < 0:
            raise ValueError("volume and spread thresholds cannot be negative")


@dataclass(frozen=True)
class DiscoveryRow:
    symbol: str
    eligible: bool
    rejection_reasons: tuple[str, ...]
    listing_age: timedelta | None
    quote_volume: Decimal | None
    bid: Decimal | None
    ask: Decimal | None
    spread_percent: Decimal | None
    range_24h_percent: Decimal | None
    price_change_24h_percent: Decimal | None
    preliminary_rank: int | None
    discovery_timestamp: datetime


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def scan_universe(
    client: DiscoveryClient,
    config: DiscoveryConfig = DiscoveryConfig(),
    *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> list[DiscoveryRow]:
    """Return one auditable row per exchange-info symbol, ranked if eligible."""

    timestamp = now()
    if timestamp.tzinfo is None:
        raise ValueError("discovery timestamp must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    ticker_by_symbol = {str(x.get("symbol")): x for x in client.tickers_24h()}
    book_by_symbol = {str(x.get("symbol")): x for x in client.book_tickers()}
    rows: list[DiscoveryRow] = []

    for market in client.exchange_info().get("symbols", []):
        symbol = str(market.get("symbol", ""))
        ticker = ticker_by_symbol.get(symbol, {})
        book = book_by_symbol.get(symbol, {})
        reasons: list[str] = []
        if market.get("contractType") != "PERPETUAL": reasons.append("not_perpetual")
        if market.get("status") != "TRADING": reasons.append("not_trading")
        if market.get("quoteAsset") != "USDT": reasons.append("not_usdt_quote")
        if str(market.get("baseAsset", "")).upper() in config.excluded_base_assets:
            reasons.append("stablecoin_like_base")

        onboard_ms = _decimal(market.get("onboardDate"))
        listing_age = None
        if onboard_ms is None:
            reasons.append("missing_listing_date")
        else:
            listed = datetime.fromtimestamp(float(onboard_ms / 1000), tz=timezone.utc)
            listing_age = timestamp - listed
            if listing_age < config.minimum_listing_age: reasons.append("listing_too_recent")

        volume = _decimal(ticker.get("quoteVolume"))
        if volume is None: reasons.append("missing_quote_volume")
        elif volume < config.minimum_quote_volume: reasons.append("quote_volume_below_minimum")
        bid, ask = _decimal(book.get("bidPrice")), _decimal(book.get("askPrice"))
        spread = None
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            reasons.append("invalid_or_missing_book_ticker")
        else:
            midpoint = (ask + bid) / 2
            spread = (ask - bid) / midpoint * 100
            if spread > config.maximum_spread_percent: reasons.append("spread_above_maximum")

        high, low, last = (_decimal(ticker.get(key)) for key in ("highPrice", "lowPrice", "lastPrice"))
        range_percent = None
        if high is None or low is None or last is None or last <= 0 or high < low:
            reasons.append("invalid_or_missing_24h_prices")
        else:
            range_percent = (high - low) / last * 100
        change = _decimal(ticker.get("priceChangePercent"))
        if change is None: reasons.append("missing_price_change_percent")
        rows.append(DiscoveryRow(symbol, not reasons, tuple(reasons), listing_age, volume, bid, ask,
                                 spread, range_percent, change, None, timestamp))

    eligible = sorted((r for r in rows if r.eligible), key=lambda r: (
        -r.quote_volume, -r.range_24h_percent, -abs(r.price_change_24h_percent),
        r.spread_percent, r.symbol,
    ))
    ranks = {row.symbol: rank for rank, row in enumerate(eligible, 1)}
    ordered = sorted(rows, key=lambda r: (r.symbol not in ranks, ranks.get(r.symbol, 0), r.symbol))
    return [DiscoveryRow(**{**row.__dict__, "preliminary_rank": ranks.get(row.symbol)})
            for row in ordered]
