from datetime import datetime, timedelta, timezone
from decimal import Decimal

from crypto_strategy_lab.data.binance.universe import DiscoveryConfig, scan_universe


NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)


class Client:
    def __init__(self, symbols, tickers, books):
        self.symbols, self.tickers, self.books = symbols, tickers, books
        self.calls = []

    def exchange_info(self): self.calls.append("exchangeInfo"); return {"symbols": self.symbols}
    def tickers_24h(self): self.calls.append("24hr"); return self.tickers
    def book_tickers(self): self.calls.append("bookTicker"); return self.books

    def klines(self, *args):
        raise AssertionError("discovery must never request candle history")


def market(symbol, *, base=None, age=100, **overrides):
    value = {"symbol": symbol, "baseAsset": base or symbol.removesuffix("USDT"),
             "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING",
             "onboardDate": int((NOW - timedelta(days=age)).timestamp() * 1000)}
    value.update(overrides)
    return value


def ticker(symbol, volume="20000000", high="110", low="90", last="100", change="5"):
    return {"symbol": symbol, "quoteVolume": volume, "highPrice": high, "lowPrice": low,
            "lastPrice": last, "priceChangePercent": change}


def book(symbol, bid="99.95", ask="100.05"):
    return {"symbol": symbol, "bidPrice": bid, "askPrice": ask}


def config():
    return DiscoveryConfig(timedelta(days=30), Decimal("10000000"), Decimal("0.2"))


def test_ranking_is_deterministic_and_symbol_breaks_complete_tie():
    symbols = [market("ZZZUSDT"), market("AAAUSDT"), market("MIDUSDT")]
    tickers = [ticker("ZZZUSDT"), ticker("AAAUSDT"), ticker("MIDUSDT", change="8")]
    books = [book(x["symbol"]) for x in symbols]
    first = scan_universe(Client(symbols, tickers, books), config(), now=lambda: NOW)
    second = scan_universe(Client(list(reversed(symbols)), list(reversed(tickers)), list(reversed(books))),
                           config(), now=lambda: NOW)
    assert [(x.symbol, x.preliminary_rank) for x in first] == [
        ("MIDUSDT", 1), ("AAAUSDT", 2), ("ZZZUSDT", 3)]
    assert [(x.symbol, x.preliminary_rank) for x in second] == [
        ("MIDUSDT", 1), ("AAAUSDT", 2), ("ZZZUSDT", 3)]


def test_all_eligibility_filters_and_reasons_are_auditable_without_klines():
    symbols = [
        market("GOODUSDT"),
        market("USDCUSDT", base="USDC"),
        market("NEWUSDT", age=2),
        market("OLDUSDT", contractType="CURRENT_QUARTER", status="SETTLING", quoteAsset="BUSD"),
        market("LOWUSDT"), market("WIDEUSDT"),
    ]
    tickers = [ticker(x["symbol"], volume="1" if x["symbol"] == "LOWUSDT" else "20000000")
               for x in symbols]
    books = [book(x["symbol"], bid="99", ask="101") if x["symbol"] == "WIDEUSDT"
             else book(x["symbol"]) for x in symbols]
    client = Client(symbols, tickers, books)
    rows = scan_universe(client, config(), now=lambda: NOW)
    by_symbol = {x.symbol: x for x in rows}

    assert by_symbol["GOODUSDT"].eligible
    assert by_symbol["GOODUSDT"].listing_age == timedelta(days=100)
    assert by_symbol["USDCUSDT"].rejection_reasons == ("stablecoin_like_base",)
    assert by_symbol["NEWUSDT"].rejection_reasons == ("listing_too_recent",)
    assert by_symbol["LOWUSDT"].rejection_reasons == ("quote_volume_below_minimum",)
    assert by_symbol["WIDEUSDT"].rejection_reasons == ("spread_above_maximum",)
    assert by_symbol["OLDUSDT"].rejection_reasons[:3] == (
        "not_perpetual", "not_trading", "not_usdt_quote")
    assert all(row.preliminary_rank is None for row in rows if not row.eligible)
    assert client.calls == ["24hr", "bookTicker", "exchangeInfo"]
