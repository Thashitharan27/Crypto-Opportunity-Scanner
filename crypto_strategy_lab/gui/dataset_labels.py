"""Human-readable dataset labels for catalog and Data Library presentation."""
from __future__ import annotations


_DATASET_FAMILY_LABELS = {
    "klines": "Market Price Candles",
    "mark_price_klines": "Mark Price Candles",
    "index_price_klines": "Index Price Candles",
    "premium_index_klines": "Premium Index Candles",
    "funding_rate": "Funding",
    "metrics": "Futures Positioning",
    "agg_trades": "Trades",
    "trades": "Trades",
    "book_depth": "Order Book",
    "book_ticker": "Order Book",
}


def dataset_family_label(dataset) -> str:
    """Return a stable friendly label without collapsing distinct candle sources."""
    raw = getattr(dataset, "value", dataset)
    value = str(raw).lower()
    if value in _DATASET_FAMILY_LABELS:
        return _DATASET_FAMILY_LABELS[value]
    if "fund" in value:
        return "Funding"
    if "metric" in value or "interest" in value or "ratio" in value:
        return "Futures Positioning"
    if "agg" in value or value == "trades":
        return "Trades"
    if "book" in value or "depth" in value:
        return "Order Book"
    return str(raw).replace("_", " ").title()
