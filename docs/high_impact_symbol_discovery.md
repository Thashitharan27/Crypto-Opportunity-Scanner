# High-impact symbol discovery contract

The universe scanner uses only Binance USD-M Futures `exchangeInfo`, exchange-wide
24-hour ticker, and exchange-wide book-ticker responses. It does not request or
cache candles. Every `exchangeInfo` symbol produces an immutable, timestamped row,
including its eligibility, all rejection reasons, listing age, quote volume, bid,
ask, 24-hour high/low/last prices, midpoint spread percentage, 24-hour range percentage,
price-change percentage, and (for eligible rows) preliminary rank.

Configuration explicitly controls minimum listing age, minimum 24-hour quote
volume, maximum spread percentage, and the stablecoin-like base-asset denylist.
Configured denylist values are normalized once using trimmed uppercase values.
Missing, invalid, or non-finite required public numeric fields reject rather than
silently admit a symbol. Malformed or duplicate exchange-wide rows raise an error
instead of being silently overwritten.

## Preliminary ranking

Eligible symbols are sorted lexicographically by:

1. 24-hour range percentage (`(high - low) / last * 100`), descending;
2. absolute 24-hour price-change percentage, descending;
3. quote volume, descending;
4. midpoint spread percentage (`(ask - bid) / ((ask + bid) / 2) * 100`), ascending;
5. symbol, ascending.

Ranks start at one. The final symbol key guarantees stable ties independent of API
response order. Minimum quote volume is an eligibility floor; once a contract clears
that floor, daily movement is intentionally prioritized over raw liquidity. This is
a cheap acquisition-priority heuristic, not a trading score or entry decision.
Candle history and strategy indicators belong to later stages.
