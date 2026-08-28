"""Canonical adapters for compact timestamped Binance futures datasets."""

from __future__ import annotations

import pandas as pd

from ..schemas import ArchiveRecord, DatasetKind
from .base_adapter import BinanceArchiveAdapter, normalize_header_columns, open_csv_stream, timestamp_series


def _read_header_frame(record: ArchiveRecord) -> pd.DataFrame:
    with open_csv_stream(record.path) as stream:
        frame = pd.read_csv(stream, low_memory=False)
    return normalize_header_columns(frame)


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Expected one of columns {candidates}, found {list(frame.columns)}")


def _base_event_frame(record: ArchiveRecord, event_time: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": record.exchange,
            "market": record.market.value,
            "dataset": record.dataset.value,
            "symbol": record.symbol,
            "interval": None,
            "event_time": event_time,
            "period_start": event_time,
            "period_end": event_time,
            "available_at": event_time,
            "source_archive": str(record.path),
            "source_fingerprint": record.fingerprint,
        }
    )


def _numeric(raw: pd.DataFrame, candidates: tuple[str, ...], *, integer: bool = False) -> pd.Series:
    values = pd.to_numeric(raw[_first_column(raw, candidates)], errors="raise")
    return values.astype("int64") if integer else values.astype(float)


class BookTickerArchiveAdapter(BinanceArchiveAdapter):
    """Canonical Binance best-bid/best-ask events (not a reconstructed book)."""

    dataset = DatasetKind.BOOK_TICKER
    canonical_schema_version = 1

    def read(self, record: ArchiveRecord) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"BookTickerArchiveAdapter cannot read {record.dataset.value}")
        raw = _read_header_frame(record)
        if raw.empty:
            return pd.DataFrame()
        event_time = timestamp_series(raw[_first_column(raw, ("event_time", "eventtime", "e"))])
        transaction_time = timestamp_series(
            raw[_first_column(raw, ("transaction_time", "transactiontime", "t"))]
        )
        frame = _base_event_frame(record, event_time)
        frame["update_id"] = _numeric(raw, ("update_id", "updateid", "u"), integer=True)
        frame["best_bid_price"] = _numeric(raw, ("best_bid_price", "best_bid", "bid_price", "b"))
        frame["best_bid_qty"] = _numeric(raw, ("best_bid_qty", "best_bid_quantity", "bid_qty", "b_qty", "bq"))
        frame["best_ask_price"] = _numeric(raw, ("best_ask_price", "best_ask", "ask_price", "a"))
        frame["best_ask_qty"] = _numeric(raw, ("best_ask_qty", "best_ask_quantity", "ask_qty", "a_qty", "aq"))
        frame["transaction_time"] = transaction_time
        compare = ["event_time", "transaction_time", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
        duplicates = frame["update_id"].duplicated(keep=False)
        for _, group in frame.loc[duplicates].groupby("update_id", sort=False):
            if any(group[column].nunique(dropna=False) > 1 for column in compare):
                raise ValueError("Conflicting duplicate bookTicker update_id")
        frame = frame.drop_duplicates("update_id", keep="first")
        return frame.sort_values(["event_time", "update_id"], kind="stable").reset_index(drop=True)


class BookDepthArchiveAdapter(BinanceArchiveAdapter):
    """Canonical public percentage-distance depth-band snapshots."""

    dataset = DatasetKind.BOOK_DEPTH
    canonical_schema_version = 1

    def read(self, record: ArchiveRecord) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"BookDepthArchiveAdapter cannot read {record.dataset.value}")
        raw = _read_header_frame(record)
        if raw.empty:
            return pd.DataFrame()
        event_time = timestamp_series(raw[_first_column(raw, ("event_time", "timestamp", "time", "t"))])
        frame = _base_event_frame(record, event_time)
        frame["percentage"] = _numeric(raw, ("percentage", "percent"))
        frame["depth"] = _numeric(raw, ("depth",))
        frame["notional"] = _numeric(raw, ("notional",))
        key = ["event_time", "percentage"]
        duplicates = frame.duplicated(key, keep=False)
        for _, group in frame.loc[duplicates].groupby(key, sort=False):
            if group[["depth", "notional"]].drop_duplicates().shape[0] > 1:
                raise ValueError("Conflicting duplicate bookDepth timestamp/percentage")
        frame = frame.drop_duplicates(key, keep="first")
        return frame.sort_values(key, kind="stable").reset_index(drop=True)


class FuturesMetricsArchiveAdapter(BinanceArchiveAdapter):
    """Open-interest/positioning snapshot fields from Binance Vision metrics."""

    dataset = DatasetKind.FUTURES_METRICS

    _FIELD_MAP = {
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "count_toptrader_long_short_ratio": "top_trader_account_long_short_ratio",
        "sum_toptrader_long_short_ratio": "top_trader_position_long_short_ratio",
        "count_long_short_ratio": "global_long_short_account_ratio",
        "sum_taker_long_short_vol_ratio": "taker_long_short_volume_ratio",
    }

    def read(self, record: ArchiveRecord) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"FuturesMetricsArchiveAdapter cannot read {record.dataset.value}")
        raw = _read_header_frame(record)
        if raw.empty:
            return pd.DataFrame()
        timestamp_col = _first_column(raw, ("create_time", "timestamp", "time"))
        event_time = timestamp_series(raw[timestamp_col])
        frame = _base_event_frame(record, event_time)
        for source, target in self._FIELD_MAP.items():
            if source in raw.columns:
                frame[target] = pd.to_numeric(raw[source], errors="coerce")
        if "symbol" in raw.columns:
            symbols = raw["symbol"].astype(str).str.upper()
            mismatch = symbols.ne(record.symbol) & symbols.ne("")
            if bool(mismatch.any()):
                raise ValueError(f"Metrics symbol mismatch in {record.path}")
        frame = frame.sort_values("event_time", kind="stable")
        return frame.drop_duplicates(subset=["symbol", "event_time"], keep="last").reset_index(drop=True)


class FundingRateArchiveAdapter(BinanceArchiveAdapter):
    """Funding settlement events from Binance Vision monthly archives."""

    dataset = DatasetKind.FUNDING_RATE

    def read(self, record: ArchiveRecord) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"FundingRateArchiveAdapter cannot read {record.dataset.value}")
        raw = _read_header_frame(record)
        if raw.empty:
            return pd.DataFrame()
        timestamp_col = _first_column(raw, ("calc_time", "funding_time", "fundingtime", "timestamp", "time"))
        rate_col = _first_column(raw, ("last_funding_rate", "funding_rate", "fundingrate"))
        event_time = timestamp_series(raw[timestamp_col])
        frame = _base_event_frame(record, event_time)
        frame["funding_rate"] = pd.to_numeric(raw[rate_col], errors="raise")
        interval_col = next((name for name in ("funding_interval_hours", "funding_interval") if name in raw.columns), None)
        if interval_col is not None:
            frame["funding_interval_hours"] = pd.to_numeric(raw[interval_col], errors="coerce")
        if "symbol" in raw.columns:
            symbols = raw["symbol"].astype(str).str.upper()
            mismatch = symbols.ne(record.symbol) & symbols.ne("")
            if bool(mismatch.any()):
                raise ValueError(f"Funding symbol mismatch in {record.path}")
        frame = frame.sort_values("event_time", kind="stable")
        return frame.drop_duplicates(subset=["symbol", "event_time"], keep="last").reset_index(drop=True)
