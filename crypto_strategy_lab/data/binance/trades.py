"""Canonical adapters for Binance trade-event archives."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import ArchiveRecord, DatasetKind
from .base_adapter import BinanceArchiveAdapter, open_csv_stream, timestamp_series


_AGG_TRADE_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)

_TRADE_COLUMNS = (
    "trade_id", "price", "quantity", "quote_quantity", "transact_time",
    "is_buyer_maker",
)


def _key(value: object) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


def _boolean_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "t": True,
        "f": False,
    }
    unknown = ~normalized.isin(mapping)
    if bool(unknown.any()):
        samples = sorted(normalized.loc[unknown].dropna().unique().tolist())[:5]
        raise ValueError(f"Unrecognized Binance buyer-maker values: {samples}")
    return normalized.map(mapping).astype(bool)


class AggTradesArchiveAdapter(BinanceArchiveAdapter):
    """Normalize Binance aggregate trade events without deriving strategy features."""

    dataset = DatasetKind.AGG_TRADES

    def read(self, record: ArchiveRecord, *, start=None, end=None) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"AggTradesArchiveAdapter cannot read {record.dataset.value}")
        with open_csv_stream(record.path) as stream:
            raw = pd.read_csv(stream, header=None, low_memory=False)
        if raw.empty:
            return pd.DataFrame()

        first = _key(raw.iloc[0, 0])
        if first in {"aggtradeid", "aggregatetradeid", "id"}:
            raw = raw.iloc[1:].reset_index(drop=True)
        if raw.shape[1] < len(_AGG_TRADE_COLUMNS):
            raise ValueError(
                f"Unexpected Binance aggTrades schema in {record.path}: {raw.shape[1]} columns"
            )
        raw = raw.iloc[:, : len(_AGG_TRADE_COLUMNS)].copy()
        raw.columns = _AGG_TRADE_COLUMNS

        event_time = timestamp_series(raw["transact_time"])
        if start is not None or end is not None:
            mask = pd.Series(True, index=raw.index)
            if start is not None:
                mask &= event_time >= start
            if end is not None:
                mask &= event_time < end
            raw = raw.loc[mask].reset_index(drop=True)
            event_time = event_time.loc[mask].reset_index(drop=True)
        price = pd.to_numeric(raw["price"], errors="raise")
        quantity = pd.to_numeric(raw["quantity"], errors="raise")
        buyer_maker = _boolean_series(raw["is_buyer_maker"])
        frame = pd.DataFrame(
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
                "agg_trade_id": pd.to_numeric(raw["agg_trade_id"], errors="raise").astype("int64"),
                "price": price,
                "quantity": quantity,
                "quote_quantity": price * quantity,
                "first_trade_id": pd.to_numeric(raw["first_trade_id"], errors="raise").astype("int64"),
                "last_trade_id": pd.to_numeric(raw["last_trade_id"], errors="raise").astype("int64"),
                "is_buyer_maker": buyer_maker,
                # Buyer maker means the aggressive side was the seller.
                "taker_side": np.where(buyer_maker.to_numpy(bool), "SELL", "BUY"),
                "source_archive": str(record.path),
                "source_fingerprint": record.fingerprint,
            }
        )
        invalid = (
            ~np.isfinite(frame["price"].to_numpy(float))
            | ~np.isfinite(frame["quantity"].to_numpy(float))
            | (frame["price"].to_numpy(float) <= 0)
            | (frame["quantity"].to_numpy(float) <= 0)
            | (frame["last_trade_id"].to_numpy(np.int64) < frame["first_trade_id"].to_numpy(np.int64))
        )
        if bool(invalid.any()):
            raise ValueError(f"Invalid Binance aggTrades rows found in {record.path}")
        if bool(frame["agg_trade_id"].duplicated().any()):
            raise ValueError(f"Duplicate Binance aggregate trade IDs found in {record.path}")
        return frame.sort_values(["event_time", "agg_trade_id"], kind="stable").reset_index(drop=True)


class TradesArchiveAdapter(BinanceArchiveAdapter):
    """Normalize Binance USD-M individual trade archives."""

    dataset = DatasetKind.TRADES

    def read(self, record: ArchiveRecord, *, start=None, end=None) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"TradesArchiveAdapter cannot read {record.dataset.value}")
        with open_csv_stream(record.path) as stream:
            raw = pd.read_csv(stream, header=None, low_memory=False)
        if raw.empty:
            return pd.DataFrame()
        if _key(raw.iloc[0, 0]) in {"tradeid", "id"}:
            raw = raw.iloc[1:].reset_index(drop=True)
        if raw.shape[1] < len(_TRADE_COLUMNS):
            raise ValueError(f"Unexpected Binance trades schema in {record.path}: {raw.shape[1]} columns")
        raw = raw.iloc[:, :len(_TRADE_COLUMNS)].copy()
        raw.columns = _TRADE_COLUMNS
        event_time = timestamp_series(raw["transact_time"])
        if start is not None or end is not None:
            mask = pd.Series(True, index=raw.index)
            if start is not None:
                mask &= event_time >= start
            if end is not None:
                mask &= event_time < end
            raw = raw.loc[mask].reset_index(drop=True)
            event_time = event_time.loc[mask].reset_index(drop=True)
        price = pd.to_numeric(raw["price"], errors="raise")
        quantity = pd.to_numeric(raw["quantity"], errors="raise")
        quote = pd.to_numeric(raw["quote_quantity"], errors="raise")
        maker = _boolean_series(raw["is_buyer_maker"])
        frame = pd.DataFrame({
            "exchange": record.exchange, "market": record.market.value,
            "dataset": record.dataset.value, "symbol": record.symbol, "interval": None,
            "event_time": event_time, "period_start": event_time, "period_end": event_time,
            "available_at": event_time,
            "trade_id": pd.to_numeric(raw["trade_id"], errors="raise").astype("int64"),
            "price": price, "quantity": quantity, "quote_quantity": quote,
            "is_buyer_maker": maker,
            "taker_side": np.where(maker.to_numpy(bool), "SELL", "BUY"),
            "source_archive": str(record.path), "source_fingerprint": record.fingerprint,
        })
        invalid = (~np.isfinite(frame["price"]) | ~np.isfinite(frame["quantity"])
                   | ~np.isfinite(frame["quote_quantity"]) | (frame["price"] <= 0)
                   | (frame["quantity"] <= 0) | (frame["quote_quantity"] < 0))
        if bool(invalid.any()):
            raise ValueError(f"Invalid Binance trades rows found in {record.path}")
        if bool(frame["trade_id"].duplicated().any()):
            raise ValueError(f"Duplicate Binance trade IDs found in {record.path}")
        return frame.sort_values(["event_time", "trade_id"], kind="stable").reset_index(drop=True)
