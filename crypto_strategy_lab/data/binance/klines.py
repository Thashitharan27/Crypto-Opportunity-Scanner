"""Canonical adapters for Binance regular and reference-price kline archives."""

from __future__ import annotations

import pandas as pd

from ..schemas import ArchiveRecord, DatasetKind
from ..timing import interval_to_timedelta
from .base_adapter import BinanceArchiveAdapter, open_csv_stream, timestamp_series


_BINANCE_COLUMNS = [
    "open_time_raw",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_raw",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

_KLINE_DATASETS = {
    DatasetKind.KLINES,
    DatasetKind.MARK_PRICE_KLINES,
    DatasetKind.INDEX_PRICE_KLINES,
    DatasetKind.PREMIUM_INDEX_KLINES,
}


class KlineLikeArchiveAdapter(BinanceArchiveAdapter):
    """Normalizes all Binance kline-shaped public-data families."""

    def __init__(self, dataset: DatasetKind) -> None:
        if dataset not in _KLINE_DATASETS:
            raise ValueError(f"Not a kline-shaped dataset: {dataset.value}")
        self.dataset = dataset

    def read(self, record: ArchiveRecord) -> pd.DataFrame:
        if record.dataset != self.dataset:
            raise ValueError(f"{self.__class__.__name__} cannot read {record.dataset.value}")
        if not record.interval:
            raise ValueError(f"Kline archive has no fixed interval metadata: {record.path}")

        with open_csv_stream(record.path) as stream:
            raw = pd.read_csv(stream, header=None, low_memory=False)
        if raw.empty:
            return pd.DataFrame()
        first = str(raw.iloc[0, 0]).strip().lower().replace(" ", "_")
        if first in {"open_time", "opentime"}:
            raw = raw.iloc[1:].reset_index(drop=True)
        if raw.shape[1] < 6:
            raise ValueError(f"Unexpected Binance kline schema in {record.path}: {raw.shape[1]} columns")
        raw = raw.iloc[:, : min(raw.shape[1], len(_BINANCE_COLUMNS))].copy()
        raw.columns = _BINANCE_COLUMNS[: raw.shape[1]]

        period_start = timestamp_series(raw["open_time_raw"])
        delta = interval_to_timedelta(record.interval)
        period_end = period_start + delta

        frame = pd.DataFrame(
            {
                "exchange": record.exchange,
                "market": record.market.value,
                "dataset": record.dataset.value,
                "symbol": record.symbol,
                "interval": record.interval,
                "event_time": period_start,
                "period_start": period_start,
                "period_end": period_end,
                "available_at": period_end,
                "open": pd.to_numeric(raw["open"], errors="raise"),
                "high": pd.to_numeric(raw["high"], errors="raise"),
                "low": pd.to_numeric(raw["low"], errors="raise"),
                "close": pd.to_numeric(raw["close"], errors="raise"),
                "volume": pd.to_numeric(raw["volume"], errors="raise"),
                "source_archive": str(record.path),
                "source_fingerprint": record.fingerprint,
            }
        )
        optional_numeric = {
            "quote_volume": "quote_volume",
            "trade_count": "trade_count",
            "taker_buy_base_volume": "taker_buy_base_volume",
            "taker_buy_quote_volume": "taker_buy_quote_volume",
        }
        for output_name, raw_name in optional_numeric.items():
            if raw_name in raw.columns:
                frame[output_name] = pd.to_numeric(raw[raw_name], errors="coerce")

        frame = frame.sort_values("period_start", kind="stable")
        frame = frame.drop_duplicates(subset=["symbol", "interval", "period_start"], keep="last")
        frame = frame.reset_index(drop=True)
        invalid = (frame["low"] > frame["high"]) | (frame["volume"] < 0)
        invalid |= (frame["open"] < frame["low"]) | (frame["open"] > frame["high"])
        invalid |= (frame["close"] < frame["low"]) | (frame["close"] > frame["high"])
        if bool(invalid.any()):
            raise ValueError(f"Invalid OHLCV rows found in Binance archive: {record.path}")
        return frame


class KlineArchiveAdapter(KlineLikeArchiveAdapter):
    def __init__(self) -> None:
        super().__init__(DatasetKind.KLINES)
