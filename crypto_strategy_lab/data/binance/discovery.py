"""Discover Binance public-data archives without requiring a fixed collector layout."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

from ..cache import stat_fingerprint
from ..schemas import ArchiveRecord, DatasetKind, MarketKind


_DATASET_ALIASES = {
    "klines": DatasetKind.KLINES,
    "metrics": DatasetKind.FUTURES_METRICS,
    "fundingrate": DatasetKind.FUNDING_RATE,
    "funding": DatasetKind.FUNDING_RATE,
    "markpriceklines": DatasetKind.MARK_PRICE_KLINES,
    "indexpriceklines": DatasetKind.INDEX_PRICE_KLINES,
    "premiumindexklines": DatasetKind.PREMIUM_INDEX_KLINES,
    "premiumpriceklines": DatasetKind.PREMIUM_INDEX_KLINES,
    "aggtrades": DatasetKind.AGG_TRADES,
    "trades": DatasetKind.TRADES,
    "bookdepth": DatasetKind.BOOK_DEPTH,
    "bookticker": DatasetKind.BOOK_TICKER,
}

_INTERVAL_DATASETS = {
    DatasetKind.KLINES,
    DatasetKind.MARK_PRICE_KLINES,
    DatasetKind.INDEX_PRICE_KLINES,
    DatasetKind.PREMIUM_INDEX_KLINES,
}

_INTERVAL_RE = re.compile(r"^[1-9][0-9]*(?:s|m|h|d|w)$", re.IGNORECASE)
_DAILY_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_MONTHLY_RE = re.compile(r"(?<!\d)(\d{4}-\d{2})(?!-\d{2})(?!\d)")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _infer_market(parts: tuple[str, ...]) -> MarketKind | None:
    keys = {_key(part) for part in parts}
    if keys.intersection({"um", "usdm", "usdtm", "usdsm"}):
        return MarketKind.FUTURES_UM
    if keys.intersection({"cm", "coinm", "coinmargined"}):
        return MarketKind.FUTURES_CM
    if "spot" in keys:
        return MarketKind.SPOT
    return None


def _period_from_name(name: str, frequency: str) -> tuple[datetime | None, datetime | None]:
    if frequency == "daily":
        match = _DAILY_RE.search(name)
        if match:
            start = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return start, start + timedelta(days=1)
    if frequency == "monthly":
        match = _MONTHLY_RE.search(name)
        if match:
            start = datetime.strptime(match.group(1), "%Y-%m").replace(tzinfo=timezone.utc)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
            return start, end
    return None, None


def infer_archive_record(path: Path, raw_root: Path) -> ArchiveRecord | None:
    """Infer canonical archive metadata from standard or collector-preserved paths."""

    path = Path(path)
    raw_root = Path(raw_root)
    if path.suffix.lower() not in {".zip", ".csv"} or not path.is_file():
        return None
    try:
        relative = path.relative_to(raw_root)
    except ValueError:
        return None
    parts = relative.parts
    market = _infer_market(parts)
    if market is None:
        return None

    dataset: DatasetKind | None = None
    dataset_index: int | None = None
    for index, part in enumerate(parts[:-1]):
        candidate = _DATASET_ALIASES.get(_key(part))
        if candidate is not None:
            dataset = candidate
            dataset_index = index
            break
    if dataset is None or dataset_index is None:
        return None

    frequency = "unknown"
    part_keys = {_key(part) for part in parts}
    if "daily" in part_keys:
        frequency = "daily"
    elif "monthly" in part_keys:
        frequency = "monthly"

    tail = list(parts[dataset_index + 1 : -1])
    if not tail:
        return None
    symbol = tail[0].upper()
    interval: str | None = None
    if dataset in _INTERVAL_DATASETS and len(tail) >= 2 and _INTERVAL_RE.fullmatch(tail[1]):
        interval = tail[1]

    period_start, period_end = _period_from_name(path.name, frequency)
    stat = path.stat()
    return ArchiveRecord(
        raw_root=raw_root.resolve(),
        path=path.resolve(),
        market=market,
        dataset=dataset,
        symbol=symbol,
        interval=interval,
        frequency=frequency,
        period_start=period_start,
        period_end=period_end,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        fingerprint=stat_fingerprint(path),
    )


def discover_archives(raw_root: Path) -> list[ArchiveRecord]:
    """Recursively catalog recognized Binance ZIP/CSV files under `raw_root`."""

    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Binance data-lake root does not exist: {root}")
    records: list[ArchiveRecord] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".zip", ".csv"}:
            continue
        record = infer_archive_record(path, root)
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: (item.market.value, item.dataset.value, item.symbol, item.interval or "", item.period_start or datetime.min.replace(tzinfo=timezone.utc), str(item.path)))
    return records
