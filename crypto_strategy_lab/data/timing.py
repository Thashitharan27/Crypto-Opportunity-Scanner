"""Canonical timing and availability rules used to prevent look-ahead."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

from .schemas import DatasetKind


_INTERVAL_RE = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>[smhdw])$")

# Binance archive/native kline interval names. Requests may arrive from the
# existing GUI as minute counts (for example 240m for a 4-hour strategy), so we
# canonicalize equivalent fixed durations before catalog lookup/cache identity.
_BINANCE_FIXED_INTERVALS = (
    "1s",
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w",
)

_CANDLE_DATASETS = {
    DatasetKind.KLINES,
    DatasetKind.MARK_PRICE_KLINES,
    DatasetKind.INDEX_PRICE_KLINES,
    DatasetKind.PREMIUM_INDEX_KLINES,
}


def ensure_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime, treating a naive value as UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def interval_to_timedelta(interval: str) -> timedelta:
    """Convert fixed Binance intervals such as 1m, 15m, 4h or 1d."""

    text = str(interval).strip()
    match = _INTERVAL_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Unsupported fixed interval: {interval!r}")
    count = int(match.group("count"))
    unit = match.group("unit")
    if unit == "s":
        return timedelta(seconds=count)
    if unit == "m":
        return timedelta(minutes=count)
    if unit == "h":
        return timedelta(hours=count)
    if unit == "d":
        return timedelta(days=count)
    return timedelta(weeks=count)


def normalize_binance_interval(interval: str) -> str:
    """Return the Binance-native name for an equivalent fixed interval.

    This intentionally normalizes only durations Binance publishes natively.
    Examples: ``60m -> 1h`` and ``240m -> 4h``. Non-native intervals remain in
    their original fixed-duration spelling so a later resampling layer can deal
    with them explicitly rather than silently changing their meaning.
    """

    text = str(interval).strip()
    if not text:
        raise ValueError("interval must not be empty")
    if text == "1M":  # Binance calendar-month interval; not a fixed timedelta.
        return text
    lowered = text.lower()
    requested = interval_to_timedelta(lowered)
    for native in _BINANCE_FIXED_INTERVALS:
        if interval_to_timedelta(native) == requested:
            return native
    return lowered


_AUTHORITATIVE_FIXED_GRIDS = {"1m", "5m", "15m", "1h", "4h", "1d"}


def floor_fixed_candle_grid(value: datetime, interval: str) -> datetime:
    """Floor ``value`` to an authoritative Binance fixed-candle UTC grid.

    Decision and event timestamps must not use this helper.  It is exclusively
    for the bounds of fixed-cadence candle requests.
    """

    normalized = normalize_binance_interval(interval)
    if normalized not in _AUTHORITATIVE_FIXED_GRIDS:
        raise ValueError(f"Unsupported authoritative candle grid: {interval!r}")
    utc = ensure_utc(value)
    step = int(interval_to_timedelta(normalized).total_seconds())
    epoch_seconds = int(utc.timestamp())
    return datetime.fromtimestamp(epoch_seconds - epoch_seconds % step, tz=timezone.utc)


def canonical_available_at(
    dataset: DatasetKind,
    event_time: datetime,
    *,
    interval: str | None = None,
    period_end: datetime | None = None,
) -> datetime:
    """Return the earliest timestamp at which a source value may be consumed.

    Candle datasets are withheld until their candle is complete. Futures
    metrics in Binance Vision are timestamped snapshots; if a provider later
    supplies an explicit aggregate interval/period end, that later boundary is
    used. Event datasets (funding, trades and order-book events) are usable at
    their event timestamp.
    """

    event_time = ensure_utc(event_time)
    if dataset in _CANDLE_DATASETS:
        if period_end is not None:
            return ensure_utc(period_end)
        if interval is None:
            raise ValueError(f"{dataset.value} requires interval or period_end")
        return event_time + interval_to_timedelta(interval)
    if dataset == DatasetKind.FUTURES_METRICS:
        if period_end is not None:
            return ensure_utc(period_end)
        if interval is not None:
            return event_time + interval_to_timedelta(interval)
        return event_time
    return event_time
