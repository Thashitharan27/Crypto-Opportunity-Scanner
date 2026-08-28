"""Causal OHLCV resampling that emits only complete higher-timeframe bars."""

from __future__ import annotations

import pandas as pd

from .timing import interval_to_timedelta


_REQUIRED = ("period_start", "period_end", "available_at", "open", "high", "low", "close", "volume")


def resample_complete_ohlcv(
    frame: pd.DataFrame,
    *,
    source_interval: str,
    target_interval: str,
) -> pd.DataFrame:
    """Aggregate complete source candles into complete target candles only.

    A target bar is emitted only when it contains the expected number of unique,
    contiguous source bars beginning exactly at the target boundary and ending
    exactly at the target boundary. Its availability is the latest dependency
    availability, never merely its label time.
    """

    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing canonical OHLCV columns: {missing}")
    source_delta = interval_to_timedelta(source_interval)
    target_delta = interval_to_timedelta(target_interval)
    source_ns = int(source_delta.total_seconds() * 1_000_000_000)
    target_ns = int(target_delta.total_seconds() * 1_000_000_000)
    if target_ns <= source_ns or target_ns % source_ns:
        raise ValueError("target_interval must be an integer multiple greater than source_interval")
    expected_count = target_ns // source_ns

    work = frame.copy()
    for column in ("period_start", "period_end", "available_at"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    work = work.sort_values("period_start", kind="stable").drop_duplicates("period_start", keep="last")
    start_ns = work["period_start"].astype("int64")
    bucket_ns = (start_ns // target_ns) * target_ns
    work["_bucket"] = pd.to_datetime(bucket_ns, utc=True)

    rows: list[dict[str, object]] = []
    for bucket, group in work.groupby("_bucket", sort=True):
        group = group.sort_values("period_start", kind="stable")
        target_end = bucket + target_delta
        expected_starts = pd.date_range(bucket, periods=expected_count, freq=source_delta, tz="UTC")
        actual_starts = pd.DatetimeIndex(group["period_start"])
        complete = len(group) == expected_count and actual_starts.equals(expected_starts)
        complete = complete and bool((group["period_end"].iloc[-1] == target_end))
        if not complete:
            continue
        row: dict[str, object] = {
            "period_start": bucket,
            "period_end": target_end,
            "event_time": bucket,
            "available_at": group["available_at"].max(),
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "volume": group["volume"].sum(),
            "source_bars": int(len(group)),
            "interval": target_interval,
        }
        for metadata in ("exchange", "market", "dataset", "symbol"):
            if metadata in group.columns:
                values = group[metadata].dropna().unique()
                if len(values) > 1:
                    raise ValueError(f"Cannot resample mixed {metadata} values in one frame")
                row[metadata] = values[0] if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)
