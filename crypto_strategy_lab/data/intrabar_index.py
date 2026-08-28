"""Searchsorted intrabar slicing without changing simulator exit logic.

The mature engine asks for intrabar rows with the idiom::

    frame[(frame.timestamp >= start) & (frame.timestamp < end)]

On a multi-year 1m DataFrame that boolean expression scans every row for every
strategy candle. ``SearchsortedIntrabarFrame`` preserves that public expression,
but turns the paired timestamp predicates into positional ``searchsorted`` bounds.
The Data Lake production engine can additionally request an array-backed
``FastIntrabarWindow`` so hot exit scans avoid constructing pandas windows while
still consuming the exact same timestamp/OHLC sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class _TimestampRange:
    owner_id: int
    lower: pd.Timestamp | None = None
    lower_inclusive: bool = True
    upper: pd.Timestamp | None = None
    upper_inclusive: bool = False

    def __and__(self, other):
        if not isinstance(other, _TimestampRange) or other.owner_id != self.owner_id:
            return NotImplemented
        lower = self.lower if self.lower is not None else other.lower
        lower_inclusive = self.lower_inclusive if self.lower is not None else other.lower_inclusive
        if self.lower is not None and other.lower is not None:
            if other.lower > self.lower:
                lower, lower_inclusive = other.lower, other.lower_inclusive
            elif other.lower == self.lower:
                lower_inclusive = self.lower_inclusive and other.lower_inclusive

        upper = self.upper if self.upper is not None else other.upper
        upper_inclusive = self.upper_inclusive if self.upper is not None else other.upper_inclusive
        if self.upper is not None and other.upper is not None:
            if other.upper < self.upper:
                upper, upper_inclusive = other.upper, other.upper_inclusive
            elif other.upper == self.upper:
                upper_inclusive = self.upper_inclusive and other.upper_inclusive
        return _TimestampRange(
            owner_id=self.owner_id,
            lower=lower,
            lower_inclusive=lower_inclusive,
            upper=upper,
            upper_inclusive=upper_inclusive,
        )


@dataclass(frozen=True, slots=True)
class FastIntrabarWindow:
    """Zero-copy positional view over the sorted intrabar arrays."""

    left: int
    right: int
    timestamps: pd.DatetimeIndex
    timestamp_values: np.ndarray
    timestamp_unit: str
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray

    @property
    def empty(self) -> bool:
        return self.left >= self.right

    @property
    def first_timestamp(self):
        return pd.NaT if self.empty else self.timestamps[self.left]

    def gap_pairs(self, expected: pd.Timedelta) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
        """Return adjacent timestamp pairs whose spacing exceeds ``expected``."""
        if self.right - self.left <= 1:
            return ()
        values = self.timestamp_values[self.left : self.right]
        expected_units = int(expected / pd.Timedelta(1, unit=self.timestamp_unit))
        gap_offsets = np.flatnonzero(np.diff(values) > expected_units)
        if gap_offsets.size == 0:
            return ()
        return tuple(
            (
                self.timestamps[self.left + int(offset)],
                self.timestamps[self.left + int(offset) + 1],
            )
            for offset in gap_offsets
        )

    def rows(self):
        """Yield the same absolute row index and OHLC values as the pandas slice."""
        for index in range(self.left, self.right):
            yield (
                index,
                # Construct directly from the UTC integer datetime array. Indexing a
                # DatetimeIndex boxes through several pandas layers for every
                # candle; the public Timestamp constructor preserves the exact
                # tz-aware scalar semantics without that indexing chain.
                pd.Timestamp(
                    self.timestamp_values[index], unit=self.timestamp_unit, tz="UTC"
                ),
                float(self.opens[index]),
                float(self.highs[index]),
                float(self.lows[index]),
            )


class _TimestampAccessor:
    """Series-compatible timestamp view with lazy range predicates."""

    def __init__(self, frame: "SearchsortedIntrabarFrame") -> None:
        self._frame = frame

    @staticmethod
    def _utc(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @property
    def _series(self) -> pd.Series:
        return self._frame._timestamp_series()

    def _indexed_endpoint(self, *, maximum: bool):
        index = getattr(self._frame, "_intrabar_timestamp_index", None)
        if (
            index is not None
            and getattr(self._frame, "intrabar_index_mode", None) == "searchsorted"
        ):
            if len(index) == 0:
                return pd.NaT
            return index[-1] if maximum else index[0]
        return None

    def max(self, *args, **kwargs):
        # The production engine calls timestamp.max() after each intrabar scan
        # to distinguish end-of-data from ordinary overlap/gap cases. On a
        # sorted 1m frame the last indexed timestamp is already the exact max,
        # so avoid rescanning the full multi-year Series each time.
        if not args and not kwargs:
            endpoint = self._indexed_endpoint(maximum=True)
            if endpoint is not None:
                return endpoint
        return self._series.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        if not args and not kwargs:
            endpoint = self._indexed_endpoint(maximum=False)
            if endpoint is not None:
                return endpoint
        return self._series.min(*args, **kwargs)

    def __ge__(self, value):
        return _TimestampRange(id(self._frame), lower=self._utc(value), lower_inclusive=True)

    def __gt__(self, value):
        return _TimestampRange(id(self._frame), lower=self._utc(value), lower_inclusive=False)

    def __lt__(self, value):
        return _TimestampRange(id(self._frame), upper=self._utc(value), upper_inclusive=False)

    def __le__(self, value):
        return _TimestampRange(id(self._frame), upper=self._utc(value), upper_inclusive=True)

    def __getitem__(self, key):
        return self._series.__getitem__(key)

    def __len__(self) -> int:
        return len(self._series)

    def __array__(self, dtype=None):
        return np.asarray(self._series, dtype=dtype)

    def __getattr__(self, name):
        return getattr(self._series, name)


class _IntrabarWindowFrame(pd.DataFrame):
    """Internal compatibility window with Series-free row iteration."""

    @property
    def _constructor(self):
        return pd.DataFrame

    def iterrows(self):
        rows = pd.DataFrame.itertuples(self, index=False, name="IntrabarRow")
        return zip(self.index, rows)


class SearchsortedIntrabarFrame(pd.DataFrame):
    """DataFrame whose timestamp range masks use a pre-built DatetimeIndex."""

    _metadata = [
        "_intrabar_timestamp_index",
        "_intrabar_open_values",
        "_intrabar_high_values",
        "_intrabar_low_values",
        "intrabar_index_mode",
        "intrabar_iteration_mode",
    ]

    @property
    def _constructor(self):
        return SearchsortedIntrabarFrame

    def _timestamp_series(self) -> pd.Series:
        return pd.DataFrame.__getitem__(self, "timestamp")

    @property
    def timestamp(self):
        return _TimestampAccessor(self)

    def fast_window(self, start, end) -> FastIntrabarWindow | None:
        """Return an array-backed [start, end) window when the sorted index is valid."""
        index = getattr(self, "_intrabar_timestamp_index", None)
        opens = getattr(self, "_intrabar_open_values", None)
        highs = getattr(self, "_intrabar_high_values", None)
        lows = getattr(self, "_intrabar_low_values", None)
        if (
            index is None
            or opens is None
            or highs is None
            or lows is None
            or getattr(self, "intrabar_index_mode", None) != "searchsorted"
            or len(index) != len(self)
            or len(opens) != len(self)
            or len(highs) != len(self)
            or len(lows) != len(self)
        ):
            return None
        start_ts = _TimestampAccessor._utc(start)
        end_ts = _TimestampAccessor._utc(end)
        left = int(index.searchsorted(start_ts, side="left"))
        right = int(index.searchsorted(end_ts, side="left"))
        return FastIntrabarWindow(
            left, right, index, index.asi8, index.unit, opens, highs, lows
        )

    def _boolean_mask_for_range(self, bounds: _TimestampRange) -> np.ndarray:
        series = pd.to_datetime(self._timestamp_series(), utc=True)
        mask = np.ones(len(self), dtype=bool)
        if bounds.lower is not None:
            mask &= (series >= bounds.lower).to_numpy() if bounds.lower_inclusive else (series > bounds.lower).to_numpy()
        if bounds.upper is not None:
            mask &= (series <= bounds.upper).to_numpy() if bounds.upper_inclusive else (series < bounds.upper).to_numpy()
        return mask

    @staticmethod
    def _engine_window(frame) -> _IntrabarWindowFrame:
        return _IntrabarWindowFrame(frame, copy=False)

    def __getitem__(self, key):
        if isinstance(key, _TimestampRange) and key.owner_id == id(self):
            index = getattr(self, "_intrabar_timestamp_index", None)
            if index is not None and getattr(self, "intrabar_index_mode", None) == "searchsorted":
                left = 0
                right = len(self)
                if key.lower is not None:
                    left = int(index.searchsorted(key.lower, side="left" if key.lower_inclusive else "right"))
                if key.upper is not None:
                    right = int(index.searchsorted(key.upper, side="right" if key.upper_inclusive else "left"))
                return self._engine_window(self.iloc[left:right])
            return self._engine_window(
                pd.DataFrame.__getitem__(self, self._boolean_mask_for_range(key))
            )
        return pd.DataFrame.__getitem__(self, key)


def as_searchsorted_intrabar(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Wrap a sorted intrabar frame; fall back transparently if it is unsorted."""

    if frame is None:
        return None
    if "timestamp" not in frame.columns:
        raise ValueError("Intrabar data requires a timestamp column")

    wrapped = SearchsortedIntrabarFrame(frame.copy(deep=False))
    timestamps = pd.DatetimeIndex(pd.to_datetime(wrapped._timestamp_series(), utc=True))
    wrapped._intrabar_timestamp_index = timestamps
    wrapped._intrabar_open_values = pd.DataFrame.__getitem__(wrapped, "open").to_numpy(float, copy=False)
    wrapped._intrabar_high_values = pd.DataFrame.__getitem__(wrapped, "high").to_numpy(float, copy=False)
    wrapped._intrabar_low_values = pd.DataFrame.__getitem__(wrapped, "low").to_numpy(float, copy=False)
    wrapped.intrabar_index_mode = (
        "searchsorted"
        if not timestamps.hasnans and timestamps.is_monotonic_increasing
        else "boolean_fallback"
    )
    wrapped.intrabar_iteration_mode = "array_window"
    return wrapped
