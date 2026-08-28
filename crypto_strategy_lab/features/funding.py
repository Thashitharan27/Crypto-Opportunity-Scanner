"""Causal Binance funding context aligned to strategy candle decisions."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.alignment import causal_asof_join
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind

from .base import FeatureDefinition, OutputField, ParameterDefinition


FUNDING_CONTEXT_FEATURE_NAME = "funding_context"
FUNDING_CONTEXT_FEATURE_VERSION = "5"


def _funding_bias(rate: np.ndarray) -> np.ndarray:
    state = np.full(len(rate), "UNKNOWN", dtype=object)
    finite = np.isfinite(rate)
    state[finite] = "NEUTRAL"
    state[finite & (rate > 0)] = "POSITIVE"
    state[finite & (rate < 0)] = "NEGATIVE"
    return state


def _datetime_ns(values) -> np.ndarray:
    """Return UTC timestamps as integer nanoseconds independent of pandas dtype unit.

    Parquet/DuckDB paths can preserve timezone-aware datetimes at microsecond
    resolution. ``DatetimeIndex.asi8`` then returns microseconds, while
    ``Timedelta.value`` and settlement transport use nanoseconds. Normalize
    explicitly so cache/input representation cannot change funding arithmetic.
    """
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def _trailing_window_sums(
    decision_times: pd.Series,
    event_times: pd.Series,
    rates: np.ndarray,
    window: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    decisions_ns = _datetime_ns(decision_times)
    events_ns = _datetime_ns(event_times)
    finite = np.isfinite(rates)
    safe_rates = np.where(finite, rates, 0.0)
    prefix_sum = np.concatenate(([0.0], np.cumsum(safe_rates)))
    prefix_count = np.concatenate(([0], np.cumsum(finite.astype(int))))
    window_ns = int(window.value)

    right = np.searchsorted(events_ns, decisions_ns, side="right")
    # A trailing H window is (T-H, T], avoiding double counting the boundary event.
    left = np.searchsorted(events_ns, decisions_ns - window_ns, side="right")
    sums = prefix_sum[right] - prefix_sum[left]
    counts = prefix_count[right] - prefix_count[left]
    return sums.astype(float), counts.astype(int)


def _settlement_batches(
    period_starts: pd.Series,
    decision_times: pd.Series,
    event_times: pd.Series,
    rates: np.ndarray,
) -> np.ndarray:
    """Serialize every settlement inside each strategy candle as compact JSON.

    Funding is an execution cashflow, not merely a decision-time feature. Keeping
    the exact event timestamps here lets the native simulator account for all
    settlements even when one strategy candle spans multiple funding events (for
    example a 1D candle with three 8-hour settlements). Each event is assigned to
    exactly one causal candle window ``(period_start, decision_time]``; events at
    the lower boundary belong to the preceding candle and are never duplicated.
    """
    starts_ns = _datetime_ns(period_starts)
    decisions_ns = _datetime_ns(decision_times)
    events_ns = _datetime_ns(event_times)
    left = np.searchsorted(events_ns, starts_ns, side="right")
    right = np.searchsorted(events_ns, decisions_ns, side="right")
    batches = np.empty(len(decisions_ns), dtype=object)
    for i, (lo, hi) in enumerate(zip(left, right)):
        values = [
            [int(events_ns[j]), float(rates[j])]
            for j in range(int(lo), int(hi))
            if np.isfinite(rates[j])
        ]
        batches[i] = json.dumps(values, separators=(",", ":"))
    return batches


def _time_to_next_known_funding(
    decision_time: pd.Series,
    source_time: pd.Series,
    interval_hours: pd.Series,
) -> np.ndarray:
    """Return seconds to the next known schedule without reading a future event."""
    decision = _datetime_ns(decision_time)
    source = _datetime_ns(source_time)
    intervals = pd.to_numeric(interval_hours, errors="coerce").to_numpy(float) * 3600.0
    elapsed = (decision - source).astype(float) / 1_000_000_000.0
    result = np.full(len(decision), np.nan)
    valid = np.isfinite(intervals) & (intervals > 0) & np.isfinite(elapsed) & (elapsed >= 0)
    if not valid.any():
        return result
    interval = intervals[valid]
    age = elapsed[valid]
    remainder = np.mod(age, interval)
    next_seconds = interval - remainder
    # At an expected settlement boundary the next known event is due now. At the
    # source event itself, however, the next schedule is one full interval away.
    boundary = np.isclose(remainder, 0.0, atol=1e-6)
    next_seconds[boundary & (age > 0)] = 0.0
    next_seconds[boundary & np.isclose(age, 0.0, atol=1e-6)] = interval[
        boundary & np.isclose(age, 0.0, atol=1e-6)
    ]
    result[valid] = next_seconds
    return result


@dataclass(frozen=True, slots=True)
class FundingContextFeatureProvider:
    """Attach only funding settlements already published by each candle close."""

    definition: FeatureDefinition = FeatureDefinition(
        name=FUNDING_CONTEXT_FEATURE_NAME,
        version=FUNDING_CONTEXT_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES, DatasetKind.FUNDING_RATE),
        parameters={
            "funding_zscore_window_days": ParameterDefinition(float, 7.0),
            "funding_zscore_min_samples": ParameterDefinition(int, 6),
            "funding_extreme_zscore": ParameterDefinition(float, 2.0),
        },
        output_columns=(
            "funding_source_available_at",
            "funding_age_hours",
            "funding_rate",
            "funding_rate_bps",
            "funding_interval_hours",
            "funding_bias",
            "funding_event_changed",
            "funding_24h_sum",
            "funding_24h_sum_bps",
            "funding_24h_count",
            "funding_previous",
            "funding_change",
            "funding_3_event_mean",
            "funding_7d_zscore",
            "funding_extreme_positive",
            "funding_extreme_negative",
            "time_to_next_funding",
            "funding_settlements_json",
        ),
        output_schema={
            "funding_extreme_positive": OutputField("bool"),
            "funding_extreme_negative": OutputField("bool"),
            "funding_settlements_json": OutputField("string", nullable=False),
        },
        warmup_bars=0,
        availability_rule="funding_events_available_at_or_before_strategy_candle_close",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[DatasetKind, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        del feature_frames
        parameters = self.definition.normalize_parameters(parameters)
        try:
            klines = datasets[DatasetKind.KLINES].copy()
            funding = datasets[DatasetKind.FUNDING_RATE].copy()
        except KeyError as exc:
            raise ValueError("funding_context requires klines and funding_rate") from exc

        required_kline = {"period_start", "available_at"}
        missing_kline = sorted(required_kline - set(klines.columns))
        if missing_kline:
            raise ValueError(f"Canonical kline frame is missing columns: {missing_kline}")
        if "available_at" not in funding.columns or "funding_rate" not in funding.columns:
            raise ValueError("Canonical funding frame requires available_at and funding_rate")
        if klines.empty:
            raise ValueError("Cannot align funding context to an empty kline frame")

        klines = (
            klines.sort_values("period_start", kind="stable")
            .drop_duplicates("period_start", keep="last")
            .reset_index(drop=True)
        )
        decisions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(klines["period_start"], utc=True),
                "decision_time": pd.to_datetime(klines["available_at"], utc=True),
            }
        )

        funding_columns = ["available_at", "funding_rate"]
        if "funding_interval_hours" in funding.columns:
            funding_columns.append("funding_interval_hours")
        right = funding[funding_columns].copy()
        right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
        right["funding_rate"] = pd.to_numeric(right["funding_rate"], errors="coerce")
        right = (
            right.sort_values("available_at", kind="stable")
            .drop_duplicates("available_at", keep="last")
            .reset_index(drop=True)
        )

        explicit_interval = (
            pd.to_numeric(right["funding_interval_hours"], errors="coerce")
            if "funding_interval_hours" in right.columns
            else pd.Series(np.nan, index=right.index, dtype=float)
        )
        observed_interval = (
            right["available_at"].diff().dt.total_seconds() / 3600.0
        )
        valid_explicit = explicit_interval.where(explicit_interval > 0)
        valid_observed = observed_interval.where(observed_interval > 0)
        # A missing interval may be inferred only from the current and previous
        # already-published events. No future row participates in this value.
        right["funding_interval_hours"] = valid_explicit.combine_first(valid_observed)

        # All derivatives are calculated on the event timeline before alignment.
        right["funding_previous"] = right["funding_rate"].shift(1)
        right["funding_change"] = right["funding_rate"] - right["funding_previous"]
        right["funding_3_event_mean"] = right["funding_rate"].rolling(
            3, min_periods=3
        ).mean()
        event_series = pd.Series(
            right["funding_rate"].to_numpy(float),
            index=pd.DatetimeIndex(right["available_at"]),
        )
        rolling = event_series.rolling(
            f'{float(parameters["funding_zscore_window_days"])}D',
            min_periods=int(parameters["funding_zscore_min_samples"]),
        )
        std = rolling.std(ddof=0)
        right["funding_7d_zscore"] = (
            (event_series - rolling.mean()) / std.where(std > 0)
        ).to_numpy()

        joined = causal_asof_join(decisions, right)
        output = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(joined["timestamp"], utc=True),
                "available_at": pd.to_datetime(joined["decision_time"], utc=True),
                "funding_source_available_at": pd.to_datetime(
                    joined["available_at"], utc=True
                ),
                "funding_rate": pd.to_numeric(joined["funding_rate"], errors="coerce"),
                "funding_previous": pd.to_numeric(
                    joined["funding_previous"], errors="coerce"
                ),
                "funding_change": pd.to_numeric(joined["funding_change"], errors="coerce"),
                "funding_3_event_mean": pd.to_numeric(
                    joined["funding_3_event_mean"], errors="coerce"
                ),
                "funding_7d_zscore": pd.to_numeric(
                    joined["funding_7d_zscore"], errors="coerce"
                ),
                "funding_interval_hours": pd.to_numeric(
                    joined["funding_interval_hours"], errors="coerce"
                ),
            }
        )
        output["funding_age_hours"] = (
            output["available_at"] - output["funding_source_available_at"]
        ).dt.total_seconds() / 3600.0
        output["funding_rate_bps"] = output["funding_rate"] * 10000.0
        output["funding_bias"] = _funding_bias(output["funding_rate"].to_numpy(float))

        threshold = float(parameters["funding_extreme_zscore"])
        z = output["funding_7d_zscore"]
        positive = pd.Series(pd.array([pd.NA] * len(output), dtype="boolean"), index=output.index)
        negative = pd.Series(pd.array([pd.NA] * len(output), dtype="boolean"), index=output.index)
        known = z.notna()
        positive.loc[known] = z.loc[known] >= threshold
        negative.loc[known] = z.loc[known] <= -threshold
        output["funding_extreme_positive"] = positive
        output["funding_extreme_negative"] = negative
        output["time_to_next_funding"] = _time_to_next_known_funding(
            output["available_at"],
            output["funding_source_available_at"],
            output["funding_interval_hours"],
        )

        current_source = output["funding_source_available_at"]
        previous_source = current_source.shift(1)
        output["funding_event_changed"] = current_source.notna() & current_source.ne(
            previous_source
        )

        event_rates = right["funding_rate"].to_numpy(float)
        trailing_sum, trailing_count = _trailing_window_sums(
            output["available_at"],
            right["available_at"],
            event_rates,
            pd.Timedelta(hours=24),
        )
        output["funding_24h_sum"] = trailing_sum
        output["funding_24h_sum_bps"] = trailing_sum * 10000.0
        output["funding_24h_count"] = trailing_count
        output["funding_settlements_json"] = _settlement_batches(
            output["timestamp"],
            output["available_at"],
            right["available_at"],
            event_rates,
        )

        source_available = output["funding_source_available_at"]
        leak = source_available.notna() & (source_available > output["available_at"])
        if bool(leak.any()):
            raise AssertionError("Funding context attached a future funding event")

        output.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                "effective_warmup_bars": self.definition.warmup_bars,
                "trailing_window_hours": 24,
                "request_cache_key": request.cache_key(),
            }
        )
        return output
