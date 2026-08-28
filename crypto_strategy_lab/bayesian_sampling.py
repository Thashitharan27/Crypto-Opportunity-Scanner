"""Causal market-grid sampling for Bayesian direction research.

This module deliberately lives outside normal portfolio simulation.  It reuses the
native execution engine to label hypothetical LONG/SHORT observations at a stable
strategy-candle cadence while keeping research observations independent from one
another's equity and allowing their holding periods to overlap.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from crypto_strategy_lab.bayesian_research import enrich_bayesian_trade_probabilities
from crypto_strategy_lab.config import EntryMode
from crypto_strategy_lab.rule_native_engine import RuleAwareDataLakeProductionBacktestEngine


BAYES_RESEARCH_SAMPLE_VERSION = "BAYES_MARKET_GRID_V1"
DEFAULT_DIRECTIONS = ("LONG", "SHORT")


def resolve_sampling_interval(
    strategy_timeframe_minutes: int,
    sampling_interval_minutes: int | None = None,
) -> tuple[int, int]:
    """Return ``(interval_minutes, strategy_candle_stride)`` for a causal grid.

    ``None`` means every completed strategy candle.  A custom interval must stay
    on the strategy grid: it cannot be lower than the strategy timeframe and it
    must be an exact integer multiple of that timeframe.
    """
    strategy_minutes = int(strategy_timeframe_minutes)
    if strategy_minutes <= 0:
        raise ValueError("strategy timeframe must be positive")
    interval = strategy_minutes if sampling_interval_minutes is None else int(sampling_interval_minutes)
    if interval < strategy_minutes:
        raise ValueError("Bayes sampling interval cannot be lower than the strategy timeframe")
    if interval % strategy_minutes:
        raise ValueError("Bayes sampling interval must be an integer multiple of the strategy timeframe")
    return interval, interval // strategy_minutes


def _normalize_directions(directions: Iterable[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value).upper() for value in directions))
    if not result:
        raise ValueError("at least one Bayes research direction is required")
    unsupported = sorted(set(result) - {"LONG", "SHORT"})
    if unsupported:
        raise ValueError(f"unsupported Bayes research directions: {unsupported}")
    return result


def is_sampling_timestamp(timestamp, interval_minutes: int) -> bool:
    """Return whether a UTC candle-open instant belongs to the stable sample grid."""
    interval = int(interval_minutes)
    if interval <= 0:
        raise ValueError("sampling interval must be positive")
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    epoch_minutes = int(value.value // 60_000_000_000)
    return epoch_minutes % interval == 0


class _BayesResearchSamplingEngine(RuleAwareDataLakeProductionBacktestEngine):
    """Native engine variant that labels independent overlapping market samples."""

    research_forced_side = "LONG"
    research_sampling_interval_minutes = 0

    def _selected_direction(self, i):
        return self.research_forced_side

    def _should_enter(self, i):
        # Research observations are not a portfolio.  Existing open observations,
        # timeout history and max-active-trade limits therefore cannot suppress the
        # next market sample.  All indicator/profile context still has to exist.
        if not np.isfinite(self.risk[i]) or self.risk[i] <= 0 or not self._in_trading_window(i):
            return False
        if self._profile_context(i) is None:
            return False
        return is_sampling_timestamp(self.times[i], self.research_sampling_interval_minutes)

    def _collect_closed_pairs(self, force=False):
        # Keep every hypothetical observation on the same fixed research equity.
        # This preserves native sizing/fees for each observation but prevents one
        # outcome from changing the size or leverage of later overlapping samples.
        still = []
        for pair in self.active_pairs:
            if force or not pair.is_open:
                pair.equity_after_trade = pair.equity_before_trade
                self.completed_pairs.append(pair)
            else:
                still.append(pair)
        self.active_pairs = still


def _research_native_config(native_config, prepared_rows: int):
    """Remove strategy selection rules while preserving native execution settings."""
    profiles = {
        key: replace(
            profile,
            enabled=True,
            flip_direction=False,
            entry_rules=(),
        )
        for key, profile in native_config.strategy_profiles.items()
    }
    return replace(
        native_config,
        strategy_profiles=profiles,
        entry_mode=EntryMode.EVERY_N_CANDLES,
        entry_interval=1,
        enable_daily_entry_schedule=False,
        max_active_pairs=max(1, int(prepared_rows) * 2 + 2),
    )


def _resolved_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove right-censored observations force-closed only because data ended."""
    if frame.empty:
        return frame.copy()
    reasons = []
    for _, row in frame.iterrows():
        side = str(row.get("side", "")).lower()
        reasons.append(str(row.get(f"{side}_exit_reason", "")).upper())
    mask = pd.Series(reasons, index=frame.index) != "END_OF_DATA"
    return frame.loc[mask].copy().reset_index(drop=True)


def _validate_sample_causality(samples: pd.DataFrame, prepared) -> None:
    if samples.empty:
        return
    required = {
        "research_signal_index",
        "research_signal_candle_open_time",
        "entry_time",
        "exit_time",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Bayes research samples are missing causal fields: {missing}")

    indices = pd.to_numeric(samples["research_signal_index"], errors="raise").astype(int)
    if (indices < 0).any() or (indices >= len(prepared)).any():
        raise ValueError("Bayes research sample points outside the prepared strategy frame")

    prepared_times = pd.to_datetime(prepared.timestamp, utc=True)
    available = pd.to_datetime(prepared.decision_available_at, utc=True)
    sample_candles = pd.to_datetime(samples["research_signal_candle_open_time"], utc=True)
    entries = pd.to_datetime(samples["entry_time"], utc=True)
    exits = pd.to_datetime(samples["exit_time"], utc=True)
    for row_number, strategy_index in enumerate(indices):
        if sample_candles.iloc[row_number] != prepared_times[strategy_index]:
            raise ValueError("Bayes research sample is attached to the wrong strategy candle")
        if entries.iloc[row_number] < available[strategy_index]:
            raise ValueError("Bayes research sample enters before its evidence is available")
    if (exits < entries).any():
        raise ValueError("Bayes research sample exits before entry")


def generate_bayesian_research_samples(
    prepared,
    intrabar,
    native_config,
    *,
    sampling_interval_minutes: int | None = None,
    directions: Iterable[str] = DEFAULT_DIRECTIONS,
) -> pd.DataFrame:
    """Generate a separate causal market-grid dataset and score it walk-forward.

    The selected strategy timeframe defines the default cadence.  For example, a
    15m run samples every completed 15m candle, 1h samples hourly, 4h samples every
    four hours and 1D samples daily.  ``sampling_interval_minutes`` can request a
    coarser aligned cadence without changing the run's strategy timeframe.

    LONG and SHORT are evaluated as separate hypothetical trades.  Their holding
    periods may overlap, and each starts from the same fixed research equity.
    Strategy Entry/Veto/flip rules are deliberately removed; regime-specific
    execution profiles, stops, targets, partials, break-even, trailing, timeout,
    fees, slippage, intrabar resolution and S/R execution semantics are preserved.
    """
    strategy_minutes = int(native_config.strategy_timeframe_minutes)
    interval_minutes, _stride = resolve_sampling_interval(
        strategy_minutes, sampling_interval_minutes
    )
    selected_directions = _normalize_directions(directions)
    research_config = _research_native_config(native_config, len(prepared))

    frames: list[pd.DataFrame] = []
    for side in selected_directions:
        engine = _BayesResearchSamplingEngine.from_prepared(
            prepared,
            intrabar,
            research_config,
        )
        engine.research_forced_side = side
        engine.research_sampling_interval_minutes = interval_minutes
        side_frame = _resolved_samples(engine.run())
        if side_frame.empty:
            continue
        side_frame["bayes_sample_population"] = "MARKET_GRID"
        side_frame["bayes_sample_version"] = BAYES_RESEARCH_SAMPLE_VERSION
        side_frame["bayes_sampling_side"] = side
        side_frame["bayes_sampling_interval_minutes"] = interval_minutes
        side_frame["bayes_sampling_strategy_timeframe_minutes"] = strategy_minutes
        side_frame["bayes_sampling_overlap_allowed"] = True
        side_frame["bayes_sampling_independent_equity"] = True
        side_frame["bayes_sampling_entry_policy"] = "AFTER_COMPLETED_STRATEGY_CANDLE"
        side_frame["bayes_sample_id"] = (
            side_frame["research_signal_index"].astype("int64").astype(str)
            + "-"
            + side
        )
        frames.append(side_frame)

    if not frames:
        empty = pd.DataFrame()
        empty.attrs["bayesian_sampling"] = {
            "version": BAYES_RESEARCH_SAMPLE_VERSION,
            "strategy_timeframe_minutes": strategy_minutes,
            "sampling_interval_minutes": interval_minutes,
            "directions": list(selected_directions),
            "overlap_allowed": True,
            "independent_equity": True,
        }
        return empty

    samples = pd.concat(frames, ignore_index=True, sort=False)
    _validate_sample_causality(samples, prepared)
    entries = pd.to_datetime(samples["entry_time"], utc=True)
    side_order = samples["side"].astype(str).str.upper().map({"LONG": 0, "SHORT": 1}).fillna(2)
    samples = (
        samples.assign(_bayes_entry_sort=entries.astype("int64"), _bayes_side_sort=side_order)
        .sort_values(["_bayes_entry_sort", "_bayes_side_sort", "bayes_sample_id"], kind="stable")
        .drop(columns=["_bayes_entry_sort", "_bayes_side_sort"])
        .reset_index(drop=True)
    )
    samples = enrich_bayesian_trade_probabilities(samples)
    samples.attrs["bayesian_sampling"] = {
        "version": BAYES_RESEARCH_SAMPLE_VERSION,
        "strategy_timeframe_minutes": strategy_minutes,
        "sampling_interval_minutes": interval_minutes,
        "directions": list(selected_directions),
        "overlap_allowed": True,
        "independent_equity": True,
        "resolved_rows": len(samples),
    }
    return samples
