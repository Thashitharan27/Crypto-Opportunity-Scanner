"""Native, array-oriented contracts at the simulator preparation boundary.

This module deliberately contains no indicator or loader logic.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np
import pandas as pd


_FLOAT_FIELDS = (
    "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
    "plus_di", "minus_di", "bb_width", "bb_width_pct", "session_vwap",
    "close_location", "mean_reversion_mean", "mean_reversion_distance_atr",
    "mean_reversion_distance_atr_previous", "mean_reversion_sigma",
    "mean_reversion_bb_upper", "mean_reversion_bb_lower",
    "mean_reversion_bb_zscore", "mean_reversion_rsi",
    "di_spread", "di_spread_1", "di_spread_3", "di_spread_5",
    "di_spread_change", "di_ratio", "plus_di_change", "minus_di_change",
    "di_pressure_spread_change", "long_directional_di_change",
    "long_opposing_di_change", "short_directional_di_change",
    "short_opposing_di_change", "bb_middle", "bb_upper", "bb_lower",
    "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
    "bb_width_change_pct", "mean_reversion_distance_change_atr",
)
_TEXT_FIELDS = (
    "long_di_pressure_state", "short_di_pressure_state",
    "mean_reversion_state", "mean_reversion_motion",
    "mean_reversion_strength_label", "mean_reversion_bb_location",
    "mean_reversion_rsi_state", "mean_reversion_reentry_confirmation",
    "mean_reversion_signal", "mean_reversion_signal_direction",
    "mean_reversion_setup_strength", "bb_reentry", "mr_signal",
    "mr_signal_direction",
)
_REQUIRED_FINITE = ("open", "high", "low", "close", "volume")


def _readonly(values, dtype, name: str) -> np.ndarray:
    array = np.asarray(values)
    if dtype == "datetime64[ns]":
        try:
            array = pd.to_datetime(array, utc=True).to_numpy(dtype="datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain UTC-compatible timestamps") from exc
    else:
        if array.dtype.kind not in "fiu" and dtype is np.float64:
            raise TypeError(f"{name} must have a numeric dtype")
        if array.dtype.kind != "b" and dtype is np.bool_:
            raise TypeError(f"{name} must have boolean dtype")
        array = np.asarray(array, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _aligned_timestamps(values, name: str) -> np.ndarray:
    """Normalize a timeline for adapter-level row-for-row alignment checks."""
    result = _readonly(values, "datetime64[ns]", name)
    if np.isnat(result).any():
        raise ValueError(f"{name} contains missing timestamps")
    return result


def _require_exact_timeline(reference: np.ndarray, values, label: str) -> np.ndarray:
    candidate = _aligned_timestamps(values, f"{label} timestamp")
    if len(candidate) != len(reference) or not np.array_equal(candidate, reference):
        raise ValueError(f"{label} timestamps are not exactly aligned to strategy timestamps")
    return candidate


@dataclass(frozen=True, slots=True)
class ResearchContext:
    """One reporting-only, timestamp-aligned feature block."""

    name: str
    available_at: np.ndarray
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("research context name is required")
        available = _readonly(self.available_at, "datetime64[ns]", "research available_at")
        if np.isnat(available).any():
            raise ValueError("research available_at contains missing timestamps")
        clean: dict[str, np.ndarray] = {}
        for name, value in self.values.items():
            if name in {"timestamp", "available_at"}:
                raise ValueError(f"research value name is reserved: {name}")
            array = np.asarray(value)
            if array.ndim != 1:
                raise ValueError(f"research field {name} must be one-dimensional")
            array = np.array(array, copy=True)
            array.setflags(write=False)
            clean[name] = array
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "values", MappingProxyType(clean))


@dataclass(frozen=True, slots=True)
class PreparedBacktestFrame:
    """Validated values on the strategy candle-open timeline.

    OHLCV and ATR are execution-critical.  The remaining core arrays are
    strategy-decision context already prepared elsewhere. Research blocks are
    isolated and cannot be accessed as core execution fields.
    """

    timestamp: np.ndarray
    strategy_interval: pd.Timedelta
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr: np.ndarray
    atr_pct: np.ndarray
    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    bb_width: np.ndarray
    bb_width_pct: np.ndarray
    session_vwap: np.ndarray
    close_location: np.ndarray
    mean_reversion_mean: np.ndarray
    mean_reversion_distance_atr: np.ndarray
    mean_reversion_distance_atr_previous: np.ndarray
    mean_reversion_sigma: np.ndarray
    mean_reversion_bb_upper: np.ndarray
    mean_reversion_bb_lower: np.ndarray
    mean_reversion_bb_zscore: np.ndarray
    mean_reversion_bb_location: np.ndarray
    mean_reversion_rsi: np.ndarray
    mean_reversion_rsi_state: np.ndarray
    mean_reversion_long_reentry: np.ndarray
    mean_reversion_short_reentry: np.ndarray
    mean_reversion_reentry_confirmation: np.ndarray
    mean_reversion_signal: np.ndarray
    mean_reversion_signal_direction: np.ndarray
    mean_reversion_setup_strength: np.ndarray
    bb_reentry: np.ndarray
    mr_signal: np.ndarray
    mr_signal_direction: np.ndarray
    di_spread: np.ndarray
    di_spread_1: np.ndarray
    di_spread_3: np.ndarray
    di_spread_5: np.ndarray
    di_spread_change: np.ndarray
    di_ratio: np.ndarray
    plus_di_change: np.ndarray
    minus_di_change: np.ndarray
    di_pressure_spread_change: np.ndarray
    long_directional_di_change: np.ndarray
    long_opposing_di_change: np.ndarray
    long_di_pressure_state: np.ndarray
    short_directional_di_change: np.ndarray
    short_opposing_di_change: np.ndarray
    short_di_pressure_state: np.ndarray
    bb_middle: np.ndarray
    bb_upper: np.ndarray
    bb_lower: np.ndarray
    bb_width_1: np.ndarray
    bb_width_3: np.ndarray
    bb_width_5: np.ndarray
    bb_width_change: np.ndarray
    bb_width_change_pct: np.ndarray
    mean_reversion_distance_change_atr: np.ndarray
    mean_reversion_state: np.ndarray
    mean_reversion_motion: np.ndarray
    mean_reversion_strength: np.ndarray
    mean_reversion_strength_label: np.ndarray
    bull_regime_return: np.ndarray
    market_regime: np.ndarray
    momentum_returns_by_hours: Mapping[int, np.ndarray]
    decision_available_at: np.ndarray
    research: tuple[ResearchContext, ...] = ()

    def __post_init__(self) -> None:
        interval = pd.Timedelta(self.strategy_interval)
        if interval <= pd.Timedelta(0):
            raise ValueError("strategy_interval must be positive")
        object.__setattr__(self, "strategy_interval", interval)
        object.__setattr__(self, "timestamp", _readonly(self.timestamp, "datetime64[ns]", "timestamp"))
        object.__setattr__(self, "decision_available_at", _readonly(
            self.decision_available_at, "datetime64[ns]", "decision_available_at"
        ))
        for name in _FLOAT_FIELDS:
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float64, name))
        for name in ("mean_reversion_long_reentry", "mean_reversion_short_reentry"):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.bool_, name))
        for name in _TEXT_FIELDS:
            object.__setattr__(self, name, _readonly(getattr(self, name), object, name))
        object.__setattr__(self, "mean_reversion_strength", _readonly(
            self.mean_reversion_strength, np.int64, "mean_reversion_strength"
        ))
        object.__setattr__(self, "bull_regime_return", _readonly(
            self.bull_regime_return, np.float64, "bull_regime_return"
        ))
        object.__setattr__(self, "market_regime", _readonly(
            self.market_regime, object, "market_regime"
        ))
        momentum = {
            int(hours): _readonly(values, np.float64, f"momentum return {hours}h")
            for hours, values in self.momentum_returns_by_hours.items()
        }
        if any(hours <= 0 for hours in momentum):
            raise ValueError("momentum lookback hours must be positive")
        object.__setattr__(self, "momentum_returns_by_hours", MappingProxyType(momentum))

        length = len(self.timestamp)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray) and len(value) != length:
                raise ValueError(f"{field.name} length {len(value)} does not match timestamp length {length}")
        if any(len(value) != length for value in self.momentum_returns_by_hours.values()):
            raise ValueError("momentum return arrays are not aligned to strategy rows")
        if np.isnat(self.timestamp).any():
            raise ValueError("timestamps cannot contain missing values")
        if np.isnat(self.decision_available_at).any():
            raise ValueError("decision_available_at cannot contain missing values")
        if length and (np.diff(self.timestamp).astype("timedelta64[ns]") <= np.timedelta64(0, "ns")).any():
            raise ValueError("timestamps must be strictly increasing and unique")
        for name in _REQUIRED_FINITE:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains missing or non-finite execution values")
        if np.any(self.decision_available_at < self.timestamp):
            raise ValueError("decision features cannot be available before their strategy candle opens")
        if np.any(self.decision_available_at > self.timestamp + interval.to_timedelta64()):
            raise ValueError("decision features use data unavailable at strategy candle completion")
        object.__setattr__(self, "research", tuple(self.research))
        names: set[str] = set()
        for block in self.research:
            if not isinstance(block, ResearchContext):
                raise TypeError("research entries must be ResearchContext instances")
            if block.name in names:
                raise ValueError(f"duplicate research context name: {block.name}")
            names.add(block.name)
            if len(block.available_at) != length or any(len(value) != length for value in block.values.values()):
                raise ValueError(f"research context {block.name} is not aligned to strategy rows")
            if np.any(block.available_at < self.timestamp) or np.any(
                block.available_at > self.timestamp + interval.to_timedelta64()
            ):
                raise ValueError(f"research context {block.name} violates causal availability")

    def __len__(self) -> int:
        return len(self.timestamp)


@dataclass(frozen=True, slots=True)
class IntrabarExecutionData:
    """Execution-resolution OHLC arrays, separate from strategy decisions."""

    timestamp: np.ndarray
    interval: pd.Timedelta
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray

    def __post_init__(self) -> None:
        interval = pd.Timedelta(self.interval)
        if interval <= pd.Timedelta(0):
            raise ValueError("intrabar interval must be positive")
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "timestamp", _readonly(self.timestamp, "datetime64[ns]", "intrabar timestamp"))
        for name in ("open", "high", "low"):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float64, f"intrabar {name}"))
        length = len(self.timestamp)
        if any(len(getattr(self, name)) != length for name in ("open", "high", "low")):
            raise ValueError("intrabar arrays must have equal lengths")
        if np.isnat(self.timestamp).any():
            raise ValueError("intrabar timestamps cannot contain missing values")
        if length and (np.diff(self.timestamp) <= np.timedelta64(0, "ns")).any():
            raise ValueError("intrabar timestamps must be strictly increasing and unique")
        if any(not np.isfinite(getattr(self, name)).all() for name in ("open", "high", "low")):
            raise ValueError("intrabar OHLC contains missing or non-finite values")

    def validate_compatible(self, strategy: PreparedBacktestFrame) -> None:
        """Validate execution-grid compatibility without requiring full-period coverage."""
        if self.interval >= strategy.strategy_interval:
            raise ValueError("intrabar interval must be smaller than the strategy interval")
        if strategy.strategy_interval % self.interval:
            raise ValueError("strategy interval must be an exact multiple of intrabar interval")
        if len(self.timestamp) and len(strategy):
            strategy_start = strategy.timestamp[0]
            strategy_end = strategy.timestamp[-1] + strategy.strategy_interval.to_timedelta64()
            offsets = (self.timestamp - strategy_start).astype("timedelta64[ns]").astype(np.int64)
            if np.any(offsets % self.interval.value):
                raise ValueError("intrabar timestamps are not aligned to the strategy time grid")
            if self.timestamp[-1] < strategy_start or self.timestamp[0] >= strategy_end:
                raise ValueError("intrabar data does not overlap the strategy execution period")

    @property
    def max_timestamp(self) -> pd.Timestamp | None:
        return pd.Timestamp(self.timestamp[-1]) if len(self.timestamp) else None

    def fast_window(self, start, end):
        """Return an array-backed execution window without constructing pandas rows."""
        start64 = np.datetime64(pd.Timestamp(start).tz_localize(None), "ns")
        end64 = np.datetime64(pd.Timestamp(end).tz_localize(None), "ns")
        left = int(np.searchsorted(self.timestamp, start64, side="left"))
        right = int(np.searchsorted(self.timestamp, end64, side="left"))
        return IntrabarExecutionWindow(self, left, right)


@dataclass(frozen=True, slots=True)
class IntrabarExecutionWindow:
    """Zero-copy slice coordinates over :class:`IntrabarExecutionData`."""

    data: IntrabarExecutionData
    start: int
    stop: int

    @property
    def empty(self) -> bool:
        return self.start == self.stop

    @property
    def first_timestamp(self) -> pd.Timestamp | None:
        return None if self.empty else pd.Timestamp(self.data.timestamp[self.start])

    def gap_pairs(self, expected: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        values = self.data.timestamp[self.start:self.stop]
        gaps = np.flatnonzero(np.diff(values) > expected.to_timedelta64())
        return [(pd.Timestamp(values[i]), pd.Timestamp(values[i + 1])) for i in gaps]

    def rows(self):
        for i in range(self.start, self.stop):
            yield (
                i,
                pd.Timestamp(self.data.timestamp[i]),
                float(self.data.open[i]),
                float(self.data.high[i]),
                float(self.data.low[i]),
            )


def from_data_lake_bundle(bundle, config=None) -> tuple[PreparedBacktestFrame, IntrabarExecutionData | None]:
    """Bounded adapter from today's Data Lake bundle; no simulator routing."""
    strategy, technical, context = bundle.strategy, bundle.technical_features, bundle.context_features
    required_strategy = {
        "period_start", "available_at", "open", "high", "low", "close", "volume"
    }
    directional_names = {
        "atr", "atr_pct", "adx", "plus_di", "minus_di", "di_spread",
        "di_spread_1", "di_spread_3", "di_spread_5", "di_spread_change",
        "di_ratio", "plus_di_change", "minus_di_change",
        "di_pressure_spread_change", "long_directional_di_change",
        "long_opposing_di_change", "long_di_pressure_state",
        "short_directional_di_change", "short_opposing_di_change",
        "short_di_pressure_state",
    }
    required_technical = {"timestamp", "available_at", *directional_names}
    required_context = {
        "timestamp", "available_at", "bb_middle", "bb_upper", "bb_lower",
        "bb_width", "bb_width_pct", "bb_width_1", "bb_width_3", "bb_width_5",
        "bb_width_change", "bb_width_change_pct", "session_vwap",
        "close_location", "mean_reversion_mean", "mean_reversion_distance_atr",
        "mean_reversion_distance_atr_previous", "mean_reversion_sigma",
        "mean_reversion_bb_upper", "mean_reversion_bb_lower", "mean_reversion_bb_zscore",
        "mean_reversion_bb_location", "mean_reversion_rsi", "mean_reversion_rsi_state",
        "mean_reversion_long_reentry", "mean_reversion_short_reentry",
        "mean_reversion_reentry_confirmation", "mean_reversion_signal",
        "mean_reversion_signal_direction", "mean_reversion_setup_strength",
        "bb_reentry", "mr_signal", "mr_signal_direction",
        "mean_reversion_distance_change_atr", "mean_reversion_state",
        "mean_reversion_motion", "mean_reversion_strength",
        "mean_reversion_strength_label",
    }
    for label, frame, required in (
        ("strategy data", strategy, required_strategy),
        ("technical features", technical, required_technical),
        ("context features", context, required_context),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

    strategy_timestamps = _aligned_timestamps(
        strategy["period_start"], "strategy data period_start"
    )
    _require_exact_timeline(strategy_timestamps, technical["timestamp"], "technical features")
    _require_exact_timeline(strategy_timestamps, context["timestamp"], "context features")

    technical_available = _aligned_timestamps(
        technical["available_at"], "technical features available_at"
    )
    context_available = _aligned_timestamps(
        context["available_at"], "context features available_at"
    )
    available = np.maximum(technical_available, context_available)

    kwargs = {name: context[name].to_numpy() for name in (
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_width_pct",
        "bb_width_1", "bb_width_3", "bb_width_5", "bb_width_change",
        "bb_width_change_pct", "session_vwap", "close_location",
        "mean_reversion_mean", "mean_reversion_distance_atr",
        "mean_reversion_distance_atr_previous", "mean_reversion_sigma",
        "mean_reversion_bb_upper", "mean_reversion_bb_lower", "mean_reversion_bb_zscore",
        "mean_reversion_bb_location", "mean_reversion_rsi", "mean_reversion_rsi_state",
        "mean_reversion_long_reentry", "mean_reversion_short_reentry",
        "mean_reversion_reentry_confirmation", "mean_reversion_signal",
        "mean_reversion_signal_direction", "mean_reversion_setup_strength",
        "bb_reentry", "mr_signal", "mr_signal_direction",
        "mean_reversion_distance_change_atr", "mean_reversion_state",
        "mean_reversion_motion", "mean_reversion_strength", "mean_reversion_strength_label",
    )}

    research_blocks: list[ResearchContext] = []
    frames = dict(bundle.research_features)
    support_resistance = getattr(bundle, "support_resistance_features", None)
    if support_resistance is not None:
        frames["support_resistance"] = support_resistance
    for name, frame in sorted(frames.items()):
        missing = sorted({"timestamp", "available_at"} - set(frame.columns))
        if missing:
            raise ValueError(f"research feature {name} is missing required columns: {missing}")
        _require_exact_timeline(strategy_timestamps, frame["timestamp"], f"research feature {name}")
        research_blocks.append(
            ResearchContext(
                name,
                frame["available_at"].to_numpy(),
                {
                    column: frame[column].to_numpy()
                    for column in frame.columns
                    if column not in {"timestamp", "available_at"}
                },
            )
        )
    research = tuple(research_blocks)

    bull_return = np.full(len(strategy), np.nan)
    regimes = np.full(len(strategy), None, dtype=object)
    momentum: dict[int, np.ndarray] = {}
    if config is not None:
        from crypto_strategy_lab.features.market_regime import prepare_policy_market_features
        bull_return, regimes, momentum = prepare_policy_market_features(
            strategy_timestamps, strategy["close"].to_numpy(float), config,
            getattr(bundle, "structural_benchmark", None),
        )
    prepared = PreparedBacktestFrame(
        timestamp=strategy_timestamps,
        strategy_interval=pd.Timedelta(bundle.request.strategy_interval),
        open=strategy["open"].to_numpy(), high=strategy["high"].to_numpy(),
        low=strategy["low"].to_numpy(), close=strategy["close"].to_numpy(),
        volume=strategy["volume"].to_numpy(),
        **{name: technical[name].to_numpy() for name in directional_names},
        bull_regime_return=bull_return, market_regime=regimes,
        momentum_returns_by_hours=momentum,
        decision_available_at=available,
        research=research, **kwargs,
    )
    intrabar = None
    if bundle.intrabar is not None:
        intrabar = IntrabarExecutionData(
            bundle.intrabar["period_start"].to_numpy(),
            pd.Timedelta(bundle.request.intrabar_interval),
            bundle.intrabar["open"].to_numpy(), bundle.intrabar["high"].to_numpy(),
            bundle.intrabar["low"].to_numpy(),
        )
        intrabar.validate_compatible(prepared)
    return prepared, intrabar


def intrabar_from_data_lake_bundle(bundle) -> IntrabarExecutionData | None:
    """Project execution-resolution data without touching persistent L2/L3."""
    if bundle.intrabar is None:
        return None
    return IntrabarExecutionData(
        bundle.intrabar["period_start"].to_numpy(),
        pd.Timedelta(bundle.request.intrabar_interval),
        bundle.intrabar["open"].to_numpy(), bundle.intrabar["high"].to_numpy(),
        bundle.intrabar["low"].to_numpy(),
    )
