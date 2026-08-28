"""Causal market-regime features computed outside the execution simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind

from .base import FeatureDefinition, OutputField, ParameterDefinition


POLICY_MARKET_FEATURE_NAME = "policy_market_context"
POLICY_MARKET_FEATURE_VERSION = "1"


def _normalize_market_regime_method(value: object) -> str:
    method = str(value).upper()
    if method not in {"ASSET_RETURN", "BTC_STRUCTURAL", "ASSET_STRUCTURAL"}:
        raise ValueError(f"unsupported market regime method {value!r}")
    return method


def _normalize_momentum_lookback_hours(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        values = [item.strip() for item in str(value).split(",") if item.strip()]
    elif isinstance(value, (int, float, np.integer, np.floating)):
        values = [value]
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("momentum lookbacks must be a number or iterable") from exc
    normalized = tuple(sorted({int(item) for item in values}))
    if not normalized or any(item <= 0 for item in normalized):
        raise ValueError("momentum lookbacks must contain positive hours")
    return normalized


def _policy_output_schema(parameters: Mapping[str, object]) -> Mapping[str, OutputField]:
    return {
        f"momentum_return_{hours}h": OutputField("numeric")
        for hours in parameters["momentum_lookback_hours"]  # type: ignore[index]
    }


STRUCTURAL_REGIME_DEFINITION = FeatureDefinition(
    name="structural_market_regime",
    version="1",
    required_datasets=(DatasetKind.KLINES,),
    output_columns=("market_regime",),
    availability_rule="completed_utc_day_available_next_midnight",
)


def causal_trailing_return(strategy_times, close, delta: pd.Timedelta) -> np.ndarray:
    """Close return against the latest candle at or before ``time - delta``."""
    times = pd.DatetimeIndex(pd.to_datetime(strategy_times, utc=True))
    values = np.asarray(close, dtype=float)
    result = np.full(len(times), np.nan)
    prior = np.searchsorted(times.asi8, (times - delta).asi8, side="right") - 1
    valid = prior >= 0
    result[valid] = values[valid] / values[prior[valid]] - 1.0
    return result


def prepare_policy_market_features(strategy_times, close, config, benchmark=None):
    """Compatibility-shaped adapter whose implementation is registry authoritative."""
    from .registry import FeatureRegistry

    times = pd.DatetimeIndex(pd.to_datetime(strategy_times, utc=True, errors="raise"))
    closes = np.asarray(close, dtype=float)
    if len(times) != len(closes):
        raise ValueError("policy market timestamps and close values are not aligned")
    if not len(times):
        return np.array([], dtype=float), np.array([], dtype=object), {}

    strategy_minutes = int(getattr(config, "strategy_timeframe_minutes", 0) or 0)
    if strategy_minutes <= 0:
        raise ValueError("strategy_timeframe_minutes must be positive for policy features")
    interval = pd.Timedelta(minutes=strategy_minutes)
    source = pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + interval,
            "close": closes,
        }
    )
    request = DataRequest(
        symbol=str(getattr(config, "market_symbol", "POLICY")),
        start=times[0].to_pydatetime(),
        end=(times[-1] + interval).to_pydatetime(),
        strategy_interval=f"{strategy_minutes}m",
    )
    lookbacks = tuple(
        sorted(
            {
                int(profile.momentum_lookback_hours)
                for profile in config.strategy_profiles.values()
            }
        )
    )
    parameters = {
        POLICY_MARKET_FEATURE_NAME: {
            "market_regime_method": config.market_regime_method,
            "bull_regime_lookback_days": config.bull_regime_lookback_days,
            "bull_regime_return_threshold": config.bull_regime_return_threshold,
            "structural_regime_sma_days": config.structural_regime_sma_days,
            "structural_regime_slope_lookback_days": config.structural_regime_slope_lookback_days,
            "momentum_lookback_hours": lookbacks,
        }
    }
    registry = FeatureRegistry()
    registry.register(PolicyMarketFeatureProvider(structural_benchmark=benchmark))
    frame = registry.execute(
        [POLICY_MARKET_FEATURE_NAME],
        request,
        {DatasetKind.KLINES: source},
        parameters=parameters,
    )[POLICY_MARKET_FEATURE_NAME]
    momentum = {
        hours: frame[f"momentum_return_{hours}h"].to_numpy(float)
        for hours in lookbacks
    }
    return (
        frame["bull_regime_return"].to_numpy(float),
        frame["market_regime"].to_numpy(dtype=object),
        momentum,
    )


@dataclass(frozen=True, slots=True)
class PolicyMarketFeatureProvider:
    """Prepare asset/structural regime and policy momentum through the registry."""

    structural_benchmark: pd.DataFrame | None = None

    definition: FeatureDefinition = FeatureDefinition(
        name=POLICY_MARKET_FEATURE_NAME,
        version=POLICY_MARKET_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        parameters={
            "market_regime_method": ParameterDefinition(
                _normalize_market_regime_method, "ASSET_RETURN"
            ),
            "bull_regime_lookback_days": ParameterDefinition(int, 90),
            "bull_regime_return_threshold": ParameterDefinition(float, 0.20),
            "structural_regime_sma_days": ParameterDefinition(int, 200),
            "structural_regime_slope_lookback_days": ParameterDefinition(int, 30),
            "momentum_lookback_hours": ParameterDefinition(
                _normalize_momentum_lookback_hours, (24,)
            ),
        },
        output_schema={
            "bull_regime_return": OutputField("numeric"),
            "market_regime": OutputField("string"),
        },
        output_schema_factory=_policy_output_schema,
        warmup_bars=0,
        availability_rule=(
            "completed_strategy_candle; structural_daily_state_available_following_utc_midnight"
        ),
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[DatasetKind, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        del feature_frames
        try:
            source = datasets[DatasetKind.KLINES].copy()
        except KeyError as exc:
            raise ValueError("policy_market_context requires canonical strategy klines") from exc
        required = {"period_start", "available_at", "close"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical policy kline frame is missing columns: {missing}")
        if source.empty:
            raise ValueError("Cannot prepare policy market features from an empty frame")

        source = source.sort_values("period_start", kind="stable").drop_duplicates(
            "period_start", keep="last"
        ).reset_index(drop=True)
        strategy_times = pd.to_datetime(source["period_start"], utc=True, errors="raise")
        available_at = pd.to_datetime(source["available_at"], utc=True, errors="raise")
        close = pd.to_numeric(source["close"], errors="raise").to_numpy(float)

        bull_days = int(parameters["bull_regime_lookback_days"])
        if bull_days <= 0:
            raise ValueError("bull_regime_lookback_days must be positive")
        bull = causal_trailing_return(
            strategy_times, close, pd.Timedelta(days=bull_days)
        )

        method = str(parameters["market_regime_method"])
        if method == "ASSET_RETURN":
            threshold = abs(float(parameters["bull_regime_return_threshold"]))
            regime = np.array(
                [
                    None if not np.isfinite(value) else
                    ("BULL" if value >= threshold else "BEAR" if value <= -threshold else "SIDEWAYS")
                    for value in bull
                ],
                dtype=object,
            )
        else:
            regime = structural_regime_values(
                strategy_times,
                self.structural_benchmark,
                sma_days=int(parameters["structural_regime_sma_days"]),
                slope_lookback_days=int(parameters["structural_regime_slope_lookback_days"]),
            )

        output = pd.DataFrame(
            {
                "timestamp": strategy_times,
                "available_at": available_at,
                "bull_regime_return": bull,
                "market_regime": regime,
            }
        )
        for hours in parameters["momentum_lookback_hours"]:  # type: ignore[index]
            output[f"momentum_return_{hours}h"] = causal_trailing_return(
                strategy_times,
                close,
                pd.Timedelta(hours=int(hours)),
            )
        output.attrs.update(
            feature_name=self.definition.name,
            feature_version=self.definition.version,
            request_cache_key=request.cache_key(),
        )
        return output


def _benchmark_timestamp_column(frame: pd.DataFrame) -> str:
    for column in ("period_start", "timestamp", "open_time", "time", "datetime", "date"):
        if column in frame.columns:
            return column
    raise ValueError("Structural regime benchmark requires a timestamp column")


def structural_regime_values(
    strategy_times,
    benchmark: pd.DataFrame,
    *,
    sma_days: int,
    slope_lookback_days: int,
) -> np.ndarray:
    """Map a completed-daily structural regime to strategy timestamps causally.

    The benchmark may be a canonical Data Lake kline frame (``period_start``)
    or a legacy-like OHLCV frame (``timestamp``). The close of UTC day D is not
    usable until 00:00 UTC on day D+1. Mapping is therefore always backward from
    each strategy timestamp to the latest already-available daily state.
    """

    if sma_days < 2:
        raise ValueError("sma_days must be at least 2")
    if slope_lookback_days < 1:
        raise ValueError("slope_lookback_days must be positive")
    if benchmark is None or benchmark.empty:
        raise ValueError("Structural regime benchmark is empty")
    if "close" not in benchmark.columns:
        raise ValueError("Structural regime benchmark requires close")

    time_col = _benchmark_timestamp_column(benchmark)
    source = benchmark[[time_col, "close"]].copy()
    source["timestamp"] = pd.to_datetime(source[time_col], utc=True, errors="coerce")
    source["close"] = pd.to_numeric(source["close"], errors="coerce")
    source = (
        source.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
    )
    if source.empty:
        raise ValueError("Structural regime benchmark has no valid rows")

    daily = source.set_index("timestamp")["close"].resample("1D").last().dropna().to_frame()
    daily["sma"] = daily["close"].rolling(sma_days, min_periods=sma_days).mean()
    daily["prior_sma"] = daily["sma"].shift(slope_lookback_days)
    daily["market_regime"] = np.where(
        (daily["close"] > daily["sma"]) & (daily["sma"] > daily["prior_sma"]),
        "BULL",
        np.where(
            (daily["close"] < daily["sma"]) & (daily["sma"] < daily["prior_sma"]),
            "BEAR",
            "SIDEWAYS",
        ),
    )
    daily.loc[daily[["sma", "prior_sma"]].isna().any(axis=1), "market_regime"] = None

    available = daily.reset_index()[["timestamp", "market_regime"]]
    available["available_at"] = available["timestamp"] + pd.Timedelta(days=1)
    available = available[["available_at", "market_regime"]].sort_values("available_at")

    target = pd.DataFrame({"strategy_time": pd.to_datetime(strategy_times, utc=True, errors="raise")})
    target["_order"] = np.arange(len(target))
    mapped = pd.merge_asof(
        target.sort_values("strategy_time"),
        available,
        left_on="strategy_time",
        right_on="available_at",
        direction="backward",
    ).sort_values("_order")
    return mapped["market_regime"].to_numpy(dtype=object)
