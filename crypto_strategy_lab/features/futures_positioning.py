"""Causal, source-native Binance futures positioning research facts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.alignment import causal_asof_join
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from .base import FeatureDataResource, FeatureDefinition, ParameterDefinition

FUTURES_POSITIONING_FEATURE_NAME = "futures_positioning"
FUTURES_POSITIONING_FEATURE_VERSION = "4"
FUTURES_POSITIONING_PRICE_INTERVAL = "1h"
RATIOS = (
    "top_trader_account_long_short_ratio",
    "top_trader_position_long_short_ratio",
    "global_long_short_account_ratio",
    "taker_long_short_volume_ratio",
)


def futures_positioning_price_resource(
    interval: str = FUTURES_POSITIONING_PRICE_INTERVAL,
) -> FeatureDataResource:
    """Auxiliary completed-kline source used for the genuine 1h price change."""
    return FeatureDataResource(DatasetKind.KLINES, interval, FUTURES_POSITIONING_FEATURE_NAME)


def _elapsed_change(
    frame: pd.DataFrame,
    column: str,
    horizon: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare each observation with the last observation at or before T-H."""
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    times = pd.DatetimeIndex(pd.to_datetime(frame["available_at"], utc=True)).asi8
    prior_i = np.searchsorted(times, times - horizon.value, side="right") - 1
    prior = np.full(len(frame), np.nan)
    valid = prior_i >= 0
    prior[valid] = values[prior_i[valid]]
    finite = np.isfinite(values) & np.isfinite(prior)
    change = np.full(len(frame), np.nan)
    change[finite] = values[finite] - prior[finite]
    pct = np.divide(
        change,
        prior,
        out=np.full(len(frame), np.nan),
        where=finite & (prior != 0),
    )
    return change, pct


def _time_zscore(
    frame: pd.DataFrame,
    column: str,
    days: float,
    minimum: int,
) -> np.ndarray:
    series = pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(frame["available_at"], utc=True)),
    )
    rolling = series.rolling(f"{days}D", min_periods=minimum)
    std = rolling.std(ddof=0)
    return ((series - rolling.mean()) / std.where(std > 0)).to_numpy(float)


def _state(price: np.ndarray, oi: np.ndarray) -> np.ndarray:
    out = np.full(len(price), "UNKNOWN", object)
    finite = np.isfinite(price) & np.isfinite(oi)
    out[finite] = "FLAT_OR_MIXED"
    out[finite & (price > 0) & (oi > 0)] = "PRICE_UP_OI_UP"
    out[finite & (price > 0) & (oi < 0)] = "PRICE_UP_OI_DOWN"
    out[finite & (price < 0) & (oi > 0)] = "PRICE_DOWN_OI_UP"
    out[finite & (price < 0) & (oi < 0)] = "PRICE_DOWN_OI_DOWN"
    return out


@dataclass(frozen=True, slots=True)
class FuturesPositioningFeatureProvider:
    definition: FeatureDefinition = FeatureDefinition(
        name=FUTURES_POSITIONING_FEATURE_NAME,
        version=FUTURES_POSITIONING_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES, DatasetKind.FUTURES_METRICS),
        parameters={
            "oi_zscore_window_days": ParameterDefinition(float, 7.0),
            "oi_zscore_min_samples": ParameterDefinition(int, 20),
        },
        output_columns=(
            "metrics_source_available_at",
            "metrics_age_seconds",
            "price_source_available_at",
            "price_age_seconds",
            "open_interest",
            "open_interest_value",
            "oi_change_5m",
            "oi_change_pct_5m",
            "oi_change_1h",
            "oi_change_pct_1h",
            "oi_change_24h",
            "oi_change_pct_24h",
            "oi_zscore_7d",
            "price_change_pct_1h",
            "oi_vs_price_state_1h",
            "open_interest_change_1bar_pct",
            "open_interest_change_3bar_pct",
            "open_interest_value_change_1bar_pct",
            "price_return_1bar",
            "price_oi_state",
            *RATIOS,
            "top_trader_account_bias",
            "top_trader_position_bias",
            "global_long_short_account_bias",
            "taker_long_short_volume_bias",
        ),
        availability_rule="source_native_metrics_and_1h_price_then_available_at_asof_strategy_decision",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[object, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames=None,
    ) -> pd.DataFrame:
        del feature_frames
        strategy = datasets[DatasetKind.KLINES].copy()
        metrics = datasets[DatasetKind.FUTURES_METRICS].copy()
        if not {"period_start", "available_at", "close"} <= set(strategy):
            raise ValueError(
                "Canonical kline frame requires period_start, available_at and close"
            )
        if "available_at" not in metrics or "open_interest" not in metrics:
            raise ValueError(
                "Canonical futures metrics frame requires available_at and open_interest"
            )

        params = self.definition.normalize_parameters(parameters)
        strategy = (
            strategy.sort_values("period_start", kind="stable")
            .drop_duplicates("period_start", keep="last")
            .reset_index(drop=True)
        )
        decisions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(strategy["period_start"], utc=True),
                "decision_time": pd.to_datetime(strategy["available_at"], utc=True),
                "close": pd.to_numeric(strategy["close"], errors="coerce"),
            }
        )

        metrics["available_at"] = pd.to_datetime(metrics["available_at"], utc=True)
        metrics = (
            metrics.sort_values("available_at", kind="stable")
            .drop_duplicates("available_at", keep="last")
            .reset_index(drop=True)
        )
        for column in ("open_interest", "open_interest_value", *RATIOS):
            if column not in metrics:
                metrics[column] = np.nan
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
        for label, delta in (
            ("5m", pd.Timedelta(minutes=5)),
            ("1h", pd.Timedelta(hours=1)),
            ("24h", pd.Timedelta(hours=24)),
        ):
            (
                metrics[f"oi_change_{label}"],
                metrics[f"oi_change_pct_{label}"],
            ) = _elapsed_change(metrics, "open_interest", delta)
        metrics["oi_zscore_7d"] = _time_zscore(
            metrics,
            "open_interest",
            float(params["oi_zscore_window_days"]),
            int(params["oi_zscore_min_samples"]),
        )

        joined = causal_asof_join(decisions, metrics)
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(joined["timestamp"], utc=True),
                "available_at": pd.to_datetime(joined["decision_time"], utc=True),
                "metrics_source_available_at": pd.to_datetime(
                    joined["available_at"], utc=True
                ),
            }
        )
        out["metrics_age_seconds"] = (
            out["available_at"] - out["metrics_source_available_at"]
        ).dt.total_seconds()
        for column in (
            "open_interest",
            "open_interest_value",
            *RATIOS,
            "oi_change_5m",
            "oi_change_pct_5m",
            "oi_change_1h",
            "oi_change_pct_1h",
            "oi_change_24h",
            "oi_change_pct_24h",
            "oi_zscore_7d",
        ):
            out[column] = pd.to_numeric(joined[column], errors="coerce")

        price_resource = futures_positioning_price_resource()
        price_source = datasets.get(price_resource)
        if price_source is None or price_source.empty:
            out["price_source_available_at"] = pd.Series(
                pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
            )
            out["price_age_seconds"] = np.nan
            out["price_change_pct_1h"] = np.nan
        else:
            price = price_source.copy()
            required_price = {"available_at", "close"}
            missing_price = sorted(required_price - set(price.columns))
            if missing_price:
                raise ValueError(
                    f"Canonical 1h positioning price source is missing {missing_price}"
                )
            price["available_at"] = pd.to_datetime(price["available_at"], utc=True)
            price = (
                price.sort_values("available_at", kind="stable")
                .drop_duplicates("available_at", keep="last")
                .reset_index(drop=True)
            )
            price["close"] = pd.to_numeric(price["close"], errors="coerce")
            _, price["price_change_pct_1h"] = _elapsed_change(
                price, "close", pd.Timedelta(hours=1)
            )
            price_joined = causal_asof_join(
                decisions,
                price[["available_at", "price_change_pct_1h"]],
            )
            out["price_source_available_at"] = pd.to_datetime(
                price_joined["available_at"], utc=True
            )
            out["price_age_seconds"] = (
                out["available_at"] - out["price_source_available_at"]
            ).dt.total_seconds()
            out["price_change_pct_1h"] = pd.to_numeric(
                price_joined["price_change_pct_1h"], errors="coerce"
            )

        out["oi_vs_price_state_1h"] = _state(
            out["price_change_pct_1h"].to_numpy(float),
            out["oi_change_pct_1h"].to_numpy(float),
        )

        # Retained research fields with explicit strategy-bar semantics.
        oi = out["open_interest"]
        oi_value = out["open_interest_value"]
        strategy_close = decisions["close"].to_numpy(float)
        out["open_interest_change_1bar_pct"] = oi.pct_change(fill_method=None)
        out["open_interest_change_3bar_pct"] = oi.pct_change(3, fill_method=None)
        out["open_interest_value_change_1bar_pct"] = oi_value.pct_change(
            fill_method=None
        )
        out["price_return_1bar"] = pd.Series(strategy_close).pct_change(
            fill_method=None
        )
        out["price_oi_state"] = _state(
            out["price_return_1bar"].to_numpy(float),
            out["open_interest_change_1bar_pct"].to_numpy(float),
        )
        for ratio, bias in zip(
            RATIOS,
            (
                "top_trader_account_bias",
                "top_trader_position_bias",
                "global_long_short_account_bias",
                "taker_long_short_volume_bias",
            ),
        ):
            out[bias] = out[ratio] - 1.0

        metrics_source = out["metrics_source_available_at"]
        if bool((metrics_source.notna() & (metrics_source > out["available_at"])).any()):
            raise AssertionError("Futures positioning attached a future metrics snapshot")
        price_source_at = out["price_source_available_at"]
        if bool((price_source_at.notna() & (price_source_at > out["available_at"])).any()):
            raise AssertionError("Futures positioning attached a future 1h price candle")

        out.attrs.update(
            feature_name=self.definition.name,
            feature_version=self.definition.version,
            effective_warmup_bars=0,
            request_cache_key=request.cache_key(),
            price_interval=FUTURES_POSITIONING_PRICE_INTERVAL,
        )
        return out
