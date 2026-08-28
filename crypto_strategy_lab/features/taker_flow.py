"""Compact taker flow derived from completed Binance kline aggregates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.alignment import causal_asof_join
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from .base import FeatureDataResource, FeatureDefinition, ParameterDefinition

TAKER_FLOW_CONTEXT_FEATURE_NAME = "taker_flow_context"
TAKER_FLOW_CONTEXT_FEATURE_VERSION = "2"


def taker_flow_resource(interval: str = "5m") -> FeatureDataResource:
    return FeatureDataResource(DatasetKind.KLINES, interval, "taker_flow")


@dataclass(frozen=True, slots=True)
class TakerFlowContextFeatureProvider:
    definition: FeatureDefinition = FeatureDefinition(
        name=TAKER_FLOW_CONTEXT_FEATURE_NAME,
        version=TAKER_FLOW_CONTEXT_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        parameters={
            "taker_flow_interval": ParameterDefinition(str, "5m"),
            "volume_tolerance": ParameterDefinition(float, 1e-9),
        },
        output_columns=(
            "taker_source_available_at",
            "taker_age_seconds",
            "taker_buy_volume",
            "taker_sell_volume",
            "taker_buy_sell_ratio",
            "taker_delta",
            "taker_delta_pct",
            "taker_delta_15m",
            "taker_delta_pct_15m",
            "taker_delta_1h",
            "taker_delta_pct_1h",
            "flow_acceleration",
            "flow_persistence",
        ),
        availability_rule="completed_auxiliary_kline_available_at_or_before_strategy_decision",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[object, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames=None,
    ) -> pd.DataFrame:
        del feature_frames
        params = self.definition.normalize_parameters(parameters)
        interval = str(params["taker_flow_interval"])
        strategy = datasets[DatasetKind.KLINES].copy()
        resource = taker_flow_resource(interval)
        source = datasets.get(resource)
        if source is None:
            raise ValueError(f"taker_flow_context requires auxiliary {resource!r}")
        source = source.copy()

        required_strategy = {"period_start", "available_at"}
        missing_strategy = sorted(required_strategy - set(strategy.columns))
        if missing_strategy:
            raise ValueError(f"Strategy klines missing {missing_strategy}")
        required_source = {"available_at", "volume", "taker_buy_base_volume"}
        missing_source = sorted(required_source - set(source.columns))
        if missing_source:
            raise ValueError(f"Taker-flow klines missing {missing_source}")

        strategy = (
            strategy.sort_values("period_start", kind="stable")
            .drop_duplicates("period_start", keep="last")
            .reset_index(drop=True)
        )
        source["available_at"] = pd.to_datetime(source["available_at"], utc=True)
        source = (
            source.sort_values("available_at", kind="stable")
            .drop_duplicates("available_at", keep="last")
            .reset_index(drop=True)
        )

        total = pd.to_numeric(source["volume"], errors="coerce").to_numpy(float)
        buy = pd.to_numeric(
            source["taker_buy_base_volume"], errors="coerce"
        ).to_numpy(float)
        finite = np.isfinite(total) & np.isfinite(buy)
        if np.any(finite & ((total < 0) | (buy < 0))):
            raise ValueError("Taker-flow volume fields cannot be negative")
        tolerance = float(params["volume_tolerance"]) * np.maximum(1.0, np.abs(total))
        excess = buy - total
        if np.any(finite & (excess > tolerance)):
            raise ValueError("taker_buy_base_volume exceeds volume beyond tolerance")

        sell = total - buy
        # A tiny negative caused only by floating point tolerance is exactly zero;
        # material negative sell volume remains an integrity error above.
        tiny_negative = finite & (sell < 0) & (np.abs(sell) <= tolerance)
        sell[tiny_negative] = 0.0
        delta = buy - sell
        source["taker_buy_volume"] = buy
        source["taker_sell_volume"] = sell
        source["taker_buy_sell_ratio"] = np.divide(
            buy,
            sell,
            out=np.full(len(source), np.nan),
            where=np.isfinite(buy) & np.isfinite(sell) & (sell > 0),
        )
        source["taker_delta"] = delta
        source["taker_delta_pct"] = np.divide(
            delta,
            total,
            out=np.full(len(source), np.nan),
            where=np.isfinite(delta) & np.isfinite(total) & (total != 0),
        )

        indexed = pd.DataFrame(
            {"delta": delta, "volume": total},
            index=pd.DatetimeIndex(source["available_at"]),
        )
        rolling_frames: dict[str, pd.DataFrame] = {}
        for label, window in (("15m", "15min"), ("1h", "1h")):
            # Elapsed windows are (T-H, T]. On 5m data, 15m therefore means
            # exactly the three completed candles ending at T, not four.
            rolling = indexed.rolling(window, closed="right", min_periods=1).sum()
            rolling_frames[label] = rolling
            source[f"taker_delta_{label}"] = rolling["delta"].to_numpy()
            source[f"taker_delta_pct_{label}"] = np.divide(
                rolling["delta"].to_numpy(float),
                rolling["volume"].to_numpy(float),
                out=np.full(len(source), np.nan),
                where=rolling["volume"].to_numpy(float) != 0,
            )

        source["flow_acceleration"] = (
            source["taker_delta_15m"] - source["taker_delta_15m"].shift(1)
        )

        signs = np.sign(delta)
        sign_frame = pd.DataFrame(
            {
                "positive": (signs > 0).astype(float),
                "negative": (signs < 0).astype(float),
                "count": np.ones(len(source), dtype=float),
            },
            index=pd.DatetimeIndex(source["available_at"]),
        )
        sign_counts = sign_frame.rolling(
            "1h", closed="right", min_periods=1
        ).sum()
        aggregate_sign = np.sign(source["taker_delta_1h"].to_numpy(float))
        count = sign_counts["count"].to_numpy(float)
        persistence = np.full(len(source), np.nan)
        valid = (count >= 2) & (aggregate_sign != 0)
        positive = valid & (aggregate_sign > 0)
        negative = valid & (aggregate_sign < 0)
        persistence[positive] = (
            sign_counts["positive"].to_numpy(float)[positive] / count[positive]
        )
        persistence[negative] = (
            sign_counts["negative"].to_numpy(float)[negative] / count[negative]
        )
        source["flow_persistence"] = persistence

        decisions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(strategy["period_start"], utc=True),
                "decision_time": pd.to_datetime(strategy["available_at"], utc=True),
            }
        )
        facts = [
            name
            for name in self.definition.output_columns
            if name
            not in {
                "timestamp",
                "available_at",
                "taker_source_available_at",
                "taker_age_seconds",
            }
        ]
        joined = causal_asof_join(decisions, source[["available_at", *facts]])
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(joined["timestamp"], utc=True),
                "available_at": pd.to_datetime(joined["decision_time"], utc=True),
                "taker_source_available_at": pd.to_datetime(
                    joined["available_at"], utc=True
                ),
            }
        )
        out["taker_age_seconds"] = (
            out["available_at"] - out["taker_source_available_at"]
        ).dt.total_seconds()
        for column in facts:
            out[column] = pd.to_numeric(joined[column], errors="coerce")

        source_available = out["taker_source_available_at"]
        if bool((source_available.notna() & (source_available > out["available_at"])).any()):
            raise AssertionError("Taker flow attached a future auxiliary kline")

        out.attrs.update(
            feature_name=self.definition.name,
            feature_version=self.definition.version,
            effective_warmup_bars=0,
            request_cache_key=request.cache_key(),
            source_interval=interval,
        )
        return out
