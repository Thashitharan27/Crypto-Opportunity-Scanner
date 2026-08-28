"""Cached causal daily features consumed by state-transition research."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.state_transition_research import (
    StateTransitionResearchConfig,
    daily_state_frame,
)

from .base import FeatureDefinition, ParameterDefinition


STATE_TRANSITION_DAILY_FEATURE_NAME = "state_transition_daily"
STATE_TRANSITION_DAILY_FEATURE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StateTransitionDailyFeatureProvider:
    """Prepare the daily regime/volatility frame once before simulation/reporting."""

    definition: FeatureDefinition = FeatureDefinition(
        name=STATE_TRANSITION_DAILY_FEATURE_NAME,
        version=STATE_TRANSITION_DAILY_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        parameters={
            "regime_lookback_days": ParameterDefinition(int, 20),
            "bull_return_threshold": ParameterDefinition(float, 0.05),
            "bear_return_threshold": ParameterDefinition(float, -0.05),
            "volatility_lookback_days": ParameterDefinition(int, 20),
            "volatility_reference_days": ParameterDefinition(int, 252),
            "volatility_low_quantile": ParameterDefinition(float, 0.33),
            "volatility_high_quantile": ParameterDefinition(float, 0.67),
            "di_bucket_edges": ParameterDefinition(tuple, (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, float("inf"))),
            "di_stable_tolerance": ParameterDefinition(float, 0.5),
            "minimum_state_observations": ParameterDefinition(int, 30),
            "minimum_trade_observations": ParameterDefinition(int, 20),
        },
        output_columns=(
            "date",
            "available_at",
            "close",
            "return_1d",
            "return_lookback",
            "regime_state",
            "volatility",
            "volatility_state",
        ),
        warmup_bars=0,
        availability_rule="daily_state_available_from_following_utc_midnight",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[DatasetKind, pd.DataFrame],
        parameters: Mapping[str, object],
    ) -> pd.DataFrame:
        try:
            source = datasets[DatasetKind.KLINES].copy()
        except KeyError as exc:
            raise ValueError("state_transition_daily requires canonical kline data") from exc
        required = {"period_start", "close"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")

        config = StateTransitionResearchConfig(
            regime_lookback_days=int(parameters.get("regime_lookback_days", 20)),
            bull_return_threshold=float(parameters.get("bull_return_threshold", 0.05)),
            bear_return_threshold=float(parameters.get("bear_return_threshold", -0.05)),
            volatility_lookback_days=int(parameters.get("volatility_lookback_days", 20)),
            volatility_reference_days=int(parameters.get("volatility_reference_days", 252)),
            volatility_low_quantile=float(parameters.get("volatility_low_quantile", 0.33)),
            volatility_high_quantile=float(parameters.get("volatility_high_quantile", 0.67)),
            di_bucket_edges=tuple(parameters.get(
                "di_bucket_edges",
                (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, float("inf")),
            )),
            di_stable_tolerance=float(parameters.get("di_stable_tolerance", 0.5)),
            minimum_state_observations=int(parameters.get("minimum_state_observations", 30)),
            minimum_trade_observations=int(parameters.get("minimum_trade_observations", 20)),
        )
        strategy = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(source["period_start"], utc=True),
                "close": pd.to_numeric(source["close"], errors="raise"),
            }
        )
        daily = daily_state_frame(strategy, config)
        daily["date"] = pd.to_datetime(daily["date"], utc=True)
        daily.insert(1, "available_at", daily["date"] + pd.Timedelta(days=1))
        if bool((daily["available_at"] <= daily["date"]).any()):
            raise ValueError("daily state-transition feature has invalid availability")
        daily.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                "request_cache_key": request.cache_key(),
                "regime_lookback_days": config.regime_lookback_days,
                "bull_return_threshold": config.bull_return_threshold,
                "bear_return_threshold": config.bear_return_threshold,
                "volatility_lookback_days": config.volatility_lookback_days,
                "volatility_reference_days": config.volatility_reference_days,
                "volatility_low_quantile": config.volatility_low_quantile,
                "volatility_high_quantile": config.volatility_high_quantile,
                "minimum_state_observations": config.minimum_state_observations,
                "minimum_trade_observations": config.minimum_trade_observations,
            }
        )
        return daily
