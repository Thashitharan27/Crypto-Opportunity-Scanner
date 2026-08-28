"""Causal scanner-only activity features (never trading signals)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from .base import FeatureDefinition, ParameterDefinition

OPPORTUNITY_ACTIVITY_FEATURE_NAME = "opportunity_activity"
OPPORTUNITY_ACTIVITY_FEATURE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class OpportunityActivityFeatureProvider:
    """Compute the four activity measures absent from Strategy Lab's registry.

    The 49-bar warmup comprises 48 quote-volume candles and the prior close
    needed for 24 log returns.  Missing quote volume stays missing.
    """

    definition: FeatureDefinition = FeatureDefinition(
        name=OPPORTUNITY_ACTIVITY_FEATURE_NAME,
        version=OPPORTUNITY_ACTIVITY_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        parameters={"window": ParameterDefinition(int, 24), "denominator_floor": ParameterDefinition(float, 1e-9)},
        output_columns=("realized_volatility", "recent_range_pct", "range_expansion", "volume_ratio"),
        warmup_bars=49,
        availability_rule="current_completed_kline_available_at",
    )

    def compute(self, request: DataRequest, datasets: Mapping[DatasetKind, pd.DataFrame],
                parameters: Mapping[str, object], feature_frames=None) -> pd.DataFrame:
        del feature_frames
        source = datasets.get(DatasetKind.KLINES)
        if source is None:
            raise ValueError("opportunity_activity requires canonical strategy klines")
        required = {"period_start", "available_at", "high", "low", "close"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")
        n = int(parameters.get("window", 24)); floor = float(parameters.get("denominator_floor", 1e-9))
        if n <= 0 or floor <= 0:
            raise ValueError("window and denominator_floor must be positive")
        frame = source.sort_values("period_start", kind="stable").drop_duplicates("period_start", keep="last").reset_index(drop=True)
        high = pd.to_numeric(frame.high, errors="coerce"); low = pd.to_numeric(frame.low, errors="coerce")
        close = pd.to_numeric(frame.close, errors="coerce")
        returns = np.log(close / close.shift(1))
        rv = returns.rolling(n, min_periods=n).std(ddof=0) * np.sqrt(n)
        recent_range = (high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min()) / close
        candle_range = (high - low) / close
        # Legacy parity: compare the current candle with the *older* preceding
        # block, not the immediately preceding 24 candles.
        preceding_median = candle_range.shift(n).rolling(n, min_periods=n).median().clip(lower=floor)
        expansion = candle_range / preceding_median
        if "quote_volume" in frame:
            quote = pd.to_numeric(frame.quote_volume, errors="coerce")
            latest = quote.rolling(n, min_periods=n).sum()
            previous = quote.shift(n).rolling(n, min_periods=n).sum().clip(lower=1.0)
            volume_ratio = latest / previous
        else:
            volume_ratio = pd.Series(np.nan, index=frame.index)
        output = pd.DataFrame({"timestamp": pd.to_datetime(frame.period_start, utc=True),
            "available_at": pd.to_datetime(frame.available_at, utc=True), "realized_volatility": rv,
            "recent_range_pct": recent_range, "range_expansion": expansion, "volume_ratio": volume_ratio})
        output.attrs.update(feature_name=self.definition.name, feature_version=self.definition.version,
                            effective_warmup_bars=2*n+1, request_cache_key=request.cache_key())
        return output
