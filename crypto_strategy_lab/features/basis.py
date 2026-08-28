"""Causal futures basis context from Binance mark/index/premium klines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.alignment import causal_asof_join
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind

from .base import FeatureDefinition, ParameterDefinition


BASIS_CONTEXT_FEATURE_NAME = "basis_context"
BASIS_CONTEXT_FEATURE_VERSION = "3"


def _basis_state(values: np.ndarray, neutral_bps: float = 1.0) -> np.ndarray:
    bps = values * 10000.0
    state = np.full(len(values), "UNKNOWN", dtype=object)
    finite = np.isfinite(bps)
    state[finite] = "NEUTRAL"
    state[finite & (bps > neutral_bps)] = "POSITIVE"
    state[finite & (bps < -neutral_bps)] = "NEGATIVE"
    return state


def _relative(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.divide(
        left - right,
        right,
        out=np.full(len(left), np.nan, dtype=float),
        where=np.isfinite(left) & np.isfinite(right) & (right != 0),
    )


def _reference_source(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            {
                "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
                f"{prefix}_price": pd.Series(dtype=float),
            }
        )
    required = {"available_at", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical {prefix} kline frame is missing columns: {missing}")
    result = frame[["available_at", "close"]].copy()
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    result[f"{prefix}_price"] = pd.to_numeric(result.pop("close"), errors="raise")
    return (
        result.sort_values("available_at", kind="stable")
        .drop_duplicates("available_at", keep="last")
        .reset_index(drop=True)
    )


def _time_zscore(
    values: pd.Series,
    available_at: pd.Series,
    *,
    days: float,
    minimum: int,
) -> np.ndarray:
    series = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(available_at, utc=True)),
    )
    rolling = series.rolling(f"{days}D", min_periods=minimum)
    std = rolling.std(ddof=0)
    return ((series - rolling.mean()) / std.where(std > 0)).to_numpy(float)


@dataclass(frozen=True, slots=True)
class BasisContextFeatureProvider:
    """Attach only reference-price facts available by strategy candle close."""

    definition: FeatureDefinition = FeatureDefinition(
        name=BASIS_CONTEXT_FEATURE_NAME,
        version=BASIS_CONTEXT_FEATURE_VERSION,
        required_datasets=(
            DatasetKind.KLINES,
            DatasetKind.MARK_PRICE_KLINES,
            DatasetKind.INDEX_PRICE_KLINES,
        ),
        optional_datasets=(DatasetKind.PREMIUM_INDEX_KLINES,),
        parameters={
            "basis_zscore_window_days": ParameterDefinition(float, 7.0),
            "basis_zscore_min_samples": ParameterDefinition(int, 5),
        },
        output_columns=(
            "mark_source_available_at",
            "mark_age_seconds",
            "mark_price",
            "index_source_available_at",
            "index_age_seconds",
            "index_price",
            "premium_source_available_at",
            "premium_age_seconds",
            "premium_index_close",
            "mark_index_basis",
            "mark_index_basis_bps",
            "mark_index_basis_state",
            "trade_mark_basis",
            "trade_mark_basis_bps",
            "trade_index_basis",
            "trade_index_basis_bps",
            "mark_index_basis_change",
            "mark_index_basis_zscore_7d",
            "premium_index_change",
            "premium_index_zscore_7d",
        ),
        warmup_bars=0,
        availability_rule="source_native_reference_facts_available_at_or_before_strategy_candle_close",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[DatasetKind, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        del feature_frames
        params = self.definition.normalize_parameters(parameters)
        try:
            klines = datasets[DatasetKind.KLINES].copy()
            mark = datasets[DatasetKind.MARK_PRICE_KLINES].copy()
            index = datasets[DatasetKind.INDEX_PRICE_KLINES].copy()
        except KeyError as exc:
            raise ValueError(
                "basis_context requires klines, mark_price_klines and index_price_klines"
            ) from exc
        premium = datasets.get(DatasetKind.PREMIUM_INDEX_KLINES, pd.DataFrame()).copy()

        required = {"period_start", "available_at", "close"}
        missing = sorted(required - set(klines.columns))
        if missing:
            raise ValueError(f"Canonical kline frame is missing columns: {missing}")
        if klines.empty:
            raise ValueError("Cannot align basis context to an empty kline frame")

        klines = (
            klines.sort_values("period_start", kind="stable")
            .drop_duplicates("period_start", keep="last")
            .reset_index(drop=True)
        )
        decisions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(klines["period_start"], utc=True),
                "decision_time": pd.to_datetime(klines["available_at"], utc=True),
                "trade_price": pd.to_numeric(klines["close"], errors="raise"),
            }
        )

        mark_source = _reference_source(mark, prefix="mark")
        index_source = _reference_source(index, prefix="index")
        if mark_source.empty or index_source.empty:
            raise ValueError("basis_context requires non-empty mark and index price sources")

        # Mark/index basis is calculated on the native mark timeline. The index
        # snapshot is joined backward to each completed mark candle before any
        # strategy-time sampling, so changes/z-scores never become 4h-lag aliases
        # merely because the strategy happens to run at 4h.
        native_decisions = pd.DataFrame(
            {
                "timestamp": mark_source["available_at"],
                "decision_time": mark_source["available_at"],
            }
        )
        index_on_mark = causal_asof_join(native_decisions, index_source)
        native_basis = pd.DataFrame(
            {
                "available_at": mark_source["available_at"],
                "mark_source_available_at": mark_source["available_at"],
                "mark_price": mark_source["mark_price"],
                "index_source_available_at": pd.to_datetime(
                    index_on_mark["available_at"], utc=True
                ),
                "index_price": pd.to_numeric(
                    index_on_mark["index_price"], errors="coerce"
                ),
            }
        )
        native_basis["mark_index_basis"] = _relative(
            native_basis["mark_price"].to_numpy(float),
            native_basis["index_price"].to_numpy(float),
        )
        native_basis["mark_index_basis_bps"] = (
            native_basis["mark_index_basis"] * 10000.0
        )
        native_basis["mark_index_basis_change"] = native_basis[
            "mark_index_basis"
        ].diff()
        native_basis["mark_index_basis_zscore_7d"] = _time_zscore(
            native_basis["mark_index_basis"],
            native_basis["available_at"],
            days=float(params["basis_zscore_window_days"]),
            minimum=int(params["basis_zscore_min_samples"]),
        )

        aligned_basis = causal_asof_join(decisions, native_basis)

        premium_source = _reference_source(premium, prefix="premium")
        if premium_source.empty:
            premium_source_at = pd.Series(
                pd.NaT, index=decisions.index, dtype="datetime64[ns, UTC]"
            )
            premium_close = pd.Series(np.nan, index=decisions.index, dtype=float)
            premium_change = pd.Series(np.nan, index=decisions.index, dtype=float)
            premium_zscore = pd.Series(np.nan, index=decisions.index, dtype=float)
        else:
            premium_source["premium_index_change"] = premium_source[
                "premium_price"
            ].diff()
            premium_source["premium_index_zscore_7d"] = _time_zscore(
                premium_source["premium_price"],
                premium_source["available_at"],
                days=float(params["basis_zscore_window_days"]),
                minimum=int(params["basis_zscore_min_samples"]),
            )
            premium_aligned = causal_asof_join(
                decisions,
                premium_source[
                    [
                        "available_at",
                        "premium_price",
                        "premium_index_change",
                        "premium_index_zscore_7d",
                    ]
                ],
            )
            premium_source_at = pd.to_datetime(
                premium_aligned["available_at"], utc=True
            )
            premium_close = pd.to_numeric(
                premium_aligned["premium_price"], errors="coerce"
            )
            premium_change = pd.to_numeric(
                premium_aligned["premium_index_change"], errors="coerce"
            )
            premium_zscore = pd.to_numeric(
                premium_aligned["premium_index_zscore_7d"], errors="coerce"
            )

        output = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(decisions["timestamp"], utc=True),
                "available_at": pd.to_datetime(decisions["decision_time"], utc=True),
                "mark_source_available_at": pd.to_datetime(
                    aligned_basis["mark_source_available_at"], utc=True
                ),
                "mark_price": pd.to_numeric(
                    aligned_basis["mark_price"], errors="coerce"
                ),
                "index_source_available_at": pd.to_datetime(
                    aligned_basis["index_source_available_at"], utc=True
                ),
                "index_price": pd.to_numeric(
                    aligned_basis["index_price"], errors="coerce"
                ),
                "premium_source_available_at": premium_source_at,
                "premium_index_close": premium_close,
                "mark_index_basis": pd.to_numeric(
                    aligned_basis["mark_index_basis"], errors="coerce"
                ),
                "mark_index_basis_bps": pd.to_numeric(
                    aligned_basis["mark_index_basis_bps"], errors="coerce"
                ),
                "mark_index_basis_change": pd.to_numeric(
                    aligned_basis["mark_index_basis_change"], errors="coerce"
                ),
                "mark_index_basis_zscore_7d": pd.to_numeric(
                    aligned_basis["mark_index_basis_zscore_7d"], errors="coerce"
                ),
                "premium_index_change": premium_change,
                "premium_index_zscore_7d": premium_zscore,
            }
        )
        for prefix in ("mark", "index", "premium"):
            output[f"{prefix}_age_seconds"] = (
                output["available_at"] - output[f"{prefix}_source_available_at"]
            ).dt.total_seconds()

        trade = decisions["trade_price"].to_numpy(float)
        mark_price = output["mark_price"].to_numpy(float)
        index_price = output["index_price"].to_numpy(float)
        output["mark_index_basis_state"] = _basis_state(
            output["mark_index_basis"].to_numpy(float)
        )
        output["trade_mark_basis"] = _relative(trade, mark_price)
        output["trade_mark_basis_bps"] = output["trade_mark_basis"] * 10000.0
        output["trade_index_basis"] = _relative(trade, index_price)
        output["trade_index_basis_bps"] = output["trade_index_basis"] * 10000.0

        for prefix in ("mark", "index", "premium"):
            source = output[f"{prefix}_source_available_at"]
            leak = source.notna() & (source > output["available_at"])
            if bool(leak.any()):
                raise AssertionError(f"Basis context attached a future {prefix} candle")

        output.attrs.update(
            {
                "feature_name": self.definition.name,
                "feature_version": self.definition.version,
                "effective_warmup_bars": self.definition.warmup_bars,
                "request_cache_key": request.cache_key(),
            }
        )
        return output
