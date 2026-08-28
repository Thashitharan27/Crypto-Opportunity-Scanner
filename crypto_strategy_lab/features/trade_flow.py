"""Causal research features derived exclusively from compact trade aggregates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from .base import FeatureDataResource, FeatureDefinition, OutputField, ParameterDefinition


TRADE_FLOW_FEATURE_NAME = "trade_flow_context"
TRADE_FLOW_FEATURE_VERSION = "2"


def trade_flow_resource(source: DatasetKind, interval: str = "1m") -> FeatureDataResource:
    if source not in {DatasetKind.AGG_TRADES, DatasetKind.TRADES}:
        raise ValueError("trade flow source must be agg_trades or trades")
    return FeatureDataResource(source, interval, "trade_flow_aggregate")


def _windows(value: object) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple)) else str(value).split(",")
    mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
    try:
        values = tuple(mapping[str(item).strip()] for item in raw)
    except KeyError as exc:
        raise ValueError("trade_flow_windows support only 1m,5m,15m,1h") from exc
    if not values or len(set(values)) != len(values):
        raise ValueError("trade_flow_windows must be non-empty and unique")
    return values


def _utc_ns(values: pd.Series) -> pd.Series:
    """Normalize UTC timestamps to one merge-compatible nanosecond dtype.

    Binance public archives can switch between millisecond and microsecond
    epochs. Pandas may preserve those as datetime64[ms/us, UTC], while strategy
    klines are commonly datetime64[ns, UTC]. merge_asof requires the units to
    match exactly even when the instants are equivalent.
    """

    return pd.to_datetime(values, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


_BASE_FIELDS = (
    "underlying_trade_count",
    "source_event_count",
    "base_volume",
    "quote_volume",
    "aggressive_buy_base_volume",
    "aggressive_sell_base_volume",
    "trade_delta_base",
)
_LARGE_FIELDS = (
    "large_source_event_count",
    "large_source_event_quote_volume",
    "large_buy_quote_volume",
    "large_sell_quote_volume",
)


def _schema(params):
    schema = {
        "trade_flow_source": OutputField("string", False),
        "trade_source_covered": OutputField("bool", False),
        "cvd_utc_day": OutputField("numeric"),
        "cvd_1h": OutputField("numeric"),
        "trade_intensity_change": OutputField("numeric"),
        "last_trade_event_at": OutputField("datetime"),
        "trade_event_age_seconds": OutputField("numeric"),
    }
    for minutes in _windows(params["trade_flow_windows"]):
        suffix = "1h" if minutes == 60 else f"{minutes}m"
        for name in (
            "trade_count",
            "source_event_count",
            "volume",
            "quote_volume",
            "aggressive_buy_volume",
            "aggressive_sell_volume",
            "trade_delta",
            "trade_delta_pct",
            "trade_intensity",
            "source_event_intensity",
            "average_trade_size",
            "average_source_event_size",
            "trade_vwap",
            "large_source_event_volume_share",
            "large_buy_share",
            "large_sell_share",
        ):
            schema[f"{name}_{suffix}"] = OutputField("numeric")
        if minutes == 1:
            # For TRADES this is the true individual-trade median. For
            # AGG_TRADES it is deliberately named as a source-event median.
            schema["median_source_event_size_1m"] = OutputField("numeric")
    return schema


@dataclass(frozen=True, slots=True)
class TradeFlowContextFeatureProvider:
    definition: FeatureDefinition = FeatureDefinition(
        name=TRADE_FLOW_FEATURE_NAME,
        version=TRADE_FLOW_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        parameters={
            "trade_flow_source": ParameterDefinition(str, "AGG_TRADES"),
            "trade_flow_windows": ParameterDefinition(
                lambda x: tuple(x)
                if isinstance(x, (list, tuple))
                else tuple(str(x).split(",")),
                ("1m", "5m", "15m", "1h"),
            ),
        },
        output_schema_factory=_schema,
        availability_rule="completed_1m_trade_aggregates_available_at_bucket_close",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[object, pd.DataFrame],
        parameters: Mapping[str, object],
        feature_frames=None,
    ) -> pd.DataFrame:
        del request, feature_frames
        source = DatasetKind[str(parameters["trade_flow_source"]).upper()]
        resource = trade_flow_resource(source)
        if resource not in datasets:
            raise ValueError(
                "trade_flow_context requires a compact trade_flow_aggregate resource"
            )
        agg = datasets[resource].copy()
        forbidden = {
            "agg_trade_id",
            "trade_id",
            "is_buyer_maker",
            "first_trade_id",
            "last_trade_id",
            "taker_side",
        }
        if forbidden & set(agg):
            raise ValueError("raw trade-event columns are forbidden in the trade-flow provider")

        required = {
            "period_start",
            "period_end",
            "available_at",
            "trade_flow_source_covered",
            "weighted_price_sum",
            "last_event_at",
            "median_source_event_size",
            *_BASE_FIELDS,
            *_LARGE_FIELDS,
        }
        missing = sorted(required - set(agg))
        if missing:
            raise ValueError(f"compact trade aggregate missing {missing}")

        for column in ("period_start", "period_end", "available_at", "last_event_at"):
            agg[column] = _utc_ns(agg[column])
        agg = agg.sort_values("available_at", kind="stable").reset_index(drop=True)
        if agg[["period_start", "period_end", "available_at"]].isna().any().any():
            raise ValueError("compact trade aggregate has malformed bucket timestamps")
        if agg["available_at"].duplicated().any():
            raise ValueError("compact trade aggregate available_at must be unique")

        covered = agg["trade_flow_source_covered"].astype(bool)
        facts = pd.DataFrame(
            {"available_at": agg["available_at"], "trade_source_covered": covered}
        )
        windows = _windows(parameters["trade_flow_windows"])

        numeric = {
            name: pd.to_numeric(agg[name], errors="coerce")
            for name in (*_BASE_FIELDS, *_LARGE_FIELDS, "weighted_price_sum")
        }
        for minutes in windows:
            suffix = "1h" if minutes == 60 else f"{minutes}m"
            valid = covered.rolling(minutes, min_periods=minutes).sum().eq(minutes)
            sums = {
                name: numeric[name]
                .rolling(minutes, min_periods=minutes)
                .sum()
                .where(valid)
                for name in _BASE_FIELDS
            }
            large = {
                name: numeric[name]
                .rolling(minutes, min_periods=minutes)
                .sum()
                .where(valid)
                for name in _LARGE_FIELDS
            }
            weighted = (
                numeric["weighted_price_sum"]
                .rolling(minutes, min_periods=minutes)
                .sum()
                .where(valid)
            )
            facts[f"trade_count_{suffix}"] = sums["underlying_trade_count"]
            facts[f"source_event_count_{suffix}"] = sums["source_event_count"]
            facts[f"volume_{suffix}"] = sums["base_volume"]
            facts[f"quote_volume_{suffix}"] = sums["quote_volume"]
            facts[f"aggressive_buy_volume_{suffix}"] = sums[
                "aggressive_buy_base_volume"
            ]
            facts[f"aggressive_sell_volume_{suffix}"] = sums[
                "aggressive_sell_base_volume"
            ]
            facts[f"trade_delta_{suffix}"] = sums["trade_delta_base"]
            facts[f"trade_delta_pct_{suffix}"] = sums["trade_delta_base"] / sums[
                "base_volume"
            ].replace(0, np.nan)
            facts[f"trade_vwap_{suffix}"] = weighted / sums["base_volume"].replace(
                0, np.nan
            )
            facts[f"average_trade_size_{suffix}"] = sums["base_volume"] / sums[
                "underlying_trade_count"
            ].replace(0, np.nan)
            facts[f"average_source_event_size_{suffix}"] = sums["base_volume"] / sums[
                "source_event_count"
            ].replace(0, np.nan)
            facts[f"trade_intensity_{suffix}"] = sums["underlying_trade_count"] / minutes
            facts[f"source_event_intensity_{suffix}"] = sums["source_event_count"] / minutes
            facts[f"large_source_event_volume_share_{suffix}"] = large[
                "large_source_event_quote_volume"
            ] / sums["quote_volume"].replace(0, np.nan)
            facts[f"large_buy_share_{suffix}"] = large["large_buy_quote_volume"] / large[
                "large_source_event_quote_volume"
            ].replace(0, np.nan)
            facts[f"large_sell_share_{suffix}"] = large["large_sell_quote_volume"] / large[
                "large_source_event_quote_volume"
            ].replace(0, np.nan)
            if minutes == 1:
                facts["median_source_event_size_1m"] = pd.to_numeric(
                    agg["median_source_event_size"], errors="coerce"
                ).where(covered)

        # CVD is deterministic with respect to request start. The aggregate
        # loader supplies the UTC-day prefix. If a source minute is unavailable,
        # later CVD values for that UTC day remain unknown rather than treating the
        # missing flow as zero.
        bucket_day = agg["period_start"].dt.floor("D")
        day_complete = covered.groupby(bucket_day).cummin().astype(bool)
        day_cvd = numeric["trade_delta_base"].where(covered).groupby(bucket_day).cumsum()
        day_cvd = day_cvd.where(day_complete)
        # The 23:59--00:00 bucket belongs to the previous UTC day. At exactly
        # 00:00 the new day's completed-bucket CVD is zero.
        crossed_midnight = agg["available_at"].dt.floor("D") != bucket_day
        day_cvd.loc[crossed_midnight & covered] = 0.0
        facts["cvd_utc_day"] = day_cvd

        valid60 = covered.rolling(60, min_periods=60).sum().eq(60)
        facts["cvd_1h"] = (
            numeric["trade_delta_base"]
            .rolling(60, min_periods=60)
            .sum()
            .where(valid60)
        )

        current_valid = covered.rolling(5, min_periods=5).sum().eq(5)
        current5 = (
            numeric["underlying_trade_count"].rolling(5, min_periods=5).sum() / 5
        ).where(current_valid)
        prior60 = (
            numeric["underlying_trade_count"]
            .shift(5)
            .rolling(60, min_periods=60)
            .sum()
            / 60
        )
        reference_valid = covered.shift(5).rolling(60, min_periods=60).sum().eq(60)
        facts["trade_intensity_change"] = (
            current5 / prior60.replace(0, np.nan) - 1
        ).where(current_valid & reference_valid)

        # Never carry a last-event timestamp across an unavailable source gap.
        segment = (~covered).cumsum()
        last_event = agg["last_event_at"].where(covered)
        facts["last_trade_event_at"] = last_event.groupby(segment).ffill().where(covered)
        facts["trade_event_age_seconds"] = (
            facts["available_at"] - facts["last_trade_event_at"]
        ).dt.total_seconds()
        facts["trade_flow_source"] = source.value

        klines = datasets[DatasetKind.KLINES].copy()
        required_kline = {"period_start", "available_at"}
        missing_kline = sorted(required_kline - set(klines))
        if missing_kline:
            raise ValueError(f"strategy klines missing {missing_kline}")
        klines["period_start"] = _utc_ns(klines["period_start"])
        klines["available_at"] = _utc_ns(klines["available_at"])
        klines = klines.sort_values("available_at", kind="stable")
        aligned = pd.merge_asof(
            pd.DataFrame(
                {
                    "timestamp": klines["period_start"],
                    "available_at": klines["available_at"],
                }
            ).sort_values("available_at"),
            facts.sort_values("available_at"),
            on="available_at",
            direction="backward",
        )
        aligned["trade_flow_source"] = aligned["trade_flow_source"].fillna(source.value)
        aligned["trade_source_covered"] = (
            aligned["trade_source_covered"].fillna(False).astype(bool)
        )
        return aligned.reset_index(drop=True)
