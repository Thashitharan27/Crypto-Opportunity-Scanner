"""Causal descriptive features from compact public order-book snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from .base import FeatureDataResource, FeatureDefinition, OutputField, ParameterDefinition

ORDER_BOOK_FEATURE_NAME = "order_book_context"
ORDER_BOOK_FEATURE_VERSION = "2"


def book_ticker_resource(interval="1m"):
    # Both order-book resources deliberately use the provider's generic base
    # role. DatasetKind keeps the two identities distinct without requiring a
    # feature-specific exception inside FeatureRegistry.
    return FeatureDataResource(DatasetKind.BOOK_TICKER, interval, "order_book")


def book_depth_resource(interval="1m"):
    return FeatureDataResource(DatasetKind.BOOK_DEPTH, interval, "order_book")


_TOP_VALUES = (
    "book_best_bid_price",
    "book_best_bid_qty",
    "book_best_ask_price",
    "book_best_ask_qty",
    "book_spread",
    "book_spread_bps",
    "book_midprice",
    "book_imbalance_l1",
    "book_microprice",
    "book_microprice_offset_bps",
)


def _schema(_params):
    result = {name: OutputField("numeric") for name in _TOP_VALUES}
    result.update(
        {
            "book_ticker_event_at": OutputField("datetime"),
            "book_ticker_age_seconds": OutputField("numeric"),
            "book_ticker_covered": OutputField("bool", False),
            "book_ticker_observed": OutputField("bool", False),
            "book_ticker_stale": OutputField("bool", False),
            "book_depth_event_at": OutputField("datetime"),
            "book_depth_age_seconds": OutputField("numeric"),
            "book_depth_covered": OutputField("bool", False),
            "book_depth_observed": OutputField("bool", False),
            "book_depth_stale": OutputField("bool", False),
            "book_depth_snapshot_complete": OutputField("bool", True),
        }
    )
    for band in range(1, 6):
        for side in ("bid", "ask"):
            result[f"book_{side}_depth_{band}pct"] = OutputField("numeric")
            result[f"book_{side}_notional_{band}pct"] = OutputField("numeric")
        result[f"book_depth_imbalance_{band}pct"] = OutputField("numeric")
        result[f"book_depth_ratio_{band}pct"] = OutputField("numeric")
        result[f"book_notional_imbalance_{band}pct"] = OutputField("numeric")
    return result


def _utc_ns(values):
    return pd.to_datetime(values, utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )


@dataclass(frozen=True, slots=True)
class OrderBookContextFeatureProvider:
    definition: FeatureDefinition = FeatureDefinition(
        name=ORDER_BOOK_FEATURE_NAME,
        version=ORDER_BOOK_FEATURE_VERSION,
        required_datasets=(DatasetKind.KLINES,),
        optional_datasets=(DatasetKind.BOOK_TICKER, DatasetKind.BOOK_DEPTH),
        parameters={
            "book_ticker_max_age_seconds": ParameterDefinition(float, 5.0),
            "book_depth_max_age_seconds": ParameterDefinition(float, 90.0),
        },
        output_schema_factory=_schema,
        availability_rule="latest_completed_1m_book_snapshot_at_or_before_strategy_decision",
    )

    def compute(
        self,
        request: DataRequest,
        datasets: Mapping[object, pd.DataFrame],
        parameters,
        feature_frames=None,
    ):
        del request, feature_frames
        ticker_max_age = float(parameters["book_ticker_max_age_seconds"])
        depth_max_age = float(parameters["book_depth_max_age_seconds"])
        if ticker_max_age < 0 or depth_max_age < 0:
            raise ValueError("order-book maximum ages must be non-negative")

        strategy = datasets[DatasetKind.KLINES]
        decisions = pd.DataFrame(
            {
                "timestamp": _utc_ns(strategy["period_start"]),
                "available_at": _utc_ns(strategy["available_at"]),
            }
        )
        result = decisions.copy()
        ticker = datasets.get(book_ticker_resource())
        depth = datasets.get(book_depth_resource())
        if ticker is None and depth is None:
            raise ValueError(
                "order_book_context requires at least one compact order-book resource"
            )

        self._reject_raw_events(ticker, DatasetKind.BOOK_TICKER)
        self._reject_raw_events(depth, DatasetKind.BOOK_DEPTH)
        result = self._ticker(result, ticker, ticker_max_age)
        result = self._depth(result, depth, depth_max_age)
        return result.loc[:, ["timestamp", "available_at", *_schema(parameters).keys()]]

    @staticmethod
    def _reject_raw_events(source, dataset: DatasetKind) -> None:
        if source is None:
            return
        if dataset is DatasetKind.BOOK_TICKER:
            forbidden = {"update_id", "event_time", "transaction_time"}
        else:
            forbidden = {"event_time", "percentage", "depth", "notional"}
        leaked = sorted(forbidden & set(source.columns))
        if leaked:
            raise ValueError(
                "raw order-book event columns are forbidden in order_book_context: "
                f"{leaked}"
            )

    @staticmethod
    def _align(result, source):
        facts = source.copy()
        facts["available_at"] = _utc_ns(facts["available_at"])
        facts["source_event_at"] = _utc_ns(facts["source_event_at"])
        return pd.merge_asof(
            result.sort_values("available_at"),
            facts.sort_values("available_at"),
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )

    def _ticker(self, result, source, max_age):
        if source is None:
            for column in _TOP_VALUES:
                result[column] = np.nan
            result["book_ticker_event_at"] = pd.NaT
            result["book_ticker_age_seconds"] = np.nan
            for column in (
                "book_ticker_covered",
                "book_ticker_observed",
                "book_ticker_stale",
            ):
                result[column] = False
            return result

        source = source.rename(
            columns={
                "best_bid_price": "book_best_bid_price",
                "best_bid_qty": "book_best_bid_qty",
                "best_ask_price": "book_best_ask_price",
                "best_ask_qty": "book_best_ask_qty",
            }
        )
        coverage = source[["available_at", "book_ticker_covered"]].copy()
        coverage["available_at"] = _utc_ns(coverage["available_at"])
        coverage = pd.merge_asof(
            result[["available_at"]].sort_values("available_at"),
            coverage.sort_values("available_at"),
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        source = source.loc[source["book_ticker_observed"].fillna(False)].copy()
        keep = [
            "available_at",
            "source_event_at",
            "book_ticker_covered",
            "book_ticker_observed",
            "book_best_bid_price",
            "book_best_bid_qty",
            "book_best_ask_price",
            "book_best_ask_qty",
        ]
        out = self._align(result, source[keep])
        out["book_ticker_covered"] = coverage["book_ticker_covered"].to_numpy()
        out["book_ticker_event_at"] = out.pop("source_event_at")
        out["book_ticker_age_seconds"] = (
            out["available_at"] - out["book_ticker_event_at"]
        ).dt.total_seconds()
        out["book_ticker_stale"] = out["book_ticker_observed"].fillna(False) & out[
            "book_ticker_age_seconds"
        ].gt(max_age)

        bid, ask = out["book_best_bid_price"], out["book_best_ask_price"]
        bq, aq = out["book_best_bid_qty"], out["book_best_ask_qty"]
        out["book_spread"] = ask - bid
        out["book_midprice"] = (bid + ask) / 2
        out["book_spread_bps"] = (
            out["book_spread"]
            / out["book_midprice"].replace(0, np.nan)
            * 10000
        )
        denom = (bq + aq).replace(0, np.nan)
        out["book_imbalance_l1"] = (bq - aq) / denom
        out["book_microprice"] = (ask * bq + bid * aq) / denom
        out["book_microprice_offset_bps"] = (
            (out["book_microprice"] - out["book_midprice"])
            / out["book_midprice"].replace(0, np.nan)
            * 10000
        )
        out.loc[out["book_ticker_stale"], list(_TOP_VALUES)] = np.nan
        for column in (
            "book_ticker_covered",
            "book_ticker_observed",
            "book_ticker_stale",
        ):
            out[column] = out[column].fillna(False).astype(bool)
        return out

    def _depth(self, result, source, max_age):
        value_columns = [
            f"book_{side}_{kind}_{band}pct"
            for band in range(1, 6)
            for side in ("bid", "ask")
            for kind in ("depth", "notional")
        ]
        if source is None:
            result["book_depth_event_at"] = pd.NaT
            result["book_depth_age_seconds"] = np.nan
            for column in (
                "book_depth_covered",
                "book_depth_observed",
                "book_depth_stale",
            ):
                result[column] = False
            result["book_depth_snapshot_complete"] = pd.Series(
                pd.NA, index=result.index, dtype="boolean"
            )
            for column in value_columns:
                result[column] = np.nan
        else:
            coverage = source[["available_at", "book_depth_covered"]].copy()
            coverage["available_at"] = _utc_ns(coverage["available_at"])
            coverage = pd.merge_asof(
                result[["available_at"]].sort_values("available_at"),
                coverage.sort_values("available_at"),
                on="available_at",
                direction="backward",
                allow_exact_matches=True,
            )
            source = source.loc[source["book_depth_observed"].fillna(False)].copy()
            keep = [
                "available_at",
                "source_event_at",
                "book_depth_covered",
                "book_depth_observed",
                "book_depth_snapshot_complete",
                *value_columns,
            ]
            result = self._align(result, source[keep])
            result["book_depth_covered"] = coverage["book_depth_covered"].to_numpy()
            result["book_depth_event_at"] = result.pop("source_event_at")
            result["book_depth_age_seconds"] = (
                result["available_at"] - result["book_depth_event_at"]
            ).dt.total_seconds()
            result["book_depth_stale"] = result["book_depth_observed"].fillna(False) & result[
                "book_depth_age_seconds"
            ].gt(max_age)
            stale = result["book_depth_stale"]
            result.loc[stale, value_columns] = np.nan
            result["book_depth_snapshot_complete"] = result[
                "book_depth_snapshot_complete"
            ].astype("boolean")
            result.loc[stale, "book_depth_snapshot_complete"] = pd.NA
            for column in (
                "book_depth_covered",
                "book_depth_observed",
                "book_depth_stale",
            ):
                result[column] = result[column].fillna(False).astype(bool)

        for band in range(1, 6):
            bid, ask = (
                result[f"book_bid_depth_{band}pct"],
                result[f"book_ask_depth_{band}pct"],
            )
            result[f"book_depth_imbalance_{band}pct"] = (bid - ask) / (
                bid + ask
            ).replace(0, np.nan)
            result[f"book_depth_ratio_{band}pct"] = bid / ask.replace(0, np.nan)
            bidn, askn = (
                result[f"book_bid_notional_{band}pct"],
                result[f"book_ask_notional_{band}pct"],
            )
            result[f"book_notional_imbalance_{band}pct"] = (bidn - askn) / (
                bidn + askn
            ).replace(0, np.nan)
        return result
