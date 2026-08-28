import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data import DataRequest, DatasetKind
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.features.trade_flow import (
    TradeFlowContextFeatureProvider,
    trade_flow_resource,
)
from feature_causality_harness import CausalityCase, assert_future_mutation_invariant


def _aggregate(source: DatasetKind) -> pd.DataFrame:
    starts = pd.date_range("2026-01-01", periods=240, freq="1min", tz="UTC")
    x = np.arange(len(starts), dtype=float)
    buy = 1.0 + (x % 5) * 0.1
    sell = 0.5 + (x % 3) * 0.05
    base = buy + sell
    quote = base * (100.0 + x * 0.01)
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=1),
            "available_at": starts + pd.Timedelta(minutes=1),
            "trade_flow_source_covered": True,
            "source_event_count": np.full(len(starts), 2.0),
            "underlying_trade_count": np.full(len(starts), 3.0 if source is DatasetKind.AGG_TRADES else 2.0),
            "base_volume": base,
            "quote_volume": quote,
            "aggressive_buy_base_volume": buy,
            "aggressive_sell_base_volume": sell,
            "aggressive_buy_quote_volume": buy * (100.0 + x * 0.01),
            "aggressive_sell_quote_volume": sell * (100.0 + x * 0.01),
            "trade_delta_base": buy - sell,
            "trade_delta_quote": (buy - sell) * (100.0 + x * 0.01),
            "weighted_price_sum": quote,
            "large_source_event_count": np.full(len(starts), np.nan),
            "large_source_event_quote_volume": np.full(len(starts), np.nan),
            "large_buy_quote_volume": np.full(len(starts), np.nan),
            "large_sell_quote_volume": np.full(len(starts), np.nan),
            "median_source_event_size": base / 2.0,
            "last_event_at": starts + pd.Timedelta(seconds=50),
        }
    )


def _klines() -> pd.DataFrame:
    starts = pd.date_range("2026-01-01", periods=240, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=1),
            "available_at": starts + pd.Timedelta(minutes=1),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
        }
    )


def _mutate_future_aggregate(frame: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
    assert mask.any()
    frame.loc[mask, "trade_delta_base"] *= -7
    frame.loc[mask, "underlying_trade_count"] *= 4
    frame.loc[mask, "source_event_count"] *= 3
    frame.loc[mask, "base_volume"] *= 2
    frame.loc[mask, "quote_volume"] *= 2
    frame.loc[mask, "weighted_price_sum"] *= 2


@pytest.mark.parametrize("source", [DatasetKind.AGG_TRADES, DatasetKind.TRADES])
def test_trade_flow_context_uses_generic_available_at_causality_harness(source):
    klines = _klines()
    aggregate = _aggregate(source)
    resource = trade_flow_resource(source)
    request = DataRequest(
        "BTCUSDT",
        klines.period_start.iloc[0].to_pydatetime(),
        klines.period_end.iloc[-1].to_pydatetime(),
        "1m",
    )
    case = CausalityCase(
        feature_name="trade_flow_context",
        registry_factory=lambda _: production_feature_registry(),
        request=request,
        datasets={DatasetKind.KLINES: klines, resource: aggregate},
        parameters={
            "trade_flow_context": {
                "trade_flow_source": source.name,
                "trade_flow_windows": ("1m", "5m", "15m", "1h"),
            }
        },
        future_mutators={resource: _mutate_future_aggregate},
    )
    assert_future_mutation_invariant(case)


def test_trade_flow_alignment_normalizes_microsecond_aggregate_and_nanosecond_strategy():
    source = DatasetKind.AGG_TRADES
    aggregate = _aggregate(source)
    # Mirror recent Binance/Parquet behavior where the compact aggregate
    # materializes as datetime64[us, UTC] while strategy klines remain ns.
    for column in ("period_start", "period_end", "available_at", "last_event_at"):
        aggregate[column] = aggregate[column].astype("datetime64[us, UTC]")

    klines = _klines().iloc[::60].reset_index(drop=True)
    assert str(aggregate["available_at"].dtype) == "datetime64[us, UTC]"
    assert str(klines["available_at"].dtype) == "datetime64[ns, UTC]"

    request = DataRequest(
        "BTCUSDT",
        klines.period_start.iloc[0].to_pydatetime(),
        klines.period_end.iloc[-1].to_pydatetime(),
        "1h",
    )
    output = TradeFlowContextFeatureProvider().compute(
        request,
        {
            DatasetKind.KLINES: klines,
            trade_flow_resource(source): aggregate,
        },
        {
            "trade_flow_source": source.name,
            "trade_flow_windows": ("1m", "5m", "15m", "1h"),
        },
    )

    assert len(output) == len(klines)
    assert str(output["available_at"].dtype) == "datetime64[ns, UTC]"
    assert output["trade_source_covered"].any()
