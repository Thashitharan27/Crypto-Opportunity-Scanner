from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data import DataRequest, DatasetKind, MarketKind
from crypto_strategy_lab.features import production_feature_registry
from feature_causality_harness import CausalityCase, assert_future_mutation_invariant


INTERVAL = "4h"
DELTA = pd.Timedelta(INTERVAL)


def _klines(n=300):
    times = pd.date_range("2025-01-01", periods=n, freq=INTERVAL, tz="UTC")
    x = np.arange(n, dtype=float)
    center = 100 + x * .035 + np.sin(x / 6) * 4 + np.sin(x / 19) * 2
    open_ = center + np.sin(x / 5) * .35
    close = center + np.cos(x / 4) * .55
    frame = pd.DataFrame({
        "period_start": times,
        "period_end": times + DELTA,
        "available_at": times + DELTA,
        "open": open_,
        "high": np.maximum(open_, close) + 1.4,
        "low": np.minimum(open_, close) - 1.4,
        "close": close,
        "volume": 100 + (x % 31) * 3,
    })
    frame.attrs["canonical_source_identity"] = "fixture-klines"
    return frame


def _request(frame):
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + DELTA).to_pydatetime(),
        strategy_interval=INTERVAL,
        market=MarketKind.FUTURES_UM,
    )


def _reference(source, offset, identity):
    frame = pd.DataFrame({
        "period_start": source.period_start,
        "period_end": source.period_end,
        "available_at": source.available_at,
        "close": source.close + offset,
    })
    frame.attrs["canonical_source_identity"] = identity
    return frame


def _metrics(source):
    times = pd.date_range(
        source.period_start.iloc[0] + pd.Timedelta(hours=1),
        source.period_end.iloc[-1], freq="2h", inclusive="left"
    )
    x = np.arange(len(times), dtype=float)
    frame = pd.DataFrame({
        "event_time": times, "period_start": times, "available_at": times,
        "open_interest": 10_000 + x * 3,
        "open_interest_value": 1_000_000 + x * 250,
        "top_trader_account_long_short_ratio": 1.05 + np.sin(x / 11) * .1,
        "top_trader_position_long_short_ratio": 1.10 + np.cos(x / 13) * .1,
        "global_long_short_account_ratio": .95 + np.sin(x / 17) * .08,
        "taker_long_short_volume_ratio": 1 + np.cos(x / 7) * .12,
    })
    frame.attrs["canonical_source_identity"] = "fixture-metrics"
    return frame


def _funding(source):
    times = pd.date_range(source.period_start.iloc[0], source.period_end.iloc[-1],
                          freq="8h", inclusive="left")
    x = np.arange(len(times), dtype=float)
    frame = pd.DataFrame({
        "event_time": times, "period_start": times, "available_at": times,
        "funding_rate": np.sin(x / 5) * .0002,
        "funding_interval_hours": 8.0,
    })
    frame.attrs["canonical_source_identity"] = "fixture-funding"
    return frame


def _agg(source):
    rows, agg_id = [], 1
    for i, row in source.iterrows():
        for hours, maker, scale in ((1, False, .999), (2, True, 1.001)):
            event = row.period_start + pd.Timedelta(hours=hours)
            rows.append({
                "event_time": event, "period_start": event, "available_at": event,
                "agg_trade_id": agg_id, "price": float(row.close) * scale,
                "quantity": 1 + (i % 7) * .1, "is_buyer_maker": maker,
                "first_trade_id": agg_id * 2, "last_trade_id": agg_id * 2 + 1,
            })
            agg_id += 1
    frame = pd.DataFrame(rows)
    frame.attrs["canonical_source_identity"] = "fixture-agg"
    return frame


def _benchmark():
    times = pd.date_range("2024-12-01", periods=24 * 90, freq="1h", tz="UTC")
    x = np.arange(len(times), dtype=float)
    return pd.DataFrame({
        "period_start": times,
        "available_at": times + pd.Timedelta(hours=1),
        "close": 80 + x * .012 + np.sin(x / 40) * 3,
    })


def _registry(context):
    return production_feature_registry(structural_benchmark=context.get("structural_benchmark"))


def _future(frame, cutoff):
    mask = pd.to_datetime(frame.available_at, utc=True) > cutoff
    assert mask.any()
    return mask


def _mutate_klines(frame, cutoff):
    mask = _future(frame, cutoff)
    frame.loc[mask, ["open", "high", "low", "close"]] *= 1.35
    frame.loc[mask, "volume"] *= 2


def _mutate_metrics(frame, cutoff):
    mask = _future(frame, cutoff)
    for column in (
        "open_interest", "open_interest_value",
        "top_trader_account_long_short_ratio", "top_trader_position_long_short_ratio",
        "global_long_short_account_ratio", "taker_long_short_volume_ratio",
    ):
        frame.loc[mask, column] *= 1.7


def _mutate_funding(frame, cutoff):
    mask = _future(frame, cutoff)
    frame.loc[mask, "funding_rate"] = frame.loc[mask, "funding_rate"] * -3 + .0007


def _mutate_reference(frame, cutoff):
    frame.loc[_future(frame, cutoff), "close"] *= 1.4


def _mutate_agg(frame, cutoff):
    mask = _future(frame, cutoff)
    frame.loc[mask, "price"] *= 1.2
    frame.loc[mask, "quantity"] *= 3
    frame.loc[mask, "is_buyer_maker"] = ~frame.loc[mask, "is_buyer_maker"].astype(bool)


def _mutate_benchmark(frame, cutoff):
    frame.loc[_future(frame, cutoff), "close"] *= 1.6


def _core():
    return {"core_directional": {"atr_period": 7, "adx_period": 6, "di_pressure_lookback": 3}}


def _market():
    return {**_core(), "market_context": {
        "bb_period": 12, "bb_stddevs": 2.0, "mean_reversion_period": 10,
    }}


def _production():
    return {**_core(), "production_market_context": {
        "bb_period": 12, "bb_stddevs": 2.0, "mean_reversion_period": 10,
        "mean_reversion_mean_type": "SMA", "mean_reversion_bb_stddevs": 2.0,
        "mean_reversion_rsi_period": 7, "mean_reversion_rsi_oversold": 30.0,
        "mean_reversion_rsi_overbought": 70.0, "mean_reversion_require_reentry": True,
    }}


def _support():
    return {**_core(), "support_resistance": {
        "atr_period": 7, "sr_timeframe_minutes": 240, "sr_pivot_left": 2,
        "sr_pivot_right": 2, "sr_lookback_bars": 60, "sr_zone_width_atr": .5,
        "sr_near_distance_atr": .75, "enable_sr_hold_confirmation": False,
        "sr_hold_confirmation_bars": 3, "sr_hold_confirmation_atr": .25,
        "sr_break_tolerance_atr": .25, "sr_break_basis": "CLOSE",
    }}


def _policy(method):
    return {"policy_market_context": {
        "market_regime_method": method, "bull_regime_lookback_days": 5,
        "bull_regime_return_threshold": .02, "structural_regime_sma_days": 5,
        "structural_regime_slope_lookback_days": 2,
        "momentum_lookback_hours": (8, 24),
    }}


def _cases():
    k = _klines(); request = _request(k)
    metrics, funding, agg = _metrics(k), _funding(k), _agg(k)
    mark = _reference(k, .25, "mark")
    index = _reference(k, -.15, "index")
    premium = _reference(k, .05, "premium")
    return [
        CausalityCase("core_directional", _registry, request, {DatasetKind.KLINES: k},
                      _core(), {DatasetKind.KLINES: _mutate_klines}),
        CausalityCase("market_context", _registry, request, {DatasetKind.KLINES: k},
                      _market(), {DatasetKind.KLINES: _mutate_klines}),
        CausalityCase("production_market_context", _registry, request, {DatasetKind.KLINES: k},
                      _production(), {DatasetKind.KLINES: _mutate_klines}),
        CausalityCase("policy_market_context", _registry, request, {DatasetKind.KLINES: k},
                      _policy("ASSET_RETURN"), {DatasetKind.KLINES: _mutate_klines}),
        CausalityCase(
            "policy_market_context", _registry, request, {DatasetKind.KLINES: k},
            _policy("BTC_STRUCTURAL"), {DatasetKind.KLINES: _mutate_klines},
            context={"structural_benchmark": _benchmark()},
            future_context_mutators={"structural_benchmark": _mutate_benchmark},
        ),
        CausalityCase("support_resistance", _registry, request, {DatasetKind.KLINES: k},
                      _support(), {DatasetKind.KLINES: _mutate_klines}),
        CausalityCase(
            "state_transition_daily", _registry, request, {DatasetKind.KLINES: k},
            {"state_transition_daily": {
                "regime_lookback_days": 5, "bull_return_threshold": .02,
                "bear_return_threshold": -.02, "volatility_lookback_days": 5,
                "volatility_reference_days": 40, "volatility_low_quantile": .33,
                "volatility_high_quantile": .67, "minimum_state_observations": 5,
                "minimum_trade_observations": 5,
            }},
            {DatasetKind.KLINES: _mutate_klines},
        ),
        CausalityCase(
            "futures_positioning", _registry, request,
            {DatasetKind.KLINES: k, DatasetKind.FUTURES_METRICS: metrics}, {},
            {DatasetKind.KLINES: _mutate_klines, DatasetKind.FUTURES_METRICS: _mutate_metrics},
        ),
        CausalityCase(
            "funding_context", _registry, request,
            {DatasetKind.KLINES: k, DatasetKind.FUNDING_RATE: funding}, {},
            {DatasetKind.KLINES: _mutate_klines, DatasetKind.FUNDING_RATE: _mutate_funding},
        ),
        CausalityCase(
            "basis_context", _registry, request,
            {DatasetKind.KLINES: k, DatasetKind.MARK_PRICE_KLINES: mark,
             DatasetKind.INDEX_PRICE_KLINES: index, DatasetKind.PREMIUM_INDEX_KLINES: premium}, {},
            {DatasetKind.KLINES: _mutate_klines,
             DatasetKind.MARK_PRICE_KLINES: _mutate_reference,
             DatasetKind.INDEX_PRICE_KLINES: _mutate_reference,
             DatasetKind.PREMIUM_INDEX_KLINES: _mutate_reference},
        ),
    ]


_IDS = [
    "core-directional-di", "market-context-mr", "production-context-mr-v2",
    "policy-regime-asset-return", "policy-regime-structural", "support-resistance",
    "state-transition-daily", "futures-positioning", "funding-context",
    "basis-context",
]


@pytest.mark.parametrize("case", _cases(), ids=_IDS)
def test_registered_feature_is_invariant_to_future_source_mutation(case):
    assert_future_mutation_invariant(case)
