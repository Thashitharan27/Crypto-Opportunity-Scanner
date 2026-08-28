from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_strategy_lab.config import DailyEntryMissedPolicy, RiskMode
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.features.technical import CoreDirectionalFeatureProvider
from crypto_strategy_lab.gui.enhanced_config import EnhancedBacktestConfig
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, StrategyProfile


def _canonical_15m(n: int = 60) -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="15min")
    close = 100.0 + np.sin(np.arange(n) / 6.0) * 2.0
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + pd.Timedelta(minutes=15),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0,
            "source_fingerprint": "research-trade-row-source",
        }
    )


def _legacy(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": frame["period_start"],
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "volume": frame["volume"],
        }
    )


def _profiles() -> dict[str, StrategyProfile]:
    profile = StrategyProfile(
        enabled=True,
        stop_loss_multiple=1.0,
        reward_risk_ratio=1.0,
    )
    return {key: profile for key in PROFILE_KEYS}


def _config() -> EnhancedBacktestConfig:
    return EnhancedBacktestConfig(
        strategy_timeframe_minutes=15,
        intrabar_timeframe_minutes=1,
        telemetry_interval_minutes=15,
        use_intrabar_data=False,
        risk_mode=RiskMode.FIXED,
        fixed_r=1.0,
        initial_equity=1000.0,
        risk_per_leg=0.01,
        taker_fee=0.0,
        maker_fee=0.0,
        slippage=0.0,
        atr_period=3,
        adx_period=3,
        di_pressure_lookback=1,
        market_regime_method="ASSET_RETURN",
        enable_daily_entry_schedule=True,
        daily_entry_time="12:00",
        daily_entry_timezone="UTC",
        daily_entry_missed_policy=DailyEntryMissedPolicy.SKIP_DAY,
        enable_support_resistance_analysis=False,
        strategy_profiles=_profiles(),
    )


def _technical(frame: pd.DataFrame, config: EnhancedBacktestConfig) -> pd.DataFrame:
    request = DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(minutes=15)).to_pydatetime(),
        strategy_interval="15m",
    )
    prepared = CoreDirectionalFeatureProvider().compute(
        request,
        {DatasetKind.KLINES: frame},
        {
            "atr_period": config.atr_period,
            "adx_period": config.adx_period,
            "di_pressure_lookback": config.di_pressure_lookback,
        },
    )
    prepared.attrs["feature_cache_key"] = "research-trade-row-directional"
    return prepared


def _research_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    timestamps = pd.to_datetime(frame["period_start"], utc=True)
    available = pd.to_datetime(frame["available_at"], utc=True)
    index = np.arange(len(frame), dtype=float)

    positioning = pd.DataFrame(
        {
            "timestamp": timestamps,
            "available_at": available,
            "open_interest": 1000.0 + index * 10.0,
            "price_oi_state": np.where(index % 2 == 0, "PRICE_UP_OI_UP", "PRICE_DOWN_OI_UP"),
        }
    )
    positioning.attrs.update(feature_name="futures_positioning", feature_version="1")

    funding = pd.DataFrame(
        {
            "timestamp": timestamps,
            "available_at": available,
            "funding_rate": index / 1_000_000.0,
            "funding_bias": np.where(index == 0, "NEUTRAL", "POSITIVE"),
        }
    )
    funding.attrs.update(feature_name="funding_context", feature_version="1")
    return {"futures_positioning": positioning, "funding_context": funding}


def test_scheduled_entry_trade_row_uses_previous_completed_signal_candle_research() -> None:
    canonical = _canonical_15m()
    config = _config()
    research = _research_frames(canonical)
    engine = DataLakeProductionBacktestEngine(
        _legacy(canonical),
        config,
        technical_features=_technical(canonical, config),
        research_features=research,
    )

    # Keep the execution test focused on research-timestamp semantics rather than
    # waiting for the synthetic series to establish an ASSET_RETURN regime/DI side.
    engine.market_regime_values[:] = "SIDEWAYS"
    engine.plus_di_values[:] = 50.0
    engine.minus_di_values[:] = 10.0
    engine.di_spread[:] = 40.0

    trades = engine.run()
    assert len(trades) == 1
    row = trades.iloc[0]

    execution_index = 48  # 12:00 UTC candle in a 15-minute series starting at midnight.
    indicator_index = execution_index - 1
    expected_signal_open = pd.Timestamp("2026-01-01T11:45:00Z")
    expected_signal_available = pd.Timestamp("2026-01-01T12:00:00Z")

    assert pd.Timestamp(row["strategy_candle_open_time"]) == pd.Timestamp("2026-01-01T12:00:00Z")
    assert int(row["research_signal_index"]) == indicator_index
    assert pd.Timestamp(row["research_signal_candle_open_time"]) == expected_signal_open
    assert pd.Timestamp(row["research_signal_available_at"]) == expected_signal_available

    assert row["open_interest"] == research["futures_positioning"].iloc[indicator_index]["open_interest"]
    assert row["price_oi_state"] == research["futures_positioning"].iloc[indicator_index]["price_oi_state"]
    assert row["funding_rate"] == research["funding_context"].iloc[indicator_index]["funding_rate"]
    assert row["funding_bias"] == research["funding_context"].iloc[indicator_index]["funding_bias"]
    assert pd.Timestamp(row["futures_positioning_feature_available_at"]) == expected_signal_available
    assert pd.Timestamp(row["funding_context_feature_available_at"]) == expected_signal_available

    # The execution candle has different synthetic research values. Seeing them
    # here would mean a scheduled-entry look-ahead regression.
    assert row["open_interest"] != research["futures_positioning"].iloc[execution_index]["open_interest"]
    assert row["funding_rate"] != research["funding_context"].iloc[execution_index]["funding_rate"]
