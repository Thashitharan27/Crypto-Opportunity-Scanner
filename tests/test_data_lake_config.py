from __future__ import annotations

from crypto_strategy_lab.data_lake_config import build_data_lake_backtest_config, normalize_data_lake_config
from crypto_strategy_lab.gui.enhanced_config import EnhancedBacktestConfig


def test_data_lake_config_does_not_require_csv_paths() -> None:
    config = build_data_lake_backtest_config(
        {
            "config_version": 2,
            "market_symbol": "BTCUSDT",
            "strategy_timeframe_minutes": 240,
            "intrabar_timeframe_minutes": 1,
            "use_intrabar_data": True,
            "telemetry_interval_minutes": 240,
            "market_regime_method": "BTC_STRUCTURAL",
        }
    )

    assert isinstance(config, EnhancedBacktestConfig)
    assert config.strategy_timeframe_minutes == 240
    assert config.intrabar_timeframe_minutes == 1
    assert config.market_regime_method == "BTC_STRUCTURAL"
    assert str(config.input_csv) == "__DATA_LAKE_STRATEGY__"
    assert config.intrabar_csv is None
    assert config.structural_regime_benchmark_csv is None


def test_data_lake_config_preserves_enhanced_strategy_settings() -> None:
    config = build_data_lake_backtest_config(
        {
            "config_version": 2,
            "strategy_timeframe_minutes": 240,
            "intrabar_timeframe_minutes": 1,
            "use_intrabar_data": False,
            "telemetry_interval_minutes": 240,
            "di_pressure_allow_expanding": True,
            "di_pressure_allow_contracting": False,
            "di_pressure_allow_mixed": True,
            "mean_reversion_mean_type": "SMA",
            "mean_reversion_rsi_period": 9,
            "sr_timeframe_minutes": 480,
            "sr_take_profit_mode": "FIXED_R",
        }
    )

    assert config.di_pressure_allow_contracting is False
    assert config.mean_reversion_mean_type == "SMA"
    assert config.mean_reversion_rsi_period == 9
    assert config.sr_timeframe_minutes == 480
    assert config.sr_take_profit_mode == "FIXED_R"


def test_data_lake_config_rejects_market_data_path_fields() -> None:
    try:
        normalize_data_lake_config(
            {
                "config_version": 2,
                "input_csv": "BTCUSDT_4h.csv",
            }
        )
    except ValueError as exc:
        assert "input_csv" in str(exc)
    else:
        raise AssertionError("Data Lake config accepted a legacy CSV path field")
