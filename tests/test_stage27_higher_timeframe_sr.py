import os
import sys

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.enhanced_engine import EnhancedBacktestEngine
from crypto_strategy_lab.gui.enhanced_config import build_enhanced_backtest_config, enhanced_default_gui_config
from crypto_strategy_lab.higher_timeframe_sr import resample_ohlc_for_sr


def _candles(periods=16):
    close = np.arange(periods, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC"),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        }
    )


def test_sr_higher_timeframe_config_is_optional_and_validated():
    values = enhanced_default_gui_config()
    assert values["sr_timeframe_minutes"] == 0
    values.update({"strategy_timeframe_minutes": 15, "sr_timeframe_minutes": 60, "use_intrabar_data": False})
    cfg = build_enhanced_backtest_config(values, require_paths=False)
    assert cfg.sr_timeframe_minutes == 60

    invalid = dict(values)
    invalid.update({
        "strategy_timeframe_minutes": 240,
        "sr_timeframe_minutes": 60,
        "telemetry_interval_minutes": 240,
    })
    with pytest.raises(ValueError, match="cannot be lower"):
        build_enhanced_backtest_config(invalid, require_paths=False)


def test_resample_sr_uses_only_complete_higher_timeframe_bars():
    data = _candles(9)  # two complete hours plus one partial 15m candle
    htf = resample_ohlc_for_sr(data, 15, 60)
    assert len(htf) == 2
    assert htf.iloc[0]["open"] == pytest.approx(data.iloc[0]["open"])
    assert htf.iloc[0]["close"] == pytest.approx(data.iloc[3]["close"])
    assert htf.iloc[0]["end_time"] == pd.Timestamp("2024-01-01 01:00:00+00:00")
    assert htf.iloc[1]["end_time"] == pd.Timestamp("2024-01-01 02:00:00+00:00")


def test_engine_exposes_only_completed_htf_candle_at_entry():
    values = enhanced_default_gui_config()
    values.update(
        {
            "strategy_timeframe_minutes": 15,
            "use_intrabar_data": False,
            "enable_support_resistance_analysis": True,
            "sr_timeframe_minutes": 60,
            "atr_period": 2,
            "adx_period": 2,
            "sr_pivot_left": 1,
            "sr_pivot_right": 1,
            "market_regime_method": "ASSET_RETURN",
        }
    )
    cfg = build_enhanced_backtest_config(values, require_paths=False)
    engine = EnhancedBacktestEngine(_candles(16), cfg)
    assert engine.sr_uses_higher_timeframe is True
    assert engine.sr_timeframe_minutes == 60
    # 00:45 strategy candle enters at 01:00, so the 00:00-01:00 HTF candle is now complete.
    assert engine._latest_completed_sr_index(3) == 0
    # 00:30 candle enters at 00:45; no 1h candle has closed yet.
    assert engine._latest_completed_sr_index(2) == -1


def test_same_timeframe_keeps_legacy_sr_path():
    values = enhanced_default_gui_config()
    values.update(
        {
            "strategy_timeframe_minutes": 15,
            "use_intrabar_data": False,
            "enable_support_resistance_analysis": True,
            "sr_timeframe_minutes": 0,
            "atr_period": 2,
            "adx_period": 2,
            "market_regime_method": "ASSET_RETURN",
        }
    )
    engine = EnhancedBacktestEngine(_candles(16), build_enhanced_backtest_config(values, require_paths=False))
    assert engine.sr_uses_higher_timeframe is False
    assert engine.sr_timeframe_minutes == 15


def test_gui_exposes_sr_structure_timeframe():
    qtwidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from crypto_strategy_lab.gui.enhanced_main_window import MainWindow

    app = qtwidgets.QApplication.instance() or qtwidgets.QApplication(sys.argv)
    window = MainWindow()
    try:
        labels = [window.sr_timeframe.itemText(i) for i in range(window.sr_timeframe.count())]
        assert labels == ["Same as Strategy", "1h", "4h", "1d"]
        window.strategy_timeframe.setCurrentText("15m")
        window.sr_timeframe.setCurrentIndex(window.sr_timeframe.findData(60))
        assert window.values()["sr_timeframe_minutes"] == 60
    finally:
        window.close()
