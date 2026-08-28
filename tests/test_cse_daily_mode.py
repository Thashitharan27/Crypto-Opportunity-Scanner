import pandas as pd

from crypto_strategy_lab.gui.sr_dynamic_tp_main_window import MainWindow
from crypto_strategy_lab.loader import load_ohlcv_csv


def test_daily_timeframe_conversion_is_explicit():
    assert MainWindow._timeframe_minutes("1d") == 1440
    assert MainWindow._timeframe_label(1440) == "1d"
    assert MainWindow._timeframe_label(2880) == "2d"


def test_cse_style_daily_csv_loads_with_date_alias_and_timezone(tmp_path):
    path = tmp_path / "HHL-N0000.CM.csv"
    frame = pd.DataFrame(
        {
            "Date": [
                "2026-08-17 00:00:00+05:30",
                "2026-08-18 00:00:00+05:30",
                "2026-08-19 00:00:00+05:30",
            ],
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1000, 1200, 1100],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        }
    )
    frame.to_csv(path, index=False)

    loaded = load_ohlcv_csv(
        str(path),
        timestamp_unit="ms",
        expected_timeframe_minutes=1440,
        label="CSE daily",
        strict_timeframe=True,
    )

    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(loaded) == 3
    assert loaded.attrs["summary"].detected_timeframe_minutes == 1440
