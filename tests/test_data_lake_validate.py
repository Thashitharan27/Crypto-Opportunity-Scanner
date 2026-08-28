from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pandas as pd

from crypto_strategy_lab.data import DataRequest


def _tool():
    path = Path(__file__).resolve().parents[1] / "tools/data_lake_validate.py"
    spec = importlib.util.spec_from_file_location("data_lake_validate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_accepts_native_canonical_identity_and_causality() -> None:
    request = DataRequest(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        strategy_interval="4h",
    )
    starts = pd.date_range(request.start, periods=2, freq="4h")
    frame = pd.DataFrame({
        "period_start": starts, "period_end": starts + pd.Timedelta(hours=4),
        "available_at": starts + pd.Timedelta(hours=4),
        "open": [1.0, 2.0], "high": [2.0, 3.0], "low": [.5, 1.5],
        "close": [1.5, 2.5], "volume": [10.0, 11.0],
        "symbol": "BTCUSDT", "exchange": "binance", "market": "futures_um",
        "dataset": "klines", "interval": "4h",
    })
    frame.attrs["canonical_source_identity"] = "source-key"

    _tool().validate_canonical_klines(frame, request, "strategy klines")


def test_validator_cli_has_no_legacy_csv_options() -> None:
    options = {action.dest for action in _tool().build_parser()._actions}
    assert "legacy_strategy_csv" not in options
    assert "timestamp_unit" not in options


def test_validator_cli_delegates_to_shared_quality_service() -> None:
    source = (Path(__file__).resolve().parents[1] / "tools/data_lake_validate.py").read_text(
        encoding="utf-8"
    )
    assert "store.data_quality_report(" in source
    assert "store.load_execution_klines" not in source
    assert "store.load_klines(intrabar_request" not in source
