from __future__ import annotations

import importlib.util
from pathlib import Path

from crypto_strategy_lab.gui.config_logic import CONFIG_VERSION, DEFAULT_GUI_CONFIG


def _load_tool():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "rewrite_config_current.py"
    spec = importlib.util.spec_from_file_location("rewrite_config_current", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rewrite_config_removes_retired_fields_and_keeps_current_values() -> None:
    tool = _load_tool()
    source = {
        "config_version": 1,
        "market_symbol": "BTCUSDT",
        "strategy_timeframe_minutes": 240,
        "risk_per_leg": 0.0075,
        "di_pressure_allow_contracting": True,
        "mean_reversion_rsi_period": 14,
    }

    current, retired = tool.rewrite_config(source)

    assert current["config_version"] == CONFIG_VERSION
    assert current["market_symbol"] == "BTCUSDT"
    assert current["strategy_timeframe_minutes"] == 240
    assert current["risk_per_leg"] == 0.0075
    assert set(retired) == {"di_pressure_allow_contracting", "mean_reversion_rsi_period"}
    assert set(current) == set(DEFAULT_GUI_CONFIG)
    assert "di_pressure_allow_contracting" not in current
    assert "mean_reversion_rsi_period" not in current
