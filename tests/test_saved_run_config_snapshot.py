from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from crypto_strategy_lab.config import BacktestConfig, RiskMode
from crypto_strategy_lab.output_manager import config_to_dict, load_config_snapshot


def test_saved_run_snapshot_restores_current_fields_and_reports_retired(tmp_path: Path) -> None:
    config = replace(
        BacktestConfig(),
        input_csv=tmp_path / "strategy.csv",
        intrabar_csv=tmp_path / "intrabar.csv",
        output_dir=tmp_path / "output",
        timestamp_unit="ms",
        strategy_timeframe_minutes=240,
        intrabar_timeframe_minutes=1,
        telemetry_interval_minutes=240,
        data_start_date="2025-01-01",
        risk_mode=RiskMode.ATR,
    )
    raw = config_to_dict(config)
    raw["mean_reversion_rsi_period"] = 14
    raw["sr_take_profit_mode"] = "ROOM_BASED"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    restored, ignored = load_config_snapshot(path, require_paths=False)

    assert restored.strategy_timeframe_minutes == 240
    assert restored.intrabar_timeframe_minutes == 1
    assert restored.telemetry_interval_minutes == 240
    assert restored.data_start_date == "2025-01-01"
    assert restored.timestamp_unit == "ms"
    assert restored.risk_mode is RiskMode.ATR
    assert restored.input_csv == tmp_path / "strategy.csv"
    assert ignored == ("mean_reversion_rsi_period", "sr_take_profit_mode")


def test_saved_run_snapshot_requires_existing_csvs_when_requested(tmp_path: Path) -> None:
    raw = config_to_dict(
        replace(
            BacktestConfig(),
            input_csv=tmp_path / "missing-strategy.csv",
            intrabar_csv=None,
            use_intrabar_data=False,
        )
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        load_config_snapshot(path, require_paths=True)
    except ValueError as exc:
        assert "Strategy CSV does not exist" in str(exc)
    else:
        raise AssertionError("missing strategy CSV should be rejected")
