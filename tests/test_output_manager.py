from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.output_manager import periodic_results, run_folder_name


def with_all_profiles(config, **changes):
    profiles = {key: replace(profile, **changes) for key, profile in config.strategy_profiles.items()}
    return replace(config, strategy_profiles=profiles)


def test_run_folder_name_uses_current_profile_contract_and_run_name():
    cfg = BacktestConfig(input_csv=Path("data/BTCUSDT_15m.csv"), run_name="My Run", atr_period=14)
    name = run_folder_name(cfg, datetime(2026, 7, 23, 12, 34, 56))
    assert name == "My_Run_BTC_15m_ATR14x1_PROFILES-COMBINED_SL2_TP2_2026-07-23_12-34-56"


def test_run_folder_name_describes_profile_partial_stop():
    cfg = BacktestConfig(input_csv=Path("data/BTCUSDT_15m.csv"))
    cfg = with_all_profiles(cfg, partial_stop_enabled=True, sl1_r=2, sl1_close_pct=75, sl2_r=10, reward_risk_ratio=1)
    name = run_folder_name(cfg, datetime(2026, 7, 23, 12, 34, 56))
    assert name == "BTC_15m_ATR14x1_PROFILES-COMBINED_PSL2x75-SL10_TP2_2026-07-23_12-34-56"


def test_run_folder_name_describes_profile_partial_take_profit():
    cfg = BacktestConfig(input_csv=Path("data/BTCUSDT_15m.csv"))
    cfg = with_all_profiles(
        cfg,
        stop_loss_multiple=2,
        partial_profit_enabled=True,
        tp1_r=3,
        tp1_close_pct=50,
        tp2_r=12,
    )
    name = run_folder_name(cfg, datetime(2026, 7, 23, 12, 34, 56))
    assert name == "BTC_15m_ATR14x1_PROFILES-COMBINED_SL2_PTP3x50-TP12_2026-07-23_12-34-56"


def test_run_folder_name_describes_identical_isolated_profiles():
    cfg = BacktestConfig(
        input_csv=Path("data/XRPUSDT_1h.csv"),
        strategy_timeframe_minutes=60,
        telemetry_interval_minutes=60,
        strategy_profile_run_mode="ISOLATED_PROFILES",
    )
    name = run_folder_name(cfg, datetime(2026, 8, 5, 19, 24, 49))
    assert name == "XRP_60m_ATR14x1_PROFILES-ISOLATED_SL2_TP2_2026-08-05_19-24-49"


def test_run_folder_name_marks_different_profile_exits_as_mixed():
    cfg = BacktestConfig(strategy_profile_run_mode="BOTH")
    profiles = dict(cfg.strategy_profiles)
    profiles["bull_long"] = replace(profiles["bull_long"], reward_risk_ratio=2)
    cfg = replace(cfg, strategy_profiles=profiles)
    name = run_folder_name(cfg, datetime(2026, 8, 5, 19, 24, 49))
    assert "_PROFILES-BOTH_MIXED_EXITS_" in name


def test_periodic_results_groups_by_exit_period():
    trades = pd.DataFrame({
        "exit_time": ["2024-01-02", "2024-01-21", "2024-02-02"],
        "pair_net_pnl": [10.0, -2.0, 5.0],
        "pair_net_r": [1.0, -0.2, 0.5],
    })
    monthly = periodic_results(trades, "ME")
    assert list(monthly["pair_count"]) == [2, 1]
    assert list(monthly["net_pnl"]) == [8.0, 5.0]


def test_periodic_results_does_not_use_reset_index_names_keyword(monkeypatch):
    trades = pd.DataFrame({
        "exit_time": ["2024-01-01", "2024-02-01"],
        "pair_net_pnl": [10.0, 5.0],
        "pair_net_r": [1.0, 0.5],
    })
    original_reset_index = pd.DataFrame.reset_index

    def legacy_reset_index(self, *args, **kwargs):
        if "names" in kwargs:
            raise TypeError("DataFrame.reset_index() got an unexpected keyword argument 'names'")
        return original_reset_index(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "reset_index", legacy_reset_index)
    monthly = periodic_results(trades, "ME")
    assert list(monthly.columns) == ["period", "pair_count", "net_pnl", "net_r"]
    assert list(monthly["net_pnl"]) == [10.0, 5.0]


def test_create_run_dir_creates_profile_named_folder_and_latest_pointer(tmp_path):
    from crypto_strategy_lab.output_manager import create_run_dir, update_latest

    cfg = BacktestConfig(output_dir=tmp_path, input_csv=Path("data/BTCUSDT_15m.csv"), atr_period=14)
    run_dir = create_run_dir(cfg)
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("BTC_15m_ATR14x1_PROFILES-COMBINED_SL2_TP2_")
    assert (run_dir / "charts").is_dir()

    (run_dir / "trade_list.csv").write_text("trade\n")
    update_latest(tmp_path, run_dir)
    assert (tmp_path / "latest" / "trade_list.csv").exists()


def test_periodic_results_accepts_newer_month_year_aliases_on_installed_pandas():
    trades = pd.DataFrame({
        "exit_time": ["2024-01-01", "2024-02-01"],
        "pair_net_pnl": [10.0, 5.0],
        "pair_net_r": [1.0, 0.5],
    })
    assert list(periodic_results(trades, "ME")["net_pnl"]) == [10.0, 5.0]
    assert list(periodic_results(trades, "YE")["net_pnl"]) == [15.0]


def test_no_unresolved_merge_markers_remain():
    root = Path(__file__).resolve().parents[1]
    excluded_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "output", "outputs", "paper_output"}
    offenders = []
    for path in root.rglob("*"):
        if path.is_file() and not excluded_dirs.intersection(path.parts) and path.suffix not in {".pyc", ".png", ".csv"}:
            text = path.read_text(errors="ignore")
            if any(line.startswith(("<<<<<<<", "=======", ">>>>>>>")) for line in text.splitlines()):
                offenders.append(path.relative_to(root))
    assert offenders == []
