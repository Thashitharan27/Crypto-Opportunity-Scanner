from pathlib import Path

import pandas as pd

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.state_transition_worker import StateTransitionBacktestWorker


def test_state_transition_worker_writes_reports(monkeypatch, tmp_path):
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC"),
        "close": [100 + i for i in range(80)],
    })
    trades = pd.DataFrame({
        "strategy_entry_time": pd.to_datetime(["2024-03-01", "2024-03-02"], utc=True),
        "directional_di": [32.0, 28.0],
        "directional_di_change": [1.0, -1.0],
        "pair_net_r": [1.0, -1.0],
    })
    monkeypatch.setattr(
        "crypto_strategy_lab.gui.state_transition_worker.load_backtest_data",
        lambda config, strategy_data: (data, None),
    )
    config = BacktestConfig(use_intrabar_data=False, output_dir=tmp_path)
    worker = StateTransitionBacktestWorker(config)

    worker._write_state_transition_reports({}, trades, pd.DataFrame(), tmp_path)

    destination = tmp_path / "state_transition_research"
    assert (destination / "daily_states.csv").exists()
    assert (destination / "regime_transition_matrix.csv").exists()
    assert (destination / "volatility_transition_matrix.csv").exists()
    assert (destination / "di_state_volatility_trade_performance.csv").exists()
    assert (destination / "current_regime_probabilities.csv").exists()


def test_state_transition_worker_report_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "crypto_strategy_lab.gui.state_transition_worker.load_backtest_data",
        lambda config, strategy_data: (_ for _ in ()).throw(RuntimeError("research failed")),
    )
    config = BacktestConfig(use_intrabar_data=False, output_dir=tmp_path)
    worker = StateTransitionBacktestWorker(config)
    messages = []
    monkeypatch.setattr(worker, "_log", messages.append)

    worker._write_state_transition_reports({}, pd.DataFrame(), pd.DataFrame(), Path(tmp_path))

    assert any("WARNING: state-transition research report failed" in message for message in messages)
