from dataclasses import replace

import pandas as pd

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui import worker as worker_module


def test_pop_heavy_trade_attrs_preserves_other_metadata():
    skipped = [{"reason": "FILTER_REJECTED"}]
    trades = pd.DataFrame({"pair_net_pnl": [1.0]})
    trades.attrs["skipped_signals"] = skipped
    trades.attrs["daily_schedule_stats"] = {"signals": 1}

    detached = worker_module._pop_heavy_trade_attrs(trades)

    assert "skipped_signals" not in trades.attrs
    assert detached["skipped_signals"] is skipped
    assert trades.attrs["daily_schedule_stats"] == {"signals": 1}


def test_isolated_only_runs_each_profile_without_shared_account(tmp_path, monkeypatch):
    calls = []

    class FakeEngine:
        def __init__(self, data, config, intrabar, progress_callback=None, progress_interval=50):
            calls.append([key for key, value in config.strategy_profiles.items() if value.enabled])
            self.progress_callback = progress_callback
            self.progress_interval = progress_interval

        def run(self):
            assert self.progress_callback is not None
            assert self.progress_interval == 50
            self.progress_callback(1, 1, 0, 0)
            return pd.DataFrame()

    monkeypatch.setattr(worker_module, "BacktestEngine", FakeEngine)
    monkeypatch.setattr(worker_module, "update_latest", lambda *_: None)
    config = replace(
        BacktestConfig(),
        strategy_profile_run_mode="ISOLATED_PROFILES",
        output_dir=tmp_path,
        output_run_dir=tmp_path / "isolated_run",
    )
    worker = worker_module.BacktestWorker(config)
    results = []
    worker.finished.connect(lambda summary, trades, equity, path: results.append((summary, trades, equity, path)))
    worker._run_isolated_only(pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"], utc=True)}), None)
    assert len(calls) == 6 and all(len(enabled) == 1 for enabled in calls)
    summary, trades, equity, path = results[0]
    assert summary["strategy_profile_run_mode"] == "ISOLATED_PROFILES"
    assert summary["profiles_tested"] == 6
    assert (path / "strategy_profile_comparison.csv").is_file()
    assert not (path / "blocked_profile_opportunities.csv").exists()


def test_isolated_only_detaches_heavy_attrs_before_statistics(tmp_path, monkeypatch):
    skipped = [{"reason": "FILTER_REJECTED"} for _ in range(100)]

    class FakeEngine:
        def __init__(self, data, config, intrabar, progress_callback=None, progress_interval=50):
            pass

        def run(self):
            trades = pd.DataFrame({"pair_net_r": [1.0], "pair_net_pnl": [1.0]})
            trades.attrs["skipped_signals"] = skipped
            return trades

    def checked_summarize(trades, initial_equity):
        assert "skipped_signals" not in trades.attrs
        return {
            "ending_equity": initial_equity + 1.0,
            "total_return_percentage": 0.1,
            "win_rate": 1.0,
            "profit_factor": 1.0,
            "maximum_drawdown_percentage": 0.0,
        }

    monkeypatch.setattr(worker_module, "BacktestEngine", FakeEngine)
    monkeypatch.setattr(worker_module, "summarize", checked_summarize)
    monkeypatch.setattr(worker_module, "equity_curve", lambda *_: pd.DataFrame())
    monkeypatch.setattr(worker_module, "update_latest", lambda *_: None)
    profiles = {
        key: replace(value, enabled=(key == "bull_long"))
        for key, value in BacktestConfig().strategy_profiles.items()
    }
    config = replace(
        BacktestConfig(),
        strategy_profile_run_mode="ISOLATED_PROFILES",
        strategy_profiles=profiles,
        output_dir=tmp_path,
        output_run_dir=tmp_path / "isolated_run",
    )
    worker_module.BacktestWorker(config)._run_isolated_only(
        pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"], utc=True)}), None
    )
