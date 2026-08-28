from types import SimpleNamespace

import numpy as np
import pandas as pd

import crypto_strategy_lab.research_adapters as adapters


def test_native_simulator_releases_duplicate_rule_rejections_after_signal_capture(monkeypatch):
    """Heavy Entry Rule rejection history must not ride on trades into reporting."""
    rejection_count = 5_000
    times = pd.date_range(
        "2024-01-01", periods=rejection_count + 1, freq="h", tz="UTC"
    ).tz_localize(None).to_numpy(dtype="datetime64[ns]")
    availability = times + np.timedelta64(1, "h")
    prepared = SimpleNamespace(
        timestamp=times,
        decision_available_at=availability,
    )

    skipped_signals = [
        {
            "strategy_candle_open_time": times[index],
            "strategy_entry_price": 100.0 + index / 1000.0,
            "plus_di": 35.0,
            "minus_di": 15.0,
            "di_spread": 20.0,
            "di_pressure_state": "EXPANDING",
            "entry_filter_reason": "Strategy profile rejected by entry rules",
            "strategy_profile_key": "bull_long",
            # Representative legacy diagnostics that canonical signals do not
            # need to duplicate on the completed-trades DataFrame metadata.
            "diagnostic_payload": tuple(range(32)),
        }
        for index in range(rejection_count)
    ]

    trades = pd.DataFrame(
        [
            {
                "pair_id": 1,
                "research_signal_index": rejection_count,
                "research_signal_candle_open_time": times[-1],
                "research_signal_available_at": availability[-1],
                "side": "LONG",
                "strategy_profile_key": "bull_long",
                "strategy_entry_price": 105.0,
            }
        ]
    )
    trades.attrs["skipped_signals"] = skipped_signals
    trades.attrs["daily_schedule_stats"] = {
        "scheduled_entry_opportunities": 1,
        "trades_opened_on_schedule": 1,
    }

    fake_engine = SimpleNamespace(
        skipped_signals=skipped_signals,
        telemetry_rows=(),
        run=lambda: trades,
    )

    class FakeEngineFactory:
        @staticmethod
        def from_prepared(*args, **kwargs):
            return fake_engine

    monkeypatch.setattr(
        adapters, "RuleAwareDataLakeProductionBacktestEngine", FakeEngineFactory
    )
    monkeypatch.setattr(
        adapters, "native_simulator_config", lambda *args, **kwargs: object()
    )

    simulator = adapters.NativeSimulator()
    result = simulator.run(
        prepared,
        None,
        adapters.BoundNativeStrategyPolicy(object()),
        object(),
        data_config=object(),
        feature_config=object(),
    )

    assert result is trades
    assert "skipped_signals" not in result.attrs
    assert result.attrs["daily_schedule_stats"]["trades_opened_on_schedule"] == 1
    assert fake_engine.skipped_signals == []

    signals = simulator.last_signals
    assert signals is not None
    assert len(signals) == rejection_count + 1
    assert int(signals["decision"].eq("REJECT").sum()) == rejection_count
    assert int(signals["decision"].eq("ENTER").sum()) == 1
    assert signals.loc[signals["decision"].eq("REJECT"), "reason_code"].notna().all()
    assert "diagnostic_payload" not in signals.columns
