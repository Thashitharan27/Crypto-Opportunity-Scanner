from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.features.state_transition import StateTransitionDailyFeatureProvider
from crypto_strategy_lab.state_transition_prepared_reports import generate_prepared_state_transition_reports
from crypto_strategy_lab.state_transition_research import StateTransitionResearchConfig


def _canonical_days(count: int = 420) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=count, freq="1D", tz="UTC")
    x = np.arange(count, dtype=float)
    close = 100.0 * np.exp(0.0008 * x + 0.08 * np.sin(x / 18.0))
    return pd.DataFrame(
        {
            "period_start": times,
            "available_at": times + pd.Timedelta(days=1),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "source_fingerprint": "state-transition-test",
        }
    )


def _request(frame: pd.DataFrame) -> DataRequest:
    return DataRequest(
        symbol="BTCUSDT",
        start=frame.period_start.iloc[0].to_pydatetime(),
        end=(frame.period_start.iloc[-1] + pd.Timedelta(days=1)).to_pydatetime(),
        strategy_interval="1d",
    )


def test_daily_state_feature_is_available_only_after_completed_day():
    frame = _canonical_days()
    daily = StateTransitionDailyFeatureProvider().compute(
        _request(frame), {DatasetKind.KLINES: frame}, {}
    )
    dates = pd.to_datetime(daily["date"], utc=True)
    available = pd.to_datetime(daily["available_at"], utc=True)
    assert (available == dates + pd.Timedelta(days=1)).all()


def test_future_mutation_cannot_change_past_daily_state_feature():
    frame = _canonical_days()
    provider = StateTransitionDailyFeatureProvider()
    before = provider.compute(_request(frame), {DatasetKind.KLINES: frame}, {})
    cutoff = 300
    changed = frame.copy()
    changed.loc[cutoff + 1 :, "close"] *= 3.0
    future_close = changed.loc[cutoff + 1 :, "close"].to_numpy()[:, None]
    changed.loc[cutoff + 1 :, ["open", "high", "low"]] = np.repeat(future_close, 3, axis=1)
    after = provider.compute(_request(changed), {DatasetKind.KLINES: changed}, {})
    pdt.assert_frame_equal(
        before.iloc[: cutoff + 1].reset_index(drop=True),
        after.iloc[: cutoff + 1].reset_index(drop=True),
        check_dtype=False,
    )


def test_prepared_reports_do_not_rebuild_daily_state(monkeypatch, tmp_path: Path):
    frame = _canonical_days()
    daily = StateTransitionDailyFeatureProvider().compute(
        _request(frame), {DatasetKind.KLINES: frame}, {}
    )
    trades = pd.DataFrame(
        {
            "strategy_entry_time": frame["period_start"].iloc[300:310].to_numpy(),
            "trade_direction": ["LONG", "SHORT"] * 5,
            "directional_di": [12, 18, 22, 27, 31, 35, 9, 14, 24, 33],
            "directional_di_change": [1, -1, 0, 2, -2, 1, 0, 1, -1, 2],
            "pair_net_r": [1, -1, 1, 1, -1, 2, -1, 1, -1, 1],
        }
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("daily_state_frame was recomputed after simulation")

    monkeypatch.setattr("crypto_strategy_lab.state_transition_research.daily_state_frame", forbidden)
    reports = generate_prepared_state_transition_reports(
        daily, trades, tmp_path,
        StateTransitionResearchConfig(minimum_trade_observations=1),
    )
    assert not reports["daily_states"].empty
    assert (tmp_path / "state_transition_research" / "daily_states.csv").exists()
    assert (tmp_path / "trade_list.csv").exists()
