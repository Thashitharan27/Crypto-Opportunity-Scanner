from pathlib import Path

import numpy as np
import pandas as pd

from crypto_strategy_lab.state_transition_research import (
    RESEARCH_TRADE_COLUMNS,
    StateTransitionResearchConfig,
    daily_state_frame,
    enrich_trade_research_context,
    generate_state_transition_reports,
    regime_alignment_analysis,
    regime_trade_outcome_analysis,
    trade_state_analysis,
    transition_matrix,
)


def _strategy_days(count: int = 420) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=count, freq="1D", tz="UTC")
    # Deterministic trend + cyclical component gives all three return regimes.
    x = np.arange(count, dtype=float)
    close = 100.0 * np.exp(0.0008 * x + 0.08 * np.sin(x / 18.0))
    return pd.DataFrame({"timestamp": timestamps, "close": close})


def test_transition_matrix_probabilities_sum_to_one() -> None:
    report = transition_matrix(["BULL", "BULL", "SIDEWAYS", "BEAR", "BEAR", "BULL"])
    totals = report.groupby("current_state")["probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_daily_states_use_expected_regime_thresholds() -> None:
    cfg = StateTransitionResearchConfig(regime_lookback_days=2, bull_return_threshold=0.05, bear_return_threshold=-0.05)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="1D", tz="UTC"),
            "close": [100, 100, 106, 100, 94, 100, 106],
        }
    )
    states = daily_state_frame(frame, cfg)
    assert "BULL" in set(states["regime_state"])
    assert "BEAR" in set(states["regime_state"])
    assert "SIDEWAYS" in set(states["regime_state"])
    assert (states.loc[states["return_lookback"].isna(), "regime_state"] == "UNKNOWN").all()


def test_trade_state_analysis_buckets_di_and_outcome() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"),
            "volatility_state": ["NORMAL", "ELEVATED", "LOW", "NORMAL"],
        }
    )
    trades = pd.DataFrame(
        {
            "strategy_entry_time": pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"),
            "directional_di": [7.0, 17.0, 32.0, 32.0],
            "directional_di_change": [1.0, -2.0, 0.1, 2.0],
            "pair_net_r": [1.0, -1.0, 2.0, -0.5],
        }
    )
    result = trade_state_analysis(trades, daily, StateTransitionResearchConfig(minimum_trade_observations=1))
    assert result["trades"].sum() == 4
    assert set(result["di_state"]) == {"RISING", "FALLING", "STABLE"}
    assert set(result["volatility_state"]) == {"NORMAL", "ELEVATED", "LOW"}


def test_trade_regime_context_uses_last_completed_day_and_causal_transition_history() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC"),
            "return_lookback": [0.10, -0.10, 0.08, 0.00, -0.08],
            "regime_state": ["BULL", "BEAR", "BULL", "SIDEWAYS", "BEAR"],
        }
    )
    trades = pd.DataFrame(
        {
            "strategy_entry_time": pd.to_datetime(
                ["2026-01-04 00:00:00+00:00", "2026-01-04 12:00:00+00:00", "2026-01-05 12:00:00+00:00"]
            ),
            "trade_direction": ["LONG", "SHORT", "LONG"],
            "pair_net_r": [1.0, -1.0, 2.0],
        }
    )

    enriched = enrich_trade_research_context(trades, daily)

    # Jan 4 entries can only use the fully completed Jan 3 state, never Jan 4.
    assert list(enriched["research_regime_date"][:2]) == ["2026-01-03", "2026-01-03"]
    assert list(enriched["research_regime_state"][:2]) == ["BULL", "BULL"]
    assert np.allclose(enriched["research_regime_return_20d"][:2].to_numpy(float), [0.08, 0.08])

    # Through Jan 3, the only observed transition out of BULL is BULL -> BEAR.
    # The later Jan 4 BULL -> SIDEWAYS outcome must not leak into Jan 4 trades.
    assert list(enriched["research_regime_expected_next_state"][:2]) == ["BEAR", "BEAR"]
    assert np.allclose(enriched["research_regime_expected_next_probability"][:2].to_numpy(float), [1.0, 1.0])
    assert list(enriched["research_regime_trade_alignment"][:2]) == ["AGREE", "COUNTER"]
    assert list(enriched["research_regime_transition_agreement"][:2]) == ["COUNTER", "AGREE"]

    # Jan 5 uses the completed Jan 4 SIDEWAYS state.
    assert enriched.loc[2, "research_regime_state"] == "SIDEWAYS"
    assert enriched.loc[2, "research_regime_trade_alignment"] == "NEUTRAL"


def test_regime_direction_and_alignment_outcome_reports() -> None:
    trades = pd.DataFrame(
        {
            "research_regime_state": ["BULL", "BULL", "BEAR", "BEAR", "SIDEWAYS", "SIDEWAYS"],
            "research_regime_trade_alignment": ["AGREE", "COUNTER", "COUNTER", "AGREE", "NEUTRAL", "NEUTRAL"],
            "trade_direction": ["LONG", "SHORT", "LONG", "SHORT", "LONG", "SHORT"],
            "pair_net_r": [1.0, -1.0, -0.5, 2.0, 0.5, -0.25],
        }
    )
    cfg = StateTransitionResearchConfig(minimum_trade_observations=1)

    direction = regime_trade_outcome_analysis(trades, cfg)
    bull_long = direction[(direction["research_regime_state"] == "BULL") & (direction["trade_direction"] == "LONG")].iloc[0]
    bear_short = direction[(direction["research_regime_state"] == "BEAR") & (direction["trade_direction"] == "SHORT")].iloc[0]
    assert len(direction) == 6
    assert bull_long["trades"] == 1 and bull_long["wins"] == 1 and bull_long["losses"] == 0
    assert bear_short["net_r"] == 2.0

    alignment = regime_alignment_analysis(trades, cfg).set_index("research_regime_trade_alignment")
    assert alignment.loc["AGREE", "trades"] == 2
    assert alignment.loc["AGREE", "wins"] == 2
    assert alignment.loc["COUNTER", "losses"] == 2
    assert alignment.loc["NEUTRAL", "trades"] == 2


def test_generate_reports_writes_research_folder_and_enriched_trade_list(tmp_path: Path) -> None:
    strategy = _strategy_days()
    trades = pd.DataFrame(
        {
            "strategy_entry_time": strategy["timestamp"].iloc[300:310].to_numpy(),
            "trade_direction": ["LONG", "SHORT"] * 5,
            "directional_di": [12, 18, 22, 27, 31, 35, 9, 14, 24, 33],
            "directional_di_change": [1, -1, 0, 2, -2, 1, 0, 1, -1, 2],
            "pair_net_r": [1, -1, 1, 1, -1, 2, -1, 1, -1, 1],
        }
    )
    reports = generate_state_transition_reports(strategy, trades, tmp_path, StateTransitionResearchConfig(minimum_trade_observations=1))
    folder = tmp_path / "state_transition_research"
    assert (folder / "daily_states.csv").exists()
    assert (folder / "regime_transition_matrix.csv").exists()
    assert (folder / "volatility_transition_matrix.csv").exists()
    assert (folder / "di_state_volatility_trade_performance.csv").exists()
    assert (folder / "regime_direction_trade_performance.csv").exists()
    assert (folder / "regime_alignment_trade_performance.csv").exists()
    assert (folder / "current_regime_probabilities.csv").exists()
    assert (tmp_path / "trade_list.csv").exists()
    exported = pd.read_csv(tmp_path / "trade_list.csv")
    assert set(RESEARCH_TRADE_COLUMNS).issubset(exported.columns)
    assert len(exported) == len(trades)
    assert not reports["daily_states"].empty
    assert len(reports["regime_direction_trade_performance"]) == 6
