"""State-transition report writer for already prepared daily research features."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_strategy_lab.state_transition_research import (
    REGIME_STATES,
    StateTransitionResearchConfig,
    current_state_probabilities,
    enrich_trade_research_context,
    regime_alignment_analysis,
    regime_trade_outcome_analysis,
    trade_state_analysis,
    transition_matrix,
)


def generate_prepared_state_transition_reports(
    daily_states: pd.DataFrame,
    trades: pd.DataFrame,
    run_dir: Path,
    config: StateTransitionResearchConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Write research reports without rebuilding daily state from strategy candles."""
    cfg = config or StateTransitionResearchConfig()
    daily = daily_states.copy()
    required = {"date", "regime_state", "volatility_state"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Prepared daily state-transition feature is missing columns: {missing}")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    if "available_at" in daily.columns:
        available = pd.to_datetime(daily["available_at"], utc=True)
        if bool((available <= daily["date"]).any()):
            raise ValueError("Prepared daily state-transition availability is not causal")

    destination = Path(run_dir) / "state_transition_research"
    destination.mkdir(parents=True, exist_ok=True)
    regime = transition_matrix(
        daily.loc[daily["regime_state"].isin(REGIME_STATES), "regime_state"],
        cfg.minimum_state_observations,
    )
    volatility = transition_matrix(
        daily.loc[daily["volatility_state"] != "UNKNOWN", "volatility_state"],
        cfg.minimum_state_observations,
    )
    enriched_trades = enrich_trade_research_context(trades, daily)
    trade_states = trade_state_analysis(trades, daily, cfg)
    regime_direction = regime_trade_outcome_analysis(enriched_trades, cfg)
    regime_alignment = regime_alignment_analysis(enriched_trades, cfg)
    current = current_state_probabilities(daily, regime)

    daily.to_csv(destination / "daily_states.csv", index=False)
    regime.to_csv(destination / "regime_transition_matrix.csv", index=False)
    volatility.to_csv(destination / "volatility_transition_matrix.csv", index=False)
    trade_states.to_csv(destination / "di_state_volatility_trade_performance.csv", index=False)
    regime_direction.to_csv(destination / "regime_direction_trade_performance.csv", index=False)
    regime_alignment.to_csv(destination / "regime_alignment_trade_performance.csv", index=False)
    current.to_csv(destination / "current_regime_probabilities.csv", index=False)
    enriched_trades.to_csv(Path(run_dir) / "trade_list.csv", index=False)

    return {
        "daily_states": daily,
        "regime_transition_matrix": regime,
        "volatility_transition_matrix": volatility,
        "di_state_volatility_trade_performance": trade_states,
        "regime_direction_trade_performance": regime_direction,
        "regime_alignment_trade_performance": regime_alignment,
        "current_regime_probabilities": current,
        "enriched_trade_list": enriched_trades,
    }