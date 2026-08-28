"""Causal Markov/state-transition research for Crypto Strategy Lab.

This module is intentionally research-only. It does not change trade selection,
position sizing, stops, targets, or execution. It produces diagnostics that can
later be evaluated with walk-forward tests before any strategy integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REGIME_STATES = ("BEAR", "SIDEWAYS", "BULL")
REGIME_REPORT_ORDER = ("BULL", "BEAR", "SIDEWAYS")
VOLATILITY_STATES = ("LOW", "NORMAL", "ELEVATED")
DI_STATES = ("FALLING", "STABLE", "RISING")
TRADE_DIRECTIONS = ("LONG", "SHORT")
TRADE_ALIGNMENTS = ("AGREE", "COUNTER", "NEUTRAL")

RESEARCH_TRADE_COLUMNS = (
    "research_regime_state",
    "research_regime_date",
    "research_regime_return_20d",
    "research_regime_trade_alignment",
    "research_regime_expected_next_state",
    "research_regime_expected_next_probability",
    "research_regime_transition_agreement",
)


@dataclass(frozen=True)
class StateTransitionResearchConfig:
    """Settings for analysis-only state-transition research."""

    regime_lookback_days: int = 20
    bull_return_threshold: float = 0.05
    bear_return_threshold: float = -0.05
    volatility_lookback_days: int = 20
    volatility_reference_days: int = 252
    volatility_low_quantile: float = 0.33
    volatility_high_quantile: float = 0.67
    di_bucket_edges: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, float("inf"))
    di_stable_tolerance: float = 0.5
    minimum_state_observations: int = 30
    minimum_trade_observations: int = 20

    def __post_init__(self) -> None:
        if self.regime_lookback_days < 1 or self.volatility_lookback_days < 2:
            raise ValueError("state-transition lookbacks must be positive")
        if self.volatility_reference_days < self.volatility_lookback_days:
            raise ValueError("volatility_reference_days must be >= volatility_lookback_days")
        if not self.bear_return_threshold < self.bull_return_threshold:
            raise ValueError("bear threshold must be lower than bull threshold")
        if not 0 < self.volatility_low_quantile < self.volatility_high_quantile < 1:
            raise ValueError("volatility quantiles must satisfy 0 < low < high < 1")
        if self.minimum_state_observations < 1 or self.minimum_trade_observations < 1:
            raise ValueError("minimum samples must be positive")


def _timestamp_column(frame: pd.DataFrame) -> str:
    for candidate in ("timestamp", "strategy_entry_time", "entry_time", "signal_time"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("No supported timestamp column found")


def _close_column(frame: pd.DataFrame) -> str:
    for candidate in ("close", "strategy_close", "entry_close"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("No supported close-price column found")


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def daily_state_frame(strategy_data: pd.DataFrame, config: StateTransitionResearchConfig | None = None) -> pd.DataFrame:
    """Create causal daily regime and volatility states from strategy candles.

    Volatility boundaries use only information available before each day: the
    rolling volatility series is compared with rolling historical quantiles
    shifted by one day. This avoids using future observations to label the
    current state.
    """
    cfg = config or StateTransitionResearchConfig()
    if strategy_data.empty:
        return pd.DataFrame(columns=["date", "close", "return_1d", "return_lookback", "regime_state", "volatility", "volatility_state"])

    ts_col = _timestamp_column(strategy_data)
    close_col = _close_column(strategy_data)
    frame = pd.DataFrame({"timestamp": _as_utc(strategy_data[ts_col]), "close": pd.to_numeric(strategy_data[close_col], errors="coerce")})
    frame = frame.dropna().sort_values("timestamp")
    daily = frame.set_index("timestamp")["close"].resample("1D").last().dropna().to_frame()
    daily["return_1d"] = daily["close"].pct_change()
    daily["return_lookback"] = daily["close"].pct_change(cfg.regime_lookback_days)
    daily["regime_state"] = np.select(
        [daily["return_lookback"] >= cfg.bull_return_threshold, daily["return_lookback"] <= cfg.bear_return_threshold],
        ["BULL", "BEAR"],
        default="SIDEWAYS",
    )
    daily.loc[daily["return_lookback"].isna(), "regime_state"] = "UNKNOWN"

    daily["volatility"] = daily["return_1d"].rolling(cfg.volatility_lookback_days, min_periods=cfg.volatility_lookback_days).std()
    prior_vol = daily["volatility"].shift(1)
    low = prior_vol.rolling(cfg.volatility_reference_days, min_periods=max(cfg.volatility_lookback_days, 30)).quantile(cfg.volatility_low_quantile)
    high = prior_vol.rolling(cfg.volatility_reference_days, min_periods=max(cfg.volatility_lookback_days, 30)).quantile(cfg.volatility_high_quantile)
    daily["volatility_state"] = np.select(
        [daily["volatility"] <= low, daily["volatility"] >= high],
        ["LOW", "ELEVATED"],
        default="NORMAL",
    )
    daily.loc[daily["volatility"].isna(), "volatility_state"] = "UNKNOWN"
    return daily.reset_index().rename(columns={"timestamp": "date"})


def transition_counts(states: Iterable[str]) -> pd.DataFrame:
    values = pd.Series(list(states), dtype="string").dropna()
    if len(values) < 2:
        return pd.DataFrame(columns=["current_state", "next_state", "count"])
    pairs = pd.DataFrame({"current_state": values.iloc[:-1].to_numpy(), "next_state": values.iloc[1:].to_numpy()})
    return pairs.value_counts(sort=False).rename("count").reset_index()


def transition_matrix(states: Iterable[str], minimum_observations: int = 1) -> pd.DataFrame:
    """Return long-form transition probabilities with sample counts."""
    counts = transition_counts(states)
    if counts.empty:
        return pd.DataFrame(columns=["current_state", "next_state", "count", "current_state_observations", "probability"])
    totals = counts.groupby("current_state")["count"].sum().rename("current_state_observations")
    result = counts.join(totals, on="current_state")
    result["probability"] = result["count"] / result["current_state_observations"]
    result["meets_minimum_sample"] = result["current_state_observations"] >= minimum_observations
    return result.sort_values(["current_state", "probability"], ascending=[True, False]).reset_index(drop=True)


def _di_value(trades: pd.DataFrame) -> pd.Series:
    if "directional_di" in trades.columns:
        return pd.to_numeric(trades["directional_di"], errors="coerce")
    if {"plus_di", "minus_di"}.issubset(trades.columns):
        return pd.concat([
            pd.to_numeric(trades["plus_di"], errors="coerce"),
            pd.to_numeric(trades["minus_di"], errors="coerce"),
        ], axis=1).max(axis=1)
    if "di_spread" in trades.columns:
        return pd.to_numeric(trades["di_spread"], errors="coerce")
    return pd.Series(np.nan, index=trades.index, dtype=float)


def _di_change(trades: pd.DataFrame) -> pd.Series:
    for candidate in ("directional_di_change", "di_change", "di_spread_entry_5bar_change", "di_spread_change"):
        if candidate in trades.columns:
            return pd.to_numeric(trades[candidate], errors="coerce")
    return pd.Series(np.nan, index=trades.index, dtype=float)


def _trade_r(trades: pd.DataFrame) -> pd.Series:
    for candidate in ("pair_net_r", "pair_net_account_r", "net_r"):
        if candidate in trades.columns:
            return pd.to_numeric(trades[candidate], errors="coerce")
    return pd.Series(np.nan, index=trades.index, dtype=float)


def _trade_direction(trades: pd.DataFrame) -> pd.Series:
    for candidate in ("trade_direction", "side", "sizing_direction", "di_sizing_direction"):
        if candidate in trades.columns:
            values = trades[candidate].astype("string").str.upper()
            return values.where(values.isin(TRADE_DIRECTIONS))
    return pd.Series(pd.NA, index=trades.index, dtype="string")


def _di_bucket(values: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f"{left:g}+" if np.isinf(right) else f"{left:g}-{right:g}")
    return pd.cut(values, bins=list(edges), labels=labels, include_lowest=True, right=False)


def _direction_alignment(regime: pd.Series, direction: pd.Series) -> pd.Series:
    regime_values = regime.astype("string").str.upper()
    direction_values = direction.astype("string").str.upper()
    result = pd.Series(pd.NA, index=regime.index, dtype="string")
    result.loc[
        ((regime_values == "BULL") & (direction_values == "LONG"))
        | ((regime_values == "BEAR") & (direction_values == "SHORT"))
    ] = "AGREE"
    result.loc[
        ((regime_values == "BULL") & (direction_values == "SHORT"))
        | ((regime_values == "BEAR") & (direction_values == "LONG"))
    ] = "COUNTER"
    result.loc[(regime_values == "SIDEWAYS") & direction_values.isin(TRADE_DIRECTIONS)] = "NEUTRAL"
    return result


def _causal_regime_context(daily_states: pd.DataFrame) -> pd.DataFrame:
    """Build per-day regime context using transition history known by that close.

    For a state dated D, its close becomes usable at D+1 00:00 UTC. Expected-next
    probabilities use only transitions whose destination state is already known
    by D; future transitions are never included.
    """
    columns = [
        "state_available_at",
        "research_regime_state",
        "research_regime_date",
        "research_regime_return_20d",
        "research_regime_expected_next_state",
        "research_regime_expected_next_probability",
    ]
    required = {"date", "regime_state", "return_lookback"}
    if daily_states.empty or not required.issubset(daily_states.columns):
        return pd.DataFrame(columns=columns)

    history = daily_states[["date", "regime_state", "return_lookback"]].copy()
    history["date"] = _as_utc(history["date"]).dt.floor("D")
    history["regime_state"] = history["regime_state"].astype("string").str.upper()
    history["return_lookback"] = pd.to_numeric(history["return_lookback"], errors="coerce")
    history = history.dropna(subset=["date", "regime_state"])
    history = history.loc[history["regime_state"].isin(REGIME_STATES)].sort_values("date").reset_index(drop=True)

    counts = {state: {next_state: 0 for next_state in REGIME_STATES} for state in REGIME_STATES}
    expected_states: list[object] = []
    expected_probabilities: list[float] = []

    for i, row in history.iterrows():
        current = str(row["regime_state"])
        if i > 0:
            previous = str(history.iloc[i - 1]["regime_state"])
            if previous in counts and current in counts[previous]:
                counts[previous][current] += 1

        current_counts = counts.get(current, {})
        total = int(sum(current_counts.values()))
        if total:
            highest = max(current_counts.values())
            expected = next(state for state in REGIME_STATES if current_counts.get(state, 0) == highest)
            expected_states.append(expected)
            expected_probabilities.append(highest / total)
        else:
            expected_states.append(pd.NA)
            expected_probabilities.append(np.nan)

    context = pd.DataFrame(
        {
            "state_available_at": history["date"] + pd.Timedelta(days=1),
            "research_regime_state": history["regime_state"].astype("string"),
            "research_regime_date": history["date"].dt.strftime("%Y-%m-%d"),
            "research_regime_return_20d": history["return_lookback"],
            "research_regime_expected_next_state": pd.Series(expected_states, dtype="string"),
            "research_regime_expected_next_probability": expected_probabilities,
        }
    )
    return context[columns]


def enrich_trade_research_context(
    trades: pd.DataFrame,
    daily_states: pd.DataFrame,
) -> pd.DataFrame:
    """Add causal 20-day regime research telemetry to each completed trade row."""
    enriched = trades.copy()
    for column in RESEARCH_TRADE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA

    if enriched.empty:
        return enriched

    ts_col = _timestamp_column(enriched)
    entry_times = _as_utc(enriched[ts_col])
    context = _causal_regime_context(daily_states)

    if context.empty or entry_times.notna().sum() == 0:
        enriched["research_regime_expected_next_probability"] = pd.to_numeric(
            enriched["research_regime_expected_next_probability"], errors="coerce"
        )
        return enriched

    valid = pd.DataFrame(
        {
            "_row_position": np.arange(len(enriched), dtype=int),
            "_trade_entry_time": entry_times.to_numpy(),
        }
    )
    valid = valid.loc[valid["_trade_entry_time"].notna()].sort_values("_trade_entry_time")
    merged = pd.merge_asof(
        valid,
        context.sort_values("state_available_at"),
        left_on="_trade_entry_time",
        right_on="state_available_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_position")

    for column in (
        "research_regime_state",
        "research_regime_date",
        "research_regime_return_20d",
        "research_regime_expected_next_state",
        "research_regime_expected_next_probability",
    ):
        values = pd.Series(pd.NA, index=np.arange(len(enriched)), dtype="object")
        values.iloc[merged["_row_position"].to_numpy(dtype=int)] = merged[column].to_numpy()
        enriched[column] = values.to_numpy()

    directions = _trade_direction(enriched)
    enriched["research_regime_trade_alignment"] = _direction_alignment(
        enriched["research_regime_state"], directions
    )
    enriched["research_regime_transition_agreement"] = _direction_alignment(
        enriched["research_regime_expected_next_state"], directions
    )
    enriched["research_regime_return_20d"] = pd.to_numeric(enriched["research_regime_return_20d"], errors="coerce")
    enriched["research_regime_expected_next_probability"] = pd.to_numeric(
        enriched["research_regime_expected_next_probability"], errors="coerce"
    )
    return enriched


def trade_state_analysis(
    trades: pd.DataFrame,
    daily_states: pd.DataFrame,
    config: StateTransitionResearchConfig | None = None,
) -> pd.DataFrame:
    """Summarize trade outcome by DI bucket × DI movement × volatility state."""
    cfg = config or StateTransitionResearchConfig()
    columns = ["di_bucket", "di_state", "volatility_state", "trades", "wins", "win_rate", "net_r", "average_r", "meets_minimum_sample"]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    ts_col = _timestamp_column(trades)
    frame = pd.DataFrame(index=trades.index)
    frame["date"] = _as_utc(trades[ts_col]).dt.floor("D")
    frame["di_value"] = _di_value(trades)
    frame["di_change"] = _di_change(trades)
    frame["trade_r"] = _trade_r(trades)
    frame["di_bucket"] = _di_bucket(frame["di_value"], cfg.di_bucket_edges).astype("string")
    frame["di_state"] = np.select(
        [frame["di_change"] > cfg.di_stable_tolerance, frame["di_change"] < -cfg.di_stable_tolerance],
        ["RISING", "FALLING"],
        default="STABLE",
    )

    state_map = daily_states[["date", "volatility_state"]].copy() if not daily_states.empty else pd.DataFrame(columns=["date", "volatility_state"])
    state_map["date"] = _as_utc(state_map["date"]).dt.floor("D")
    frame = frame.merge(state_map, on="date", how="left")
    frame["volatility_state"] = frame["volatility_state"].fillna("UNKNOWN")
    frame = frame.dropna(subset=["trade_r", "di_bucket"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    grouped = frame.groupby(["di_bucket", "di_state", "volatility_state"], observed=True, dropna=False)["trade_r"]
    result = grouped.agg(trades="size", wins=lambda x: int((x > 0).sum()), net_r="sum", average_r="mean").reset_index()
    result["win_rate"] = result["wins"] / result["trades"]
    result["meets_minimum_sample"] = result["trades"] >= cfg.minimum_trade_observations
    return result[columns].sort_values(["di_bucket", "di_state", "volatility_state"]).reset_index(drop=True)


def _outcome_summary(
    frame: pd.DataFrame,
    group_column: str,
    group_order: tuple[str, ...],
    config: StateTransitionResearchConfig,
) -> pd.DataFrame:
    columns = [group_column, "trades", "wins", "losses", "win_rate", "net_r", "average_r", "meets_minimum_sample"]
    work = pd.DataFrame(
        {
            group_column: frame.get(group_column, pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string"),
            "trade_r": _trade_r(frame),
        }
    ).dropna(subset=[group_column, "trade_r"])

    if work.empty:
        result = pd.DataFrame({group_column: list(group_order)})
        result["trades"] = 0
        result["wins"] = 0
        result["losses"] = 0
        result["win_rate"] = np.nan
        result["net_r"] = 0.0
        result["average_r"] = np.nan
        result["meets_minimum_sample"] = False
        return result[columns]

    grouped = work.groupby(group_column, observed=True)["trade_r"].agg(
        trades="size",
        wins=lambda x: int((x > 0).sum()),
        losses=lambda x: int((x < 0).sum()),
        net_r="sum",
        average_r="mean",
    )
    result = grouped.reindex(list(group_order)).reset_index()
    for column in ("trades", "wins", "losses"):
        result[column] = result[column].fillna(0).astype(int)
    result["net_r"] = result["net_r"].fillna(0.0)
    result["win_rate"] = np.where(result["trades"] > 0, result["wins"] / result["trades"], np.nan)
    result["meets_minimum_sample"] = result["trades"] >= config.minimum_trade_observations
    return result[columns]


def regime_trade_outcome_analysis(
    enriched_trades: pd.DataFrame,
    config: StateTransitionResearchConfig | None = None,
) -> pd.DataFrame:
    """Show LONG vs SHORT outcomes for each causal 20-day regime state."""
    cfg = config or StateTransitionResearchConfig()
    columns = [
        "research_regime_state",
        "trade_direction",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "net_r",
        "average_r",
        "meets_minimum_sample",
    ]
    if enriched_trades.empty:
        index = pd.MultiIndex.from_product(
            [REGIME_REPORT_ORDER, TRADE_DIRECTIONS],
            names=["research_regime_state", "trade_direction"],
        )
        result = index.to_frame(index=False)
        result[["trades", "wins", "losses"]] = 0
        result["win_rate"] = np.nan
        result["net_r"] = 0.0
        result["average_r"] = np.nan
        result["meets_minimum_sample"] = False
        return result[columns]

    work = pd.DataFrame(
        {
            "research_regime_state": enriched_trades["research_regime_state"].astype("string").str.upper(),
            "trade_direction": _trade_direction(enriched_trades),
            "trade_r": _trade_r(enriched_trades),
        }
    )
    work = work[
        work["research_regime_state"].isin(REGIME_REPORT_ORDER)
        & work["trade_direction"].isin(TRADE_DIRECTIONS)
    ].dropna(subset=["trade_r"])

    index = pd.MultiIndex.from_product(
        [REGIME_REPORT_ORDER, TRADE_DIRECTIONS],
        names=["research_regime_state", "trade_direction"],
    )
    if work.empty:
        grouped = pd.DataFrame(index=index, columns=["trades", "wins", "losses", "net_r", "average_r"])
    else:
        grouped = work.groupby(["research_regime_state", "trade_direction"], observed=True)["trade_r"].agg(
            trades="size",
            wins=lambda x: int((x > 0).sum()),
            losses=lambda x: int((x < 0).sum()),
            net_r="sum",
            average_r="mean",
        ).reindex(index)

    result = grouped.reset_index()
    for column in ("trades", "wins", "losses"):
        result[column] = result[column].fillna(0).astype(int)
    result["net_r"] = result["net_r"].fillna(0.0)
    result["win_rate"] = np.where(result["trades"] > 0, result["wins"] / result["trades"], np.nan)
    result["meets_minimum_sample"] = result["trades"] >= cfg.minimum_trade_observations
    return result[columns]


def regime_alignment_analysis(
    enriched_trades: pd.DataFrame,
    config: StateTransitionResearchConfig | None = None,
) -> pd.DataFrame:
    """Summarize AGREE/COUNTER/NEUTRAL trade outcomes for the causal regime."""
    cfg = config or StateTransitionResearchConfig()
    return _outcome_summary(enriched_trades, "research_regime_trade_alignment", TRADE_ALIGNMENTS, cfg)


def current_state_probabilities(daily_states: pd.DataFrame, transition_report: pd.DataFrame) -> pd.DataFrame:
    """Return the transition probabilities applicable to the most recent state."""
    if daily_states.empty or transition_report.empty:
        return pd.DataFrame(columns=transition_report.columns)
    current = str(daily_states.iloc[-1]["regime_state"])
    return transition_report.loc[transition_report["current_state"] == current].copy().reset_index(drop=True)


def generate_state_transition_reports(
    strategy_data: pd.DataFrame,
    trades: pd.DataFrame,
    run_dir: Path,
    config: StateTransitionResearchConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Write research-only state-transition CSV reports into a backtest run."""
    cfg = config or StateTransitionResearchConfig()
    destination = Path(run_dir) / "state_transition_research"
    destination.mkdir(parents=True, exist_ok=True)

    daily = daily_state_frame(strategy_data, cfg)
    regime = transition_matrix(
        daily.loc[daily["regime_state"].isin(REGIME_STATES), "regime_state"],
        cfg.minimum_state_observations,
    )
    volatility = transition_matrix(daily.loc[daily["volatility_state"] != "UNKNOWN", "volatility_state"], cfg.minimum_state_observations)
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

    # The state-transition GUI is a research-only wrapper. Overwrite its normal
    # trade export with the same rows plus causal research telemetry; no engine
    # state or trade decision is modified.
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
