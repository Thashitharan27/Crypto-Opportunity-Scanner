"""Strategy-aware opportunity sampling for resilience research.

This module is deliberately separate from portfolio simulation.  It evaluates
only entries that the configured strategy would actually accept, but removes
portfolio suppression so viable opportunities can overlap.  Every synthetic
trade starts from the same research equity and uses the native execution engine
for stops, targets, timeout, fees, slippage and intrabar resolution.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from crypto_strategy_lab.config import EntryMode
from crypto_strategy_lab.research_sampling_fast import ResearchSamplingFastExitMixin
from crypto_strategy_lab.rule_native_engine import RuleAwareDataLakeProductionBacktestEngine


RESEARCH_SAMPLING_VERSION = "STRATEGY_OPPORTUNITY_V1"
RESEARCH_SAMPLING_MODES = {
    "PORTFOLIO",
    "EVERY_VIABLE_ENTRY",
    "FIXED_INTERVAL",
    "EPISODE_FIRST",
}


class StrategyResearchSamplingEngine(
    ResearchSamplingFastExitMixin,
    RuleAwareDataLakeProductionBacktestEngine,
):
    """Native engine variant that ignores portfolio overlap suppression only."""

    def _should_enter(self, i):
        # Research observations are not a portfolio. Existing open observations,
        # timeout history and max-active-pair limits must not suppress a new
        # opportunity. Strategy profile availability and the normal downstream
        # Entry/Veto rule evaluation remain authoritative.
        if not np.isfinite(self.risk[i]) or self.risk[i] <= 0:
            return False
        if not self._in_trading_window(i):
            return False
        return self._profile_context(i) is not None

    def _collect_closed_pairs(self, force=False):
        # Keep all observations on fixed independent research equity. The normal
        # engine may calculate a trade PnL, but no completed observation may alter
        # the sizing base of a later overlapping observation.
        still = []
        for pair in self.active_pairs:
            if force or not pair.is_open:
                pair.equity_after_trade = pair.equity_before_trade
                self.completed_pairs.append(pair)
            else:
                still.append(pair)
        self.active_pairs = still


def research_native_config(native_config, prepared_rows: int):
    """Remove portfolio constraints while preserving strategy and trade semantics."""
    return replace(
        native_config,
        entry_mode=EntryMode.EVERY_N_CANDLES,
        entry_interval=1,
        enable_daily_entry_schedule=False,
        max_active_pairs=max(1, int(prepared_rows) * 2 + 2),
        max_combined_effective_leverage=None,
    )


def _exit_reason(row: pd.Series) -> str:
    side = str(row.get("side", "")).lower()
    return str(row.get(f"{side}_exit_reason", "")).upper()


def _annotate_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    """Label uninterrupted stretches of the same viable configured setup."""
    if frame.empty:
        result = frame.copy()
        for name, dtype in (
            ("research_episode_id", "string"),
            ("research_episode_entry_number", "int64"),
            ("research_episode_viable_entries", "int64"),
        ):
            result[name] = pd.Series(dtype=dtype)
        return result

    required = {"research_signal_index", "side"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"research sampling rows are missing episode fields: {missing}")

    result = frame.copy()
    result["research_signal_index"] = pd.to_numeric(
        result["research_signal_index"], errors="raise"
    ).astype("int64")
    profile = result.get(
        "strategy_profile_key", pd.Series("", index=result.index, dtype="string")
    ).fillna("").astype(str)
    side = result["side"].fillna("").astype(str).str.upper()
    result = result.assign(_profile_sort=profile, _side_sort=side).sort_values(
        ["research_signal_index", "_profile_sort", "_side_sort"], kind="stable"
    ).reset_index(drop=True)

    previous_index = result["research_signal_index"].shift(1)
    previous_profile = result["_profile_sort"].shift(1)
    previous_side = result["_side_sort"].shift(1)
    new_episode = (
        previous_index.isna()
        | result["research_signal_index"].ne(previous_index + 1)
        | result["_profile_sort"].ne(previous_profile)
        | result["_side_sort"].ne(previous_side)
    )
    episode_number = new_episode.cumsum().astype("int64")
    result["research_episode_id"] = episode_number.map(
        lambda value: f"episode-{int(value):06d}"
    )
    result["research_episode_entry_number"] = (
        result.groupby("research_episode_id", sort=False).cumcount() + 1
    ).astype("int64")
    result["research_episode_viable_entries"] = result.groupby(
        "research_episode_id", sort=False
    )["research_episode_id"].transform("size").astype("int64")
    return result.drop(columns=["_profile_sort", "_side_sort"])


def _resolved_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Treat right-censored end-of-data observations as unknown, not losses."""
    if frame.empty:
        return frame.copy()
    mask = frame.apply(_exit_reason, axis=1).ne("END_OF_DATA")
    return frame.loc[mask].copy().reset_index(drop=True)


def _select_sampling_mode(
    viable: pd.DataFrame,
    mode: str,
    interval_candles: int,
) -> pd.DataFrame:
    mode = str(mode).upper()
    interval = int(interval_candles)
    if mode not in RESEARCH_SAMPLING_MODES:
        raise ValueError(f"unsupported research sampling mode: {mode}")
    if interval <= 0:
        raise ValueError("research sampling interval must be positive")
    if mode == "PORTFOLIO":
        return viable.iloc[0:0].copy()
    if mode == "EVERY_VIABLE_ENTRY":
        return viable.copy().reset_index(drop=True)
    if mode == "EPISODE_FIRST":
        return viable.loc[viable["research_episode_entry_number"].eq(1)].copy().reset_index(drop=True)
    return viable.loc[
        (viable["research_episode_entry_number"] - 1).mod(interval).eq(0)
    ].copy().reset_index(drop=True)


def generate_strategy_research_samples(
    prepared,
    intrabar,
    native_config,
    *,
    mode: str,
    interval_candles: int = 1,
) -> pd.DataFrame:
    """Evaluate configured strategy opportunities without portfolio suppression."""
    normalized_mode = str(mode).upper()
    if normalized_mode not in RESEARCH_SAMPLING_MODES:
        raise ValueError(f"unsupported research sampling mode: {normalized_mode}")
    interval = int(interval_candles)
    if interval <= 0:
        raise ValueError("research sampling interval must be positive")
    if normalized_mode == "PORTFOLIO":
        empty = pd.DataFrame()
        empty.attrs["research_sampling"] = {
            "enabled": False,
            "mode": normalized_mode,
            "interval_candles": interval,
        }
        return empty

    config = research_native_config(native_config, len(prepared))
    engine = StrategyResearchSamplingEngine.from_prepared(prepared, intrabar, config)
    raw = engine.run()
    exit_optimization = engine.research_exit_optimization_stats()
    raw = _annotate_episodes(raw)
    viable_episode_count = int(raw["research_episode_id"].nunique()) if not raw.empty else 0
    resolved = _resolved_samples(raw)
    selected = _select_sampling_mode(resolved, normalized_mode, interval)

    if not selected.empty:
        selected["research_sampling_version"] = RESEARCH_SAMPLING_VERSION
        selected["research_sampling_mode"] = normalized_mode
        selected["research_sampling_interval_candles"] = interval
        selected["research_sampling_overlap_allowed"] = True
        selected["research_sampling_independent_equity"] = True
        selected["research_sampling_population"] = "STRATEGY_VIABLE"
        selected["research_sample_id"] = (
            selected["research_signal_index"].astype("int64").astype(str)
            + "-"
            + selected["side"].astype(str).str.upper()
            + "-"
            + selected["research_episode_id"].astype(str)
        )

    selected.attrs["research_sampling"] = {
        "version": RESEARCH_SAMPLING_VERSION,
        "enabled": True,
        "mode": normalized_mode,
        "interval_candles": interval,
        "strategy_timeframe_minutes": int(native_config.strategy_timeframe_minutes),
        "overlap_allowed": True,
        "independent_equity": True,
        "portfolio_exposure_ignored": True,
        "combined_leverage_cap_ignored": True,
        "per_trade_execution_semantics_preserved": True,
        **exit_optimization,
        "viable_entries": int(len(raw)),
        "resolved_viable_entries": int(len(resolved)),
        "selected_entries": int(len(selected)),
        "viable_episodes": viable_episode_count,
        "end_of_data_samples_censored": int(len(raw) - len(resolved)),
        "bayesian_cluster_key": "research_episode_id",
    }
    return selected


def build_episode_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sampled outcomes at the correlation-cluster/episode level."""
    columns = [
        "research_episode_id", "episode_start_time", "episode_end_time",
        "strategy_profile_key", "side", "sampled_entries", "viable_entries",
        "wins", "losses", "entry_win_rate", "episode_net_r", "episode_avg_r",
        "episode_positive",
    ]
    if samples.empty:
        return pd.DataFrame(columns=columns)

    frame = samples.copy()
    frame["pair_net_r"] = pd.to_numeric(frame.get("pair_net_r"), errors="coerce")
    frame["_win"] = frame["pair_net_r"].gt(0)
    entry_times = pd.to_datetime(frame.get("entry_time"), utc=True, errors="coerce")
    frame["_entry_time"] = entry_times
    rows = []
    for episode_id, group in frame.groupby("research_episode_id", sort=False):
        net_r = float(group["pair_net_r"].sum(min_count=1))
        finite = group["pair_net_r"].dropna()
        wins = int(group["_win"].sum())
        entries = int(len(group))
        rows.append({
            "research_episode_id": episode_id,
            "episode_start_time": group["_entry_time"].min(),
            "episode_end_time": group["_entry_time"].max(),
            "strategy_profile_key": group.get("strategy_profile_key", pd.Series([None])).iloc[0],
            "side": str(group["side"].iloc[0]).upper(),
            "sampled_entries": entries,
            "viable_entries": int(group["research_episode_viable_entries"].max()),
            "wins": wins,
            "losses": entries - wins,
            "entry_win_rate": (wins / entries) if entries else np.nan,
            "episode_net_r": net_r,
            "episode_avg_r": float(finite.mean()) if not finite.empty else np.nan,
            "episode_positive": bool(np.isfinite(net_r) and net_r > 0),
        })
    return pd.DataFrame(rows, columns=columns)


def build_sampling_summary(samples: pd.DataFrame, metadata: dict | None = None) -> dict:
    """Return entry-level and episode-level resilience statistics."""
    metadata = dict(metadata or samples.attrs.get("research_sampling", {}))
    episodes = build_episode_table(samples)
    r = pd.to_numeric(samples.get("pair_net_r", pd.Series(dtype=float)), errors="coerce")
    wins = int(r.gt(0).sum())
    entries = int(len(samples))
    episode_count = int(len(episodes))
    positive_episodes = int(episodes["episode_positive"].sum()) if episode_count else 0

    summary = {
        **metadata,
        "synthetic_entries": entries,
        "wins": wins,
        "losses": entries - wins,
        "entry_win_rate": (wins / entries) if entries else None,
        "average_r_per_entry": float(r.mean()) if r.notna().any() else None,
        "net_r": float(r.sum(min_count=1)) if r.notna().any() else None,
        "unique_episodes": episode_count,
        "positive_episodes": positive_episodes,
        "negative_or_flat_episodes": episode_count - positive_episodes,
        "episode_success_rate": (positive_episodes / episode_count) if episode_count else None,
        "average_sampled_entries_per_episode": float(episodes["sampled_entries"].mean()) if episode_count else None,
        "median_sampled_entries_per_episode": float(episodes["sampled_entries"].median()) if episode_count else None,
        "average_episode_r": float(episodes["episode_net_r"].mean()) if episode_count else None,
    }
    if episode_count:
        best = episodes.loc[episodes["episode_net_r"].idxmax()]
        worst = episodes.loc[episodes["episode_net_r"].idxmin()]
        summary["best_episode"] = {
            "research_episode_id": best["research_episode_id"],
            "net_r": float(best["episode_net_r"]),
        }
        summary["worst_episode"] = {
            "research_episode_id": worst["research_episode_id"],
            "net_r": float(worst["episode_net_r"]),
        }
    else:
        summary["best_episode"] = None
        summary["worst_episode"] = None

    for output_name, candidates in (
        ("average_mae_r", ("mae_r", "max_adverse_excursion_r")),
        ("average_mfe_r", ("mfe_r", "max_favorable_excursion_r")),
    ):
        source = next((name for name in candidates if name in samples.columns), None)
        if source is not None:
            values = pd.to_numeric(samples[source], errors="coerce")
            summary[output_name] = float(values.mean()) if values.notna().any() else None
    return summary


def build_context_breakdown(samples: pd.DataFrame) -> pd.DataFrame:
    """Compact entry/episode outcome table across available research context."""
    columns = ["dimension", "value", "entries", "episodes", "wins", "win_rate", "avg_r", "net_r"]
    if samples.empty:
        return pd.DataFrame(columns=columns)

    frame = samples.copy()
    frame["pair_net_r"] = pd.to_numeric(frame["pair_net_r"], errors="coerce")
    frame["_win"] = frame["pair_net_r"].gt(0)
    side = frame["side"].astype(str).str.upper()
    plus_di = pd.to_numeric(frame.get("plus_di"), errors="coerce")
    minus_di = pd.to_numeric(frame.get("minus_di"), errors="coerce")
    directional_di = plus_di.where(side.eq("LONG"), minus_di)
    frame["research_di_bucket"] = pd.cut(
        directional_di,
        [-np.inf, 10, 20, 30, 40, np.inf],
        labels=["<10", "10-20", "20-30", "30-40", "40+"],
        right=False,
    ).astype("string")
    adx = pd.to_numeric(frame.get("adx"), errors="coerce")
    frame["research_adx_bucket"] = pd.cut(
        adx,
        [-np.inf, 20, 30, 40, np.inf],
        labels=["<20", "20-30", "30-40", "40+"],
        right=False,
    ).astype("string")

    dimensions = [
        "strategy_profile_key", "market_regime", "research_di_bucket",
        "research_adx_bucket", "di_pressure_state", "mean_reversion_state",
        "funding_bias", "oi_vs_price_state_1h",
    ]
    rows = []
    for dimension in dimensions:
        if dimension not in frame.columns:
            continue
        available = frame.loc[frame[dimension].notna()].copy()
        if available.empty:
            continue
        for value, group in available.groupby(dimension, dropna=False, sort=True):
            entries = int(len(group))
            wins = int(group["_win"].sum())
            r = group["pair_net_r"]
            rows.append({
                "dimension": dimension,
                "value": str(value),
                "entries": entries,
                "episodes": int(group["research_episode_id"].nunique()),
                "wins": wins,
                "win_rate": wins / entries if entries else np.nan,
                "avg_r": float(r.mean()) if r.notna().any() else np.nan,
                "net_r": float(r.sum(min_count=1)) if r.notna().any() else np.nan,
            })
    return pd.DataFrame(rows, columns=columns)