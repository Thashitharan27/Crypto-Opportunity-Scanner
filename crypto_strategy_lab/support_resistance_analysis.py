"""Support/Resistance analysis report generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from crypto_strategy_lab.report_workbooks import build_support_resistance_workbook


def build_sr_analysis_tables(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate support/resistance analysis reports from trade data.

    Args:
        trades: Trade results DataFrame from engine.results_frame()
    Returns:
        Dictionary of workbook sheet name -> DataFrame
    """
    if trades.empty or "long_sr_location" not in trades.columns:
        return {}
    
    reports: dict[str, pd.DataFrame] = {}
    
    # Report 1: Support/Resistance Analysis by Location and Direction
    sr_analysis = _build_sr_analysis(trades)
    if not sr_analysis.empty:
        reports["Location"] = sr_analysis
    
    # Report 2: Support/Resistance Regime Analysis
    if "market_regime" in trades.columns:
        sr_regime = _build_sr_regime_analysis(trades)
        if not sr_regime.empty:
            reports["Regime"] = sr_regime
    
    # Report 3: Support/Resistance Distance Buckets
    sr_distance = _build_sr_distance_buckets(trades)
    if not sr_distance.empty:
        reports["Distance"] = sr_distance

    # Report: S/R event-context analysis (NEAR_SUPPORT/SUPPORT_BOUNCE/etc, entry-time snapshot, may be multi-label)
    sr_event_context = _build_sr_event_context_analysis(trades)
    if not sr_event_context.empty:
        reports["Event Context"] = sr_event_context

    for sheet_name, builder in (
        ("Hold Analysis", _build_sr_hold_analysis),
        ("Rejection Strength", _build_sr_rejection_analysis),
        ("Test Count", _build_sr_test_count_analysis),
    ):
        report = builder(trades)
        if not report.empty:
            reports[sheet_name] = report

    overview_parts = []
    location = reports.get("Location")
    if location is not None and not location.empty:
        context_column = "location" if "location" in location else location.columns[0]
        concise = location[location[context_column].astype(str).str.contains("SUPPORT|RESISTANCE|GOOD|NEUTRAL|BAD", case=False, regex=True)].copy()
        concise.insert(0, "source", "Location")
        overview_parts.append(concise.head(18))
    events = reports.get("Event Context")
    if events is not None and not events.empty and "context" in events:
        key_events = {"SUPPORT_BOUNCE", "RESISTANCE_REJECTION", "RESISTANCE_BREAKOUT", "SUPPORT_BREAKDOWN"}
        concise = events[events["context"].isin(key_events)].copy()
        concise.insert(0, "source", "Event Context")
        overview_parts.append(concise)
    reports["Overview"] = pd.concat(overview_parts, ignore_index=True, sort=False) if overview_parts else pd.DataFrame()
    
    return reports


def generate_sr_analysis_reports(trades: pd.DataFrame, run_dir: Path) -> dict[str, pd.DataFrame]:
    """Build S/R tables once and consolidate them into one workbook."""
    reports = build_sr_analysis_tables(trades)
    build_support_resistance_workbook(reports, run_dir)
    return reports


def _interaction_rows(trades: pd.DataFrame, metric: str) -> list[dict]:
    rows = []
    for direction, prefix, side_values in (("LONG", "long_", {"LONG", "BOTH"}), ("SHORT", "short_", {"SHORT", "BOTH"})):
        if "side" not in trades.columns:
            continue
        selected = trades[trades["side"].isin(side_values)]
        for structure, state_col, value_col in (("Support", f"{prefix}sr_support_state", f"{prefix}sr_support_{metric}"), ("Resistance", f"{prefix}sr_resistance_state", f"{prefix}sr_resistance_{metric}")):
            if state_col not in selected.columns:
                continue
            group_columns = [state_col] + (["market_regime"] if "market_regime" in selected.columns else [])
            for keys, group in selected.groupby(group_columns, dropna=False, observed=True):
                keys = (keys,) if not isinstance(keys, tuple) else keys
                state = keys[0]
                if pd.isna(state):
                    continue
                row = _sr_stats_row(str(state), direction, group, prefix)
                row["structure_type"] = structure
                row["state"] = state
                if "market_regime" in selected.columns:
                    row["market_regime"] = keys[1]
                if value_col in group.columns:
                    row[f"average_{metric}"] = group[value_col].mean()
                rows.append(row)
    return rows


def _build_sr_hold_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    rows = _interaction_rows(trades, "held")
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    columns = ["direction", "market_regime", "structure_type", "state", "trade_count", "winners", "losers", "win_rate", "total_r", "avg_r", "total_pnl", "avg_pnl"]
    return result[[column for column in columns if column in result.columns]]


def _build_sr_rejection_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = [(0.0, 0.25, "0-0.25 ATR"), (0.25, 0.50, "0.25-0.50 ATR"), (0.50, 0.75, "0.50-0.75 ATR"), (0.75, 1.00, "0.75-1.00 ATR"), (1.00, 1.50, "1.00-1.50 ATR"), (1.50, np.inf, "1.50+ ATR")]
    for direction, prefix, side_values in (("LONG", "long_", {"LONG", "BOTH"}), ("SHORT", "short_", {"SHORT", "BOTH"})):
        selected = trades[trades["side"].isin(side_values)] if "side" in trades.columns else trades
        for structure in ("Support", "Resistance"):
            value_col = f"{prefix}sr_{structure.lower()}_rejection_atr"
            if value_col not in selected.columns:
                continue
            for lower, upper, label in buckets:
                mask = selected[value_col].notna() & (selected[value_col] >= lower) & (selected[value_col] < upper)
                bucket_trades = selected[mask]
                if bucket_trades.empty:
                    continue
                grouped = bucket_trades.groupby("market_regime", dropna=False, observed=True) if "market_regime" in bucket_trades.columns else [(None, bucket_trades)]
                for regime, group in grouped:
                    row = _sr_stats_row(label, direction, group, prefix)
                    row.update({"market_regime": regime, "structure_type": structure, "rejection_bucket": label})
                    rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_sr_test_count_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, prefix, side_values in (("LONG", "long_", {"LONG", "BOTH"}), ("SHORT", "short_", {"SHORT", "BOTH"})):
        selected = trades[trades["side"].isin(side_values)] if "side" in trades.columns else trades
        for structure in ("Support", "Resistance"):
            count_col = f"{prefix}sr_{structure.lower()}_test_count"
            state_col = f"{prefix}sr_{structure.lower()}_state"
            if count_col not in selected.columns:
                continue
            bucket = selected[count_col].fillna(0).astype(int).map(lambda count: "1st test" if count == 1 else ("2nd test" if count == 2 else ("3rd test" if count == 3 else ("4+ tests" if count >= 4 else "No test"))))
            group_columns = ["_sr_test_bucket"] + ([state_col] if state_col in selected.columns else []) + (["market_regime"] if "market_regime" in selected.columns else [])
            grouped = selected.assign(_sr_test_bucket=bucket).groupby(group_columns, dropna=False, observed=True)
            for keys, group in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys, None)
                row = _sr_stats_row(str(keys[0]), direction, group, prefix)
                state_index = 1 if state_col in selected.columns else None
                regime_index = (state_index + 1) if state_index is not None and "market_regime" in selected.columns else (1 if "market_regime" in selected.columns else None)
                row.update({"market_regime": keys[regime_index] if regime_index is not None else None, "structure_type": structure, "test_bucket": keys[0], "state": keys[state_index] if state_index is not None else None})
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_sr_event_context_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Public entry point for GUI/report consumers: Event Context workbook table."""
    return _build_sr_event_context_analysis(trades)


def _build_sr_event_context_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """S/R performance grouped by entry-time event context labels (NEAR_SUPPORT, SUPPORT_BOUNCE, RESISTANCE_REJECTION,
    RESISTANCE_BREAKOUT, SUPPORT_BREAKDOWN, NO_NEARBY_SR). A trade may carry multiple labels."""
    rows = []
    for direction, prefix, side_values in (("LONG", "long_", {"LONG", "BOTH"}), ("SHORT", "short_", {"SHORT", "BOTH"})):
        context_col = f"{prefix}sr_context"
        if context_col not in trades.columns or "side" not in trades.columns:
            continue
        selected = trades[trades["side"].isin(side_values)].copy()
        selected = selected[selected[context_col].notna()]
        if selected.empty:
            continue
        selected["_sr_context_label"] = selected[context_col].str.split("|")
        exploded = selected.explode("_sr_context_label")
        for label, group in exploded.groupby("_sr_context_label", observed=True):
            rows.append(_sr_stats_row(label, direction, group, prefix))
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).rename(columns={"location": "context"})
    cols = ["context", "direction"] + [c for c in result.columns if c not in ("context", "direction")]
    return result[cols]


def _build_sr_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Build support/resistance analysis grouped by location and direction."""
    rows = []
    
    # Identify SR location columns for LONG and SHORT
    long_sr_cols = [c for c in trades.columns if c.startswith("long_sr_")]
    short_sr_cols = [c for c in trades.columns if c.startswith("short_sr_")]
    
    if not long_sr_cols or not short_sr_cols:
        return pd.DataFrame()
    
    # Process LONG trades
    if "long_sr_location" in trades.columns:
        long_trades = trades[trades["side"].isin(["LONG", "BOTH"])].copy()
        if not long_trades.empty:
            location_groups = long_trades.groupby("long_sr_location", observed=True)
            for location, group in location_groups:
                if pd.isna(location):
                    continue
                rows.append(_sr_stats_row(location, "LONG", group, "long_"))
    
    # Process SHORT trades
    if "short_sr_location" in trades.columns:
        short_trades = trades[trades["side"].isin(["SHORT", "BOTH"])].copy()
        if not short_trades.empty:
            location_groups = short_trades.groupby("short_sr_location", observed=True)
            for location, group in location_groups:
                if pd.isna(location):
                    continue
                rows.append(_sr_stats_row(location, "SHORT", group, "short_"))
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_sr_regime_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Build support/resistance analysis by regime, location, and direction."""
    rows = []
    
    if "market_regime" not in trades.columns:
        return pd.DataFrame()
    
    # Process LONG trades by regime and location
    if "long_sr_location" in trades.columns:
        long_trades = trades[trades["side"].isin(["LONG", "BOTH"])].copy()
        if not long_trades.empty:
            for regime in long_trades["market_regime"].dropna().unique():
                regime_trades = long_trades[long_trades["market_regime"] == regime]
                location_groups = regime_trades.groupby("long_sr_location", observed=True)
                for location, group in location_groups:
                    if pd.isna(location):
                        continue
                    row = _sr_stats_row(location, "LONG", group, "long_")
                    row["regime"] = regime
                    rows.append(row)
    
    # Process SHORT trades by regime and location
    if "short_sr_location" in trades.columns:
        short_trades = trades[trades["side"].isin(["SHORT", "BOTH"])].copy()
        if not short_trades.empty:
            for regime in short_trades["market_regime"].dropna().unique():
                regime_trades = short_trades[short_trades["market_regime"] == regime]
                location_groups = regime_trades.groupby("short_sr_location", observed=True)
                for location, group in location_groups:
                    if pd.isna(location):
                        continue
                    row = _sr_stats_row(location, "SHORT", group, "short_")
                    row["regime"] = regime
                    rows.append(row)
    
    if rows:
        result = pd.DataFrame(rows)
        # Reorder columns
        cols = ["regime", "location", "direction"] + [c for c in result.columns if c not in ["regime", "location", "direction"]]
        return result[cols]
    
    return pd.DataFrame()


def _build_sr_distance_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    """Build support/resistance analysis by ATR distance buckets."""
    rows = []
    
    # Define distance buckets (in ATR units)
    buckets = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, np.inf)]
    bucket_labels = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50-2.00", "2.00+"]
    
    # Process LONG trades - support distances
    if "long_sr_support_distance_atr" in trades.columns:
        long_trades = trades[trades["side"].isin(["LONG", "BOTH"])].copy()
        if not long_trades.empty:
            for (lower, upper), bucket_label in zip(buckets, bucket_labels):
                mask = (long_trades["long_sr_support_distance_atr"] >= lower) & \
                       (long_trades["long_sr_support_distance_atr"] < upper) & \
                       (long_trades["long_sr_support_distance_atr"].notna())
                if mask.any():
                    group = long_trades[mask]
                    row = _sr_stats_row(f"Support {bucket_label} ATR", "LONG", group, "long_")
                    rows.append(row)
    
    # Process LONG trades - resistance distances
    if "long_sr_resistance_distance_atr" in trades.columns:
        long_trades = trades[trades["side"].isin(["LONG", "BOTH"])].copy()
        if not long_trades.empty:
            for (lower, upper), bucket_label in zip(buckets, bucket_labels):
                mask = (long_trades["long_sr_resistance_distance_atr"] >= lower) & \
                       (long_trades["long_sr_resistance_distance_atr"] < upper) & \
                       (long_trades["long_sr_resistance_distance_atr"].notna())
                if mask.any():
                    group = long_trades[mask]
                    row = _sr_stats_row(f"Resistance {bucket_label} ATR", "LONG", group, "long_")
                    rows.append(row)
    
    # Process SHORT trades - support distances
    if "short_sr_support_distance_atr" in trades.columns:
        short_trades = trades[trades["side"].isin(["SHORT", "BOTH"])].copy()
        if not short_trades.empty:
            for (lower, upper), bucket_label in zip(buckets, bucket_labels):
                mask = (short_trades["short_sr_support_distance_atr"] >= lower) & \
                       (short_trades["short_sr_support_distance_atr"] < upper) & \
                       (short_trades["short_sr_support_distance_atr"].notna())
                if mask.any():
                    group = short_trades[mask]
                    row = _sr_stats_row(f"Support {bucket_label} ATR", "SHORT", group, "short_")
                    rows.append(row)
    
    # Process SHORT trades - resistance distances
    if "short_sr_resistance_distance_atr" in trades.columns:
        short_trades = trades[trades["side"].isin(["SHORT", "BOTH"])].copy()
        if not short_trades.empty:
            for (lower, upper), bucket_label in zip(buckets, bucket_labels):
                mask = (short_trades["short_sr_resistance_distance_atr"] >= lower) & \
                       (short_trades["short_sr_resistance_distance_atr"] < upper) & \
                       (short_trades["short_sr_resistance_distance_atr"].notna())
                if mask.any():
                    group = short_trades[mask]
                    row = _sr_stats_row(f"Resistance {bucket_label} ATR", "SHORT", group, "short_")
                    rows.append(row)
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _sr_stats_row(location: str, direction: str, group: pd.DataFrame, prefix: str) -> dict:
    """Calculate statistics for a group of trades."""
    
    # Identify which columns contain PnL data (handles both LONG and SHORT prefixes)
    if f"{prefix}pair_net_r" in group.columns:
        r_col = f"{prefix}pair_net_r"
    elif "pair_net_r" in group.columns:
        r_col = "pair_net_r"
    else:
        r_col = None
    
    if f"{prefix}pair_net_pnl" in group.columns:
        pnl_col = f"{prefix}pair_net_pnl"
    elif "pair_net_pnl" in group.columns:
        pnl_col = "pair_net_pnl"
    else:
        pnl_col = None
    
    if f"{prefix}holding_minutes" in group.columns:
        hold_col = f"{prefix}holding_minutes"
    elif "holding_minutes" in group.columns:
        hold_col = "holding_minutes"
    else:
        hold_col = None
    
    # Calculate win/loss metrics
    trade_count = len(group)
    if pnl_col and pnl_col in group.columns:
        winners = (group[pnl_col] > 0).sum()
        losers = (group[pnl_col] < 0).sum()
        breakeven = trade_count - winners - losers
        win_rate = winners / trade_count if trade_count > 0 else np.nan
        avg_pnl = group[pnl_col].mean()
        total_pnl = group[pnl_col].sum()
    else:
        winners = losers = breakeven = 0
        win_rate = avg_pnl = total_pnl = np.nan
    
    # Calculate R metrics
    if r_col and r_col in group.columns:
        avg_r = group[r_col].mean()
        total_r = group[r_col].sum()
    else:
        avg_r = total_r = np.nan
    
    # Calculate holding time
    if hold_col and hold_col in group.columns:
        avg_holding_minutes = group[hold_col].mean()
    else:
        avg_holding_minutes = np.nan
    
    return {
        "location": location,
        "direction": direction,
        "trade_count": trade_count,
        "winners": winners,
        "losers": losers,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "total_r": total_r,
        "avg_r": avg_r,
        "avg_holding_minutes": avg_holding_minutes,
    }
