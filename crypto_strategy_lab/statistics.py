"""Summary statistics for completed directional trades."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _max_streak(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    groups = mask.ne(mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max() or 0)


def _daily_schedule_summary(trades: pd.DataFrame, stats: dict | None, skipped: list | None) -> dict[str, object]:
    stats = stats or {}; skipped = skipped or []
    skipped_reasons = pd.Series([r.get("reason") for r in skipped], dtype=object)
    delays = pd.to_numeric(trades.get("entry_delay_minutes", pd.Series(dtype=float)), errors="coerce") if not trades.empty else pd.Series(dtype=float)
    entry_days = pd.to_datetime(trades.get("actual_entry_timestamp", pd.Series(dtype=object)), errors="coerce", utc=True).dt.date if not trades.empty else pd.Series(dtype=object)
    skipped_days = pd.to_datetime(pd.Series([r.get("scheduled_timestamp") for r in skipped]), errors="coerce", utc=True).dt.date if skipped else pd.Series(dtype=object)
    return {
        "scheduled_entry_opportunities": int(stats.get("scheduled_entry_opportunities", 0)),
        "trades_opened_on_schedule": int(stats.get("trades_opened_on_schedule", 0)),
        "scheduled_entries_skipped_because_trade_was_open": int((skipped_reasons == "ACTIVE_TRADE").sum()),
        "scheduled_entries_skipped_by_filters": int((skipped_reasons == "FILTER_REJECTED").sum()),
        "scheduled_entries_skipped_due_to_missing_data": int((skipped_reasons == "MISSING_DATA").sum()),
        "average_entry_delay": float(delays.mean()) if len(delays) else 0.0,
        "maximum_entry_delay": float(delays.max()) if len(delays) else 0.0,
        "days_with_trades": int(entry_days.dropna().nunique()),
        "days_without_trades": int(skipped_days.dropna().nunique()),
    }


def _bool_column(trades: pd.DataFrame, name: str) -> pd.Series:
    return trades.get(name, pd.Series(False, index=trades.index)).fillna(False).astype(bool)


def _single_trade_exit_reason(trades: pd.DataFrame) -> pd.Series:
    """Return the exit reason of the one directional position on each result row."""
    result = trades.get("exit_reason", pd.Series(index=trades.index, dtype=object)).copy()
    for side in ("long", "short"):
        values = trades.get(f"{side}_exit_reason", pd.Series(index=trades.index, dtype=object))
        result = result.where(result.notna(), values)
    return result


def _single_trade_be_triggered(trades: pd.DataFrame) -> pd.Series:
    return _bool_column(trades, "long_be_triggered") | _bool_column(trades, "short_be_triggered")


def summarize(trades: pd.DataFrame, initial_equity: float = 1000.0) -> dict[str, object]:
    if trades.empty:
        stats = trades.attrs.get("daily_schedule_stats", {})
        skipped = trades.attrs.get("skipped_daily_entries", [])
        return {
            "total_pairs": 0,
            "total_trades": 0,
            "ending_equity": initial_equity,
            "exit_source_counts": {"1M_INTRABAR": 0, "15M_FALLBACK": 0, "END_OF_DATA": 0},
            **_daily_schedule_summary(pd.DataFrame(), stats, skipped),
        }

    wins = trades["pair_net_pnl"] > 0
    losses = trades["pair_net_pnl"] < 0
    flats = trades["pair_net_pnl"] == 0
    gross_profit = trades.loc[wins, "pair_net_pnl"].sum()
    gross_loss = -trades.loc[losses, "pair_net_pnl"].sum()
    equity = trades["equity_after_trade"]
    running_peak = equity.cummax().clip(lower=initial_equity)
    drawdown = equity - running_peak
    max_dd = float(drawdown.min())
    max_dd_pct = float((drawdown / running_peak).min()) if not running_peak.empty else 0.0

    exit_source_cols = [c for c in ("long_exit_source", "short_exit_source") if c in trades]
    source_values = pd.concat([trades[c] for c in exit_source_cols], ignore_index=True) if exit_source_cols else trades.get("exit_source", pd.Series(dtype=object))
    source_counts = {
        "1M_INTRABAR": int((source_values == "1M_INTRABAR").sum()),
        "15M_FALLBACK": int((source_values == "15M_FALLBACK").sum()),
        "END_OF_DATA": int((source_values == "END_OF_DATA").sum()),
    }

    timeout_mask = _bool_column(trades, "profile_timeout_triggered")
    timeout_trades = trades[timeout_mask]

    exit_reason = _single_trade_exit_reason(trades)
    be_triggered = _single_trade_be_triggered(trades)
    be_trades = trades[be_triggered]
    be_exit_mask = exit_reason.isin(["BE", "BE_COST_ADJUSTED", "BE_R_OFFSET"])
    tp_after_be = be_triggered & exit_reason.eq("TP")

    fallback_cols = [c for c in ("long_fallback_reason", "short_fallback_reason") if c in trades]
    fallback_reasons = (
        pd.concat([trades[c] for c in fallback_cols], ignore_index=True)
        if fallback_cols
        else pd.Series(dtype=object)
    ).dropna().value_counts().to_dict()

    be_same_candle_ambiguity_count = 0
    for side in ("long", "short"):
        ambiguous = _bool_column(trades, f"{side}_be_same_candle_ambiguous")
        enabled = _bool_column(trades, f"{side}_be_enabled")
        be_same_candle_ambiguity_count += int((ambiguous & enabled).sum())

    outcome_labels = exit_reason.fillna("UNKNOWN").astype(str)
    outcome_labels = outcome_labels.where(~tp_after_be, "TP_AFTER_BE_MOVE")
    outcomes = {}
    for key, group in trades.groupby(outcome_labels, dropna=False):
        outcomes[str(key)] = {
            "count": int(len(group)),
            "percentage": float(len(group) / len(trades)),
            "average_net_r": float(group["pair_net_r"].mean()),
            "total_net_r": float(group["pair_net_r"].sum()),
        }

    adx = pd.to_numeric(trades.get("adx", pd.Series(dtype=float)), errors="coerce")
    plus_di = pd.to_numeric(trades.get("plus_di", pd.Series(dtype=float)), errors="coerce")
    minus_di = pd.to_numeric(trades.get("minus_di", pd.Series(dtype=float)), errors="coerce")
    daily_stats = _daily_schedule_summary(
        trades,
        trades.attrs.get("daily_schedule_stats", {}),
        trades.attrs.get("skipped_daily_entries", []),
    )

    return {
        **daily_stats,
        # ``total_pairs`` and pair_* PnL/R columns remain compatibility names in
        # the current result schema. There is exactly one directional trade per row.
        "total_pairs": int(len(trades)),
        "total_trades": int(len(trades)),
        "average_winner": float(trades.loc[wins, "pair_net_pnl"].mean()) if wins.any() else 0.0,
        "average_loser": float(trades.loc[losses, "pair_net_pnl"].mean()) if losses.any() else 0.0,
        "expectancy": float(trades["pair_net_pnl"].mean()),
        "signals_evaluated": int(trades.get("signals_evaluated", pd.Series([len(trades)])).iloc[0]) if "signals_evaluated" in trades else int(len(trades)),
        "signals_skipped_by_adx": int(trades.get("signals_skipped_by_adx", pd.Series([0])).iloc[0]) if "signals_skipped_by_adx" in trades else 0,
        "signals_skipped_by_filters": int(trades.get("signals_skipped_by_filters", pd.Series([0])).iloc[0]) if "signals_skipped_by_filters" in trades else 0,
        "signals_traded": int(trades.get("signals_traded", pd.Series([len(trades)])).iloc[0]) if "signals_traded" in trades else int(len(trades)),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "flat_pairs": int(flats.sum()),
        "flat_trades": int(flats.sum()),
        "win_rate": float(wins.mean()),
        "loss_rate": float(losses.mean()),
        "average_gross_account_r": float(trades.get("pair_gross_account_r", trades["pair_gross_r"]).mean()),
        "average_net_r": float(trades["pair_net_r"].mean()),
        "average_net_account_r": float(trades.get("pair_net_account_r", trades["pair_net_r"]).mean()),
        "median_net_r": float(trades["pair_net_r"].median()),
        "total_net_r": float(trades["pair_net_r"].sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
        "ending_equity": float(equity.iloc[-1]),
        "total_return_percentage": float((equity.iloc[-1] / initial_equity - 1) * 100),
        "maximum_drawdown": max_dd,
        "maximum_drawdown_percentage": max_dd_pct * 100,
        "maximum_consecutive_wins": _max_streak(wins),
        "maximum_consecutive_losses": _max_streak(losses),
        "average_holding_time": float(trades["holding_hours"].mean()),
        "total_fees": float(trades["pair_total_fees"].sum()),
        "profile_timeout_trade_count": int(timeout_mask.sum()),
        "average_profile_timeout_pnl": float(timeout_trades["pair_net_pnl"].mean()) if not timeout_trades.empty else 0.0,
        "total_profile_timeout_pnl": float(timeout_trades["pair_net_pnl"].sum()) if not timeout_trades.empty else 0.0,
        "profile_timeout_trades_profitable": int((timeout_trades["pair_net_pnl"] > 0).sum()) if not timeout_trades.empty else 0,
        "profile_timeout_trades_losing": int((timeout_trades["pair_net_pnl"] < 0).sum()) if not timeout_trades.empty else 0,
        "be_triggered_trade_count": int(be_triggered.sum()),
        "be_exit_count": int(be_exit_mask.sum()),
        "tp_after_be_count": int(tp_after_be.sum()),
        "average_pnl_of_be_triggered_trades": float(be_trades["pair_net_pnl"].mean()) if not be_trades.empty else 0.0,
        "total_pnl_of_be_triggered_trades": float(be_trades["pair_net_pnl"].sum()) if not be_trades.empty else 0.0,
        "be_same_candle_ambiguity_count": be_same_candle_ambiguity_count,
        "ambiguous_event_count": int(trades["ambiguous_candle"].sum()),
        "ambiguous_intrabar_count": int(trades.get("ambiguous_intrabar", trades["ambiguous_candle"]).sum()),
        "missing_intrabar_interval_count": int(trades.get("missing_intrabar_data", pd.Series(dtype=bool)).sum()),
        "exit_source_counts": source_counts,
        "intrabar_exit_count": source_counts["1M_INTRABAR"],
        "fallback_15m_exit_count": source_counts["15M_FALLBACK"],
        "end_of_data_exit_count": source_counts["END_OF_DATA"],
        "fallback_reason_counts": {str(k): int(v) for k, v in fallback_reasons.items()},
        "leverage_capped_trade_count": int(trades.get("leverage_capped", pd.Series(dtype=bool)).sum()),
        "average_combined_effective_leverage": float(trades.get("combined_effective_leverage", pd.Series(dtype=float)).mean()),
        "maximum_combined_effective_leverage": float(trades.get("combined_effective_leverage", pd.Series(dtype=float)).max()),
        "average_fees_as_percentage_of_expected_winning_profit": float(trades.get("fees_as_percentage_of_expected_winning_profit", pd.Series(dtype=float)).mean()),
        "average_adx_of_winning_trades": float(adx[wins].mean()) if not adx[wins].empty else np.nan,
        "average_adx_of_losing_trades": float(adx[losses].mean()) if not adx[losses].empty else np.nan,
        "average_plus_di_of_winners": float(plus_di[wins].mean()) if not plus_di[wins].empty else np.nan,
        "average_plus_di_of_losers": float(plus_di[losses].mean()) if not plus_di[losses].empty else np.nan,
        "average_minus_di_of_winners": float(minus_di[wins].mean()) if not minus_di[wins].empty else np.nan,
        "average_minus_di_of_losers": float(minus_di[losses].mean()) if not minus_di[losses].empty else np.nan,
        "exit_combinations": outcomes,
        "outcomes": outcomes,
    }


def _bucket_exit_counts(trades: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    reasons = _single_trade_exit_reason(trades).loc[mask]
    be_reasons = {"BE", "BE_COST_ADJUSTED", "BE_R_OFFSET"}
    timeout = _bool_column(trades, "profile_timeout_triggered").loc[mask]
    return {
        "TP Count": int(reasons.eq("TP").sum()),
        "SL Count": int(reasons.eq("SL").sum()),
        "BE Count": int(reasons.isin(be_reasons).sum()),
        "Profile Timeout Count": int(timeout.sum()),
        "End Of Data Count": int(reasons.eq("END_OF_DATA").sum()),
    }


def adx_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    buckets = [(0,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,None)]
    rows=[]
    adx_values = pd.to_numeric(trades.get("adx", pd.Series(dtype=float)), errors="coerce")
    for lo, hi in buckets:
        label = f"{lo}+" if hi is None else f"{lo}-{hi}"
        mask = adx_values >= lo if hi is None else ((adx_values >= lo) & (adx_values < hi))
        pnl = pd.to_numeric(trades.get("pair_net_pnl", pd.Series(index=trades.index, dtype=float)), errors="coerce").loc[mask]
        duration = pd.to_numeric(trades.get("holding_minutes", pd.Series(index=trades.index, dtype=float)), errors="coerce").loc[mask]
        wins = pnl > 0; losses = pnl < 0
        rows.append({
            "Bucket": label,
            "Trades": int(mask.sum()),
            "Wins": int(wins.sum()),
            "Losses": int(losses.sum()),
            "Win rate": float(wins.mean()) if len(pnl) else 0.0,
            "Average PnL": float(pnl.mean()) if len(pnl) else 0.0,
            "Average duration": float(duration.mean()) if len(duration) else 0.0,
            **_bucket_exit_counts(trades, mask),
        })
    return pd.DataFrame(rows)


def equity_curve(trades: pd.DataFrame, initial_equity: float = 1000.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["time", "equity", "drawdown"])
    equity = trades["equity_after_trade"]
    exit_cols=[c for c in ("long_exit_time","short_exit_time") if c in trades]
    if exit_cols:
        parsed = [pd.to_datetime(trades[c], errors="coerce", utc=True) for c in exit_cols]
        times = parsed[0]
        for candidate in parsed[1:]:
            times = times.where(candidate.isna() | (times.notna() & times.ge(candidate)), candidate)
    else:
        times = pd.to_datetime(trades.get("exit_time", trades["entry_time"]),errors="coerce",utc=True)
    return pd.DataFrame({
        "time": pd.to_datetime(times),
        "equity": equity,
        "drawdown": equity - equity.cummax().clip(lower=initial_equity),
    })


def bucket_analysis(trades: pd.DataFrame, column: str, buckets: list[tuple[float, float | None]], pct_labels: bool = False) -> pd.DataFrame:
    rows=[]; values=pd.to_numeric(trades.get(column, pd.Series(index=trades.index, dtype=float)), errors="coerce")
    all_pnl=pd.to_numeric(trades.get("pair_net_pnl",pd.Series(index=trades.index,dtype=float)),errors="coerce")
    all_holding=pd.to_numeric(trades.get("holding_minutes",pd.Series(index=trades.index,dtype=float)),errors="coerce")
    for lo, hi in buckets:
        label = f"{lo:g}+" if hi is None else f"{lo:g}-{hi:g}"
        if pct_labels:
            label = f"{lo:g}%+" if hi is None else f"{lo:g}-{hi:g}%"
        mask = values >= lo if hi is None else ((values >= lo) & (values < hi))
        pnl=all_pnl.loc[mask]; holding=all_holding.loc[mask]
        wins=pnl>0; losses=pnl<0
        rows.append({
            "Bucket":label,
            "Trades":int(mask.sum()),
            "Wins":int(wins.sum()),
            "Losses":int(losses.sum()),
            "Win Rate":float(wins.mean()) if len(pnl) else 0.0,
            "Average Net PnL":float(pnl.mean()) if len(pnl) else 0.0,
            "Average Holding Time":float(holding.mean()) if len(holding) else 0.0,
            **_bucket_exit_counts(trades, mask),
        })
    return pd.DataFrame(rows)


def bb_width_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    return bucket_analysis(trades, "bb_width_pct", [(0,2),(2,4),(4,6),(6,8),(8,10),(10,None)], pct_labels=True)


def di_spread_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    return bucket_analysis(
        trades,
        "di_spread",
        [
            (0, 5),
            (5, 10),
            (10, 15),
            (15, 20),
            (20, 25),
            (25, 30),
            (30, 35),
            (35, 40),
            (40, 45),
            (45, 50),
            (50, None),
        ],
    )


def di_pressure_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Build readable pressure, regime, and spread-change performance tables."""
    columns = ["Section", "Direction", "Market Regime", "DI Pressure State", "DI Spread Change Bucket", "Trades", "Wins", "Losses", "Win Rate", "Average Net PnL", "Net PnL", "Average Net R", "Total Net R"]
    if trades.empty or "di_pressure_state" not in trades:
        return pd.DataFrame(columns=columns)
    frame=trades.copy()
    frame["Direction"]=frame.get("side", frame.get("di_sizing_direction", "UNKNOWN")).astype(str).str.upper()
    frame["DI Pressure State"]=frame["di_pressure_state"].fillna("UNKNOWN").astype(str).str.upper()
    frame["Market Regime"]=frame.get("market_regime", pd.Series("UNKNOWN",index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    pnl=pd.to_numeric(frame.get("pair_net_pnl"),errors="coerce"); net_r=pd.to_numeric(frame.get("pair_net_r"),errors="coerce")
    frame["_pnl"]=pnl; frame["_r"]=net_r
    change=pd.to_numeric(frame.get("di_spread_change"),errors="coerce")
    frame["DI Spread Change Bucket"]=pd.cut(change,[-np.inf,-10,-5,0,5,10,np.inf],right=False,labels=["< -10","-10 to -5","-5 to 0","0 to +5","+5 to +10","> +10"])
    def grouped(section, groups):
        rows=[]
        for keys,g in frame.groupby(groups,dropna=False,observed=True):
            keys=keys if isinstance(keys,tuple) else (keys,); gp=g["_pnl"]; gr=g["_r"]
            row={"Section":section,**dict(zip(groups,keys)),"Trades":len(g),"Wins":int((gp>0).sum()),"Losses":int((gp<0).sum()),"Win Rate":float((gp>0).mean()),"Average Net PnL":float(gp.mean()),"Net PnL":float(gp.sum()),"Average Net R":float(gr.mean()),"Total Net R":float(gr.sum())}; rows.append(row)
        return rows
    rows=grouped("Direction + Pressure",["Direction","DI Pressure State"])
    rows+=grouped("Regime + Direction + Pressure",["Market Regime","Direction","DI Pressure State"])
    rows+=grouped("DI Spread Change",["DI Spread Change Bucket"])
    return pd.DataFrame(rows).reindex(columns=columns)


def mean_reversion_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab mean-reversion behavior across the complete DI-pressure range."""
    columns=["Section","DI Pressure Bucket","Direction","Market Regime","DI Pressure State","Mean Reversion Alignment","Mean Reversion Motion","Mean Reversion State","Mean Reversion Strength","Trades","Wins","Losses","Win Rate","Average Net PnL","Net PnL","Average Net R","Total Net R"]
    required={"di_spread","mean_reversion_alignment","mean_reversion_motion"}
    if trades.empty or not required.issubset(trades.columns): return pd.DataFrame(columns=columns)
    frame=trades.copy(); di=pd.to_numeric(frame["di_spread"],errors="coerce")
    edges=[0,5,10,15,20,25,30,35,40,45,50,np.inf]; labels=["0-5","5-10","10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50","50+"]
    frame["DI Pressure Bucket"]=pd.cut(di,edges,right=False,labels=labels)
    frame["Direction"]=frame.get("di_sizing_direction",frame.get("sizing_direction",pd.Series("UNKNOWN",index=frame.index))).fillna("UNKNOWN").astype(str).str.upper()
    frame["Market Regime"]=frame.get("market_regime",pd.Series("UNKNOWN",index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["DI Pressure State"]=frame.get("di_pressure_state",pd.Series("UNKNOWN",index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion Alignment"]=frame["mean_reversion_alignment"].fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion Motion"]=frame["mean_reversion_motion"].fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion State"]=frame.get("mean_reversion_state",pd.Series("UNKNOWN",index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion Strength"]=frame.get("mean_reversion_strength_label",pd.Series("UNKNOWN",index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["_pnl"]=pd.to_numeric(frame.get("pair_net_pnl"),errors="coerce"); frame["_r"]=pd.to_numeric(frame.get("pair_net_r"),errors="coerce")
    def grouped(section, groups):
        rows=[]
        for keys,g in frame.groupby(groups,dropna=False,observed=True):
            keys=keys if isinstance(keys,tuple) else (keys,); pnl=g["_pnl"]; rr=g["_r"]
            rows.append({"Section":section,**dict(zip(groups,keys)),"Trades":int(len(g)),"Wins":int((pnl>0).sum()),"Losses":int((pnl<0).sum()),"Win Rate":float((pnl>0).mean()),"Average Net PnL":float(pnl.mean()),"Net PnL":float(pnl.sum()),"Average Net R":float(rr.mean()),"Total Net R":float(rr.sum())})
        return rows
    rows=grouped("DI Bucket + Reversion",["DI Pressure Bucket","Mean Reversion Alignment","Mean Reversion Motion"])
    rows+=grouped("Direction + DI Bucket + Reversion",["Direction","DI Pressure Bucket","Mean Reversion Alignment","Mean Reversion Motion"])
    rows+=grouped("DI State + DI Bucket + Reversion",["DI Pressure State","DI Pressure Bucket","Mean Reversion Alignment","Mean Reversion Motion"])
    rows+=grouped("Regime + DI Bucket + Reversion",["Market Regime","DI Pressure Bucket","Mean Reversion Alignment","Mean Reversion Motion"])
    rows+=grouped("Mean Reversion State",["Mean Reversion State","Mean Reversion Strength"])
    return pd.DataFrame(rows).reindex(columns=columns)
