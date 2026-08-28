"""Additive trade-lifecycle indicator reports built from existing telemetry."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from collections.abc import Callable

import numpy as np
import pandas as pd


INDICATORS = {"adx": "adx", "di_spread": "di_spread", "atr": "atr", "bb_width": "bb_width"}
PHASE_LABELS = ("0-25%", "25-50%", "50-75%", "75-100%")
VALIDATIONS = ("exit_before_entry", "missing_trade_id",
               "missing_telemetry", "duplicate_telemetry_timestamp", "non_monotonic_telemetry_time",
               "missing_indicator_values", "invalid_r_value")


def _num(row, *names):
    for name in names:
        value = row.get(name, np.nan)
        if pd.notna(value):
            return float(value)
    return np.nan


def _leg_rows(trades: pd.DataFrame):
    """Yield individual completed legs without guessing identity from timestamps."""
    for _, row in trades.iterrows():
        pair_id = row.get("pair_id")
        explicit = str(row.get("side", "")).upper()
        sides = [explicit.lower()] if explicit in ("LONG", "SHORT") else [s for s in ("long", "short") if pd.notna(row.get(f"{s}_exit_time"))]
        for side in sides:
            exit_time = row.get(f"{side}_exit_time", row.get("exit_time"))
            if pd.isna(exit_time):
                continue
            existing_leg_id = row.get("leg_id") if len(sides) == 1 else row.get(f"{side}_leg_id")
            leg_id = existing_leg_id if pd.notna(existing_leg_id) else (f"{pair_id}_{side.upper()}" if pd.notna(pair_id) else np.nan)
            trade_id = row.get("trade_id") if len(sides) == 1 and pd.notna(row.get("trade_id")) else leg_id
            entry_time = row.get(f"{side}_entry_time", row.get("entry_time"))
            yield row, side, trade_id, leg_id, pair_id, pd.Timestamp(entry_time), pd.Timestamp(exit_time)


def _nearest_at_or_before(g, timestamp, column):
    numeric = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = g.loc[(g["timestamp"] <= timestamp) & numeric.notna(), ["timestamp"]].copy()
    values["value"] = numeric[numeric.notna()]
    return values.iloc[-1]["value"] if len(values) else np.nan


def _phase_values(g, column, entry, exit_, phases):
    valid = g[["timestamp"]].copy()
    valid[column] = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna(subset=[column])
    if valid.empty:
        return [np.nan] * phases
    duration = (exit_ - entry).total_seconds()
    valid["fraction"] = ((valid.timestamp - entry).dt.total_seconds() / duration).clip(0, 1) if duration > 0 else 0.0
    result = []
    for phase in range(phases):
        lo, hi = phase / phases, (phase + 1) / phases
        mask = (valid.fraction >= lo) & (valid.fraction <= hi if phase == phases - 1 else valid.fraction < hi)
        if mask.any():
            result.append(float(valid.loc[mask, column].mean()))
        else:  # Sparse/short trades use the observation nearest the phase midpoint.
            midpoint = (lo + hi) / 2
            result.append(float(valid.loc[(valid.fraction - midpoint).abs().idxmin(), column]))
    return result


def build_lifecycle_analysis(trades: pd.DataFrame, telemetry: pd.DataFrame, phases: int = 4,
                             checkpoints=(15, 30, 60),
                             progress: Callable[[str, int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one lifecycle row per completed leg and all validation failures.

    Telemetry is joined strictly by ``pair_id`` (the telemetry's unique parent ID),
    then bounded inclusively by the leg's true entry and exit times. Empty phase
    bins use the nearest observation to the phase midpoint.
    """
    # Normalize telemetry and group it once.  The previous implementation scanned
    # the complete telemetry frame for every leg, which made this O(trades *
    # telemetry).  Group indices retain the exact original rows without making
    # thousands of per-trade copies.
    tel = telemetry
    if "timestamp" in tel and not pd.api.types.is_datetime64_any_dtype(tel["timestamp"]):
        tel = tel.copy()
        tel["timestamp"] = pd.to_datetime(tel["timestamp"])
    grouping_started = perf_counter()
    if not tel.empty and "leg_id" in tel:
        telemetry_groups = {key: group for key, group in tel.groupby("leg_id", sort=False, dropna=False)}
        grouping_key = "leg_id"
    elif not tel.empty and {"long_leg_id", "short_leg_id"}.intersection(tel.columns):
        telemetry_groups = {}
        for side_name in ("long", "short"):
            column = f"{side_name}_leg_id"
            if column in tel:
                telemetry_groups.update({key: group for key, group in tel.groupby(column, sort=False, dropna=False)})
        grouping_key = "leg_id"
    elif not tel.empty and "pair_id" in tel:
        telemetry_groups = {key: group for key, group in tel.groupby("pair_id", sort=False, dropna=False)}
        grouping_key = "pair_id"
    else:
        telemetry_groups = {}
        grouping_key = "pair_id"
    print(f"Lifecycle timing - telemetry grouping: {perf_counter() - grouping_started:.3f}s")
    rows, failures = [], []
    legs = list(_leg_rows(trades))
    total = len(legs)
    processing_started = perf_counter()
    for number, (trade, side, trade_id, leg_id, pair_id, entry, exit_) in enumerate(legs, 1):
        if progress:
            progress("lifecycle analysis", number, total)
        raw = telemetry_groups.get(leg_id if grouping_key == "leg_id" else pair_id, tel.iloc[0:0])
        flags = {name: False for name in VALIDATIONS}
        flags["missing_trade_id"] = pd.isna(trade_id) or pd.isna(leg_id) or pd.isna(pair_id)
        flags["exit_before_entry"] = exit_ < entry
        source_before = int((raw.timestamp < entry).sum()) if not raw.empty else 0
        source_after = int((raw.timestamp > exit_).sum()) if not raw.empty else 0
        if not raw.empty:
            flags["duplicate_telemetry_timestamp"] = bool(raw.timestamp.duplicated().any())
            flags["non_monotonic_telemetry_time"] = not raw.timestamp.is_monotonic_increasing
        g = raw[(raw.timestamp >= entry) & (raw.timestamp <= exit_)].sort_values("timestamp", kind="stable") if not raw.empty else raw
        included_before = int((g.timestamp < entry).sum()) if not g.empty else 0
        included_after = int((g.timestamp > exit_).sum()) if not g.empty else 0
        flags["missing_telemetry"] = g.empty
        missing = {prefix: col not in g or pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).isna().any()
                   for prefix, col in INDICATORS.items()}
        flags["missing_indicator_values"] = any(missing.values())
        risk = _num(trade, f"{side}_existing_r", "r_distance")
        flags["invalid_r_value"] = not np.isfinite(risk) or risk <= 0
        holding = max(0.0, (exit_ - entry).total_seconds() / 60)
        first_used = g.timestamp.min() if not g.empty else pd.NaT
        last_used = g.timestamp.max() if not g.empty else pd.NaT
        range_valid = bool(not g.empty and first_used >= entry and last_used <= exit_ and not included_before and not included_after)
        reconciliation = {"source_telemetry_exists_after_exit": source_after > 0,
            "analysis_included_telemetry_after_exit": included_after > 0,
            "source_rows_after_leg_exit": source_after, "included_rows_after_leg_exit": included_after,
            "source_rows_before_leg_entry": source_before, "included_rows_before_leg_entry": included_before,
            "first_used_telemetry_time": first_used, "last_used_telemetry_time": last_used,
            "used_telemetry_row_count": len(g), "excluded_before_entry_count": source_before,
            "excluded_after_exit_count": source_after, "lifecycle_range_valid": range_valid,
            **{f"missing_{name}": value for name, value in missing.items()}}
        out = {"trade_id": trade_id, "leg_id": leg_id, "pair_id": pair_id, "symbol": trade.get("symbol", trade.get("asset", "")), "side": side.upper(),
               "entry_time": entry, "exit_time": exit_, "exit_reason": trade.get(f"{side}_final_exit_reason", trade.get(f"{side}_exit_reason", trade.get("exit_reason"))),
               "is_winner": _num(trade, f"{side}_net_pnl", "pair_net_pnl") > 0, "holding_minutes": holding,
               "holding_hours": holding / 60, "final_gross_r": _num(trade, f"{side}_gross_r", "pair_gross_r"),
               "final_net_r": _num(trade, f"{side}_net_r", f"{side}_account_r", "pair_net_r"),
               "gross_pnl": _num(trade, f"{side}_gross_pnl", "pair_gross_pnl"),
               "fees": _num(trade, f"{side}_fees", f"{side}_total_fees", "pair_total_fees"),
               "net_pnl": _num(trade, f"{side}_net_pnl", "pair_net_pnl"), "telemetry_row_count": len(g), **reconciliation}
        for name, failed in flags.items():
            out[name] = failed
        out["lifecycle_validation_failed"] = any(flags.values())
        for prefix, column in INDICATORS.items():
            s = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan) if column in g else pd.Series(dtype=float)
            valid = g.loc[s.notna(), ["timestamp"]].copy(); valid["value"] = s[s.notna()].astype(float)
            first = valid.value.iloc[0] if len(valid) else np.nan; last = valid.value.iloc[-1] if len(valid) else np.nan
            change = last - first if np.isfinite(first) and np.isfinite(last) else np.nan
            out.update({f"{prefix}_entry": first, f"{prefix}_exit": last, f"{prefix}_min": valid.value.min(),
                        f"{prefix}_max": valid.value.max(), f"{prefix}_mean": valid.value.mean(), f"{prefix}_median": valid.value.median(),
                        f"{prefix}_std": valid.value.std(), f"{prefix}_change": change,
                        f"{prefix}_pct_change": change / first * 100 if np.isfinite(first) and first != 0 else np.nan})
            hours = (valid.timestamp - entry).dt.total_seconds().to_numpy() / 3600 if len(valid) else np.array([])
            out[f"{prefix}_slope_per_hour"] = float(np.polyfit(hours, valid.value, 1)[0]) if len(valid) > 1 and np.ptp(hours) > 0 else np.nan
            for extreme in ("max", "min"):
                idx = valid.value.idxmax() if extreme == "max" and len(valid) else (valid.value.idxmin() if len(valid) else None)
                when = valid.loc[idx, "timestamp"] if idx is not None else pd.NaT
                out[f"{prefix}_{extreme}_time"] = when
                out[f"{prefix}_minutes_to_{extreme}"] = (when - entry).total_seconds() / 60 if pd.notna(when) else np.nan
            phase = _phase_values(g, column, entry, exit_, phases) if column in g else [np.nan] * phases
            for i, value in enumerate(phase, 1): out[f"{prefix}_phase_{i}_mean"] = value
            for i in range(phases - 1): out[f"{prefix}_phase_{i+1}_to_{i+2}_change"] = phase[i+1] - phase[i]
            for minutes in checkpoints:
                out[f"{prefix}_change_{minutes}m"] = (_nearest_at_or_before(g, entry + pd.Timedelta(minutes=minutes), column) - first) if holding >= minutes and column in g and np.isfinite(first) else np.nan
            for pct in (25, 50):
                checkpoint = entry + (exit_ - entry) * (pct / 100)
                value = _nearest_at_or_before(g, checkpoint, column) if column in g else np.nan
                out[f"{prefix}_change_{pct}pct"] = value - first if np.isfinite(value) and np.isfinite(first) else np.nan
        unrealized = f"{side}_unrealized_profit_r"
        if unrealized in g:
            for i, value in enumerate(_phase_values(g, unrealized, entry, exit_, phases), 1): out[f"unrealized_r_phase_{i}_mean"] = value
        price_col = "high" if side == "long" else "low"; adverse_col = "low" if side == "long" else "high"
        if price_col in g and adverse_col in g and not g.empty:
            fav_idx = pd.to_numeric(g[price_col], errors="coerce").idxmax() if side == "long" else pd.to_numeric(g[price_col], errors="coerce").idxmin()
            adv_idx = pd.to_numeric(g[adverse_col], errors="coerce").idxmin() if side == "long" else pd.to_numeric(g[adverse_col], errors="coerce").idxmax()
            fav, adv = float(g.loc[fav_idx, price_col]), float(g.loc[adv_idx, adverse_col]); entry_price = _num(trade, f"{side}_entry_price", "entry_price")
            direction = 1 if side == "long" else -1
            out.update({"mfe_price": fav, "mae_price": adv,
                        "mfe_r": direction * (fav - entry_price) / risk if not flags["invalid_r_value"] else np.nan,
                        "mae_r": direction * (adv - entry_price) / risk if not flags["invalid_r_value"] else np.nan,
                        "minutes_to_mfe": (g.loc[fav_idx, "timestamp"] - entry).total_seconds()/60,
                        "minutes_to_mae": (g.loc[adv_idx, "timestamp"] - entry).total_seconds()/60,
                        "mfe_before_mae": g.loc[fav_idx, "timestamp"] < g.loc[adv_idx, "timestamp"],
                        "mae_before_mfe": g.loc[adv_idx, "timestamp"] < g.loc[fav_idx, "timestamp"]})
        rows.append(out)
        validation = {"trade_id": trade_id, "leg_id": leg_id, "pair_id": pair_id, "symbol": out["symbol"], "side": side.upper(),
                      "entry_time": entry, "exit_time": exit_, **reconciliation, **flags,
                      "genuine_validation_failure": any(flags.values()) or not range_valid}
        failures.append(validation)
    print(f"Lifecycle timing - leg lifecycle processing: {perf_counter() - processing_started:.3f}s")
    return pd.DataFrame(rows), pd.DataFrame(failures)


def _safe_stats(series):
    """Return nullable descriptive statistics without NumPy empty-slice warnings."""
    valid = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return (np.nan,) * 5
    return valid.mean(), valid.median(), valid.std(), valid.min(), valid.max()


def lifecycle_summary(analysis):
    rows=[]; numeric=analysis.select_dtypes(include=np.number).columns
    winners=analysis[analysis.is_winner.astype(bool)] if not analysis.empty else analysis; losers=analysis[~analysis.is_winner.astype(bool)] if not analysis.empty else analysis
    for metric in numeric:
        wm, *_ = _safe_stats(winners[metric]); lm, *_ = _safe_stats(losers[metric])
        for label, group in (("WINNER", winners), ("LOSER", losers), ("ALL", analysis)):
            s=pd.to_numeric(group[metric],errors="coerce"); valid=s.dropna()
            mean, median, std, minimum, maximum = _safe_stats(valid)
            rows.append({"group":label,"metric":metric,"trade_count":len(group),"non_null_count":len(valid),"mean":mean,"median":median,"std":std,"min":minimum,"max":maximum,"winner_mean":wm,"loser_mean":lm,"mean_difference":wm-lm,"percentage_difference":((wm-lm)/abs(lm)*100 if pd.notna(lm) and lm else np.nan),"correlation_with_final_net_r":analysis[[metric,"final_net_r"]].corr().iloc[0,1] if metric != "final_net_r" else 1.0,"correlation_with_net_pnl":analysis[[metric,"net_pnl"]].corr().iloc[0,1] if metric != "net_pnl" else 1.0})
    return pd.DataFrame(rows)


def phase_comparison(analysis, phases=4):
    rows=[]
    for prefix in (*INDICATORS, "unrealized_r"):
        for phase in range(1, phases+1):
            col=f"{prefix}_phase_{phase}_mean"
            if col not in analysis: continue
            w=analysis.loc[analysis.is_winner.astype(bool),col].dropna(); l=analysis.loc[~analysis.is_winner.astype(bool),col].dropna()
            rows.append({"indicator":prefix.upper(),"phase":PHASE_LABELS[phase-1] if phases==4 else f"{(phase-1)*100/phases:g}-{phase*100/phases:g}%","winner_trade_count":len(w),"loser_trade_count":len(l),"winner_mean":w.mean(),"loser_mean":l.mean(),"difference":w.mean()-l.mean(),"winner_median":w.median(),"loser_median":l.median()})
    return pd.DataFrame(rows)


def early_warning_analysis(analysis, minimum_sample=20):
    rows=[]
    metrics=[c for c in analysis if "_change_" in c and (c.endswith("m") or c.endswith("pct"))]
    for metric in metrics:
        valid=analysis.dropna(subset=[metric]).copy()
        if valid.empty: continue
        try: valid["bucket"]=pd.qcut(valid[metric], min(4, valid[metric].nunique()), duplicates="drop")
        except ValueError: valid["bucket"]="ALL"
        for bucket,g in valid.groupby("bucket", observed=True):
            winners=int(g.is_winner.sum()); count=len(g)
            rows.append({"metric":metric,"bucket":str(bucket),"trade_count":count,"winner_count":winners,"loser_count":count-winners,"win_rate":winners/count,"average_net_r":g.final_net_r.mean(),"median_net_r":g.final_net_r.median(),"average_net_pnl":g.net_pnl.mean(),"small_sample_warning":count<minimum_sample})
    return pd.DataFrame(rows)


def sequence_analysis(analysis, flat_threshold_pct=5.0):
    rows=[]
    if analysis.empty:
        return pd.DataFrame(columns=["indicator","pattern","trade_count","winner_count","loser_count","win_rate","average_net_r","average_net_pnl"])
    for prefix in INDICATORS:
        cols=[f"{prefix}_phase_{i}_mean" for i in range(1,5)]
        classified=[]
        for _,r in analysis.dropna(subset=cols).iterrows():
            v=r[cols].to_numpy(float); d=np.diff(v); scale=max(abs(v[0]),1e-12)
            if np.ptp(v)/scale*100 <= flat_threshold_pct: pattern="FLAT"
            elif np.all(d>=0): pattern="RISING"
            elif np.all(d<=0): pattern="FALLING"
            elif d[0]>0 and d[-1]<0: pattern="RISE_THEN_FALL"
            elif d[0]<0 and d[-1]>0: pattern="FALL_THEN_RISE"
            else: pattern="VOLATILE"
            classified.append((pattern,r))
        for pattern in ("RISING","FALLING","FLAT","RISE_THEN_FALL","FALL_THEN_RISE","VOLATILE"):
            selected=[r for p,r in classified if p==pattern]
            if not selected: continue
            g=pd.DataFrame(selected); count=len(g); wins=int(g.is_winner.sum())
            rows.append({"indicator":prefix.upper(),"pattern":pattern,"trade_count":count,"winner_count":wins,"loser_count":count-wins,"win_rate":wins/count,"average_net_r":g.final_net_r.mean(),"average_net_pnl":g.net_pnl.mean()})
    return pd.DataFrame(rows)


def save_lifecycle_charts(analysis, charts_dir):
    if analysis.empty: return
    import matplotlib.pyplot as plt
    charts_dir=Path(charts_dir); charts_dir.mkdir(parents=True,exist_ok=True)
    for prefix in (*INDICATORS,"unrealized_r"):
        cols=[f"{prefix}_phase_{i}_mean" for i in range(1,5)]
        if any(c not in analysis for c in cols): continue
        fig,ax=plt.subplots()
        plotted=False
        for label,mask in (("Winners",analysis.is_winner.astype(bool)),("Losers",~analysis.is_winner.astype(bool))):
            g=analysis.loc[mask,cols]
            if len(g) and g.notna().any().any(): ax.plot([12.5,37.5,62.5,87.5],g.mean(),marker="o",label=label); plotted=True
        if plotted:
            ax.set(xlabel="Normalized trade lifetime (%)",ylabel=prefix.upper(),title=f"{prefix.upper()} lifecycle (n={len(analysis)})"); ax.legend(); fig.tight_layout(); fig.savefig(charts_dir/f"{prefix}_lifecycle_winners_vs_losers.png")
        plt.close(fig)
    early = early_warning_analysis(analysis)
    if not early.empty:
        usable = early[~early.small_sample_warning]
        if not usable.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            shown = usable.head(20)
            ax.bar(range(len(shown)), shown.win_rate)
            ax.set_xticks(range(len(shown)), (shown.metric + "\n" + shown.bucket).tolist(), rotation=90)
            ax.set(ylabel="Win rate", title="Early indicator change win rate (buckets n>=20)")
            fig.tight_layout(); fig.savefig(charts_dir / "early_indicator_change_win_rate.png"); plt.close(fig)


def export_lifecycle_reports(trades, telemetry, run_dir, *, phases=4, checkpoints=(15,30,60), minimum_sample=20, charts=True, flat_threshold_pct=5.0, progress=None):
    total_started = perf_counter()
    analysis, validation=build_lifecycle_analysis(trades,telemetry,phases,checkpoints,progress)
    phase_started = perf_counter()
    phase = phase_comparison(analysis, phases)
    print(f"Lifecycle timing - phase report generation: {perf_counter() - phase_started:.3f}s")
    sequence_started = perf_counter()
    sequence = sequence_analysis(analysis, flat_threshold_pct)
    print(f"Lifecycle timing - sequence report generation: {perf_counter() - sequence_started:.3f}s")
    validation_started = perf_counter()
    validation = validation.copy()
    print(f"Lifecycle timing - validation report generation: {perf_counter() - validation_started:.3f}s")
    reports={"indicator_lifecycle_analysis.csv":analysis,"indicator_lifecycle_summary.csv":lifecycle_summary(analysis),"indicator_phase_comparison.csv":phase,"indicator_early_warning_analysis.csv":early_warning_analysis(analysis,minimum_sample),"indicator_sequence_analysis.csv":sequence,"indicator_lifecycle_validation.csv":validation}
    writing_started = perf_counter()
    for name,frame in reports.items():
        if progress: progress(f"writing {name}", 0, 0)
        frame.to_csv(Path(run_dir)/name,index=False)
    print(f"Lifecycle timing - CSV writing: {perf_counter() - writing_started:.3f}s")
    if charts:
        if progress: progress("creating lifecycle charts", 0, 0)
        save_lifecycle_charts(analysis,Path(run_dir)/"charts")
    print(f"Lifecycle timing - total lifecycle reports: {perf_counter() - total_started:.3f}s")
    return reports
