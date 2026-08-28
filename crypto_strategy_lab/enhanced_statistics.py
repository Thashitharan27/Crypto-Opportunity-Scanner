"""Extended reporting for the Bollinger + RSI mean-reversion telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_strategy_lab.statistics import mean_reversion_analysis as legacy_mean_reversion_analysis


EXTRA_COLUMNS = [
    "MR Signal",
    "MR Signal Direction",
    "BB Location",
    "RSI State",
    "BB Re-entry",
    "Mean Type",
    "Confirmed MR",
    "Confirmed Direction",
    "Confirmed Alignment",
]


def mean_reversion_analysis_v2(trades: pd.DataFrame) -> pd.DataFrame:
    """Return legacy MR tables plus BB/RSI/re-entry research cross-tabs.

    The primary validation section deliberately collapses the classical MR idea
    into only three trade-alignment groups for DI 0-30: AGREE, DISAGREE, and
    NO_SIGNAL. Potential outside-band setups remain available in the detailed
    telemetry but are not treated as confirmed mean reversion.
    """
    legacy = legacy_mean_reversion_analysis(trades)
    columns = list(legacy.columns) + [column for column in EXTRA_COLUMNS if column not in legacy.columns]
    legacy = legacy.reindex(columns=columns)

    required = {"di_spread", "mean_reversion_signal"}
    if trades.empty or not required.issubset(trades.columns):
        return legacy

    frame = trades.copy()
    di = pd.to_numeric(frame["di_spread"], errors="coerce")
    edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, np.inf]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50+"]
    frame["DI Pressure Bucket"] = pd.cut(di, edges, right=False, labels=labels)
    frame["Direction"] = frame.get(
        "di_sizing_direction",
        frame.get("sizing_direction", pd.Series("UNKNOWN", index=frame.index)),
    ).fillna("UNKNOWN").astype(str).str.upper()
    frame["Market Regime"] = frame.get("market_regime", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["DI Pressure State"] = frame.get("di_pressure_state", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion Alignment"] = frame.get("mean_reversion_alignment", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion Motion"] = frame.get("mean_reversion_motion", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion State"] = frame.get("mean_reversion_state", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Mean Reversion Strength"] = frame.get("mean_reversion_strength_label", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["MR Signal"] = frame["mean_reversion_signal"].fillna("UNKNOWN").astype(str).str.upper()
    frame["MR Signal Direction"] = frame.get("mean_reversion_signal_direction", pd.Series("NONE", index=frame.index)).fillna("NONE").astype(str).str.upper()
    frame["BB Location"] = frame.get("mean_reversion_bb_location", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["RSI State"] = frame.get("mean_reversion_rsi_state", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["BB Re-entry"] = frame.get("bb_reentry", frame.get("mean_reversion_reentry_confirmation", pd.Series("NONE", index=frame.index))).fillna("NONE").astype(str).str.upper()
    frame["Mean Type"] = frame.get("mean_reversion_mean_type", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str).str.upper()
    frame["Confirmed MR"] = frame.get("mr_signal", pd.Series("NO_SIGNAL", index=frame.index)).fillna("NO_SIGNAL").astype(str).str.upper()
    frame["Confirmed Direction"] = frame.get("mr_signal_direction", pd.Series("NONE", index=frame.index)).fillna("NONE").astype(str).str.upper()
    frame["Confirmed Alignment"] = frame.get("mr_trade_alignment", pd.Series("NO_SIGNAL", index=frame.index)).fillna("NO_SIGNAL").astype(str).str.upper()
    frame["_pnl"] = pd.to_numeric(frame.get("pair_net_pnl"), errors="coerce")
    frame["_r"] = pd.to_numeric(frame.get("pair_net_r"), errors="coerce")

    def grouped(source: pd.DataFrame, section: str, groups: list[str]) -> list[dict]:
        rows: list[dict] = []
        for keys, group in source.groupby(groups, dropna=False, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            pnl = group["_pnl"]
            rr = group["_r"]
            rows.append(
                {
                    "Section": section,
                    **dict(zip(groups, keys)),
                    "Trades": int(len(group)),
                    "Wins": int((pnl > 0).sum()),
                    "Losses": int((pnl < 0).sum()),
                    "Win Rate": float((pnl > 0).mean()),
                    "Average Net PnL": float(pnl.mean()),
                    "Net PnL": float(pnl.sum()),
                    "Average Net R": float(rr.mean()),
                    "Total Net R": float(rr.sum()),
                }
            )
        return rows

    rows: list[dict] = []

    # Primary hypothesis test: one broad confirmed-MR rule across the entire
    # low-to-moderate DI range, with potential/unconfirmed setups counted as NO_SIGNAL.
    primary = frame.loc[di.ge(0) & di.lt(30)].copy()
    if not primary.empty:
        rows += grouped(primary, "DI 0-30 Confirmed MR Alignment", ["Confirmed Alignment"])
        rows += grouped(primary, "DI 0-30 Confirmed MR Direction", ["Confirmed Direction", "Confirmed Alignment"])
        rows += grouped(primary, "DI 0-30 Confirmed MR + DI State", ["DI Pressure State", "Confirmed Alignment"])

    # Detailed research views remain available after the broad test is established.
    rows += grouped(frame, "MR V2 Signal", ["MR Signal"])
    rows += grouped(frame, "DI Bucket + MR V2 Signal", ["DI Pressure Bucket", "MR Signal"])
    rows += grouped(frame, "Direction + DI Bucket + MR V2 Signal", ["Direction", "DI Pressure Bucket", "MR Signal"])
    rows += grouped(frame, "DI State + DI Bucket + MR V2 Signal", ["DI Pressure State", "DI Pressure Bucket", "MR Signal"])
    rows += grouped(frame, "Regime + DI Bucket + MR V2 Signal", ["Market Regime", "DI Pressure Bucket", "MR Signal"])
    rows += grouped(frame, "BB Location + RSI State", ["BB Location", "RSI State"])
    rows += grouped(frame, "BB Re-entry", ["BB Re-entry", "MR Signal"])

    enhanced = pd.DataFrame(rows).reindex(columns=columns)
    return pd.concat([legacy, enhanced], ignore_index=True)
