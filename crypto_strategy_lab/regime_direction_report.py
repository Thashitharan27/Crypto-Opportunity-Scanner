"""Market-regime and trade-direction summary reports.

These reports are diagnostic only. They do not change entry, exit, sizing, or
filter behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PROFILE_ORDER = [
    ("BULL", "LONG", "Bull Long"),
    ("BEAR", "SHORT", "Bear Short"),
    ("BULL", "SHORT", "Bull Short"),
    ("BEAR", "LONG", "Bear Long"),
    ("SIDEWAYS", "LONG", "Sideways Long"),
    ("SIDEWAYS", "SHORT", "Sideways Short"),
]


def _direction_series(trades: pd.DataFrame) -> pd.Series:
    """Return the effective entry direction using the best available telemetry."""
    index = trades.index
    direction = pd.Series(index=index, dtype=object)

    for column in ("sizing_direction", "di_sizing_direction"):
        if column in trades:
            values = trades[column].astype("string").str.upper()
            direction = direction.where(direction.notna(), values)

    # Older trade lists may not contain an explicit selected direction. DI is a
    # safe fallback because it is the direction source used by the strategy.
    if direction.isna().any() and "plus_di" in trades and "minus_di" in trades:
        plus_di = pd.to_numeric(trades["plus_di"], errors="coerce")
        minus_di = pd.to_numeric(trades["minus_di"], errors="coerce")
        inferred = pd.Series(
            np.where(plus_di > minus_di, "LONG", np.where(minus_di > plus_di, "SHORT", None)),
            index=index,
            dtype=object,
        )
        direction = direction.where(direction.notna(), inferred)

    return direction.astype("string").str.upper()


def _holding_minutes(trades: pd.DataFrame) -> pd.Series:
    if "holding_minutes" in trades:
        return pd.to_numeric(trades["holding_minutes"], errors="coerce")
    if "holding_hours" in trades:
        return pd.to_numeric(trades["holding_hours"], errors="coerce") * 60.0
    return pd.Series(np.nan, index=trades.index, dtype=float)


def _metrics(profile: str, group: pd.DataFrame) -> dict[str, object]:
    pnl = pd.to_numeric(group.get("pair_net_pnl", pd.Series(index=group.index, dtype=float)), errors="coerce")
    holding = _holding_minutes(group)
    wins = pnl > 0
    losses = pnl < 0
    gross_profit = float(pnl.loc[wins].sum())
    gross_loss = float(-pnl.loc[losses].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    return {
        "Profile": profile,
        "Trades": int(len(group)),
        "Wins": int(wins.sum()),
        "Losses": int(losses.sum()),
        "Win Rate": float(wins.mean()) if len(group) else 0.0,
        "Average Net PnL": float(pnl.mean()) if len(group) else 0.0,
        "Total Net PnL": float(pnl.sum()) if len(group) else 0.0,
        "Average Holding Time": float(holding.mean()) if len(group) and holding.notna().any() else 0.0,
        "Profit Factor": float(profit_factor),
    }


def _prepared(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["_report_regime"] = frame.get(
        "market_regime", pd.Series(index=frame.index, dtype=object)
    ).astype("string").str.upper()
    frame["_report_direction"] = _direction_series(frame)
    frame["_report_di_spread"] = pd.to_numeric(
        frame.get("di_spread", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    )
    return frame


def regime_direction_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize Bull/Bear/Sideways x Long/Short performance."""
    frame = _prepared(trades)
    rows: list[dict[str, object]] = []
    for regime, direction, profile in PROFILE_ORDER:
        group = frame.loc[
            frame["_report_regime"].eq(regime) & frame["_report_direction"].eq(direction)
        ]
        rows.append(_metrics(profile, group))
    return pd.DataFrame(rows)


def regime_direction_di_summary(trades: pd.DataFrame, threshold: float = 30.0) -> pd.DataFrame:
    """Compare DI spread below/above ``threshold`` for each regime/direction profile."""
    frame = _prepared(trades)
    rows: list[dict[str, object]] = []
    bands = [
        (f"DI < {threshold:g}", frame["_report_di_spread"] < threshold),
        (f"DI >= {threshold:g}", frame["_report_di_spread"] >= threshold),
    ]
    for regime, direction, profile in PROFILE_ORDER:
        profile_mask = frame["_report_regime"].eq(regime) & frame["_report_direction"].eq(direction)
        for band, di_mask in bands:
            row = _metrics(profile, frame.loc[profile_mask & di_mask])
            row["DI Band"] = band
            rows.append(row)

    columns = [
        "Profile", "DI Band", "Trades", "Wins", "Losses", "Win Rate",
        "Average Net PnL", "Total Net PnL", "Average Holding Time", "Profit Factor",
    ]
    return pd.DataFrame(rows, columns=columns)
