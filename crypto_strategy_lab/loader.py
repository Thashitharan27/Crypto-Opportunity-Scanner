"""CSV loading utilities for Binance OHLCV files."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
TIMESTAMP_ALIASES = ("timestamp", "open_time", "time", "datetime", "date")

@dataclass
class DataSummary:
    label: str; rows_loaded: int; start: object; end: object; detected_timeframe_minutes: int | None; missing_candles: int; gaps: list[tuple[object, object, int]]

def _detect(diffs: pd.Series) -> int | None:
    if diffs.empty: return None
    mode = diffs.dt.total_seconds().div(60).mode()
    return int(mode.iloc[0]) if not mode.empty else None

def load_ohlcv_csv(path: str, timestamp_unit: str | None = "ms", expected_timeframe_minutes: int | None = None, label: str = "Data", strict_timeframe: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    time_col = next((c for c in TIMESTAMP_ALIASES if c in df.columns), None)
    if time_col is None: raise ValueError(f"Missing timestamp column. Accepted aliases: {list(TIMESTAMP_ALIASES)}")
    if time_col != "timestamp": df = df.rename(columns={time_col: "timestamp"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df = df.loc[:, list(REQUIRED_COLUMNS)].copy(); before = len(df)
    raw = df["timestamp"].copy(); unit = timestamp_unit or None
    df["timestamp"] = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
    if df["timestamp"].isna().any() and unit is not None:
        df["timestamp"] = df["timestamp"].fillna(pd.to_datetime(raw, utc=True, errors="coerce"))
    if df["timestamp"].isna().any(): raise ValueError("timestamps are invalid")
    for c in ("open","high","low","close","volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    invalid = (df["high"] < df[["open","close","low"]].max(axis=1)) | (df["low"] > df[["open","close","high"]].min(axis=1)) | (df["volume"] < 0)
    invalid_count = int(invalid.sum()); df = df.loc[~invalid]
    df = df.sort_values("timestamp", kind="stable")
    dup = int(df["timestamp"].duplicated().sum()); df = df.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    if df.empty: raise ValueError("No valid OHLCV rows loaded")
    diffs = df["timestamp"].diff().dropna(); detected = _detect(diffs)
    basis = expected_timeframe_minutes or 15
    expected = pd.Timedelta(minutes=basis)
    gaps_s = diffs[diffs > expected]
    gaps=[]; missing_candles=0
    for idx, delta in gaps_s.items():
        miss = int(delta / expected) - 1; missing_candles += miss; gaps.append((df.loc[idx-1,"timestamp"], df.loc[idx,"timestamp"], miss))
    if strict_timeframe and expected_timeframe_minutes and detected != expected_timeframe_minutes:
        raise ValueError(f"{label} data is not {expected_timeframe_minutes}-minute data (detected {detected})")
    print(f"{label}:\nRows loaded: {len(df)}\nStart: {df.timestamp.min()}\nEnd: {df.timestamp.max()}\nDetected timeframe: {detected} minutes\nMissing candles: {missing_candles}\nmissing_{basis}m_candles={missing_candles}")
    if missing_candles: print(f"WARNING: {label} has {missing_candles} missing candles across {len(gaps)} gaps.")
    df.attrs["summary"] = DataSummary(label, len(df), df.timestamp.min(), df.timestamp.max(), detected, missing_candles, gaps)
    return df

def load_backtest_data(config, strategy_data: pd.DataFrame | None = None):
    """Load input data, reusing an already validated strategy frame when supplied."""
    strat = strategy_data if strategy_data is not None else load_ohlcv_csv(str(config.input_csv), config.timestamp_unit, config.strategy_timeframe_minutes, "Strategy data", True)
    if getattr(config, "data_start_date", None):
        data_start = pd.Timestamp(config.data_start_date, tz="UTC")
        strat = strat.loc[strat.timestamp >= data_start].reset_index(drop=True)
        if strat.empty: raise ValueError("No strategy rows remain on or after data_start_date")
    intra = None
    if config.use_intrabar_data and config.intrabar_csv:
        intra = load_ohlcv_csv(str(config.intrabar_csv), config.timestamp_unit, config.intrabar_timeframe_minutes, "Intrabar data", True)
        if intra.timestamp.max() <= strat.timestamp.min() or intra.timestamp.min() >= strat.timestamp.max() + pd.Timedelta(minutes=config.strategy_timeframe_minutes):
            raise ValueError("intrabar data does not overlap the strategy period")
    return strat, intra
