"""Output directory management for backtest runs."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_strategy_lab.config import BacktestConfig


def _safe_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-")


def _format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _risk_label(config: BacktestConfig) -> str:
    mode = getattr(config.risk_mode, "value", config.risk_mode)
    if mode == "ATR":
        return f"ATR{config.atr_period}x{_format_number(config.atr_multiplier)}"
    if mode == "PERCENT":
        return f"PCT{_format_number(config.percent_r * 100)}"
    return f"FIXED{_format_number(config.fixed_r)}"


def _profile_exit_labels(config: BacktestConfig) -> tuple[str, str]:
    profiles = [profile for profile in config.strategy_profiles.values() if profile.enabled]
    signatures = {
        (
            profile.partial_stop_enabled,
            profile.sl1_r,
            profile.sl1_close_pct,
            profile.sl2_r,
            profile.stop_loss_multiple,
            profile.partial_profit_enabled,
            profile.tp1_r,
            profile.tp1_close_pct,
            profile.tp2_r,
            profile.reward_risk_ratio,
        )
        for profile in profiles
    }
    if not profiles or len(signatures) != 1:
        return "MIXED", "EXITS"

    profile = profiles[0]
    if profile.partial_stop_enabled:
        stop = (
            f"PSL{_format_number(profile.sl1_r)}x{_format_number(profile.sl1_close_pct)}"
            f"-SL{_format_number(profile.sl2_r)}"
        )
    else:
        stop = f"SL{_format_number(profile.stop_loss_multiple)}"
    if profile.partial_profit_enabled:
        target = (
            f"PTP{_format_number(profile.tp1_r)}x{_format_number(profile.tp1_close_pct)}"
            f"-TP{_format_number(profile.tp2_r)}"
        )
    else:
        target = f"TP{_format_number(profile.stop_loss_multiple * profile.reward_risk_ratio)}"
    return stop, target


def _profile_mode_label(config: BacktestConfig) -> str:
    labels = {
        "ISOLATED_PROFILES": "PROFILES-ISOLATED",
        "COMBINED_SHARED_CAPITAL": "PROFILES-COMBINED",
        "BOTH": "PROFILES-BOTH",
    }
    return labels.get(config.strategy_profile_run_mode, f"PROFILES-{_safe_part(config.strategy_profile_run_mode)}")


def infer_symbol(config: BacktestConfig) -> str:
    stem = Path(config.input_csv).stem.upper()
    match = re.match(r"([A-Z]+?)(?:USDT|USD|BTC|ETH)?(?:_|-|$)", stem)
    return match.group(1) if match else "BACKTEST"


def run_folder_name(config: BacktestConfig, timestamp: datetime | None = None) -> str:
    stamp = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    stop, target = _profile_exit_labels(config)
    parts = [
        infer_symbol(config),
        f"{config.strategy_timeframe_minutes}m",
        _risk_label(config),
    ]
    parts.append(_profile_mode_label(config))
    parts.extend([
        stop,
        target,
        stamp,
    ])
    base = "_".join(parts)
    run_name = _safe_part(config.run_name or "")
    return f"{run_name}_{base}" if run_name else base


def planned_run_dir(config: BacktestConfig) -> Path:
    if config.output_run_dir is not None:
        return Path(config.output_run_dir)
    return Path(config.output_dir) / run_folder_name(config)


def create_run_dir(config: BacktestConfig) -> Path:
    path = planned_run_dir(config)
    counter = 1
    while path.exists():
        path = path.with_name(f"{path.name}_{counter}")
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    (path / "charts").mkdir(exist_ok=True)
    return path


def config_to_dict(config: BacktestConfig) -> dict[str, Any]:
    raw = asdict(config) if is_dataclass(config) else {f.name: getattr(config, f.name) for f in fields(BacktestConfig)}
    return {key: _jsonable(value) for key, value in raw.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_config_snapshot(
    path: str | Path,
    *,
    require_paths: bool = True,
) -> tuple[BacktestConfig, tuple[str, ...]]:
    """Rehydrate a saved run ``config.json`` into the current config contract.

    Run outputs intentionally serialize the entire :class:`BacktestConfig`, so
    historical snapshots can contain fields that have since been retired. Only
    fields that still exist in the current dataclass are restored; retired fields
    are returned to the caller for explicit reporting rather than silently
    affecting current behavior.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Saved run config JSON must contain an object.")
    if "input_csv" not in raw:
        raise ValueError("Saved run config is missing input_csv.")

    allowed = {field.name for field in fields(BacktestConfig)}
    ignored = tuple(sorted(set(raw) - allowed))
    current = {key: value for key, value in raw.items() if key in allowed}
    for key in (
        "input_csv",
        "intrabar_csv",
        "output_dir",
        "structural_regime_benchmark_csv",
        "output_run_dir",
    ):
        if key in current and current[key] not in (None, ""):
            current[key] = Path(current[key])

    config = BacktestConfig(**current)
    if require_paths:
        if not Path(config.input_csv).is_file():
            raise ValueError(f"Strategy CSV does not exist: {config.input_csv}")
        if config.use_intrabar_data and config.intrabar_csv and not Path(config.intrabar_csv).is_file():
            raise ValueError(f"Intrabar CSV does not exist: {config.intrabar_csv}")
    return config, ignored


def write_config(config: BacktestConfig, run_dir: Path) -> None:
    (run_dir / "config.json").write_text(json.dumps(config_to_dict(config), indent=2, default=str))


def write_run_info(config: BacktestConfig, summary: dict[str, Any], run_dir: Path) -> None:
    enabled_profiles = {key: profile for key, profile in config.strategy_profiles.items() if profile.enabled}
    trailing_profiles = [key for key, profile in enabled_profiles.items() if profile.trailing_enabled or profile.r_step_trailing_enabled]
    break_even_profiles = [key for key, profile in enabled_profiles.items() if profile.break_even_enabled]
    timeout_profiles = [key for key, profile in enabled_profiles.items() if profile.timeout_enabled]
    lines = [
        "Backtest Run Information",
        "========================",
        f"Output folder: {run_dir.resolve()}",
        f"Run name: {config.run_name or '(none)'}",
        f"Strategy CSV: {config.input_csv}",
        f"Intrabar CSV: {config.intrabar_csv if config.use_intrabar_data else '(disabled)'}",
        f"Symbol: {infer_symbol(config)}",
        f"Strategy timeframe: {config.strategy_timeframe_minutes}m",
        f"Stop distance basis: {config.risk_mode.value}",
        f"Base account risk per trade: {config.risk_per_leg * 100:g}%",
        f"ATR distance-unit period/multiplier: {config.atr_period} / {config.atr_multiplier}",
        "Strategy Profiles: " + (", ".join(enabled_profiles) if enabled_profiles else "none enabled"),
        f"Strategy Profile run mode: {config.strategy_profile_run_mode}",
        "Trailing profiles: " + (", ".join(trailing_profiles) if trailing_profiles else "none"),
        "Break-even profiles: " + (", ".join(break_even_profiles) if break_even_profiles else "none"),
        "Timeout profiles: " + (", ".join(timeout_profiles) if timeout_profiles else "none"),
        f"Partial intrabar ordering: {'STOP_FIRST' if config.tie_policy.value == 'PESSIMISTIC' else 'TP1_THEN_TP2_THEN_STOP'}",
        f"Initial equity: {config.initial_equity}",
        f"Total pairs: {summary.get('total_pairs')}",
        f"Ending equity: {summary.get('ending_equity')}",
        f"Total return %: {summary.get('total_return_percentage')}",
    ]
    (run_dir / "run_info.txt").write_text("\n".join(lines) + "\n")


def compatible_resample_freq(freq: str) -> str:
    """Map legacy pandas aliases to modern aliases by default."""
    return {"M": "ME", "Y": "YE"}.get(freq, freq)


def periodic_results(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "pair_count", "net_pnl", "net_r"])
    exits = pd.to_datetime(trades.get("exit_time", pd.Series(pd.NaT, index=trades.index)), errors="coerce", utc=True)
    side_exit_columns = [column for column in ("long_exit_time", "short_exit_time") if column in trades]
    if side_exit_columns and exits.isna().any():
        side_exits = pd.concat(
            [pd.to_datetime(trades[column], errors="coerce", utc=True) for column in side_exit_columns],
            axis=1,
        ).max(axis=1)
        exits = exits.fillna(side_exits)
    if exits.isna().all():
        raise ValueError("Periodic results require at least one valid trade exit timestamp.")
    # Keep this narrow. Copying the full telemetry-rich trade frame also deep
    # copies its large attrs (notably skipped_signals) in recent pandas.
    frame = pd.DataFrame({"exit_time": exits, "pair_net_pnl": trades["pair_net_pnl"], "pair_net_r": trades["pair_net_r"]}).set_index("exit_time")
    frame = frame.loc[frame.index.notna()]
    candidates = [compatible_resample_freq(freq)]
    fallback = {"ME": "M", "YE": "Y", "M": "ME", "Y": "YE"}.get(candidates[0])
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            periodic = frame.resample(candidate).agg(pair_count=("pair_net_pnl", "size"), net_pnl=("pair_net_pnl", "sum"), net_r=("pair_net_r", "sum"))
            periodic = periodic.reset_index()
            return periodic.rename(columns={periodic.columns[0]: "period"})
        except ValueError as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError(f"Unsupported resample frequency: {freq}")


def update_latest(output_root: Path, run_dir: Path) -> None:
    latest = output_root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(run_dir, latest)

TRADE_R_COLUMN_METADATA = {
    "distance_unit_price": "Entry-time price-distance unit selected by the configured distance basis (for example ATR × multiplier).",
    "trade_r_price_distance": "Full initial stop distance in price units. This is the price meaning of 1 trade R.",
    "configured_account_risk_percentage": "Configured account-equity percentage planned to be lost at the initial full stop before fees and slippage.",
    "estimated_all_in_stop_risk_percentage": "Estimated account-equity loss at stop after entry fee, stop-exit fee, and configured slippage.",
    "*_price_r": "Realized price movement divided by the full initial stop distance; 1.0 means one trade R of favourable price movement.",
    "*_account_r": "Realized account PnL divided by planned account risk at entry; 1.0 means one configured account-risk unit.",
    "*_effective_leverage": "Entry notional divided by equity at the moment that leg was sized.",
    "pair_account_r": "Pair-level realized account PnL divided by the pair planned account risk at entry.",
}


def write_trade_column_metadata(run_dir: Path) -> None:
    (run_dir / "trade_list_column_metadata.json").write_text(json.dumps(TRADE_R_COLUMN_METADATA, indent=2), encoding="utf-8")
