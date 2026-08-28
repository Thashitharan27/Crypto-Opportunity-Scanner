"""Bounded composition adapters around the proven native simulator implementation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
import time

import pandas as pd

from .bayesian_research import enrich_bayesian_trade_probabilities
from .funding_execution import FundingAwareRuleBacktestEngine as RuleAwareDataLakeProductionBacktestEngine
from .gui.enhanced_config import enhanced_default_gui_config, build_enhanced_backtest_config
from .strategy_profiles import StrategyProfile


@dataclass(frozen=True)
class BoundNativeStrategyPolicy:
    """The strategy policy explicitly bound to one prepared market-fact frame."""

    config: object


@dataclass(frozen=True)
class PreparedPolicyConfig:
    """Only policy-market inputs physically materialized into PreparedBacktestFrame."""

    strategy_timeframe_minutes: int
    market_regime_method: str
    bull_regime_lookback_days: int
    bull_regime_return_threshold: float
    structural_regime_sma_days: int
    structural_regime_slope_lookback_days: int
    strategy_profiles: dict[str, object]
    market_symbol: str = "POLICY"


def prepared_policy_config(run_config) -> PreparedPolicyConfig:
    """Project the composed config to the exact inputs that participate in L3."""

    profiles = {
        key: SimpleNamespace(
            rsi_period=int(profile.rsi_period),
            momentum_lookback_hours=int(profile.momentum_lookback_hours),
        )
        for key, profile in run_config.strategy.profiles.items()
    }
    return PreparedPolicyConfig(
        strategy_timeframe_minutes=int(run_config.data.strategy_timeframe_minutes),
        market_regime_method=str(run_config.features.market_regime_method),
        bull_regime_lookback_days=int(run_config.features.bull_regime_lookback_days),
        bull_regime_return_threshold=float(run_config.features.bull_regime_return_threshold),
        structural_regime_sma_days=int(run_config.features.structural_regime_sma_days),
        structural_regime_slope_lookback_days=int(
            run_config.features.structural_regime_slope_lookback_days
        ),
        strategy_profiles=profiles,
    )


def native_simulator_config(data_config, feature_config, strategy_config, execution_config):
    """Translate composition only at the mature simulator boundary.

    ReportingConfig is intentionally absent: report/output changes cannot alter
    simulation configuration or L3 identity. The returned EnhancedBacktestConfig
    is an implementation adapter, not the authoritative serialized contract.
    """

    values = enhanced_default_gui_config()
    for component in (data_config, feature_config, strategy_config, execution_config):
        for name, value in asdict(component).items():
            if name != "profiles" and name in values:
                values[name] = value
    values["strategy_profiles"] = {
        key: asdict(
            StrategyProfile(
                **{
                    **asdict(strategy_config.profiles[key]),
                    **asdict(execution_config.profiles[key]),
                }
            )
        )
        for key in strategy_config.profiles
    }
    # Inert compatibility fields required by the mature config constructor.
    # Telemetry/reporting is owned outside the simulator in the composed path;
    # using the strategy interval here only satisfies the legacy validation rule.
    values.update(
        input_csv="",
        intrabar_csv=None,
        output_dir="output",
        structural_regime_benchmark_csv=None,
        telemetry_interval_minutes=int(data_config.strategy_timeframe_minutes),
        enable_trade_telemetry=False,
        save_full_telemetry_csv=False,
        save_trade_journey_summary=False,
        save_trade_journey_charts=False,
        enable_indicator_lifecycle_analysis=False,
        config_version=2,
    )
    return build_enhanced_backtest_config(values, require_paths=False)


def _signal_frame(prepared, trades: pd.DataFrame, skipped_signals) -> pd.DataFrame:
    """Serialize decisions already made by the same simulator run.

    Rejections come from ``engine.skipped_signals``, which is populated at the
    decision point. Entries come from completed trade rows.  The only join used
    for rejected rows is an exact strategy-candle timestamp lookup into the
    already-bound PreparedBacktestFrame; there is no nearest/asof lookup and no
    strategy re-evaluation.

    PreparedBacktestFrame stores UTC instants as ``datetime64[ns]`` without a
    timezone tag.  Signal artifacts deliberately use that same UTC-normalized
    storage representation so DuckDB comparisons are deterministic on hosts
    whose local timezone is not UTC.
    """
    columns = [
        "signal_id",
        "strategy_index",
        "candle_open_time",
        "decision_available_at",
        "side",
        "profile",
        "decision",
        "reason_code",
        "proposed_entry",
        "proposed_stop",
        "proposed_target",
    ]

    prepared_times = pd.to_datetime(prepared.timestamp, utc=True)
    availability = pd.to_datetime(prepared.decision_available_at, utc=True)
    exact_index = {timestamp.value: i for i, timestamp in enumerate(prepared_times)}
    if len(exact_index) != len(prepared_times):
        raise ValueError("prepared strategy timestamps are not unique for signal capture")

    rows: list[dict[str, object]] = []
    for number, raw in enumerate(skipped_signals or (), start=1):
        candle = pd.Timestamp(raw.get("strategy_candle_open_time"))
        if candle.tzinfo is None:
            candle = candle.tz_localize("UTC")
        else:
            candle = candle.tz_convert("UTC")
        try:
            strategy_index = exact_index[candle.value]
        except KeyError as exc:
            raise ValueError(
                "rejected signal does not map to an exact prepared strategy row"
            ) from exc
        plus_di = pd.to_numeric(raw.get("plus_di"), errors="coerce")
        minus_di = pd.to_numeric(raw.get("minus_di"), errors="coerce")
        side = None
        if pd.notna(plus_di) and pd.notna(minus_di) and plus_di != minus_di:
            side = "LONG" if plus_di > minus_di else "SHORT"
        rows.append(
            {
                "signal_id": f"reject-{number}",
                "strategy_index": int(strategy_index),
                "candle_open_time": prepared_times[strategy_index],
                "decision_available_at": availability[strategy_index],
                "side": side,
                "profile": raw.get("strategy_profile_key"),
                "decision": "REJECT",
                "reason_code": raw.get("entry_filter_reason") or raw.get("reason"),
                "proposed_entry": raw.get("strategy_entry_price"),
                "proposed_stop": None,
                "proposed_target": None,
            }
        )

    if not trades.empty:
        required = {
            "research_signal_index",
            "research_signal_candle_open_time",
            "research_signal_available_at",
            "side",
        }
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(
                f"completed trades cannot produce exact signal provenance: {sorted(missing)}"
            )
        for row_number, (_, trade) in enumerate(trades.iterrows(), start=1):
            strategy_index = int(trade["research_signal_index"])
            if strategy_index < 0 or strategy_index >= len(prepared_times):
                raise ValueError("entered signal strategy index is outside prepared frame")
            candle = pd.Timestamp(trade["research_signal_candle_open_time"])
            available = pd.Timestamp(trade["research_signal_available_at"])
            if candle.tzinfo is None:
                candle = candle.tz_localize("UTC")
            else:
                candle = candle.tz_convert("UTC")
            if available.tzinfo is None:
                available = available.tz_localize("UTC")
            else:
                available = available.tz_convert("UTC")
            if candle != prepared_times[strategy_index] or available != availability[strategy_index]:
                raise ValueError("entered signal causal attachment disagrees with prepared frame")
            pair_id = trade.get("pair_id", row_number)
            rows.append(
                {
                    "signal_id": f"enter-{pair_id}",
                    "strategy_index": strategy_index,
                    "candle_open_time": candle,
                    "decision_available_at": available,
                    "side": str(trade["side"]),
                    "profile": trade.get("strategy_profile_key"),
                    "decision": "ENTER",
                    "reason_code": "ENTERED",
                    "proposed_entry": trade.get("strategy_entry_price"),
                    "proposed_stop": trade.get("initial_stop_price"),
                    "proposed_target": trade.get("initial_target_price"),
                }
            )

    if not rows:
        return pd.DataFrame(
            {
                "signal_id": pd.Series(dtype="string"),
                "strategy_index": pd.Series(dtype="int64"),
                "candle_open_time": pd.Series(dtype="datetime64[ns]"),
                "decision_available_at": pd.Series(dtype="datetime64[ns]"),
                "side": pd.Series(dtype="string"),
                "profile": pd.Series(dtype="string"),
                "decision": pd.Series(dtype="string"),
                "reason_code": pd.Series(dtype="string"),
                "proposed_entry": pd.Series(dtype="float64"),
                "proposed_stop": pd.Series(dtype="float64"),
                "proposed_target": pd.Series(dtype="float64"),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    result["strategy_index"] = pd.to_numeric(
        result["strategy_index"], errors="raise"
    ).astype("int64")
    result["candle_open_time"] = pd.to_datetime(
        result["candle_open_time"], utc=True
    ).dt.tz_localize(None)
    result["decision_available_at"] = pd.to_datetime(
        result["decision_available_at"], utc=True
    ).dt.tz_localize(None)
    return result.sort_values(
        ["strategy_index", "decision", "signal_id"], kind="stable"
    ).reset_index(drop=True)


def _release_canonicalized_rejection_metadata(trades: pd.DataFrame) -> None:
    """Drop the legacy duplicate after rejected decisions have canonical storage.

    ``BacktestEngine.results_frame`` attaches the complete ``skipped_signals``
    list to ``DataFrame.attrs`` for legacy callers.  In the composed v3 path the
    same decisions are already materialized in ``signals.parquet`` via
    ``_signal_frame``.  Keeping the high-cardinality list on the trades frame
    makes pandas copy/deepcopy that payload again during artifact publication,
    which can make heavily filtered Entry Rule runs spend minutes in reporting.

    Trade columns, signal rows, daily schedule summary metadata and trading
    semantics are intentionally untouched.
    """
    trades.attrs.pop("skipped_signals", None)


def _is_completed_trade_result(trades: pd.DataFrame) -> bool:
    """Distinguish real completed-result frames from narrow provenance test doubles.

    Native production results always publish PnL and an exit timestamp.  Some
    adapter tests deliberately use a smaller frame containing only signal
    provenance because they are exercising rejection-memory behavior.  Empty
    frames are still enriched so normal no-trade runs receive the stable Bayesian
    output schema.  Once a frame has the two production sentinels, the Bayesian
    module remains strict about every other required column.
    """
    return trades.empty or {"pair_net_pnl", "exit_time"}.issubset(trades.columns)


class NativeSimulator:
    """Composition adapter; deliberately not an engine subclass."""

    def __init__(self):
        self.last_engine_init_seconds = 0.0
        self.last_simulation_seconds = 0.0
        self.last_signals: pd.DataFrame | None = None
        self.last_telemetry: pd.DataFrame | None = None

    def run(
        self,
        prepared,
        intrabar,
        strategy,
        execution_config,
        *,
        data_config,
        feature_config,
    ):
        if not isinstance(strategy, BoundNativeStrategyPolicy):
            raise TypeError("NativeSimulator requires a bound strategy policy")
        native_config = native_simulator_config(
            data_config,
            feature_config,
            strategy.config,
            execution_config,
        )
        self.last_signals = None
        self.last_telemetry = None
        started = time.perf_counter()
        engine = RuleAwareDataLakeProductionBacktestEngine.from_prepared(
            prepared, intrabar, native_config
        )
        self.last_engine_init_seconds = time.perf_counter() - started
        started = time.perf_counter()
        trades = engine.run()
        self.last_simulation_seconds = time.perf_counter() - started

        # Passive observation only: use records produced by this exact run.
        skipped_signals = getattr(engine, "skipped_signals", ())
        self.last_signals = _signal_frame(prepared, trades, skipped_signals)
        # The canonical signals frame now owns rejected-decision provenance.
        # Do not carry the same potentially huge list into pandas/DuckDB report
        # serialization through legacy DataFrame metadata, and release the wide
        # engine records before the reporting stage starts.
        _release_canonicalized_rejection_metadata(trades)
        # Bayesian scoring is downstream-only. It uses only completed outcomes
        # available before each entry and therefore cannot alter this run's fills.
        # Narrow provenance-only test doubles intentionally do not masquerade as
        # completed results, while real production frames stay schema-strict.
        if _is_completed_trade_result(trades):
            trades = enrich_bayesian_trade_probabilities(trades)
        clear_rejections = getattr(skipped_signals, "clear", None)
        if clear_rejections is not None:
            clear_rejections()
        telemetry_rows = getattr(engine, "telemetry_rows", ())
        if telemetry_rows:
            self.last_telemetry = pd.DataFrame(telemetry_rows)
        return trades


class NativeStrategyPolicy:
    """Prepared-fact strategy policy supplied explicitly to the simulator boundary."""

    def bind(self, prepared, config):
        # Prepared facts are immutable; binding establishes explicit strategy
        # ownership without moving indicator or fill logic into this adapter.
        del prepared
        return BoundNativeStrategyPolicy(config)
