"""Production composition for the PAPER-only opportunity scanner."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.data.backtest_service import load_backtest_bundle
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.data.store import MarketDataStore
from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    OpportunityScanRequest,
    create_opportunity_scanner_service,
)
from crypto_strategy_lab.paper_scanner import (
    LatestNativeStrategyEvaluator,
    PaperScannerConfig,
    PaperScannerRunner,
)
from crypto_strategy_lab.prepared_backtest import from_data_lake_bundle
from crypto_strategy_lab.rule_native_engine import (
    RuleAwareDataLakeProductionBacktestEngine,
)


def create_production_paper_scanner(
    *,
    market_data_root: Path,
    cache_root: Path,
    scan_output_root: Path,
    paper_config: PaperScannerConfig,
    scan_request_factory: Callable[[], OpportunityScanRequest],
    strategy_config: BacktestConfig,
    scanner_pipeline_options: Mapping[str, Any] | None = None,
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> PaperScannerRunner:
    """Compose the real Task-9 scanner, Data Lake, and native PAPER evaluator.

    The caller supplies configuration and paths, never an engine builder.  This
    boundary contains no broker, authenticated API, order client, or execution
    mode; the native engine is inspected only through its non-mutating entry
    decision API.
    """
    raw_root, cache = Path(market_data_root), Path(cache_root)
    scanner = create_opportunity_scanner_service(
        raw_root,
        cache,
        Path(scan_output_root),
        **dict(scanner_pipeline_options or {}),
    )
    store = MarketDataStore(raw_root, cache)

    def engine_builder(candidate: dict[str, Any]):
        request = _strategy_request(candidate)
        bundle = load_backtest_bundle(
            store,
            request,
            market_regime_method=strategy_config.market_regime_method,
            structural_regime_sma_days=strategy_config.structural_regime_sma_days,
            structural_regime_slope_lookback_days=(
                strategy_config.structural_regime_slope_lookback_days
            ),
            atr_period=strategy_config.atr_period,
            adx_period=strategy_config.adx_period,
            di_pressure_lookback=strategy_config.di_pressure_lookback,
            bb_period=strategy_config.bb_period,
            bb_stddevs=strategy_config.bb_stddevs,
            mean_reversion_period=strategy_config.mean_reversion_period,
            enable_support_resistance_analysis=(
                strategy_config.enable_support_resistance_analysis
            ),
            sr_pivot_left=strategy_config.sr_pivot_left,
            sr_pivot_right=strategy_config.sr_pivot_right,
            sr_lookback_bars=strategy_config.sr_lookback_bars,
            sr_zone_width_atr=strategy_config.sr_zone_width_atr,
            sr_near_distance_atr=strategy_config.sr_near_distance_atr,
            enable_sr_hold_confirmation=strategy_config.enable_sr_hold_confirmation,
            sr_hold_confirmation_bars=strategy_config.sr_hold_confirmation_bars,
            sr_hold_confirmation_atr=strategy_config.sr_hold_confirmation_atr,
            sr_break_tolerance_atr=strategy_config.sr_break_tolerance_atr,
            sr_break_basis=strategy_config.sr_break_basis,
        )
        prepared, intrabar = from_data_lake_bundle(bundle, strategy_config)
        engine = RuleAwareDataLakeProductionBacktestEngine.from_prepared(
            prepared, intrabar, strategy_config
        )
        return prepared, engine

    return PaperScannerRunner(
        paper_config,
        scanner,
        scan_request_factory,
        LatestNativeStrategyEvaluator(engine_builder),
        clock=clock,
        sleeper=sleeper,
    )


def _strategy_request(candidate: dict[str, Any]) -> DataRequest:
    """Recover the canonical Task-5 request flattened into Task-7 publication."""
    return DataRequest(
        symbol=str(candidate["symbol"]),
        start=_timestamp(candidate["strategy_request_start"]),
        end=_timestamp(candidate["strategy_request_end"]),
        strategy_interval=str(candidate["strategy_interval"]),
        market=MarketKind(str(candidate["strategy_request_market"])),
        exchange=str(candidate.get("strategy_request_exchange", "binance")),
        datasets=(DatasetKind.KLINES,),
    )


def _timestamp(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("published strategy request timestamp must be timezone-aware")
    return timestamp.to_pydatetime()
