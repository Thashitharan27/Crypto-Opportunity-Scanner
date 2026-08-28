"""GUI worker adapter for Data Lake v2 market data and prepared features.

The mature reporting/output pipeline remains in place while market data and
stateless context are prepared once through the Data Lake before simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import traceback

from PySide6.QtCore import Slot

from crypto_strategy_lab.data import DataRequest, MarketDataStore
from crypto_strategy_lab.data.backtest_service import BacktestDataBundle, load_backtest_bundle
from crypto_strategy_lab.data_lake_production_engine import DataLakeProductionBacktestEngine
from crypto_strategy_lab.state_transition_prepared_reports import generate_prepared_state_transition_reports
from crypto_strategy_lab.prepared_cache import prepare_bundle_with_cache
from crypto_strategy_lab.gui.worker import BacktestWorker


@dataclass(frozen=True, slots=True)
class DataLakeGuiRunSpec:
    """Runtime-only market-data selection passed from the GUI to the worker."""

    raw_root: Path
    cache_root: Path
    symbol: str
    start: datetime
    end: datetime
    intrabar_start: datetime | None = None
    refresh_catalog: bool = False
    include_agg_trade_flow: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_root", Path(self.raw_root))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not self.symbol:
            raise ValueError("Data Lake GUI symbol must not be empty")
        if self.start >= self.end:
            raise ValueError("Data Lake GUI start must be before end")


class DataLakeGuiBacktestWorker(BacktestWorker):
    """Existing GUI/report worker with Data Lake market data and feature inputs."""

    def __init__(self, config, run_spec: DataLakeGuiRunSpec):
        super().__init__(config, strategy_data=None)
        self.run_spec = run_spec
        self.data_bundle: BacktestDataBundle | None = None
        self._prepared_inputs = None
        self.finished.connect(self._write_data_lake_metadata_and_research)

    def _prepare_bundle(self) -> BacktestDataBundle:
        strategy_interval = f"{int(self.config.strategy_timeframe_minutes)}m"
        intrabar_interval = (
            f"{int(self.config.intrabar_timeframe_minutes)}m"
            if self.config.use_intrabar_data else None
        )
        request = DataRequest(
            symbol=self.run_spec.symbol,
            start=self.run_spec.start,
            end=self.run_spec.end,
            strategy_interval=strategy_interval,
            intrabar_interval=intrabar_interval,
        )
        store = MarketDataStore(self.run_spec.raw_root, self.run_spec.cache_root)
        return load_backtest_bundle(
            store,
            request,
            market_regime_method=self.config.market_regime_method,
            structural_regime_sma_days=self.config.structural_regime_sma_days,
            structural_regime_slope_lookback_days=self.config.structural_regime_slope_lookback_days,
            refresh_catalog=self.run_spec.refresh_catalog,
            intrabar_start=self.run_spec.intrabar_start,
            atr_period=self.config.atr_period,
            adx_period=self.config.adx_period,
            di_pressure_lookback=self.config.di_pressure_lookback,
            bb_period=self.config.bb_period,
            bb_stddevs=self.config.bb_stddevs,
            mean_reversion_period=self.config.mean_reversion_period,
            mean_reversion_mean_type=getattr(self.config, "mean_reversion_mean_type", "SMA"),
            mean_reversion_bb_stddevs=getattr(self.config, "mean_reversion_bb_stddevs", 2.0),
            mean_reversion_rsi_period=getattr(self.config, "mean_reversion_rsi_period", 14),
            mean_reversion_rsi_oversold=getattr(self.config, "mean_reversion_rsi_oversold", 30.0),
            mean_reversion_rsi_overbought=getattr(self.config, "mean_reversion_rsi_overbought", 70.0),
            mean_reversion_require_reentry=getattr(self.config, "mean_reversion_require_reentry", True),
            enable_support_resistance_analysis=self.config.enable_support_resistance_analysis,
            sr_timeframe_minutes=int(getattr(self.config, "sr_timeframe_minutes", 0) or 0),
            sr_pivot_left=self.config.sr_pivot_left,
            sr_pivot_right=self.config.sr_pivot_right,
            sr_lookback_bars=self.config.sr_lookback_bars,
            sr_zone_width_atr=self.config.sr_zone_width_atr,
            sr_near_distance_atr=self.config.sr_near_distance_atr,
            enable_sr_hold_confirmation=self.config.enable_sr_hold_confirmation,
            sr_hold_confirmation_bars=self.config.sr_hold_confirmation_bars,
            sr_hold_confirmation_atr=self.config.sr_hold_confirmation_atr,
            sr_break_tolerance_atr=self.config.sr_break_tolerance_atr,
            sr_break_basis=self.config.sr_break_basis,
            trade_flow_enabled=self.run_spec.include_agg_trade_flow,
        )

    @Slot()
    def run(self) -> None:
        try:
            self._emit_stage("Loading Binance Data Lake", 0)
            self.data_bundle = self._prepare_bundle()
            bundle = self.data_bundle
            directional = bundle.technical_features
            context = bundle.context_features
            sr = bundle.support_resistance_features
            self._log(
                f"Data Lake source: {self.run_spec.symbol} | "
                f"strategy={bundle.request.strategy_interval} ({len(bundle.strategy):,} rows) | "
                f"intrabar={bundle.request.intrabar_interval or 'disabled'} "
                f"({len(bundle.intrabar) if bundle.intrabar is not None else 0:,} rows)"
            )
            self._log(
                f"Prepared features: {directional.attrs.get('feature_name')}@"
                f"{directional.attrs.get('feature_version')} "
                f"(cache={'hit' if directional.attrs.get('feature_cache_hit') else 'miss'}), "
                f"{context.attrs.get('feature_name')}@{context.attrs.get('feature_version')} "
                f"(cache={'hit' if context.attrs.get('feature_cache_hit') else 'miss'})"
            )
            if sr is not None:
                self._log(
                    f"Prepared S/R: {sr.attrs.get('feature_name')}@{sr.attrs.get('feature_version')} "
                    f"(cache={'hit' if sr.attrs.get('feature_cache_hit') else 'miss'})"
                )
            elif self.config.enable_support_resistance_analysis:
                self._log("WARNING: S/R enabled but no prepared S/R context was produced")
            if bundle.research_features:
                labels = [
                    f"{name}@{frame.attrs.get('feature_version')} "
                    f"(cache={'hit' if frame.attrs.get('feature_cache_hit') else 'miss'})"
                    for name, frame in sorted(bundle.research_features.items())
                ]
                self._log("Futures research: " + ", ".join(labels))
            else:
                self._log("Futures research: no local compact/reference coverage for this request")
            if self.run_spec.include_agg_trade_flow and "trade_flow_context" not in bundle.research_features:
                self._log("AggTrades research requested but no local aggTrades coverage was found")
            if bundle.structural_benchmark is not None:
                self._log(
                    f"Structural benchmark: {bundle.structural_benchmark_symbol} "
                    f"{bundle.structural_benchmark_interval} "
                    f"({len(bundle.structural_benchmark):,} rows)"
                )

            # Keep L3 validation/materialization in this worker's failure path.
            # If it raises, the GUI must receive failed() so its thread/running
            # state is cleaned up instead of remaining stuck.
            self._prepared_inputs = prepare_bundle_with_cache(
                self.run_spec.cache_root, bundle, self.config
            )[:2]
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
            return

        super().run()

    def _load_runtime_inputs(self):
        if self._prepared_inputs is None:
            raise RuntimeError("Data Lake inputs were not prepared")
        return self._prepared_inputs

    def _runtime_period(self, data):
        return data.timestamp[0], data.timestamp[-1]

    def _build_engine(self, data, config, intrabar, **kwargs):
        return DataLakeProductionBacktestEngine.from_prepared(
            data, intrabar, config, **kwargs
        )

    @staticmethod
    def _feature_manifest(frame):
        if frame is None:
            return None
        return {
            "name": frame.attrs.get("feature_name"),
            "version": frame.attrs.get("feature_version"),
            "rows": len(frame),
            "cache_hit": bool(frame.attrs.get("feature_cache_hit", False)),
            "cache_key": frame.attrs.get("feature_cache_key"),
            "effective_warmup_bars": frame.attrs.get("effective_warmup_bars"),
        }

    @Slot(dict, object, object, object)
    def _write_data_lake_metadata_and_research(self, summary, trades, _equity, run_dir) -> None:
        if self.data_bundle is None:
            return
        run_dir = Path(run_dir)
        bundle = self.data_bundle
        request = bundle.request
        directional = bundle.technical_features
        context = bundle.context_features
        sr = bundle.support_resistance_features
        daily_state = bundle.state_transition_daily_features
        try:
            summary.update(
                {
                    "data_source": "binance_data_lake_v2",
                    "market_symbol": request.symbol,
                    "strategy_interval": request.strategy_interval,
                    "intrabar_interval": request.intrabar_interval,
                    "data_request_start": request.start.isoformat(),
                    "data_request_end": request.end.isoformat(),
                    "technical_feature_name": directional.attrs.get("feature_name"),
                    "technical_feature_version": directional.attrs.get("feature_version"),
                    "context_feature_name": context.attrs.get("feature_name"),
                    "context_feature_version": context.attrs.get("feature_version"),
                    "support_resistance_feature_name": sr.attrs.get("feature_name") if sr is not None else None,
                    "support_resistance_feature_version": sr.attrs.get("feature_version") if sr is not None else None,
                    "state_transition_daily_feature_version": daily_state.attrs.get("feature_version") if daily_state is not None else None,
                    "research_feature_names": sorted(bundle.research_features),
                    "include_agg_trade_flow": bool(self.run_spec.include_agg_trade_flow),
                }
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            directional_manifest = self._feature_manifest(directional)
            directional_manifest.update(
                atr_period=directional.attrs.get("atr_period"),
                adx_period=directional.attrs.get("adx_period"),
                di_pressure_lookback=directional.attrs.get("di_pressure_lookback"),
            )
            context_manifest = self._feature_manifest(context)
            context_manifest.update(
                bb_period=context.attrs.get("bb_period"),
                bb_stddevs=context.attrs.get("bb_stddevs"),
                mean_reversion_period=context.attrs.get("mean_reversion_period"),
                mean_reversion_mean_type=context.attrs.get("mean_reversion_mean_type"),
                mean_reversion_bb_stddevs=context.attrs.get("mean_reversion_bb_stddevs"),
                mean_reversion_rsi_period=context.attrs.get("mean_reversion_rsi_period"),
                mean_reversion_rsi_oversold=context.attrs.get("mean_reversion_rsi_oversold"),
                mean_reversion_rsi_overbought=context.attrs.get("mean_reversion_rsi_overbought"),
                mean_reversion_require_reentry=context.attrs.get("mean_reversion_require_reentry"),
            )
            manifest = {
                "data_source": "binance_data_lake_v2",
                "raw_root": str(self.run_spec.raw_root.resolve()),
                "cache_root": str(self.run_spec.cache_root.resolve()),
                "request": {
                    "symbol": request.symbol,
                    "start": request.start.isoformat(),
                    "end": request.end.isoformat(),
                    "strategy_interval": request.strategy_interval,
                    "intrabar_interval": request.intrabar_interval,
                    "intrabar_start": (
                        self.run_spec.intrabar_start.isoformat()
                        if self.run_spec.intrabar_start is not None else None
                    ),
                },
                "research_options": {
                    "include_agg_trade_flow": bool(self.run_spec.include_agg_trade_flow),
                },
                "strategy_rows": len(bundle.strategy),
                "intrabar_rows": len(bundle.intrabar) if bundle.intrabar is not None else 0,
                "features": {
                    "core_directional": directional_manifest,
                    "production_market_context": context_manifest,
                    "support_resistance": self._feature_manifest(sr),
                    "state_transition_daily": self._feature_manifest(daily_state),
                    "research": {
                        name: self._feature_manifest(frame)
                        for name, frame in sorted(bundle.research_features.items())
                    },
                },
                "production_engine": "DataLakeProductionBacktestEngine",
                "structural_benchmark_symbol": bundle.structural_benchmark_symbol,
                "structural_benchmark_interval": bundle.structural_benchmark_interval,
                "benchmark_rows": (
                    len(bundle.structural_benchmark)
                    if bundle.structural_benchmark is not None else 0
                ),
                "trade_rows": len(trades),
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            if daily_state is None:
                raise ValueError("Data Lake state-transition reports require prepared daily features")
            generate_prepared_state_transition_reports(daily_state, trades, run_dir)
            self._log("Data Lake manifest and state-transition research reports saved")
        except Exception as exc:
            self._log(f"WARNING: Data Lake post-run metadata/research failed: {exc}")
