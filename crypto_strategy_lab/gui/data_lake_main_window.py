"""Data-Lake-native top-level GUI composition.

This class keeps the existing strategy/profile/report controls while replacing
CSV selection/validation with filename-free Binance Data Lake requests.  It is a
migration layer around the current large MainWindow; future GUI cleanup can split
those controls into smaller widgets without coupling them back to market files.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
import re
import time
import traceback

import pandas as pd
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QCheckBox, QFileDialog, QFormLayout, QLabel, QMessageBox

from crypto_strategy_lab.data import MarketDataStore
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.data.timing import normalize_binance_interval
from crypto_strategy_lab.data_lake_config import (
    DATA_LAKE_CONFIG_FIELDS,
    build_data_lake_backtest_config,
    normalize_data_lake_config,
)
from crypto_strategy_lab.gui.config_logic import default_gui_config
from crypto_strategy_lab.gui.data_lake_worker import DataLakeGuiBacktestWorker, DataLakeGuiRunSpec
from crypto_strategy_lab.gui.state_transition_main_window import MainWindow as StateTransitionMainWindow
from crypto_strategy_lab.output_manager import planned_run_dir
from crypto_strategy_lab.paths import CACHE_DIR, MARKET_DATA_ROOT


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def strategy_warmup_period(config) -> timedelta:
    """Conservative strategy-candle warm-up for features still inside the engine."""

    profile_rsi = max((p.rsi_period for p in config.strategy_profiles.values()), default=14)
    momentum_hours = max((p.momentum_lookback_hours for p in config.strategy_profiles.values()), default=24)
    bars = max(
        int(config.atr_period) + 5,
        int(config.adx_period) * 2 + 5,
        int(config.bb_period) + 5,
        int(config.mean_reversion_period) + 5,
        int(profile_rsi) + 5,
        int(config.sr_lookback_bars) + 5 if config.enable_support_resistance_analysis else 0,
        30,
    )
    bar_days = bars * int(config.strategy_timeframe_minutes) / 1440.0
    days = max(bar_days, momentum_hours / 24.0, 22.0)  # state-transition research also uses 20d context
    if config.market_regime_method == "ASSET_RETURN":
        days = max(days, float(config.bull_regime_lookback_days) + 2.0)
    return timedelta(days=math.ceil(days) + 2)


def _parse_gui_period(start_text: str, end_text: str, coverage_start, coverage_end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve GUI period to inclusive start / exclusive end in UTC."""

    start = _utc_timestamp(start_text) if start_text.strip() else _utc_timestamp(coverage_start)
    if end_text.strip():
        end = _utc_timestamp(end_text)
        if _DATE_ONLY.fullmatch(end_text.strip()):
            end += pd.Timedelta(days=1)
    else:
        end = _utc_timestamp(coverage_end)
    if start >= end:
        raise ValueError("Backtest start must be before end")
    return start, end


class MainWindow(StateTransitionMainWindow):
    """Existing full research GUI with Data Lake v2 as its market-data source."""

    def __init__(self, startup_status=None):
        self._data_lake_store: MarketDataStore | None = None
        self._data_lake_catalog_ready = False
        self._validated_data_lake_spec: DataLakeGuiRunSpec | None = None
        self._validated_data_lake_config = None
        super().__init__(startup_status=startup_status)
        self.market_data_folder = MARKET_DATA_ROOT
        self._decorate_data_lake_controls()
        self._sync_dataset_paths()

    def _decorate_data_lake_controls(self) -> None:
        self.shared_data_note.setText(f"Binance Data Lake: {self.market_data_folder}")
        self.data_help.setText(
            "Strategy, intrabar and structural-regime candles are selected automatically "
            "from the Binance archive catalog. Compact futures research is attached when "
            "available; aggTrades flow is optional because it is much heavier."
        )
        self.include_agg_trade_flow = QCheckBox("Include aggTrades trade-flow research (heavier)")
        self.include_agg_trade_flow.setChecked(False)
        self.include_agg_trade_flow.setToolTip(
            "Loads Binance aggregate-trade archives and adds causal buy/sell pressure columns. "
            "Leave off for normal fast backtests."
        )

        # QFormLayout labels are QLabel children of the same Data group box.
        parent = self.input_csv.parentWidget()
        if parent is not None:
            for label in parent.findChildren(QLabel):
                if label.text() == "Strategy CSV":
                    label.setText("Strategy Data")
                elif label.text() == "Intrabar CSV":
                    label.setText("Intrabar Data")
                elif label.text() == "Shared Data":
                    label.setText("Data Lake Root")
            layout = parent.layout()
            if isinstance(layout, QFormLayout):
                layout.addRow("Trade Flow Research", self.include_agg_trade_flow)
            elif layout is not None:
                layout.addWidget(self.include_agg_trade_flow)

    def _sync_dataset_paths(self, *_args) -> None:
        if not hasattr(self, "input_csv"):
            return
        self.input_csv.setText("Automatic from Binance Data Lake v2")
        self.input_csv.setEnabled(False)
        if self.use_intrabar.isChecked():
            self.intrabar_csv.setText("Automatic from Binance Data Lake v2")
        else:
            self.intrabar_csv.setText("Disabled")
        self.intrabar_csv.setEnabled(False)
        if hasattr(self, "dataset_info"):
            self.dataset_info.setText(
                "Select pair, timeframes and period, then Validate Data to inspect archive coverage."
            )
        if hasattr(self, "shared_data_note"):
            self.shared_data_note.setText(f"Binance Data Lake: {self.market_data_folder}")
        if hasattr(self, "planned_output"):
            self.update_planned_output()

    def browse_csv(self) -> None:
        QMessageBox.information(
            self,
            "Data Lake v2",
            "Strategy candles are selected automatically from the Binance Data Lake. "
            "There is no strategy CSV to browse for.",
        )

    def browse_intrabar_csv(self) -> None:
        QMessageBox.information(
            self,
            "Data Lake v2",
            "Intrabar candles are selected automatically from the Binance Data Lake. "
            "There is no intrabar CSV to browse for.",
        )

    def _data_lake_strategy_values(self) -> dict:
        values = super().values()
        return {key: values[key] for key in DATA_LAKE_CONFIG_FIELDS if key in values}

    def _build_data_lake_config(self):
        values = super().values()
        strategy_values = {key: values[key] for key in DATA_LAKE_CONFIG_FIELDS if key in values}
        config = build_data_lake_backtest_config(strategy_values)
        symbol = str(values.get("market_symbol", "BTCUSDT")).strip().upper().replace("/", "")
        output_dir = Path(values.get("output_dir") or "output")
        # These path fields remain on BacktestConfig only until the config split is
        # complete. They are inert in the Data Lake worker. The symbol-shaped
        # placeholder keeps existing output-folder naming human-readable.
        return replace(
            config,
            input_csv=Path(f"{symbol}_DATA_LAKE.csv"),
            intrabar_csv=None,
            output_dir=output_dir,
            structural_regime_benchmark_csv=None,
        )

    def _store(self) -> MarketDataStore:
        if self._data_lake_store is None:
            self._data_lake_store = MarketDataStore(MARKET_DATA_ROOT, CACHE_DIR)
        return self._data_lake_store

    def _coverage(self, store: MarketDataStore, symbol: str, interval: str):
        coverage = store.catalog.coverage(
            MARKET_DATA_ROOT,
            market=MarketKind.FUTURES_UM,
            dataset=DatasetKind.KLINES,
            symbol=symbol,
            interval=normalize_binance_interval(interval),
        )
        if coverage.archive_count <= 0 or coverage.first_period is None or coverage.last_period is None:
            raise ValueError(f"No Binance Data Lake kline coverage for {symbol} {interval}")
        return coverage

    def validate_data(self, *args, config=None, force_refresh: bool | None = None):
        """Validate catalog coverage and prepare the runtime Data Lake request."""

        try:
            if config is None:
                config = self._build_data_lake_config()
            store = self._store()
            clicked_manually = bool(args) and isinstance(args[0], bool)
            should_refresh = (
                force_refresh
                if force_refresh is not None
                else (clicked_manually or not self._data_lake_catalog_ready)
            )
            archive_count = None
            if should_refresh:
                archive_count = store.refresh_catalog()
                self._data_lake_catalog_ready = True

            symbol = self.market_symbol.currentText().strip().upper().replace("/", "")
            strategy_interval = normalize_binance_interval(
                f"{int(config.strategy_timeframe_minutes)}m"
            )
            intrabar_interval = (
                normalize_binance_interval(f"{int(config.intrabar_timeframe_minutes)}m")
                if config.use_intrabar_data
                else None
            )
            strategy_coverage = self._coverage(store, symbol, strategy_interval)
            trade_coverage_start = _utc_timestamp(strategy_coverage.first_period)
            trade_coverage_end = _utc_timestamp(strategy_coverage.last_period)

            intrabar_coverage = None
            if intrabar_interval:
                intrabar_coverage = self._coverage(store, symbol, intrabar_interval)
                trade_coverage_start = max(trade_coverage_start, _utc_timestamp(intrabar_coverage.first_period))
                trade_coverage_end = min(trade_coverage_end, _utc_timestamp(intrabar_coverage.last_period))
                if trade_coverage_start >= trade_coverage_end:
                    raise ValueError(
                        f"{symbol} {strategy_interval} and {intrabar_interval} coverage do not overlap"
                    )

            if self.entire_dataset.isChecked():
                trade_start = trade_coverage_start
                trade_end = trade_coverage_end
                normalized_config = replace(config, trading_start_date=None, trading_end_date=None)
            else:
                trade_start, trade_end = _parse_gui_period(
                    self.trading_start.text(),
                    self.trading_end.text(),
                    trade_coverage_start,
                    trade_coverage_end,
                )
                if trade_start < trade_coverage_start or trade_end > trade_coverage_end:
                    raise ValueError(
                        "Requested trading period is outside available Data Lake coverage. "
                        f"Available overlap: {trade_coverage_start} to {trade_coverage_end}"
                    )
                normalized_config = replace(
                    config,
                    trading_start_date=trade_start.isoformat(),
                    trading_end_date=(trade_end - pd.Timedelta(microseconds=1)).isoformat(),
                )

            request_start = max(
                _utc_timestamp(strategy_coverage.first_period),
                trade_start - strategy_warmup_period(normalized_config),
            )
            request_end = trade_end
            intrabar_start = trade_start if intrabar_interval else None
            include_agg_trade_flow = bool(self.include_agg_trade_flow.isChecked())

            self._validated_data_lake_spec = DataLakeGuiRunSpec(
                raw_root=MARKET_DATA_ROOT,
                cache_root=CACHE_DIR,
                symbol=symbol,
                start=request_start.to_pydatetime(),
                end=request_end.to_pydatetime(),
                intrabar_start=(intrabar_start.to_pydatetime() if intrabar_start is not None else None),
                refresh_catalog=False,
                include_agg_trade_flow=include_agg_trade_flow,
            )
            self._validated_data_lake_config = normalized_config
            self._validated_strategy_data = self._validated_data_lake_spec

            lines = [
                f"Pair: {symbol}",
                f"Strategy coverage ({strategy_interval}): {strategy_coverage.first_period} to {strategy_coverage.last_period}",
            ]
            if intrabar_coverage is not None:
                lines.append(
                    f"Intrabar coverage ({intrabar_interval}): {intrabar_coverage.first_period} to {intrabar_coverage.last_period}"
                )
            lines.extend(
                [
                    f"Trading period: {trade_start} to {trade_end} (end exclusive)",
                    f"Strategy load incl. warm-up: {request_start} to {request_end}",
                    f"AggTrades trade-flow research: {'enabled' if include_agg_trade_flow else 'disabled'}",
                ]
            )
            if include_agg_trade_flow:
                agg_coverage = store.catalog.coverage(
                    MARKET_DATA_ROOT,
                    market=MarketKind.FUTURES_UM,
                    dataset=DatasetKind.AGG_TRADES,
                    symbol=symbol,
                    interval=None,
                )
                if agg_coverage.archive_count:
                    lines.append(
                        f"AggTrades coverage: {agg_coverage.first_period} to {agg_coverage.last_period} "
                        f"({agg_coverage.archive_count:,} archives)"
                    )
                else:
                    lines.append("AggTrades coverage: none found; run will continue without trade-flow columns")
            if archive_count is not None:
                lines.append(f"Cataloged archives: {archive_count:,}")
            self.dataset_info.setText("\n".join(lines))
            self.append_log("Data Lake validation passed.")
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Data Lake Validation", str(exc))
            self.append_log(traceback.format_exc())
            self._validated_data_lake_spec = None
            self._validated_data_lake_config = None
            return False

    def update_planned_output(self) -> None:
        try:
            config = self._build_data_lake_config()
            self.planned_output.setText(str(planned_run_dir(config).resolve()))
        except Exception:
            self.planned_output.setText("Output run folder: unavailable until configuration is valid")

    def run_backtest(self) -> None:
        try:
            config = self._build_data_lake_config()
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Validation Problems", str(exc))
            return

        if not self.validate_data(config=config):
            return
        config = self._validated_data_lake_config
        spec = self._validated_data_lake_spec
        if config is None or spec is None:
            QMessageBox.warning(self, "Data Lake Validation", "No validated Data Lake request is available")
            return

        config = replace(config, output_run_dir=planned_run_dir(config))
        self.planned_output.setText(str(config.output_run_dir.resolve()))
        self._run_failed = False
        self._pending_ui_results = None
        self.output_dir = config.output_run_dir
        self.thread = QThread()
        self.worker = DataLakeGuiBacktestWorker(config, spec)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self._thread_finished)
        self.worker.status.connect(self.on_status)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.started = time.time()
        self._set_backtest_running(True)
        self.thread.start()

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Data Lake Strategy Configuration", "data_lake_strategy.json", "JSON (*.json)"
        )
        if not path:
            return
        values = normalize_data_lake_config(self._data_lake_strategy_values())
        Path(path).write_text(json.dumps(values, indent=2, default=str) + "\n", encoding="utf-8")

    def load_config(self) -> None:
        if self.thread and self.thread.isRunning():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Data Lake Strategy Configuration", "", "JSON (*.json)"
        )
        if not path:
            return
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        values = normalize_data_lake_config(raw)
        self.apply_values(values)

    def apply_values(self, values) -> None:
        # Data-Lake-native configs deliberately omit path fields. Convert them
        # only into the temporary UI bridge expected by the inherited controls.
        if isinstance(values, dict) and set(values).issubset(DATA_LAKE_CONFIG_FIELDS):
            normalized = normalize_data_lake_config(values)
            bridge = default_gui_config()
            bridge.update(normalized)
            bridge["input_csv"] = "Automatic from Binance Data Lake v2"
            bridge["intrabar_csv"] = ""
            bridge["output_dir"] = (
                self.output_folder.text() if hasattr(self, "output_folder") else "output"
            )
            bridge["structural_regime_benchmark_csv"] = None
            super().apply_values(bridge)
            self._sync_dataset_paths()
            return
        super().apply_values(values)
        self._sync_dataset_paths()
