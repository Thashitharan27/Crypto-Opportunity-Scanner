"""QThread worker that runs the existing backtesting pipeline."""
from __future__ import annotations

import json, time, traceback, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.loader import load_backtest_data
from crypto_strategy_lab.plots import save_plots
from crypto_strategy_lab.statistics import adx_analysis, bb_width_analysis, di_spread_analysis, di_pressure_analysis, mean_reversion_analysis, equity_curve, summarize
from crypto_strategy_lab.telemetry import add_journey_columns, stop_loss_journey_analysis, save_journey_charts, trade_journey_analysis, winner_loser_journey_analysis, trailing_profit_analysis, partial_take_profit_analysis
from crypto_strategy_lab.lifecycle import export_lifecycle_reports
from crypto_strategy_lab.output_manager import create_run_dir, periodic_results, update_latest, write_config, write_run_info, write_trade_column_metadata
from crypto_strategy_lab.report_workbooks import build_backtest_workbook, build_indicator_workbook, build_performance_breakdowns
from crypto_strategy_lab.support_resistance_analysis import generate_sr_analysis_reports
from dataclasses import replace
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS


def _pop_heavy_trade_attrs(trades: pd.DataFrame) -> dict[str, object]:
    """Remove export-only metadata that pandas would copy during analysis."""
    detached: dict[str, object] = {}
    for key in ("skipped_signals",):
        if key in trades.attrs:
            detached[key] = trades.attrs.pop(key)
    return detached

class BacktestWorker(QObject):
    status = Signal(str, int)
    log = Signal(str)
    finished = Signal(dict, object, object, object)
    failed = Signal(str, str)

    def __init__(self, config: BacktestConfig, strategy_data: pd.DataFrame | None = None):
        super().__init__(); self.config = config; self.strategy_data = strategy_data; self._cancel = False; self._started = 0.0; self._log_lines: list[str] = []; self._output_status = ("Preparing outputs", 95)

    @Slot()
    def cancel(self) -> None:
        self._cancel = True

    def _check(self):
        if self._cancel: raise RuntimeError("Backtest cancelled by user.")

    def _log(self, message: str) -> None:
        self._log_lines.append(str(message))
        self.log.emit(str(message))

    def _elapsed(self) -> float:
        return max(0.0, time.time() - self._started) if self._started else 0.0

    @staticmethod
    def _fmt_duration(seconds: float | None) -> str:
        if seconds is None or seconds == float("inf"):
            return "calculating"
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _emit_stage(self, stage: str, percent: int, processed: int = 0, total: int = 0, completed: int = 0, pair_total: int = 0, remaining: float | None = None) -> None:
        detail = (
            f"Stage: {stage} | Strategy candles: {processed:,} / {total:,} | "
            f"Completed pairs: {completed:,} / {pair_total:,} | "
            f"Elapsed: {self._fmt_duration(self._elapsed())} | ETA: {self._fmt_duration(remaining)}"
        )
        self.status.emit(detail, max(0, min(100, int(percent))))

    def _backtest_progress(self, processed: int, total: int, completed: int, opened: int) -> None:
        self._check()
        ratio = processed / total if total else 1.0
        percent = 20 + round(70 * ratio)
        elapsed = self._elapsed()
        remaining = (elapsed / ratio) - elapsed if ratio > 0 else None
        self._emit_stage("Backtesting", percent, processed, total, completed, opened, remaining)

    def _output_progress(self, detail: str, percent: int) -> None:
        self._output_status = (detail, percent)
        self._emit_stage(detail, percent)

    def _telemetry_progress(self, current: int, total: int) -> None:
        self._check()
        elapsed = max(0.0, time.monotonic() - self._telemetry_started)
        remaining = (elapsed / current) * (total - current) if current else None
        percent = 92 + (current / total * 2 if total else 2)
        detail = (
            f"Adding telemetry metrics: trade {current:,} / {total:,} | "
            f"Telemetry ETA: {self._fmt_duration(remaining)}"
        )
        self._output_progress(detail, int(percent))

    def _load_runtime_inputs(self):
        return load_backtest_data(self.config, self.strategy_data)

    def _runtime_period(self, data) -> tuple[object, object]:
        return data["timestamp"].min(), data["timestamp"].max()

    def _build_engine(self, data, config, intrabar, **kwargs):
        return BacktestEngine(data, config, intrabar, **kwargs)

    def _heartbeat(self, stop: threading.Event) -> None:
        """Keep Qt's queued status stream active during long pandas/matplotlib calls."""
        while not stop.wait(1.0):
            detail, percent = self._output_status
            self._emit_stage(detail, percent)

    def _run_isolated_only(self, data: pd.DataFrame, intrabar: pd.DataFrame | None) -> None:
        """Run enabled profiles independently without a shared-account run."""
        output_root=self.config.output_dir; run_dir=create_run_dir(self.config); write_config(self.config,run_dir)
        isolated_dir=run_dir/"isolated_profiles"; isolated_dir.mkdir(parents=True,exist_ok=True)
        comparison=[]; display_frames=[]
        enabled=[name for name in PROFILE_KEYS if self.config.strategy_profiles[name].enabled]
        for number,profile_name in enumerate(enabled,1):
            self._check(); self._emit_stage(f"Isolated profile {number}/{len(enabled)}: {profile_name.replace('_',' ').title()}",20+round(70*(number-1)/max(1,len(enabled))))
            self._log(f"Running isolated profile: {profile_name}")
            profiles={key:replace(value,enabled=(key==profile_name)) for key,value in self.config.strategy_profiles.items()}
            config=replace(self.config,strategy_profile_run_mode="COMBINED_SHARED_CAPITAL",strategy_profiles=profiles)
            profile_started=time.monotonic()
            profile_count=max(1,len(enabled))
            def isolated_progress(processed: int, total: int, completed: int, opened: int, *, _number=number, _name=profile_name, _started=profile_started) -> None:
                self._check()
                ratio=processed/total if total else 1.0
                percent=20+round(70*((_number-1)+ratio)/profile_count)
                elapsed=max(0.0,time.monotonic()-_started)
                remaining=(elapsed/ratio)-elapsed if ratio>0 else None
                self._emit_stage(
                    f"Isolated profile {_number}/{profile_count}: {_name.replace('_',' ').title()}",
                    percent,processed,total,completed,opened,remaining,
                )
            progress_interval=250 if config.strategy_timeframe_minutes<=1 else 50
            trades=self._build_engine(data,config,intrabar,progress_callback=isolated_progress,progress_interval=progress_interval).run()
            _pop_heavy_trade_attrs(trades)
            profile_dir=isolated_dir/profile_name; profile_dir.mkdir(parents=True,exist_ok=True)
            summary=summarize(trades,self.config.initial_equity); equity=equity_curve(trades,self.config.initial_equity)
            trades.to_csv(profile_dir/"trade_list.csv",index=False); equity.to_csv(profile_dir/"equity_curve.csv",index=False)
            (profile_dir/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
            comparison.append({"profile":profile_name,"trades":len(trades),"net_profit":summary.get("ending_equity",self.config.initial_equity)-self.config.initial_equity,"return_pct":summary.get("total_return_percentage",0.0),"win_rate":summary.get("win_rate",0.0),"profit_factor":summary.get("profit_factor",0.0),"max_drawdown_pct":summary.get("maximum_drawdown_percentage",0.0)})
            if not trades.empty:
                frame=trades.copy(); frame.insert(0,"isolated_profile",profile_name); display_frames.append(frame)
        all_trades=pd.concat(display_frames,ignore_index=True) if display_frames else pd.DataFrame()
        summary={"strategy_profile_run_mode":"ISOLATED_PROFILES","profiles_tested":len(comparison),"total_trades":sum(item["trades"] for item in comparison),"isolated_profile_comparison":comparison}
        pd.DataFrame(comparison).to_csv(run_dir/"strategy_profile_comparison.csv",index=False)
        all_trades.to_csv(run_dir/"isolated_trade_list.csv",index=False)
        (run_dir/"summary.json").write_text(json.dumps(summary,indent=2,default=str)); write_run_info(self.config,summary,run_dir)
        update_latest(output_root,run_dir); self._emit_stage("Outputs saved; preparing results view",99,len(data),len(data),len(all_trades),len(all_trades),0)
        self._log(f"Completed {len(comparison)} isolated profile runs with {len(all_trades):,} total trades")
        self._log(f"Results saved to {run_dir}"); (run_dir/"log.txt").write_text("\n".join(self._log_lines+["Isolated profile tests completed from GUI worker."])+"\n")
        self.finished.emit(summary,all_trades,pd.DataFrame(),run_dir)

    @Slot()
    def run(self) -> None:
        self._started = time.time()
        try:
            self._emit_stage("Loading data", 0)
            data, intrabar = self._load_runtime_inputs()
            self._emit_stage("Loading data", 10, 0, len(data))
            self._log(f"Loaded {len(data):,} strategy candles")
            if intrabar is not None: self._log(f"Loaded {len(intrabar):,} intrabar candles")
            period_start, period_end = self._runtime_period(data)
            self._log(f"Period: {period_start} to {period_end}")
            self._check(); self._emit_stage("ATR calculation", 10, 0, len(data))
            self._log(f"Running {self.config.risk_mode.value}, ATR({self.config.atr_period}), multiplier {self.config.atr_multiplier}")
            self._log(f"Intrabar config: use_intrabar_data={self.config.use_intrabar_data}, intrabar_csv={self.config.intrabar_csv}, intrabar_timeframe={self.config.intrabar_timeframe_minutes}m")
            if self.config.strategy_profile_run_mode=="ISOLATED_PROFILES":
                self._run_isolated_only(data,intrabar)
                return
            # Minute data contains many more candles. Less frequent GUI-only
            # updates reduce cross-thread overhead without changing simulation
            # resolution, ordering, trades, or any saved output.
            progress_interval = 250 if self.config.strategy_timeframe_minutes <= 1 else 50
            engine = self._build_engine(data, self.config, intrabar, progress_callback=self._backtest_progress, progress_interval=progress_interval)
            self._check(); self._emit_stage("ATR calculation", 20, 0, len(data))
            trades = engine.run()
            detached_trade_attrs = _pop_heavy_trade_attrs(trades)
            heartbeat_stop = threading.Event()
            initial_detail = f"Building telemetry table for {len(trades):,} completed trades" if self.config.enable_trade_telemetry else "Preparing core results"
            self._output_status = (initial_detail, 91)
            heartbeat = threading.Thread(target=self._heartbeat, args=(heartbeat_stop,), daemon=True)
            heartbeat.start()
            self._output_progress(*self._output_status)
            if self.config.enable_trade_telemetry:
                telemetry = engine.telemetry_frame()
                self._check()
                self._output_progress(f"Adding telemetry metrics to {len(trades):,} completed trades", 92)
                self._telemetry_started = time.monotonic()
                trades = add_journey_columns(trades, telemetry, progress=self._telemetry_progress)
            else:
                telemetry = pd.DataFrame()
            self._check(); self._output_progress("Calculating performance statistics", 94)
            # skipped_signals can contain tens of thousands of dictionaries. Pandas
            # deep-copies DataFrame.attrs during ordinary Series/frame operations,
            # although calculations and reports do not use that export-only payload.
            equity = equity_curve(trades, self.config.initial_equity)
            summary = summarize(trades, self.config.initial_equity)
            summary.update({"strategy_profile_run_mode": self.config.strategy_profile_run_mode, "use_intrabar_data": self.config.use_intrabar_data, "intrabar_csv": str(self.config.intrabar_csv) if self.config.intrabar_csv else None, "strategy_timeframe": self.config.strategy_timeframe_minutes, "intrabar_timeframe": self.config.intrabar_timeframe_minutes, "atr_period": self.config.atr_period, "atr_multiplier": self.config.atr_multiplier})
            if self.config.use_intrabar_data and summary.get("intrabar_exit_count") == 0:
                self._log("WARNING: use_intrabar_data=True but 1M_INTRABAR exit count is 0. Check intrabar path, overlap, and timestamp alignment.")
            self._check(); self._output_progress("preparing trade list", 95)
            output_root = self.config.output_dir
            run_dir = create_run_dir(self.config)
            write_config(self.config, run_dir)
            output_failures = []

            output_timings = []
            def run_output_step(label, action, percent=98):
                self._output_progress(label, percent)
                self._log(f"{label}...")
                started = time.perf_counter()
                try:
                    return action()
                except Exception as exc:  # noqa: BLE001 - output exports must continue independently.
                    tb = traceback.format_exc()
                    output_failures.append({"step": label, "error": str(exc), "traceback": tb})
                    self._log(f"ERROR while {label}: {exc}")
                    self._log(tb.rstrip())
                    return None
                finally:
                    elapsed = time.perf_counter() - started
                    output_timings.append((label, elapsed))
                    self._log(f"Output report timing: {label}: {elapsed:.3f}s")

            if self.config.strategy_profile_run_mode == "BOTH":
                self._output_progress("Running six isolated strategy profiles", 95)
                isolated_dir=run_dir/"isolated_profiles"; isolated_dir.mkdir(parents=True,exist_ok=True)
                comparison=[]; isolated_frames=[]
                combined_times=set(pd.to_datetime(trades.get("strategy_entry_time",pd.Series(dtype=object)),utc=True,errors="coerce").dropna())
                for profile_name in PROFILE_KEYS:
                    profile=self.config.strategy_profiles[profile_name]
                    if not profile.enabled: continue
                    self._check(); self._log(f"Running isolated profile: {profile_name}")
                    isolated_profiles={key:replace(value,enabled=(key==profile_name)) for key,value in self.config.strategy_profiles.items()}
                    isolated_config=replace(self.config,strategy_profile_run_mode="COMBINED_SHARED_CAPITAL",strategy_profiles=isolated_profiles)
                    isolated_trades=self._build_engine(data,isolated_config,intrabar).run()
                    isolated_trades.to_csv(isolated_dir/f"{profile_name}_trade_list.csv",index=False)
                    isolated_summary=summarize(isolated_trades,self.config.initial_equity)
                    comparison.append({"profile":profile_name,"trades":len(isolated_trades),"net_profit":isolated_summary.get("ending_equity",self.config.initial_equity)-self.config.initial_equity,"return_pct":isolated_summary.get("total_return_percentage",0.0),"win_rate":isolated_summary.get("win_rate",0.0),"profit_factor":isolated_summary.get("profit_factor",0.0),"max_drawdown_pct":isolated_summary.get("maximum_drawdown_percentage",0.0)})
                    if not isolated_trades.empty:
                        candidates=isolated_trades.copy(); candidates["profile"]=profile_name
                        candidates["blocked_in_combined"]=~pd.to_datetime(candidates["strategy_entry_time"],utc=True,errors="coerce").isin(combined_times)
                        isolated_frames.append(candidates[candidates["blocked_in_combined"]])
                pd.DataFrame(comparison).to_csv(run_dir/"strategy_profile_comparison.csv",index=False)
                blocked=pd.concat(isolated_frames,ignore_index=True) if isolated_frames else pd.DataFrame()
                blocked.to_csv(run_dir/"blocked_profile_opportunities.csv",index=False)
                summary["isolated_profile_comparison"]=comparison
                summary["blocked_profile_opportunities"]=len(blocked)

            run_output_step("writing trade_list.csv", lambda: trades.to_csv(run_dir / "trade_list.csv", index=False), 96)
            if self.config.save_feature_analysis_reports:
                run_output_step("writing trailing_profit_analysis.csv", lambda: trailing_profit_analysis(trades).to_csv(run_dir / "trailing_profit_analysis.csv", index=False))
                run_output_step("writing partial_take_profit_analysis.csv", lambda: partial_take_profit_analysis(trades).to_csv(run_dir / "partial_take_profit_analysis.csv", index=False))
            if self.config.enable_trade_telemetry:
                self._output_progress("preparing telemetry", 96)
                if self.config.save_full_telemetry_csv:
                    run_output_step("writing trade_telemetry.csv", lambda: telemetry.to_csv(run_dir / "trade_telemetry.csv", index=False), 97)
                if self.config.save_trade_journey_summary:
                    run_output_step("writing trade_journey_analysis.csv", lambda: trade_journey_analysis(trades).to_csv(run_dir / "trade_journey_analysis.csv", index=False))
                    run_output_step("writing winner_loser_journey_analysis.csv", lambda: winner_loser_journey_analysis(trades).to_csv(run_dir / "winner_loser_journey_analysis.csv", index=False))
                    run_output_step("writing stop_loss_journey_analysis.csv", lambda: stop_loss_journey_analysis(trades, telemetry).to_csv(run_dir / "stop_loss_journey_analysis.csv", index=False))
                if self.config.enable_indicator_lifecycle_analysis:
                    def lifecycle_progress(label, current, total):
                        detail = f"{label} trade {current:,} of {total:,}" if total else label
                        now = time.monotonic()
                        self._output_status = (detail, 97 if total and current < total else 98)
                        if current in (1, total) or now - getattr(self, "_last_lifecycle_update", 0.0) >= 0.25:
                            self._last_lifecycle_update = now
                            self._output_progress(*self._output_status)
                        self._check()
                    run_output_step("lifecycle analysis", lambda: export_lifecycle_reports(trades, telemetry, run_dir, phases=self.config.lifecycle_phases, checkpoints=self.config.lifecycle_early_checkpoints, minimum_sample=self.config.lifecycle_minimum_bucket_sample, charts=self.config.create_lifecycle_charts, flat_threshold_pct=self.config.lifecycle_flat_pattern_threshold_pct, progress=lifecycle_progress), 97)
            skipped_signals = detached_trade_attrs.get("skipped_signals", [])
            run_output_step("writing skipped_signals.csv", lambda: pd.DataFrame(skipped_signals).to_csv(run_dir / "skipped_signals.csv", index=False))
            run_output_step("writing skipped_daily_entries.csv", lambda: pd.DataFrame(trades.attrs.get("skipped_daily_entries", [])).to_csv(run_dir / "skipped_daily_entries.csv", index=False))
            monthly, yearly = periodic_results(trades, "ME"), periodic_results(trades, "YE")
            market_regime, direction_regime = build_performance_breakdowns(trades)
            parallel_reports = [
                ("Saving trade column metadata", lambda: write_trade_column_metadata(run_dir), 98),
                ("writing support/resistance analysis reports", lambda: generate_sr_analysis_reports(trades, run_dir), 98),
                ("writing equity_curve.csv", lambda: equity.to_csv(run_dir / "equity_curve.csv", index=False), 98),
                ("writing backtest_report.xlsx", lambda: build_backtest_workbook(summary, self.config, run_dir, monthly, yearly, market_regime, direction_regime), 98),
            ]
            if self.config.save_indicator_analysis_reports:
                parallel_reports.extend([
                    ("writing indicator_analysis.xlsx", lambda: build_indicator_workbook({"ADX": adx_analysis(trades), "BB Width": bb_width_analysis(trades), "DI Spread": di_spread_analysis(trades), "DI Pressure": di_pressure_analysis(trades), "Mean Reversion": mean_reversion_analysis(trades)}, run_dir), 98),
                    ("writing di_mean_reversion_analysis.csv", lambda: mean_reversion_analysis(trades).to_csv(run_dir / "di_mean_reversion_analysis.csv", index=False), 98),
                ])
            if self.config.create_standard_charts:
                parallel_reports.append(("creating charts", lambda: save_plots(trades, equity, run_dir / "charts"), 99))
            report_results = {}
            self._output_progress(f"creating {len(parallel_reports)} reports in parallel", 98)
            with ThreadPoolExecutor(max_workers=min(4, len(parallel_reports)), thread_name_prefix="report") as pool:
                pending = {pool.submit(run_output_step, label, action, percent): label for label, action, percent in parallel_reports}
                for future in as_completed(pending):
                    self._check()
                    report_results[pending[future]] = future.result()
            chart_warnings = report_results.get("creating charts") or []
            if self.config.enable_trade_telemetry and self.config.save_trade_journey_charts:
                chart_warnings.extend(run_output_step("creating journey charts", lambda: save_journey_charts(trades, telemetry, run_dir / "charts")) or [])
            for warning in chart_warnings:
              self.log.emit(f"WARNING: {warning}")
            if output_failures:
                summary["failed_output_reports"] = output_failures
                self._log("WARNING: Some output reports failed: " + ", ".join(f["step"] for f in output_failures))
            run_output_step("writing summary.json", lambda: (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str)))
            run_output_step("Saving run info", lambda: write_run_info(self.config, summary, run_dir))
            update_latest(output_root, run_dir)
            heartbeat_stop.set(); heartbeat.join(timeout=2)
            # The main window still has to install the summary and trade model.
            # Keep the bar below 100 until that GUI-side work is complete so the
            # user is never shown a finished state while Qt is still finalizing.
            self._emit_stage("Outputs saved; preparing results view", 99, len(data), len(data), len(trades), len(trades), 0)
            if output_timings:
                slowest_label, slowest_elapsed = max(output_timings, key=lambda item: item[1])
                self._log(f"Slowest output report: {slowest_label} ({slowest_elapsed:.3f}s)")
            self._log(f"Completed {len(trades):,} trade pairs")
            self._log(f"Results saved to {run_dir}")
            (run_dir / "log.txt").write_text("\n".join(self._log_lines + ["Backtest completed from GUI worker."]) + "\n")
            trades.attrs.update(detached_trade_attrs)
            self.finished.emit(summary, trades, equity, run_dir)
        except Exception as exc:
            if "heartbeat_stop" in locals(): heartbeat_stop.set()
            self.failed.emit(str(exc), traceback.format_exc())
