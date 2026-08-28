from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    return widgets.QApplication.instance() or widgets.QApplication([]), widgets


class _CompletedRuns:
    def __init__(self, manifest, summary):
        self.manifest = manifest
        self.summary = summary

    def read(self, _run_dir):
        return self.manifest, self.summary


def _window(tmp_path: Path):
    _qt, widgets = _app()
    run_dir = tmp_path / "BTCUSDT_15m_test"
    run_dir.mkdir()
    (run_dir / "data_quality.json").write_text(
        json.dumps({"status": "OK"}), encoding="utf-8"
    )
    (run_dir / "trade_list.csv").write_text(
        "pair_id,pair_funding_net_pnl\n1,-3.25\n2,1.00\n",
        encoding="utf-8",
    )
    artifacts = {
        "workbook": {"path": "backtest_report.xlsx"},
        "trade_csv": {"path": "trade_list.csv"},
        "summary": {"path": "summary.json"},
        "trades": {"path": "artifacts/trades.parquet"},
        "signals": {"path": "artifacts/signals.parquet"},
        "feature_context": {"path": "artifacts/feature_context.parquet"},
        "data_quality": {"path": "data_quality.json"},
        "source_archives": {"path": "provenance/source_archives.parquet"},
    }
    manifest = {
        "run_id": "20260824T090000Z-test",
        "request": {
            "symbol": "BTCUSDT",
            "start": "2021-01-01T00:00:00+00:00",
            "end": "2022-01-01T00:00:00+00:00",
            "requested_strategy_interval": "15m",
        },
        "artifacts": artifacts,
        "execution_result": {
            "stage_timings": {
                "data_features": 18.3139929,
                "engine_init": 0.0243927,
                "prepared_cache": 4.8673839,
                "reporting": 3.1795287,
                "simulation": 3.7319515,
                "strategy_simulation_total": 513.8310656,
            }
        },
    }
    summary = {
        "ending_equity": 1667.065916840636,
        "net_pnl": 667.065916840636,
        "total_return_percentage": 66.7065916840637,
        "total_trades": 383,
        "wins": 235,
        "losses": 148,
        "win_rate": 0.6135770234986945,
        "total_net_r": 52.97971555250051,
        "average_net_r": 0.1383282390404713,
        "profit_factor": 1.278028705431582,
        "maximum_drawdown_percentage": -11.894710564707083,
        "total_fees": 336.2731341706942,
    }

    class Window:
        def __init__(self):
            self._manifest = manifest
            self._run_dir = run_dir
            self.service = SimpleNamespace(
                completed_runs=_CompletedRuns(manifest, summary)
            )
            self.open_folder = widgets.QPushButton("legacy output folder")
            self.opened = []

        def open_artifact(self, key):
            self.opened.append(key)

    return Window()


def test_dashboard_formats_completed_run_for_humans(tmp_path):
    _qt, _widgets = _app()
    from crypto_strategy_lab.gui.results_dashboard_workspace import ResultsDashboardWorkspace

    dashboard = ResultsDashboardWorkspace(_window(tmp_path))
    dashboard.refresh_completed_run()

    assert dashboard.run_title.text() == "BTCUSDT · 15 Minutes"
    assert "2021-01-01 → 2022-01-01" in dashboard.run_context.text()
    assert dashboard.quality_status.text() == "Data Quality  PASS"
    assert dashboard.metric_values["trades"].text() == "383\n235 W · 148 L"
    assert dashboard.metric_values["win_rate"].text() == "61.4%"
    assert dashboard.metric_values["net_r"].text() == "+53.0R"
    assert dashboard.metric_values["avg_r"].text() == "+0.138R"
    assert dashboard.metric_values["net_pnl"].text() == "+$667.07"
    assert dashboard.metric_values["return"].text() == "+66.7%"
    assert dashboard.metric_values["ending_equity"].text() == "$1,667.07"
    assert dashboard.metric_values["profit_factor"].text() == "1.28"
    assert dashboard.drawdown_value.text() == "-11.9%"
    assert dashboard.fees_value.text() == "$336.27"
    assert dashboard.funding_value.text() == "-$2.25"


def test_dashboard_prefers_summary_funding_total_when_present(tmp_path):
    _qt, _widgets = _app()
    from crypto_strategy_lab.gui.results_dashboard_workspace import ResultsDashboardWorkspace

    window = _window(tmp_path)
    window.service.completed_runs.summary["total_funding_net_pnl"] = 4.5
    dashboard = ResultsDashboardWorkspace(window)
    dashboard.refresh_completed_run()

    assert dashboard.funding_value.text() == "+$4.50"


def test_dashboard_marks_pre_funding_runs_as_unavailable(tmp_path):
    _qt, _widgets = _app()
    from crypto_strategy_lab.gui.results_dashboard_workspace import ResultsDashboardWorkspace

    window = _window(tmp_path)
    (window._run_dir / "trade_list.csv").write_text(
        "pair_id,pair_net_pnl\n1,2.00\n",
        encoding="utf-8",
    )
    dashboard = ResultsDashboardWorkspace(window)
    dashboard.refresh_completed_run()

    assert dashboard.funding_value.text() == "—"


def test_dashboard_has_clean_artifact_surface_and_compact_timings(tmp_path):
    _qt, _widgets = _app()
    from crypto_strategy_lab.gui.results_dashboard_workspace import ResultsDashboardWorkspace

    dashboard = ResultsDashboardWorkspace(_window(tmp_path))
    dashboard.refresh_completed_run()

    assert "telemetry" not in dashboard.artifact_buttons
    assert dashboard.artifact_buttons["source_archives"].isEnabled()
    assert not dashboard.research_content.isVisible()
    assert not dashboard.performance_content.isVisible()
    assert dashboard.timing_values["data_features"].text() == "18.3s"
    assert dashboard.timing_values["simulation"].text() == "3.7s"
    assert dashboard.timing_values["strategy_simulation_total"].text() == "8m 34s"

    dashboard.research_toggle.setChecked(True)
    dashboard.performance_toggle.setChecked(True)
    assert dashboard.research_toggle.text() == "Research Artifacts ▾"
    assert dashboard.performance_toggle.text() == "Run Performance ▾"


def test_profit_factor_infinity_is_displayed_not_zero():
    from crypto_strategy_lab.gui.results_dashboard_workspace import _profit_factor

    assert _profit_factor(float("inf")) == "∞"


def test_active_app_composes_results_dashboard():
    import inspect
    import app

    source = inspect.getsource(app.main)
    assert "apply_results_dashboard_workspace(window)" in source
