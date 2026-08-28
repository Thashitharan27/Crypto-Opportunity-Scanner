from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.review_run_install import apply_review_run_workspace
    from crypto_strategy_lab.gui.run_progress import install_run_progress
    from crypto_strategy_lab.gui.setup_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def inventory(self, *_args):
            return []

        def coverage(self, request):
            base = dict(
                first_period=datetime(2020, 1, 1, tzinfo=timezone.utc),
                last_period=datetime(2026, 8, 21, tzinfo=timezone.utc),
                archive_count=99,
                state="AVAILABLE",
            )
            rows = [
                {"dataset": "klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "funding_rate", "interval": None, **base},
                {"dataset": "metrics", "interval": None, **base},
                {"dataset": "mark_price_klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "index_price_klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "premium_index_klines", "interval": request.strategy_timeframe, **base},
                {"dataset": "klines", "interval": "5m", **base},
            ]
            if request.intrabar_timeframe:
                rows.append({"dataset": "klines", "interval": request.intrabar_timeframe, **base})
            return rows

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    window = MainWindow(service=Service())
    window._validation_debounce.stop()
    install_run_progress(window)
    apply_review_run_workspace(window)
    return app, window


def test_active_app_installs_review_workspace_after_progress():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "apply_review_run_workspace" in source
    assert source.index("install_run_progress(window)") < source.index(
        "apply_review_run_workspace(window)"
    )


def test_review_workspace_reuses_authoritative_actions_and_is_compact():
    _app, window = _window()
    try:
        from PySide6.QtWidgets import QPushButton

        workspace = window.review_run_workspace
        assert window.pages.widget(5).findChild(type(workspace)) is workspace
        assert window.save.parent() is workspace
        assert window.load.parent() is workspace
        assert window.run_button.parent() is workspace
        assert window.review_summary.isHidden()
        assert window.output_root.isHidden()
        assert workspace.progress_status is window.run_progress_status
        assert workspace.progress_status.isHidden()

        legacy_run_actions = [
            button
            for button in window.findChildren(QPushButton)
            if button is not window.run_button
            and button.text().strip().upper() == "RUN BACKTEST"
        ]
        assert legacy_run_actions
        assert all(button.isHidden() and not button.isEnabled() for button in legacy_run_actions)
    finally:
        window.close()


def test_review_summary_uses_current_rule_model_and_calculates_base_concurrent_risk():
    _app, window = _window()
    try:
        workspace = window.review_run_workspace
        workspace.refresh(window.build_config())
        assert workspace.plan_values["direction"].text() == "DI Direction"
        assert workspace.plan_values["markets"].text() == "All 6 environments"
        assert workspace.plan_values["entry"].text() == "None — DI Direction only"
        assert "Available to Rules" not in workspace.plan_values["pressure"].text()
        assert workspace.risk_values["equity"].text() == "$1,000.00"
        assert workspace.risk_values["risk"].text() == "1.00% · $10.00"
        assert workspace.risk_values["concurrent"].text() == "1.00% · $10.00"
        assert "Canonical completed-run artifact set" in "\n".join(
            label.text() for label in workspace.findChildren(type(workspace.output_path))
        )
    finally:
        window.close()


def test_readiness_drives_primary_action_without_changing_run_path():
    _app, window = _window()
    try:
        workspace = window.review_run_workspace

        window._set_readiness(
            "NOT READY",
            "Required strategy candles are unavailable.",
            state="blocked",
        )
        assert workspace.readiness_title.text() == "NOT READY"
        assert not window.run_button.isEnabled()
        assert not workspace.go_setup.isHidden()

        window._set_readiness(
            "CHECKING DATA…",
            "Exact candle continuity will be checked before execution.",
            state="pending",
        )
        assert window.run_button.text() == "Validate & Run"
        assert window.run_button.isEnabled()
        assert workspace.go_setup.isHidden()

        window._set_readiness(
            "READY TO RUN",
            "All required run data is validated.",
            state="ready",
        )
        assert window.run_button.text() == "Run Backtest"
        assert window.run_button.isEnabled()
        # The same native button remains wired; the workspace did not create a
        # second execution control or a parallel start path.
        assert window.run_button is workspace.window.run_button
    finally:
        window.close()


def test_completed_thread_lifecycle_restores_idle_run_action():
    _app, window = _window()
    try:
        from PySide6.QtCore import QObject, QThread
        from crypto_strategy_lab.gui.review_run_install import (
            _complete_run_thread_lifecycle,
        )

        workspace = window.review_run_workspace
        thread = QThread(window)
        worker = QObject()
        window._thread = thread
        window._worker = worker
        window._set_readiness(
            "READY TO RUN",
            "All required run data is validated.",
            state="ready",
        )
        assert window.run_button.text() == "Running…"
        assert not window.run_button.isEnabled()

        _complete_run_thread_lifecycle(window, workspace, thread, worker)
        assert window._thread is None
        assert window._worker is None
        assert window.run_button.text() == "Run Backtest"
        assert window.run_button.isEnabled()
    finally:
        window.close()


def test_completion_wrappers_remain_qobject_bound_after_full_app_composition():
    app, window = _window()
    try:
        from PySide6.QtCore import QObject
        from crypto_strategy_lab.gui.results_dashboard_install import (
            apply_results_dashboard_workspace,
        )

        review_bridge = window._review_run_completion_bridge
        assert isinstance(review_bridge, QObject)
        assert getattr(window._finished, "__self__", None) is review_bridge
        assert review_bridge.thread() is app.thread()
        assert getattr(window._failed, "__self__", None) is review_bridge

        apply_results_dashboard_workspace(window)
        results_bridge = window._results_completion_bridge
        assert isinstance(results_bridge, QObject)
        assert getattr(window._finished, "__self__", None) is results_bridge
        assert results_bridge.thread() is app.thread()
        # Results chains into the Review bridge, which is also GUI-thread bound.
        assert getattr(results_bridge.original_finished, "__self__", None) is review_bridge
    finally:
        window.close()


def test_progress_is_hidden_when_idle_and_revealed_by_real_run_events():
    _app, window = _window()
    try:
        status = window.run_progress_status
        assert status.isHidden()
        window._run_progress_relay.render(
            {"kind": "stage", "label": "SIMULATION", "detail": "Running strategy"}
        )
        assert not status.isHidden()
        assert window.stage.text() == "SIMULATION"
        assert window.run_progress_detail.text() == "Running strategy"
    finally:
        window.close()
