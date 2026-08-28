"""Acceptance tests for the minimal active Run Output workspace."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from crypto_strategy_lab.data_lake_config import ResearchRunConfig


ROOT = Path(__file__).resolve().parents[1]


def test_active_app_installs_run_output_composition():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "apply_reports_diagnostics_workspace" in source
    assert "apply_reports_diagnostics_workspace(window)" in source


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.reports_diagnostics_install import (
        apply_reports_diagnostics_workspace,
    )
    from crypto_strategy_lab.gui.rule_main_window import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])

    class Catalog:
        def symbols(self):
            return ["BTCUSDT"]

        def coverage(self, _request):
            return []

        def inventory(self, *_args):
            return []

    class Service:
        catalog = Catalog()

        def refresh_catalog(self):
            return 0

    window = MainWindow(service=Service())
    apply_reports_diagnostics_workspace(window)
    return app, window


def _page_text(page, qt_widgets) -> str:
    labels = "\n".join(
        child.text() for child in page.findChildren(qt_widgets.QLabel)
    )
    groups = "\n".join(
        child.title() for child in page.findChildren(qt_widgets.QGroupBox)
    )
    buttons = "\n".join(
        child.text() for child in page.findChildren(qt_widgets.QPushButton)
    )
    return "\n".join((labels, groups, buttons))


def test_run_output_page_reuses_run_name_and_has_no_report_modes():
    _app, window = _window()
    try:
        qt_widgets = pytest.importorskip("PySide6.QtWidgets")
        workspace = window.reports_diagnostics_workspace
        assert workspace.run_name is window.reporting_form.widgets["run_name"]

        text = _page_text(window.pages.widget(4), qt_widgets)
        assert "Run Output" in text or "Completed Run Output" in text
        for retired in (
            "Report Preset",
            "Apply Preset",
            "Core",
            "Review — Recommended",
            "Deep Diagnostics",
            "Trade Journey",
            "Indicator Lifecycle",
            "Telemetry Interval",
            "Create standard result charts",
        ):
            assert retired not in text
    finally:
        window.close()


def test_run_output_page_documents_the_exact_clean_artifact_set():
    _app, window = _window()
    try:
        qt_widgets = pytest.importorskip("PySide6.QtWidgets")
        text = _page_text(window.pages.widget(4), qt_widgets)
        for expected in (
            "backtest_report.xlsx",
            "trade_list.csv",
            "summary.json",
            "data_quality.json",
            "artifacts/trades.parquet",
            "artifacts/feature_context.parquet",
            "artifacts/signals.parquet",
            "provenance/source_archives.parquet",
            "run_manifest.json",
        ):
            assert expected in text
        assert "telemetry.parquet" not in text
    finally:
        window.close()


def test_run_details_only_change_reporting_identity_and_output_folder():
    _app, window = _window()
    try:
        workspace = window.reports_diagnostics_workspace
        before = window.build_config()

        workspace.run_name.setText("clean-output-run")
        workspace.output_dir.setText("output/clean-output-run")
        after = window.build_config()

        assert after.reporting.run_name == "clean-output-run"
        assert after.reporting.output_dir == "output/clean-output-run"
        assert window.output_root.text() == "output/clean-output-run"
        assert window.output_root.isHidden()
        assert (after.data, after.features, after.strategy, after.execution) == (
            before.data,
            before.features,
            before.strategy,
            before.execution,
        )
    finally:
        window.close()


def test_config_load_refreshes_native_run_details_without_second_state():
    _app, window = _window()
    try:
        base = ResearchRunConfig()
        reporting = replace(
            base.reporting,
            run_name="loaded-run",
            output_dir="output/loaded-run",
        )
        window.apply_config(replace(base, reporting=reporting))

        workspace = window.reports_diagnostics_workspace
        assert workspace.run_name.text() == "loaded-run"
        assert workspace.output_dir.text() == "output/loaded-run"
        assert window.build_config().reporting == reporting
    finally:
        window.close()


def test_review_summary_exposes_one_canonical_output_contract():
    _app, window = _window()
    try:
        window._render_research_summary(window.build_config())
        text = window.review_summary.text()
        assert "Output: Canonical completed-run set" in text
        assert "Reports: QUICK" not in text
        assert "Reports: STANDARD" not in text
        assert "Reports: DEEP" not in text
    finally:
        window.close()
