from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    core = pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
    return widgets.QApplication.instance() or widgets.QApplication([]), widgets, core


def _manifest():
    from crypto_strategy_lab.data_lake_config import DataConfig, FeatureConfig, ResearchRunConfig

    config = ResearchRunConfig(
        data=DataConfig(
            strategy_timeframe_minutes=60,
            intrabar_timeframe_minutes=1,
            use_intrabar_data=True,
            intrabar_missing_policy="ERROR",
        ),
        features=replace(FeatureConfig(), funding_zscore_window_days=14.0),
    )
    return {
        "run_id": "af8461472bc54b04a837aac78dec95a5",
        "code_commit": "7da41427b2ac704589b13fc0b38ca0c04babf610",
        # Historical run-manifest v1 intentionally has no exchange field.
        "request": {
            "market": "futures_um",
            "symbol": "BTCUSDT",
            "start": "2020-09-01T00:00:00+00:00",
            "end": "2022-12-31T00:00:00+00:00",
            "requested_strategy_interval": "1h",
            "requested_intrabar_interval": "1m",
            "effective_intrabar_interval": "1m",
        },
        "research": {
            "request": {
                "symbol": "BTCUSDT",
                "start": "2020-09-01T00:00:00+00:00",
                "end": "2022-12-31T00:00:00+00:00",
                "strategy_interval": "1h",
                "intrabar_interval": "1m",
            }
        },
        "config": config.to_dict(),
    }


def test_manifest_seed_restores_request_and_full_v3_config():
    from crypto_strategy_lab.gui.completed_run_research import research_seed_from_manifest

    seed = research_seed_from_manifest(_manifest())

    assert seed.run_id == "af8461472bc54b04a837aac78dec95a5"
    assert seed.code_commit.startswith("7da41427")
    assert seed.request.exchange == "binance"
    assert str(seed.request.market) == "futures_um"
    assert seed.request.symbol == "BTCUSDT"
    assert seed.request.period_start.date().isoformat() == "2020-09-01"
    assert seed.request.period_end.date().isoformat() == "2022-12-31"
    assert seed.request.strategy_timeframe == "1h"
    assert seed.request.intrabar_timeframe == "1m"
    assert seed.config.data.strategy_timeframe_minutes == 60
    assert seed.config.features.funding_zscore_window_days == 14.0


def test_manifest_seed_rejects_request_config_timeframe_disagreement():
    from crypto_strategy_lab.gui.completed_run_research import research_seed_from_manifest

    manifest = _manifest()
    manifest["request"]["requested_strategy_interval"] = "15m"

    with pytest.raises(ValueError, match="disagree on the strategy timeframe"):
        research_seed_from_manifest(manifest)


class _EditableWindow:
    def __init__(self):
        _qt, widgets, core = _app()
        from crypto_strategy_lab.data import MarketKind
        from crypto_strategy_lab.data_lake_config import ResearchRunConfig
        from crypto_strategy_lab.gui.v2_main_window import TimeframeCombo, TIMEFRAME_LABELS

        self.exchange = widgets.QComboBox()
        self.exchange.addItem("Binance", "binance")
        self.market = widgets.QComboBox()
        self.market.addItem("USD-M Futures", MarketKind.FUTURES_UM)
        self.symbol = widgets.QComboBox()
        self.symbol.setEditable(True)
        self.symbol.addItem("ETHUSDT")
        self.symbol.setCurrentText("ETHUSDT")
        self.start = widgets.QDateEdit(core.QDate(2024, 1, 1))
        self.end = widgets.QDateEdit(core.QDate(2026, 8, 24))
        self.strategy_tf = TimeframeCombo()
        for value in ("15m", "1h", "4h", "1d"):
            self.strategy_tf.addItem(TIMEFRAME_LABELS[value], value)
        self.strategy_tf.setCurrentText("15m")
        self.intrabar_tf = TimeframeCombo()
        self.intrabar_tf.addItem("None", None)
        for value in ("1m", "5m", "15m"):
            self.intrabar_tf.addItem(TIMEFRAME_LABELS[value], value)
        self.intrabar_tf.setCurrentText("1m")

        self.pages = widgets.QStackedWidget()
        setup = widgets.QWidget()
        setup_layout = widgets.QVBoxLayout(setup)
        setup_layout.addWidget(widgets.QLabel("Setup"))
        other = widgets.QWidget()
        other_layout = widgets.QVBoxLayout(other)
        other_layout.addWidget(widgets.QLabel("Results Dashboard"))
        self.pages.addWidget(setup)
        self.pages.addWidget(other)
        self.pages.setCurrentIndex(1)

        self.config = ResearchRunConfig()
        self.applied_config = None
        self.invalidations = 0
        self.coverage_refreshes = 0
        self.data_refreshes = 0
        self.summary_refreshes = 0

    def apply_config(self, config):
        self.applied_config = config
        self.config = config

    def _invalidate_range_validation(self):
        self.invalidations += 1

    def refresh_coverage(self):
        self.coverage_refreshes += 1

    def _refresh_run_data_view(self):
        self.data_refreshes += 1

    def _refresh_summary_from_widgets(self):
        self.summary_refreshes += 1


def test_use_as_new_research_populates_authoritative_editable_request():
    _qt, _widgets, _core = _app()
    from crypto_strategy_lab.gui.completed_run_research import (
        apply_completed_run_seed,
        research_seed_from_manifest,
    )

    window = _EditableWindow()
    seed = research_seed_from_manifest(_manifest())
    apply_completed_run_seed(window, seed)

    assert window.applied_config == seed.config
    assert window.symbol.currentText() == "BTCUSDT"
    assert window.start.date().toPython().isoformat() == "2020-09-01"
    assert window.end.date().toPython().isoformat() == "2022-12-31"
    assert window.strategy_tf.currentData() == "1h"
    assert window.intrabar_tf.currentData() == "1m"
    assert window.exchange.currentData() == "binance"
    assert getattr(window.market.currentData(), "value", window.market.currentData()) == "futures_um"
    assert window.invalidations == 1
    assert window.coverage_refreshes == 1
    assert window.data_refreshes == 1
    assert window.summary_refreshes == 1
    assert window.pages.currentIndex() == 0


def test_open_completed_run_is_read_only_for_setup(tmp_path: Path):
    _qt, _widgets, _core = _app()
    from crypto_strategy_lab.gui.completed_run_research import load_completed_run_read_only

    window = _EditableWindow()
    original = (
        window.symbol.currentText(),
        window.start.date(),
        window.end.date(),
        window.strategy_tf.currentData(),
        window.intrabar_tf.currentData(),
        window.config,
        window.pages.currentIndex(),
    )
    manifest = _manifest()

    class CompletedRuns:
        def read(self, run_dir):
            assert run_dir == tmp_path
            return manifest, {"total_trades": 1}

    window.service = SimpleNamespace(completed_runs=CompletedRuns())
    window._manifest = None
    window._run_dir = None

    class Workspace:
        def __init__(self):
            self.refreshes = 0

        def refresh_completed_run(self):
            self.refreshes += 1

    workspace = Workspace()
    path = tmp_path / "run_manifest.json"
    path.write_text("{}", encoding="utf-8")

    loaded = load_completed_run_read_only(window, workspace, path)

    assert loaded is manifest
    assert window._manifest is manifest
    assert window._run_dir == tmp_path
    assert workspace.refreshes == 1
    assert (
        window.symbol.currentText(),
        window.start.date(),
        window.end.date(),
        window.strategy_tf.currentData(),
        window.intrabar_tf.currentData(),
        window.config,
        window.pages.currentIndex(),
    ) == original
