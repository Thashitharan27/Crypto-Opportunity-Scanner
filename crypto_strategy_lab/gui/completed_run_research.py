"""Explicitly reuse immutable completed runs as new editable research.

Opening a completed run is read-only.  Reusing it is a separate, deliberate
operation that copies the historical request plus v3 configuration into the
current GUI, while leaving the completed run and its artifacts untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from crypto_strategy_lab.data import MarketKind
from crypto_strategy_lab.data.timing import normalize_binance_interval
from crypto_strategy_lab.data_lake_config import (
    ResearchRunConfig,
    normalize_data_lake_config,
)
from crypto_strategy_lab.gui.v2_controller import GuiResearchRequest


@dataclass(frozen=True)
class CompletedRunResearchSeed:
    """Editable research intent reconstructed from one completed run."""

    run_id: str
    code_commit: str | None
    request: GuiResearchRequest
    config: ResearchRunConfig


def _utc_datetime(value: Any, label: str):
    if value in (None, ""):
        raise ValueError(f"Completed run is missing {label}.")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Completed run has an invalid {label}.")
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    return timestamp.to_pydatetime()


def _request_sources(manifest: Mapping[str, Any]) -> tuple[dict, dict]:
    top = manifest.get("request")
    if not isinstance(top, Mapping):
        top = {}
    research = manifest.get("research")
    nested = research.get("request") if isinstance(research, Mapping) else {}
    if not isinstance(nested, Mapping):
        nested = {}
    return dict(top), dict(nested)


def research_seed_from_manifest(manifest: Mapping[str, Any]) -> CompletedRunResearchSeed:
    """Reconstruct the editable request/config without mutating the historical run."""

    if not isinstance(manifest, Mapping):
        raise ValueError("Completed run manifest must be an object.")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Completed run does not contain a reusable v3 configuration snapshot.")
    config = normalize_data_lake_config(dict(raw_config))
    if not isinstance(config, ResearchRunConfig):
        raise ValueError("Completed run configuration is not the native v3 research contract.")

    request, legacy_request = _request_sources(manifest)
    symbol = str(request.get("symbol") or legacy_request.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Completed run is missing its research symbol.")

    strategy_interval = str(
        request.get("requested_strategy_interval")
        or legacy_request.get("strategy_interval")
        or ""
    ).strip()
    if not strategy_interval:
        raise ValueError("Completed run is missing its requested strategy timeframe.")

    if "requested_intrabar_interval" in request:
        intrabar_interval = request.get("requested_intrabar_interval")
    else:
        intrabar_interval = legacy_request.get("intrabar_interval")
    intrabar_interval = (
        str(intrabar_interval).strip() if intrabar_interval not in (None, "") else None
    )

    start = _utc_datetime(request.get("start") or legacy_request.get("start"), "start date")
    end = _utc_datetime(request.get("end") or legacy_request.get("end"), "end date")
    if start >= end:
        raise ValueError("Completed run has an invalid research date range.")

    # run-manifest v1 was created for the Binance-only workstation and did not
    # persist exchange explicitly.  Treat that historical omission as Binance;
    # newer manifests can provide an explicit exchange without changing this API.
    exchange = str(request.get("exchange") or "binance").strip().lower()
    market = request.get("market") or MarketKind.FUTURES_UM.value

    expected_strategy = normalize_binance_interval(
        f"{config.data.strategy_timeframe_minutes}m"
    )
    if normalize_binance_interval(strategy_interval) != expected_strategy:
        raise ValueError(
            "Completed run request and configuration disagree on the strategy timeframe."
        )
    expected_intrabar = (
        normalize_binance_interval(f"{config.data.intrabar_timeframe_minutes}m")
        if config.data.use_intrabar_data
        else None
    )
    requested_intrabar = (
        normalize_binance_interval(intrabar_interval) if intrabar_interval else None
    )
    if requested_intrabar != expected_intrabar:
        raise ValueError(
            "Completed run request and configuration disagree on the intrabar timeframe."
        )

    return CompletedRunResearchSeed(
        run_id=str(manifest.get("run_id") or ""),
        code_commit=(str(manifest.get("code_commit")) if manifest.get("code_commit") else None),
        request=GuiResearchRequest(
            exchange=exchange,
            market=market,
            symbol=symbol,
            period_start=start,
            period_end=end,
            strategy_timeframe=normalize_binance_interval(strategy_interval),
            intrabar_timeframe=requested_intrabar,
        ),
        config=config,
    )


def _market_value(value: Any):
    native = getattr(value, "value", value)
    try:
        return MarketKind(str(native))
    except ValueError as exc:
        raise ValueError(f"This Crypto Strategy Lab build does not support market {native!r}.") from exc


def _supported_request_indices(window, request: GuiResearchRequest) -> dict[str, int]:
    values = {
        "exchange": window.exchange.findData(request.exchange),
        "market": window.market.findData(_market_value(request.market)),
        "strategy": window.strategy_tf.findData(request.strategy_timeframe),
        "intrabar": window.intrabar_tf.findData(request.intrabar_timeframe),
    }
    unsupported = [name for name, index in values.items() if index < 0]
    if unsupported:
        raise ValueError(
            "This completed run cannot be edited by the current workstation because these "
            "request values are unsupported: " + ", ".join(unsupported)
        )
    return values


def _open_setup_page(window) -> None:
    pages = getattr(window, "pages", None)
    if pages is None:
        return
    for index in range(pages.count()):
        page = pages.widget(index)
        if any(label.text() == "Setup" for label in page.findChildren(QLabel)):
            pages.setCurrentIndex(index)
            return


def apply_completed_run_seed(window, seed: CompletedRunResearchSeed) -> None:
    """Copy a completed run into the authoritative editable GUI state."""

    indices = _supported_request_indices(window, seed.request)
    request_widgets = (
        window.exchange,
        window.market,
        window.symbol,
        window.start,
        window.end,
        window.strategy_tf,
        window.intrabar_tf,
    )
    prior_signal_states = [widget.blockSignals(True) for widget in request_widgets]
    try:
        # Existing composition installers wrap this authoritative loader, so using
        # it also restores the current Strategy/Features/Risk/Reports workspaces.
        window.apply_config(seed.config)
        window.exchange.setCurrentIndex(indices["exchange"])
        window.market.setCurrentIndex(indices["market"])
        window.symbol.setCurrentText(seed.request.symbol)
        window.start.setDate(
            QDate(seed.request.period_start.year, seed.request.period_start.month, seed.request.period_start.day)
        )
        window.end.setDate(
            QDate(seed.request.period_end.year, seed.request.period_end.month, seed.request.period_end.day)
        )
        window.strategy_tf.setCurrentIndex(indices["strategy"])
        window.intrabar_tf.setCurrentIndex(indices["intrabar"])
    finally:
        for widget, prior in zip(request_widgets, prior_signal_states):
            widget.blockSignals(prior)

    if hasattr(window, "_invalidate_range_validation"):
        window._invalidate_range_validation()
    if hasattr(window, "refresh_coverage"):
        window.refresh_coverage()
    if hasattr(window, "_refresh_run_data_view"):
        window._refresh_run_data_view()
    if hasattr(window, "_refresh_summary_from_widgets"):
        window._refresh_summary_from_widgets()
    _open_setup_page(window)


def load_completed_run_read_only(window, workspace, manifest_path: Path) -> dict:
    """Open a completed run in Results without changing the editable research request."""

    path = Path(manifest_path)
    if path.name.lower() != "run_manifest.json":
        raise ValueError("Select a completed run_manifest.json file.")
    if not path.is_file():
        raise ValueError("The selected run manifest does not exist.")
    manifest, _summary = window.service.completed_runs.read(path.parent)
    window._run_dir = path.parent
    window._manifest = manifest
    workspace.refresh_completed_run()
    return manifest


def install_completed_run_research_actions(window, workspace) -> None:
    """Add explicit read-only open and deliberate reuse actions to Results Dashboard."""

    if getattr(workspace, "completed_run_reuse_group", None) is not None:
        return

    group = QGroupBox("Completed Research")
    layout = QVBoxLayout(group)
    note = QLabel(
        "Opening a completed run is read-only. Use as New Research explicitly copies its "
        "request and v3 configuration into the editable workstation. The completed run stays "
        "unchanged and any new backtest uses the current application code."
    )
    note.setWordWrap(True)
    note.setStyleSheet("color:#52606d")
    layout.addWidget(note)

    detail = QLabel("No completed run selected.")
    detail.setWordWrap(True)
    layout.addWidget(detail)

    actions = QHBoxLayout()
    open_button = QPushButton("Open Completed Run…")
    use_button = QPushButton("Use as New Research")
    use_button.setEnabled(False)
    actions.addWidget(open_button)
    actions.addWidget(use_button)
    actions.addStretch()
    layout.addLayout(actions)

    dashboard_layout = workspace.layout()
    dashboard_layout.insertWidget(1, group)
    workspace.completed_run_reuse_group = group
    workspace.open_completed_run_button = open_button
    workspace.use_as_new_research_button = use_button
    workspace.completed_run_reuse_detail = detail

    original_refresh = workspace.refresh_completed_run

    def refresh_with_reuse_state():
        original_refresh()
        manifest = dict(getattr(window, "_manifest", {}) or {})
        run_dir = getattr(window, "_run_dir", None)
        available = bool(manifest and run_dir)
        use_button.setEnabled(available)
        if not available:
            detail.setText("No completed run selected.")
            return
        run_id = str(manifest.get("run_id") or "")
        code_commit = str(manifest.get("code_commit") or "")
        bits = []
        if run_id:
            bits.append(f"Run {run_id}")
        if code_commit:
            bits.append(f"Original code {code_commit[:12]}")
        detail.setText(" · ".join(bits) if bits else "Completed run loaded read-only.")

    workspace.refresh_completed_run = refresh_with_reuse_state

    def choose_completed_run():
        start_dir = str(getattr(window, "output_root", None).text()) if getattr(window, "output_root", None) else ""
        selected, _filter = QFileDialog.getOpenFileName(
            window,
            "Open completed Crypto Strategy Lab run",
            start_dir,
            "Run manifest (run_manifest.json);;JSON (*.json)",
        )
        if not selected:
            return
        try:
            load_completed_run_read_only(window, workspace, Path(selected))
        except Exception as exc:
            QMessageBox.warning(window, "Cannot open completed run", str(exc))

    def use_as_new_research():
        manifest = dict(getattr(window, "_manifest", {}) or {})
        if not manifest:
            return
        try:
            seed = research_seed_from_manifest(manifest)
            apply_completed_run_seed(window, seed)
        except Exception as exc:
            QMessageBox.warning(window, "Cannot reuse completed run", str(exc))

    open_button.clicked.connect(choose_completed_run)
    use_button.clicked.connect(use_as_new_research)
