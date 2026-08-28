"""Final GUI composition for the organized Risk & Execution workspace."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel

from crypto_strategy_lab.strategy_rule_model import uses_support_resistance_rules
from .risk_execution_workspace import RiskExecutionWorkspace


def _rules_require_support_resistance(window) -> bool:
    builder = getattr(window, "rule_builder", None)
    if builder is None:
        return False
    try:
        return uses_support_resistance_rules(
            builder.required_rules.rules(),
            builder.veto_rules.rules(),
            builder.flip_rules.rules(),
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _sync_target_sr_dependency(window) -> None:
    """S/R-capped targets own their causal S/R dependency automatically."""
    if getattr(window, "_syncing_target_sr_dependency", False):
        return
    execution_form = getattr(window, "execution_form", None)
    feature_form = getattr(window, "feature_form", None)
    if execution_form is None or feature_form is None:
        return
    target_mode = execution_form.widgets.get("sr_take_profit_mode")
    sr_toggle = feature_form.widgets.get("enable_support_resistance_analysis")
    if target_mode is None or sr_toggle is None:
        return

    window._syncing_target_sr_dependency = True
    try:
        required_by_target = str(target_mode.currentData() or "FIXED_R") == "SR_CAPPED_R"
        was_required = bool(getattr(window, "_sr_target_dependency_active", False))
        panel = getattr(window, "research_features_panel", None)

        if required_by_target:
            if not was_required:
                window._sr_target_previous_checked = bool(sr_toggle.isChecked())
            window._sr_target_dependency_active = True
            if not sr_toggle.isChecked():
                sr_toggle.setChecked(True)
            sr_toggle.setEnabled(False)
            sr_toggle.setText("Support / Resistance calculation required by target policy")
            if panel is not None:
                panel.sr_card.set_status("REQUIRED BY TARGET POLICY")
                panel.refresh_visibility()
            return

        window._sr_target_dependency_active = False
        if was_required and not _rules_require_support_resistance(window):
            previous = bool(getattr(window, "_sr_target_previous_checked", False))
            sr_toggle.setChecked(previous)
        if panel is not None:
            panel._sync_sr_requirement()
        else:
            sr_toggle.setEnabled(True)
    finally:
        window._syncing_target_sr_dependency = False


def apply_risk_execution_workspace(window) -> None:
    """Replace the long execution forms with one compact mode-aware workspace."""
    if getattr(window, "risk_execution_workspace", None) is not None:
        return
    if not all(
        hasattr(window, name)
        for name in ("execution_form", "base_execution_form", "_page", "_scroll", "_replace_page")
    ):
        return

    # Keep the legacy explanation label alive because the existing live-summary
    # method still updates it. It becomes a hidden compatibility sink while the
    # new workspace owns the visible Effective Plan summary.
    legacy_risk = getattr(window, "risk_explanation", None)
    if isinstance(legacy_risk, QLabel):
        legacy_risk.setParent(window)
        legacy_risk.hide()
        window._legacy_risk_explanation = legacy_risk

    workspace = RiskExecutionWorkspace(window)
    note = QLabel(
        "Account risk, stop sizing, profit policy and trade management are shown as one execution plan. "
        "Only controls relevant to the selected mode stay visible; advanced assumptions remain available below."
    )
    note.setWordWrap(True)
    note.setStyleSheet("background:#eef5fb; padding:8px; border:1px solid #c8d9e8")
    page = window._page("Risk & Execution", note, window._scroll(workspace))
    window._replace_page(3, page)
    window.risk_execution_workspace = workspace

    window.execution_form.changed.connect(workspace.refresh_summary_from_widgets)
    window.base_execution_form.changed.connect(workspace.refresh_summary_from_widgets)
    window.execution_form.widgets["sr_take_profit_mode"].currentIndexChanged.connect(
        lambda _index: _sync_target_sr_dependency(window)
    )
    window.feature_form.widgets["enable_support_resistance_analysis"].toggled.connect(
        lambda _checked: _sync_target_sr_dependency(window)
    )
    if hasattr(window, "rule_builder"):
        window.rule_builder.changed.connect(lambda: _sync_target_sr_dependency(window))

    # Config loading can update numeric spin boxes with signals blocked. Wrap the
    # instance apply hook so the composed page always refreshes after a load/reset.
    original_apply_config = window.apply_config

    def apply_config_and_refresh(config):
        result = original_apply_config(config)
        workspace.refresh_visibility()
        workspace.refresh_summary_from_widgets()
        _sync_target_sr_dependency(window)
        return result

    window.apply_config = apply_config_and_refresh

    _sync_target_sr_dependency(window)
    workspace.refresh_visibility()
    workspace.refresh_summary_from_widgets()
