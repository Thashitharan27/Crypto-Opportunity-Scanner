"""Install the ChatGPT/MCP connection lifecycle into the composed v3 GUI."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from crypto_strategy_lab.gui.chatgpt_connection import ChatGPTIntegrationWidget


_AUTOSTART_MIGRATION_KEY = "chatgpt_auto_start_default_v1"
_AUTOSTART_SETTING_KEY = "auto_start_chatgpt_connection"
_AUTOSTART_PREFERENCE_KEY = "chatgpt_auto_start_preference_v2"


def _setting_is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _persist_autostart(widget: ChatGPTIntegrationWidget, checked: bool | None = None) -> None:
    """Persist the user's auto-start choice immediately and durably."""
    value = widget.auto_start.isChecked() if checked is None else bool(checked)
    settings = widget.settings
    # Keep the legacy key coherent for the reusable widget, but use the v2 key
    # as the authoritative preference because the legacy widget can overwrite
    # its own key while restoring other controls during construction.
    settings.setValue(_AUTOSTART_SETTING_KEY, value)
    settings.setValue(_AUTOSTART_PREFERENCE_KEY, value)
    settings.sync()


def _restore_preference(widget: ChatGPTIntegrationWidget) -> None:
    """Restore the durable preference after the legacy widget finishes loading."""
    settings = widget.settings
    if settings.contains(_AUTOSTART_PREFERENCE_KEY):
        desired = _setting_is_true(settings.value(_AUTOSTART_PREFERENCE_KEY, True))
    else:
        # v2 migration: auto-start is the product default. Once the user changes
        # it, the v2 key preserves that explicit choice on subsequent launches.
        desired = True
        settings.setValue(_AUTOSTART_PREFERENCE_KEY, desired)

    widget.auto_start.blockSignals(True)
    try:
        widget.auto_start.setChecked(desired)
    finally:
        widget.auto_start.blockSignals(False)

    settings.setValue(_AUTOSTART_SETTING_KEY, desired)
    settings.setValue(_AUTOSTART_MIGRATION_KEY, True)
    settings.sync()


def _auto_start_if_ready(widget: ChatGPTIntegrationWidget) -> None:
    """Start silently only when the saved connection configuration is usable."""
    if not widget.auto_start.isChecked():
        return
    _key, errors = widget._validated()
    if errors:
        widget.manager._log(
            "GUI",
            "Automatic ChatGPT connection skipped because configuration is incomplete.",
        )
        return
    widget.start()


def apply_chatgpt_autostart(window) -> None:
    """Make the current v3 ChatGPT widget default-on and lifecycle-aware."""
    if getattr(window, "_chatgpt_autostart_installed", False):
        return

    widget = window.findChild(ChatGPTIntegrationWidget)
    if widget is None:
        raise RuntimeError("Active GUI does not contain the ChatGPT integration widget")

    _restore_preference(widget)
    widget.auto_start.toggled.connect(
        lambda checked: _persist_autostart(widget, checked)
    )

    original_post_show = window.start_post_show_tasks

    def start_post_show_tasks() -> None:
        original_post_show()
        QTimer.singleShot(0, lambda: _auto_start_if_ready(widget))

    def shutdown() -> None:
        # Force the final checkbox value to disk before child-process cleanup.
        _persist_autostart(widget)
        widget.shutdown()

    window.start_post_show_tasks = start_post_show_tasks
    window.chatgpt_integration = widget

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(shutdown)

    window._chatgpt_autostart_installed = True
