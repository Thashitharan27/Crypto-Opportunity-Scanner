"""GUI entry point for Crypto Strategy Lab."""
import os
import sys
import traceback

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from crypto_strategy_lab.gui.run_progress import install_run_progress
from crypto_strategy_lab.gui.research_feature_ownership import (
    apply_research_feature_ownership,
)
from crypto_strategy_lab.gui.risk_execution_install import (
    apply_risk_execution_workspace,
)
from crypto_strategy_lab.gui.reports_diagnostics_install import (
    apply_reports_diagnostics_workspace,
)
from crypto_strategy_lab.gui.review_run_install import apply_review_run_workspace
from crypto_strategy_lab.gui.results_dashboard_install import (
    apply_results_dashboard_workspace,
)
from crypto_strategy_lab.gui.data_library_install import (
    apply_data_library_workspace,
)
from crypto_strategy_lab.gui.chatgpt_autostart_install import (
    apply_chatgpt_autostart,
)
from crypto_strategy_lab.gui.github_sync_install import apply_github_sync_safety
from crypto_strategy_lab.gui.desktop_style import (
    apply_application_style,
    apply_shell_style,
)
# The active researcher window layers run-faithful readiness over the compact
# Setup workspace while keeping the proven v2 data/results shell.
from crypto_strategy_lab.gui.v2_main_window import MainWindow as StableGuiShell
from crypto_strategy_lab.gui import run_readiness

MainWindow = run_readiness.MainWindow
assert issubclass(MainWindow, StableGuiShell)


def _splash_pixmap() -> QPixmap:
    pixmap = QPixmap(480, 220)
    pixmap.fill(QColor("#172033"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("#f5f7fb"))
    painter.setFont(QFont("Segoe UI", 24, QFont.Weight.DemiBold))
    painter.drawText(pixmap.rect().adjusted(30, 20, -30, -55), Qt.AlignCenter,
                     "Crypto Strategy Lab")
    painter.end()
    return pixmap


def main() -> int:
    # tunnel-client's health/admin listener defaults to 127.0.0.1:8080.
    # Use an OS-assigned loopback port so Crypto Strategy Lab can run beside
    # another tunnel-client instance without colliding on port 8080.
    os.environ["HEALTH_LISTEN_ADDR"] = "127.0.0.1:0"

    app = QApplication(sys.argv)
    apply_application_style(app)
    splash = QSplashScreen(_splash_pixmap(), Qt.WindowStaysOnTopHint)
    splash.showMessage("Loading settings...", Qt.AlignBottom | Qt.AlignHCenter,
                       QColor("#d8dfeb"))
    splash.show()
    app.processEvents()

    def status(message: str) -> None:
        splash.showMessage(message, Qt.AlignBottom | Qt.AlignHCenter, QColor("#d8dfeb"))
        app.processEvents()

    try:
        window = MainWindow(startup_status=status)
        apply_research_feature_ownership(window)
        apply_risk_execution_workspace(window)
        apply_reports_diagnostics_workspace(window)
        install_run_progress(window)
        apply_review_run_workspace(window)
        apply_results_dashboard_workspace(window)
        apply_data_library_workspace(window)
        apply_chatgpt_autostart(window)
        apply_github_sync_safety(window)
        apply_shell_style(window)
        status("Ready")
        window.show()
        splash.finish(window)
        # Auto-connect only after the fully constructed main window is visible.
        window.start_post_show_tasks()
    except Exception as exc:
        splash.close()
        traceback.print_exc()
        QMessageBox.critical(None, "Crypto Strategy Lab - Startup Error",
                             f"Crypto Strategy Lab could not start.\n\n{exc}")
        return 1

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
