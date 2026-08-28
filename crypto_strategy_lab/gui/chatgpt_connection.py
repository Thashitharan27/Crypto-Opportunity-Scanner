"""Safe Qt process management and UI for the OpenAI Secure Tunnel."""
from __future__ import annotations

import importlib.util
import os
import re
import socket
import sys
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QPlainTextEdit, QSpinBox, QVBoxLayout, QWidget)

KEYRING_SERVICE = "CryptoStrategyLab.OpenAITunnel"
KEYRING_USERNAME = "runtime_api_key"
TUNNEL_ARGUMENTS = ["run", "--log.level=info", "--log.format=struct-text"]
_SECRET_RE = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|CONTROL_PLANE_API_KEY\s*[=:]\s*\S+)")
CREATE_NO_WINDOW = 0x08000000


def configure_hidden_process(process: QProcess) -> None:
    """Best-effort suppression of a Windows console for a Qt child process.

    Some PySide6 builds do not expose Qt's process-argument modifier. Console
    suppression is optional in that case: the child must remain launchable and
    its Qt-managed output pipes must remain intact.
    """
    if sys.platform != "win32":
        return

    modifier = getattr(process, "setCreateProcessArgumentsModifier", None)
    if not callable(modifier):
        return

    def add_no_window_flag(arguments) -> None:
        arguments.flags |= CREATE_NO_WINDOW

    modifier(add_no_window_flag)


def redact_secrets(value: str) -> str:
    """Remove plausible credentials from process output before retaining it."""
    return _SECRET_RE.sub("[REDACTED]", value)


def validate_configuration(path: str, tunnel_id: str, api_key: str | None,
                           output_dir: str | Path, port: int) -> list[str]:
    errors = []
    executable = Path(path).expanduser()
    if not executable.is_file():
        errors.append("Tunnel client executable does not exist.")
    if not tunnel_id.strip():
        errors.append("Tunnel ID is required.")
    if not api_key:
        errors.append("Runtime API key is not configured in Credential Manager.")
    if not Path(output_dir).expanduser().is_dir():
        errors.append("The configured output directory does not exist.")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        errors.append("MCP port must be between 1 and 65535.")
    if importlib.util.find_spec("mcp_server.server") is None:
        errors.append("The mcp_server.server Python module is not importable.")
    return errors


def tunnel_environment(api_key: str, tunnel_id: str, endpoint: str) -> QProcessEnvironment:
    env = QProcessEnvironment.systemEnvironment()
    env.insert("CONTROL_PLANE_API_KEY", api_key)
    env.insert("CONTROL_PLANE_TUNNEL_ID", tunnel_id)
    env.insert("MCP_SERVER_URL", endpoint)
    return env


class ChatGPTConnectionManager(QObject):
    """Own exactly the child processes started by this GUI."""
    state_changed = Signal(str, str, str)
    diagnostic_changed = Signal(str)
    log_added = Signal(str)
    error = Signal(str)

    def __init__(self, output_dir, parent=None):
        super().__init__(parent)
        self.output_dir = output_dir
        self.mcp = QProcess(self); self.tunnel = QProcess(self)
        configure_hidden_process(self.mcp); configure_hidden_process(self.tunnel)
        self.mcp.setProcessChannelMode(QProcess.MergedChannels)
        self.tunnel.setProcessChannelMode(QProcess.MergedChannels)
        self.mcp.readyReadStandardOutput.connect(lambda: self._read(self.mcp, "MCP"))
        self.tunnel.readyReadStandardOutput.connect(lambda: self._read(self.tunnel, "TUNNEL"))
        self.mcp.started.connect(lambda: self._process_started("MCP", self.mcp))
        self.tunnel.started.connect(lambda: self._process_started("TUNNEL", self.tunnel))
        self.mcp.errorOccurred.connect(lambda err: self._process_error("MCP", self.mcp, err))
        self.tunnel.errorOccurred.connect(lambda err: self._process_error("Tunnel", self.tunnel, err))
        self.mcp.finished.connect(lambda code, status: self._child_finished("MCP", code, status))
        self.tunnel.finished.connect(lambda code, status: self._child_finished("Tunnel", code, status))
        self.logs = deque(maxlen=2000); self.state = "Disconnected"
        self.last_diagnostic = ""
        self.port = 8765; self._starting = False; self._stopping = False
        self._tunnel_ready = False
        self._mcp_started = False; self._tunnel_started = False; self._deadline = 0
        self.monitor = QTimer(self); self.monitor.setInterval(250)
        self.monitor.timeout.connect(self._poll)

    @property
    def owns_running_processes(self):
        return ((self._mcp_started and self.mcp.state() != QProcess.NotRunning) or
                (self._tunnel_started and self.tunnel.state() != QProcess.NotRunning))

    def _log(self, source, message):
        for line in redact_secrets(str(message)).splitlines():
            entry = f"[{source}] {line}"; self.logs.append(entry); self.log_added.emit(entry)

    def _set_diagnostic(self, message):
        message = redact_secrets(str(message)).strip()
        self.last_diagnostic = message
        self.diagnostic_changed.emit(message)
        if message:
            self._log("ERROR", message)

    def _read(self, process, source):
        output = bytes(process.readAllStandardOutput()).decode(errors="replace")
        if source == "TUNNEL" and any(marker in output.lower() for marker in
                                      ("tunnel-client started", "tunnel client started", "tunnel connected")):
            self._tunnel_ready = True
        self._log(source, output)

    def _process_started(self, name, process):
        self._log(name, f"Process started (PID {process.processId()}).")

    def _process_error(self, name, process, error):
        detail = process.errorString() or str(error)
        self._log(name.upper(), f"QProcess error {error}: {detail}")
        if self._stopping:
            return
        self._starting = False
        self.state = "Error"
        message = f"{name} process error: {detail}"
        self._set_diagnostic(message)
        self.error.emit(message + "\n\nOpen Logs for the tunnel-client output and exit details.")
        self._emit()

    def _reachable(self):
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=.08): return True
        except OSError: return False

    def _emit(self):
        reachable = self._reachable()
        mcp = "Running" if reachable and self._mcp_started else ("Port in use" if reachable else "Stopped")
        tunnel = "Running" if self.tunnel.state() == QProcess.Running else "Stopped"
        self.state_changed.emit(self.state, mcp, tunnel)

    def start(self, path, tunnel_id, api_key, port):
        if self._starting or self.owns_running_processes:
            self._log("GUI", "Start ignored: connection is already active or starting."); return False
        self.port = port
        if self._reachable():
            self.state = "Error"; self._log("GUI", f"MCP port {port} is already in use.")
            self._set_diagnostic(f"MCP port {port} is already in use.")
            self.error.emit(f"MCP port {port} is already in use. The existing process was not changed."); self._emit(); return False
        self._path, self._tunnel_id, self._api_key = path, tunnel_id, api_key
        self._tunnel_ready = False
        self.last_diagnostic = ""; self.diagnostic_changed.emit("")
        self._starting = True; self.state = "Starting MCP..."; self._deadline = 40
        env = QProcessEnvironment.systemEnvironment()
        env.insert("CRYPTO_STRATEGY_LAB_MCP_PORT", str(port))
        env.insert("CRYPTO_STRATEGY_LAB_OUTPUT_DIR", str(Path(self.output_dir()).resolve()))
        self.mcp.setProcessEnvironment(env); self._mcp_started = True
        self._log("GUI", f"Starting local MCP server on 127.0.0.1:{port}.")
        self.mcp.start(sys.executable, ["-m", "mcp_server.server"]); self.monitor.start(); self._emit(); return True

    def _poll(self):
        if self._stopping: return
        if self._starting:
            self._deadline -= 1
            if self.state == "Starting MCP..." and self._reachable() and self.mcp.state() == QProcess.Running:
                self.state = "Starting Tunnel..."; self._deadline = 60; self._tunnel_started = True
                endpoint = f"http://127.0.0.1:{self.port}/mcp"
                self.tunnel.setProcessEnvironment(tunnel_environment(self._api_key, self._tunnel_id, endpoint))
                self._log("GUI", f"MCP is ready; starting secure tunnel for {endpoint}.")
                self.tunnel.start(self._path, TUNNEL_ARGUMENTS)
            elif self.state == "Starting Tunnel..." and self.tunnel.state() == QProcess.Running and self._tunnel_ready:
                self._starting = False; self.state = "Connected"; self._log("GUI", "ChatGPT connection is running.")
            elif self._deadline <= 0:
                self._starting = False; self.state = "Error"
                message = "Connection startup timed out. The MCP or tunnel did not become ready in time."
                self._set_diagnostic(message)
                self.error.emit(message + "\n\nOpen Logs for details.")
                self._emit()
        elif self.state == "Connected" and (not self._reachable() or self.tunnel.state() != QProcess.Running):
            self.state = "Error"
            message = "Connection health check failed: the MCP endpoint or secure tunnel stopped responding."
            self._set_diagnostic(message)
        self._emit()

    @staticmethod
    def _exit_status_name(exit_status):
        return "NormalExit" if exit_status == QProcess.NormalExit else "CrashExit"

    def _child_finished(self, name, exit_code=0, exit_status=QProcess.NormalExit):
        process = self.mcp if name == "MCP" else self.tunnel
        self._read(process, name.upper())
        status_name = self._exit_status_name(exit_status)
        self._log("GUI", f"{name} process exited with code {exit_code} ({status_name}).")
        if name == "MCP":
            self._mcp_started = False
        elif name == "Tunnel":
            self._tunnel_started = False
        if not self._stopping and (self._starting or self.state == "Connected"):
            self._starting = False; self.state = "Error"
            message = f"{name} process exited with code {exit_code} ({status_name})."
            self._set_diagnostic(message)
            self.error.emit(message + "\n\nOpen Logs for the tunnel-client output immediately before the exit.")
            self._emit()

    def _stop_owned_process(self, process, owned, name):
        """Stop one owned child, escalating only if graceful exit times out."""
        if not owned or process.state() == QProcess.NotRunning:
            return
        self._log("GUI", f"Stopping {name} process.")
        process.terminate()
        if process.waitForFinished(1500):
            return
        self._log("GUI", f"{name} did not stop gracefully; killing owned process.")
        process.kill()
        process.waitForFinished(1500)

    def stop(self):
        self._starting = False; self._stopping = True; self.state = "Stopping..."; self._emit()
        # The tunnel depends on MCP, so stop it first. Ownership flags prevent
        # an externally managed process (including one using our port) from
        # ever being terminated here.
        self._stop_owned_process(self.tunnel, self._tunnel_started, "Tunnel")
        self._stop_owned_process(self.mcp, self._mcp_started, "MCP")
        self._tunnel_started = self._mcp_started = self._stopping = False
        self.state = "Disconnected"; self.monitor.stop(); self._log("GUI", "Connection stopped."); self._emit()


class ChatGPTIntegrationWidget(QWidget):
    def __init__(self, settings, output_dir, parent=None):
        super().__init__(parent); self.settings = settings
        self.manager = ChatGPTConnectionManager(output_dir, self)
        self._build(); self._load()
        self.manager.state_changed.connect(self._status)
        self.manager.diagnostic_changed.connect(self._diagnostic)
        self.manager.error.connect(lambda text: QMessageBox.warning(self, "ChatGPT Connection", text))

    def _build(self):
        outer = QVBoxLayout(self); title = QLabel("ChatGPT Integration"); title.setStyleSheet("font-size:20px;font-weight:600")
        outer.addWidget(title)
        connection = QGroupBox("Connection"); form = QFormLayout(connection)
        self.connection_status=QLabel("● Disconnected"); self.mcp_status=QLabel("● Stopped"); self.tunnel_status=QLabel("● Stopped"); self.endpoint=QLabel(); self.last_error=QLabel("—")
        self.last_error.setWordWrap(True)
        form.addRow("Connection Status",self.connection_status); form.addRow("Local MCP Server",self.mcp_status); form.addRow("Secure Tunnel",self.tunnel_status); form.addRow("MCP Endpoint",self.endpoint); form.addRow("Last Connection Error",self.last_error)
        buttons=QHBoxLayout(); self.start_button=QPushButton("Start ChatGPT Connection"); self.stop_button=QPushButton("Stop Connection"); self.stop_button.setEnabled(False); buttons.addWidget(self.start_button); buttons.addWidget(self.stop_button); form.addRow(buttons)
        self.auto_start=QCheckBox("Start automatically with Crypto Strategy Lab"); form.addRow(self.auto_start); outer.addWidget(connection)
        config=QGroupBox("Configuration"); cf=QFormLayout(config)
        self.path=QLineEdit(); browse=QPushButton("Browse"); row=QHBoxLayout(); row.addWidget(self.path); row.addWidget(browse); cf.addRow("Tunnel Client",row)
        self.tunnel_id=QLineEdit(); cf.addRow("Tunnel ID",self.tunnel_id)
        self.key_status=QLabel("Not configured"); self.key_button=QPushButton("Set / Change API Key"); clear=QPushButton("Clear API Key"); kr=QHBoxLayout(); kr.addWidget(self.key_status); kr.addWidget(self.key_button); kr.addWidget(clear); cf.addRow("API Key",kr)
        self.port=QSpinBox(); self.port.setRange(1,65535); cf.addRow("MCP Port",self.port); outer.addWidget(config)
        actions=QHBoxLayout(); test=QPushButton("Test Configuration"); logs=QPushButton("Open Logs"); actions.addWidget(test); actions.addWidget(logs); actions.addStretch(); outer.addLayout(actions); outer.addStretch()
        self.start_button.clicked.connect(self.start); self.stop_button.clicked.connect(self.manager.stop); browse.clicked.connect(self.browse); self.key_button.clicked.connect(self.set_key); clear.clicked.connect(self.clear_key); test.clicked.connect(self.test); logs.clicked.connect(self.open_logs)
        self.path.editingFinished.connect(self._save); self.tunnel_id.editingFinished.connect(self._save); self.port.valueChanged.connect(self._save); self.auto_start.toggled.connect(self._save)

    def _credential(self):
        try: return importlib.import_module("keyring").get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception as exc: raise RuntimeError(f"Windows Credential Manager is unavailable: {exc}") from exc

    def _load(self):
        default_path = next((str(p) for p in (Path(r"C:\OpenAI-Tunnel\tunnel-client.exe"), Path.cwd()/"tunnel-client.exe") if p.is_file()), "")
        self.path.setText(str(self.settings.value("tunnel_client_path", default_path)))
        self.tunnel_id.setText(str(self.settings.value("tunnel_id", os.getenv("CONTROL_PLANE_TUNNEL_ID", ""))))
        try: port=int(self.settings.value("mcp_port", os.getenv("CRYPTO_STRATEGY_LAB_MCP_PORT", "8765")))
        except (TypeError,ValueError): port=8765
        self.port.setValue(port if 1 <= port <= 65535 else 8765)
        self.auto_start.setChecked(str(self.settings.value("auto_start_chatgpt_connection", "false")).lower()=="true")
        self._refresh_key(); self._update_endpoint()

    def _save(self):
        self.settings.setValue("tunnel_client_path",self.path.text().strip()); self.settings.setValue("tunnel_id",self.tunnel_id.text().strip())
        self.settings.setValue("mcp_port",self.port.value()); self.settings.setValue("auto_start_chatgpt_connection",self.auto_start.isChecked()); self._update_endpoint()

    def _refresh_key(self):
        try: configured=bool(self._credential())
        except RuntimeError: configured=False
        self.key_status.setText("Configured" if configured else "Not configured")

    def _update_endpoint(self): self.endpoint.setText(f"http://127.0.0.1:{self.port.value()}/mcp")
    def browse(self):
        value,_=QFileDialog.getOpenFileName(self,"Select tunnel client",self.path.text(),"Executables (*.exe);;All files (*)")
        if value: self.path.setText(value); self._save()
    def set_key(self):
        value,ok=QInputDialog.getText(self,"Runtime API Key","API key:",QLineEdit.Password)
        if ok and value:
            try: importlib.import_module("keyring").set_password(KEYRING_SERVICE,KEYRING_USERNAME,value)
            except Exception as exc: QMessageBox.critical(self,"Credential Manager",f"Could not securely store the key: {exc}")
            self._refresh_key()
    def clear_key(self):
        try: importlib.import_module("keyring").delete_password(KEYRING_SERVICE,KEYRING_USERNAME)
        except Exception as exc: QMessageBox.critical(self,"Credential Manager",f"Could not clear the key: {exc}")
        self._refresh_key()
    def _validated(self):
        try: key=self._credential()
        except RuntimeError as exc: return None,[str(exc)]
        return key,validate_configuration(self.path.text(),self.tunnel_id.text(),key,self.manager.output_dir(),self.port.value())
    def test(self):
        _,errors=self._validated(); QMessageBox.information(self,"Configuration Test","Configuration is ready." if not errors else "Please fix:\n\n"+"\n".join(errors))
    def start(self):
        self._save(); key,errors=self._validated()
        if errors: QMessageBox.warning(self,"ChatGPT Configuration","Please fix:\n\n"+"\n".join(errors)); return
        self.manager.start(self.path.text(),self.tunnel_id.text(),key,self.port.value())
    def auto_start_connection(self):
        if self.auto_start.isChecked(): self.start()
    def shutdown(self):
        """Stop child processes owned by this integration widget."""
        self.manager.stop()
    def _diagnostic(self,text):
        self.last_error.setText(text or "—")
    def _status(self,state,mcp,tunnel):
        color="#16833b" if state=="Connected" else ("#b42318" if state=="Error" else "#666")
        self.connection_status.setText(f"● {state}"); self.connection_status.setStyleSheet(f"color:{color};font-weight:600")
        self.mcp_status.setText(f"● {mcp}"); self.tunnel_status.setText(f"● {tunnel}")
        active=state in ("Starting MCP...","Starting Tunnel...","Connected","Stopping...") or self.manager.owns_running_processes
        self.start_button.setEnabled(not active); self.start_button.setText("Connection Active" if state=="Connected" else "Start ChatGPT Connection"); self.stop_button.setEnabled(active and state!="Stopping...")
    def open_logs(self):
        dialog=QDialog(self); dialog.setWindowTitle("ChatGPT Connection Logs"); dialog.resize(800,450); layout=QVBoxLayout(dialog); edit=QPlainTextEdit(); edit.setReadOnly(True); edit.setMaximumBlockCount(2000); edit.setPlainText("\n".join(self.manager.logs)); layout.addWidget(edit); self.manager.log_added.connect(edit.appendPlainText); dialog.setAttribute(Qt.WA_DeleteOnClose); dialog.show(); self._log_dialog=dialog
