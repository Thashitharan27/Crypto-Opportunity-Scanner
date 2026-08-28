from pathlib import Path
from unittest.mock import Mock, patch
from PySide6.QtCore import QProcess, QSettings
from crypto_strategy_lab.gui.chatgpt_connection import (ChatGPTConnectionManager,
    ChatGPTIntegrationWidget, TUNNEL_ARGUMENTS, redact_secrets,
    tunnel_environment, validate_configuration)


class FakeProcess:
    def __init__(self, name, events, graceful=True):
        self.name=name; self.events=events; self.running=True; self.graceful=graceful
        self.terminate=Mock(side_effect=self._terminate)
        self.kill=Mock(side_effect=self._kill)
        self.waitForFinished=Mock(side_effect=self._wait)

    def state(self): return QProcess.Running if self.running else QProcess.NotRunning
    def _terminate(self): self.events.append(f"terminate {self.name}")
    def _kill(self): self.events.append(f"kill {self.name}"); self.running=False
    def _wait(self, _timeout):
        self.events.append(f"wait {self.name}")
        if self.graceful: self.running=False
        return not self.running

def test_validation_reports_each_missing_item(tmp_path):
    with patch('crypto_strategy_lab.gui.chatgpt_connection.importlib.util.find_spec',return_value=object()):
        errors=validate_configuration(str(tmp_path/'missing.exe'),'',None,tmp_path/'missing-output',0)
    for phrase in ('executable','Tunnel ID','API key','output directory','port'):
        assert any(phrase in error for error in errors)

def test_valid_configuration(tmp_path):
    exe=tmp_path/'tunnel client.exe'; exe.touch(); out=tmp_path/'output'; out.mkdir()
    with patch('crypto_strategy_lab.gui.chatgpt_connection.importlib.util.find_spec',return_value=object()):
        assert validate_configuration(str(exe),'tunnel_example','secret',out,8765)==[]

def test_redaction():
    text=redact_secrets('sk-exampleSecret123 CONTROL_PLANE_API_KEY=another-secret')
    assert 'exampleSecret' not in text and 'another-secret' not in text
    assert text.count('[REDACTED]')==2

def test_tunnel_command_and_environment_do_not_mix_secret():
    secret='super-secret-value'; env=tunnel_environment(secret,'tunnel_example','http://127.0.0.1:8765/mcp')
    assert TUNNEL_ARGUMENTS==['run','--log.level=info','--log.format=struct-text']
    assert secret not in TUNNEL_ARGUMENTS
    assert env.value('CONTROL_PLANE_API_KEY')==secret
    assert env.value('CONTROL_PLANE_TUNNEL_ID')=='tunnel_example'
    assert env.value('MCP_SERVER_URL').endswith('/mcp')

def test_duplicate_start_and_stop_without_processes(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path)); manager._starting=True
    assert manager.start('client.exe','tunnel','key',8765) is False
    manager._starting=False; manager.stop(); assert manager.state=='Disconnected'

def test_shutdown_stops_owned_tunnel_before_owned_mcp(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path)); events=[]
    manager.tunnel=FakeProcess('tunnel',events)
    manager.mcp=FakeProcess('mcp',events)
    manager._tunnel_started=manager._mcp_started=True

    manager.stop()

    assert events == ['terminate tunnel','wait tunnel','terminate mcp','wait mcp']
    assert manager.tunnel.state()==manager.mcp.state()==QProcess.NotRunning

def test_shutdown_kills_owned_child_after_graceful_timeout(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path)); events=[]
    manager.tunnel=FakeProcess('tunnel',events,graceful=False)
    manager.mcp=FakeProcess('mcp',events,graceful=False)
    manager._tunnel_started=manager._mcp_started=True

    manager.stop()

    assert events == ['terminate tunnel','wait tunnel','kill tunnel','wait tunnel',
                      'terminate mcp','wait mcp','kill mcp','wait mcp']
    assert manager.tunnel.state()==manager.mcp.state()==QProcess.NotRunning

def test_shutdown_does_not_touch_unowned_processes_or_occupied_port(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path)); events=[]
    manager.tunnel=FakeProcess('external tunnel',events)
    manager.mcp=FakeProcess('external mcp',events)
    manager._reachable=Mock(return_value=True)

    manager.stop()

    assert events == []
    manager._reachable.assert_called()

def test_repeated_widget_shutdown_is_safe(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path)); events=[]
    manager.tunnel=FakeProcess('tunnel',events)
    manager.mcp=FakeProcess('mcp',events)
    manager._tunnel_started=manager._mcp_started=True
    widget=Mock(manager=manager)

    ChatGPTIntegrationWidget.shutdown(widget)
    ChatGPTIntegrationWidget.shutdown(widget)

    assert events == ['terminate tunnel','wait tunnel','terminate mcp','wait mcp']

def test_main_window_close_shuts_down_chatgpt_widget():
    from crypto_strategy_lab.gui.main_window import MainWindow
    window=Mock(chatgpt_tab=Mock())
    event=Mock()

    MainWindow.closeEvent(window,event)

    window.chatgpt_tab.shutdown.assert_called_once_with()
    event.accept.assert_called_once_with()

def test_clean_shutdown_allows_next_start_without_false_port_conflict(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path)); events=[]
    manager.tunnel=FakeProcess('tunnel',events)
    manager.mcp=FakeProcess('mcp',events)
    manager._tunnel_started=manager._mcp_started=True
    manager.stop()
    replacement_mcp=Mock()
    replacement_mcp.state.return_value=QProcess.NotRunning
    manager.mcp=replacement_mcp

    with patch.object(manager,'_reachable',return_value=False):
        assert manager.start('client.exe','tunnel','key',8765) is True

    replacement_mcp.start.assert_called_once()

def test_tunnel_process_error_is_exposed_without_stopping_owned_mcp(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path))
    manager._starting=True; manager._mcp_started=True; manager._tunnel_started=True
    process=Mock(); process.errorString.return_value='another tunnel instance is already active'

    with patch.object(manager,'_emit') as emit:
        manager._process_error('Tunnel',process,QProcess.FailedToStart)

    assert manager.state=='Error'
    assert manager._starting is False
    assert manager._mcp_started is True
    assert 'another tunnel instance is already active' in manager.last_diagnostic
    assert any('another tunnel instance is already active' in line for line in manager.logs)
    emit.assert_called_once_with()

def test_unexpected_tunnel_exit_records_exit_code_and_status(qapp,tmp_path):
    manager=ChatGPTConnectionManager(lambda:str(tmp_path))
    manager._starting=True; manager._mcp_started=True; manager._tunnel_started=True

    with patch.object(manager,'_read'), patch.object(manager,'_emit'):
        manager._child_finished('Tunnel',17,QProcess.CrashExit)

    assert manager.state=='Error'
    assert manager._tunnel_started is False
    assert manager._mcp_started is True
    assert 'code 17 (CrashExit)' in manager.last_diagnostic
    assert any('code 17 (CrashExit)' in line for line in manager.logs)

def test_api_key_is_not_a_settings_key(tmp_path):
    settings=QSettings(str(tmp_path/'settings.ini'),QSettings.IniFormat)
    for key,value in {'tunnel_client_path':'client.exe','tunnel_id':'tunnel_x','mcp_port':8765,'auto_start_chatgpt_connection':True}.items(): settings.setValue(key,value)
    settings.sync()
    assert not any('api' in key.lower() or 'credential' in key.lower() for key in settings.allKeys())
    assert 'secret' not in Path(settings.fileName()).read_text()
