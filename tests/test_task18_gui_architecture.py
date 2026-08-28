"""Focused acceptance tests for the Task-18 GUI boundary."""
from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pytest
from crypto_strategy_lab.data import MarketKind
from crypto_strategy_lab.data.quality import (DataQualityIssue, DataQualityReport,
    DataQualityStatus, DatasetQualityReport)
from crypto_strategy_lab.data_lake_config import (ExecutionProfileConfig, FeatureConfig,
    ResearchRunConfig, StrategyProfileConfig)
from crypto_strategy_lab.gui.v2_controller import (CompletedRunReader,
    CatalogStatusService, GuiApplicationService, GuiResearchRequest)
from crypto_strategy_lab.run_manifest import (FEATURE_RESEARCH_ARTIFACT_CONTRACT,
    FEATURE_RESEARCH_ARTIFACT_VERSION, RUN_MANIFEST_CONTRACT, RUN_MANIFEST_VERSION,
    RunArtifactError, file_sha256)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = [ROOT / "crypto_strategy_lab/gui/v2_main_window.py",
          ROOT / "crypto_strategy_lab/gui/v2_controller.py"]


def test_app_launches_only_authoritative_v2_window():
    tree = ast.parse((ROOT / "app.py").read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "crypto_strategy_lab.gui.v2_main_window" in imports
    assert not any(x and x.endswith(("main_window", "data_lake_main_window")) and
                   x != "crypto_strategy_lab.gui.v2_main_window" for x in imports)


def test_main_window_directly_inherits_qmainwindow_and_has_no_csv_selectors():
    tree = ast.parse(ACTIVE[0].read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow")
    assert [ast.unparse(x) for x in cls.bases] == ["QMainWindow"]
    source = ACTIVE[0].read_text()
    assert "Strategy CSV" not in source and "Intrabar CSV" not in source
    assert "input_csv" not in source and "intrabar_csv" not in source


def test_active_import_boundaries_and_no_monkey_patches():
    forbidden = {"load_backtest_bundle", "prepare_bundle_with_cache",
                 "DataLakeProductionBacktestEngine", "BinanceArchiveAdapter",
                 "BacktestWorker", "mcp_server"}
    for path in ACTIVE:
        tree = ast.parse(path.read_text())
        imported = {alias.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names}
        assert not forbidden.intersection(imported)
        # No module-level attribute assignment (the retired stack replaced classes this way).
        assert not any(isinstance(n, (ast.Assign, ast.AnnAssign)) and
                       isinstance(getattr(n, "target", None), ast.Attribute)
                       for n in tree.body)


def test_request_model_is_filename_free_and_complete():
    request = GuiResearchRequest("binance", MarketKind.FUTURES_UM, "btcusdt",
        datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,2,1,tzinfo=timezone.utc),
        "4h", "1m")
    native = request.to_data_request()
    assert native.symbol == "BTCUSDT" and native.exchange == "binance"
    assert native.strategy_interval == "4h" and native.intrabar_interval == "1m"
    assert not any("csv" in name for name in native.__dataclass_fields__)


def test_controller_invokes_injected_runner_exactly_once(tmp_path):
    class Catalog:
        def inventory(self, *_a, **_k): return []
    class Store:
        raw_root=tmp_path; catalog=Catalog()
    calls=[]
    class Runner:
        def run(self, request, config): calls.append((request, config)); return "result"
    service=object.__new__(GuiApplicationService); service.raw_root=tmp_path
    service.cache_root=tmp_path; service.store=Store()
    service._runner_factory=lambda output: Runner()
    request=GuiResearchRequest("binance",MarketKind.FUTURES_UM,"BTCUSDT",
        datetime(2026,1,1,tzinfo=timezone.utc),datetime(2026,1,2,tzinfo=timezone.utc),"15m","1m")
    assert service.run(request,ResearchRunConfig()) == "result"
    assert len(calls) == 1


def test_catalog_coverage_is_utc_normalized_and_path_free(tmp_path):
    class Catalog:
        def inventory(self, *_args, **_kwargs):
            return [{"exchange":"binance","symbol":"BTCUSDT","dataset":"klines",
                "interval":"4h","first_period":datetime(2026,1,1),
                "last_period":datetime(2026,1,3),"archive_count":2}]
    store=type("Store",(),{"raw_root":tmp_path,"catalog":Catalog()})()
    request=GuiResearchRequest("binance",MarketKind.FUTURES_UM,"BTCUSDT",
        datetime(2026,1,1,tzinfo=timezone.utc),datetime(2026,1,2,tzinfo=timezone.utc),"4h","1m")
    row=CatalogStatusService(store).coverage(request)[0]
    assert row["first_period"].tzinfo is not None and row["last_period"].tzinfo is not None
    assert row["state"] == "AVAILABLE" and "path" not in row


def test_native_v3_save_load_preserves_profiles_and_output_is_nonsemantic(tmp_path):
    service=object.__new__(GuiApplicationService); path=tmp_path/"native.json"
    original=ResearchRunConfig(); changed=replace(original,
        reporting=replace(original.reporting,output_dir=str(tmp_path/"runs")))
    service.save_config(path,changed); loaded=service.load_config(path)
    assert loaded == changed
    assert loaded.strategy.profiles == original.strategy.profiles
    assert loaded.execution.profiles == original.execution.profiles


def test_results_are_resolved_from_manifest_catalog(tmp_path):
    (tmp_path/"artifacts").mkdir(); (tmp_path/"artifacts/summary.json").write_text('{"total_trades": 7}')
    (tmp_path/"artifacts/signals.parquet").write_bytes(b"canonical-signal-double")
    summary_path=tmp_path/"artifacts/summary.json"
    manifest={"run_manifest_contract":RUN_MANIFEST_CONTRACT,"run_manifest_version":RUN_MANIFEST_VERSION,
              "run_status":"COMPLETED","research":{"artifact_contract":FEATURE_RESEARCH_ARTIFACT_CONTRACT,
              "artifact_version":FEATURE_RESEARCH_ARTIFACT_VERSION},
              "artifacts":{"summary":{"path":"artifacts/summary.json","sha256":file_sha256(summary_path)},
              "signals":{"path":"artifacts/signals.parquet","sha256":file_sha256(tmp_path/"artifacts/signals.parquet")}}}
    (tmp_path/"run_manifest.json").write_text(json.dumps(manifest))
    loaded,summary=CompletedRunReader().read(tmp_path)
    assert summary["total_trades"] == 7
    assert CompletedRunReader.artifact_path(tmp_path,loaded,"signals").name == "signals.parquet"
    (tmp_path/"artifacts/signals.parquet").write_bytes(b"tampered")
    with pytest.raises(RunArtifactError,match="integrity"):
        CompletedRunReader.artifact_path(tmp_path,loaded,"signals")


def test_only_controller_constructs_research_runner():
    gui_source=ACTIVE[0].read_text(); controller_source=ACTIVE[1].read_text()
    assert "ResearchRunner" not in gui_source
    assert "ResearchRunner(" in controller_source
    assert 'run_manifest.json").write' not in gui_source + controller_source


def _qt_window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from crypto_strategy_lab.gui.v2_main_window import MainWindow
    app = widgets.QApplication.instance() or widgets.QApplication([])
    class Catalog:
        def symbols(self): return ["BTCUSDT"]
        def coverage(self, _request): return []
    class Service:
        catalog=Catalog()
        def refresh_catalog(self): return 0
    return app, MainWindow(service=Service())


def test_main_window_constructs_offscreen_and_timeframes_roundtrip():
    _app, window = _qt_window()
    try:
        from crypto_strategy_lab.gui.v2_main_window import timeframe_label, timeframe_minutes
        assert {label: timeframe_minutes(label) for label in ("15m","1h","4h","1d")} == {
            "15m":15,"1h":60,"4h":240,"1d":1440}
        assert [timeframe_label(value) for value in (15,60,240,1440)] == ["15m","1h","4h","1d"]
        for label in ("15m","1h","4h","1d"):
            config=ResearchRunConfig(data=replace(ResearchRunConfig().data,
                strategy_timeframe_minutes=timeframe_minutes(label), use_intrabar_data=False))
            window.apply_config(config)
            assert window.build_config() == config
    finally: window.close()


def test_research_oriented_visible_groups_and_unique_native_fields():
    _app, window = _qt_window()
    try:
        from crypto_strategy_lab.gui.v2_main_window import (EXECUTION_GROUPS,
            EXECUTION_PROFILE_GROUPS, FEATURE_GROUPS, STRATEGY_GROUPS, STRATEGY_PROFILE_GROUPS)
        assert window.feature_form.section_titles == ("Price / Volatility", "DI", "Mean Reversion",
            "Regime", "Support / Resistance", "Open Interest", "Funding", "Positioning",
            "Taker Flow", "Trade Flow", "Order Book")
        execution = set(window.execution_form.section_titles) | set(
            window.profile_editor.execution_form.section_titles)
        assert {"Risk", "Stop Loss", "Take Profit", "Break-even", "Trailing", "Partials",
                "Timeout", "Fees", "Slippage", "Tie / Same-bar Policy"} <= execution
        strategy = set(window.strategy_form.section_titles) | {"Profiles"} | set(
            window.profile_editor.strategy_form.section_titles)
        assert {"Profiles", "Direction", "Entry Filters"} <= strategy
        for form in (window.strategy_form, window.feature_form, window.execution_form,
                     window.profile_editor.strategy_form, window.profile_editor.execution_form):
            grouped = [name for _title, names, _note in {
                "StrategyConfig": STRATEGY_GROUPS, "FeatureConfig": FEATURE_GROUPS,
                "ExecutionConfig": EXECUTION_GROUPS, "StrategyProfileConfig": STRATEGY_PROFILE_GROUPS,
                "ExecutionProfileConfig": EXECUTION_PROFILE_GROUPS,
            }[form.cls.__name__] for name in names]
            assert len(grouped) == len(set(grouped)) == len(form.widgets)
    finally: window.close()


def test_golden_timeframe_selection_roundtrips_4h_and_1m():
    _app, window = _qt_window()
    try:
        window.strategy_tf.setCurrentText("4h"); window.intrabar_tf.setCurrentText("1m")
        config = window.build_config()
        assert config.data.strategy_timeframe_minutes == 240
        assert config.data.intrabar_timeframe_minutes == 1
        assert config.data.use_intrabar_data is True
        window.apply_config(config)
        assert window.strategy_tf.currentText() == "4h" and window.intrabar_tf.currentText() == "1m"
        assert window.build_config() == config
    finally: window.close()


def test_full_native_config_and_profiles_survive_gui_roundtrip():
    _app, window = _qt_window()
    try:
        base=ResearchRunConfig()
        rules=({"indicator":"ADX","operator":">=","value":23.5,"action":"REJECT","enabled":True},)
        strategy_profiles=dict(base.strategy.profiles); execution_profiles=dict(base.execution.profiles)
        strategy_profiles["bull_long"]=replace(strategy_profiles["bull_long"],enabled=False,
            flip_direction=True,entry_rules=rules,flip_rule_match_mode="ALL",reject_rule_match_mode="ALL",
            rsi_period=base.features.mean_reversion_rsi_period,momentum_lookback_hours=48)
        execution_profiles["bull_long"]=replace(execution_profiles["bull_long"],reward_risk_ratio=2.25,
            risk_multiplier=.75,partial_stop_enabled=True,partial_profit_enabled=True,trailing_enabled=True,
            break_even_enabled=True,timeout_enabled=True,r_step_trailing_enabled=True,
            atr_checkpoint_tp_extension_enabled=True)
        config=replace(base,
            features=replace(base.features,trade_flow_enabled=True,trade_flow_windows=("1m","1h"),
                order_book_enabled=True,enable_support_resistance_analysis=True),
            strategy=replace(base.strategy,profiles=strategy_profiles,entry_interval=3,
                enable_daily_entry_schedule=True),
            execution=replace(base.execution,profiles=execution_profiles,risk_mode="PERCENT",
                maker_fee=.00123456,tie_policy="OPTIMISTIC"))
        window.apply_config(config)
        assert set(window.strategy_form.widgets) == set(config.strategy.__dataclass_fields__) - {"profiles"}
        assert set(window.feature_form.widgets) == set(config.features.__dataclass_fields__)
        assert set(window.execution_form.widgets) == set(config.execution.__dataclass_fields__) - {"profiles"}
        primary_profile_fields = set(StrategyProfileConfig.__dataclass_fields__) - {
            "enabled", "entry_rules", "rsi_period", "momentum_lookback_hours"
        }
        assert set(window.profile_editor.strategy_form.widgets) == primary_profile_fields
        assert set(window.profile_editor.native_calculation_widgets) == {
            "rsi_period", "momentum_lookback_hours"
        }
        assert set(window.profile_editor.permission_checks) == set(config.strategy.profiles)
        assert window.profile_editor.permission_checks["bull_long"].isChecked() is False
        assert set(window.profile_editor.execution_form.widgets) == set(ExecutionProfileConfig.__dataclass_fields__)
        assert window.build_config() == config
        assert window.build_config().strategy.profiles["bull_long"].entry_rules == rules
    finally: window.close()


def test_apply_config_does_not_corrupt_current_profile():
    _app, window = _qt_window()
    try:
        base=ResearchRunConfig(); profiles=dict(base.strategy.profiles)
        profiles["bear_short"]=replace(profiles["bear_short"],enabled=False,flip_direction=True)
        config=replace(base,strategy=replace(base.strategy,profiles=profiles))
        window.profile_editor.selector.setCurrentText("bear_short")
        window.apply_config(config)
        assert window.build_config().strategy.profiles == profiles
    finally: window.close()


def test_data_quality_completion_uses_overall_and_dataset_status():
    _app, window = _qt_window()
    try:
        issue=DataQualityIssue("GAP",DataQualityStatus.WARN,"coverage gap")
        dataset=DatasetQualityReport("klines","BTCUSDT","4h",True,"a","b","a","b","a","b",10,
            None,DataQualityStatus.WARN,(issue,),False)
        window.render_data_quality(DataQualityReport((dataset,)))
        assert window.quality.text() == "Data quality: WARN"
        assert window.quality_table.item(0,4).text() == "WARN"
        assert window.quality_table.item(0,5).text() == "GAP"
        window.render_resolution({"request":{"requested_intrabar_interval":"1m",
            "effective_intrabar_interval":"15m"}})
        assert window.resolution.text() == "Requested: 1m | Effective: 15m | FALLBACK"
    finally: window.close()
