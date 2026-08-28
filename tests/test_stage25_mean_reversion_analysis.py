import os
import sys

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.gui.config_logic import build_backtest_config, default_gui_config
from crypto_strategy_lab.mean_reversion import (
    DI_PRESSURE_BUCKETS,
    classify_alignment,
    classify_motion,
    classify_state,
    classify_strength,
    di_pressure_bucket,
    distance_from_mean_atr,
    ema,
)
from crypto_strategy_lab.statistics import mean_reversion_analysis


def test_mean_reversion_classification_keeps_location_motion_and_alignment_separate():
    assert classify_state(-1.6) == "STRONGLY_BELOW_MEAN"
    assert classify_state(1.6) == "STRONGLY_ABOVE_MEAN"
    assert classify_motion(-1.2, -1.8) == "TOWARD_MEAN"
    assert classify_motion(-1.8, -1.2) == "AWAY_FROM_MEAN"
    assert classify_alignment(-1.2, "LONG") == "FAVORS_REVERSION"
    assert classify_alignment(-1.2, "SHORT") == "AGAINST_REVERSION"
    assert classify_alignment(0.2, "LONG") == "NEUTRAL"
    assert classify_strength(1.7) == (3, "STRONG")


def test_mean_reversion_ema_and_atr_distance_are_causal_and_raw():
    close=np.array([10.,11.,12.,13.,14.,15.])
    mean=ema(close,3)
    assert np.isnan(mean[0]) and np.isnan(mean[1])
    atr=np.ones(len(close))*2
    distance=distance_from_mean_atr(close,mean,atr)
    assert np.isfinite(distance[-1])
    assert distance[-1] == pytest.approx((close[-1]-mean[-1])/2)


def test_di_pressure_buckets_cover_full_range_without_hard_coded_cutoff():
    assert DI_PRESSURE_BUCKETS[0] == (0,5)
    assert DI_PRESSURE_BUCKETS[-1] == (50,None)
    assert di_pressure_bucket(2.5) == "0-5"
    assert di_pressure_bucket(27.0) == "25-30"
    assert di_pressure_bucket(42.0) == "40-45"
    assert di_pressure_bucket(80.0) == "50+"


def test_mean_reversion_report_includes_all_di_buckets_present_in_data():
    rows=[]
    for di, alignment, motion, pnl in [
        (2.0,"FAVORS_REVERSION","TOWARD_MEAN",1.0),
        (27.0,"AGAINST_REVERSION","AWAY_FROM_MEAN",-1.0),
        (42.0,"FAVORS_REVERSION","TOWARD_MEAN",2.0),
        (55.0,"NEUTRAL","FLAT",0.2),
    ]:
        rows.append({"di_spread":di,"di_sizing_direction":"LONG","market_regime":"BULL","di_pressure_state":"EXPANDING","mean_reversion_alignment":alignment,"mean_reversion_motion":motion,"mean_reversion_state":"BELOW_MEAN","mean_reversion_strength_label":"STRONG","pair_net_pnl":pnl,"pair_net_r":pnl})
    report=mean_reversion_analysis(pd.DataFrame(rows))
    primary=report[report["Section"].eq("DI Bucket + Reversion")]
    assert {"0-5","25-30","40-45","50+"}.issubset(set(primary["DI Pressure Bucket"].astype(str)))
    assert not any("<30" in str(v) for v in report.astype(str).to_numpy().ravel())


def test_config_exposes_mean_reversion_as_analysis_settings():
    values=default_gui_config()
    assert values["enable_mean_reversion_analysis"] is True
    assert values["mean_reversion_period"] == 20
    cfg=build_backtest_config(values,require_paths=False)
    assert cfg.enable_mean_reversion_analysis is True
    assert cfg.mean_reversion_period == 20


def test_mean_reversion_toggle_changes_telemetry_not_di_direction():
    n=30
    candles=pd.DataFrame({
        "timestamp":pd.date_range("2024-01-01",periods=n,freq="15min",tz="UTC"),
        "open":np.arange(n,dtype=float)+100,
        "high":np.arange(n,dtype=float)+102,
        "low":np.arange(n,dtype=float)+99,
        "close":np.arange(n,dtype=float)+101,
        "volume":1.0,
    })
    enabled=BacktestEngine(candles,BacktestConfig(adx_period=2,mean_reversion_period=5,enable_mean_reversion_analysis=True))
    disabled=BacktestEngine(candles,BacktestConfig(adx_period=2,mean_reversion_period=5,enable_mean_reversion_analysis=False))
    for engine in (enabled,disabled):
        engine.plus_di_values[:]=30
        engine.minus_di_values[:]=10
        engine.di_spread[:]=20
    i=n-1
    assert enabled._selected_direction(i)==disabled._selected_direction(i)=="LONG"
    assert enabled._mean_reversion_snapshot(i,"LONG")["mean_reversion_state"] != "UNKNOWN"
    assert disabled._mean_reversion_snapshot(i,"LONG")["mean_reversion_state"] == "UNKNOWN"


def test_di_tab_exposes_record_only_mean_reversion_controls():
    qtwidgets=pytest.importorskip("PySide6.QtWidgets",exc_type=ImportError)
    from crypto_strategy_lab.gui.main_window import MainWindow
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    app=qtwidgets.QApplication.instance() or qtwidgets.QApplication(sys.argv)
    window=MainWindow()
    try:
        assert window.enable_mean_reversion_analysis.text()=="Analyze mean reversion"
        assert window.mean_reversion_period.value()==20
        idx=[window.tabs.tabText(i) for i in range(window.tabs.count())].index("DI Direction & Pressure")
        page=window.tabs.widget(idx)
        texts=[w.text() for w in page.findChildren(qtwidgets.QLabel)]
        assert any("RECORD ONLY" in text for text in texts)
        assert any("no DI cutoff is hard-coded" in text for text in texts)
        values=window.values()
        assert values["enable_mean_reversion_analysis"] is True
        assert values["mean_reversion_period"]==20
    finally:
        window.close()
