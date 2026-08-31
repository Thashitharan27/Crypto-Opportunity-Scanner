from dataclasses import replace
import hashlib, json
from datetime import datetime, timezone

import pandas as pd
import pytest
import numpy as np

from crypto_strategy_lab.data_lake_config import ResearchRunConfig, ExecutionConfig
from crypto_strategy_lab.historical_strategy_validation import (
    EVERY_VIABLE_ENTRY, STANDARD_SINGLE_SYMBOL, HistoricalStrategyValidator,
    SymbolResearchResult, aggregate_reports, attach_candidate_trades,
    build_validation_data_request, load_validation_config,
    latest_strategy_coverage, validation_execution_config,
)
from crypto_strategy_lab.research_warmup import WARMUP_POLICY_VERSION, validation_warmup_bars
from crypto_strategy_lab.research_adapters import native_simulator_config
from crypto_strategy_lab.features.market_regime import prepare_policy_market_features
from crypto_strategy_lab.support_resistance import SupportResistanceDetector
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind


def candidates(symbols=("SOLUSDT",)):
    return pd.DataFrame({"decision_timestamp":pd.to_datetime(["2025-01-01T00:00Z"]*len(symbols)),
        "final_rank":range(1,len(symbols)+1),"symbol":symbols,"scan_run_id":["scan"]*len(symbols)})


def test_half_open_horizon_neutral_and_incomplete_coverage():
    c=candidates(); trades=pd.DataFrame({"symbol":["SOLUSDT"]*4,"pair_id":[1,2,3,4],
        "entry_time":pd.to_datetime(["2024-12-31T23:59Z","2025-01-01T00:00Z","2025-01-01T12:00Z","2025-01-02T00:00Z"]),
        "pair_net_r":[9,1,0,-1],"side":["LONG"]*4})
    rows,assoc=attach_candidate_trades(c,trades,population=STANDARD_SINGLE_SYMBOL,run_id="run",horizon=pd.Timedelta("24h"),available_through=pd.Timestamp("2025-01-02T00:00Z"))
    assert rows.iloc[0].entry_count == 2 and rows.iloc[0].result == "MULTIPLE"
    assert (rows.iloc[0].wins,rows.iloc[0].losses,rows.iloc[0].neutrals)==(1,0,1)
    assert rows.iloc[0].win_rate_resolved == 1 and rows.iloc[0].win_share_all_completed == .5
    unresolved,_=attach_candidate_trades(c,trades.iloc[0:0],population=STANDARD_SINGLE_SYMBOL,run_id="run",horizon=pd.Timedelta("24h"),available_through=pd.Timestamp("2025-01-01T23:59Z"))
    assert unresolved.iloc[0].result == "UNRESOLVED"


def test_overlap_conversion_counts_windows_unique_trade_once():
    c=pd.concat([candidates(),candidates().assign(decision_timestamp=pd.Timestamp("2025-01-02T00:00Z"))],ignore_index=True)
    trade=pd.DataFrame({"symbol":["SOLUSDT"],"research_sample_id":["sample-1"],"entry_time":pd.to_datetime(["2025-01-03T00:00Z"]),"pair_net_r":[1.0],"side":["LONG"]})
    rows,assoc=attach_candidate_trades(c,trade,population=EVERY_VIABLE_ENTRY,run_id="native",horizon=pd.Timedelta("7D"),available_through=pd.Timestamp("2025-01-10T00:00Z"))
    summary,*_=aggregate_reports(rows,assoc); row=summary.iloc[0]
    assert row.candidate_windows_with_entry == 2 and row.candidate_to_entry_conversion == 1
    assert row.unique_trade_count == row.unique_wins == 1


def test_one_sequential_invocation_per_unique_symbol_and_combined_period(tmp_path):
    symbols=[f"S{i}USDT" for i in range(7)]; c=pd.concat([candidates(symbols)]*15,ignore_index=True).iloc[:100].copy()
    # create distinct decisions without creating duplicate natural identities
    c["decision_timestamp"]=[pd.Timestamp("2025-01-01T00:00Z")+pd.Timedelta(days=i//7) for i in range(100)]
    calls=[]
    def execute(symbol,start,end,config):
        calls.append((symbol,pd.Timestamp(start),pd.Timestamp(end),config.strategy,config.execution))
        return SymbolResearchResult("run-"+symbol,pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"]),pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"]),pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"]),end,tmp_path/symbol,start,end)
    config=ResearchRunConfig(); path=tmp_path/"config.json"; path.write_text("{}")
    result=HistoricalStrategyValidator(execute,warmup_bars=lambda _:10).run(c,config,config_path=path,publish=False)
    assert len(calls)==7 and [x[0] for x in calls]==sorted(symbols)
    assert all(x[3] is config.strategy and x[4] is config.execution for x in calls)
    assert len(result.outcomes)==200 and set(result.outcomes.population)=={STANDARD_SINGLE_SYMBOL,EVERY_VIABLE_ENTRY}
    assert all(end >= c[c.symbol.eq(symbol)].decision_timestamp.max()+pd.Timedelta("24h") for symbol,_,end,_,_ in calls)


def test_eve_censored_entry_is_valid_and_unresolved():
    empty=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
    censored=pd.DataFrame({"entry_time":pd.to_datetime(["2025-01-01T01:00Z"]),
        "pair_net_r":[None],"side":["SHORT"],"research_sample_id":["eod-1"]})
    rows,associations=attach_candidate_trades(candidates(),empty,population=EVERY_VIABLE_ENTRY,
        run_id="real-run",horizon=pd.Timedelta("24h"),available_through=pd.Timestamp("2025-01-02T00:00Z"),censored_trades=censored)
    assert rows.iloc[0].valid_entry and rows.iloc[0].completed_trade_count == 0
    assert rows.iloc[0].result == "UNRESOLVED"
    assert associations.iloc[0].status == "CENSORED" and associations.iloc[0].trade_identity.endswith("eod-1")


def test_timeout_tail_and_immutable_config_snapshot_are_published(tmp_path):
    base=ResearchRunConfig(); execution=replace(base.execution,profiles={
        key:replace(profile,timeout_enabled=True,timeout_minutes=1440 if key==next(iter(base.execution.profiles)) else 60)
        for key,profile in base.execution.profiles.items()})
    config=replace(base,execution=execution); external=tmp_path/"external.json"
    external.write_text(json.dumps(config.to_dict(),default=str))
    calls=[]
    def execute(symbol,start,end,used):
        calls.append((pd.Timestamp(start),pd.Timestamp(end),used.strategy,used.execution))
        # Mutating the external file mid-run must not alter snapshotted provenance.
        external.write_text('{"changed":true}')
        empty=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"])
        viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        return SymbolResearchResult("authoritative-run",empty,viable,viable.copy(),end,tmp_path/"native",start,end,("source-sha",))
    readiness={"symbol":"SOLUSDT","dataset":"klines","interval":"1m","required_role":"INTRABAR","state":"ACQUIRED"}
    result=HistoricalStrategyValidator(execute,warmup_bars=lambda _:5,output_root=tmp_path/"validation",
        preflight=lambda *args,**kwargs:[readiness]).run(
        candidates().assign(scanner_source_identity="scanner-source"),config,config_path=external)
    assert calls[0][1] == pd.Timestamp("2025-01-03T00:00Z")
    snapshot=(result.run_dir/"strategy_config_snapshot.json").read_bytes()
    assert hashlib.sha256(snapshot).hexdigest()==result.manifest["strategy_config_sha256"]
    assert b'"changed"' not in snapshot
    assert result.manifest["native_research_run_ids_by_symbol"]["SOLUSDT"]=="authoritative-run"
    assert result.manifest["authoritative_research_config_sha256"]==result.manifest["strategy_config_sha256"]
    assert result.manifest["validation_warmup_policy_version"]==WARMUP_POLICY_VERSION
    assert result.manifest["validation_warmup_bars"]==5
    assert result.manifest["native_research_runs_by_symbol"]["SOLUSDT"]["request_start"]==pd.Timestamp(calls[0][0]).isoformat()
    assert result.manifest["scanner_candidate_sources"][0]["scanner_source_identity"]=="scanner-source"
    association=result.run_dir/"strategy_validation_trade_associations.csv"
    assert association.exists() and result.manifest["artifacts"][association.name]["rows"]==0
    readiness_path=result.run_dir/"strategy_validation_data_readiness.csv"
    assert readiness_path.exists() and result.manifest["artifacts"][readiness_path.name]["rows"]==1


def test_timeout_disabled_uses_latest_coverage_without_forcing_exit(tmp_path):
    seen=[]; latest=pd.Timestamp("2025-02-01T00:00Z")
    def execute(symbol,start,end,config):
        seen.append(pd.Timestamp(end)); empty=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"]); viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        return SymbolResearchResult("run",empty,viable,viable.copy(),latest,tmp_path/"native",start,end)
    HistoricalStrategyValidator(execute,warmup_bars=lambda _:0,latest_available=lambda _:latest).run(candidates(),ResearchRunConfig(),config_path=tmp_path/"unused",publish=False)
    assert seen==[latest]


def test_strict_v3_load_and_production_request_projection(tmp_path):
    config=ResearchRunConfig(); path=tmp_path/"config.json"; path.write_text(json.dumps(config.to_dict()))
    loaded=load_validation_config(path); assert loaded==config
    bad=tmp_path/"bad.json"; bad.write_text('{"config_version":3,"unknown":1}')
    import pytest
    with pytest.raises(ValueError,match="Unknown"): load_validation_config(bad)
    registry=type("Registry",(),{"names":lambda self:("technical",),"required_datasets":lambda self,names:(DatasetKind.FUNDING_RATE,)})()
    request=build_validation_data_request("btcusdt",datetime(2025,1,1,tzinfo=timezone.utc),datetime(2025,1,2,tzinfo=timezone.utc),loaded,registry)
    assert request.market==MarketKind.FUTURES_UM and request.strategy_interval=="15m" and request.intrabar_interval=="1m"
    assert request.datasets==(DatasetKind.KLINES,)
    assert request.exchange=="binance"


def test_native_interval_aliases_find_actual_catalog_coverage():
    rows=[{"symbol":"BTCUSDT","dataset":"klines","interval":interval,
           "last_period":pd.Timestamp(end)} for interval,end in
          (("15m","2025-01-02T00:15Z"),("1h","2025-01-02T01:00Z"),
           ("4h","2025-01-02T04:00Z"),("1d","2025-01-03T00:00Z"))]
    expected={15:"2025-01-02T00:15Z",60:"2025-01-02T01:00Z",240:"2025-01-02T04:00Z",1440:"2025-01-03T00:00Z"}
    for minutes,value in expected.items():
        assert pd.Timestamp(latest_strategy_coverage(rows,"BTCUSDT",minutes))==pd.Timestamp(value)


def test_catalog_coverage_uses_partition_end_without_inventing_a_candle():
    rows=[{"symbol":"BTCUSDT","dataset":"klines","interval":"4h","last_period":value}
          for value in ("2025-01-01T04:00Z","2025-01-03T00:00Z","2025-01-02T20:00Z")]
    assert pd.Timestamp(latest_strategy_coverage(rows,"BTCUSDT",240))==pd.Timestamp("2025-01-03T00:00Z")
    assert latest_strategy_coverage(rows,"ETHUSDT",240) is None
    partial=[{"symbol":"BTCUSDT","dataset":"klines","interval":"1d","last_period":"2025-01-02T12:00Z"}]
    assert pd.Timestamp(latest_strategy_coverage(partial,"BTCUSDT",1440))==pd.Timestamp("2025-01-02T12:00Z")


@pytest.mark.parametrize("minutes",[240,1440])
def test_configured_warmup_covers_asset_return_and_higher_timeframe_sr(minutes):
    base=ResearchRunConfig()
    data=replace(base.data,strategy_timeframe_minutes=minutes,
        intrabar_timeframe_minutes=60 if minutes==240 else 240)
    features=replace(base.features,market_regime_method="ASSET_RETURN",
        bull_regime_lookback_days=120,enable_support_resistance_analysis=True,
        sr_timeframe_minutes=1440,sr_lookback_bars=80,sr_pivot_left=7,
        sr_pivot_right=9,enable_sr_hold_confirmation=True,sr_hold_confirmation_bars=4)
    config=replace(base,data=data,features=features)
    registry=type("Registry",(),{"names":lambda self:("core",),"effective_warmup":lambda self,names:10})()
    bars=validation_warmup_bars(config,registry)
    duration=pd.Timedelta(minutes=bars*minutes)
    assert duration>=pd.Timedelta(days=124)  # 120d regime + mature safety margin
    assert duration>=pd.Timedelta(minutes=(80+7+9+4+5)*1440)


def test_partial_catalog_end_keeps_candidate_unresolved():
    coverage=latest_strategy_coverage([{"symbol":"SOLUSDT","dataset":"klines","interval":"1h",
        "last_period":"2025-01-01T23:00Z"}],"SOLUSDT",60)
    empty=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"])
    rows,_=attach_candidate_trades(candidates(),empty,population=STANDARD_SINGLE_SYMBOL,
        run_id="native",horizon=pd.Timedelta("24h"),available_through=coverage)
    assert rows.iloc[0].entry_horizon_status=="INSUFFICIENT_FUTURE_DATA"
    assert rows.iloc[0].result=="UNRESOLVED"


def test_asset_return_context_at_first_candidate_matches_long_native_history():
    base=ResearchRunConfig(); config=replace(base,
        data=replace(base.data,strategy_timeframe_minutes=240,intrabar_timeframe_minutes=60),
        features=replace(base.features,market_regime_method="ASSET_RETURN",bull_regime_lookback_days=90))
    registry=type("Registry",(),{"names":lambda self:("policy_market_context",),"effective_warmup":lambda self,names:0})()
    warmup=validation_warmup_bars(config,registry)
    times=pd.date_range("2024-01-01",periods=1400,freq="4h",tz="UTC")
    closes=np.linspace(100,180,len(times)); candidate_index=1300
    native=native_simulator_config(config.data,config.features,config.strategy,config.execution)
    _,long_regime,_=prepare_policy_market_features(times[:candidate_index+1],closes[:candidate_index+1],native)
    start=candidate_index-warmup
    _,validation_regime,_=prepare_policy_market_features(times[start:candidate_index+1],closes[start:candidate_index+1],native)
    assert validation_regime[-1]==long_regime[-1]


def test_sr_context_at_first_candidate_matches_long_native_history():
    base=ResearchRunConfig(); config=replace(base,
        data=replace(base.data,strategy_timeframe_minutes=1440,intrabar_timeframe_minutes=240),
        features=replace(base.features,enable_support_resistance_analysis=True,
            sr_timeframe_minutes=1440,sr_lookback_bars=80,sr_pivot_left=4,sr_pivot_right=4,
            enable_sr_hold_confirmation=True,sr_hold_confirmation_bars=3))
    registry=type("Registry",(),{"names":lambda self:("support_resistance",),"effective_warmup":lambda self,names:0})()
    warmup=validation_warmup_bars(config,registry); count=220; x=np.arange(count)
    close=100+8*np.sin(x/7); open_=close-.2; high=close+1; low=close-1; atr=np.full(count,2.0)
    kwargs=dict(pivot_left=4,pivot_right=4,lookback_bars=80,enable_hold_confirmation=True,hold_confirmation_bars=3)
    long=SupportResistanceDetector(**kwargs).analyze_price_location(count-1,open_,high,low,close,atr,"LONG")
    start=max(0,count-1-warmup)
    validation=SupportResistanceDetector(**kwargs).analyze_price_location(count-1-start,open_[start:],high[start:],low[start:],close[start:],atr[start:],"LONG")
    assert validation.nearest_support_price==pytest.approx(long.nearest_support_price)
    assert validation.nearest_resistance_price==pytest.approx(long.nearest_resistance_price)
    assert validation.support_state==long.support_state and validation.resistance_state==long.resistance_state


def test_standard_end_of_data_is_auditable_and_unresolved_denominator_counts_it():
    c=candidates().assign(scanner_source_identity="scanner-source")
    trades=pd.DataFrame({"symbol":["SOLUSDT"],"pair_id":["pair-7"],
        "entry_time":pd.to_datetime(["2025-01-01T02:00Z"]),"pair_net_r":[-1.0],
        "side":["LONG"],"long_exit_reason":["END_OF_DATA"]})
    rows,associations=attach_candidate_trades(c,trades,population=STANDARD_SINGLE_SYMBOL,
        run_id="native-run",horizon=pd.Timedelta("24h"),available_through=pd.Timestamp("2025-01-02T00:00Z"))
    row=rows.iloc[0]
    assert row.entry_horizon_status=="COMPLETE" and row.outcome_resolution_status=="UNRESOLVED"
    assert row.valid_entry and row.result=="UNRESOLVED" and row.side=="LONG"
    assert row.first_entry_time==row.last_entry_time==pd.Timestamp("2025-01-01T02:00Z")
    association=associations.iloc[0]
    assert association.status=="CENSORED" and association.trade_identity.endswith("pair-7")
    assert association.scan_run_id=="scan" and association.scanner_source_identity=="scanner-source"
    summary,*_=aggregate_reports(rows,associations)
    assert summary.iloc[0].unresolved_candidate_windows==1
    assert summary.iloc[0].candidate_to_entry_conversion==1


def test_validation_reporting_override_preserves_strategy_and_execution_identity():
    base=ResearchRunConfig(); source=replace(base,reporting=replace(base.reporting,
        research_sampling_mode="EVERY_VIABLE_ENTRY",analysis_level="DEEP",
        enable_trade_telemetry=True,telemetry_interval_minutes=15))
    effective=validation_execution_config(source)
    assert effective.strategy is source.strategy and effective.execution is source.execution
    assert effective.reporting.research_sampling_mode=="PORTFOLIO"
    assert effective.reporting.analysis_level=="STANDARD" and not effective.reporting.enable_trade_telemetry


def test_source_eve_deep_reporting_is_overridden_once_per_symbol(tmp_path):
    base=ResearchRunConfig(); source=replace(base,reporting=replace(base.reporting,
        research_sampling_mode="EVERY_VIABLE_ENTRY",analysis_level="DEEP"))
    calls=[]
    def execute(symbol,start,end,effective):
        calls.append(effective)
        empty=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"])
        viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        return SymbolResearchResult("native",empty,viable,viable.copy(),end,tmp_path/symbol,start,end)
    HistoricalStrategyValidator(execute,warmup_bars=lambda _:0).run(candidates(("SOLUSDT","BTCUSDT")),source,config_path=tmp_path/"config",publish=False)
    assert len(calls)==2
    assert all(c.reporting.research_sampling_mode=="PORTFOLIO" and c.reporting.analysis_level=="STANDARD" for c in calls)
    assert all(c.strategy is source.strategy and c.execution is source.execution for c in calls)


def test_cancellation_prevents_next_symbol_and_never_publishes_complete(tmp_path):
    calls=[]; cancelled=[False]
    def execute(symbol,start,end,config):
        calls.append(symbol); cancelled[0]=True
        empty=pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"])
        viable=pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"])
        return SymbolResearchResult("native",empty,viable,viable.copy(),end,tmp_path/symbol,start,end)
    validator=HistoricalStrategyValidator(execute,warmup_bars=lambda _:0,output_root=tmp_path/"output")
    from crypto_strategy_lab.historical_strategy_validation import ValidationCancelled
    with pytest.raises(ValidationCancelled):
        validator.run(candidates(("SOLUSDT","BTCUSDT")),ResearchRunConfig(),config_path=tmp_path/"config",cancelled=lambda:cancelled[0])
    assert len(calls)==1 and not (tmp_path/"output").exists()


def test_required_data_preflight_failure_prevents_native_run_and_publication(tmp_path):
    calls=[]
    def blocked(*args,**kwargs): raise RuntimeError("Required data unavailable: 1m KLINES")
    validator=HistoricalStrategyValidator(lambda *args:calls.append(args),warmup_bars=lambda _:0,
        output_root=tmp_path/"output",preflight=blocked)
    with pytest.raises(RuntimeError,match="1m KLINES"):
        validator.run(candidates(),ResearchRunConfig(),config_path=tmp_path/"config")
    assert calls==[] and not (tmp_path/"output").exists()
