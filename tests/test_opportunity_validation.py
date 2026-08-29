from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from crypto_strategy_lab.opportunity_validation import (
    OpportunityValidationConfig,
    StrategyResearchSource,
    join_research_results,
    load_scanner_observations,
    validate_opportunities,
)
from crypto_strategy_lab.opportunity_validation_reporting import write_validation_reports
from crypto_strategy_lab.opportunity_scoring import OpportunityOutcome


def _observations() -> pd.DataFrame:
    rows=[]
    for year, regime, offset in ((2024,"BULL",0),(2025,"BEAR",4)):
        decision=pd.Timestamp(f"{year}-01-01T00:00:00Z")
        for rank in range(1,5):
            rows.append({"decision_timestamp":decision,"symbol":f"S{offset+rank}","eligible":True,
                "preliminary":rank<=3,"final":rank<=3,"final_rank":rank if rank<=3 else None,
                "valid_entry":rank in (1,3),"absolute_movement":float(6-rank),
                "trade_expectancy_value":2.0 if rank==1 else -1.0 if rank==3 else None,
                "trade_won":rank==1,"market_regime":regime,"year":year,
                "scan_run_id":f"scan-{year}", "scanner_source_identity":f"scanner-{year}",
                "research_run_id":"research-1", "research_source_identities":'["research-source"]',
                "evaluation_horizon":"1 days 00:00:00",
                "score__legacy_volatility_v1":float(6-rank),"score__balanced_activity_v1":float(rank),
                "component__atr_pct":float(rank),"component__realized_volatility":float(rank*2)})
    return pd.DataFrame(rows)


def test_calculations_top_k_rank_decay_and_stability():
    result=validate_opportunities(_observations(),OpportunityValidationConfig(top_ks=(2,5),rank_bucket_size=2,minimum_sample_warning=10))
    final=result.overall[result.overall.selection=="final_candidates"].iloc[0]
    assert final.sample_count==6
    assert final.candidate_to_entry_conversion==pytest.approx(4/6)
    assert final.expectancy==pytest.approx(.5)
    assert final.opportunity_capture==pytest.approx(24/28)
    assert list(result.top_k.top_k)==[2,5]
    assert list(result.top_k.max_rank_present)==[2,3]
    assert list(result.rank_decay.rank_bucket)==["1-2","3-4"]
    assert list(result.by_year.year.unique())==[2024,2025]
    assert list(result.by_regime.market_regime.unique())==["BEAR","BULL"]
    rank_move=result.associations.query("predictor == 'final_rank' and outcome == 'absolute_movement'").iloc[0]
    assert rank_move.sample_count==6 and rank_move.spearman_rho==pytest.approx(-1)
    assert result.components.iloc[0].spearman_rho==pytest.approx(1)


def test_ordering_is_deterministic_and_missing_ranks_are_not_fabricated():
    source=_observations().sample(frac=1,random_state=4)
    result=validate_opportunities(source,OpportunityValidationConfig(top_ks=(5,10)))
    assert list(result.observations.symbol)==["S1","S2","S3","S4","S5","S6","S7","S8"]
    assert list(result.by_rank["rank"])==[1,2,3]
    assert all(result.top_k.max_rank_present==3)


def test_native_research_adapter_uses_entry_horizon_and_native_expectancy():
    scanner=pd.DataFrame({"decision_timestamp":["2025-01-01T00:00:00Z"]*2,
        "symbol":["btc","eth"], "market_regime":["SIDEWAYS","BEAR"]})
    decision=datetime(2025,1,1,tzinfo=timezone.utc)
    outcomes=[OpportunityOutcome(symbol,decision,100.0,.3,movement,.1,1.0)
        for symbol,movement in (("BTC",.1),("ETH",.2))]
    trades=pd.DataFrame({"entry_time":["2025-01-01T23:59:00Z","2025-01-02T00:00:00Z"],
        "pair_net_r":[2.0,99.0],"market_regime":["BULL","BULL"]})
    source=StrategyResearchSource("research-run","24h",trades,outcomes,("lake-b","lake-a"),symbol="BTC")
    joined=join_research_results(scanner,source)
    assert joined.valid_entry.tolist()==[True,False]
    assert joined.trade_expectancy_value.tolist()[0]==2.0
    assert pd.isna(joined.trade_expectancy_value.tolist()[1])
    assert joined.market_regime.tolist()==["BULL","BEAR"]
    assert joined.research_run_id.dropna().unique().tolist()==["research-run"]
    assert joined.evaluation_horizon.dropna().unique().tolist()==["1 days 00:00:00"]


def test_empty_insufficient_and_reports(tmp_path):
    empty=validate_opportunities(pd.DataFrame())
    assert empty.overall.empty
    result=validate_opportunities(_observations(),OpportunityValidationConfig(minimum_sample_warning=100))
    assert result.overall.small_sample_warning.all()
    write_validation_reports(tmp_path,result)
    expected={"validation_overall.csv","validation_by_rank.csv","validation_top_k.csv",
        "validation_rank_decay.csv","validation_by_year.csv","validation_by_regime.csv",
        "validation_components.csv","validation_summary.json"}
    assert expected=={path.name for path in tmp_path.iterdir()}
    summary=json.loads((tmp_path/"validation_summary.json").read_text())
    assert summary["natural_key"]==["decision_timestamp","symbol"]
    assert "winning" not in summary["definitions"]["opportunity_capture"]
    assert summary["scan_run_ids"]==["scan-2024","scan-2025"]
    assert summary["research_run_ids"]==["research-1"]
    assert summary["research_source_identities"]==["research-source"]


def test_missing_entry_coverage_denominator_is_visible_in_every_table():
    observations=_observations()
    observations["valid_entry"]=observations.valid_entry.astype("boolean")
    observations.loc[observations.symbol.isin(["S2","S6"]),"valid_entry"]=pd.NA
    result=validate_opportunities(observations)
    assert result.overall.entry_observation_count.tolist()==[6,4,4]
    assert "entry_observation_count" in result.by_rank
    assert "entry_observation_count" in result.top_k
    assert "entry_observation_count" in result.by_year
    assert "entry_observation_count" in result.by_regime


def test_loader_preserves_discovery_components_and_reports_redundancy(tmp_path,monkeypatch):
    universe=pd.DataFrame({"symbol":["AAA","BBB"],"eligible":[True,True],"discovery_rank":[1,2],
        "reference_available_at":["2024-12-31T23:00:00Z"]*2,"discovery_source_identity":["a","b"],
        "range_percent":[1.0,2.0],"absolute_price_change_percent":[2.0,4.0],
        "quote_volume":[10.0,20.0],"spread_percent":[.1,.2]})
    universe.to_csv(tmp_path/"universe_snapshot.csv",index=False)
    pd.DataFrame({"symbol":["AAA","BBB"],"strategy_source_identity":["sa","sb"]}).to_csv(tmp_path/"preliminary_candidates.csv",index=False)
    pd.DataFrame({"symbol":["AAA","BBB"],"final_rank":[1,2]}).to_csv(tmp_path/"final_candidates.csv",index=False)
    manifest={"run_type":"OPPORTUNITY_SCAN","run_id":"scan-native","opportunity_scan":{
        "discovery_mode":"HISTORICAL","decision_timestamp":"2025-01-01T00:00:00Z"},
        "artifacts":{"universe_snapshot":{},"preliminary_candidates":{},"final_candidates":{}}}
    monkeypatch.setattr("crypto_strategy_lab.opportunity_validation.load_completed_manifest",lambda _:manifest)
    monkeypatch.setattr("crypto_strategy_lab.opportunity_validation.artifact_path",lambda directory,manifest,name:directory/f"{name}.csv")
    loaded=load_scanner_observations([tmp_path])
    assert loaded["discovery__range_percent"].tolist()==[1.0,2.0]
    assert loaded["discovery__quote_volume"].tolist()==[10.0,20.0]
    loaded=loaded.assign(valid_entry=[False,True],absolute_movement=[1.0,2.0],
        trade_expectancy_value=[None,1.0],trade_won=[None,True],market_regime=["BEAR","BULL"],year=2025)
    components=validate_opportunities(loaded).components
    pair=components[(components.analysis_type=="REDUNDANCY") &
        (components.component_a=="discovery__absolute_price_change_percent") &
        (components.component_b=="discovery__quote_volume")].iloc[0]
    assert pair.sample_count==2 and pair.spearman_rho==pytest.approx(1)


def test_constant_or_insufficient_association_is_explicit_null():
    one=_observations().iloc[:1].copy()
    result=validate_opportunities(one)
    assert result.associations.spearman_rho.isna().all()
