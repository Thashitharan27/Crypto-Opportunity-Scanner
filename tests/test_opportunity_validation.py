from __future__ import annotations

import json

import pandas as pd
import pytest

from crypto_strategy_lab.opportunity_validation import (
    OpportunityValidationConfig,
    join_research_results,
    validate_opportunities,
)
from crypto_strategy_lab.opportunity_validation_reporting import write_validation_reports


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


def test_join_enforces_natural_key_and_causal_entry_boundary():
    scanner=pd.DataFrame({"decision_timestamp":["2025-01-01T00:00:00Z"],"symbol":["btc"]})
    research=pd.DataFrame({"decision_timestamp":["2025-01-01T00:00:00Z"],"symbol":["BTC"],
        "valid_entry":[True],"forward_max_abs_excursion_pct":[.1],"trade_expectancy_value":[2.0],
        "market_regime":["SIDEWAYS"],"entry_timestamp":["2024-12-31T23:00:00Z"]})
    with pytest.raises(ValueError,match="predates scan decision"):
        join_research_results(scanner,research)
    research.entry_timestamp="2025-01-01T01:00:00Z"
    joined=join_research_results(scanner,research)
    assert joined.loc[0,"symbol"]=="BTC" and joined.loc[0,"year"]==2025
    with pytest.raises(ValueError,match="duplicate natural research key"):
        join_research_results(scanner,pd.concat([research,research]))


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


def test_constant_or_insufficient_association_is_explicit_null():
    one=_observations().iloc[:1].copy()
    result=validate_opportunities(one)
    assert result.associations.spearman_rho.isna().all()
