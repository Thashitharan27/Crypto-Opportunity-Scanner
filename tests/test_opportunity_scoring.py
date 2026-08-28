from datetime import datetime, timezone
import random
import pandas as pd

from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.features.opportunity_activity import OpportunityActivityFeatureProvider
from crypto_strategy_lab.opportunity_scoring import *

T=datetime(2025,1,3,tzinfo=timezone.utc)
SIG=SourceSignature(DatasetKind.KLINES,"same",1)

def snap(symbol,rank,offset=0,missing=False):
    values={"realized_volatility":1+offset,"atr":2,"atr_pct":2+offset,"recent_range_pct":3+offset,
        "range_expansion":4+offset,"volume_ratio":None if missing else 5+offset,"adx":6+offset,
        "di_spread":7+offset,"di_spread_1":6+offset,"momentum_return_24h":-(8+offset),"market_regime":"SIDEWAYS"}
    return OpportunityFeatureSnapshot(symbol,rank,T,"1h",T,T,values,SIG,{"core_directional":"1","policy_market_context":"1","opportunity_activity":"1"})

def test_exact_legacy_weights_and_ordinal_ties():
    assert [c.weight for c in LEGACY_VOLATILITY_V1.components]==[.30,.25,.20,.10,.15]
    result=score_opportunities([snap("B",2),snap("A",1)],T,OpportunityScoringConfig(models=(LEGACY_VOLATILITY_V1,)))
    # Equal values deliberately retain discovery-rank/symbol input order, unlike alternatives.
    assert [r.score for r in result.rows]==[0.0,1.0]

def test_alternative_ties_order_and_missing_are_stable():
    source=[snap("B",2),snap("A",1),snap("C",3,missing=True)]
    expected=None
    for _ in range(5):
        random.shuffle(source); rows=score_opportunities(source,T,OpportunityScoringConfig(models=(BALANCED_ACTIVITY_V1,))).rows
        current=[(r.symbol,r.score,r.model_rank,r.status) for r in rows]
        expected=expected or current; assert current==expected
    assert expected[:2]==[("A",.5,1,ScoreStatus.SCORABLE),("B",.5,2,ScoreStatus.SCORABLE)]
    assert expected[2][3]==ScoreStatus.UNSCORABLE

def test_decision_boundary_and_cross_timestamp_isolation():
    other=snap("Z",1,99); other=OpportunityFeatureSnapshot(other.symbol,other.discovery_rank,datetime(2025,1,4,tzinfo=timezone.utc),"1h",other.feature_timestamp,other.available_at,other.values,SIG)
    base=score_opportunities([snap("A",1),snap("B",2,1)],T)
    assert score_opportunities([snap("A",1),snap("B",2,1),other],T)==base

def test_activity_uses_quote_volume_and_future_rows_are_clipped():
    times=pd.date_range("2025-01-01",periods=51,freq="h",tz="UTC")
    frame=pd.DataFrame({"period_start":times,"available_at":times+pd.Timedelta(hours=1),"high":range(101,152),"low":range(99,150),"close":range(100,151),"quote_volume":[10.]*51})
    provider=OpportunityActivityFeatureProvider(); params={"window":24,"denominator_floor":1e-12}
    from crypto_strategy_lab.data.query import DataRequest
    req=DataRequest("X",times[0].to_pydatetime(),(times[-1]+pd.Timedelta(hours=1)).to_pydatetime(),"1h")
    features=provider.compute(req,{DatasetKind.KLINES:frame},params)
    assert features.iloc[-1].volume_ratio==1
    causal=latest_causal_snapshot("X",1,times[-2].to_pydatetime(),"1h",features,SIG)
    changed=features.copy(); changed.loc[50,"realized_volatility"]=999
    assert latest_causal_snapshot("X",1,times[-2].to_pydatetime(),"1h",changed,SIG)==causal
    assert not hasattr(score_opportunities([snap("A",1)],T).rows[0],"side")

def test_future_outcome_changes_not_frozen_score():
    row=score_opportunities([snap("A",1)],T).rows[0]
    future=pd.DataFrame({"period_start":[pd.Timestamp(T)+pd.Timedelta(hours=1)],"high":[110],"low":[90],"close":[105]})
    evaluator=HistoricalOpportunityEvaluator(); first=evaluator.evaluate(row,future,100)
    future["high"]=120
    second=evaluator.evaluate(row,future,100)
    assert first!=second
    assert row==score_opportunities([snap("A",1)],T).rows[0]

def test_source_identity_is_canonical_signature():
    assert snap("A",1).source_signature.cache_identity()==snap("B",2).source_signature.cache_identity()
    changed=SourceSignature(DatasetKind.KLINES,"changed",2)
    assert changed.cache_identity()!=SIG.cache_identity()
