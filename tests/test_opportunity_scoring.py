from datetime import datetime, timezone
import random
import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.features.opportunity_activity import OpportunityActivityFeatureProvider
from crypto_strategy_lab.features.technical import CoreDirectionalFeatureProvider
from crypto_strategy_lab.features.market_regime import PolicyMarketFeatureProvider
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
    assert OpportunityActivityFeatureProvider().definition.parameters["denominator_floor"].default==1e-9
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
    times=pd.date_range("2025-01-01",periods=60,freq="h",tz="UTC")
    frame=pd.DataFrame({"period_start":times,"available_at":times+pd.Timedelta(hours=1),"high":range(101,161),"low":range(99,159),"close":range(100,160),"quote_volume":[10.]*60})
    provider=OpportunityActivityFeatureProvider(); params={"window":24,"denominator_floor":1e-9}
    from crypto_strategy_lab.data.query import DataRequest
    req=DataRequest("X",times[0].to_pydatetime(),(times[-1]+pd.Timedelta(hours=1)).to_pydatetime(),"1h")
    features=provider.compute(req,{DatasetKind.KLINES:frame},params)
    assert features.iloc[-1].volume_ratio==1
    returns=pd.Series(frame.close).apply(float).pipe(lambda x: np.log(x/x.shift())).iloc[-24:]
    assert features.iloc[-1].realized_volatility==pytest.approx(returns.std(ddof=0)*24**.5)
    assert features.iloc[-1].recent_range_pct==pytest.approx((frame.high.iloc[-24:].max()-frame.low.iloc[-24:].min())/frame.close.iloc[-1])
    ranges=(frame.high-frame.low)/frame.close
    assert features.iloc[-1].range_expansion==pytest.approx(ranges.iloc[-1]/ranges.shift(24).rolling(24).median().iloc[-1])
    causal=latest_causal_snapshot("X",1,times[-2].to_pydatetime(),"1h",features,SIG)
    changed=features.copy(); changed.loc[50,"realized_volatility"]=999
    assert latest_causal_snapshot("X",1,times[-2].to_pydatetime(),"1h",changed,SIG)==causal
    assert not hasattr(score_opportunities([snap("A",1)],T).rows[0],"side")

def test_future_outcome_changes_not_frozen_score():
    row=score_opportunities([snap("A",1)],T).rows[0]
    times=pd.date_range(T,periods=25,freq="h")
    future=pd.DataFrame({"period_start":times,"high":[110]*24+[999],"low":[90]*24+[1],"close":[105]*24+[999]})
    evaluator=HistoricalOpportunityEvaluator(); first=evaluator.evaluate(row,future,100)
    # The horizon-boundary candle is excluded; the decision candle is included.
    assert first.forward_range_pct==.2
    future.loc[0,"high"]=120
    second=evaluator.evaluate(row,future,100)
    assert first!=second
    assert row==score_opportunities([snap("A",1)],T).rows[0]

def test_forward_outcome_requires_complete_fixed_grid():
    row=score_opportunities([snap("A",1)],T).rows[0]
    times=pd.date_range(T,periods=24,freq="h")
    complete=pd.DataFrame({"period_start":times,"high":110,"low":90,"close":100})
    evaluator=HistoricalOpportunityEvaluator()
    assert evaluator.evaluate(row,complete,100) is not None
    assert evaluator.evaluate(row,complete.iloc[:-1],100) is None
    assert evaluator.evaluate(row,complete.drop(index=10),100) is None

def test_non_hourly_legacy_is_explicitly_unscorable_and_config_is_enforced():
    four=snap("A",1)
    four=OpportunityFeatureSnapshot(four.symbol,four.discovery_rank,four.decision_time,"4h",four.feature_timestamp,four.available_at,four.values,SIG)
    legacy=score_opportunities([four],T,OpportunityScoringConfig("4h",(LEGACY_VOLATILITY_V1,))).rows[0]
    assert legacy.status==ScoreStatus.UNSCORABLE and "supports only 1h" in legacy.unscorable_reason
    mismatch=score_opportunities([four],T,OpportunityScoringConfig("1h",(BALANCED_ACTIVITY_V1,))).rows[0]
    assert mismatch.status==ScoreStatus.UNSCORABLE and "does not match config" in mismatch.unscorable_reason

def test_unscorable_preserves_raw_telemetry_and_has_no_trade_contract():
    row=score_opportunities([snap("A",1,missing=True)],T,OpportunityScoringConfig(models=(LEGACY_VOLATILITY_V1,))).rows[0]
    assert row.raw_components["realized_volatility"]==1
    assert row.raw_components["volume_ratio"] is None
    forbidden={"side","entry","stop","target","trade_eligibility","long","short"}
    assert forbidden.isdisjoint(row.__dataclass_fields__)

def test_registry_snapshot_reuses_existing_feature_outputs_at_boundary():
    at=pd.DataFrame({"timestamp":[T],"available_at":[T],"atr":[2.],"atr_pct":[.02],"adx":[20.],"di_spread":[5.],"di_spread_1":[4.]})
    policy=pd.DataFrame({"timestamp":[T],"available_at":[T],"momentum_return_24h":[-.1],"market_regime":["BEAR"]})
    activity=pd.DataFrame({"timestamp":[T],"available_at":[T],"realized_volatility":[.2],"recent_range_pct":[.3],"range_expansion":[1.2],"volume_ratio":[1.1]})
    at.attrs["feature_version"]=CoreDirectionalFeatureProvider().definition.version
    policy.attrs["feature_version"]=PolicyMarketFeatureProvider().definition.version
    activity.attrs["feature_version"]=OpportunityActivityFeatureProvider().definition.version
    result=snapshot_from_registry_features("A",1,T,"1h",{"core_directional":at,"policy_market_context":policy,"opportunity_activity":activity},SIG)
    assert result.values["atr"]==2 and result.values["adx"]==20 and result.values["di_spread"]==5
    later=at.copy(); later.loc[1]=[T,T+pd.Timedelta(hours=1),999,.99,99,99,99]
    unchanged=snapshot_from_registry_features("A",1,T,"1h",{"core_directional":later,"policy_market_context":policy,"opportunity_activity":activity},SIG)
    assert unchanged.values["atr"]==2

def test_comparison_population_strata_and_missing_lift():
    first=score_opportunities([snap("A",1),snap("B",2,missing=True)],T,OpportunityScoringConfig(models=(LEGACY_VOLATILITY_V1,))).rows
    later_time=datetime(2026,1,3,tzinfo=timezone.utc)
    later_snap=snap("C",1); later_snap=OpportunityFeatureSnapshot("C",1,later_time,"1h",later_time,later_time,{**later_snap.values,"market_regime":"BULL"},SIG)
    second=score_opportunities([later_snap],later_time,OpportunityScoringConfig(models=(LEGACY_VOLATILITY_V1,))).rows
    outcome=OpportunityOutcome("A",T,100,.2,.1,.05,1.5)
    report=build_comparison(first+second,{("A",T):outcome},top_ks=(1,))
    overall=next(r for r in report if r["model"]=="legacy_volatility_v1" and r["stratum_type"]=="ALL")
    assert (overall["candidate_count"],overall["scorable_count"],overall["selected_count"],overall["outcome_count"])==(3,2,2,1)
    assert overall["unscorable_rate"]==pytest.approx(1/3)
    year=next(r for r in report if r["model"]=="legacy_volatility_v1" and r["stratum"]=="2025")
    assert (year["decision_count"],year["candidate_count"],year["scorable_count"],year["outcome_count"])==(1,2,1,1)
    bull=next(r for r in report if r["model"]=="legacy_volatility_v1" and r["stratum"]=="BULL")
    assert (bull["candidate_count"],bull["scorable_count"],bull["outcome_count"])==(1,1,0)
    assert bull["mean_forward_range_pct"] is None and bull["lift_vs_discovery_order_forward_range_pct"] is None

def test_source_identity_is_canonical_signature():
    assert snap("A",1).source_signature.cache_identity()==snap("B",2).source_signature.cache_identity()
    changed=SourceSignature(DatasetKind.KLINES,"changed",2)
    assert changed.cache_identity()!=SIG.cache_identity()
