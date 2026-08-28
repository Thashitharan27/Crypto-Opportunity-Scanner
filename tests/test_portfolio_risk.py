import pandas as pd
import pytest

from crypto_strategy_lab.portfolio import _shared_equity_replay, run_portfolio


def test_total_portfolio_risk_cap_blocks_excess_overlapping_entry():
    start=pd.Timestamp("2024-01-01",tz="UTC")
    trades=pd.DataFrame([
        {"entry_time":start,"exit_time":start+pd.Timedelta(hours=2),"portfolio_trade_key":"A-1","asset":"A","pair_net_r":1.0},
        {"entry_time":start,"exit_time":start+pd.Timedelta(hours=2),"portfolio_trade_key":"B-1","asset":"B","pair_net_r":1.0},
        {"entry_time":start,"exit_time":start+pd.Timedelta(hours=2),"portfolio_trade_key":"C-1","asset":"C","pair_net_r":1.0},
        {"entry_time":start+pd.Timedelta(hours=3),"exit_time":start+pd.Timedelta(hours=4),"portfolio_trade_key":"C-2","asset":"C","pair_net_r":1.0},
    ])
    realized,assignments,max_open,_,max_risk,blocked=_shared_equity_replay(trades,1000,.02,.05)
    assert blocked==1 and max_open==2 and max_risk==pytest.approx(.04)
    assert assignments["C-1"]["portfolio_accepted"] is False
    assert assignments["C-1"]["portfolio_block_reason"]=="MAXIMUM_TOTAL_PORTFOLIO_RISK"
    assert assignments["C-2"]["portfolio_accepted"] is True
    assert set(realized["portfolio_trade_key"])=={"A-1","B-1","C-2"}


def test_portfolio_rejects_cap_below_per_asset_risk():
    with pytest.raises(ValueError,match="risk per asset"):
        run_portfolio([("A","a.json"),("B","b.json")],risk_per_asset=.06,maximum_total_risk=.05)
