from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.historical_strategy_validation import (
    EVERY_VIABLE_ENTRY, STANDARD_SINGLE_SYMBOL, HistoricalStrategyValidator,
    SymbolResearchResult, aggregate_reports, attach_candidate_trades,
)


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
        return SymbolResearchResult("run-"+symbol,pd.DataFrame(columns=["entry_time","pair_net_r","side","pair_id"]),pd.DataFrame(columns=["entry_time","pair_net_r","side","research_sample_id"]),end)
    config=ResearchRunConfig(); path=tmp_path/"config.json"; path.write_text("{}")
    result=HistoricalStrategyValidator(execute,warmup_bars=lambda _:10).run(c,config,config_path=path,publish=False)
    assert len(calls)==7 and [x[0] for x in calls]==sorted(symbols)
    assert all(x[3] is config.strategy and x[4] is config.execution for x in calls)
    assert len(result.outcomes)==200 and set(result.outcomes.population)=={STANDARD_SINGLE_SYMBOL,EVERY_VIABLE_ENTRY}
    assert all(end >= c[c.symbol.eq(symbol)].decision_timestamp.max()+pd.Timedelta("24h") for symbol,_,end,_,_ in calls)
