"""Native structural benchmark data requirement shared by GUI and validation."""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import timedelta
from .data import DatasetKind

STRUCTURAL_BENCHMARK_INTERVAL="1h"
STRUCTURAL_REGIME_METHODS={"BTC_STRUCTURAL","ASSET_STRUCTURAL"}

@dataclass(frozen=True)
class StructuralBenchmarkRequirement:
    symbol: str
    interval: str
    warmup_days: int
    data_request: object

def structural_benchmark_requirement(request,features):
    if request is None or features is None: return None
    method=str(getattr(features,"market_regime_method","")).upper()
    if method not in STRUCTURAL_REGIME_METHODS: return None
    warmup_days=int(features.structural_regime_sma_days)+int(features.structural_regime_slope_lookback_days)+7
    symbol="BTCUSDT" if method=="BTC_STRUCTURAL" else request.symbol.upper()
    base=request.to_data_request()
    data_request=replace(base,symbol=symbol,start=request.period_start-timedelta(days=warmup_days),
        end=request.period_end,strategy_interval=STRUCTURAL_BENCHMARK_INTERVAL,
        intrabar_interval=None,datasets=(DatasetKind.KLINES,))
    return StructuralBenchmarkRequirement(symbol,STRUCTURAL_BENCHMARK_INTERVAL,warmup_days,data_request)
