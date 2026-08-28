"""Migrate legacy GUI JSON files to the six independent strategy profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, StrategyProfile, profiles_to_dict


def _range_from_mode(enabled, mode, minimum, maximum, disabled="Disabled"):
    if not enabled or mode == disabled:
        return False, minimum, maximum
    text=str(mode).lower()
    return True, (minimum if "minimum" in text or "range" in text else -1000.0), (maximum if "maximum" in text or "range" in text else 1000.0)


def migrate(values):
    profiles={}
    regime_filter=bool(values.get("enable_regime_direction_filter",False))
    global_di=_range_from_mode(values.get("enable_di_spread_filter",False),values.get("di_spread_filter_mode","Disabled"),float(values.get("di_spread_minimum",0)),float(values.get("di_spread_maximum",1000)))
    global_adx=_range_from_mode(values.get("enable_adx_filter",False),values.get("adx_filter_mode","Disabled"),float(values.get("adx_minimum",0)),float(values.get("adx_maximum",1000)))
    global_bb=_range_from_mode(values.get("enable_bb_width_filter",False),values.get("bb_width_filter_mode","Disabled"),float(values.get("bb_width_minimum",0)),float(values.get("bb_width_maximum",1000)))
    notes=[]
    for key in PROFILE_KEYS:
        regime,direction=key.split("_"); upper=direction.upper()
        enabled=bool(values.get(f"allow_{key}",True)) if regime_filter else True
        if regime=="bull" and direction=="short" and values.get("enable_bull_regime_short_filter",False): enabled=False
        rr=float(values.get(f"di_{direction}_{regime}_reward_risk_ratio",values.get(f"di_{direction}_reward_risk_ratio",values.get("di_reward_risk_ratio",1)))) if values.get("enable_di_regime_reward_risk",False) else float(values.get(f"di_{direction}_reward_risk_ratio",values.get("di_reward_risk_ratio",1)))

        di_min=max(0.0,float(values.get(f"di_direction_{direction}_minimum_spread",values.get("di_direction_minimum_spread",0))))
        di_max=1000.0; di_enabled=di_min>0
        if global_di[0]: di_enabled=True; di_min=max(di_min,global_di[1]); di_max=min(di_max,global_di[2])
        if values.get("enable_directional_di_spread_range",False):
            di_enabled=True; di_min=max(di_min,float(values.get(f"directional_{direction}_di_spread_minimum",0))); di_max=min(di_max,float(values.get(f"directional_{direction}_di_spread_maximum",1000)))

        adx_min=0.0; adx_max=1000.0; adx_enabled=False
        if global_adx[0]: adx_enabled=True; adx_min=max(adx_min,global_adx[1]); adx_max=min(adx_max,global_adx[2])
        if values.get("enable_directional_adx_range",False):
            adx_enabled=True
            adx_min=max(adx_min,float(values.get(f"directional_{direction}_adx_{'minimum' if direction=='long' else 'range_minimum'}",0)))
            adx_max=min(adx_max,float(values.get(f"directional_{direction}_adx_{'range_maximum' if direction=='long' else 'maximum'}",1000)))
        if values.get("enable_directional_adx_filter",False):
            adx_enabled=True
            if direction=="long": adx_max=min(adx_max,float(values.get("directional_long_adx_maximum",1000)))
            else: adx_min=max(adx_min,float(values.get("directional_short_adx_minimum",0)))
        if direction=="short" and values.get("enable_biased_short_adx_cap",False): adx_enabled=True; adx_max=min(adx_max,float(values.get("biased_short_adx_maximum",1000)))
        if regime=="bear" and values.get("enable_bear_regime_adx_filter",False): adx_enabled=True; adx_min=max(adx_min,float(values.get("bear_regime_adx_minimum",0)))

        atr_enabled=bool(values.get("enable_directional_atr_pct_range",False)); rsi_enabled=bool(values.get("enable_directional_rsi_range",False)); close_enabled=bool(values.get("enable_directional_close_location_range",False)); momentum_enabled=bool(values.get("enable_directional_momentum_range",False))
        momentum_min=float(values.get(f"directional_{direction}_momentum_minimum",-10)); momentum_max=float(values.get(f"directional_{direction}_momentum_maximum",10)); momentum_hours=int(values.get("directional_momentum_lookback_hours",24))
        if direction=="long" and values.get("enable_long_momentum_filter",False): momentum_enabled=True; momentum_min=max(momentum_min,float(values.get("long_momentum_minimum_return",-10))); momentum_hours=int(values.get("long_momentum_lookback_hours",24))
        vwap_enabled=direction=="short" and bool(values.get("enable_short_vwap_distance_filter",False))
        profile=StrategyProfile(
            enabled=enabled,reward_risk_ratio=rr,risk_multiplier=1.0,
            di_spread_enabled=di_enabled,di_spread_minimum=di_min,di_spread_maximum=di_max,
            adx_enabled=adx_enabled,adx_minimum=adx_min,adx_maximum=adx_max,
            atr_pct_enabled=atr_enabled,atr_pct_minimum=float(values.get(f"directional_{direction}_atr_pct_minimum",0)),atr_pct_maximum=float(values.get(f"directional_{direction}_atr_pct_maximum",1)),
            rsi_enabled=rsi_enabled,rsi_period=int(values.get("directional_rsi_period",14)),rsi_minimum=float(values.get(f"directional_{direction}_rsi_minimum",0)),rsi_maximum=float(values.get(f"directional_{direction}_rsi_maximum",100)),
            bb_width_enabled=global_bb[0],bb_width_minimum=global_bb[1],bb_width_maximum=global_bb[2],
            close_location_enabled=close_enabled,close_location_minimum=float(values.get(f"directional_{direction}_close_location_minimum",0)),close_location_maximum=float(values.get(f"directional_{direction}_close_location_maximum",1)),
            momentum_enabled=momentum_enabled,momentum_lookback_hours=momentum_hours,momentum_minimum=momentum_min,momentum_maximum=momentum_max,
            vwap_distance_enabled=vwap_enabled,vwap_distance_minimum=float(values.get("short_vwap_minimum_distance_atr",-1000)) if vwap_enabled else -1000,vwap_distance_maximum=1000,
        )
        profile.validate(key); profiles[key]=profile
    if values.get("enable_bull_long_momentum_target_extension",False): notes.append("Bull Long momentum-based variable target cannot be represented by one fixed profile target; Bull Long uses its normal regime reward/risk ratio.")
    values["enable_strategy_profiles"]=True
    values["strategy_profile_run_mode"]="BOTH"
    values["strategy_profiles"]=profiles_to_dict(profiles)
    if notes: values["strategy_profile_migration_notes"]=notes
    return values


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("folder",type=Path); args=parser.parse_args()
    for path in sorted(args.folder.glob("*.json")):
        values=json.loads(path.read_text())
        path.write_text(json.dumps(migrate(values),indent=2)+"\n")
        print(path.name)


if __name__=="__main__": main()
