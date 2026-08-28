from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_strategy_lab.rule_native_engine import (
    RuleAwareDataLakeProductionBacktestEngine,
)
from crypto_strategy_lab.strategy_rule_model import (
    compile_profiles,
    new_rule,
    normalize_rule,
)
from crypto_strategy_lab.strategy_profiles import RULE_INDICATORS


def _engine_with_dataframe_research():
    engine = object.__new__(RuleAwareDataLakeProductionBacktestEngine)
    engine.research_features = {
        "futures_positioning": pd.DataFrame(
            {
                "oi_change_pct_5m": [0.01],
                "oi_change_pct_1h": [0.025],
                "oi_change_pct_24h": [-0.04],
                "oi_zscore_7d": [1.75],
                "price_change_pct_1h": [0.012],
                "oi_vs_price_state_1h": ["PRICE_UP_OI_UP"],
                "top_trader_account_bias": [0.20],
                "top_trader_position_bias": [0.15],
                "global_long_short_account_bias": [-0.05],
                "taker_long_short_volume_bias": [0.08],
            }
        ),
        "funding_context": pd.DataFrame(
            {
                "funding_rate_bps": [-0.75],
                "funding_bias": ["NEGATIVE"],
                "funding_24h_sum_bps": [-1.5],
                "funding_change": [-0.00002],
                "funding_3_event_mean": [-0.00010],
                "funding_7d_zscore": [-2.25],
                "funding_extreme_positive": [False],
                "funding_extreme_negative": [True],
            }
        ),
        "basis_context": pd.DataFrame(
            {
                "mark_index_basis_bps": [-2.5],
                "mark_index_basis_state": ["NEGATIVE"],
                "mark_index_basis_zscore_7d": [-1.8],
                "trade_mark_basis_bps": [1.2],
                "trade_index_basis_bps": [-1.3],
                "premium_index_zscore_7d": [-2.1],
            }
        ),
        "taker_flow_context": pd.DataFrame(
            {
                "taker_buy_sell_ratio": [1.4],
                "taker_delta_pct": [0.10],
                "taker_delta_pct_15m": [0.16],
                "taker_delta_pct_1h": [0.22],
                "flow_persistence": [0.75],
            }
        ),
    }
    return engine


def test_lightweight_futures_rule_indicators_are_registered():
    expected = {
        "OI_CHANGE_PCT_1H",
        "OI_VS_PRICE_STATE_1H",
        "FUNDING_RATE_BPS",
        "FUNDING_BIAS",
        "FUNDING_EXTREME_NEGATIVE",
        "MARK_INDEX_BASIS_BPS",
        "MARK_INDEX_BASIS_STATE",
        "TAKER_DELTA_PCT_1H",
    }
    assert expected <= set(RULE_INDICATORS)


def test_dataframe_research_values_feed_numeric_and_categorical_rules():
    engine = _engine_with_dataframe_research()
    profile = SimpleNamespace()

    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "OI_CHANGE_PCT_1H"
    ) == 0.025
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "OI_VS_PRICE_STATE_1H"
    ) == 1.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "FUNDING_RATE_BPS"
    ) == -0.75
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "FUNDING_BIAS"
    ) == 1.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "FUNDING_EXTREME_NEGATIVE"
    ) == 1.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "FUNDING_CHANGE_BPS"
    ) == -0.2
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "MARK_INDEX_BASIS_STATE"
    ) == 1.0
    assert engine._strategy_profile_rule_value(
        0, "LONG", profile, "TAKER_DELTA_PCT_1H"
    ) == 0.22


def test_native_prepared_research_block_is_supported_without_dataframe_conversion():
    engine = object.__new__(RuleAwareDataLakeProductionBacktestEngine)
    engine.research_features = {
        "futures_positioning": SimpleNamespace(
            values={
                "oi_change_pct_1h": np.array([0.03]),
                "oi_vs_price_state_1h": np.array(["PRICE_DOWN_OI_UP"], dtype=object),
            }
        ),
        "funding_context": SimpleNamespace(
            values={
                "funding_rate_bps": np.array([-1.0]),
                "funding_bias": np.array(["NEGATIVE"], dtype=object),
            }
        ),
    }

    assert engine._prepared_research_value(0, "OI_CHANGE_PCT_1H") == 0.03
    assert engine._prepared_research_value(0, "OI_VS_PRICE_STATE_1H") == 3.0
    assert engine._prepared_research_value(0, "FUNDING_RATE_BPS") == -1.0
    assert engine._prepared_research_value(0, "FUNDING_BIAS") == 1.0


def test_negative_funding_is_a_first_class_categorical_entry_rule():
    rule = new_rule(kind="REQUIRED", evidence="FUNDING_BIAS")
    assert rule["operator"] == "IS"
    assert rule["value"] == "NEGATIVE"

    profiles, _execution = compile_profiles(
        direction_mode="DI",
        market_permissions=("BULL_LONG",),
        required_rules=(rule,),
    )
    native = profiles["bull_long"].entry_rules[0]
    assert native["indicator"] == "FUNDING_BIAS"
    assert native["condition"] == "OUTSIDE"
    assert native["minimum"] == native["maximum"] == 1.0
    assert native["_builder_kind"] == "REQUIRED"


def test_missing_required_futures_evidence_fails_closed_but_veto_does_not_fire():
    engine = object.__new__(RuleAwareDataLakeProductionBacktestEngine)
    engine.research_features = {}
    profile = SimpleNamespace()

    required = {
        "indicator": "FUNDING_BIAS",
        "condition": "OUTSIDE",
        "minimum": 1.0,
        "maximum": 1.0,
        "_builder_kind": "REQUIRED",
    }
    veto = {
        "indicator": "FUNDING_BIAS",
        "condition": "INSIDE",
        "minimum": 1.0,
        "maximum": 1.0,
        "_builder_kind": "VETO",
    }
    assert engine._strategy_profile_entry_rule_matches(0, "LONG", profile, required)
    assert not engine._strategy_profile_entry_rule_matches(0, "LONG", profile, veto)


def test_normalize_rule_accepts_price_oi_state_and_extreme_funding_boolean():
    oi_rule = normalize_rule(
        {
            "kind": "VETO",
            "evidence": "OI_VS_PRICE_STATE_1H",
            "operator": "IS",
            "value": "PRICE_DOWN_OI_UP",
            "regime": "ALL",
            "side": "ALL",
        }
    )
    funding_rule = normalize_rule(
        {
            "kind": "VETO",
            "evidence": "FUNDING_EXTREME_POSITIVE",
            "operator": "IS",
            "value": "TRUE",
            "regime": "ALL",
            "side": "ALL",
        }
    )
    assert oi_rule["value"] == "PRICE_DOWN_OI_UP"
    assert funding_rule["value"] == "TRUE"
