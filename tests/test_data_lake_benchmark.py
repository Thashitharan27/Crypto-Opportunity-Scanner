from __future__ import annotations

import pandas as pd

from tools.data_lake_benchmark import _median, _trade_fingerprint


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": 1,
                "side": "LONG",
                "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
                "long_exit_reason": "TP",
                "long_exit_source": "1M_INTRABAR",
                "pair_net_pnl": 10.0,
                "pair_net_r": 1.0,
                "equity_after_trade": 1010.0,
            },
            {
                "pair_id": 2,
                "side": "SHORT",
                "entry_time": pd.Timestamp("2026-01-01T02:00:00Z"),
                "exit_time": pd.Timestamp("2026-01-01T03:00:00Z"),
                "short_exit_reason": "SL",
                "short_exit_source": "1M_INTRABAR",
                "pair_net_pnl": -5.0,
                "pair_net_r": -1.0,
                "equity_after_trade": 1005.0,
            },
        ]
    )


def test_trade_fingerprint_is_stable_across_dataframe_attrs() -> None:
    left = _trades()
    right = _trades()
    left.attrs["diagnostic"] = {"large": "payload"}
    right.attrs["diagnostic"] = {"different": "payload"}

    assert _trade_fingerprint(left) == _trade_fingerprint(right)


def test_trade_fingerprint_changes_when_core_execution_result_changes() -> None:
    baseline = _trades()
    changed = _trades()
    changed.loc[1, "pair_net_r"] = -0.5

    assert _trade_fingerprint(baseline) != _trade_fingerprint(changed)


def test_trade_fingerprint_ignores_non_execution_research_columns() -> None:
    baseline = _trades()
    changed = _trades()
    changed["research_note"] = ["alpha", "beta"]

    assert _trade_fingerprint(baseline) == _trade_fingerprint(changed)


def test_warm_median_uses_numeric_timing_values() -> None:
    records = [
        {"simulation_seconds": 3.0},
        {"simulation_seconds": 1.0},
        {"simulation_seconds": 2.0},
    ]
    assert _median(records, "simulation_seconds") == 2.0
