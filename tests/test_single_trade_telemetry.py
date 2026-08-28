from __future__ import annotations

from pathlib import Path

import pandas as pd

import crypto_strategy_lab.telemetry as telemetry_module
from crypto_strategy_lab.telemetry import (
    outcome_label,
    partial_take_profit_analysis,
    stop_loss_journey_analysis,
    trade_journey_analysis,
)


def test_outcome_label_uses_one_actual_trade_exit() -> None:
    assert outcome_label(pd.Series({"long_exit_reason": "TP"})) == "TP"
    assert outcome_label(pd.Series({"short_exit_reason": "SL"})) == "SL"
    assert outcome_label(pd.Series({"long_exit_reason": "BE_R_OFFSET"})) == "BE"
    assert outcome_label(pd.Series({"short_exit_reason": "PROFILE_TIMEOUT"})) == "PROFILE_TIMEOUT"
    assert outcome_label(pd.Series({"long_exit_reason": "TP", "long_be_triggered": True})) == "TP_AFTER_BE_MOVE"


def test_trade_journey_analysis_has_directional_outcomes_not_pair_combinations() -> None:
    trades = pd.DataFrame(
        [
            {"pair_id": 1, "long_exit_reason": "TP", "holding_hours": 2.0, "pair_net_pnl": 10.0},
            {"pair_id": 2, "short_exit_reason": "SL", "holding_hours": 1.0, "pair_net_pnl": -5.0},
        ]
    )

    result = trade_journey_analysis(trades)
    counts = dict(zip(result["outcome"], result["trade_count"]))

    assert counts["TP"] == 1
    assert counts["SL"] == 1
    assert "Long TP / Short SL" not in counts
    assert "Long SL / Short TP" not in counts
    assert "Long SL / Short SL" not in counts


def test_stop_loss_journey_analysis_tracks_the_single_stopped_trade() -> None:
    stop_time = pd.Timestamp("2026-01-01T01:00:00Z")
    trades = pd.DataFrame(
        [
            {
                "pair_id": 1,
                "side": "LONG",
                "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "long_exit_reason": "SL",
                "long_exit_time": stop_time,
                "holding_hours": 1.0,
                "pair_net_pnl": -10.0,
                "adx_entry": 20.0,
            },
            {
                "pair_id": 2,
                "side": "SHORT",
                "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "short_exit_reason": "TP",
                "short_exit_time": stop_time,
                "holding_hours": 1.0,
                "pair_net_pnl": 10.0,
                "adx_entry": 30.0,
            },
        ]
    )
    telemetry = pd.DataFrame(
        [
            {"pair_id": 1, "timestamp": pd.Timestamp("2026-01-01T00:30:00Z"), "adx": 22.0, "di_spread": 10.0, "bb_width": 2.0, "atr": 1.0},
            {"pair_id": 1, "timestamp": stop_time, "adx": 25.0, "di_spread": 12.0, "bb_width": 2.5, "atr": 1.2},
            {"pair_id": 2, "timestamp": stop_time, "adx": 35.0, "di_spread": 14.0, "bb_width": 3.0, "atr": 1.5},
        ]
    )

    result = stop_loss_journey_analysis(trades, telemetry)

    assert result["pair_id"].tolist() == [1]
    assert result.loc[0, "side"] == "LONG"
    assert result.loc[0, "stop_loss_time"] == stop_time
    assert result.loc[0, "adx_at_stop"] == 25.0
    assert result.loc[0, "adx_max_before_stop"] == 25.0


def test_partial_take_profit_summary_counts_directional_trades() -> None:
    trades = pd.DataFrame(
        [
            {
                "long_entry_price": 100.0,
                "long_tp1_hit": True,
                "long_tp2_hit": False,
                "long_tp1_net_pnl": 0.4,
                "long_stop_net_pnl": 0.5,
                "pair_gross_pnl": 1.0,
                "pair_total_fees": 0.1,
                "pair_net_pnl": 0.9,
                "equity_after_trade": 1000.9,
            },
            {
                "short_entry_price": 100.0,
                "short_tp1_hit": True,
                "short_tp2_hit": True,
                "short_tp1_net_pnl": 0.5,
                "short_tp2_net_pnl": 1.4,
                "short_stop_net_pnl": 0.0,
                "pair_gross_pnl": 2.0,
                "pair_total_fees": 0.1,
                "pair_net_pnl": 1.9,
                "equity_after_trade": 1002.8,
            },
        ]
    )

    result = partial_take_profit_analysis(trades).iloc[0]

    assert result["total_trades"] == 2
    assert result["long_trades"] == 1
    assert result["short_trades"] == 1
    assert result["long_tp1_hit_rate"] == 1.0
    assert result["short_tp2_hit_rate"] == 1.0
    assert "total_legs" not in result.index


def test_retired_double_sl_journey_alias_is_removed() -> None:
    assert not hasattr(telemetry_module, "double_sl_journey_analysis")


def test_gui_worker_uses_stop_loss_journey_export_name() -> None:
    source = Path("crypto_strategy_lab/gui/worker.py").read_text(encoding="utf-8")
    assert "stop_loss_journey_analysis.csv" in source
    assert "double_sl_journey_analysis" not in source
