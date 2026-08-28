from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_strategy_lab.bayesian_research import (
    EVIDENCE_FAMILIES,
    FAMILY_LOG_ODDS_CAP,
    BayesianTradeModel,
    enrich_bayesian_trade_probabilities,
    threshold_simulation,
)


def _trade(
    entry: str,
    exit_: str,
    *,
    side: str = "LONG",
    pnl: float = 1.0,
    net_r: float | None = None,
    regime: str = "BULL",
    plus_di: float = 30.0,
    minus_di: float = 15.0,
    plus_change: float = 2.0,
    minus_change: float = -1.0,
    adx: float = 25.0,
    **extra,
) -> dict[str, object]:
    row = {
        "entry_time": pd.Timestamp(entry, tz="UTC"),
        "exit_time": pd.Timestamp(exit_, tz="UTC"),
        "side": side,
        "pair_net_pnl": pnl,
        "pair_net_r": pnl if net_r is None else net_r,
        "market_regime": regime,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "plus_di_change": plus_change,
        "minus_di_change": minus_change,
        "adx": adx,
        # Keep the synthetic baseline representative of a normal native trade row
        # so v2 has the required three independent evidence families.
        "mean_reversion_state": "BELOW_MEAN",
    }
    row.update(extra)
    return row


def _rich_trade(*, good: bool, pnl: float, day: int = 1) -> dict[str, object]:
    return _trade(
        f"2024-02-{day:02d} 00:00",
        f"2024-02-{day:02d} 01:00",
        pnl=pnl,
        atr_pct=0.015 if good else 0.04,
        bb_width=0.05 if good else 0.12,
        mean_reversion_state="BELOW_MEAN" if good else "ABOVE_MEAN",
        mean_reversion_motion="TOWARD_MEAN" if good else "AWAY_FROM_MEAN",
        mean_reversion_signal_direction="LONG" if good else "SHORT",
        mean_reversion_rsi=38.0 if good else 72.0,
        long_trade_location_rating="GOOD" if good else "BAD",
        long_room_in_direction_atr=3.5 if good else 0.5,
        long_near_support=good,
        long_near_resistance=not good,
        oi_vs_price_state_1h="PRICE_UP_OI_UP" if good else "PRICE_DOWN_OI_UP",
        oi_change_pct_1h=0.02 if good else -0.02,
        oi_zscore_7d=0.6 if good else 2.5,
        funding_bias="NEGATIVE" if good else "POSITIVE",
        funding_rate_bps=-1.0 if good else 2.0,
        mark_index_basis_state="NEUTRAL" if good else "POSITIVE",
        mark_index_basis_bps=0.5 if good else 4.0,
        taker_buy_sell_ratio=1.3 if good else 0.7,
        taker_delta_pct_15m=0.20 if good else -0.20,
        taker_delta_pct_1h=0.18 if good else -0.18,
        flow_persistence=0.8 if good else 0.4,
        trade_source_covered=True,
        trade_delta_pct_15m=0.18 if good else -0.18,
        book_ticker_observed=True,
        book_ticker_stale=False,
        book_imbalance_l1=0.4 if good else -0.4,
        book_depth_observed=True,
        book_depth_stale=False,
        book_depth_imbalance_1pct=0.35 if good else -0.35,
    )


def test_future_outcome_cannot_change_an_earlier_probability():
    rows = [
        _trade("2024-01-01 00:00", "2024-01-01 02:00", pnl=1.0),
        # This entry happens before trade 1 has closed, so it must still see a
        # completely neutral model regardless of trade 1's eventual result.
        _trade("2024-01-01 01:00", "2024-01-01 03:00", pnl=-1.0),
        _trade("2024-01-01 04:00", "2024-01-01 05:00", pnl=1.0),
    ]
    first = enrich_bayesian_trade_probabilities(pd.DataFrame(rows))

    changed = pd.DataFrame(rows)
    changed.loc[0, "pair_net_pnl"] = -100.0
    changed.loc[0, "pair_net_r"] = -100.0
    second = enrich_bayesian_trade_probabilities(changed)

    assert first.loc[1, "bayes_long_probability"] == pytest.approx(0.5)
    assert second.loc[1, "bayes_long_probability"] == pytest.approx(0.5)
    assert first.loc[1, "bayes_long_probability"] == second.loc[1, "bayes_long_probability"]
    # By the third entry, trade 1 and trade 2 have completed, so their outcomes
    # are allowed to affect the posterior.
    assert first.loc[2, "bayes_long_probability"] != second.loc[2, "bayes_long_probability"]


def test_small_context_sample_is_shrunk_instead_of_trusting_raw_win_rate():
    model = BayesianTradeModel(prior_strength=20.0)
    observations = [
        _trade("2024-01-01 00:00", "2024-01-01 01:00", pnl=1.0),
        _trade("2024-01-02 00:00", "2024-01-02 01:00", pnl=1.0),
        _trade("2024-01-03 00:00", "2024-01-03 01:00", pnl=1.0),
        _trade("2024-01-04 00:00", "2024-01-04 01:00", pnl=-1.0),
    ]
    for raw in observations:
        model.observe(pd.Series(raw))

    estimate = model.estimate(pd.Series(observations[-1]), "LONG")

    assert estimate.context_samples == 4
    assert estimate.probability < 0.75
    assert estimate.probability > 0.5
    assert estimate.confidence == "LOW"


def test_completed_trade_becomes_available_at_a_later_entry():
    frame = pd.DataFrame(
        [
            _trade("2024-01-01 00:00", "2024-01-01 01:00", pnl=1.0),
            _trade("2024-01-01 02:00", "2024-01-01 03:00", pnl=-1.0),
        ]
    )

    scored = enrich_bayesian_trade_probabilities(frame)

    assert scored.loc[0, "bayes_long_side_samples"] == 0
    assert scored.loc[0, "bayes_long_context_samples"] == 0
    assert scored.loc[1, "bayes_long_side_samples"] == 1
    assert scored.loc[1, "bayes_long_context_samples"] == 1
    assert scored.loc[1, "bayes_long_probability"] > 0.5


def test_long_and_short_are_scored_independently():
    frame = pd.DataFrame(
        [
            _trade("2024-01-01 00:00", "2024-01-01 01:00", side="LONG", pnl=1.0),
            _trade(
                "2024-01-01 02:00",
                "2024-01-01 03:00",
                side="SHORT",
                pnl=-1.0,
                plus_di=15.0,
                minus_di=30.0,
                plus_change=-1.0,
                minus_change=2.0,
                regime="BEAR",
            ),
            _trade("2024-01-01 04:00", "2024-01-01 05:00", side="LONG", pnl=1.0),
        ]
    )

    scored = enrich_bayesian_trade_probabilities(frame)
    last = scored.iloc[-1]

    assert last["bayes_long_side_samples"] == 1
    assert last["bayes_short_side_samples"] == 1
    assert 0.0 < last["bayes_long_probability"] < 1.0
    assert 0.0 < last["bayes_short_probability"] < 1.0
    assert last["bayes_actual_side_probability"] == pytest.approx(last["bayes_long_probability"])


def test_threshold_flags_require_probability_samples_and_multiple_families():
    rows = []
    for day in range(1, 27):
        rows.append(
            _trade(
                f"2024-01-{day:02d} 00:00",
                f"2024-01-{day:02d} 01:00",
                pnl=1.0 if day <= 24 else -1.0,
            )
        )
    scored = enrich_bayesian_trade_probabilities(
        pd.DataFrame(rows), min_context_samples=20
    )

    assert not scored.loc[10, "bayes_take_55_min20"]
    assert scored.loc[21, "bayes_actual_context_samples"] >= 20
    assert scored.loc[21, "bayes_actual_evidence_families"] >= 3
    assert scored.loc[21, "bayes_take_55_min20"]

    summary = threshold_simulation(scored, min_context_samples=20)
    row_55 = summary.loc[np.isclose(summary["minimum_probability"], 0.55)].iloc[0]
    expected = scored["bayes_take_55_min20"].sum()
    assert row_55["trades"] == expected
    assert row_55["trades"] > 0


def test_input_order_does_not_change_scores_for_distinct_entry_times():
    frame = pd.DataFrame(
        [
            _trade("2024-01-01 00:00", "2024-01-01 01:00", pnl=1.0),
            _trade("2024-01-01 02:00", "2024-01-01 03:00", pnl=-1.0),
            _trade("2024-01-01 04:00", "2024-01-01 05:00", pnl=1.0),
        ]
    )
    normal = enrich_bayesian_trade_probabilities(frame.copy()).set_index("entry_time")
    shuffled = enrich_bayesian_trade_probabilities(
        frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    ).set_index("entry_time")

    assert normal["bayes_long_probability"].sort_index().tolist() == pytest.approx(
        shuffled["bayes_long_probability"].sort_index().tolist()
    )


def test_v2_uses_broad_evidence_families_without_exact_context_fragmentation():
    model = BayesianTradeModel()
    # Same side baseline, but broad evidence consistently separates the two states.
    for i in range(40):
        model.observe(pd.Series(_rich_trade(good=True, pnl=1.0 if i < 32 else -1.0, day=i % 28 + 1)))
    for i in range(40):
        model.observe(pd.Series(_rich_trade(good=False, pnl=1.0 if i < 12 else -1.0, day=i % 28 + 1)))

    good = model.estimate(pd.Series(_rich_trade(good=True, pnl=1.0)), "LONG")
    bad = model.estimate(pd.Series(_rich_trade(good=False, pnl=-1.0)), "LONG")

    assert good.evidence_families == len(EVIDENCE_FAMILIES)
    assert bad.evidence_families == len(EVIDENCE_FAMILIES)
    assert good.context_samples == 40
    assert bad.context_samples == 40
    assert good.probability > bad.probability
    assert good.expected_net_r > bad.expected_net_r
    # No single correlated family is allowed to dominate the combined score.
    assert all(abs(lift) <= FAMILY_LOG_ODDS_CAP + 1e-12 for _, lift in good.family_lifts)


def test_optional_stale_order_book_evidence_is_not_used():
    model = BayesianTradeModel()
    good = _rich_trade(good=True, pnl=1.0)
    stale = dict(good)
    stale["book_ticker_observed"] = False
    stale["book_ticker_stale"] = True
    stale["book_depth_observed"] = False
    stale["book_depth_stale"] = True

    for _ in range(30):
        model.observe(pd.Series(good))
    estimate = model.estimate(pd.Series(stale), "LONG")
    families = {name for name, _ in estimate.family_lifts}

    # Trade-flow fields are still valid, but stale book fields add no extra family:
    # the family remains bounded and usable from covered trade-flow evidence only.
    assert "microstructure" in families
    assert abs(dict(estimate.family_lifts)["microstructure"]) <= FAMILY_LOG_ODDS_CAP


def test_enriched_rows_expose_family_diagnostics_for_mcp_analysis():
    rows = []
    for day in range(1, 24):
        row = _rich_trade(good=True, pnl=1.0, day=(day - 1) % 22 + 1)
        row["entry_time"] = pd.Timestamp("2024-03-01", tz="UTC") + pd.Timedelta(days=day)
        row["exit_time"] = row["entry_time"] + pd.Timedelta(hours=1)
        rows.append(row)

    scored = enrich_bayesian_trade_probabilities(pd.DataFrame(rows))
    last = scored.iloc[-1]

    assert last["bayes_model_version"] == "BAYES_EVIDENCE_V2"
    assert last["bayes_actual_evidence_families"] >= 7
    assert last["bayes_actual_evidence_items"] > last["bayes_actual_evidence_families"]
    assert "bayes_actual_oi_positioning_log_odds_lift" in scored
    assert "bayes_actual_funding_basis_log_odds_lift" in scored
    assert "bayes_top_positive_evidence" in scored


def test_empty_frame_gets_stable_bayesian_schema():
    frame = pd.DataFrame()
    scored = enrich_bayesian_trade_probabilities(frame)

    assert scored.empty
    assert "bayes_long_probability" in scored
    assert "bayes_short_probability" in scored
    assert "bayes_take_60_min20" in scored
    assert "bayes_actual_evidence_families" in scored
    assert "bayes_actual_microstructure_log_odds_lift" in scored
