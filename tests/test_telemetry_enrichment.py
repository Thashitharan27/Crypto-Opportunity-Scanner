import pandas as pd

from crypto_strategy_lab.telemetry import INDICATORS, add_journey_columns


def test_journey_enrichment_preserves_skipped_signal_metadata():
    trades = pd.DataFrame({"pair_id": [1], "holding_minutes": [0]})
    skipped = [{"entry_filter_reason": "Profile rule rejected entry"}]
    trades.attrs["skipped_signals"] = skipped
    telemetry = pd.DataFrame(
        {"pair_id": [1], "elapsed_minutes": [0], **{name: [1.0] for name in INDICATORS}}
    )
    enriched = add_journey_columns(trades, telemetry)
    assert enriched.attrs["skipped_signals"] == skipped


def test_journey_enrichment_reports_each_completed_trade():
    trades = pd.DataFrame({"pair_id": [1, 2], "holding_minutes": [0, 0]})
    telemetry = pd.DataFrame(
        {
            "pair_id": [1, 2],
            "elapsed_minutes": [0, 0],
            **{name: [1.0, 1.0] for name in INDICATORS},
        }
    )
    calls = []
    add_journey_columns(trades, telemetry, progress=lambda current, total: calls.append((current, total)))
    assert calls == [(1, 2), (2, 2)]
