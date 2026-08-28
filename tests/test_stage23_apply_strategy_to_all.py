from copy import deepcopy
from dataclasses import fields, replace

from crypto_strategy_lab.gui.profile_editor import StrategyProfilesWidget
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, StrategyProfile


def test_apply_strategy_to_all_copies_complete_baseline_but_preserves_profile_identity(qtbot):
    widget = StrategyProfilesWidget()
    qtbot.addWidget(widget)

    source_key = "bull_long"
    source = replace(
        widget.profiles[source_key],
        enabled=True,
        flip_direction=False,
        risk_multiplier=1.75,
        flip_rule_match_mode="ALL",
        reject_rule_match_mode="ALL",
        rsi_period=9,
        momentum_lookback_hours=36,
        entry_rules=(
            {
                "action": "REJECT",
                "indicator": "ADX",
                "condition": "OUTSIDE",
                "minimum": 12.0,
                "maximum": 35.0,
            },
        ),
        stop_loss_multiple=2.5,
        reward_risk_ratio=1.8,
        partial_stop_enabled=True,
        sl1_r=0.75,
        sl1_close_pct=40.0,
        sl2_r=2.5,
        partial_profit_enabled=True,
        tp1_r=1.25,
        tp1_close_pct=45.0,
        tp2_r=3.5,
        break_even_enabled=True,
        break_even_activation_r=1.1,
        break_even_offset_r=0.2,
        timeout_enabled=True,
        timeout_minutes=720,
    )
    widget.profiles[source_key] = source
    widget.current = source_key

    identity_before = {}
    for index, key in enumerate(PROFILE_KEYS):
        if key == source_key:
            continue
        enabled = index % 2 == 0
        flip_direction = index % 2 == 1
        widget.profiles[key] = replace(
            widget.profiles[key],
            enabled=enabled,
            flip_direction=flip_direction,
            risk_multiplier=0.25,
            stop_loss_multiple=7.0,
            reward_risk_ratio=4.0,
            entry_rules=(),
        )
        identity_before[key] = (enabled, flip_direction)

    widget._apply_strategy_to_all()

    assert widget.profiles[source_key] == source
    for key in PROFILE_KEYS:
        if key == source_key:
            continue
        target = widget.profiles[key]
        assert (target.enabled, target.flip_direction) == identity_before[key]
        for field in fields(StrategyProfile):
            if field.name in {"enabled", "flip_direction"}:
                continue
            assert getattr(target, field.name) == deepcopy(getattr(source, field.name)), field.name
