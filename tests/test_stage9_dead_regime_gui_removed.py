from pathlib import Path


def test_dead_regime_reward_risk_gui_remnants_are_gone():
    root = Path(__file__).resolve().parents[1]
    text = (root / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    for stale in (
        'QGroupBox("Regime-Specific Reward/Risk")',
        'conditional target replaces Long Bull Reward/Risk',
        'regime_targets_help=',
        'bull_long_conditional_help=',
        'values.update({})',
    ):
        assert stale not in text


def test_di_tab_still_exposes_active_direction_and_pressure_controls():
    root = Path(__file__).resolve().parents[1]
    text = (root / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert 'QGroupBox("DI Direction Selection")' in text
    assert 'QGroupBox("DI Pressure Analysis")' in text
    assert 'self.profile_editor.values()' in text
