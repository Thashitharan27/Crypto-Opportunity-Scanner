from dataclasses import fields
from pathlib import Path

from crypto_strategy_lab.config import BacktestConfig
from crypto_strategy_lab.gui.config_logic import DEFAULT_GUI_CONFIG

GLOBAL_KEYS = {
    "enable_bull_long_r_step_trailing", "bull_long_r_step_activation_r",
    "bull_long_r_step_distance_r", "bull_long_r_step_size_r",
    "bull_long_r_step_maximum_r", "bull_long_r_step_activation_close_pct",
    "enable_atr_checkpoint_tp_extension", "atr_checkpoint_di_spread_minimum",
    "atr_checkpoint_bb_width_minimum", "atr_checkpoint_profit_lock_start",
    "atr_checkpoint_profit_lock_distance",
}

def test_global_special_exit_config_fields_are_gone():
    names = {f.name for f in fields(BacktestConfig)}
    assert not (GLOBAL_KEYS & names)
    assert not (GLOBAL_KEYS & set(DEFAULT_GUI_CONFIG))

def test_no_global_special_exit_config_fallbacks_remain():
    root = Path(__file__).resolve().parents[1] / "crypto_strategy_lab"
    checked = [
        root / "engine.py",
        root / "output_manager.py",
        root / "gui" / "config_logic.py",
        root / "gui" / "main_window.py",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for key in GLOBAL_KEYS:
            assert f"self.config.{key}" not in text, f"global fallback {key} remains in {path}"

def test_removed_global_gui_names_do_not_return():
    root = Path(__file__).resolve().parents[1] / "crypto_strategy_lab"
    for path in (root / "gui" / "config_logic.py", root / "gui" / "main_window.py"):
        text = path.read_text(encoding="utf-8")
        for key in GLOBAL_KEYS:
            assert key not in text, f"retired global GUI field {key} remains in {path}"

def test_position_level_special_exit_state_is_retained():
    text = (Path(__file__).resolve().parents[1] / "crypto_strategy_lab" / "trade.py").read_text(encoding="utf-8")
    assert "r_step_trailing_enabled" in text
    assert "atr_checkpoint_extension_enabled" in text
    assert "atr_checkpoint_di_spread_minimum" in text


def test_profile_special_exit_ownership_is_retained():
    text = (Path(__file__).resolve().parents[1] / "crypto_strategy_lab" / "strategy_profiles.py").read_text(encoding="utf-8")
    assert "r_step_trailing_enabled" in text
    assert "atr_checkpoint_tp_extension_enabled" in text
    assert "atr_checkpoint_di_spread_minimum" in text
