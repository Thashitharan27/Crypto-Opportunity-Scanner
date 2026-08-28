from types import SimpleNamespace

from crypto_strategy_lab.enhanced_engine import EnhancedBacktestEngine
from crypto_strategy_lab.gui.enhanced_config import enhanced_default_gui_config


def _engine_for(state, *, expanding=True, contracting=True, mixed=True, enabled=True):
    engine = object.__new__(EnhancedBacktestEngine)
    engine.config = SimpleNamespace(
        enable_di_pressure_analysis=enabled,
        di_pressure_allow_expanding=expanding,
        di_pressure_allow_contracting=contracting,
        di_pressure_allow_mixed=mixed,
    )
    engine._selected_direction = lambda _i: "LONG"
    engine._di_pressure_snapshot = lambda _i, _direction: {"di_pressure_state": state}
    return engine


def test_enhanced_defaults_preserve_record_only_behavior():
    values = enhanced_default_gui_config()
    assert values["di_pressure_allow_expanding"] is True
    assert values["di_pressure_allow_contracting"] is True
    assert values["di_pressure_allow_mixed"] is True


def test_all_states_allowed_does_not_filter_unknown_warmup_state():
    engine = _engine_for("UNKNOWN")
    assert engine._di_pressure_filter_result(10) == (True, None)


def test_expanding_only_rejects_contracting_and_mixed():
    contracting = _engine_for("CONTRACTING", contracting=False, mixed=False)
    mixed = _engine_for("MIXED", contracting=False, mixed=False)
    expanding = _engine_for("EXPANDING", contracting=False, mixed=False)

    assert expanding._di_pressure_filter_result(10) == (True, None)
    assert contracting._di_pressure_filter_result(10) == (False, "DI_PRESSURE_CONTRACTING_FILTERED")
    assert mixed._di_pressure_filter_result(10) == (False, "DI_PRESSURE_MIXED_FILTERED")


def test_disabled_analysis_never_filters_entries():
    engine = _engine_for("CONTRACTING", expanding=True, contracting=False, mixed=False, enabled=False)
    assert engine._di_pressure_filter_result(10) == (True, None)
