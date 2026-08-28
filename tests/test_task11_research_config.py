from dataclasses import replace

import pytest

from crypto_strategy_lab.data_lake_config import (
    ExecutionProfileConfig,
    FeatureConfig,
    ResearchRunConfig,
    StrategyProfileConfig,
    normalize_data_lake_config,
)


def test_v3_is_strict_and_flat_aliases_are_rejected():
    with pytest.raises(ValueError, match="Unknown Data Lake configuration sections"):
        normalize_data_lake_config({"config_version": 3, "atr_period": 7})
    with pytest.raises(ValueError, match="Unknown features settings"):
        normalize_data_lake_config(
            {"config_version": 3, "features": {"output_dir": "x"}}
        )


def test_v3_defaults_preserve_pre_split_data_lake_semantics():
    config = ResearchRunConfig()
    assert config.data.strategy_timeframe_minutes == 15
    assert config.data.intrabar_timeframe_minutes == 1
    assert config.data.intrabar_missing_policy == "ERROR"
    assert config.features.market_regime_method == "BTC_STRUCTURAL"
    assert config.features.bull_regime_lookback_days == 90
    assert config.features.bull_regime_return_threshold == pytest.approx(0.20)
    assert config.strategy.enable_di_direction_selection is True
    assert config.strategy.enable_di_pressure_analysis is True
    assert config.strategy.enable_mean_reversion_analysis is True
    assert config.execution.percent_r == pytest.approx(0.002)
    assert config.execution.risk_per_leg == pytest.approx(0.01)
    assert config.execution.slippage == pytest.approx(0.0005)
    assert config.reporting.telemetry_interval_minutes == 15
    assert config.reporting.save_indicator_analysis_reports is True
    assert config.reporting.create_standard_charts is True


def test_strategy_and_execution_profiles_are_separate_authoritative_types():
    config = ResearchRunConfig()
    strategy_profile = config.strategy.profiles["bull_long"]
    execution_profile = config.execution.profiles["bull_long"]
    assert isinstance(strategy_profile, StrategyProfileConfig)
    assert isinstance(execution_profile, ExecutionProfileConfig)
    assert hasattr(strategy_profile, "entry_rules")
    assert not hasattr(strategy_profile, "stop_loss_multiple")
    assert hasattr(execution_profile, "stop_loss_multiple")
    assert not hasattr(execution_profile, "entry_rules")


@pytest.mark.parametrize(
    "component,field,value",
    [
        ("reporting", "output_dir", "elsewhere"),
        ("reporting", "create_standard_charts", False),
        ("execution", "maker_fee", 0.9),
        ("execution", "slippage", 0.1),
    ],
)
def test_non_market_configuration_does_not_change_feature_parameters(
    component, field, value
):
    original = ResearchRunConfig()
    changed_part = replace(getattr(original, component), **{field: value})
    changed = replace(original, **{component: changed_part})
    assert changed.features.registry_parameters() == original.features.registry_parameters()


def test_feature_period_deterministically_changes_registry_parameters():
    original = ResearchRunConfig()
    changed = replace(original, features=replace(original.features, adx_period=21))
    assert changed.features.registry_parameters() != original.features.registry_parameters()


def test_sr_registry_parameters_are_owned_by_feature_config():
    features = FeatureConfig(
        enable_support_resistance_analysis=True, sr_timeframe_minutes=0
    )
    params = features.registry_parameters(strategy_timeframe_minutes=240)
    assert params["support_resistance"]["sr_timeframe_minutes"] == 240
    assert params["support_resistance"]["atr_period"] == features.atr_period


def test_inactive_telemetry_interval_does_not_block_higher_timeframe_run():
    base = ResearchRunConfig()
    config = replace(
        base,
        data=replace(base.data, strategy_timeframe_minutes=240),
        reporting=replace(
            base.reporting,
            enable_trade_telemetry=False,
            telemetry_interval_minutes=15,
        ),
    )
    config.validate()


def test_active_telemetry_interval_must_align_to_strategy_timeframe():
    base = ResearchRunConfig()
    invalid = replace(
        base,
        data=replace(base.data, strategy_timeframe_minutes=240),
        reporting=replace(
            base.reporting,
            enable_trade_telemetry=True,
            telemetry_interval_minutes=15,
        ),
    )
    with pytest.raises(ValueError, match="telemetry interval must be a multiple"):
        invalid.validate()

    valid = replace(
        invalid,
        reporting=replace(invalid.reporting, telemetry_interval_minutes=240),
    )
    valid.validate()
