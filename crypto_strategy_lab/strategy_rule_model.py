"""Rule-based strategy authoring compiled to the mature simulator contract.

The researcher-facing model is intentionally not profile based. Market
permissions, required filters, vetoes, and optional direction flips are authored
once with regime/side scopes. Only at the simulator boundary are those rules
expanded into the six mature regime/direction inputs still consumed by the proven
engine. Builder metadata is embedded in rule dictionaries so save/load can round
trip the rule model without exposing profiles in the UI.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

from crypto_strategy_lab.data_lake_config import (
    ExecutionProfileConfig,
    StrategyProfileConfig,
)
from crypto_strategy_lab.strategy_profiles import PROFILE_KEYS, RULE_INDICATORS

# Direction selection and trade permission are intentionally separate concepts.
# DI remains the raw directional control. DMI_TREND keeps the same +DI/-DI side
# selection but adds a deliberately simple, non-optimized trend confirmation
# baseline at compile time. LONG/SHORT eligibility still belongs to the market
# permission grid below.
DIRECTION_MODES = ("DI", "DMI_TREND")
REGIMES = ("BULL", "BEAR", "SIDEWAYS")
SIDES = ("LONG", "SHORT")
MARKET_PERMISSIONS = tuple(f"{regime}_{side}" for regime in REGIMES for side in SIDES)
NUMERIC_RULE_OPERATORS = ("GT", "GTE", "LT", "LTE", "BETWEEN", "OUTSIDE")
CATEGORICAL_RULE_OPERATORS = ("IS", "IS_NOT")
RULE_OPERATORS = (*NUMERIC_RULE_OPERATORS, *CATEGORICAL_RULE_OPERATORS)
RULE_KINDS = ("REQUIRED", "VETO", "FLIP")

_BOOL_VALUES = ("TRUE", "FALSE")
CATEGORICAL_RULE_VALUES = {
    "DI_PRESSURE_STATE": ("EXPANDING", "CONTRACTING", "MIXED"),
    "SR_NEAR_SUPPORT": _BOOL_VALUES,
    "SR_NEAR_RESISTANCE": _BOOL_VALUES,
    "SR_INSIDE_SUPPORT_ZONE": _BOOL_VALUES,
    "SR_INSIDE_RESISTANCE_ZONE": _BOOL_VALUES,
    "SR_SUPPORT_HELD": _BOOL_VALUES,
    "SR_RESISTANCE_HELD": _BOOL_VALUES,
    "SR_SUPPORT_STATE": (
        "NO_SUPPORT_NEARBY",
        "APPROACHING_SUPPORT",
        "SUPPORT_TESTING",
        "SUPPORT_HELD",
        "SUPPORT_BROKEN",
    ),
    "SR_RESISTANCE_STATE": (
        "NO_RESISTANCE_NEARBY",
        "APPROACHING_RESISTANCE",
        "RESISTANCE_TESTING",
        "RESISTANCE_HELD",
        "RESISTANCE_BROKEN",
    ),
    "SR_TRADE_LOCATION_RATING": (
        "GOOD_LOCATION",
        "NEUTRAL_LOCATION",
        "BAD_LOCATION",
    ),
    "OI_VS_PRICE_STATE_1H": (
        "PRICE_UP_OI_UP",
        "PRICE_UP_OI_DOWN",
        "PRICE_DOWN_OI_UP",
        "PRICE_DOWN_OI_DOWN",
        "FLAT_OR_MIXED",
    ),
    "FUNDING_BIAS": ("NEGATIVE", "NEUTRAL", "POSITIVE"),
    "FUNDING_EXTREME_POSITIVE": _BOOL_VALUES,
    "FUNDING_EXTREME_NEGATIVE": _BOOL_VALUES,
    "MARK_INDEX_BASIS_STATE": ("NEGATIVE", "NEUTRAL", "POSITIVE"),
}
CATEGORICAL_VALUE_CODES = {
    "DI_PRESSURE_STATE": {"EXPANDING": 1.0, "CONTRACTING": 2.0, "MIXED": 3.0},
    "SR_NEAR_SUPPORT": {"TRUE": 1.0, "FALSE": 0.0},
    "SR_NEAR_RESISTANCE": {"TRUE": 1.0, "FALSE": 0.0},
    "SR_INSIDE_SUPPORT_ZONE": {"TRUE": 1.0, "FALSE": 0.0},
    "SR_INSIDE_RESISTANCE_ZONE": {"TRUE": 1.0, "FALSE": 0.0},
    "SR_SUPPORT_HELD": {"TRUE": 1.0, "FALSE": 0.0},
    "SR_RESISTANCE_HELD": {"TRUE": 1.0, "FALSE": 0.0},
    "SR_SUPPORT_STATE": {
        "NO_SUPPORT_NEARBY": 1.0,
        "APPROACHING_SUPPORT": 2.0,
        "SUPPORT_TESTING": 3.0,
        "SUPPORT_HELD": 4.0,
        "SUPPORT_BROKEN": 5.0,
    },
    "SR_RESISTANCE_STATE": {
        "NO_RESISTANCE_NEARBY": 1.0,
        "APPROACHING_RESISTANCE": 2.0,
        "RESISTANCE_TESTING": 3.0,
        "RESISTANCE_HELD": 4.0,
        "RESISTANCE_BROKEN": 5.0,
    },
    "SR_TRADE_LOCATION_RATING": {
        "GOOD_LOCATION": 1.0,
        "NEUTRAL_LOCATION": 2.0,
        "BAD_LOCATION": 3.0,
    },
    "OI_VS_PRICE_STATE_1H": {
        "PRICE_UP_OI_UP": 1.0,
        "PRICE_UP_OI_DOWN": 2.0,
        "PRICE_DOWN_OI_UP": 3.0,
        "PRICE_DOWN_OI_DOWN": 4.0,
        "FLAT_OR_MIXED": 5.0,
    },
    "FUNDING_BIAS": {"NEGATIVE": 1.0, "NEUTRAL": 2.0, "POSITIVE": 3.0},
    "FUNDING_EXTREME_POSITIVE": {"TRUE": 1.0, "FALSE": 0.0},
    "FUNDING_EXTREME_NEGATIVE": {"TRUE": 1.0, "FALSE": 0.0},
    "MARK_INDEX_BASIS_STATE": {"NEGATIVE": 1.0, "NEUTRAL": 2.0, "POSITIVE": 3.0},
}
SUPPORT_RESISTANCE_RULE_EVIDENCE = frozenset(
    indicator for indicator in RULE_INDICATORS if indicator.startswith("SR_")
)
LOW = -1e308
HIGH = 1e308

_META_PREFIX = "_builder_"


def is_categorical_evidence(evidence: str) -> bool:
    return str(evidence).upper() in CATEGORICAL_RULE_VALUES


def is_support_resistance_evidence(evidence: str) -> bool:
    return str(evidence).upper() in SUPPORT_RESISTANCE_RULE_EVIDENCE


def uses_support_resistance_rules(*rule_groups) -> bool:
    return any(
        is_support_resistance_evidence(rule.get("evidence", ""))
        for group in rule_groups
        for rule in (group or ())
    )


def rule_operator_options(evidence: str) -> tuple[str, ...]:
    return CATEGORICAL_RULE_OPERATORS if is_categorical_evidence(evidence) else NUMERIC_RULE_OPERATORS


def rule_value_options(evidence: str) -> tuple[str, ...]:
    return CATEGORICAL_RULE_VALUES.get(str(evidence).upper(), ())


def new_rule(*, kind: str = "REQUIRED", evidence: str = "DI_SPREAD") -> dict:
    """Return one researcher-facing rule row with stable round-trip identity."""
    evidence = str(evidence).upper()
    if is_categorical_evidence(evidence):
        operator = "IS"
        value = rule_value_options(evidence)[0]
        value2 = None
    else:
        operator = "GTE"
        value = 30.0 if evidence == "DI_SPREAD" else 0.0
        value2 = 0.0
    return {
        "id": uuid4().hex,
        "kind": kind,
        "evidence": evidence,
        "operator": operator,
        "value": value,
        "value2": value2,
        "regime": "ALL",
        "side": "ALL",
    }


def normalize_rule(rule: dict, *, expected_kind: str | None = None) -> dict:
    if not isinstance(rule, dict):
        raise ValueError("strategy rule must be an object")
    value = dict(rule)
    value.setdefault("id", uuid4().hex)
    value.setdefault("kind", expected_kind or "REQUIRED")
    value.setdefault("evidence", "DI_SPREAD")
    value["kind"] = str(value["kind"]).upper()
    value["evidence"] = str(value["evidence"]).upper()
    categorical = is_categorical_evidence(value["evidence"])
    value.setdefault("operator", "IS" if categorical else "GTE")
    value.setdefault("value", rule_value_options(value["evidence"])[0] if categorical else 0.0)
    value.setdefault("value2", None if categorical else 0.0)
    value.setdefault("regime", "ALL")
    value.setdefault("side", "ALL")
    if expected_kind is not None:
        value["kind"] = expected_kind
    value["operator"] = str(value["operator"]).upper()
    value["regime"] = str(value["regime"]).upper()
    value["side"] = str(value["side"]).upper()

    if value["kind"] not in RULE_KINDS:
        raise ValueError(f"unsupported strategy rule kind: {value['kind']}")
    if value["evidence"] not in RULE_INDICATORS:
        raise ValueError(f"unsupported strategy rule evidence: {value['evidence']}")
    if value["operator"] not in rule_operator_options(value["evidence"]):
        raise ValueError(
            f"unsupported operator {value['operator']} for {value['evidence']}"
        )
    if value["regime"] not in ("ALL", *REGIMES):
        raise ValueError(f"unsupported strategy rule regime scope: {value['regime']}")
    if value["side"] not in ("ALL", *SIDES):
        raise ValueError(f"unsupported strategy rule side scope: {value['side']}")

    if categorical:
        value["value"] = str(value["value"]).upper()
        value["value2"] = None
        if value["value"] not in rule_value_options(value["evidence"]):
            raise ValueError(
                f"unsupported value {value['value']} for {value['evidence']}"
            )
    else:
        value["value"] = float(value["value"])
        value["value2"] = float(value["value2"])
        if value["operator"] in {"BETWEEN", "OUTSIDE"} and value["value"] > value["value2"]:
            raise ValueError("strategy rule lower value cannot exceed upper value")
    return value


def normalize_rules(rules, *, kind: str) -> tuple[dict, ...]:
    return tuple(normalize_rule(rule, expected_kind=kind) for rule in (rules or ()))


def _profile_scope(profile_key: str) -> tuple[str, str]:
    regime, source_side = profile_key.upper().split("_", 1)
    return regime, source_side


def effective_side(source_side: str, direction_mode: str) -> str:
    """Return the DI-selected side; the chosen mode may add confirmations."""
    mode = str(direction_mode).upper()
    if mode not in DIRECTION_MODES:
        raise ValueError(f"unsupported direction mode: {direction_mode}")
    return source_side


def _applies(rule: dict, regime: str, side: str) -> bool:
    return rule["regime"] in ("ALL", regime) and rule["side"] in ("ALL", side)


def _range(rule: dict) -> tuple[float, float, str]:
    operator = rule["operator"]
    evidence = rule["evidence"]
    if is_categorical_evidence(evidence):
        code = CATEGORICAL_VALUE_CODES[evidence][rule["value"]]
        return code, code, "INSIDE" if operator == "IS" else "OUTSIDE"

    first, second = float(rule["value"]), float(rule["value2"])
    if operator == "GT":
        return LOW, first, "OUTSIDE"
    if operator == "GTE":
        return first, HIGH, "INSIDE"
    if operator == "LT":
        return first, HIGH, "OUTSIDE"
    if operator == "LTE":
        return LOW, first, "INSIDE"
    if operator == "BETWEEN":
        return first, second, "INSIDE"
    if operator == "OUTSIDE":
        return first, second, "OUTSIDE"
    raise ValueError(f"unsupported strategy rule operator: {operator}")


def _native_rule(rule: dict, *, required: bool) -> dict:
    minimum, maximum, matching_condition = _range(rule)
    # A required rule rejects when its desired condition is false. A veto or
    # flip rule acts when its desired condition is true.
    condition = (
        "OUTSIDE" if matching_condition == "INSIDE" else "INSIDE"
    ) if required else matching_condition
    action = "FLIP" if rule["kind"] == "FLIP" else "REJECT"
    return {
        "action": action,
        "indicator": rule["evidence"],
        "condition": condition,
        "minimum": minimum,
        "maximum": maximum,
        f"{_META_PREFIX}id": rule["id"],
        f"{_META_PREFIX}kind": rule["kind"],
        f"{_META_PREFIX}operator": rule["operator"],
        f"{_META_PREFIX}value": rule["value"],
        f"{_META_PREFIX}value2": rule["value2"],
        f"{_META_PREFIX}regime": rule["regime"],
        f"{_META_PREFIX}side": rule["side"],
    }


_DMI_TREND_MODE_MARKER = "_strategy_direction_mode"
_DMI_TREND_RULE_MARKER = "_strategy_builtin_rule"


def _dmi_trend_native_rules() -> tuple[dict, ...]:
    """Return the fixed v1 DMI Trend confirmations.

    This is intentionally a conservative research baseline rather than an
    optimized parameter set: raw DI chooses the side, ADX must be at least 20,
    ADX must be non-falling versus the previous completed strategy candle, and
    directional DI pressure must be expanding over the configured pressure
    lookback. The rules are hidden from researcher-authored Entry Rules but carry
    REQUIRED metadata so missing causal evidence rejects the candidate.
    """
    required = {
        "action": "REJECT",
        "condition": "OUTSIDE",
        f"{_META_PREFIX}kind": "REQUIRED",
        _DMI_TREND_MODE_MARKER: "DMI_TREND",
    }
    expanding = CATEGORICAL_VALUE_CODES["DI_PRESSURE_STATE"]["EXPANDING"]
    return (
        {
            **required,
            "indicator": "ADX",
            "minimum": 20.0,
            "maximum": HIGH,
            _DMI_TREND_RULE_MARKER: "ADX_MIN_20",
        },
        {
            **required,
            "indicator": "ADX_CHANGE",
            "minimum": 0.0,
            "maximum": HIGH,
            _DMI_TREND_RULE_MARKER: "ADX_NON_FALLING",
        },
        {
            **required,
            "indicator": "DI_PRESSURE_STATE",
            "minimum": expanding,
            "maximum": expanding,
            _DMI_TREND_RULE_MARKER: "DI_PRESSURE_EXPANDING",
        },
    )


def compile_profiles(
    *,
    direction_mode: str,
    market_permissions,
    required_rules=(),
    veto_rules=(),
    flip_rules=(),
    rsi_period: int = 14,
    momentum_lookback_hours: int = 24,
    base_execution: ExecutionProfileConfig | None = None,
) -> tuple[dict[str, StrategyProfileConfig], dict[str, ExecutionProfileConfig]]:
    """Compile rule authoring into the exact mature six-input engine contract."""
    mode = str(direction_mode).upper()
    if mode not in DIRECTION_MODES:
        raise ValueError(f"unsupported direction mode: {direction_mode}")
    permissions = {str(item).upper() for item in market_permissions}
    unknown_permissions = sorted(permissions - set(MARKET_PERMISSIONS))
    if unknown_permissions:
        raise ValueError("unknown market permissions: " + ", ".join(unknown_permissions))
    required = normalize_rules(required_rules, kind="REQUIRED")
    veto = normalize_rules(veto_rules, kind="VETO")
    flips = normalize_rules(flip_rules, kind="FLIP")
    if int(rsi_period) <= 0 or int(momentum_lookback_hours) <= 0:
        raise ValueError("rule calculation periods must be positive")

    strategy_profiles: dict[str, StrategyProfileConfig] = {}
    execution_profiles: dict[str, ExecutionProfileConfig] = {}
    execution = base_execution or ExecutionProfileConfig()

    for key in PROFILE_KEYS:
        regime, source_side = _profile_scope(key)
        side = effective_side(source_side, mode)
        enabled = f"{regime}_{side}" in permissions
        native_rules = list(_dmi_trend_native_rules()) if mode == "DMI_TREND" else []
        for rule in required:
            if _applies(rule, regime, side):
                native_rules.append(_native_rule(rule, required=True))
        for rule in veto:
            if _applies(rule, regime, side):
                native_rules.append(_native_rule(rule, required=False))
        for rule in flips:
            if _applies(rule, regime, side):
                native_rules.append(_native_rule(rule, required=False))

        strategy_profiles[key] = StrategyProfileConfig(
            enabled=enabled,
            flip_direction=False,
            entry_rules=tuple(native_rules),
            flip_rule_match_mode="ANY",
            reject_rule_match_mode="ANY",
            rsi_period=int(rsi_period),
            momentum_lookback_hours=int(momentum_lookback_hours),
        )
        execution_profiles[key] = replace(execution)

    return strategy_profiles, execution_profiles


def _builder_rule(native_rule: dict) -> dict | None:
    if not isinstance(native_rule, dict) or f"{_META_PREFIX}id" not in native_rule:
        return None
    return normalize_rule({
        "id": native_rule[f"{_META_PREFIX}id"],
        "kind": native_rule.get(f"{_META_PREFIX}kind", "VETO"),
        "evidence": native_rule.get("indicator", "DI_SPREAD"),
        "operator": native_rule.get(f"{_META_PREFIX}operator", "BETWEEN"),
        "value": native_rule.get(f"{_META_PREFIX}value", native_rule.get("minimum", 0.0)),
        "value2": native_rule.get(f"{_META_PREFIX}value2", native_rule.get("maximum", 0.0)),
        "regime": native_rule.get(f"{_META_PREFIX}regime", "ALL"),
        "side": native_rule.get(f"{_META_PREFIX}side", "ALL"),
    })


def decompile_rules(strategy_profiles) -> dict[str, tuple[dict, ...]]:
    """Recover builder-authored rules from metadata embedded in compiled rules."""
    by_kind: dict[str, dict[str, dict]] = {kind: {} for kind in RULE_KINDS}
    for profile in strategy_profiles.values():
        for native in getattr(profile, "entry_rules", ()):
            rule = _builder_rule(native)
            if rule is None:
                continue
            by_kind[rule["kind"]][rule["id"]] = rule
    return {
        kind: tuple(deepcopy(list(items.values())))
        for kind, items in by_kind.items()
    }


def infer_direction_mode(strategy_profiles) -> str:
    """Recover the authored direction strategy from compiled native metadata."""
    if any(bool(profile.flip_direction) for profile in strategy_profiles.values()):
        raise ValueError(
            "Static profile direction overrides are retired; use Direction Flip Rules instead"
        )
    for profile in strategy_profiles.values():
        for rule in getattr(profile, "entry_rules", ()):
            if str(rule.get(_DMI_TREND_MODE_MARKER, "")).upper() == "DMI_TREND":
                return "DMI_TREND"
    return "DI"


def infer_market_permissions(strategy_profiles, direction_mode: str) -> tuple[str, ...]:
    # Validate the authored direction strategy even though DI preserves source side.
    effective_side("LONG", direction_mode)
    permissions = set()
    for key, profile in strategy_profiles.items():
        if not profile.enabled:
            continue
        regime, source_side = _profile_scope(key)
        permissions.add(f"{regime}_{source_side}")
    return tuple(item for item in MARKET_PERMISSIONS if item in permissions)


def common_execution_profile(execution_profiles) -> ExecutionProfileConfig:
    """Use the first profile as the base; builder-authored saves write all six equally."""
    for key in PROFILE_KEYS:
        if key in execution_profiles:
            return replace(execution_profiles[key])
    return ExecutionProfileConfig()
