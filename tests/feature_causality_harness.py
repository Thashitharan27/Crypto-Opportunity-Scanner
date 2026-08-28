"""One cache-safe future-mutation experiment for every feature provider."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import pandas as pd


FrameMutator = Callable[[pd.DataFrame, pd.Timestamp], None]
RegistryFactory = Callable[[Mapping[str, pd.DataFrame]], object]


@dataclass(frozen=True)
class CausalityCase:
    feature_name: str
    registry_factory: RegistryFactory
    request: object
    datasets: Mapping[object, pd.DataFrame]
    parameters: Mapping[str, Mapping[str, object]]
    future_mutators: Mapping[object, FrameMutator]
    context: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    future_context_mutators: Mapping[str, FrameMutator] = field(default_factory=dict)


def _copy_frames(frames):
    return {key: value.copy(deep=True) for key, value in frames.items()}


def _execute(case: CausalityCase, datasets, context):
    registry = case.registry_factory(context)
    frame = registry.execute(
        [case.feature_name],
        case.request,
        datasets,
        parameters=case.parameters,
        cache=None,
    )[case.feature_name]
    return registry, frame


def _comparison_columns(registry, case: CausalityCase) -> list[str]:
    definition = registry.get(case.feature_name).definition
    target_parameters = case.parameters.get(case.feature_name)
    return list(definition.schema_for(target_parameters))


def _resource_label(key: object) -> str:
    value = getattr(key, "value", None)
    if value is not None:
        return str(value)
    dataset = getattr(key, "dataset", None)
    interval = getattr(key, "interval", None)
    role = getattr(key, "role", None)
    if dataset is not None and interval is not None and role is not None:
        return f"{getattr(dataset, 'value', dataset)}:{interval}:{role}"
    return repr(key)


def assert_future_mutation_invariant(case: CausalityCase) -> None:
    """Recompute the complete registry graph without cache for every material source.

    The cutoff is an output ``available_at`` timestamp. Mutators may modify only
    source observations not yet available at that cutoff. The target feature is
    then recomputed through FeatureRegistry and every already-available declared
    output field must remain semantically identical.
    """
    base_datasets = _copy_frames(case.datasets)
    base_context = _copy_frames(case.context)
    registry, baseline = _execute(case, base_datasets, base_context)
    available = pd.to_datetime(baseline["available_at"], utc=True, errors="raise")
    resolved = registry.resolve([case.feature_name], case.parameters)
    effective_warmup = next(
        item.effective_warmup_bars
        for item in resolved
        if item.definition.name == case.feature_name
    )
    if len(available) < 3:
        raise AssertionError(f"{case.feature_name} causality fixture is too small")
    minimum_index = min(max(int(effective_warmup), 1), len(available) - 2)
    cutoff_index = max(minimum_index, len(available) // 2)
    cutoff_index = min(cutoff_index, len(available) - 2)
    cutoff = available.iloc[cutoff_index]
    columns = _comparison_columns(registry, case)
    expected = baseline.loc[available <= cutoff, columns].reset_index(drop=True)
    assert len(expected), f"{case.feature_name} has no already-available output at cutoff"

    for kind, mutate in case.future_mutators.items():
        changed_datasets = _copy_frames(case.datasets)
        changed_context = _copy_frames(case.context)
        before = changed_datasets[kind].copy(deep=True)
        mutate(changed_datasets[kind], cutoff)
        assert not changed_datasets[kind].equals(before), (
            f"mutator did not change future {_resource_label(kind)} source "
            f"for {case.feature_name}"
        )
        _, actual = _execute(case, changed_datasets, changed_context)
        actual_available = pd.to_datetime(actual["available_at"], utc=True, errors="raise")
        actual = actual.loc[actual_available <= cutoff, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_like=False)

    for name, mutate in case.future_context_mutators.items():
        changed_datasets = _copy_frames(case.datasets)
        changed_context = _copy_frames(case.context)
        before = changed_context[name].copy(deep=True)
        mutate(changed_context[name], cutoff)
        assert not changed_context[name].equals(before), (
            f"mutator did not change future {name} context for {case.feature_name}"
        )
        _, actual = _execute(case, changed_datasets, changed_context)
        actual_available = pd.to_datetime(actual["available_at"], utc=True, errors="raise")
        actual = actual.loc[actual_available <= cutoff, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_like=False)
