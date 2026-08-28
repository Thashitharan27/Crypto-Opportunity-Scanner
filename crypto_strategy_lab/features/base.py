"""Small, authoritative contracts for derived features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype,
    is_string_dtype,
)

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind


_MISSING = object()


@dataclass(frozen=True, slots=True)
class FeatureDataResource:
    """Identity for an auxiliary feature input sharing a DatasetKind.

    Unlike the strategy ``KLINES`` slot, this key carries interval and role, so
    an independently requested research timeline cannot overwrite it.
    """

    dataset: DatasetKind
    interval: str
    role: str


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """One accepted parameter, including its canonical representation."""

    normalizer: Callable[[object], object]
    default: object = _MISSING

    @property
    def required(self) -> bool:
        return self.default is _MISSING

    def normalize(self, value: object) -> object:
        try:
            if self.normalizer is bool and isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    return False
                raise ValueError(f"invalid boolean parameter value {value!r}")
            return self.normalizer(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid parameter value {value!r}") from exc


@dataclass(frozen=True, slots=True)
class OutputField:
    """Logical (not storage-specific) output type."""

    kind: str = "any"  # any, numeric, string, bool, datetime
    nullable: bool = True

    def validate(self, series: pd.Series, name: str) -> None:
        checks = {
            "any": lambda _: True,
            "numeric": is_numeric_dtype,
            "string": lambda s: is_string_dtype(s.dtype) or is_object_dtype(s.dtype),
            "bool": is_bool_dtype,
            "datetime": is_datetime64_any_dtype,
        }
        if self.kind not in checks:
            raise ValueError(f"Unknown logical output kind {self.kind!r}")
        if not checks[self.kind](series):
            raise ValueError(f"Feature output {name!r} must be {self.kind}, got {series.dtype}")
        if not self.nullable and series.isna().any():
            raise ValueError(f"Feature output {name!r} must not contain missing values")


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Complete public contract and cache-relevant identity of one feature."""

    name: str
    version: str
    required_datasets: tuple[DatasetKind, ...]
    output_columns: tuple[str, ...] = ()
    parameters: Mapping[str, ParameterDefinition] = field(default_factory=dict)
    output_schema: Mapping[str, OutputField] = field(default_factory=dict)
    output_schema_factory: Callable[[Mapping[str, object]], Mapping[str, OutputField]] | None = None
    required_features: tuple[str, ...] = ()
    optional_datasets: tuple[DatasetKind, ...] = ()
    warmup_bars: int = 0
    availability_rule: str = "max_dependency_available_at"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("feature name and version must not be empty")
        if set(self.required_datasets) & set(self.optional_datasets):
            raise ValueError("dataset cannot be both required and optional")
        schema = dict(self.output_schema)
        if self.output_columns:
            def inferred(name: str) -> OutputField:
                if name in {"timestamp", "available_at", "date"} or name.endswith("_at") or name.endswith("_time"):
                    return OutputField("datetime")
                if name == "close_location":
                    return OutputField("numeric")
                if (
                    name in {"funding_bias", "bb_reentry"}
                    or name.endswith(("_confirmation", "_setup_strength"))
                    or any(
                        token in name
                        for token in (
                            "state", "rating", "location", "signal", "regime", "motion",
                            "strength_label", "basis_state",
                        )
                    )
                ):
                    return OutputField("string")
                if (
                    name.startswith(("near_", "inside_"))
                    or "_near_" in name
                    or "_inside_" in name
                    or name.endswith(("_tested", "_held", "_changed", "_reentry"))
                ):
                    return OutputField("bool")
                return OutputField("numeric")
            schema = {**{name: inferred(name) for name in self.output_columns}, **schema}
        if not schema and self.output_schema_factory is None:
            raise ValueError("feature must declare an output schema")
        schema.setdefault("available_at", OutputField("datetime", nullable=False))
        if "timestamp" not in schema and "date" not in schema:
            schema["timestamp"] = OutputField("datetime", nullable=False)
        object.__setattr__(self, "output_schema", schema)
        object.__setattr__(self, "output_columns", tuple(schema))
        if self.name in self.required_features:
            raise ValueError("feature cannot depend on itself")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be non-negative")
        if not self.availability_rule.strip():
            raise ValueError("availability_rule must not be empty")

    def normalize_parameters(self, supplied: Mapping[str, object] | None = None) -> dict[str, object]:
        supplied = dict(supplied or {})
        unknown = sorted(set(supplied) - set(self.parameters))
        if unknown:
            raise ValueError(f"Unknown parameters for {self.name}: {unknown}")
        normalized: dict[str, object] = {}
        for name in sorted(self.parameters):
            spec = self.parameters[name]
            if name in supplied:
                value = supplied[name]
            elif spec.required:
                raise ValueError(f"Missing required parameter for {self.name}: {name}")
            else:
                value = spec.default
            normalized[name] = spec.normalize(value)
        return normalized

    def schema_for(self, parameters: Mapping[str, object] | None = None) -> dict[str, OutputField]:
        schema = dict(self.output_schema)
        if self.output_schema_factory is not None:
            normalized = self.normalize_parameters(parameters)
            schema.update(self.output_schema_factory(normalized))
        return schema

    def validate_output(
        self,
        frame: pd.DataFrame,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        schema = self.schema_for(parameters)
        missing = sorted(set(schema) - set(frame.columns))
        if missing:
            raise ValueError(f"Invalid {self.name} output; missing columns: {missing}")
        for name, field in schema.items():
            field.validate(frame[name], name)


class FeatureProvider(Protocol):
    definition: FeatureDefinition

    def compute(self, request: DataRequest, datasets: Mapping[DatasetKind, pd.DataFrame],
                parameters: Mapping[str, object],
                feature_frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame: ...
