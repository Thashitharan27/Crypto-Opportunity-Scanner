"""Authoritative feature graph resolution, execution and identity."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from typing import Mapping, Sequence

import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.quality import validate_feature_timeline
from .base import FeatureDataResource, FeatureDefinition, FeatureProvider


@dataclass(frozen=True, slots=True)
class ResolvedFeature:
    definition: FeatureDefinition
    parameters: Mapping[str, object]
    effective_warmup_bars: int


class FeatureRegistry:
    """Owns the full graph; callers provide only requests, data and overrides."""

    def __init__(self) -> None:
        self._providers: dict[str, FeatureProvider] = {}

    def register(self, provider: FeatureProvider) -> None:
        name = provider.definition.name
        if name in self._providers:
            raise ValueError(f"Feature already registered: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> FeatureProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def dependency_order(self, feature_names: Sequence[str]) -> tuple[str, ...]:
        ordered, visiting, visited = [], [], set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(
                    "Feature dependency cycle detected: " + " -> ".join((*visiting, name))
                )
            visiting.append(name)
            definition = self.get(name).definition
            for dependency in sorted(definition.required_features):
                visit(dependency)
            visiting.pop()
            visited.add(name)
            ordered.append(name)

        for name in sorted(set(feature_names)):
            visit(name)
        return tuple(ordered)

    def resolve(
        self,
        feature_names: Sequence[str],
        parameters: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[ResolvedFeature, ...]:
        overrides = parameters or {}
        order = self.dependency_order(feature_names)
        unknown = sorted(set(overrides) - set(order))
        if unknown:
            raise ValueError(f"Parameters supplied for unrequested features: {unknown}")
        warmups: dict[str, int] = {}
        result = []
        for name in order:
            definition = self.get(name).definition
            inherited = max((warmups[d] for d in definition.required_features), default=0)
            warmups[name] = inherited + definition.warmup_bars
            result.append(
                ResolvedFeature(
                    definition,
                    definition.normalize_parameters(overrides.get(name)),
                    warmups[name],
                )
            )
        return tuple(result)

    def effective_warmup(self, feature_names: Sequence[str]) -> int:
        return max((item.effective_warmup_bars for item in self.resolve(feature_names)), default=0)

    def required_datasets(self, feature_names: Sequence[str]) -> tuple[DatasetKind, ...]:
        kinds = {
            kind
            for item in self.resolve(feature_names)
            for kind in item.definition.required_datasets
        }
        return tuple(sorted(kinds, key=lambda item: item.value))

    @staticmethod
    def identity(
        resolved: ResolvedFeature,
        request: DataRequest,
        source_identities: Mapping[object, str],
        dependency_identities: Mapping[str, str],
    ) -> str:
        definition = resolved.definition
        material_datasets = set(definition.required_datasets)
        material_datasets.update(
            kind for kind in definition.optional_datasets if kind in source_identities
        )
        base_role = definition.name.removesuffix("_context")
        roles = {definition.name, base_role, f"{base_role}_aggregate"}
        auxiliary = [
            key
            for key in source_identities
            if isinstance(key, FeatureDataResource) and key.role in roles
        ]
        missing_sources = set(definition.required_datasets) - set(source_identities)
        if missing_sources:
            raise ValueError(
                f"Missing source identities for {definition.name}: "
                f"{sorted(kind.value for kind in missing_sources)}"
            )
        payload = {
            "feature": definition.name,
            "version": definition.version,
            "cache_format_version": 2,
            "request_scope": request.feature_scope_key(),
            "parameters": dict(resolved.parameters),
            "sources": {
                kind.value: source_identities[kind]
                for kind in sorted(material_datasets, key=lambda item: item.value)
            },
            "auxiliary_sources": {
                f"{key.dataset.value}:{key.interval}:{key.role}": source_identities[key]
                for key in sorted(
                    auxiliary,
                    key=lambda item: (item.dataset.value, item.interval, item.role),
                )
            },
            "dependencies": {
                name: dependency_identities[name]
                for name in sorted(definition.required_features)
            },
            "schema": {
                name: {"kind": field.kind, "nullable": field.nullable}
                for name, field in definition.output_schema.items()
            },
        }
        return sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()

    def definition_hash(self, feature_names: Sequence[str]) -> str:
        payload = [
            {
                "name": item.definition.name,
                "version": item.definition.version,
                "parameters": list(item.definition.parameters),
                "schema": {
                    name: {"kind": field.kind, "nullable": field.nullable}
                    for name, field in item.definition.output_schema.items()
                },
                "parameterized_schema": item.definition.output_schema_factory is not None,
                "datasets": [d.value for d in item.definition.required_datasets],
                "optional_datasets": [d.value for d in item.definition.optional_datasets],
                "dependencies": list(item.definition.required_features),
                "warmup": item.definition.warmup_bars,
                "availability": item.definition.availability_rule,
            }
            for item in self.resolve(feature_names)
        ]
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def execute(
        self,
        feature_names: Sequence[str],
        request: DataRequest,
        datasets: Mapping[object, pd.DataFrame],
        *,
        parameters=None,
        cache=None,
        source_identities: Mapping[object, str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        frames, identities = {}, {}
        source_ids = source_identities or {
            kind: frame.attrs.get("canonical_source_identity")
            or (cache._source_signature(frame) if cache else "uncached")
            for kind, frame in datasets.items()
        }
        for resolved in self.resolve(feature_names, parameters):
            definition = resolved.definition
            missing = set(definition.required_datasets) - set(datasets)
            if missing:
                raise ValueError(
                    f"Missing datasets for {definition.name}: "
                    f"{sorted(kind.value for kind in missing)}"
                )
            identity = self.identity(resolved, request, source_ids, identities)
            frame = cache.load(definition, request, identity) if cache else None
            if frame is not None:
                try:
                    validate_feature_timeline(definition, frame, resolved.parameters)
                except ValueError:
                    frame = None
            if frame is None:
                provider = self.get(definition.name)
                dependencies = {
                    name: frames[name] for name in definition.required_features
                }
                if "feature_frames" in inspect.signature(provider.compute).parameters:
                    frame = provider.compute(
                        request, datasets, resolved.parameters, dependencies
                    )
                else:
                    frame = provider.compute(request, datasets, resolved.parameters)
                definition.validate_output(frame, resolved.parameters)
                validate_feature_timeline(definition, frame, resolved.parameters)
                frame.attrs.update(
                    feature_cache_hit=False,
                    feature_cache_key=identity,
                    effective_warmup_bars=resolved.effective_warmup_bars,
                )
                if cache:
                    cache.store(definition, request, identity, frame)
            frames[definition.name], identities[definition.name] = frame, identity
        return frames
