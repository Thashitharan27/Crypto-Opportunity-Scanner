"""Persistent, content-addressed L3 cache for validated prepared runs."""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping

import duckdb
import numpy as np
import pandas as pd

from .features import production_feature_registry
from .features.market_regime import POLICY_MARKET_FEATURE_NAME
from .prepared_backtest import (
    PreparedBacktestFrame,
    ResearchContext,
    from_data_lake_bundle,
    intrabar_from_data_lake_bundle,
)

PREPARED_CACHE_FORMAT_VERSION = 1
PREPARED_CONTRACT_VERSION = 1


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def prepared_policy_inputs(config, *, feature_registry=None) -> dict[str, object]:
    """Only definitions/configuration physically materialized into the prepared frame.

    TP/SL, trailing, fees, slippage, portfolio behavior and reporting/UI values
    are intentionally absent. Profile momentum lookbacks are present because
    their arrays are physically stored in the prepared contract. The authoritative
    policy-market feature definition hash is also included so implementation,
    version, schema, causality or dependency-contract changes invalidate L3.
    """
    if config is None:
        return {"policy_features": False}
    registry = feature_registry if feature_registry is not None else production_feature_registry()
    policy_definition = registry.definition_hash(
        [POLICY_MARKET_FEATURE_NAME]
    )
    return {
        "policy_features": True,
        "policy_market_definition": policy_definition,
        "market_regime_method": config.market_regime_method,
        "bull_regime_lookback_days": config.bull_regime_lookback_days,
        "bull_regime_return_threshold": config.bull_regime_return_threshold,
        "structural_regime_sma_days": config.structural_regime_sma_days,
        "structural_regime_slope_lookback_days": config.structural_regime_slope_lookback_days,
        "momentum_lookback_hours": sorted(
            {
                int(profile.momentum_lookback_hours)
                for profile in config.strategy_profiles.values()
            }
        ),
    }


class PreparedRunCache:
    """Parquet plus JSON-manifest cache; manifests are committed last."""

    def __init__(self, cache_root: Path, *, contract_version=PREPARED_CONTRACT_VERSION):
        self.root = Path(cache_root) / "prepared"
        self.contract_version = int(contract_version)

    def identity(
        self,
        *,
        request_identity: str,
        feature_identities: Mapping[str, str],
        canonical_identities: Mapping[str, str],
        prepared_inputs: Mapping[str, object],
    ) -> str:
        return _digest(
            {
                "format_version": PREPARED_CACHE_FORMAT_VERSION,
                "contract_version": self.contract_version,
                "request_identity": request_identity,
                "feature_identities": dict(sorted(feature_identities.items())),
                "canonical_identities": dict(sorted(canonical_identities.items())),
                "prepared_inputs": prepared_inputs,
            }
        )

    def paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / f"v{PREPARED_CACHE_FORMAT_VERSION}" / key[:2]
        return directory / f"{key}.parquet", directory / f"{key}.json"

    def store(
        self,
        key: str,
        frame: PreparedBacktestFrame,
        *,
        provenance: Mapping[str, object],
    ) -> None:
        parquet, manifest = self.paths(key)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        temp_parquet = parquet.with_suffix(".tmp.parquet")
        temp_manifest = manifest.with_suffix(".tmp.json")
        columns, layout = {}, {"arrays": [], "momentum": [], "research": []}
        scalar = {"strategy_interval_ns": frame.strategy_interval.value}
        for field in fields(frame):
            name, value = field.name, getattr(frame, field.name)
            if isinstance(value, np.ndarray):
                column = f"array__{name}"
                columns[column] = value
                layout["arrays"].append([name, column])
        for hours, value in sorted(frame.momentum_returns_by_hours.items()):
            column = f"momentum__{hours}"
            columns[column] = value
            layout["momentum"].append([hours, column])
        for index, block in enumerate(frame.research):
            available = f"research__{index}__available_at"
            columns[available] = block.available_at
            item = {"name": block.name, "available_at": available, "values": []}
            for value_index, (name, value) in enumerate(sorted(block.values.items())):
                column = f"research__{index}__value__{value_index}"
                columns[column] = value
                item["values"].append([name, column])
            layout["research"].append(item)
        table = pd.DataFrame(columns)
        for path in (temp_parquet, temp_manifest):
            path.unlink(missing_ok=True)
        with duckdb.connect() as con:
            con.register("prepared_frame", table)
            con.execute(
                "COPY prepared_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(temp_parquet)],
            )
        metadata = {
            "cache_format_version": PREPARED_CACHE_FORMAT_VERSION,
            "prepared_contract_version": self.contract_version,
            "l3_key": key,
            "row_count": len(frame),
            "strategy_interval": str(frame.strategy_interval),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout": layout,
            "scalars": scalar,
            **dict(provenance),
        }
        temp_manifest.write_text(
            json.dumps(metadata, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temp_parquet.replace(parquet)
        temp_manifest.replace(manifest)

    def load(self, key: str) -> PreparedBacktestFrame | None:
        parquet, manifest = self.paths(key)
        if not parquet.is_file() or not manifest.is_file():
            return None
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                metadata["l3_key"] != key
                or metadata["cache_format_version"] != PREPARED_CACHE_FORMAT_VERSION
                or metadata["prepared_contract_version"] != self.contract_version
            ):
                return None
            with duckdb.connect() as con:
                table = con.read_parquet(str(parquet)).df()
            if len(table) != metadata["row_count"]:
                return None
            layout = metadata["layout"]
            kwargs = {
                name: table[column].to_numpy() for name, column in layout["arrays"]
            }
            kwargs["strategy_interval"] = pd.Timedelta(
                metadata["scalars"]["strategy_interval_ns"], unit="ns"
            )
            kwargs["momentum_returns_by_hours"] = {
                int(hours): table[column].to_numpy()
                for hours, column in layout["momentum"]
            }
            kwargs["research"] = tuple(
                ResearchContext(
                    item["name"],
                    table[item["available_at"]].to_numpy(),
                    {
                        name: table[column].to_numpy()
                        for name, column in item["values"]
                    },
                )
                for item in layout["research"]
            )
            return PreparedBacktestFrame(**kwargs)
        except Exception:
            return None

    def get_or_build(
        self,
        key: str,
        builder: Callable[[], PreparedBacktestFrame],
        *,
        provenance,
    ):
        cached = self.load(key)
        if cached is not None:
            return cached, True
        frame = builder()  # constructor validation remains authoritative
        if not isinstance(frame, PreparedBacktestFrame):
            raise TypeError("builder must return PreparedBacktestFrame")
        self.store(key, frame, provenance=provenance)
        return frame, False


def bundle_prepared_identity(cache: PreparedRunCache, bundle, config, *, feature_registry=None):
    features = {}
    frames = {
        "core_directional": bundle.technical_features,
        "production_market_context": bundle.context_features,
        **bundle.research_features,
    }
    if bundle.support_resistance_features is not None:
        frames["support_resistance"] = bundle.support_resistance_features
    for name, frame in frames.items():
        key = frame.attrs.get("feature_cache_key")
        if key:
            features[name] = key
    canonical = {
        "strategy_ohlcv": bundle.strategy.attrs.get("canonical_source_identity", "")
    }
    benchmark = getattr(bundle, "structural_benchmark", None)
    if benchmark is not None:
        canonical["structural_benchmark"] = benchmark.attrs.get(
            "canonical_source_identity", ""
        )
    request_identity = bundle.request.feature_scope_key()
    inputs = prepared_policy_inputs(config, feature_registry=feature_registry)
    key = cache.identity(
        request_identity=request_identity,
        feature_identities=features,
        canonical_identities=canonical,
        prepared_inputs=inputs,
    )
    provenance = {
        "request_identity": request_identity,
        "input_l2_identities": features,
        "canonical_identities": canonical,
        "prepared_input_identity": _digest(inputs),
    }
    return key, provenance


def prepare_bundle_with_cache(cache_root: Path, bundle, config):
    """Return validated native inputs, reusing persistent L3 when possible.

    The current bundle still establishes the authoritative L1/L2 identities.
    On an L3 hit this function does not execute feature providers or rebuild the
    PreparedBacktestFrame; it only projects current intrabar execution data,
    which is deliberately outside L3.
    """
    cache = PreparedRunCache(cache_root)
    key, provenance = bundle_prepared_identity(cache, bundle, config)
    prepared, hit = cache.get_or_build(
        key,
        lambda: from_data_lake_bundle(bundle, config)[0],
        provenance=provenance,
    )
    intrabar = intrabar_from_data_lake_bundle(bundle)
    if intrabar is not None:
        intrabar.validate_compatible(prepared)
    return prepared, intrabar, hit, key
