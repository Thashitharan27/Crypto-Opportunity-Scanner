"""Disposable Parquet cache for versioned derived feature frames."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import duckdb
import pandas as pd

from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.source_identity import SourceSignature

from .base import FeatureDefinition


class FeatureFrameCache:
    """Persist prepared features without ever mutating the raw Binance lake."""

    def __init__(self, cache_root: Path) -> None:
        self.root = Path(cache_root) / "features"
        self.format_version = 2

    @staticmethod
    def _source_signature(canonical_source: pd.DataFrame) -> str:
        if "source_fingerprint" not in canonical_source.columns:
            return "no-source-fingerprint"
        values = sorted(
            str(value)
            for value in canonical_source["source_fingerprint"].dropna().unique().tolist()
        )
        return sha256("|".join(values).encode("utf-8")).hexdigest()

    def key(
        self,
        definition: FeatureDefinition,
        request: DataRequest,
        parameters: Mapping[str, object],
        canonical_source: pd.DataFrame,
        *,
        dependency_keys: Sequence[str] = (),
        additional_sources: Sequence[pd.DataFrame] = (),
    ) -> str:
        payload = {
            "feature": definition.name,
            "version": definition.version,
            "format_version": self.format_version,
            "request_scope": request.feature_scope_key(),
            "parameters": dict(parameters),
            "source": self._source_signature(canonical_source),
            "additional_sources": [
                self._source_signature(source) for source in additional_sources
            ],
            "dependencies": list(dependency_keys),
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return sha256(raw).hexdigest()

    def key_from_signatures(
        self,
        definition: FeatureDefinition,
        request: DataRequest,
        parameters: Mapping[str, object],
        source_signatures: Sequence[SourceSignature],
        *,
        dependency_keys: Sequence[str] = (),
    ) -> str:
        """Build a key from catalog metadata, before source frames are loaded."""

        payload = {
            "feature": definition.name,
            "version": definition.version,
            "format_version": self.format_version,
            "request_scope": request.feature_scope_key(),
            "parameters": dict(parameters),
            "sources": [signature.cache_identity() for signature in source_signatures],
            "dependencies": list(dependency_keys),
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return sha256(raw).hexdigest()

    def paths(
        self,
        definition: FeatureDefinition,
        request: DataRequest,
        key: str,
    ) -> tuple[Path, Path]:
        directory = (
            self.root
            / definition.name
            / f"v{definition.version}"
            / request.market.value
            / request.symbol
            / request.strategy_interval
        )
        return directory / f"{key}.parquet", directory / f"{key}.json"

    def load(
        self,
        definition: FeatureDefinition,
        request: DataRequest,
        key: str,
    ) -> pd.DataFrame | None:
        parquet_path, metadata_path = self.paths(definition, request, key)
        if not parquet_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("cache_format_version") != self.format_version
                or metadata.get("feature_name") != definition.name
                or metadata.get("feature_version") != definition.version
                or metadata.get("feature_cache_key") != key
            ):
                return None
            with duckdb.connect() as con:
                frame = con.read_parquet(str(parquet_path)).df()
        except Exception:
            # Cache is disposable. A broken/incomplete entry simply becomes a miss.
            return None
        for column in ("timestamp", "available_at"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        frame.attrs.update(metadata.get("frame_attrs", {}))
        frame.attrs["feature_cache_hit"] = True
        frame.attrs["feature_cache_key"] = key
        return frame

    def store(
        self,
        definition: FeatureDefinition,
        request: DataRequest,
        key: str,
        frame: pd.DataFrame,
    ) -> None:
        parquet_path, metadata_path = self.paths(definition, request, key)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_parquet = parquet_path.with_suffix(".tmp.parquet")
        temporary_metadata = metadata_path.with_suffix(".tmp.json")
        for path in (temporary_parquet, temporary_metadata):
            path.unlink(missing_ok=True)

        with duckdb.connect() as con:
            con.register("feature_frame", frame)
            escaped = str(temporary_parquet).replace("'", "''")
            con.execute(
                f"COPY feature_frame TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        metadata = {
            "cache_format_version": self.format_version,
            "feature_name": definition.name,
            "feature_version": definition.version,
            "feature_cache_key": key,
            "frame_attrs": {
                str(name): value
                for name, value in frame.attrs.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
        }
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        temporary_parquet.replace(parquet_path)
        temporary_metadata.replace(metadata_path)
