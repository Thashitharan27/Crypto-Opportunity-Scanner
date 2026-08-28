"""Deterministic source identities derived without materializing market data."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable

from .schemas import ArchiveRecord, DatasetKind

CANONICAL_IDENTITY_VERSION = 1


def canonical_partition_identity(record: ArchiveRecord, contract: dict[str, object]) -> str:
    """Central identity for one canonicalized immutable raw partition."""
    payload = {
        "identity_version": CANONICAL_IDENTITY_VERSION,
        "exchange": record.exchange,
        "market": record.market.value,
        "dataset": record.dataset.value,
        "symbol": record.symbol,
        "interval": record.interval,
        "raw_fingerprint": record.fingerprint,
        "canonical_contract": contract,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceSignature:
    """Identity of the catalog partitions contributing to a requested slice.

    Archive fingerprints remain the primary content identity.  Coverage and file
    metadata are included so catalog corrections also invalidate derived data;
    the archive filename alone is never treated as source identity.
    """

    dataset: DatasetKind
    digest: str
    partition_count: int
    identity_version: int = 2

    @classmethod
    def from_records(
        cls,
        dataset: DatasetKind,
        records: Iterable[ArchiveRecord],
    ) -> "SourceSignature":
        partitions = [
            {
                "exchange": record.exchange,
                "market": record.market.value,
                "dataset": record.dataset.value,
                "symbol": record.symbol,
                "interval": record.interval,
                "frequency": record.frequency,
                "period_start": record.period_start,
                "period_end": record.period_end,
                "size_bytes": record.size_bytes,
                "mtime_ns": record.mtime_ns,
                "fingerprint": record.fingerprint,
            }
            for record in records
        ]
        partitions.sort(
            key=lambda item: json.dumps(item, sort_keys=True, default=str)
        )
        payload = {"signature_version": 1, "dataset": dataset.value, "partitions": partitions}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return cls(dataset, sha256(encoded).hexdigest(), len(partitions))

    def cache_identity(self) -> str:
        return f"canonical-v{self.identity_version}:{self.dataset.value}:{self.partition_count}:{self.digest}"

    @classmethod
    def from_canonical_identities(cls, dataset, identities):
        values = sorted(identities)
        raw = json.dumps({"dataset": dataset.value, "canonical": values}, sort_keys=True).encode()
        return cls(dataset, sha256(raw).hexdigest(), len(values))
