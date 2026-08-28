"""Disposable cache layout for normalized market data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .schemas import ArchiveRecord


CANONICAL_CACHE_FORMAT_VERSION = 1


def stat_fingerprint(path: Path) -> str:
    """Fast source fingerprint without hashing multi-gigabyte archive contents."""

    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """All local cache locations owned by Crypto Strategy Lab."""

    root: Path

    @property
    def catalog_db(self) -> Path:
        return self.root / "catalog" / "catalog.duckdb"

    @property
    def market_root(self) -> Path:
        return self.root / "market"

    def archive_parquet(self, record: ArchiveRecord, canonical_identity: str = "legacy") -> Path:
        interval = record.interval or "no_interval"
        stem = record.path.stem.replace(".csv", "")
        return (
            self.market_root
            / record.market.value
            / record.dataset.value
            / record.symbol
            / interval
            / record.frequency
            / f"{stem}-{record.fingerprint[:12]}-{canonical_identity[:16]}.parquet"
        )

    def archive_manifest(self, record: ArchiveRecord, canonical_identity: str) -> Path:
        return self.archive_parquet(record, canonical_identity).with_suffix(".json")

    def ensure(self) -> None:
        self.catalog_db.parent.mkdir(parents=True, exist_ok=True)
        self.market_root.mkdir(parents=True, exist_ok=True)
