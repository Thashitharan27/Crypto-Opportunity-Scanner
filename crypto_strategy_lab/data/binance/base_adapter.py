"""Shared helpers for reading raw Binance ZIP/CSV archives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator
import zipfile

import pandas as pd

from ..schemas import ArchiveRecord, DatasetKind


class BinanceArchiveAdapter(ABC):
    """Exchange-specific schema adapter; never computes research features."""

    dataset: DatasetKind
    # These are deliberately dataset-adapter properties: changing the kline
    # normalizer must not invalidate funding (and vice versa).
    normalizer_version: int = 1
    canonical_schema_version: int = 1

    def canonical_contract(self) -> dict[str, object]:
        return {
            "dataset": self.dataset.value,
            "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
            "normalizer_version": self.normalizer_version,
            "schema_version": self.canonical_schema_version,
        }

    @abstractmethod
    def read(self, record: ArchiveRecord) -> pd.DataFrame:
        """Normalize one raw source archive into a canonical DataFrame."""


@contextmanager
def open_csv_stream(path: Path) -> Iterator[BinaryIO]:
    """Yield a binary CSV stream from either a .csv or Binance .zip archive."""

    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV member found in Binance archive: {path}")
            member = max(members, key=lambda name: archive.getinfo(name).file_size)
            with archive.open(member, "r") as stream:
                yield stream
        return
    with path.open("rb") as stream:
        yield stream


def timestamp_series(values: pd.Series) -> pd.Series:
    """Parse Binance timestamps expressed as ms/us/ns integers or UTC text."""

    numeric = pd.to_numeric(values, errors="coerce")
    if bool(numeric.notna().all()):
        magnitude = float(numeric.abs().median())
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        else:
            unit = "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="raise")
    return pd.to_datetime(values, utc=True, errors="raise")


def normalize_header_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Binance header spelling without changing row values."""

    result = frame.copy()
    result.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in result.columns
    ]
    return result
