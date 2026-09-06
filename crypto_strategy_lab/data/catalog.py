"""DuckDB archive catalog for the external Binance data lake."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

import duckdb

from .query import DataRequest
from .schemas import ArchiveRecord, Coverage, DatasetKind, MarketKind


class DataCatalog:
    """Small local index of immutable raw archive metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # Run-scoped observation of catalog selections. These are metadata
        # records already selected by the normal loading path; reporters never
        # need to reopen an archive.
        self.selected_records: list[ArchiveRecord] = []
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS archives (
                    raw_root VARCHAR NOT NULL,
                    path VARCHAR PRIMARY KEY,
                    exchange VARCHAR NOT NULL,
                    market VARCHAR NOT NULL,
                    dataset VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    interval VARCHAR,
                    frequency VARCHAR NOT NULL,
                    period_start TIMESTAMPTZ,
                    period_end TIMESTAMPTZ,
                    size_bytes BIGINT NOT NULL,
                    mtime_ns BIGINT NOT NULL,
                    fingerprint VARCHAR NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_archives_lookup "
                "ON archives(raw_root, market, dataset, symbol, interval)"
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS archive_scan_directories (
                    raw_root VARCHAR NOT NULL,
                    path VARCHAR NOT NULL,
                    mtime_ns BIGINT NOT NULL,
                    PRIMARY KEY (raw_root, path)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS archive_directory_members (
                    raw_root VARCHAR NOT NULL,
                    path VARCHAR NOT NULL,
                    directory_path VARCHAR NOT NULL,
                    PRIMARY KEY (raw_root, path)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_archive_members_directory "
                "ON archive_directory_members(raw_root, directory_path)"
            )

    @staticmethod
    def _archive_rows(root_text: str, records: list[ArchiveRecord]) -> list[tuple]:
        return [
            (
                root_text,
                str(record.path.absolute()),
                record.exchange,
                record.market.value,
                record.dataset.value,
                record.symbol,
                record.interval,
                record.frequency,
                record.period_start,
                record.period_end,
                record.size_bytes,
                record.mtime_ns,
                record.fingerprint,
            )
            for record in records
        ]

    @staticmethod
    def _insert_archive_rows(
        con: duckdb.DuckDBPyConnection,
        root_text: str,
        records: list[ArchiveRecord],
    ) -> None:
        rows = DataCatalog._archive_rows(root_text, records)
        if rows:
            con.executemany(
                """
                INSERT INTO archives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            con.executemany(
                """
                INSERT INTO archive_directory_members VALUES (?, ?, ?)
                """,
                [
                    (
                        root_text,
                        str(record.path.absolute()),
                        str(record.path.absolute().parent),
                    )
                    for record in records
                ],
            )

    def sync_root(
        self,
        raw_root: Path,
        records: list[ArchiveRecord],
        *,
        directory_mtimes: Mapping[str, int] | None = None,
    ) -> None:
        """Replace catalog rows for one raw root without touching raw files."""

        root_text = str(Path(raw_root).resolve())
        with self._connect() as con:
            con.execute("BEGIN")
            try:
                con.execute("DELETE FROM archives WHERE raw_root = ?", [root_text])
                con.execute(
                    "DELETE FROM archive_directory_members WHERE raw_root = ?",
                    [root_text],
                )
                con.execute(
                    "DELETE FROM archive_scan_directories WHERE raw_root = ?",
                    [root_text],
                )
                self._insert_archive_rows(con, root_text, records)
                if directory_mtimes is not None:
                    rows = [
                        (root_text, str(Path(path).absolute()), int(mtime_ns))
                        for path, mtime_ns in directory_mtimes.items()
                    ]
                    if rows:
                        con.executemany(
                            """
                            INSERT INTO archive_scan_directories VALUES (?, ?, ?)
                            """,
                            rows,
                        )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def directory_snapshot(self, raw_root: Path) -> dict[str, int]:
        root_text = str(Path(raw_root).resolve())
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT path, mtime_ns
                FROM archive_scan_directories
                WHERE raw_root = ?
                """,
                [root_text],
            ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def archive_count(self, raw_root: Path) -> int:
        root_text = str(Path(raw_root).resolve())
        with self._connect() as con:
            row = con.execute(
                "SELECT count(*) FROM archives WHERE raw_root = ?",
                [root_text],
            ).fetchone()
        return int(row[0])

    def apply_incremental_root(
        self,
        raw_root: Path,
        *,
        replacements: Mapping[str, list[ArchiveRecord]],
        removed_directories: set[str],
        directory_mtimes: Mapping[str, int],
    ) -> None:
        """Apply changed directories without replacing the complete archive catalog."""

        root_text = str(Path(raw_root).resolve())
        separator = os.sep
        with self._connect() as con:
            con.execute("BEGIN")
            try:
                # Removed subtree roots invalidate all archive members beneath
                # them. starts_with avoids wildcard escaping problems in Windows
                # paths that may contain '%' or '_'.
                for removed in sorted(removed_directories):
                    removed_text = str(Path(removed).absolute())
                    prefix = removed_text + separator
                    con.execute(
                        """
                        DELETE FROM archives
                        WHERE path IN (
                            SELECT path
                            FROM archive_directory_members
                            WHERE raw_root = ?
                              AND (
                                directory_path = ?
                                OR starts_with(directory_path, ?)
                              )
                        )
                        """,
                        [root_text, removed_text, prefix],
                    )
                    con.execute(
                        """
                        DELETE FROM archive_directory_members
                        WHERE raw_root = ?
                          AND (
                            directory_path = ?
                            OR starts_with(directory_path, ?)
                          )
                        """,
                        [root_text, removed_text, prefix],
                    )

                # Reconcile only direct archive files in directories whose mtime
                # changed (plus every directory in a newly discovered subtree).
                for directory, records in replacements.items():
                    directory_text = str(Path(directory).absolute())
                    con.execute(
                        """
                        DELETE FROM archives
                        WHERE path IN (
                            SELECT path
                            FROM archive_directory_members
                            WHERE raw_root = ? AND directory_path = ?
                        )
                        """,
                        [root_text, directory_text],
                    )
                    con.execute(
                        """
                        DELETE FROM archive_directory_members
                        WHERE raw_root = ? AND directory_path = ?
                        """,
                        [root_text, directory_text],
                    )
                    self._insert_archive_rows(con, root_text, records)

                # Directory metadata is small compared with the archive catalog,
                # so replacing this index keeps incremental state simple and
                # transactional.
                con.execute(
                    "DELETE FROM archive_scan_directories WHERE raw_root = ?",
                    [root_text],
                )
                directory_rows = [
                    (root_text, str(Path(path).absolute()), int(mtime_ns))
                    for path, mtime_ns in directory_mtimes.items()
                ]
                if directory_rows:
                    con.executemany(
                        """
                        INSERT INTO archive_scan_directories VALUES (?, ?, ?)
                        """,
                        directory_rows,
                    )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def records_for(
        self,
        raw_root: Path,
        request: DataRequest,
        dataset: DatasetKind,
        interval: str | None,
    ) -> list[ArchiveRecord]:
        root_text = str(Path(raw_root).resolve())
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT raw_root, path, market, dataset, symbol, interval, frequency,
                       period_start, period_end, size_bytes, mtime_ns, fingerprint,
                       exchange
                FROM archives
                WHERE raw_root = ?
                  AND exchange = ?
                  AND market = ?
                  AND dataset = ?
                  AND symbol = ?
                  AND (? IS NULL OR interval = ?)
                  AND (period_end IS NULL OR period_end > ?)
                  AND (period_start IS NULL OR period_start < ?)
                ORDER BY period_start NULLS FIRST, path
                """,
                [
                    root_text,
                    request.exchange,
                    request.market.value,
                    dataset.value,
                    request.symbol,
                    interval,
                    interval,
                    request.start,
                    request.end,
                ],
            ).fetchall()
        result = [
            ArchiveRecord(
                raw_root=Path(row[0]),
                path=Path(row[1]),
                market=MarketKind(row[2]),
                dataset=DatasetKind(row[3]),
                symbol=row[4],
                interval=row[5],
                frequency=row[6],
                period_start=row[7],
                period_end=row[8],
                size_bytes=row[9],
                mtime_ns=row[10],
                fingerprint=row[11],
                exchange=row[12],
            )
            for row in rows
        ]
        self.selected_records.extend(result)
        return result

    def reset_selected_records(self) -> None:
        self.selected_records.clear()

    def coverage(
        self,
        raw_root: Path,
        *,
        market: MarketKind,
        dataset: DatasetKind,
        symbol: str,
        interval: str | None = None,
    ) -> Coverage:
        root_text = str(Path(raw_root).resolve())
        with self._connect() as con:
            row = con.execute(
                """
                SELECT min(period_start), max(period_end), count(*)
                FROM archives
                WHERE raw_root = ? AND market = ? AND dataset = ? AND symbol = ?
                  AND (? IS NULL OR interval = ?)
                """,
                [root_text, market.value, dataset.value, symbol.upper(), interval, interval],
            ).fetchone()
        return Coverage(row[0], row[1], int(row[2]))

    def inventory(self, raw_root: Path, *, market: MarketKind) -> list[dict]:
        """Return presentation-safe availability metadata from the catalog only.

        This intentionally exposes no archive paths. Clients such as the GUI can
        discover symbols and coverage without walking or opening the immutable raw
        lake.
        """
        root_text = str(Path(raw_root).resolve())
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT exchange, symbol, dataset, interval, min(period_start),
                       max(period_end), count(*)
                FROM archives
                WHERE raw_root = ? AND market = ?
                GROUP BY exchange, symbol, dataset, interval
                ORDER BY symbol, dataset, interval NULLS FIRST
                """,
                [root_text, market.value],
            ).fetchall()
        return [
            {"exchange": row[0], "symbol": row[1], "dataset": row[2],
             "interval": row[3], "first_period": row[4], "last_period": row[5],
             "archive_count": int(row[6])}
            for row in rows
        ]
