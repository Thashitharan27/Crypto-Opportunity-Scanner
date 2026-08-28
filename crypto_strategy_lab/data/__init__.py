"""Centralized market-data access for Crypto Strategy Lab."""

from .query import DataRequest
from .schemas import ArchiveRecord, Coverage, DatasetKind, MarketKind
from .store import DataNotAvailableError, MarketDataStore
from .source_identity import SourceSignature
from .order_book import (BOOK_SNAPSHOT_SCHEMA_VERSION, OrderBookSnapshotResult,
                         OrderBookSnapshotStore)
from .quality import (
    DataQualityCache, DataQualityError, DataQualityIssue, DataQualityReport,
    DataQualityStatus, DatasetQualityReport, DatasetValidationContract,
    MissingCoverageRange,
    classify_archive_overlap, validate_dataset, validate_feature_timeline,
)

__all__ = [
    "ArchiveRecord",
    "Coverage",
    "DataNotAvailableError",
    "DataRequest",
    "DatasetKind",
    "MarketDataStore",
    "MarketKind",
    "SourceSignature",
    "BOOK_SNAPSHOT_SCHEMA_VERSION", "OrderBookSnapshotResult", "OrderBookSnapshotStore",
    "DataQualityCache", "DataQualityError", "DataQualityIssue", "DataQualityReport",
    "DataQualityStatus", "DatasetQualityReport", "DatasetValidationContract",
    "MissingCoverageRange",
    "classify_archive_overlap", "validate_dataset", "validate_feature_timeline",
]
