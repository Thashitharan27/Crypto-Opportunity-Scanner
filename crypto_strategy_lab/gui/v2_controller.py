"""Application boundary for the v2 GUI.

Qt widgets depend on this module rather than data-lake implementation details.
Only this controller composes and invokes :class:`ResearchRunner`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from crypto_strategy_lab.data import (
    DataQualityReport,
    DataRequest,
    DatasetKind,
    MarketDataStore,
    MarketKind,
)
from crypto_strategy_lab.bayesian_sampling_reporting import BayesianSamplingCsvManifestReporter
from crypto_strategy_lab.data_lake_config import ResearchRunConfig, load_data_lake_config
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.prepared_cache import PreparedRunCache
from crypto_strategy_lab.progress import emit_progress
from crypto_strategy_lab.research_adapters import NativeSimulator, NativeStrategyPolicy
from crypto_strategy_lab.research_runner import ResearchRunner
from crypto_strategy_lab.run_manifest import artifact_path, load_completed_manifest


def _utc(value):
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    return (timestamp.tz_localize("UTC") if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")).to_pydatetime()


def _market_kind(value: MarketKind | str) -> MarketKind:
    """Normalize Qt/user-data strings at the GUI application boundary."""
    return value if isinstance(value, MarketKind) else MarketKind(str(value))


@dataclass(frozen=True)
class GuiResearchRequest:
    exchange: str
    market: MarketKind | str
    symbol: str
    period_start: datetime
    period_end: datetime
    strategy_timeframe: str
    intrabar_timeframe: str | None

    def to_data_request(self, datasets=(DatasetKind.KLINES,)) -> DataRequest:
        return DataRequest(symbol=self.symbol, start=self.period_start, end=self.period_end,
                           strategy_interval=self.strategy_timeframe,
                           intrabar_interval=self.intrabar_timeframe, datasets=tuple(datasets),
                           market=_market_kind(self.market), exchange=self.exchange)


class CatalogStatusService:
    """Read-only catalog facade; it never discovers or opens raw archives."""
    def __init__(self, store: MarketDataStore):
        self._store = store

    def inventory(self, market=MarketKind.FUTURES_UM) -> list[dict]:
        return self._store.catalog.inventory(self._store.raw_root, market=_market_kind(market))

    def symbols(self, market=MarketKind.FUTURES_UM) -> list[str]:
        return sorted({row["symbol"] for row in self.inventory(market)})

    def coverage(self, request: GuiResearchRequest) -> list[dict]:
        rows = [r for r in self.inventory(request.market) if r["symbol"] == request.symbol.upper()]
        for row in rows:
            first = _utc(row["first_period"])
            last = _utc(row["last_period"])
            row["first_period"], row["last_period"] = first, last
            row["state"] = ("UNAVAILABLE" if not row["archive_count"] else
                            "PARTIAL" if first is None or last is None or
                            first > request.period_start or last < request.period_end else "AVAILABLE")
        return rows


class CompletedRunReader:
    """Resolve summaries and navigation solely through the canonical manifest."""
    def read(self, run_dir: Path) -> tuple[dict, dict]:
        manifest = load_completed_manifest(Path(run_dir))
        summary = json.loads(self.artifact_path(run_dir, manifest, "summary").read_text(encoding="utf-8"))
        return manifest, summary

    @staticmethod
    def artifact_path(run_dir: Path, manifest: dict, name: str) -> Path:
        return artifact_path(Path(run_dir), manifest, name, verify=True)


class GuiApplicationService:
    def __init__(self, raw_root: Path, cache_root: Path,
                 runner_factory: Callable | None = None):
        self.raw_root, self.cache_root = Path(raw_root), Path(cache_root)
        self.store = MarketDataStore(self.raw_root, self.cache_root)
        self.catalog = CatalogStatusService(self.store)
        self.completed_runs = CompletedRunReader()
        self._runner_factory = runner_factory
        self.progress_callback = None

    def refresh_catalog(self) -> int:
        """Refresh discovery only at the application-service boundary."""
        return self.store.refresh_catalog()

    def required_data_quality(self, request: GuiResearchRequest) -> DataQualityReport:
        """Validate the exact required candle slices selected in the GUI.

        Catalog metadata can establish archive-level availability without opening
        immutable raw files, but it cannot prove that every expected candle exists
        inside an archive. This explicit preflight uses the normal Task-12 quality
        cache and canonical adapters, so a cold check performs the validation once
        and a warm check is metadata-only. It never runs strategy features or the
        simulator.
        """
        callback = getattr(self, "progress_callback", None)
        intervals = tuple(dict.fromkeys(
            interval for interval in (
                request.strategy_timeframe,
                request.intrabar_timeframe,
            ) if interval
        ))
        emit_progress(
            callback,
            kind="stage",
            phase="required_data_validation",
            label="VALIDATING SELECTED CANDLE RANGE",
            detail="Checking actual candle continuity before strategy execution.",
        )
        data_request = request.to_data_request()
        reports = tuple(
            self.store.data_quality_report(
                data_request,
                DatasetKind.KLINES,
                interval=interval,
                required=True,
            )
            for interval in intervals
        )
        return DataQualityReport(reports)

    def _runner(self, output_root: Path):
        if self._runner_factory:
            return self._runner_factory(output_root)
        return ResearchRunner(self.store, production_feature_registry(),
                              PreparedRunCache(self.cache_root), NativeStrategyPolicy(),
                              NativeSimulator(), (BayesianSamplingCsvManifestReporter(output_root),))

    def run(self, request: GuiResearchRequest, config: ResearchRunConfig):
        config.validate()
        callback = getattr(self, "progress_callback", None)
        # The observer is attached only for this run. Data/cache layers can emit
        # partition-level events without depending on Qt or changing their public
        # result contracts.
        self.store.progress_callback = callback
        try:
            return self._runner(Path(config.reporting.output_dir)).run(
                request.to_data_request(), config
            )
        except Exception as exc:
            emit_progress(
                callback,
                kind="failed",
                phase="failed",
                label="RUN FAILED",
                detail=str(exc),
            )
            raise
        finally:
            self.store.progress_callback = None

    @staticmethod
    def save_config(path: Path, config: ResearchRunConfig) -> None:
        config.validate()
        Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_config(path: Path) -> ResearchRunConfig:
        return load_data_lake_config(path)