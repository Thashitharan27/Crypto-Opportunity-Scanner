"""Thin application boundary for the Opportunity Scanner workspace.

Widgets only create :class:`OpportunityScanRequest` values.  An injected Task
1--7 orchestration callable owns the pipeline, while this module owns the
canonical, integrity-checked Task-7 result reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from crypto_strategy_lab.data.binance.historical_discovery import HistoricalDiscoveryConfig
from crypto_strategy_lab.data.binance.historical_discovery import (
    DiscoveryDecisionTime, discover_historical_universe,
)
from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionState, BinanceDataHubBackend, SelectiveCandleAcquirer,
    SelectiveCandleAcquisitionConfig,
)
from crypto_strategy_lab.data.binance.universe import (
    BinanceUsdMDiscoveryClient, DiscoveryConfig, scan_universe,
)
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.data.store import MarketDataStore
from crypto_strategy_lab.data.timing import interval_to_timedelta
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.final_candidates import (
    FinalCandidateBoundaryConfig, OpportunityModelRef, build_final_candidate_set,
)
from crypto_strategy_lab.opportunity_reporting import (
    OpportunityScanPublicationInput, publish_opportunity_scan,
)
from crypto_strategy_lab.opportunity_scoring import (
    OpportunityScoringConfig, OpportunityScoringModelDefinition,
    score_opportunities, snapshot_from_registry_features,
)
from crypto_strategy_lab.rich_data_acquisition import (
    RichDataAcquisitionConfig, SelectiveRichDataAcquirer,
)
from crypto_strategy_lab.run_manifest import artifact_path, load_completed_manifest


@dataclass(frozen=True, slots=True)
class OpportunityScanRequest:
    """Existing Task 1--6 configurations plus an explicit scan boundary."""

    market: MarketKind
    mode: str
    decision_time: datetime | None
    live_discovery: DiscoveryConfig
    historical_discovery: HistoricalDiscoveryConfig
    candle_acquisition: SelectiveCandleAcquisitionConfig
    scoring: OpportunityScoringConfig | None
    final_candidates: FinalCandidateBoundaryConfig
    rich_data: RichDataAcquisitionConfig

    def __post_init__(self) -> None:
        mode = self.mode.upper()
        if mode not in {"LIVE", "HISTORICAL"}:
            raise ValueError("scan mode must be LIVE or HISTORICAL")
        object.__setattr__(self, "mode", mode)
        if mode == "HISTORICAL":
            if self.decision_time is None or self.decision_time.tzinfo is None:
                raise ValueError("historical decision time must be timezone-aware")
            object.__setattr__(self, "decision_time", self.decision_time.astimezone(timezone.utc))
        elif self.decision_time is not None:
            raise ValueError("live discovery obtains its decision time from the observed snapshot")


@dataclass(frozen=True, slots=True)
class CompletedOpportunityScan:
    run_dir: Path
    manifest: dict
    summary: dict
    universe: pd.DataFrame
    preliminary: pd.DataFrame
    final: pd.DataFrame
    readiness: pd.DataFrame
    scores: pd.DataFrame


class OpportunityScanResultReader:
    """Read only paths declared by a completed OPPORTUNITY_SCAN manifest."""

    def read(self, run_dir: Path) -> CompletedOpportunityScan:
        directory = Path(run_dir)
        manifest = load_completed_manifest(directory)
        if manifest.get("run_type") != "OPPORTUNITY_SCAN":
            raise ValueError("completed run is not an OPPORTUNITY_SCAN")

        def csv(name: str, *, optional: bool = False) -> pd.DataFrame:
            if optional and name not in manifest.get("artifacts", {}):
                return pd.DataFrame()
            return pd.read_csv(artifact_path(directory, manifest, name, verify=True))

        summary = json.loads(
            artifact_path(directory, manifest, "opportunity_summary", verify=True)
            .read_text(encoding="utf-8")
        )
        universe = csv("universe_snapshot")
        preliminary = csv("preliminary_candidates")
        scores = csv("opportunity_scores", optional=True)
        # Discovery impact values remain publication-owned; join for display only.
        if not preliminary.empty and not universe.empty:
            metrics = [c for c in ("symbol", "range_percent", "absolute_price_change_percent",
                                    "quote_volume", "spread_percent") if c in universe]
            preliminary = preliminary.merge(universe[metrics], on="symbol", how="left")
        if not preliminary.empty and not scores.empty:
            selected = scores[["symbol", "model_rank", "score"]]
            preliminary = preliminary.merge(selected, on="symbol", how="left")
        return CompletedOpportunityScan(directory, manifest, summary, universe, preliminary,
                                        csv("final_candidates"), csv("rich_data_readiness"), scores)


class OpportunityScannerApplicationService:
    """Injectable facade that invokes the existing Task 1--7 pipeline once."""

    def __init__(self, run_once: Callable[[OpportunityScanRequest, Callable[[], bool]], Path],
                 reader: OpportunityScanResultReader | None = None):
        self._run_once = run_once
        self.reader = reader or OpportunityScanResultReader()

    def run(self, request: OpportunityScanRequest, cancelled: Callable[[], bool]) -> CompletedOpportunityScan:
        return self.reader.read(self._run_once(request, cancelled))


class UnconfiguredOpportunityScannerService(OpportunityScannerApplicationService):
    """Clear boundary used when the host has not installed a pipeline adapter."""

    def __init__(self):
        def unavailable(_request, _cancelled):
            raise RuntimeError("Opportunity scanner pipeline service is not configured")
        super().__init__(unavailable)


class Task1To7OpportunityScanner:
    """Small orchestration facade over the existing Task 1--7 contracts."""

    SCORING_FEATURES = (
        "core_directional", "policy_market_context", "opportunity_activity",
    )

    def __init__(self, store: MarketDataStore, backend: BinanceDataHubBackend,
                 output_root: Path, *, live_client=None, registry=None,
                 now: Callable[[], datetime] | None = None):
        self.store = store
        self.backend = backend
        self.output_root = Path(output_root)
        self.live_client = live_client or BinanceUsdMDiscoveryClient()
        self.registry = registry or production_feature_registry()
        self.now = now or (lambda: datetime.now(timezone.utc))

    def __call__(self, request: OpportunityScanRequest,
                 cancelled: Callable[[], bool]) -> Path:
        if request.mode == "LIVE":
            discovery = scan_universe(
                self.live_client, request.live_discovery, now=self.now
            )
            discovery_config = request.live_discovery
            decision = self._live_decision(discovery)
        else:
            decision = request.decision_time
            discovery = discover_historical_universe(
                self.store, self._historical_symbols(),
                DiscoveryDecisionTime(decision), request.historical_discovery,
                market=request.market,
            )
            discovery_config = None

        interval = interval_to_timedelta(request.candle_acquisition.strategy_interval)
        required_bars = max(
            1, self.registry.effective_warmup(self.SCORING_FEATURES)
        )
        candle_start = decision - interval * (required_bars + 1)
        candles = SelectiveCandleAcquirer(
            self.store, self.backend, request.candle_acquisition
        ).acquire(discovery, candle_start, decision, cancelled=cancelled)

        scoring = self._score(request, candles, decision) if request.scoring else None
        final = build_final_candidate_set(
            discovery, candles, scoring, request.final_candidates
        )
        rich = SelectiveRichDataAcquirer(
            self.store, self.backend, request.rich_data, self.registry
        ).acquire(final, cancelled=cancelled)
        package = OpportunityScanPublicationInput(
            scan_timestamp=self.now(), discovery=discovery,
            discovery_config=discovery_config, candle_acquisition=candles,
            candle_acquisition_config=request.candle_acquisition,
            scoring_result=scoring, scoring_config=request.scoring,
            final_candidates=final, rich_data=rich,
            rich_data_config=request.rich_data,
        )
        return publish_opportunity_scan(self.output_root, package)

    @staticmethod
    def _live_decision(discovery) -> datetime:
        timestamps = {row.discovery_timestamp for row in discovery}
        if len(timestamps) != 1:
            raise ValueError("live discovery did not produce one decision timestamp")
        return next(iter(timestamps))

    def _historical_symbols(self) -> tuple[str, ...]:
        rows = self.store.catalog.inventory(
            self.store.raw_root, market=MarketKind.FUTURES_UM
        )
        return tuple(sorted({
            str(row["symbol"]).strip().upper() for row in rows
            if row.get("symbol")
        }))

    def _score(self, request, candles, decision):
        snapshots = []
        for acquired in candles.symbols:
            if acquired.state not in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED}:
                continue
            data_request = DataRequest(
                acquired.symbol, acquired.requested_start, acquired.requested_end,
                acquired.strategy_interval, datasets=(DatasetKind.KLINES,),
                market=request.market,
            )
            frame = self.store.load_dataset(
                data_request, DatasetKind.KLINES,
                interval=acquired.strategy_interval,
            )
            frames = self.registry.execute(
                self.SCORING_FEATURES, data_request,
                {DatasetKind.KLINES: frame},
                source_identities={
                    DatasetKind.KLINES: acquired.source_signature.cache_identity()
                },
            )
            snapshot = snapshot_from_registry_features(
                acquired.symbol, acquired.rank, decision,
                acquired.strategy_interval, frames, acquired.source_signature,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return score_opportunities(snapshots, decision, request.scoring)


def create_opportunity_scanner_service(raw_root: Path, cache_root: Path,
                                       output_root: Path, **pipeline_options):
    """Construct the runnable production Task 1--7 GUI service."""
    store = MarketDataStore(Path(raw_root), Path(cache_root))
    backend = BinanceDataHubBackend(Path(raw_root))
    pipeline = Task1To7OpportunityScanner(
        store, backend, Path(output_root), **pipeline_options
    )
    return OpportunityScannerApplicationService(pipeline)


def build_request(*, mode: str, decision_time: datetime | None,
                  minimum_listing_age_days: int, minimum_quote_volume: Decimal,
                  maximum_spread_percent: Decimal, preliminary_size: int,
                  final_size: int, strategy_interval: str,
                  model: OpportunityScoringModelDefinition | None,
                  enabled_features: tuple[str, ...]) -> OpportunityScanRequest:
    """Explicit GUI-to-native configuration mapping (no scanner semantics)."""
    live = DiscoveryConfig(timedelta(days=minimum_listing_age_days), minimum_quote_volume,
                           maximum_spread_percent)
    historical = HistoricalDiscoveryConfig(minimum_quote_volume=minimum_quote_volume)
    candles = SelectiveCandleAcquisitionConfig(preliminary_size, strategy_interval)
    scoring = None if model is None else OpportunityScoringConfig(strategy_interval, (model,))
    model_ref = None if model is None else OpportunityModelRef(model.name, model.version)
    final = FinalCandidateBoundaryConfig(strategy_interval, final_size, model_ref)
    rich = RichDataAcquisitionConfig(enabled_features=enabled_features)
    return OpportunityScanRequest(MarketKind.FUTURES_UM, mode, decision_time, live,
                                  historical, candles, scoring, final, rich)
