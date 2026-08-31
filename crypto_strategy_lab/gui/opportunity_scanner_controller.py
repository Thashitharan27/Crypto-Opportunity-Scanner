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
    SelectiveCandleAcquisitionConfig, resolve_binance_data_hub_project_path,
)
from crypto_strategy_lab.data.binance.universe import (
    BinanceUsdMDiscoveryClient, DiscoveryConfig, scan_universe,
)
from crypto_strategy_lab.data.query import DataRequest
from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.data.store import MarketDataStore
from crypto_strategy_lab.data.timing import interval_to_timedelta
from crypto_strategy_lab.data.timing import normalize_binance_interval
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


@dataclass(frozen=True, slots=True)
class OpportunityScanProgress:
    """Operational progress only; never part of a research/publication model."""

    stage: str
    stage_index: int
    stage_count: int = 6
    message: str = ""
    decision_timestamp: datetime | None = None
    completed_scans: int = 0
    total_scans: int = 1
    current_scan_index: int = 1
    elapsed_seconds: float = 0.0
    average_scan_seconds: float | None = None
    eta_seconds: float | None = None


MAX_HISTORICAL_REPLAY_SCANS = 1000
HISTORICAL_REPLAY_CADENCES = {
    "1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1),
}


def historical_decision_points(start: datetime, end: datetime, cadence: timedelta,
                               *, maximum: int = MAX_HISTORICAL_REPLAY_SCANS) -> tuple[datetime, ...]:
    """Return exact, inclusive UTC replay instants without grid alignment."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("historical range timestamps must be timezone-aware UTC")
    if start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0):
        raise ValueError("historical range timestamps must be UTC")
    if cadence <= timedelta(0):
        raise ValueError("historical replay cadence must be greater than zero")
    if end < start:
        raise ValueError("historical range end must be at or after start")
    count = ((end - start) // cadence) + 1
    if count < 1:
        raise ValueError("historical range must contain at least one decision point")
    if count > maximum:
        raise ValueError(f"historical range has {count} scans; maximum is {maximum}")
    return tuple(start + cadence * offset for offset in range(count))


@dataclass(frozen=True, slots=True)
class HistoricalReplayResult:
    completed: tuple[CompletedOpportunityScan, ...]
    decision_points: tuple[datetime, ...]
    elapsed_seconds: float

    @property
    def last(self) -> CompletedOpportunityScan | None:
        return self.completed[-1] if self.completed else None


class HistoricalReplayFailure(RuntimeError):
    def __init__(self, decision_time: datetime, completed: tuple[CompletedOpportunityScan, ...], cause: Exception):
        super().__init__(str(cause)); self.decision_time = decision_time; self.completed = completed


class HistoricalRangeRunner:
    """Sequentially reuse the native single-scan service boundary."""

    def __init__(self, service, *, monotonic: Callable[[], float]):
        self.service, self.monotonic = service, monotonic

    def run(self, decision_points, request_factory, cancelled, progress=lambda _event: None):
        points = tuple(decision_points)
        if not points:
            raise ValueError("historical range must contain at least one decision point")
        started = self.monotonic(); completed = []; completed_seconds = 0.0
        for index, decision in enumerate(points, 1):
            if cancelled():
                raise OpportunityScanCancelled(f"historical replay cancelled; {len(completed)} completed")
            scan_started = self.monotonic()
            def forward(event, i=index, d=decision):
                elapsed = self.monotonic() - started
                average = completed_seconds / len(completed) if completed else None
                progress(OpportunityScanProgress(event.stage, event.stage_index,
                    event.stage_count, event.message, d, len(completed), len(points), i,
                    elapsed, average, None if average is None else average * (len(points)-len(completed))))
            try:
                request = request_factory(decision)
                run_with_progress = getattr(self.service, "run_with_progress", None)
                result = (
                    run_with_progress(request, cancelled, forward)
                    if run_with_progress is not None
                    else self.service.run(request, cancelled)
                )
            except OpportunityScanCancelled:
                raise
            except Exception as exc:
                raise HistoricalReplayFailure(decision, tuple(completed), exc) from exc
            completed.append(result)
            completed_seconds += self.monotonic() - scan_started
            elapsed = self.monotonic() - started
            average = completed_seconds / len(completed)
            progress(OpportunityScanProgress("scan_complete", 6, 6, "Scan complete", decision,
                len(completed), len(points), index, elapsed, average,
                average * (len(points) - len(completed))))
        return HistoricalReplayResult(tuple(completed), points, self.monotonic() - started)


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
            selected_model = (
                manifest.get("config", {})
                .get("final_candidates", {})
                .get("opportunity_model")
            )
            if isinstance(selected_model, dict):
                selected = scores
                selected = selected.loc[
                    (selected["model_name"] == selected_model.get("name"))
                    & (selected["model_version"].astype(str)
                       == str(selected_model.get("version")))
                ]
                # A valid Task-7 package has at most one selected-model score
                # per symbol. Enforce the Task-5 boundary rather than
                # multiplying rows with Task-4 comparison models.
                if selected["symbol"].duplicated().any():
                    raise ValueError(
                        "selected opportunity model has duplicate symbol scores"
                    )
                preliminary = preliminary.merge(
                    selected[["symbol", "model_rank", "score"]],
                    on="symbol", how="left",
                )
        return CompletedOpportunityScan(directory, manifest, summary, universe, preliminary,
                                        csv("final_candidates"), csv("rich_data_readiness"), scores)


class OpportunityScannerApplicationService:
    """Injectable facade that invokes the existing Task 1--7 pipeline once."""

    def __init__(self, run_once: Callable[[OpportunityScanRequest, Callable[[], bool]], Path],
                 reader: OpportunityScanResultReader | None = None, validation_service=None):
        self._run_once = run_once
        self.reader = reader or OpportunityScanResultReader()
        self.validation_service = validation_service

    def run(self, request: OpportunityScanRequest, cancelled: Callable[[], bool]) -> CompletedOpportunityScan:
        return self.reader.read(self._run_once(request, cancelled))

    def run_with_progress(self, request, cancelled, progress) -> CompletedOpportunityScan:
        """Optional GUI boundary; the original two-argument API stays intact."""
        method = getattr(self._run_once, "run_with_progress", None)
        path = method(request, cancelled, progress) if method else self._run_once(request, cancelled)
        return self.reader.read(path)


class OpportunityScanCancelled(RuntimeError):
    """Cooperative cancellation before an authoritative Task-7 publication."""


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
        return self.run_with_progress(request, cancelled, lambda _event: None)

    def run_with_progress(self, request: OpportunityScanRequest,
                          cancelled: Callable[[], bool], progress) -> Path:
        def report(stage, index, message):
            progress(OpportunityScanProgress(stage, index, 6, message,
                                             request.decision_time))
        if request.mode == "LIVE":
            report("discovery", 1, "Live discovery")
            discovery = scan_universe(
                self.live_client, request.live_discovery, now=self.now
            )
            discovery_config = request.live_discovery
            decision = self._live_decision(discovery)
        else:
            report("historical_discovery", 1, "Historical discovery")
            decision = request.decision_time
            discovery = discover_historical_universe(
                self.store, self._historical_symbols(),
                DiscoveryDecisionTime(decision), request.historical_discovery,
                market=request.market,
            )
            discovery_config = None

        self._raise_if_cancelled(cancelled)

        required_bars = max(
            1, self.registry.effective_warmup(self.SCORING_FEATURES)
        )
        candle_start, candle_end = self._candle_bounds(
            decision, request.candle_acquisition.strategy_interval,
            required_bars + 1,
        )
        report("candle_acquisition", 2, "Selective candle acquisition")
        candles = SelectiveCandleAcquirer(
            self.store, self.backend, request.candle_acquisition
        ).acquire(discovery, candle_start, candle_end, cancelled=cancelled)

        self._raise_if_cancelled(cancelled)
        report("scoring", 3, "Opportunity scoring" if request.scoring else "Opportunity scoring skipped")
        scoring = self._score(request, candles, decision) if request.scoring else None
        report("final_candidates", 4, "Final candidate boundary")
        final = build_final_candidate_set(
            discovery, candles, scoring, request.final_candidates
        )
        report("rich_data", 5, "Selective rich-data acquisition")
        rich = SelectiveRichDataAcquirer(
            self.store, self.backend, request.rich_data, self.registry
        ).acquire(final, cancelled=cancelled)
        # Task 7 has no cancellation contract and writes an immutable completed
        # marker.  Never enter publication after a cooperative cancellation.
        self._raise_if_cancelled(cancelled)
        report("publication", 6, "Immutable OPPORTUNITY_SCAN publication")
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
    def _raise_if_cancelled(cancelled: Callable[[], bool]) -> None:
        if cancelled():
            raise OpportunityScanCancelled("opportunity scan cancelled")

    @staticmethod
    def _candle_bounds(decision: datetime, strategy_interval: str,
                       candle_count: int) -> tuple[datetime, datetime]:
        """Return a fixed-grid Task-3 window without changing the decision."""
        interval = interval_to_timedelta(strategy_interval)
        utc = decision.astimezone(timezone.utc)
        step = int(interval.total_seconds())
        epoch_seconds = int(utc.timestamp())
        end = datetime.fromtimestamp(
            epoch_seconds - (epoch_seconds % step), tz=timezone.utc
        )
        return end - interval * candle_count, end

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
                                       output_root: Path, *, project_root: Path | None = None,
                                       **pipeline_options):
    """Construct the runnable production Task 1--7 GUI service."""
    store = MarketDataStore(Path(raw_root), Path(cache_root))
    scanner_root = (Path(__file__).resolve().parents[2]
                    if project_root is None else Path(project_root))
    data_hub_path = resolve_binance_data_hub_project_path(scanner_root)
    backend = BinanceDataHubBackend(Path(raw_root), project_path=data_hub_path)
    pipeline = Task1To7OpportunityScanner(
        store, backend, Path(output_root), **pipeline_options
    )
    from crypto_strategy_lab.historical_strategy_validation import create_historical_strategy_validation_service
    validation=create_historical_strategy_validation_service(raw_root,cache_root,Path(output_root)/"opportunity_validation",backend)
    service = OpportunityScannerApplicationService(pipeline, validation_service=validation)
    service.binance_data_hub_project_path = data_hub_path
    return service


def build_request(*, mode: str, decision_time: datetime | None,
                  minimum_listing_age_days: int, minimum_quote_volume: Decimal,
                  maximum_spread_percent: Decimal, preliminary_size: int,
                  final_size: int, strategy_interval: str,
                  model: OpportunityScoringModelDefinition | None,
                  enabled_features: tuple[str, ...]) -> OpportunityScanRequest:
    """Explicit GUI-to-native configuration mapping (no scanner semantics)."""
    normalized_interval = normalize_binance_interval(strategy_interval)
    if (model is not None and model.supported_intervals is not None
            and normalized_interval not in model.supported_intervals):
        supported = ", ".join(model.supported_intervals)
        raise ValueError(
            f"{model.name} v{model.version} supports only {supported}"
        )
    live = DiscoveryConfig(timedelta(days=minimum_listing_age_days), minimum_quote_volume,
                           maximum_spread_percent)
    historical = HistoricalDiscoveryConfig(minimum_quote_volume=minimum_quote_volume)
    candles = SelectiveCandleAcquisitionConfig(preliminary_size, normalized_interval)
    scoring = None if model is None else OpportunityScoringConfig(normalized_interval, (model,))
    model_ref = None if model is None else OpportunityModelRef(model.name, model.version)
    final = FinalCandidateBoundaryConfig(normalized_interval, final_size, model_ref)
    rich = RichDataAcquisitionConfig(enabled_features=enabled_features)
    return OpportunityScanRequest(MarketKind.FUTURES_UM, mode, decision_time, live,
                                  historical, candles, scoring, final, rich)
