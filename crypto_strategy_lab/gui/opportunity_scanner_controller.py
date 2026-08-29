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
from threading import Event
from typing import Callable

import pandas as pd

from crypto_strategy_lab.data.binance.historical_discovery import HistoricalDiscoveryConfig
from crypto_strategy_lab.data.binance.selective_acquisition import SelectiveCandleAcquisitionConfig
from crypto_strategy_lab.data.binance.universe import DiscoveryConfig
from crypto_strategy_lab.data.schemas import MarketKind
from crypto_strategy_lab.final_candidates import FinalCandidateBoundaryConfig, OpportunityModelRef
from crypto_strategy_lab.opportunity_scoring import OpportunityScoringConfig, OpportunityScoringModelDefinition
from crypto_strategy_lab.rich_data_acquisition import RichDataAcquisitionConfig
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
