from datetime import datetime, timedelta, timezone
from decimal import Decimal

from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionState,
    CandleAcquisitionResult,
    SelectiveCandleAcquisitionConfig,
    SymbolAcquisitionResult,
)
from crypto_strategy_lab.data.binance.universe import DiscoveryConfig, DiscoveryRow
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.final_candidates import build_final_candidate_set
from crypto_strategy_lab.opportunity_reporting import (
    OpportunityScanPublicationInput,
    publish_opportunity_scan,
)
from crypto_strategy_lab.opportunity_scoring import (
    OpportunityScoringConfig,
    OpportunityScoringResult,
)
from crypto_strategy_lab.rich_data_acquisition import (
    RichDataAcquisitionConfig,
    RichDataAcquisitionPlan,
    RichDataAcquisitionResult,
    SymbolRichDataResult,
)
from crypto_strategy_lab.run_manifest import artifact_path, load_completed_manifest


NOW = datetime(2025, 1, 2, tzinfo=timezone.utc)


def _package() -> OpportunityScanPublicationInput:
    discovery = DiscoveryRow(
        "BTCUSDT", True, (), timedelta(days=100), Decimal("20000000"),
        Decimal("10"), Decimal("10.01"), Decimal("0.1"), Decimal("8"),
        Decimal("-3"), 1, NOW,
    )
    signature = SourceSignature(DatasetKind.KLINES, "strategy-source", 2)
    acquisition = CandleAcquisitionResult((SymbolAcquisitionResult(
        "BTCUSDT", 1, AcquisitionState.REUSED, "1h", NOW - timedelta(days=5),
        NOW, row_count=120, source_signature=signature,
    ),))
    candidates = build_final_candidate_set((discovery,), acquisition)
    rich_data = RichDataAcquisitionResult(
        RichDataAcquisitionPlan((), ()),
        (SymbolRichDataResult("BTCUSDT", 1, signature.cache_identity(), (), ()),),
    )
    return OpportunityScanPublicationInput(
        NOW, (discovery,), DiscoveryConfig(), acquisition,
        SelectiveCandleAcquisitionConfig(), OpportunityScoringResult(NOW, ()),
        OpportunityScoringConfig(), candidates, rich_data,
        RichDataAcquisitionConfig(),
    )


def test_executed_scoring_publishes_artifact_even_when_it_has_no_rows(tmp_path):
    run_dir = publish_opportunity_scan(tmp_path, _package())
    manifest = load_completed_manifest(run_dir)

    assert manifest["artifacts"]["opportunity_scores"]["rows"] == 0
    path = artifact_path(run_dir, manifest, "opportunity_scores")
    assert path.read_text(encoding="utf-8").startswith("symbol,discovery_rank,")
