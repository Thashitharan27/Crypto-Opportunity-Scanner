from datetime import datetime, timedelta, timezone
from decimal import Decimal
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from crypto_strategy_lab.data.binance.selective_acquisition import (
    AcquisitionState,
    CandleAcquisitionResult,
    SelectiveCandleAcquisitionConfig,
    SymbolAcquisitionResult,
)
from crypto_strategy_lab.data.binance.universe import DiscoveryConfig, DiscoveryRow
from crypto_strategy_lab.data.schemas import DatasetKind
from crypto_strategy_lab.data.source_identity import SourceSignature
from crypto_strategy_lab.final_candidates import (
    FinalCandidateBoundaryConfig,
    OpportunityModelRef,
    build_final_candidate_set,
)
from crypto_strategy_lab.opportunity_reporting import (
    OpportunityScanPublicationInput,
    publish_opportunity_scan,
)
from crypto_strategy_lab.opportunity_scoring import (
    OpportunityScoreRow,
    OpportunityScoringConfig,
    OpportunityScoringResult,
    ScoreStatus,
)
from crypto_strategy_lab.rich_data_acquisition import (
    RichDataAcquisitionConfig,
    RichDataAcquisitionPlan,
    RichDataAcquisitionResult,
    SymbolRichDataResult,
)
from crypto_strategy_lab.run_manifest import (
    RunArtifactError,
    artifact_path,
    load_completed_manifest,
)


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


def test_scoring_result_boundary_must_match_discovery(tmp_path):
    package = _package()
    package = replace(
        package,
        scoring_result=OpportunityScoringResult(NOW - timedelta(hours=1), ()),
    )

    with pytest.raises(ValueError, match="Task 4 result decision time"):
        publish_opportunity_scan(tmp_path, package)


def test_task6_plan_cannot_reference_a_non_final_symbol(tmp_path):
    package = _package()
    foreign = SimpleNamespace(symbol="ETHUSDT")
    rich_data = replace(
        package.rich_data,
        plan=RichDataAcquisitionPlan((foreign,), ()),
    )

    with pytest.raises(ValueError, match="non-final candidate symbol"):
        publish_opportunity_scan(tmp_path, replace(package, rich_data=rich_data))


def test_task4_score_cannot_reference_symbol_absent_from_task3(tmp_path):
    package = _package()
    score = OpportunityScoreRow(
        "ETHUSDT", 1, NOW, "1h", NOW - timedelta(hours=1),
        NOW - timedelta(hours=1), {}, {}, {}, "balanced_activity", "1", 1.0,
        1, None, None, None, None, None, None, "UNKNOWN", None, None, None,
        "foreign-source", {}, ScoreStatus.SCORABLE, None,
    )
    package = replace(
        package,
        scoring_result=OpportunityScoringResult(NOW, (score,)),
    )

    with pytest.raises(ValueError, match="absent from Task 3"):
        publish_opportunity_scan(tmp_path, package)


def test_selected_model_cannot_be_omitted_from_task5_candidate(tmp_path):
    package = _package()
    acquisition = package.candle_acquisition.symbols[0]
    score = OpportunityScoreRow(
        "BTCUSDT", 1, NOW, "1h", NOW - timedelta(hours=1),
        NOW - timedelta(hours=1), {}, {}, {}, "balanced_activity", "1", 1.0,
        1, None, None, None, None, None, None, "UNKNOWN", None, None, None,
        acquisition.source_signature.cache_identity(), {}, ScoreStatus.SCORABLE, None,
    )
    scoring = OpportunityScoringResult(NOW, (score,))
    candidates = build_final_candidate_set(
        package.discovery,
        package.candle_acquisition,
        scoring,
        FinalCandidateBoundaryConfig(
            opportunity_model=OpportunityModelRef("balanced_activity", "1")
        ),
    )
    candidate = replace(
        candidates.candidates[0],
        opportunity_model_name=None,
        opportunity_model_version=None,
        opportunity_score=None,
        opportunity_model_rank=None,
    )
    package = replace(
        package,
        scoring_result=scoring,
        final_candidates=replace(candidates, candidates=(candidate,)),
    )

    with pytest.raises(ValueError, match="candidate model disagrees"):
        publish_opportunity_scan(tmp_path, package)


def test_task6_dataset_requirement_must_belong_to_its_candidate(tmp_path):
    package = _package()
    symbol_result = package.rich_data.symbols[0]
    foreign_requirement = SimpleNamespace(
        symbol="ETHUSDT",
        final_rank=symbol_result.final_rank,
        dataset=DatasetKind.KLINES,
        interval="1h",
    )
    foreign_dataset = SimpleNamespace(requirement=foreign_requirement)
    rich_data = replace(
        package.rich_data,
        symbols=(replace(symbol_result, datasets=(foreign_dataset,)),),
    )

    with pytest.raises(ValueError, match="candidate provenance mismatch"):
        publish_opportunity_scan(tmp_path, replace(package, rich_data=rich_data))


def test_task3_duplicate_symbols_are_rejected(tmp_path):
    package = _package()
    duplicate = replace(
        package.candle_acquisition,
        symbols=package.candle_acquisition.symbols * 2,
    )

    with pytest.raises(ValueError, match="Task 3 symbols must be unique"):
        publish_opportunity_scan(
            tmp_path, replace(package, candle_acquisition=duplicate)
        )


def test_manifest_uses_exact_semantic_hash_names_and_all_selected_sources(tmp_path):
    run_dir = publish_opportunity_scan(tmp_path, _package())
    manifest = load_completed_manifest(run_dir)

    assert {
        "discovery_config_hash",
        "candle_acquisition_config_hash",
        "scoring_config_hash",
        "final_candidate_config_hash",
        "rich_data_config_hash",
        "feature_config_hash",
    } <= manifest["hashes"].keys()
    assert {"dataset": "klines", "interval": "1h"} in (
        manifest["opportunity_scan"]["selected_datasets"]
    )


def test_completed_loader_rejects_incomplete_scan_catalog(tmp_path):
    run_dir = publish_opportunity_scan(tmp_path, _package())
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["final_candidates"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunArtifactError, match="catalog is incomplete"):
        load_completed_manifest(run_dir)


def test_discovery_ranks_must_be_unique_and_contiguous(tmp_path):
    package = _package()
    second = replace(package.discovery[0], symbol="ETHUSDT")
    invalid_discovery = (package.discovery[0], second)

    with pytest.raises(ValueError, match="unique, contiguous"):
        publish_opportunity_scan(
            tmp_path, replace(package, discovery=invalid_discovery)
        )
