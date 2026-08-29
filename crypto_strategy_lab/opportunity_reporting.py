"""Task 7: immutable publication of already-completed scanner stages.

Live runs preserve the observed REST-derived universe artifact, rather than claiming
that a later REST query can replay it. Historical runs additionally publish a frozen
decision boundary and Data Lake replay specification. No pipeline stage executes here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data.binance.historical_discovery import HistoricalDiscoveryResult
from .data.binance.selective_acquisition import AcquisitionState, CandleAcquisitionResult, SelectiveCandleAcquisitionConfig
from .data.binance.universe import DiscoveryConfig, DiscoveryRow
from .data.schemas import DatasetKind
from .final_candidates import FinalCandidateSet
from .opportunity_scoring import OpportunityScoringConfig, OpportunityScoringResult
from .rich_data_acquisition import RichDataAcquisitionConfig, RichDataAcquisitionResult
from .run_manifest import (OPPORTUNITY_SCAN_ARTIFACT_CONTRACT, OPPORTUNITY_SCAN_ARTIFACT_VERSION,
    RUN_MANIFEST_CONTRACT, RUN_MANIFEST_VERSION, RunType, artifact_path, atomic_json,
    canonical_json, canonical_sha256, capture_code_provenance, catalog_entry,
    new_run_identity, runtime_provenance)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _json(value: Any) -> Any:
    if isinstance(value, datetime): return _utc(value).isoformat()
    if isinstance(value, timedelta): return value.total_seconds()
    if isinstance(value, Decimal): return format(value, "f")
    if isinstance(value, Enum): return value.value
    if is_dataclass(value): return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping): return {str(k): _json(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list, set, frozenset)): return [_json(v) for v in sorted(value, key=str) if isinstance(value, (set, frozenset))] if isinstance(value, (set, frozenset)) else [_json(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class OpportunityScanPublicationInput:
    scan_timestamp: datetime
    discovery: Sequence[DiscoveryRow] | HistoricalDiscoveryResult
    discovery_config: DiscoveryConfig | None
    candle_acquisition: CandleAcquisitionResult
    candle_acquisition_config: SelectiveCandleAcquisitionConfig
    scoring_result: OpportunityScoringResult | None
    scoring_config: OpportunityScoringConfig | None
    final_candidates: FinalCandidateSet
    rich_data: RichDataAcquisitionResult
    rich_data_config: RichDataAcquisitionConfig
    feature_config: Mapping[str, Any] | None = None
    exchange: str = "binance"
    market: str = "futures_um"


def _signature(sig: Any) -> dict[str, Any] | None:
    if sig is None: return None
    return {"dataset": _json(sig.dataset), "digest": sig.digest,
            "partition_count": sig.partition_count, "identity_version": sig.identity_version,
            "cache_identity": sig.cache_identity()}


def _valid_signature(sig: Any, dataset: DatasetKind) -> bool:
    return (
        sig is not None
        and sig.dataset is dataset
        and bool(sig.digest.strip())
        and sig.partition_count > 0
        and bool(sig.cache_identity())
    )


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: canonical_json(_json(row[c])) if isinstance(row.get(c), (dict, list, tuple)) else _json(row.get(c)) for c in columns})


class OpportunityScanPublisher:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    def publish(self, package: OpportunityScanPublicationInput) -> Path:
        run_id, started = new_run_identity()
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        provenance = run_dir / "provenance"; provenance.mkdir()
        live = not isinstance(package.discovery, HistoricalDiscoveryResult)
        if live:
            if package.discovery_config is None: raise ValueError("live discovery_config is required")
            discovery_rows = list(package.discovery)
            stamps = {_utc(r.discovery_timestamp) for r in discovery_rows}
            if len(stamps) != 1: raise ValueError("live discovery timestamps must agree")
            decision = next(iter(stamps)); mode, contract = "LIVE", "binance_usdm_live_24h_v1"
            source_by_symbol = {}
        else:
            result = package.discovery
            if result.snapshot.decision_time.value != result.decision_time.value: raise ValueError("historical decision times disagree")
            decision = _utc(result.decision_time.value); mode, contract = "HISTORICAL", result.contract
            discovery_rows = list(result.snapshot.rows)
            source_by_symbol = {s.symbol.upper(): s for s in result.sources}
            if (len(source_by_symbol) != len(result.sources)
                    or set(source_by_symbol) != {
                        row.symbol.strip().upper() for row in discovery_rows
                    }):
                raise ValueError("historical discovery sources must match the universe")
        symbols = [str(r.symbol).strip().upper() for r in discovery_rows]
        if not symbols or len(symbols) != len(set(symbols)): raise ValueError("discovery symbols must be unique")
        scan_timestamp = _utc(package.scan_timestamp)
        universe = {"exchange": package.exchange.lower(), "market": package.market.lower(),
                    "discovery_contract": contract, "symbols": sorted(symbols),
                    "config": _json(package.discovery_config if live else package.discovery.config)}
        snapshot = []
        for r in discovery_rows:
            historical = not live
            rank = r.rank if historical else r.preliminary_rank
            snapshot.append({"symbol": r.symbol.upper(), "discovery_mode": mode, "decision_timestamp": decision,
                "eligible": r.eligible, "discovery_rank": rank, "rejection_reasons": list(r.rejection_reasons),
                "quote_volume": r.quote_volume, "range_percent": r.range_percent if historical else r.range_24h_percent,
                "price_change_percent": r.price_change_percent if historical else r.price_change_24h_percent,
                "absolute_price_change_percent": (abs(r.price_change_percent) if historical and r.price_change_percent is not None else abs(r.price_change_24h_percent) if not historical and r.price_change_24h_percent is not None else None),
                "spread_percent": None if historical else r.spread_percent,
                "listing_age_seconds": None if historical or r.listing_age is None else r.listing_age.total_seconds(),
                "reference_period_start": r.period_start if historical else None, "reference_period_end": r.period_end if historical else None,
                "reference_available_at": r.available_at if historical else None, "discovery_source_identity": r.source_identity if historical else None})
        snapshot.sort(key=lambda x: x["symbol"])
        rank_by = {r["symbol"]: r["discovery_rank"] for r in snapshot}
        acquisition_by = {a.symbol.upper(): a for a in package.candle_acquisition.symbols}
        if len(acquisition_by) != len(package.candle_acquisition.symbols):
            raise ValueError("Task 3 symbols must be unique")
        if not set(acquisition_by) <= set(symbols): raise ValueError("Task 3 symbol is absent from discovery")
        prelim = []
        sources = []
        for symbol, a in sorted(acquisition_by.items(), key=lambda x: (x[1].rank, x[0])):
            if a.rank != rank_by[symbol] or a.strategy_interval != package.candle_acquisition_config.strategy_interval: raise ValueError("Task 3 rank/interval mismatch")
            sig = _signature(a.source_signature)
            if (a.state in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED}
                    and not _valid_signature(a.source_signature, DatasetKind.KLINES)):
                raise ValueError("successful Task 3 source must have a valid KLINES signature")
            if sig: sources.append({"symbol": symbol, "role": "strategy_candles", **sig,
                "interval": a.strategy_interval, "request_start": _json(a.requested_start), "request_end": _json(a.requested_end)})
            prelim.append({"symbol": symbol, "discovery_rank": a.rank, "strategy_interval": a.strategy_interval,
                "requested_start": a.requested_start, "requested_end": a.requested_end, "acquisition_state": a.state,
                "acquisition_ranges": _json(a.acquisition_ranges), "quality_status": a.quality_status, "row_count": a.row_count,
                "strategy_source_identity": sig and sig["cache_identity"], "strategy_source_signature": sig, "detail": a.detail})
        score_rows = []
        scores_by_key = {}
        if package.scoring_result is None:
            if package.scoring_config is not None: raise ValueError("scoring config supplied without scoring result")
            if package.final_candidates.config.opportunity_model is not None: raise ValueError("selected model requires explicit scoring")
        else:
            if package.scoring_config is None: raise ValueError("scoring result requires explicit scoring config")
            if _utc(package.scoring_result.decision_time) != decision:
                raise ValueError("Task 4 result decision time must match discovery")
            allowed = {(m.name, m.version) for m in package.scoring_config.models}
            for score in package.scoring_result.rows:
                if _utc(score.decision_time) != decision or score.strategy_interval != package.scoring_config.strategy_interval or (score.model_name, score.model_version) not in allowed: raise ValueError("Task 4 decision/interval/model mismatch")
                expected = acquisition_by.get(score.symbol.upper())
                if expected is None:
                    raise ValueError("Task 4 score symbol is absent from Task 3")
                if (expected.source_signature is None
                        or score.source_identity != expected.source_signature.cache_identity()):
                    raise ValueError("Task 4 source identity mismatch")
                score_key = (
                    score.symbol.upper(), score.model_name, score.model_version
                )
                if score_key in scores_by_key:
                    raise ValueError("duplicate Task 4 score row")
                scores_by_key[score_key] = score
                score_rows.append(_json(score))
                sources.append({"symbol": score.symbol.upper(), "role": "opportunity_scoring", **_signature(expected.source_signature),
                                "interval": score.strategy_interval, "request_start": None, "request_end": _json(score.decision_time)})
            selected = package.final_candidates.config.opportunity_model
            if selected and (selected.name, selected.version) not in allowed: raise ValueError("Task 5 selected model does not match scoring config")
        finals = sorted(package.final_candidates.candidates, key=lambda c: (c.final_rank, c.symbol))
        if [c.final_rank for c in finals] != list(range(1, len(finals)+1)): raise ValueError("final ranks must be unique and contiguous")
        if not {c.symbol.upper() for c in finals} <= set(acquisition_by): raise ValueError("final candidates must be Task 3 symbols")
        final_rows = []
        for c in finals:
            a = acquisition_by[c.symbol.upper()]
            if c.discovery_timestamp != decision or c.discovery_rank != rank_by[c.symbol.upper()] or c.strategy_source_identity != (a.source_signature.cache_identity() if a.source_signature else None) or c.discovery_mode.value != mode or c.discovery_contract != contract: raise ValueError("Task 5 provenance mismatch")
            selected_model = package.final_candidates.config.opportunity_model
            candidate_model = (
                c.opportunity_model_name,
                c.opportunity_model_version,
            )
            if selected_model is None:
                if (candidate_model != (None, None)
                        or c.opportunity_score is not None
                        or c.opportunity_model_rank is not None):
                    raise ValueError("Task 5 candidate has an unconfigured score")
            else:
                if candidate_model != (selected_model.name, selected_model.version):
                    raise ValueError("Task 5 candidate model disagrees with its config")
                score = scores_by_key.get((
                    c.symbol.upper(), c.opportunity_model_name,
                    c.opportunity_model_version,
                ))
                if score is None:
                    raise ValueError("Task 5 selected score is absent from Task 4")
                if (score.score != c.opportunity_score
                        or score.model_rank != c.opportunity_model_rank
                        or score.discovery_rank != c.discovery_rank
                        or score.strategy_interval != c.strategy_interval
                        or _utc(score.decision_time) != decision
                        or score.source_identity != c.strategy_source_identity):
                    raise ValueError("Task 5 selected score disagrees with Task 4")
            flat = c.serializable(); metrics = flat.pop("impact_metrics"); request = flat.pop("strategy_data_request")
            final_rows.append({**flat, **{f"impact_{k}": v for k,v in metrics.items()}, **{f"strategy_request_{k}": v for k,v in request.items()}})
        final_symbols = {c.symbol.upper() for c in finals}
        rich_symbols = {s.symbol.upper(): s for s in package.rich_data.symbols}
        if len(rich_symbols) != len(package.rich_data.symbols):
            raise ValueError("Task 6 symbols must be unique")
        if set(rich_symbols) != final_symbols:
            raise ValueError("Task 6 symbol set must equal final candidates")
        plan_symbols = {
            requirement.symbol.strip().upper()
            for requirement in package.rich_data.plan.requirements
        }
        for feature_plan in package.rich_data.plan.features:
            plan_symbols.update(
                requirement.symbol.strip().upper()
                for requirement in feature_plan.requirements
            )
        if not plan_symbols <= final_symbols:
            raise ValueError("Task 6 plan contains a non-final candidate symbol")
        rich_rows = []
        for c in finals:
            s = rich_symbols[c.symbol.upper()]
            if s.final_rank != c.final_rank or s.strategy_source_identity != c.strategy_source_identity: raise ValueError("Task 6 candidate provenance mismatch")
            readiness = {f.feature_name: f.readiness for f in s.features}
            for d in sorted(s.datasets, key=lambda x: (x.requirement.dataset.value, x.requirement.interval or "")):
                if (d.requirement.symbol.strip().upper() != s.symbol.upper()
                        or d.requirement.final_rank != s.final_rank):
                    raise ValueError(
                        "Task 6 dataset requirement candidate provenance mismatch"
                    )
                sig = _signature(d.source_signature)
                if d.state in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED}:
                    if (not _valid_signature(d.source_signature, d.requirement.dataset)
                            or d.source_identity != sig["cache_identity"]):
                        raise ValueError("Task 6 source identity mismatch")
                elif d.source_identity is not None:
                    raise ValueError("unavailable Task 6 result cannot claim a source identity")
                if d.state in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED}:
                    sources.append({"symbol": s.symbol.upper(), "role": "rich_data", **sig,
                    "interval": d.requirement.interval, "request_start": _json(d.requirement.start), "request_end": _json(d.requirement.end)})
                for feature, requiredness in sorted(d.requirement.feature_requiredness):
                    rich_rows.append({"symbol": s.symbol.upper(), "final_rank": s.final_rank, "strategy_source_identity": s.strategy_source_identity,
                        "feature_name": feature, "feature_readiness": readiness.get(feature), "dataset": d.requirement.dataset,
                        "interval": d.requirement.interval, "requested_start": d.requirement.start, "requested_end": d.requirement.end,
                        "requiredness_for_feature": requiredness, "acquisition_state": d.state, "quality_status": d.quality_status,
                        "missing_ranges": _json(d.missing_ranges), "rich_source_identity": d.source_identity,
                        "rich_source_signature": sig, "detail": d.detail})
        if not live:
            historical_rows = {row.symbol.upper(): row for row in discovery_rows}
            for symbol, src in sorted(source_by_symbol.items()):
                sig = _signature(src.signature)
                row_identity = historical_rows[symbol].source_identity
                if src.signature is not None:
                    if (not _valid_signature(src.signature, DatasetKind.KLINES)
                            or row_identity != sig["cache_identity"]):
                        raise ValueError("historical discovery source identity mismatch")
                    sources.append({"symbol": symbol, "role": "historical_discovery", **sig,
                    "interval": src.request.strategy_interval, "request_start": _json(src.request.start), "request_end": _json(src.request.end)})
                elif row_identity is not None:
                    raise ValueError("historical discovery identity has no signature")
        sources.sort(key=canonical_json); source_doc = {"contract": "opportunity_source_identities_v1", "entries": sources}
        source_digest = canonical_sha256(source_doc)
        configs = {"discovery": universe["config"], "candle_acquisition": _json(package.candle_acquisition_config),
                   "scoring": _json(package.scoring_config), "final_candidates": _json(package.final_candidates.config),
                   "rich_data": _json(package.rich_data_config), "features": _json(package.feature_config)}
        hashes = {
            "universe_definition_hash": canonical_sha256(universe),
            "discovery_config_hash": canonical_sha256(configs["discovery"]),
            "candle_acquisition_config_hash": canonical_sha256(configs["candle_acquisition"]),
            "scoring_config_hash": canonical_sha256(configs["scoring"]),
            "final_candidate_config_hash": canonical_sha256(configs["final_candidates"]),
            "rich_data_config_hash": canonical_sha256(configs["rich_data"]),
            "feature_config_hash": canonical_sha256(configs["features"]),
            "source_identity_digest": source_digest,
        }
        hashes["semantic_input_hash"] = canonical_sha256({"decision_timestamp": decision.isoformat(), "universe": universe, "configs": configs, "sources": source_doc})
        columns = {
          "universe_snapshot": list(snapshot[0].keys()), "discovery_rejections": list(snapshot[0].keys()),
          "preliminary_candidates": list(prelim[0].keys()) if prelim else ["symbol","discovery_rank","strategy_interval","requested_start","requested_end","acquisition_state","acquisition_ranges","quality_status","row_count","strategy_source_identity","strategy_source_signature","detail"],
          "final_candidates": list(final_rows[0].keys()) if final_rows else ["symbol","final_rank"],
          "final_candidate_exclusions": ["symbol","reason","detail"],
          "rich_data_readiness": list(rich_rows[0].keys()) if rich_rows else ["symbol","final_rank","feature_name","feature_readiness","dataset","interval"]}
        csv_rows = {"universe_snapshot": snapshot, "discovery_rejections": [r for r in snapshot if not r["eligible"]],
                    "preliminary_candidates": prelim, "final_candidates": final_rows,
                    "final_candidate_exclusions": [_json(e) for e in package.final_candidates.exclusions], "rich_data_readiness": rich_rows}
        artifacts = {}
        for name, rows in csv_rows.items():
            path = run_dir / f"{name}.csv"; _write_csv(path, columns[name], rows); artifacts[name] = catalog_entry(path, run_dir, "csv", len(rows))
        if package.scoring_result is not None:
            score_columns = list(score_rows[0]) if score_rows else [
                    "symbol", "discovery_rank", "decision_time", "strategy_interval",
                    "feature_timestamp", "available_at", "raw_components",
                    "normalized_components", "component_weights", "model_name",
                    "model_version", "score", "model_rank", "atr", "atr_pct", "adx",
                    "di_spread", "previous_di_spread", "di_pressure_delta",
                    "di_pressure_state", "signed_momentum_24h", "absolute_momentum_24h",
                    "market_regime", "source_identity", "feature_versions", "status",
                    "unscorable_reason",
                ]
            path = run_dir / "opportunity_scores.csv"
            _write_csv(path, score_columns, score_rows)
            artifacts["opportunity_scores"] = catalog_entry(
                path, run_dir, "csv", len(score_rows)
            )
        source_path = provenance / "source_identities.json"; atomic_json(source_path, source_doc); artifacts["source_identities"] = catalog_entry(source_path, run_dir, "json", len(sources))
        counts = {"universe_count": len(snapshot), "discovery_eligible_count": sum(bool(r["eligible"]) for r in snapshot), "discovery_rejected_count": sum(not r["eligible"] for r in snapshot), "preliminary_candidate_count": len(prelim), "final_candidate_count": len(finals)}
        summary = {"run_type": RunType.OPPORTUNITY_SCAN.value, "discovery_mode": mode, "scan_timestamp": scan_timestamp.isoformat(), "decision_timestamp": decision.isoformat(), **counts,
                   "task3_ready_count": sum(r["acquisition_state"] in {AcquisitionState.REUSED, AcquisitionState.ACQUIRED} for r in prelim), "scoring_enabled": package.scoring_result is not None,
                   "scoring_models": [{"name": name, "version": version} for name, version in sorted({(r["model_name"], r["model_version"]) for r in score_rows})], "rich_data_enabled_features": list(package.rich_data_config.enabled_features),
                   "rich_data_readiness_counts": {x: sum(f.readiness.value == x for s in package.rich_data.symbols for f in s.features) for x in ("READY","DEGRADED","UNAVAILABLE")},
                   "selected_datasets": [{"dataset": dataset, "interval": interval} for dataset, interval in sorted({(entry["dataset"], entry.get("interval")) for entry in sources if entry.get("dataset")}, key=str)], "source_identity_digest": source_digest, "semantic_input_hash": hashes["semantic_input_hash"]}
        summary_path = run_dir / "opportunity_summary.json"; atomic_json(summary_path, summary); artifacts["opportunity_summary"] = catalog_entry(summary_path, run_dir, "json")
        replay = None if live else {"discovery_mode": mode, "decision_timestamp": decision.isoformat(), "universe_symbols": sorted(symbols), "exchange": package.exchange, "market": package.market,
             "historical_discovery_contract": contract, "historical_discovery_config": configs["discovery"], "candle_acquisition_config": configs["candle_acquisition"], "scoring_config": configs["scoring"],
             "final_candidate_config": configs["final_candidates"], "rich_data_config": configs["rich_data"], "feature_config": configs["features"], "source_identity_digest": source_digest}
        payload = {"artifact_contract": OPPORTUNITY_SCAN_ARTIFACT_CONTRACT, "artifact_version": OPPORTUNITY_SCAN_ARTIFACT_VERSION,
            "scan_timestamp": scan_timestamp.isoformat(), "decision_timestamp": decision.isoformat(), "discovery_mode": mode, "discovery_contract": contract,
            "universe_definition": universe, "counts": counts, "selected_datasets": summary["selected_datasets"], "source_identity_digest": source_digest,
            "source_identities_artifact": "source_identities", "historical_replay_spec": replay,
            "reproducibility_note": "Observed snapshot only; live REST responses are not replayable." if live else "Frozen historical boundary and Data Lake identities support replay."}
        manifest = {"run_manifest_contract": RUN_MANIFEST_CONTRACT, "run_manifest_version": RUN_MANIFEST_VERSION, "run_type": RunType.OPPORTUNITY_SCAN.value,
            "run_id": run_id, "run_started_at": started, "run_completed_at": datetime.now(timezone.utc).isoformat(), "run_status": "COMPLETED",
            **capture_code_provenance(), "runtime": runtime_provenance(), "config": configs, "hashes": hashes, "artifacts": artifacts, "opportunity_scan": payload}
        for name in artifacts: artifact_path(run_dir, manifest, name)
        atomic_json(run_dir / "run_manifest.json", manifest)  # final completion-marker write
        return run_dir


def publish_opportunity_scan(output_root: Path, package: OpportunityScanPublicationInput) -> Path:
    return OpportunityScanPublisher(output_root).publish(package)
