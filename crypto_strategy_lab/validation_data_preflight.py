"""Requirement-driven Data Lake preparation for historical strategy validation."""
from __future__ import annotations
from dataclasses import dataclass, replace
from types import SimpleNamespace
import json

from .data import DataRequest, DatasetKind, MarketKind
from .data.binance.selective_acquisition import AcquisitionState, ArchiveAcquisitionRequest
from .data.quality import DataQualityStatus
from .data.timing import floor_fixed_candle_grid
from .gui.v2_controller import GuiResearchRequest
from .rich_data_acquisition import (RequirementRequiredness, RichDataAcquisitionConfig,
    resolve_rich_data_requirements)
from .rule_native_engine import (required_research_features_for_strategy,
    required_research_indicators_for_strategy)
from .structural_requirements import structural_benchmark_requirement


class ValidationDataUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationDataRequirement:
    role: str
    request: DataRequest
    dataset: DatasetKind
    interval: str | None


_FIXED_CANDLE_DATASETS = {DatasetKind.KLINES, DatasetKind.MARK_PRICE_KLINES,
    DatasetKind.INDEX_PRICE_KLINES, DatasetKind.PREMIUM_INDEX_KLINES}


def _align_candle_requirement(item: ValidationDataRequirement) -> ValidationDataRequirement:
    """Align each candle request to its own grid; leave event data exact."""
    if item.dataset not in _FIXED_CANDLE_DATASETS or item.interval is None:
        return item
    request=replace(item.request,start=floor_fixed_candle_grid(item.request.start,item.interval))
    return replace(item,request=request)


def validation_data_requirements(symbol,start,end,config) -> tuple[ValidationDataRequirement,...]:
    strategy=f"{config.data.strategy_timeframe_minutes}m"
    intrabar=f"{config.data.intrabar_timeframe_minutes}m" if config.data.use_intrabar_data else None
    base=DataRequest(symbol,start,end,strategy,intrabar,market=MarketKind.FUTURES_UM)
    requirements=[ValidationDataRequirement("STRATEGY",base,DatasetKind.KLINES,base.strategy_interval)]
    # Full native interval is intentional: warm-up trades can affect Standard
    # single-symbol suppression at the first candidate.
    if base.intrabar_interval:
        request=DataRequest(symbol,start,end,base.intrabar_interval,market=base.market)
        requirements.append(ValidationDataRequirement("INTRABAR",request,DatasetKind.KLINES,base.intrabar_interval))
    gui_request=GuiResearchRequest("binance",MarketKind.FUTURES_UM,symbol,start,end,
        base.strategy_interval,base.intrabar_interval)
    structural=structural_benchmark_requirement(gui_request,config.features)
    if structural:
        requirements.append(ValidationDataRequirement("STRUCTURAL_BENCHMARK",
            structural.data_request,DatasetKind.KLINES,structural.interval))
    features=required_research_features_for_strategy(config.strategy)
    indicators=required_research_indicators_for_strategy(config.strategy)
    if features:
        candidate=SimpleNamespace(symbol=symbol,final_rank=1,strategy_data_request=base,
            strategy_source_identity="VALIDATION_PREFLIGHT")
        plan=resolve_rich_data_requirements(SimpleNamespace(candidates=(candidate,)),
            RichDataAcquisitionConfig(enabled_features=tuple(features),
                feature_parameters=config.features.registry_parameters(
                    strategy_timeframe_minutes=config.data.strategy_timeframe_minutes)))
        for item in plan.requirements:
            strategy_required=item.requiredness is RequirementRequiredness.REQUIRED_FOR_FEATURE
            if item.dataset is DatasetKind.KLINES and item.interval=="1h" and indicators & {
                "OI_VS_PRICE_STATE_1H","PRICE_CHANGE_PCT_1H"}: strategy_required=True
            if item.dataset is DatasetKind.PREMIUM_INDEX_KLINES and "PREMIUM_INDEX_ZSCORE_7D" in indicators:
                strategy_required=True
            if strategy_required:
                requirements.append(ValidationDataRequirement("STRATEGY_CONTEXT",
                    item.data_request,item.dataset,item.interval))
    unique={}
    for item in requirements:
        key=(item.role,item.request.symbol,item.dataset,item.interval,item.request.start,item.request.end)
        unique[key]=item
    return tuple(_align_candle_requirement(item) for item in unique.values())


class ValidationDataPreparer:
    def __init__(self,store,backend): self.store,self.backend=store,backend

    def prepare(self,symbol,start,end,config,*,coverage_scope="MANDATORY_ENTRY",
                cancelled=lambda:False,progress=lambda event:None):
        rows=[]; self.store.refresh_catalog()
        for requirement in validation_data_requirements(symbol,start,end,config):
            if cancelled(): raise ValidationDataUnavailable(f"Validation blocked on {symbol}: cancelled")
            progress({"stage":"Checking required validation data","symbol":symbol,
                "dataset":requirement.dataset.value,"interval":requirement.interval})
            quality=(self.store.event_archive_quality_report
                if requirement.dataset in {DatasetKind.AGG_TRADES,DatasetKind.TRADES}
                else self.store.data_quality_report)
            report=quality(requirement.request,requirement.dataset,
                interval=requirement.interval,required=True)
            gaps=tuple(report.missing_coverage_ranges()); state=AcquisitionState.REUSED
            if report.status is not DataQualityStatus.OK:
                if report.has_non_missing_errors() or not gaps:
                    self._blocked(symbol,requirement,"QUALITY_FAILED",gaps,report)
                progress({"stage":f"Acquiring required {requirement.interval or requirement.dataset.value} data",
                    "symbol":symbol,"dataset":requirement.dataset.value,"interval":requirement.interval})
                outcome=self.backend.acquire_archive(ArchiveAcquisitionRequest(
                    requirement.request,requirement.dataset,requirement.interval,gaps),cancelled=cancelled)
                if outcome.state is not AcquisitionState.ACQUIRED:
                    self._blocked(symbol,requirement,outcome.state.value,gaps,report,outcome.detail)
                self.store.refresh_catalog()
                report=quality(requirement.request,requirement.dataset,
                    interval=requirement.interval,required=True)
                if report.status is not DataQualityStatus.OK:
                    self._blocked(symbol,requirement,"QUALITY_FAILED",gaps,report,"post-acquisition verification failed")
                state=AcquisitionState.ACQUIRED
            try: source=self.store.source_signature(requirement.request,requirement.dataset,
                interval=requirement.interval).cache_identity()
            except Exception: source=None
            rows.append({"symbol":requirement.request.symbol,"candidate_symbol":symbol,
                "dataset":requirement.dataset.value,"interval":requirement.interval,
                "required_role":requirement.role,"coverage_scope":coverage_scope,
                "requested_start":requirement.request.start,
                "requested_end":requirement.request.end,"state":state.value,
                "missing_ranges_attempted":json.dumps([{"start":x.start.isoformat(),"end":x.end.isoformat()} for x in gaps]),
                "post_verification_quality_status":report.status.value,"source_identity":source})
        return rows

    def common_available_end(self,symbol,start,end,config):
        """Latest common catalog end across strategy-decision-required inputs."""
        ends=[]
        for requirement in validation_data_requirements(symbol,start,end,config):
            coverage=self.store.catalog.coverage(self.store.raw_root,
                market=requirement.request.market,dataset=requirement.dataset,
                symbol=requirement.request.symbol,interval=requirement.interval)
            if coverage.last_period is None: return None
            value=__import__("pandas").Timestamp(coverage.last_period)
            ends.append(value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC"))
        return min(ends).to_pydatetime() if ends else None

    @staticmethod
    def _blocked(symbol,requirement,state,gaps,report,detail=None):
        ranges=", ".join(f"[{x.start.isoformat()}, {x.end.isoformat()})" for x in gaps) or "unknown range"
        raise ValidationDataUnavailable(f"Validation blocked on {symbol}\nRequired data unavailable: "
            f"{requirement.interval or requirement.dataset.value} {requirement.dataset.value}\n{ranges}\n"
            f"state={state}; quality={report.status.value}; {detail or ''}".strip())
