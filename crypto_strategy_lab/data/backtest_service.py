"""Prepare simulator inputs from the Binance Data Lake."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import pandas as pd

from crypto_strategy_lab.data.schemas import DatasetKind, MarketKind
from crypto_strategy_lab.features import production_feature_registry
from crypto_strategy_lab.features.trade_flow import TradeFlowContextFeatureProvider, trade_flow_resource
from crypto_strategy_lab.data.trade_aggregates import TradeAggregateStore
from crypto_strategy_lab.data.order_book import OrderBookSnapshotStore
from crypto_strategy_lab.features.order_book import (OrderBookContextFeatureProvider,
                                                       book_depth_resource,
                                                       book_ticker_resource)
from crypto_strategy_lab.features.basis import BasisContextFeatureProvider
from crypto_strategy_lab.features.cache import FeatureFrameCache
from crypto_strategy_lab.features.funding import FundingContextFeatureProvider
from crypto_strategy_lab.features.futures_positioning import (
    FUTURES_POSITIONING_PRICE_INTERVAL,
    FuturesPositioningFeatureProvider,
    futures_positioning_price_resource,
)
from crypto_strategy_lab.features.taker_flow import (
    TakerFlowContextFeatureProvider,
    taker_flow_resource,
)
from crypto_strategy_lab.features.technical import CORE_DIRECTIONAL_FEATURE_NAME
from .query import DataRequest
from .store import DataNotAvailableError, MarketDataStore
from .timing import interval_to_timedelta
from .quality import (
    DataQualityIssue,
    DataQualityReport,
    DataQualityStatus,
    DatasetQualityReport,
    validate_feature_timeline,
)


@dataclass(frozen=True, slots=True)
class BacktestDataBundle:
    request: DataRequest
    strategy: pd.DataFrame
    intrabar: pd.DataFrame | None
    technical_features: pd.DataFrame
    context_features: pd.DataFrame
    support_resistance_features: pd.DataFrame | None
    research_features: dict[str, pd.DataFrame]
    structural_benchmark: pd.DataFrame | None
    structural_benchmark_symbol: str | None
    structural_benchmark_interval: str | None
    state_transition_daily_features: pd.DataFrame | None = None
    data_quality: DataQualityReport | None = None
    intrabar_interval: str | None = None


def _cached_multisource_feature(
    store,
    request,
    datasets,
    provider,
    parameters=None,
    *,
    registry=None,
):
    """Cache a feature that depends on klines plus one or more event datasets."""
    registry = registry if registry is not None else production_feature_registry()
    return registry.execute(
        [provider.definition.name],
        request,
        datasets,
        parameters={provider.definition.name: dict(parameters or {})},
        cache=FeatureFrameCache(store.cache.root),
    )[provider.definition.name]


def _cached_catalog_feature(
    store,
    request,
    canonical,
    dataset,
    provider,
    parameters=None,
    registry=None,
):
    """Resolve a multisource feature cache before materializing its event data."""
    parameters = dict(parameters or {})
    kline_request = _dataset_request(request, DatasetKind.KLINES)
    signatures = (
        store.source_signature(
            kline_request, DatasetKind.KLINES, interval=request.strategy_interval
        ),
        store.source_signature(_dataset_request(request, dataset), dataset),
    )
    registry = registry if registry is not None else production_feature_registry()
    resolved = registry.resolve(
        [provider.definition.name], {provider.definition.name: parameters}
    )[0]
    source_ids = {
        kind: signature.cache_identity()
        for kind, signature in zip(provider.definition.required_datasets, signatures)
    }
    key = registry.identity(resolved, request, source_ids, {})
    cache = FeatureFrameCache(store.cache.root)
    cached = cache.load(provider.definition, request, key)
    if cached is not None:
        try:
            provider.definition.validate_output(cached, resolved.parameters)
            validate_feature_timeline(provider.definition, cached, resolved.parameters)
            return cached
        except ValueError:
            pass

    source = store.load_dataset(_dataset_request(request, dataset), dataset)
    if source.empty:
        return None
    frame = provider.compute(
        request,
        {DatasetKind.KLINES: canonical, dataset: source},
        resolved.parameters,
    )
    provider.definition.validate_output(frame, resolved.parameters)
    validate_feature_timeline(provider.definition, frame, resolved.parameters)
    frame.attrs["feature_cache_hit"] = False
    frame.attrs["feature_cache_key"] = key
    cache.store(provider.definition, request, key, frame)
    return frame


def _cached_feature_by_identities(
    store: MarketDataStore,
    request: DataRequest,
    provider,
    parameters: Mapping[str, object] | None,
    source_identities: Mapping[object, str],
    datasets_loader: Callable[[], Mapping[object, pd.DataFrame]],
    *,
    registry=None,
) -> pd.DataFrame:
    """Resolve L2 from catalog identities before materializing auxiliary sources."""
    registry = registry if registry is not None else production_feature_registry()
    normalized = dict(parameters or {})
    resolved = registry.resolve(
        [provider.definition.name],
        {provider.definition.name: normalized},
    )[0]
    key = registry.identity(resolved, request, source_identities, {})
    cache = FeatureFrameCache(store.cache.root)
    cached = cache.load(provider.definition, request, key)
    if cached is not None:
        try:
            provider.definition.validate_output(cached, resolved.parameters)
            validate_feature_timeline(provider.definition, cached, resolved.parameters)
            return cached
        except ValueError:
            pass

    datasets = datasets_loader()
    frame = provider.compute(request, datasets, resolved.parameters)
    provider.definition.validate_output(frame, resolved.parameters)
    validate_feature_timeline(provider.definition, frame, resolved.parameters)
    frame.attrs["feature_cache_hit"] = False
    frame.attrs["feature_cache_key"] = key
    cache.store(provider.definition, request, key, frame)
    return frame


def _dataset_request(
    request: DataRequest,
    dataset: DatasetKind,
    *,
    start=None,
) -> DataRequest:
    return DataRequest(
        symbol=request.symbol,
        start=start or request.start,
        end=request.end,
        strategy_interval=request.strategy_interval,
        market=request.market,
        exchange=request.exchange,
        datasets=(dataset,),
    )


def _interval_dataset_request(
    request: DataRequest,
    dataset: DatasetKind,
    interval: str,
    *,
    start=None,
) -> DataRequest:
    return DataRequest(
        symbol=request.symbol,
        start=start or request.start,
        end=request.end,
        strategy_interval=interval,
        market=request.market,
        exchange=request.exchange,
        datasets=(dataset,),
    )


_OPTIONAL_COVERAGE_ISSUES = {
    "LEADING_COVERAGE_GAP",
    "TRAILING_COVERAGE_GAP",
    "MISSING_INTERNAL_INTERVAL",
    "LEADING_SOURCE_COVERAGE_GAP",
    "TRAILING_SOURCE_COVERAGE_GAP",
}


def _quality_is_usable_optional(report: DatasetQualityReport) -> bool:
    """Allow partial research coverage while rejecting source-integrity errors."""
    if report.status is DataQualityStatus.MISSING:
        return False
    return not any(
        issue.severity is DataQualityStatus.ERROR
        and issue.code not in _OPTIONAL_COVERAGE_ISSUES
        for issue in report.issues
    )


def _nonfatal_fallback_report(
    report: DatasetQualityReport,
    *,
    requested_interval: str,
    actual_interval: str | None,
    policy: str,
) -> DatasetQualityReport:
    downgraded = tuple(
        replace(issue, severity=DataQualityStatus.WARN)
        if issue.severity is DataQualityStatus.ERROR
        else issue
        for issue in report.issues
    )
    fallback = DataQualityIssue(
        code="EXPLICIT_INTRABAR_FALLBACK",
        severity=DataQualityStatus.WARN,
        message=(
            "Requested intrabar data was unavailable or invalid; explicit "
            "fallback policy was used"
        ),
        details={
            "requested_interval": requested_interval,
            "actual_interval": actual_interval,
            "policy": policy,
        },
    )
    return replace(
        report,
        required=False,
        status=DataQualityStatus.WARN,
        issues=(*downgraded, fallback),
    )


def _research_parameters(
    feature_parameters: Mapping[str, Mapping[str, object]] | None,
    name: str,
) -> dict[str, object]:
    return dict((feature_parameters or {}).get(name, {}))


def _optional_futures_research_features(
    store: MarketDataStore,
    request: DataRequest,
    canonical: pd.DataFrame,
    *,
    trade_flow_enabled: bool = False,
    trade_flow_source: DatasetKind = DatasetKind.AGG_TRADES,
    large_trade_quote_threshold: float | None = None,
    registry=None,
    usable_datasets: set[DatasetKind] | None = None,
    feature_parameters: Mapping[str, Mapping[str, object]] | None = None,
    positioning_price_usable: bool = False,
    taker_flow_usable: bool = False,
    order_book_enabled: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load compact futures research blocks when validated coverage exists.

    Statistics are calculated on their native source timelines before causal
    alignment to strategy decisions. Partial optional history remains queryable;
    rows before source coverage stay NaN/UNKNOWN rather than becoming synthetic
    zeroes or causing the whole provider to disappear.
    """
    if request.market != MarketKind.FUTURES_UM:
        return {}

    allowed = usable_datasets
    result: dict[str, pd.DataFrame] = {}

    if order_book_enabled:
        provider = OrderBookContextFeatureProvider()
        parameters = _research_parameters(feature_parameters, provider.definition.name)
        snapshots = OrderBookSnapshotStore(store)
        datasets: dict[object, pd.DataFrame] = {DatasetKind.KLINES: canonical}
        source_ids: dict[object, str] = {
            DatasetKind.KLINES: store.source_signature(
                _dataset_request(request, DatasetKind.KLINES), DatasetKind.KLINES,
                interval=request.strategy_interval).cache_identity()
        }
        for dataset, resource in ((DatasetKind.BOOK_TICKER, book_ticker_resource()),
                                  (DatasetKind.BOOK_DEPTH, book_depth_resource())):
            if allowed is not None and dataset not in allowed:
                continue
            try:
                compact = snapshots.load(request, dataset)
                datasets[resource] = compact.frame
                source_ids[resource] = compact.source_identity
            except DataNotAvailableError:
                pass
        if len(datasets) == 1:
            raise DataNotAvailableError("order-book research requested but neither source overlaps")
        result[provider.definition.name] = _cached_feature_by_identities(
            store, request, provider, parameters, source_ids, lambda: datasets,
            registry=registry)

    if allowed is None or DatasetKind.FUTURES_METRICS in allowed:
        try:
            provider = FuturesPositioningFeatureProvider()
            parameters = _research_parameters(
                feature_parameters, provider.definition.name
            )
            warmup_days = max(
                1.0,
                float(parameters.get("oi_zscore_window_days", 7.0)),
            )
            metrics_request = _dataset_request(
                request,
                DatasetKind.FUTURES_METRICS,
                start=request.start - timedelta(days=warmup_days),
            )
            strategy_signature = store.source_signature(
                _dataset_request(request, DatasetKind.KLINES),
                DatasetKind.KLINES,
                interval=request.strategy_interval,
            )
            metrics_signature = store.source_signature(
                metrics_request,
                DatasetKind.FUTURES_METRICS,
            )
            source_ids: dict[object, str] = {
                DatasetKind.KLINES: strategy_signature.cache_identity(),
                DatasetKind.FUTURES_METRICS: metrics_signature.cache_identity(),
            }
            price_resource = futures_positioning_price_resource()
            price_request = _interval_dataset_request(
                request,
                DatasetKind.KLINES,
                FUTURES_POSITIONING_PRICE_INTERVAL,
                start=request.start - pd.Timedelta(hours=1),
            )
            if positioning_price_usable:
                price_signature = store.source_signature(
                    price_request,
                    DatasetKind.KLINES,
                    interval=FUTURES_POSITIONING_PRICE_INTERVAL,
                )
                source_ids[price_resource] = price_signature.cache_identity()

            def positioning_loader():
                datasets: dict[object, pd.DataFrame] = {
                    DatasetKind.KLINES: canonical,
                    DatasetKind.FUTURES_METRICS: store.load_dataset(
                        metrics_request, DatasetKind.FUTURES_METRICS
                    ),
                }
                if positioning_price_usable:
                    datasets[price_resource] = store.load_dataset(
                        price_request,
                        DatasetKind.KLINES,
                        interval=FUTURES_POSITIONING_PRICE_INTERVAL,
                    )
                return datasets

            result[provider.definition.name] = _cached_feature_by_identities(
                store,
                request,
                provider,
                parameters,
                source_ids,
                positioning_loader,
                registry=registry,
            )
        except DataNotAvailableError:
            pass

    if allowed is None or DatasetKind.FUNDING_RATE in allowed:
        try:
            provider = FundingContextFeatureProvider()
            parameters = _research_parameters(
                feature_parameters, provider.definition.name
            )
            warmup_days = max(
                1.0,
                float(parameters.get("funding_zscore_window_days", 7.0)),
            )
            funding_request = _dataset_request(
                request,
                DatasetKind.FUNDING_RATE,
                start=request.start - timedelta(days=warmup_days),
            )
            funding = store.load_dataset(
                funding_request, DatasetKind.FUNDING_RATE
            )
            if not funding.empty:
                result[provider.definition.name] = _cached_multisource_feature(
                    store,
                    request,
                    {
                        DatasetKind.KLINES: canonical,
                        DatasetKind.FUNDING_RATE: funding,
                    },
                    provider,
                    parameters,
                    registry=registry,
                )
        except DataNotAvailableError:
            pass

    if taker_flow_usable:
        try:
            provider = TakerFlowContextFeatureProvider()
            parameters = _research_parameters(
                feature_parameters, provider.definition.name
            )
            interval = str(parameters.get("taker_flow_interval", "5m"))
            interval_to_timedelta(interval)
            resource = taker_flow_resource(interval)
            source_request = _interval_dataset_request(
                request,
                DatasetKind.KLINES,
                interval,
                start=request.start - timedelta(hours=1),
            )
            strategy_signature = store.source_signature(
                _dataset_request(request, DatasetKind.KLINES),
                DatasetKind.KLINES,
                interval=request.strategy_interval,
            )
            source_signature = store.source_signature(
                source_request,
                DatasetKind.KLINES,
                interval=interval,
            )
            source_ids = {
                DatasetKind.KLINES: strategy_signature.cache_identity(),
                resource: source_signature.cache_identity(),
            }

            def taker_loader():
                return {
                    DatasetKind.KLINES: canonical,
                    resource: store.load_dataset(
                        source_request,
                        DatasetKind.KLINES,
                        interval=interval,
                    ),
                }

            result[provider.definition.name] = _cached_feature_by_identities(
                store,
                request,
                provider,
                parameters,
                source_ids,
                taker_loader,
                registry=registry,
            )
        except DataNotAvailableError:
            pass

    basis_provider = BasisContextFeatureProvider()
    basis_parameters = _research_parameters(
        feature_parameters, basis_provider.definition.name
    )
    basis_warmup_days = max(
        0.0,
        float(basis_parameters.get("basis_zscore_window_days", 7.0)),
    )
    reference_start = min(
        request.start - timedelta(days=basis_warmup_days),
        request.start - interval_to_timedelta(request.strategy_interval),
    )
    reference_frames: dict[DatasetKind, pd.DataFrame] = {}
    for dataset in (
        DatasetKind.MARK_PRICE_KLINES,
        DatasetKind.INDEX_PRICE_KLINES,
    ):
        if allowed is not None and dataset not in allowed:
            continue
        try:
            frame = store.load_dataset(
                _dataset_request(request, dataset, start=reference_start),
                dataset,
                interval=request.strategy_interval,
            )
            if not frame.empty:
                reference_frames[dataset] = frame
        except DataNotAvailableError:
            pass
    if (
        DatasetKind.MARK_PRICE_KLINES in reference_frames
        and DatasetKind.INDEX_PRICE_KLINES in reference_frames
    ):
        if allowed is None or DatasetKind.PREMIUM_INDEX_KLINES in allowed:
            try:
                premium = store.load_dataset(
                    _dataset_request(
                        request,
                        DatasetKind.PREMIUM_INDEX_KLINES,
                        start=reference_start,
                    ),
                    DatasetKind.PREMIUM_INDEX_KLINES,
                    interval=request.strategy_interval,
                )
                if not premium.empty:
                    reference_frames[DatasetKind.PREMIUM_INDEX_KLINES] = premium
            except DataNotAvailableError:
                pass
        result[basis_provider.definition.name] = _cached_multisource_feature(
            store,
            request,
            {DatasetKind.KLINES: canonical, **reference_frames},
            basis_provider,
            basis_parameters,
            registry=registry,
        )

    if trade_flow_enabled:
        if allowed is not None and trade_flow_source not in allowed:
            raise DataNotAvailableError(
                f"trade_flow_context requested but {trade_flow_source.value} is unavailable"
            )
        try:
            aggregate = TradeAggregateStore(store).load(
                _dataset_request(request, trade_flow_source), trade_flow_source,
                large_trade_quote_threshold=large_trade_quote_threshold)
            provider = TradeFlowContextFeatureProvider()
            resource = trade_flow_resource(trade_flow_source)
            parameters = _research_parameters(feature_parameters, provider.definition.name)
            source_ids = {DatasetKind.KLINES: store.source_signature(_dataset_request(request, DatasetKind.KLINES), DatasetKind.KLINES, interval=request.strategy_interval).cache_identity(),
                          resource: aggregate.source_identity}
            result[provider.definition.name] = _cached_feature_by_identities(
                store, request, provider, parameters, source_ids,
                lambda: {DatasetKind.KLINES: canonical, resource: aggregate.frame}, registry=registry)
            result[provider.definition.name].attrs.update(
                trade_aggregate_cache_hit=aggregate.cache_hit,
                partitions_built=aggregate.partitions_built,
                partitions_reused=aggregate.partitions_reused)
        except DataNotAvailableError as exc:
            raise DataNotAvailableError(
                f"trade_flow_context requested but {trade_flow_source.value} is unavailable"
            ) from exc
    return result


def load_backtest_bundle(
    store: MarketDataStore,
    request: DataRequest,
    *,
    market_regime_method: str | None = None,
    structural_regime_sma_days: int = 200,
    structural_regime_slope_lookback_days: int = 30,
    benchmark_interval: str = "1h",
    refresh_catalog: bool = True,
    intrabar_start: datetime | None = None,
    atr_period: int = 14,
    adx_period: int = 14,
    di_pressure_lookback: int = 3,
    bb_period: int = 20,
    bb_stddevs: float = 2.0,
    mean_reversion_period: int = 20,
    mean_reversion_mean_type: str = "SMA",
    mean_reversion_bb_stddevs: float = 2.0,
    mean_reversion_rsi_period: int = 14,
    mean_reversion_rsi_oversold: float = 30.0,
    mean_reversion_rsi_overbought: float = 70.0,
    mean_reversion_require_reentry: bool = True,
    enable_support_resistance_analysis: bool = False,
    sr_timeframe_minutes: int = 0,
    sr_pivot_left: int = 5,
    sr_pivot_right: int = 5,
    sr_lookback_bars: int = 200,
    sr_zone_width_atr: float = 0.5,
    sr_near_distance_atr: float = 0.75,
    enable_sr_hold_confirmation: bool = False,
    sr_hold_confirmation_bars: int = 3,
    sr_hold_confirmation_atr: float = 0.25,
    sr_break_tolerance_atr: float = 0.25,
    sr_break_basis: str = "CLOSE",
    trade_flow_enabled: bool = False,
    trade_flow_source: DatasetKind = DatasetKind.AGG_TRADES,
    large_trade_quote_threshold: float | None = None,
    feature_registry=None,
    feature_config=None,
    data_config=None,
) -> BacktestDataBundle:
    """Load validated canonical market data and reusable causal feature blocks.

    The composed path supplies component config objects and an injected registry.
    Individual feature keyword arguments remain temporarily for genuine
    non-composed callers; they are not the authoritative ResearchRunner contract.
    """
    if refresh_catalog:
        store.refresh_catalog()

    canonical = store.load_klines(request, request.strategy_interval)
    quality_reports = [
        store.data_quality_report(
            request,
            DatasetKind.KLINES,
            interval=request.strategy_interval,
            required=True,
            frame=canonical,
        )
    ]
    DataQualityReport(tuple(quality_reports)).raise_for_errors()
    strategy = canonical
    strategy_minutes = int(
        interval_to_timedelta(request.strategy_interval).total_seconds() // 60
    )

    if feature_config is not None:
        market_regime_method = str(feature_config.market_regime_method)
        structural_regime_sma_days = int(feature_config.structural_regime_sma_days)
        structural_regime_slope_lookback_days = int(
            feature_config.structural_regime_slope_lookback_days
        )
        enable_support_resistance_analysis = bool(
            feature_config.enable_support_resistance_analysis
        )
        trade_flow_enabled = bool(feature_config.trade_flow_enabled)
        order_book_enabled = bool(feature_config.order_book_enabled)
        trade_flow_source = DatasetKind[str(feature_config.trade_flow_source).upper()]
        large_trade_quote_threshold = feature_config.large_trade_quote_threshold
        feature_parameters = feature_config.registry_parameters(
            strategy_timeframe_minutes=strategy_minutes
        )
    else:
        order_book_enabled = False
        if market_regime_method is None:
            raise ValueError("market_regime_method is required without FeatureConfig")
        effective_sr_minutes = int(sr_timeframe_minutes or strategy_minutes)
        feature_parameters = {
            CORE_DIRECTIONAL_FEATURE_NAME: {
                "atr_period": int(atr_period),
                "adx_period": int(adx_period),
                "di_pressure_lookback": int(di_pressure_lookback),
            },
            "production_market_context": {
                "bb_period": int(bb_period),
                "bb_stddevs": float(bb_stddevs),
                "mean_reversion_period": int(mean_reversion_period),
                "mean_reversion_mean_type": str(mean_reversion_mean_type).upper(),
                "mean_reversion_bb_stddevs": float(mean_reversion_bb_stddevs),
                "mean_reversion_rsi_period": int(mean_reversion_rsi_period),
                "mean_reversion_rsi_oversold": float(mean_reversion_rsi_oversold),
                "mean_reversion_rsi_overbought": float(mean_reversion_rsi_overbought),
                "mean_reversion_require_reentry": bool(mean_reversion_require_reentry),
            },
        }
        if enable_support_resistance_analysis:
            feature_parameters["support_resistance"] = {
                "atr_period": int(atr_period),
                "sr_timeframe_minutes": effective_sr_minutes,
                "sr_pivot_left": int(sr_pivot_left),
                "sr_pivot_right": int(sr_pivot_right),
                "sr_lookback_bars": int(sr_lookback_bars),
                "sr_zone_width_atr": float(sr_zone_width_atr),
                "sr_near_distance_atr": float(sr_near_distance_atr),
                "enable_sr_hold_confirmation": bool(enable_sr_hold_confirmation),
                "sr_hold_confirmation_bars": int(sr_hold_confirmation_bars),
                "sr_hold_confirmation_atr": float(sr_hold_confirmation_atr),
                "sr_break_tolerance_atr": float(sr_break_tolerance_atr),
                "sr_break_basis": str(sr_break_basis).upper(),
            }

    usable_research_datasets: set[DatasetKind] | None = None
    positioning_price_usable = False
    taker_flow_usable = False
    if request.market == MarketKind.FUTURES_UM:
        usable_research_datasets = set()
        for dataset, interval in (
            (DatasetKind.FUTURES_METRICS, None),
            (DatasetKind.FUNDING_RATE, None),
            (DatasetKind.MARK_PRICE_KLINES, request.strategy_interval),
            (DatasetKind.INDEX_PRICE_KLINES, request.strategy_interval),
            (DatasetKind.PREMIUM_INDEX_KLINES, request.strategy_interval),
        ):
            report = store.data_quality_report(
                request,
                dataset,
                interval=interval,
                required=False,
            )
            quality_reports.append(report)
            if _quality_is_usable_optional(report):
                usable_research_datasets.add(dataset)

        price_quality_request = _interval_dataset_request(
            request,
            DatasetKind.KLINES,
            FUTURES_POSITIONING_PRICE_INTERVAL,
        )
        price_report = store.data_quality_report(
            price_quality_request,
            DatasetKind.KLINES,
            interval=FUTURES_POSITIONING_PRICE_INTERVAL,
            required=False,
        )
        quality_reports.append(price_report)
        positioning_price_usable = _quality_is_usable_optional(price_report)

        taker_parameters = _research_parameters(
            feature_parameters, "taker_flow_context"
        )
        taker_interval = str(taker_parameters.get("taker_flow_interval", "5m"))
        interval_to_timedelta(taker_interval)
        taker_quality_request = _interval_dataset_request(
            request,
            DatasetKind.KLINES,
            taker_interval,
        )
        taker_report = store.data_quality_report(
            taker_quality_request,
            DatasetKind.KLINES,
            interval=taker_interval,
            required=False,
        )
        quality_reports.append(taker_report)
        taker_flow_usable = _quality_is_usable_optional(taker_report)

        if trade_flow_enabled:
            coverage = store.catalog.coverage(store.raw_root, market=request.market,
                dataset=trade_flow_source, symbol=request.symbol)
            missing = coverage.archive_count == 0
            partial = (not missing and (coverage.first_period is None or coverage.last_period is None
                or coverage.first_period > request.start or coverage.last_period < request.end))
            issues = ()
            if missing:
                issues = (DataQualityIssue("MISSING_SOURCE", DataQualityStatus.ERROR,
                    f"No {trade_flow_source.value} source coverage", details={}),)
            elif partial:
                issues = (DataQualityIssue("PARTIAL_SOURCE_COVERAGE", DataQualityStatus.WARN,
                    f"Partial {trade_flow_source.value} source coverage", details={}),)
            agg_report = DatasetQualityReport(dataset=trade_flow_source.value,
                symbol=request.symbol, interval=None, required=True,
                requested_start=request.start.isoformat(), requested_end=request.end.isoformat(),
                observed_start=str(coverage.first_period) if coverage.first_period else None,
                observed_end=str(coverage.last_period) if coverage.last_period else None,
                complete_start=str(coverage.first_period) if coverage.first_period else None,
                complete_end=str(coverage.last_period) if coverage.last_period else None,
                row_count=0, source_identity=None,
                status=(DataQualityStatus.ERROR if missing else DataQualityStatus.WARN if partial else DataQualityStatus.OK),
                issues=issues)
            quality_reports.append(agg_report)
            DataQualityReport((agg_report,)).raise_for_errors()
            usable_research_datasets.add(trade_flow_source)

        if order_book_enabled:
            available_book_sources = 0
            for dataset in (DatasetKind.BOOK_TICKER, DatasetKind.BOOK_DEPTH):
                report = store.data_quality_report(request, dataset, required=False)
                quality_reports.append(report)
                if _quality_is_usable_optional(report):
                    usable_research_datasets.add(dataset)
                    available_book_sources += 1
            if not available_book_sources:
                missing = DatasetQualityReport(
                    dataset="order_book", symbol=request.symbol, interval="1m", required=True,
                    requested_start=request.start.isoformat(), requested_end=request.end.isoformat(),
                    observed_start=None, observed_end=None, complete_start=None, complete_end=None,
                    row_count=0, source_identity=None, status=DataQualityStatus.ERROR,
                    issues=(DataQualityIssue("MISSING_ORDER_BOOK_SOURCES", DataQualityStatus.ERROR,
                                             "Neither BOOK_TICKER nor BOOK_DEPTH overlaps the request"),))
                quality_reports.append(missing)
                DataQualityReport((missing,)).raise_for_errors()

    registry = (
        feature_registry if feature_registry is not None else production_feature_registry()
    )
    requested = ["production_market_context", "state_transition_daily"]
    if enable_support_resistance_analysis:
        requested.append("support_resistance")
    main_parameter_names = set(registry.dependency_order(requested))
    main_feature_parameters = {
        name: parameters
        for name, parameters in feature_parameters.items()
        if name in main_parameter_names
    }
    frames = registry.execute(
        requested,
        request,
        {DatasetKind.KLINES: canonical},
        parameters=main_feature_parameters,
        cache=FeatureFrameCache(store.cache.root),
    )
    directional = frames[CORE_DIRECTIONAL_FEATURE_NAME]
    context = frames["production_market_context"]
    sr_features = frames.get("support_resistance")
    state_transition_daily = frames["state_transition_daily"]

    research_features = _optional_futures_research_features(
        store,
        request,
        canonical,
        trade_flow_enabled=trade_flow_enabled,
        trade_flow_source=trade_flow_source if trade_flow_enabled else DatasetKind.AGG_TRADES,
        large_trade_quote_threshold=large_trade_quote_threshold if trade_flow_enabled else None,
        registry=registry,
        usable_datasets=usable_research_datasets,
        feature_parameters=feature_parameters,
        positioning_price_usable=positioning_price_usable,
        taker_flow_usable=taker_flow_usable,
        order_book_enabled=order_book_enabled,
    )

    intrabar = None
    actual_intrabar_interval = None
    if request.intrabar_interval:
        effective_start = (
            max(request.start, intrabar_start) if intrabar_start else request.start
        )
        intrabar_request = DataRequest(
            symbol=request.symbol,
            start=effective_start,
            end=request.end,
            strategy_interval=request.intrabar_interval,
            market=request.market,
            exchange=request.exchange,
        )
        requested_quality = store.data_quality_report(
            intrabar_request,
            DatasetKind.KLINES,
            interval=request.intrabar_interval,
            required=True,
        )
        if requested_quality.status is DataQualityStatus.ERROR:
            policy = str(
                getattr(data_config, "intrabar_missing_policy", "ERROR")
            ).upper()
            if policy == "ERROR":
                DataQualityReport((requested_quality,)).raise_for_errors()
            elif policy == "WARN_AND_USE_15M":
                fallback_interval = "15m"
                if interval_to_timedelta(fallback_interval) >= interval_to_timedelta(
                    request.strategy_interval
                ):
                    raise ValueError(
                        "15m intrabar fallback must remain below the strategy timeframe"
                    )
                quality_reports.append(
                    _nonfatal_fallback_report(
                        requested_quality,
                        requested_interval=request.intrabar_interval,
                        actual_interval=fallback_interval,
                        policy=policy,
                    )
                )
                fallback_request = DataRequest(
                    symbol=request.symbol,
                    start=effective_start,
                    end=request.end,
                    strategy_interval=fallback_interval,
                    market=request.market,
                    exchange=request.exchange,
                )
                fallback_quality = store.data_quality_report(
                    fallback_request,
                    DatasetKind.KLINES,
                    interval=fallback_interval,
                    required=True,
                )
                DataQualityReport((fallback_quality,)).raise_for_errors()
                quality_reports.append(fallback_quality)
                intrabar = store.load_execution_klines(
                    fallback_request, fallback_interval
                )
                actual_intrabar_interval = fallback_interval
            elif policy == "WARN_AND_CONTINUE":
                quality_reports.append(
                    _nonfatal_fallback_report(
                        requested_quality,
                        requested_interval=request.intrabar_interval,
                        actual_interval=None,
                        policy=policy,
                    )
                )
            else:
                raise ValueError(f"Unsupported intrabar missing policy: {policy}")
        else:
            quality_reports.append(requested_quality)
            intrabar = store.load_execution_klines(
                intrabar_request, request.intrabar_interval
            )
            actual_intrabar_interval = request.intrabar_interval

    benchmark = None
    benchmark_symbol = None
    benchmark_interval_used = None
    if market_regime_method in {"BTC_STRUCTURAL", "ASSET_STRUCTURAL"}:
        benchmark_symbol = (
            "BTCUSDT" if market_regime_method == "BTC_STRUCTURAL" else request.symbol
        )
        benchmark_interval_used = benchmark_interval
        warmup_days = (
            int(structural_regime_sma_days)
            + int(structural_regime_slope_lookback_days)
            + 7
        )
        benchmark_request = DataRequest(
            symbol=benchmark_symbol,
            start=request.start - timedelta(days=warmup_days),
            end=request.end,
            strategy_interval=benchmark_interval,
            market=request.market,
            exchange=request.exchange,
        )
        benchmark = store.load_klines(
            benchmark_request, benchmark_request.strategy_interval
        )
        benchmark_quality = store.data_quality_report(
            benchmark_request,
            DatasetKind.KLINES,
            interval=benchmark_request.strategy_interval,
            required=True,
            frame=benchmark,
        )
        quality_reports.append(benchmark_quality)
        DataQualityReport((benchmark_quality,)).raise_for_errors()

    data_quality = DataQualityReport(tuple(quality_reports))
    data_quality.raise_for_errors()
    return BacktestDataBundle(
        request=request,
        strategy=strategy,
        intrabar=intrabar,
        technical_features=directional,
        context_features=context,
        support_resistance_features=sr_features,
        research_features=research_features,
        structural_benchmark=benchmark,
        structural_benchmark_symbol=benchmark_symbol,
        structural_benchmark_interval=benchmark_interval_used,
        state_transition_daily_features=state_transition_daily,
        data_quality=data_quality,
        intrabar_interval=actual_intrabar_interval,
    )
