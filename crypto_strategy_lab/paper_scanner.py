"""Restart-safe periodic PAPER application layer over Task 1--7.

There is intentionally no broker, exchange-order client, or execution mode in
this module.  Persistence ordering for a new paper entry is: durable
``SIGNAL_PENDING`` audit row, atomic state ledger replacement, then descriptive
``SIGNAL_EMITTED``/``PAPER_ENTRY_RECORDED`` audit rows.  A crash can therefore
lose a descriptive row, but cannot re-submit the paper entry after restart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
import uuid
import time

import numpy as np
import pandas as pd

from crypto_strategy_lab.candidate_lifecycle import (
    CandidateLifecyclePolicy,
    DEFAULT_LIFECYCLE_POLICY,
    TransitionType,
    apply_candidate_lifecycle,
)
from crypto_strategy_lab.data.timing import normalize_binance_interval
from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    OpportunityScanCancelled,
    OpportunityScanRequest,
    OpportunityScannerApplicationService,
)
from crypto_strategy_lab.paper_scanner_reporting import PaperScannerAuditLog
from crypto_strategy_lab.paper_scanner_state import (
    PaperScannerState,
    PaperScannerStateStore,
)
from crypto_strategy_lab.scanner_operations import (
    DiskMonitor, HealthReporter, HealthStatus, ScannerHealthSnapshot,
    ScannerOperationalConfig, aggregate_acquisition_metrics,
    retry_delay_seconds, retryable_exception,
)


class CycleStatus(str, Enum):
    COMPLETED = "COMPLETED"
    STALE_DISCOVERY = "STALE_DISCOVERY"
    STALE_STRATEGY_DATA = "STALE_STRATEGY_DATA"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class StaleStrategyDataError(ValueError):
    """No completed causal strategy row is currently available."""


@dataclass(frozen=True, slots=True)
class PaperScannerConfig:
    scan_interval: timedelta
    stale_market_snapshot_limit: timedelta
    stale_strategy_candle_limit: timedelta
    api_retry_count: int
    retry_backoff: timedelta
    state_path: Path
    audit_log_path: Path
    max_signal_history: int = 10_000
    signal_history_retention: timedelta = timedelta(days=365)
    operational: ScannerOperationalConfig = ScannerOperationalConfig()

    def __post_init__(self) -> None:
        if self.scan_interval <= timedelta(0):
            raise ValueError("scan_interval must be positive")
        if self.stale_market_snapshot_limit <= timedelta(0):
            raise ValueError("market staleness must be positive")
        if self.stale_strategy_candle_limit <= timedelta(0):
            raise ValueError("strategy staleness must be positive")
        if self.api_retry_count < 0 or self.api_retry_count > 20:
            raise ValueError("api_retry_count must be between 0 and 20")
        if self.retry_backoff < timedelta(0) or self.retry_backoff > timedelta(hours=1):
            raise ValueError("retry_backoff must be between zero and one hour")
        if self.max_signal_history <= 0:
            raise ValueError("max_signal_history must be positive")
        if self.signal_history_retention <= self.stale_strategy_candle_limit:
            raise ValueError("signal_history_retention must exceed strategy staleness")
        object.__setattr__(self, "state_path", Path(self.state_path))
        object.__setattr__(self, "audit_log_path", Path(self.audit_log_path))


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    accepted: bool
    signal_candle_timestamp: datetime
    decision_available_at: datetime
    strategy_profile_key: str | None = None
    side: str | None = None
    reference_price: float | None = None
    detail: str = ""


class StrategyEvaluator(Protocol):
    def evaluate(
        self,
        candidate: dict[str, Any],
        decision_time: datetime,
        stale_limit: timedelta,
    ) -> StrategyEvaluation: ...


class LatestNativeStrategyEvaluator:
    """Small adapter around the native prepared frame and Entry/Veto method.

    ``engine_builder`` uses the Lab's normal data preparation/configuration and
    returns ``(PreparedBacktestFrame, RuleAware...Engine)``.  This adapter merely
    chooses the last causal completed row and invokes the engine's authoritative
    strategy-profile filter; it neither runs future exits nor research sampling.
    """

    def __init__(self, engine_builder: Callable[[dict[str, Any]], tuple[Any, Any]]):
        self.engine_builder = engine_builder

    def evaluate(
        self,
        candidate: dict[str, Any],
        decision_time: datetime,
        stale_limit: timedelta,
    ) -> StrategyEvaluation:
        prepared, engine = self.engine_builder(candidate)
        decision64 = np.datetime64(_utc(decision_time).replace(tzinfo=None), "ns")
        interval64 = pd.Timedelta(prepared.strategy_interval).to_timedelta64()
        completed_at = prepared.timestamp + interval64
        # A decision can never be available before its candle is complete, even
        # if every underlying feature advertises an earlier source timestamp.
        effective_available_at = np.maximum(
            prepared.decision_available_at, completed_at
        )
        research_by_name = {research.name: research for research in prepared.research}
        if len(research_by_name) != len(prepared.research):
            raise ValueError("prepared research block names are not unique")
        for research in research_by_name.values():
            if len(research.available_at) != len(prepared.timestamp):
                raise ValueError(
                    "research availability is not aligned to strategy rows"
                )
        base_causal = np.flatnonzero(effective_available_at <= decision64)
        index = None
        for candidate_index in reversed(base_causal.tolist()):
            required = engine.required_entry_research_features(candidate_index)
            missing = required - research_by_name.keys()
            if missing:
                raise ValueError(
                    f"required prepared research blocks are missing: {sorted(missing)}"
                )
            row_available = effective_available_at[candidate_index]
            for name in required:
                row_available = np.maximum(
                    row_available, research_by_name[name].available_at[candidate_index]
                )
            if row_available <= decision64:
                index = candidate_index
                effective_available_at[candidate_index] = row_available
                break
        if index is None:
            raise StaleStrategyDataError("no completed causal strategy candle")
        available = _from_np(effective_available_at[index])
        candle = _from_np(prepared.timestamp[index])
        if available > _utc(decision_time):
            raise ValueError("strategy decision is not causal")
        if _utc(decision_time) - available > stale_limit:
            return StrategyEvaluation(
                False, candle, available, detail="stale strategy decision row"
            )
        native = engine.evaluate_prepared_entry(index)
        return StrategyEvaluation(
            native.accepted,
            candle,
            available,
            native.strategy_profile_key,
            native.side,
            native.reference_price,
            native.detail,
        )


@dataclass(frozen=True, slots=True)
class PaperOpportunitySignal:
    signal_id: str
    scan_run_id: str
    scanner_decision_timestamp: str
    signal_candle_timestamp: str
    symbol: str
    strategy_interval: str
    strategy_profile_key: str
    side: str
    final_candidate_rank: int
    discovery_rank: int | None
    opportunity_model: str | None
    opportunity_rank: int | None
    opportunity_score: float | None
    reference_price: float | None
    strategy_source_identity: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class PaperScanCycleResult:
    cycle_id: str
    scanner_run_id: str | None
    decision_timestamp: str | None
    final_candidates: int
    candidates_evaluated: int
    fresh_candidates: int
    stale_candidates: int
    strategy_signals: int
    duplicate_signals_suppressed: int
    new_paper_entries: int
    status: CycleStatus
    metrics: dict[str, Any] | None = None


class PaperScannerRunner:
    def __init__(
        self,
        config: PaperScannerConfig,
        scanner: OpportunityScannerApplicationService,
        request_factory: Callable[[], OpportunityScanRequest],
        evaluator: StrategyEvaluator,
        *,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
        transient_error: Callable[[Exception], bool] | None = None,
        lifecycle_policy: CandidateLifecyclePolicy = DEFAULT_LIFECYCLE_POLICY,
        monotonic: Callable[[], float] | None = None,
        http_telemetry: list[Mapping[str, Any]] | None = None,
    ):
        self.config, self.scanner = config, scanner
        self.request_factory, self.evaluator = request_factory, evaluator
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper or time.sleep
        self.monotonic = monotonic or time.monotonic
        self.transient_error = transient_error or retryable_exception
        self.http_telemetry = http_telemetry
        self.store = PaperScannerStateStore(config.state_path)
        self.audit = PaperScannerAuditLog(config.audit_log_path, self.clock)
        self.state = self.store.load()  # corrupt state fails closed before runtime
        lifecycle_policy.validate_executable()
        if self.state.lifecycle_policy.identity != lifecycle_policy.identity:
            raise ValueError("configured lifecycle policy does not match durable state")
        self.lifecycle_policy = lifecycle_policy
        self.runtime_started = _utc(self.clock())
        self.health_reporter = HealthReporter(config.operational.health_path)
        self.disk_monitor = DiskMonitor(config.operational.disk)
        self.consecutive_failed_cycles = 0
        self.last_success = self.last_failure = self.last_error_category = None
        self._cycle_started = self.monotonic()
        self._metrics: dict[str, Any] = {}
        self.audit.append(
            "STATE_LOADED",
            "runtime",
            detail=(
                f"version=2 entries={len(self.state.paper_entries)} "
                f"lifecycle_policy={self.lifecycle_policy.identity}"
            ),
        )
        self._publish_health(HealthStatus.STARTING, None)

    def run_once(
        self,
        cancelled: Callable[[], bool] | None = None,
        cancellation_wait: Callable[[float], bool] | None = None,
    ) -> PaperScanCycleResult:
        cancellation_requested = cancelled or (lambda: False)
        self._cycle_started = self.monotonic()
        self._metrics = {
            "stale_market_limit_seconds": self.config.stale_market_snapshot_limit.total_seconds(),
            "stale_strategy_limit_seconds": self.config.stale_strategy_candle_limit.total_seconds(),
            "retry_count": 0,
        }
        cycle = uuid.uuid4().hex
        self.audit.append("SCAN_STARTED", cycle)
        completed = None
        for attempt in range(self.config.api_retry_count + 1):
            if cancellation_requested():
                self.audit.append("SCAN_CANCELLED", cycle)
                return self._finish(cycle, None, None, status=CycleStatus.CANCELLED)
            try:
                request = self.request_factory()
                if request.mode != "LIVE":
                    raise ValueError("paper scanner supports LIVE requests only")
                pipeline_started = self.monotonic()
                try:
                    completed = self.scanner.run(request, cancellation_requested)
                finally:
                    self._metrics["pipeline_duration_ms"] = (
                        self.monotonic() - pipeline_started
                    ) * 1000
                    self._capture_http_telemetry(cycle)
                self._metrics.update(aggregate_acquisition_metrics(
                    getattr(completed, "preliminary", None),
                    getattr(completed, "readiness", None),
                ))
                break
            except OpportunityScanCancelled:
                self.audit.append("SCAN_CANCELLED", cycle)
                return self._finish(cycle, None, None, status=CycleStatus.CANCELLED)
            except Exception as exc:
                if attempt >= self.config.api_retry_count or not self.transient_error(
                    exc
                ):
                    self.audit.append(
                        "SCAN_FAILED", cycle, detail=f"{type(exc).__name__}: {exc}"
                    )
                    return self._finish(cycle, None, None, status=CycleStatus.FAILED)
                backoff = retry_delay_seconds(
                    exc, attempt + 1, self.config.retry_backoff,
                    self.config.operational.retry_backoff_cap,
                )
                retry_after = getattr(exc, "headers", {}).get("Retry-After") if getattr(exc, "headers", None) else None
                self.audit.append(
                    "SCAN_RETRY",
                    cycle,
                    detail=f"attempt={attempt + 1} {type(exc).__name__}",
                    severity="WARNING",
                    component="scanner_pipeline",
                    fields={"error_type": type(exc).__name__, "attempt": attempt + 1,
                            "delay_seconds": backoff,
                            "http_status": getattr(exc, "code", None),
                            "retry_after_seconds": retry_after},
                )
                self._metrics["retry_count"] = attempt + 1
                self._metrics["last_retry_delay_seconds"] = backoff
                if cancellation_wait is not None:
                    if cancellation_wait(backoff):
                        self.audit.append("SCAN_CANCELLED", cycle)
                        return self._finish(
                            cycle, None, None, status=CycleStatus.CANCELLED
                        )
                else:
                    self.sleeper(backoff)
                    if cancellation_requested():
                        self.audit.append("SCAN_CANCELLED", cycle)
                        return self._finish(
                            cycle, None, None, status=CycleStatus.CANCELLED
                        )
        assert completed is not None
        run_id = str(completed.manifest["run_id"])
        decision = _parse(completed.summary["decision_timestamp"])
        self.audit.append("SCAN_COMPLETED", cycle, scan_run_id=run_id)
        age = _utc(self.clock()) - decision
        self._metrics["discovery_age_seconds"] = age.total_seconds()
        if age < timedelta(0) or age > self.config.stale_market_snapshot_limit:
            self.audit.append(
                "STALE_DISCOVERY",
                cycle,
                scan_run_id=run_id,
                detail=f"age_seconds={age.total_seconds()}",
                severity="ERROR", fields={"discovery_age_seconds": age.total_seconds()},
            )
            return self._finish(
                cycle,
                run_id,
                decision,
                len(completed.final),
                status=CycleStatus.STALE_DISCOVERY,
            )
        lifecycle_started = self.monotonic()
        try:
            lifecycle = apply_candidate_lifecycle(
                self.state.candidate_lifecycle,
                self.state.lifecycle_cursor,
                completed.final.to_dict("records"),
                decision,
                run_id,
                self.lifecycle_policy,
            )
        except ValueError as exc:
            # A non-causal scan must not modify either in-memory or durable
            # lifecycle state.  In particular, do not call _finish(), because
            # that would replace the otherwise unchanged state document.
            self.audit.append(
                "CANDIDATE_SET_REJECTED", cycle, scan_run_id=run_id,
                detail=f"decision_timestamp={decision.isoformat()} {exc}",
            )
            self._metrics["lifecycle_duration_ms"] = (
                self.monotonic() - lifecycle_started
            ) * 1000
            return self._finish_operational_only(
                cycle, run_id, decision, len(completed.final),
                error_category="LIFECYCLE_REJECTED",
            )
        self._metrics["lifecycle_duration_ms"] = (self.monotonic() - lifecycle_started) * 1000
        lifecycle_state = PaperScannerState(
            emitted_signal_ids=list(self.state.emitted_signal_ids),
            paper_entries=list(self.state.paper_entries),
            last_completed_cycle=self.state.last_completed_cycle,
            last_successful_scan_run_id=self.state.last_successful_scan_run_id,
            lifecycle_policy=self.lifecycle_policy,
            candidate_lifecycle=list(lifecycle.state),
            lifecycle_cursor=lifecycle.cursor,
        )
        try:
            self.store.save(lifecycle_state)
        except OSError as exc:
            self.audit.append(
                "STATE_PERSIST_FAILED", cycle, scan_run_id=run_id,
                detail=f"candidate lifecycle: {type(exc).__name__}: {exc}",
            )
            return self._finish_operational_only(
                cycle, run_id, decision, len(completed.final),
                error_category="LIFECYCLE_PERSIST_FAILED",
            )
        self.state = lifecycle_state
        for transition in lifecycle.transitions:
            event = {
                TransitionType.ACTIVATED: "CANDIDATE_ACTIVATED",
                TransitionType.REMOVED: "CANDIDATE_REMOVED",
                TransitionType.RANK_CHANGED: "CANDIDATE_RANK_CHANGED",
            }[transition.transition]
            self.audit.append(
                event, cycle, scan_run_id=run_id, symbol=transition.symbol,
                detail=(
                    f"decision_timestamp={transition.decision_timestamp} "
                    f"previous_final_rank={transition.previous_final_rank} "
                    f"final_rank={transition.final_rank} "
                    f"previous_discovery_rank={transition.previous_discovery_rank} "
                    f"discovery_rank={transition.discovery_rank}"
                ),
            )
        counts = {
            kind: sum(item.transition is kind for item in lifecycle.transitions)
            for kind in TransitionType
        }
        self.audit.append(
            "CANDIDATE_SET_APPLIED", cycle, scan_run_id=run_id,
            detail=(
                f"decision_timestamp={decision.isoformat()} active={len(lifecycle.active_rows)} "
                f"activated={counts[TransitionType.ACTIVATED]} "
                f"removed={counts[TransitionType.REMOVED]} "
                f"rank_changed={counts[TransitionType.RANK_CHANGED]}"
            ),
        )
        # Retention is based on the authoritative signal-candle time.  It is
        # strictly longer than the strategy staleness window, so an identity is
        # never removed while the corresponding signal could still be emitted.
        self._prune_expired_signal_history(decision)
        evaluated = fresh = stale = signals = duplicates = entries = 0
        strategy_started = self.monotonic()
        for row in lifecycle.active_rows:
            symbol = str(row["symbol"])
            try:
                evaluation = self.evaluator.evaluate(
                    row, decision, self.config.stale_strategy_candle_limit
                )
            except StaleStrategyDataError as exc:
                stale += 1
                self.audit.append(
                    "STALE_STRATEGY_DATA",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    detail=str(exc),
                )
                continue
            except (ValueError, IndexError, KeyError, TypeError) as exc:
                self.audit.append(
                    "STRATEGY_EVALUATION_FAILED",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                self._metrics["strategy_evaluation_duration_ms"] = (
                    self.monotonic() - strategy_started
                ) * 1000
                return self._finish(
                    cycle,
                    run_id,
                    decision,
                    len(completed.final),
                    evaluated,
                    fresh,
                    stale,
                    signals,
                    duplicates,
                    entries,
                    status=CycleStatus.FAILED,
                )
            evaluated += 1
            strategy_age = decision - _utc(evaluation.decision_available_at)
            if (
                strategy_age < timedelta(0)
                or strategy_age > self.config.stale_strategy_candle_limit
            ):
                stale += 1
                self.audit.append(
                    "STALE_STRATEGY_DATA",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    detail=f"age_seconds={strategy_age.total_seconds()}",
                )
                continue
            fresh += 1
            self.audit.append(
                "CANDIDATE_EVALUATED",
                cycle,
                scan_run_id=run_id,
                symbol=symbol,
                detail=evaluation.detail,
            )
            if (
                not evaluation.accepted
                or not evaluation.strategy_profile_key
                or not evaluation.side
            ):
                self.audit.append(
                    "NO_STRATEGY_ENTRY",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    detail=evaluation.detail,
                )
                continue
            signal = self._signal(completed, row, evaluation, decision)
            signals += 1
            if signal.signal_id in self.state.emitted_signal_ids:
                duplicates += 1
                self.audit.append(
                    "DUPLICATE_SUPPRESSED",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    signal_id=signal.signal_id,
                )
                continue
            if len(self.state.emitted_signal_ids) >= self.config.max_signal_history:
                # Never evict a still-live duplicate key just to satisfy the
                # bound.  Fail this cycle closed and preserve the prior ledger.
                self.audit.append(
                    "STATE_CAPACITY_REACHED",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    signal_id=signal.signal_id,
                    detail=f"max_signal_history={self.config.max_signal_history}",
                )
                self._metrics["strategy_evaluation_duration_ms"] = (
                    self.monotonic() - strategy_started
                ) * 1000
                return self._finish(
                    cycle,
                    run_id,
                    decision,
                    len(completed.final),
                    evaluated,
                    fresh,
                    stale,
                    signals,
                    duplicates,
                    entries,
                    status=CycleStatus.FAILED,
                )
            self.audit.append(
                "SIGNAL_PENDING",
                cycle,
                scan_run_id=run_id,
                symbol=symbol,
                signal_id=signal.signal_id,
            )
            next_state = PaperScannerState(
                emitted_signal_ids=[
                    *self.state.emitted_signal_ids,
                    signal.signal_id,
                ],
                paper_entries=[
                    *self.state.paper_entries,
                    {"record_type": "PAPER_ENTRY", **asdict(signal)},
                ],
                last_completed_cycle=self.state.last_completed_cycle,
                last_successful_scan_run_id=self.state.last_successful_scan_run_id,
                lifecycle_policy=self.state.lifecycle_policy,
                candidate_lifecycle=list(self.state.candidate_lifecycle),
                lifecycle_cursor=self.state.lifecycle_cursor,
            )
            try:
                self.store.save(next_state)
            except OSError as exc:
                # Do not install an uncommitted identity in memory.  The prior
                # durable state remains authoritative and a later cycle can
                # safely retry once storage is healthy.
                self.audit.append(
                    "STATE_PERSIST_FAILED",
                    cycle,
                    scan_run_id=run_id,
                    symbol=symbol,
                    signal_id=signal.signal_id,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                self._metrics["strategy_evaluation_duration_ms"] = (
                    self.monotonic() - strategy_started
                ) * 1000
                return self._finish_operational_only(
                    cycle, run_id, decision, len(completed.final), evaluated,
                    fresh, stale, signals, duplicates, entries,
                    error_category="SIGNAL_PERSIST_FAILED",
                )
            self.state = next_state
            entries += 1
            self.audit.append(
                "SIGNAL_EMITTED",
                cycle,
                scan_run_id=run_id,
                symbol=symbol,
                signal_id=signal.signal_id,
            )
            self.audit.append(
                "PAPER_ENTRY_RECORDED",
                cycle,
                scan_run_id=run_id,
                symbol=symbol,
                signal_id=signal.signal_id,
            )
        status = (
            CycleStatus.STALE_STRATEGY_DATA
            if stale and not fresh
            else CycleStatus.COMPLETED
        )
        self._metrics["strategy_evaluation_duration_ms"] = (self.monotonic() - strategy_started) * 1000
        return self._finish(
            cycle,
            run_id,
            decision,
            len(completed.final),
            evaluated,
            fresh,
            stale,
            signals,
            duplicates,
            entries,
            status=status,
        )

    def _prune_expired_signal_history(self, decision: datetime) -> None:
        cutoff = decision - self.config.signal_history_retention
        retained_entries = [
            entry
            for entry in self.state.paper_entries
            if _parse(entry["signal_candle_timestamp"]) >= cutoff
        ]
        if len(retained_entries) == len(self.state.paper_entries):
            return
        self.state.paper_entries[:] = retained_entries
        self.state.emitted_signal_ids[:] = [
            str(entry["signal_id"]) for entry in retained_entries
        ]
        self.store.save(self.state)

    def _signal(self, completed, row, evaluation, decision):
        interval = normalize_binance_interval(str(row["strategy_interval"]))
        candle = _utc(evaluation.signal_candle_timestamp)
        identity = "|".join(
            (
                str(row["symbol"]).upper(),
                interval,
                evaluation.strategy_profile_key,
                evaluation.side.upper(),
                candle.isoformat(),
            )
        )
        signal_id = hashlib.sha256(identity.encode()).hexdigest()
        model = row.get("opportunity_model_name")
        return PaperOpportunitySignal(
            signal_id,
            str(completed.manifest["run_id"]),
            decision.isoformat(),
            candle.isoformat(),
            str(row["symbol"]).upper(),
            interval,
            evaluation.strategy_profile_key,
            evaluation.side.upper(),
            int(row["final_rank"]),
            _optional_int(row.get("discovery_rank")),
            model if pd.notna(model) else None,
            _optional_int(row.get("opportunity_model_rank")),
            _optional_float(row.get("opportunity_score")),
            evaluation.reference_price,
            row.get("strategy_source_identity"),
            _utc(self.clock()).isoformat(),
        )

    def _finish(
        self,
        cycle,
        run_id,
        decision,
        final=0,
        evaluated=0,
        fresh=0,
        stale=0,
        signals=0,
        duplicates=0,
        entries=0,
        status=CycleStatus.COMPLETED,
    ):
        self._metrics.update({"fresh_candidates": fresh, "stale_candidates": stale,
                              "active_candidates": final,
                              "total_cycle_duration_ms": (self.monotonic() - self._cycle_started) * 1000})
        result = PaperScanCycleResult(
            cycle, run_id, decision.isoformat() if decision else None, final,
            evaluated, fresh, stale, signals, duplicates, entries, status,
            dict(self._metrics),
        )
        cycle_state = asdict(result)
        # Keep the on-disk schema independent of Enum.__str__ implementation
        # details ("COMPLETED", never "CycleStatus.COMPLETED").
        cycle_state["status"] = result.status.value
        self.state.last_completed_cycle = cycle_state
        if status is CycleStatus.COMPLETED:
            self.state.last_successful_scan_run_id = run_id
        self.store.save(self.state)
        return self._finalize_operations(result)

    def _finish_operational_only(
        self, cycle, run_id, decision, final=0, evaluated=0, fresh=0,
        stale=0, signals=0, duplicates=0, entries=0, *, error_category: str,
    ) -> PaperScanCycleResult:
        """Finalize health/audit without touching authoritative PAPER state."""
        self._metrics.update({
            "fresh_candidates": fresh,
            "stale_candidates": stale,
            "active_candidates": final,
            "total_cycle_duration_ms": (
                self.monotonic() - self._cycle_started
            ) * 1000,
        })
        result = PaperScanCycleResult(
            cycle, run_id, decision.isoformat() if decision else None, final,
            evaluated, fresh, stale, signals, duplicates, entries,
            CycleStatus.FAILED, dict(self._metrics),
        )
        return self._finalize_operations(result, error_category=error_category)

    def _finalize_operations(
        self, result: PaperScanCycleResult, *, error_category: str | None = None,
    ) -> PaperScanCycleResult:
        health_status, disk = self._health_status(result)
        category = error_category or result.status.value
        self.audit.append(
            "CYCLE_COMPLETED", result.cycle_id,
            scan_run_id=result.scanner_run_id, detail=result.status.value,
            severity=("ERROR" if health_status is HealthStatus.UNHEALTHY
                      else "WARNING" if health_status is HealthStatus.DEGRADED
                      else "INFO"),
            fields={"health_status": health_status.value, **self._metrics},
        )
        now = _utc(self.clock())
        if health_status is HealthStatus.UNHEALTHY:
            self.consecutive_failed_cycles += 1
            self.last_failure, self.last_error_category = now, category
        elif result.status is CycleStatus.COMPLETED:
            # Both HEALTHY and usable DEGRADED completions are successful scans.
            self.consecutive_failed_cycles = 0
            self.last_success, self.last_error_category = now, None
        # CANCELLED is operationally neutral: it is neither success nor failure
        # and therefore preserves prior success/failure accounting.
        self._publish_health(health_status, result, disk=disk)
        return result

    def run_forever(self, stop_event) -> None:
        self.audit.append("RUNTIME_STARTED", "runtime")
        try:
            while not stop_event.is_set():
                try:
                    self.run_once(stop_event.is_set, stop_event.wait)
                except Exception as exc:
                    self.consecutive_failed_cycles += 1
                    self.last_failure, self.last_error_category = _utc(self.clock()), type(exc).__name__
                    self.audit.append("RUNTIME_CYCLE_CRASH", "runtime", severity="ERROR",
                                      fields={"error_type": type(exc).__name__})
                    self._publish_health(HealthStatus.UNHEALTHY, None)
                    if stop_event.wait(self.config.operational.crash_recovery_delay.total_seconds()):
                        break
                    continue
                if stop_event.is_set():
                    break
                if stop_event.wait(self.config.scan_interval.total_seconds()):
                    break
        finally:
            self.audit.append("RUNTIME_STOPPED", "runtime")
            self._publish_health(HealthStatus.STOPPED, None)

    def _capture_http_telemetry(self, cycle: str) -> None:
        if self.http_telemetry is None or not self.http_telemetry:
            return
        observations = [dict(item) for item in self.http_telemetry]
        self.http_telemetry.clear()
        previous = self._metrics.get("binance_public_api", {})
        self._metrics["binance_public_api"] = {
            "request_count": previous.get("request_count", 0) + len(observations),
            "latest": observations[-1],
            "rate_limit_pressure": previous.get("rate_limit_pressure", False) or any(
                item.get("http_status") == 429 for item in observations
            ),
        }
        for fields in observations:
            self.audit.append(
                "BINANCE_API_RESPONSE", cycle, component="binance_public_api",
                severity=("WARNING" if fields.get("http_status") in {418, 429}
                          else "INFO"), fields=fields,
            )

    def _health_status(self, result: PaperScanCycleResult) -> tuple[HealthStatus, dict[str, Any]]:
        disk_error = False
        try:
            disk = dict(self.disk_monitor.sample())
            self._metrics.pop("disk_monitor_error", None)
        except OSError as exc:
            disk_error = True
            disk = dict(self.disk_monitor.latest)
            error = {
                "error_type": type(exc).__name__,
                "operation": "disk_cache_sample",
            }
            filename = getattr(exc, "filename", None)
            if filename is not None:
                error["path"] = str(filename)
            self._metrics["disk_monitor_error"] = error
            self.audit.append(
                "DISK_MONITOR_FAILED", result.cycle_id,
                scan_run_id=result.scanner_run_id,
                severity="WARNING", component="disk_monitor", fields=error,
            )
        disappeared = sum(
            int(value.get("entries_disappeared", 0) or 0)
            for value in disk.values() if isinstance(value, dict)
        )
        if disappeared:
            warning_fields = {
                "operation": "cache_walk",
                "entries_disappeared": disappeared,
            }
            self._metrics["disk_monitor_warning"] = warning_fields
            self.audit.append(
                "DISK_MONITOR_WARNING", result.cycle_id,
                scan_run_id=result.scanner_run_id, severity="WARNING",
                component="disk_monitor", fields=warning_fields,
            )
        else:
            self._metrics.pop("disk_monitor_warning", None)
        if result.status in {CycleStatus.FAILED, CycleStatus.STALE_DISCOVERY, CycleStatus.STALE_STRATEGY_DATA}:
            return HealthStatus.UNHEALTHY, disk
        if result.status is CycleStatus.CANCELLED:
            return HealthStatus.DEGRADED, disk
        levels = {value.get("level") for value in disk.values()
                  if isinstance(value, dict)}
        if "CRITICAL" in levels:
            return HealthStatus.UNHEALTHY, disk
        candle = self._metrics.get("strategy_candle_acquisition", {})
        rich = self._metrics.get("rich_feature_readiness", {})
        partial = any(candle.get(name, 0) for name in (
            "missing", "quality_failed", "download_failed", "cancelled"
        ))
        optional_degraded = bool(
            rich.get("degraded", 0) or rich.get("unavailable", 0)
        )
        rate_limit_pressure = self._metrics.get("binance_public_api", {}).get(
            "rate_limit_pressure", False
        )
        warning = (
            "WARNING" in levels or result.stale_candidates
            or self._metrics.get("retry_count") or partial or optional_degraded
            or rate_limit_pressure or disk_error or disappeared
        )
        return (HealthStatus.DEGRADED if warning else HealthStatus.HEALTHY), disk

    def _publish_health(
        self, status: HealthStatus, result: PaperScanCycleResult | None,
        *, disk: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            snapshot = ScannerHealthSnapshot(
                1, _utc(self.clock()).isoformat(), self.runtime_started.isoformat(), status.value,
                result.cycle_id if result else None, result.scanner_run_id if result else None,
                result.status.value if result else None,
                self.last_success.isoformat() if self.last_success else None,
                self.last_failure.isoformat() if self.last_failure else None,
                self.consecutive_failed_cycles, self.last_error_category,
                dict(self._metrics), dict(disk if disk is not None else self.disk_monitor.sample()),
            )
            self.health_reporter.publish(snapshot)
        except OSError as exc:
            self.audit.append("HEALTH_PUBLISH_FAILED", "runtime", severity="WARNING",
                              fields={"error_type": type(exc).__name__})


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse(value: str) -> datetime:
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _from_np(value) -> datetime:
    return pd.Timestamp(value).tz_localize("UTC").to_pydatetime()


def _optional_int(value):
    return None if value is None or pd.isna(value) else int(value)


def _optional_float(value):
    return None if value is None or pd.isna(value) else float(value)
