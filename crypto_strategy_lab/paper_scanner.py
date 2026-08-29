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
from typing import Any, Callable, Protocol
import uuid

import numpy as np
import pandas as pd

from crypto_strategy_lab.data.timing import normalize_binance_interval
from crypto_strategy_lab.gui.opportunity_scanner_controller import (
    OpportunityScanRequest,
    OpportunityScannerApplicationService,
)
from crypto_strategy_lab.paper_scanner_reporting import PaperScannerAuditLog
from crypto_strategy_lab.paper_scanner_state import (
    PaperScannerStateStore,
)


class CycleStatus(str, Enum):
    COMPLETED = "COMPLETED"
    STALE_DISCOVERY = "STALE_DISCOVERY"
    FAILED = "FAILED"


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
        causal = np.flatnonzero(prepared.decision_available_at <= decision64)
        if not len(causal):
            raise ValueError("no completed causal strategy candle")
        index = int(causal[-1])
        available = _from_np(prepared.decision_available_at[index])
        candle = _from_np(prepared.timestamp[index])
        if available > _utc(decision_time):
            raise ValueError("strategy decision is not causal")
        if _utc(decision_time) - available > stale_limit:
            return StrategyEvaluation(
                False, candle, available, detail="stale strategy decision row"
            )
        passed, detail = engine._strategy_profile_filter_result(
            index
        )  # native Entry/Veto boundary
        context = engine._profile_context(index)
        side = context[1] if context else None
        profile = context[2] if context else None
        return StrategyEvaluation(
            bool(passed),
            candle,
            available,
            profile,
            side,
            float(prepared.close[index]),
            str(detail),
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
    ):
        self.config, self.scanner = config, scanner
        self.request_factory, self.evaluator = request_factory, evaluator
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        import time

        self.sleeper = sleeper or time.sleep
        self.transient_error = transient_error or (
            lambda exc: isinstance(exc, (ConnectionError, TimeoutError))
        )
        self.store = PaperScannerStateStore(config.state_path)
        self.audit = PaperScannerAuditLog(config.audit_log_path, self.clock)
        self.state = self.store.load()  # corrupt state fails closed before runtime
        self.audit.append(
            "STATE_LOADED",
            "runtime",
            detail=f"version=1 entries={len(self.state.paper_entries)}",
        )

    def run_once(self) -> PaperScanCycleResult:
        cycle = uuid.uuid4().hex
        self.audit.append("SCAN_STARTED", cycle)
        completed = None
        for attempt in range(self.config.api_retry_count + 1):
            try:
                request = self.request_factory()
                if request.mode != "LIVE":
                    raise ValueError("paper scanner supports LIVE requests only")
                completed = self.scanner.run(request, lambda: False)
                break
            except Exception as exc:
                if attempt >= self.config.api_retry_count or not self.transient_error(
                    exc
                ):
                    self.audit.append(
                        "SCAN_FAILED", cycle, detail=f"{type(exc).__name__}: {exc}"
                    )
                    return self._finish(cycle, None, None, status=CycleStatus.FAILED)
                self.audit.append(
                    "SCAN_RETRY",
                    cycle,
                    detail=f"attempt={attempt + 1} {type(exc).__name__}",
                )
                self.sleeper(self.config.retry_backoff.total_seconds())
        assert completed is not None
        run_id = str(completed.manifest["run_id"])
        decision = _parse(completed.summary["decision_timestamp"])
        self.audit.append("SCAN_COMPLETED", cycle, scan_run_id=run_id)
        age = _utc(self.clock()) - decision
        if age < timedelta(0) or age > self.config.stale_market_snapshot_limit:
            self.audit.append(
                "STALE_DISCOVERY",
                cycle,
                scan_run_id=run_id,
                detail=f"age_seconds={age.total_seconds()}",
            )
            return self._finish(
                cycle,
                run_id,
                decision,
                len(completed.final),
                status=CycleStatus.STALE_DISCOVERY,
            )
        # Retention is based on the authoritative signal-candle time.  It is
        # strictly longer than the strategy staleness window, so an identity is
        # never removed while the corresponding signal could still be emitted.
        self._prune_expired_signal_history(decision)
        evaluated = fresh = stale = signals = duplicates = entries = 0
        for row in completed.final.sort_values("final_rank").to_dict("records"):
            symbol = str(row["symbol"])
            evaluation = self.evaluator.evaluate(
                row, decision, self.config.stale_strategy_candle_limit
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
            self.state.emitted_signal_ids.append(signal.signal_id)
            self.state.paper_entries.append(
                {"record_type": "PAPER_ENTRY", **asdict(signal)}
            )
            self.store.save(self.state)
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
        result = PaperScanCycleResult(
            cycle,
            run_id,
            decision.isoformat() if decision else None,
            final,
            evaluated,
            fresh,
            stale,
            signals,
            duplicates,
            entries,
            status,
        )
        self.state.last_completed_cycle = asdict(result)
        if status is CycleStatus.COMPLETED:
            self.state.last_successful_scan_run_id = run_id
        self.store.save(self.state)
        self.audit.append(
            "CYCLE_COMPLETED", cycle, scan_run_id=run_id, detail=status.value
        )
        return result

    def run_forever(self, stop_event) -> None:
        self.audit.append("RUNTIME_STARTED", "runtime")
        try:
            while not stop_event.is_set():
                self.run_once()
                if stop_event.wait(self.config.scan_interval.total_seconds()):
                    break
        finally:
            self.audit.append("RUNTIME_STOPPED", "runtime")


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
