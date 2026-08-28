"""Causal empirical-Bayes probability research for completed strategy trades.

The scorer is deliberately downstream of the simulator. It never changes a fill,
entry, exit or direction. Each entry is scored using only outcomes that were
already completed by that entry timestamp, so future trades cannot leak into the
probability estimate.

Model v2 factorizes the evidence into bounded families rather than requiring one
exact high-dimensional context match. This lets the model use much more of the
research surface while reducing double-counting among correlated indicators.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
from statistics import median
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_VERSION = "BAYES_EVIDENCE_V2"
DEFAULT_PRIOR_STRENGTH = 20.0
DEFAULT_MIN_CONTEXT_SAMPLES = 20
DEFAULT_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
DEFAULT_MIN_EVIDENCE_FAMILIES = 3
FAMILY_LOG_ODDS_CAP = 0.45
TOTAL_LOG_ODDS_CAP = 1.40

EVIDENCE_FAMILIES = (
    "core_direction",
    "trend_volatility",
    "momentum_mean_reversion",
    "support_resistance",
    "oi_positioning",
    "funding_basis",
    "taker_flow",
    "microstructure",
)


@dataclass
class _Stats:
    wins: int = 0
    losses: int = 0
    net_r_sum: float = 0.0

    @property
    def samples(self) -> int:
        return self.wins + self.losses

    def add(self, won: bool, net_r: float) -> None:
        if won:
            self.wins += 1
        else:
            self.losses += 1
        if np.isfinite(net_r):
            self.net_r_sum += float(net_r)


@dataclass(frozen=True)
class BayesianEstimate:
    side: str
    probability: float
    prior_probability: float
    expected_net_r: float
    context_samples: int
    side_samples: int
    lower_90: float
    upper_90: float
    confidence: str
    evidence_families: int
    evidence_items: int
    family_lifts: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _PendingObservation:
    side: str
    won: bool
    net_r: float
    tokens: tuple[tuple[str, str, str], ...]


def _numeric(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if np.isfinite(result) else math.nan


def _bucket(value, edges: tuple[float, ...]) -> str:
    value = _numeric(value)
    if not np.isfinite(value):
        return "UNKNOWN"
    lower = None
    for edge in edges:
        if value < edge:
            return f"<{edge:g}" if lower is None else f"{lower:g}-{edge:g}"
        lower = edge
    return f"{edges[-1]:g}+"


def _signed_bucket(value, edges: tuple[float, ...]) -> str:
    value = _numeric(value)
    if not np.isfinite(value):
        return "UNKNOWN"
    if abs(value) < 1e-15:
        return "ZERO"
    sign = "POS" if value > 0 else "NEG"
    return f"{sign}:{_bucket(abs(value), edges)}"


def _centered_bucket(value, center: float, edges: tuple[float, ...]) -> str:
    value = _numeric(value)
    if not np.isfinite(value):
        return "UNKNOWN"
    return _signed_bucket(value - center, edges)


def _category(value) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        if pd.isna(value):
            return "UNKNOWN"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    text = str(value).strip().upper()
    return text if text and text not in {"NAN", "NONE", "<NA>"} else "UNKNOWN"


def _first(row: pd.Series, *names: str):
    for name in names:
        if name not in row.index:
            continue
        value = row.get(name)
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return None


def _direction_relation(value, side: str) -> str:
    state = _category(value)
    if state in {"UNKNOWN", "NEUTRAL", "NONE", "NO_SIGNAL"}:
        return state
    if "LONG" in state:
        direction = "LONG"
    elif "SHORT" in state:
        direction = "SHORT"
    else:
        return state
    return "AGREE" if direction == side else "COUNTER"


def _pressure_state(row: pd.Series, side: str) -> str:
    direct_name = "long_di_pressure_state" if side == "LONG" else "short_di_pressure_state"
    existing = _category(row.get(direct_name))
    if existing != "UNKNOWN":
        return existing
    plus_change = _numeric(row.get("plus_di_change"))
    minus_change = _numeric(row.get("minus_di_change"))
    if not np.isfinite(plus_change) or not np.isfinite(minus_change):
        return "UNKNOWN"
    directional, opposing = (
        (plus_change, minus_change) if side == "LONG" else (minus_change, plus_change)
    )
    if directional > 0 and opposing < 0:
        return "EXPANDING"
    if directional < 0 and opposing > 0:
        return "CONTRACTING"
    return "MIXED"


def _append(
    groups: dict[str, list[tuple[str, str]]],
    family: str,
    label: str,
    state: str,
) -> None:
    state = _category(state)
    if state != "UNKNOWN":
        groups[family].append((label, state))


def _evidence_tokens(row: pd.Series, side: str) -> tuple[tuple[str, str, str], ...]:
    """Return compact side-aware evidence tokens present on this trade row."""
    side = str(side).upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"unsupported Bayesian side: {side}")
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)

    regime = _first(row, "market_regime", "research_regime_state")
    _append(groups, "core_direction", "regime", _category(regime))

    plus_di = _numeric(row.get("plus_di"))
    minus_di = _numeric(row.get("minus_di"))
    if np.isfinite(plus_di) and np.isfinite(minus_di):
        if plus_di == minus_di:
            relation = "TIE"
        else:
            dominant = "LONG" if plus_di > minus_di else "SHORT"
            relation = "AGREE" if dominant == side else "COUNTER"
        _append(groups, "core_direction", "di_relation", relation)
        directional_di = plus_di if side == "LONG" else minus_di
        _append(
            groups,
            "core_direction",
            "directional_di",
            _bucket(directional_di, (10.0, 20.0, 30.0, 40.0)),
        )
        _append(
            groups,
            "core_direction",
            "di_spread",
            _bucket(abs(plus_di - minus_di), (5.0, 10.0, 20.0, 30.0, 40.0)),
        )
    _append(groups, "core_direction", "di_pressure", _pressure_state(row, side))
    _append(
        groups,
        "core_direction",
        "di_spread_change",
        _signed_bucket(row.get("di_spread_change"), (2.5, 5.0, 10.0, 20.0)),
    )

    # Optional higher-timeframe direction labels are consumed when present.
    for timeframe, candidates in (
        ("1h", ("di_direction_1h", "direction_1h", "one_hour_di_direction")),
        ("4h", ("di_direction_4h", "direction_4h", "four_hour_di_direction")),
        ("1d", ("di_direction_1d", "direction_1d", "daily_di_direction")),
    ):
        value = _first(row, *candidates)
        if value is not None:
            _append(
                groups,
                "core_direction",
                f"di_{timeframe}_relation",
                _direction_relation(value, side),
            )

    _append(
        groups,
        "trend_volatility",
        "adx",
        _bucket(row.get("adx"), (15.0, 20.0, 25.0, 30.0, 40.0)),
    )
    _append(
        groups,
        "trend_volatility",
        "atr_pct",
        _bucket(row.get("atr_pct"), (0.005, 0.01, 0.02, 0.04, 0.08)),
    )
    _append(
        groups,
        "trend_volatility",
        "bb_width",
        _bucket(_first(row, "bb_width_pct", "bb_width"), (0.02, 0.04, 0.06, 0.10, 0.16)),
    )
    _append(
        groups,
        "trend_volatility",
        "bb_width_change",
        _signed_bucket(
            _first(row, "bb_width_change_pct", "bb_width_change"),
            (0.05, 0.15, 0.35, 0.75),
        ),
    )

    _append(
        groups,
        "momentum_mean_reversion",
        "mr_state",
        _category(row.get("mean_reversion_state")),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "mr_motion",
        _category(row.get("mean_reversion_motion")),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "mr_signal_relation",
        _direction_relation(
            _first(row, "mean_reversion_signal_direction", "mr_signal_direction"),
            side,
        ),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "mr_bb_location",
        _category(row.get("mean_reversion_bb_location")),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "mr_rsi_state",
        _category(row.get("mean_reversion_rsi_state")),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "mr_zscore",
        _signed_bucket(row.get("mean_reversion_bb_zscore"), (0.5, 1.0, 1.5, 2.0, 3.0)),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "mr_distance_atr",
        _signed_bucket(row.get("mean_reversion_distance_atr"), (0.5, 1.0, 1.5, 2.5)),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "rsi",
        _bucket(_first(row, "rsi", "mean_reversion_rsi"), (30.0, 40.0, 50.0, 60.0, 70.0)),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "momentum",
        _signed_bucket(_first(row, "momentum", "momentum_return"), (0.005, 0.015, 0.03, 0.06)),
    )
    _append(
        groups,
        "momentum_mean_reversion",
        "vwap_distance",
        _signed_bucket(
            _first(row, "vwap_distance", "vwap_distance_atr"),
            (0.25, 0.75, 1.5, 3.0),
        ),
    )

    prefix = side.lower()
    sr_candidates = {
        "location": (f"{prefix}_trade_location_rating", "sr_trade_location_rating"),
        "room": (f"{prefix}_room_in_direction_atr", "sr_room_in_direction_atr"),
        "near_support": (f"{prefix}_near_support", "sr_near_support"),
        "near_resistance": (f"{prefix}_near_resistance", "sr_near_resistance"),
        "inside_support": (f"{prefix}_inside_support_zone", "sr_inside_support_zone"),
        "inside_resistance": (f"{prefix}_inside_resistance_zone", "sr_inside_resistance_zone"),
        "support_state": (f"{prefix}_support_state", "sr_support_state"),
        "resistance_state": (f"{prefix}_resistance_state", "sr_resistance_state"),
        "support_held": (f"{prefix}_support_held", "sr_support_held"),
        "resistance_held": (f"{prefix}_resistance_held", "sr_resistance_held"),
    }
    for label in (
        "location",
        "near_support",
        "near_resistance",
        "inside_support",
        "inside_resistance",
        "support_state",
        "resistance_state",
        "support_held",
        "resistance_held",
    ):
        value = _first(row, *sr_candidates[label])
        if value is not None:
            _append(groups, "support_resistance", label, _category(value))
    room = _first(row, *sr_candidates["room"])
    if room is not None:
        _append(
            groups,
            "support_resistance",
            "room_atr",
            _bucket(room, (0.75, 1.5, 2.0, 3.0, 5.0)),
        )
    for label, candidates in (
        ("support_distance", (f"{prefix}_nearest_support_distance_atr", "sr_support_distance_atr")),
        ("resistance_distance", (f"{prefix}_nearest_resistance_distance_atr", "sr_resistance_distance_atr")),
    ):
        value = _first(row, *candidates)
        if value is not None:
            _append(
                groups,
                "support_resistance",
                label,
                _bucket(value, (0.5, 1.0, 2.0, 4.0, 8.0)),
            )

    _append(
        groups,
        "oi_positioning",
        "price_oi_state",
        _category(_first(row, "oi_vs_price_state_1h", "price_oi_state")),
    )
    for label, names, edges in (
        ("oi_change_1h", ("oi_change_pct_1h",), (0.0025, 0.01, 0.025, 0.05)),
        ("oi_change_24h", ("oi_change_pct_24h",), (0.01, 0.03, 0.07, 0.15)),
        ("oi_zscore", ("oi_zscore_7d",), (0.5, 1.0, 2.0, 3.0)),
        ("price_change_1h", ("price_change_pct_1h",), (0.0025, 0.01, 0.025, 0.05)),
        ("top_account_bias", ("top_trader_account_bias",), (0.05, 0.15, 0.35, 0.75)),
        ("top_position_bias", ("top_trader_position_bias",), (0.05, 0.15, 0.35, 0.75)),
        ("global_bias", ("global_long_short_account_bias",), (0.05, 0.15, 0.35, 0.75)),
    ):
        value = _first(row, *names)
        if value is not None:
            _append(groups, "oi_positioning", label, _signed_bucket(value, edges))

    _append(groups, "funding_basis", "funding_bias", _category(row.get("funding_bias")))
    for label, names, edges in (
        ("funding_rate", ("funding_rate_bps",), (0.25, 0.75, 2.0, 5.0)),
        ("funding_24h", ("funding_24h_sum_bps",), (0.5, 1.5, 4.0, 10.0)),
        ("funding_change", ("funding_change_bps", "funding_change"), (0.25, 0.75, 2.0, 5.0)),
        ("funding_zscore", ("funding_7d_zscore", "funding_zscore_7d"), (0.5, 1.0, 2.0, 3.0)),
        ("mark_index_basis", ("mark_index_basis_bps",), (0.5, 1.0, 2.5, 5.0, 10.0)),
        ("mark_index_zscore", ("mark_index_basis_zscore_7d",), (0.5, 1.0, 2.0, 3.0)),
        ("premium_zscore", ("premium_index_zscore_7d",), (0.5, 1.0, 2.0, 3.0)),
    ):
        value = _first(row, *names)
        if value is not None:
            _append(groups, "funding_basis", label, _signed_bucket(value, edges))
    _append(
        groups,
        "funding_basis",
        "basis_state",
        _category(row.get("mark_index_basis_state")),
    )
    for flag in ("funding_extreme_positive", "funding_extreme_negative"):
        if flag in row.index:
            _append(groups, "funding_basis", flag, _category(row.get(flag)))

    ratio = _first(row, "taker_buy_sell_ratio")
    if ratio is not None:
        _append(
            groups,
            "taker_flow",
            "buy_sell_ratio",
            _centered_bucket(ratio, 1.0, (0.05, 0.15, 0.35, 0.75, 1.5)),
        )
    for label, names, edges in (
        ("delta_15m", ("taker_delta_pct_15m",), (0.03, 0.10, 0.20, 0.40)),
        ("delta_1h", ("taker_delta_pct_1h",), (0.03, 0.10, 0.20, 0.40)),
        ("positioning_bias", ("taker_long_short_volume_bias",), (0.05, 0.15, 0.35, 0.75)),
    ):
        value = _first(row, *names)
        if value is not None:
            _append(groups, "taker_flow", label, _signed_bucket(value, edges))
    persistence = _first(row, "flow_persistence", "taker_flow_persistence")
    if persistence is not None:
        _append(
            groups,
            "taker_flow",
            "persistence",
            _bucket(persistence, (0.4, 0.6, 0.75, 0.9)),
        )

    # Expensive diagnostics are optional. Coverage/staleness flags are honored so
    # missing or stale snapshots never masquerade as genuine zero-valued evidence.
    trade_flow_allowed = (
        "trade_source_covered" not in row.index
        or bool(row.get("trade_source_covered"))
    )
    if trade_flow_allowed:
        for label, names, edges in (
            ("trade_delta_5m", ("trade_delta_pct_5m",), (0.03, 0.10, 0.20, 0.40)),
            ("trade_delta_15m", ("trade_delta_pct_15m",), (0.03, 0.10, 0.20, 0.40)),
            ("trade_delta_1h", ("trade_delta_pct_1h",), (0.03, 0.10, 0.20, 0.40)),
            ("trade_intensity_change", ("trade_intensity_change",), (0.1, 0.3, 0.75, 1.5)),
            ("cvd_1h", ("cvd_1h",), (1.0, 10.0, 100.0, 1000.0)),
        ):
            value = _first(row, *names)
            if value is not None:
                _append(groups, "microstructure", label, _signed_bucket(value, edges))
        for label, names in (
            ("large_buy_share_15m", ("large_buy_share_15m",)),
            ("large_sell_share_15m", ("large_sell_share_15m",)),
        ):
            value = _first(row, *names)
            if value is not None:
                _append(
                    groups,
                    "microstructure",
                    label,
                    _bucket(value, (0.1, 0.25, 0.5, 0.75)),
                )

    ticker_allowed = not (
        ("book_ticker_observed" in row.index and not bool(row.get("book_ticker_observed")))
        or ("book_ticker_stale" in row.index and bool(row.get("book_ticker_stale")))
    )
    if ticker_allowed:
        for label, names, edges in (
            ("book_spread_bps", ("book_spread_bps",), (0.5, 1.0, 2.0, 5.0, 10.0)),
            ("book_imbalance_l1", ("book_imbalance_l1",), (0.1, 0.25, 0.5, 0.75)),
            ("book_microprice_offset", ("book_microprice_offset_bps",), (0.25, 0.75, 2.0, 5.0)),
        ):
            value = _first(row, *names)
            if value is not None:
                _append(groups, "microstructure", label, _signed_bucket(value, edges))

    depth_allowed = not (
        ("book_depth_observed" in row.index and not bool(row.get("book_depth_observed")))
        or ("book_depth_stale" in row.index and bool(row.get("book_depth_stale")))
    )
    if depth_allowed:
        value = _first(row, "book_depth_imbalance_1pct")
        if value is not None:
            _append(
                groups,
                "microstructure",
                "book_depth_imbalance",
                _signed_bucket(value, (0.1, 0.25, 0.5, 0.75)),
            )

    return tuple(
        (family, label, state)
        for family in EVIDENCE_FAMILIES
        for label, state in groups.get(family, ())
    )


def _logit(probability: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _confidence(
    context_samples: int,
    evidence_families: int,
    side_samples: int,
) -> str:
    if context_samples >= 60 and evidence_families >= 4 and side_samples >= 120:
        return "HIGH"
    if (
        context_samples >= DEFAULT_MIN_CONTEXT_SAMPLES
        and evidence_families >= DEFAULT_MIN_EVIDENCE_FAMILIES
        and side_samples >= 40
    ):
        return "MEDIUM"
    return "LOW"


class BayesianTradeModel:
    """Causal empirical-Bayes model with bounded factorized evidence families."""

    def __init__(self, prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> None:
        if prior_strength <= 0:
            raise ValueError("Bayesian prior strength must be positive")
        self.prior_strength = float(prior_strength)
        self.global_stats = _Stats()
        self.side_stats = {"LONG": _Stats(), "SHORT": _Stats()}
        self.feature_stats: dict[tuple[str, str, str, str], _Stats] = defaultdict(_Stats)

    def observe_tokens(
        self,
        side: str,
        won: bool,
        net_r: float,
        tokens: tuple[tuple[str, str, str], ...],
    ) -> None:
        side = str(side).upper()
        if side not in self.side_stats:
            return
        self.global_stats.add(bool(won), net_r)
        self.side_stats[side].add(bool(won), net_r)
        for family, label, state in tokens:
            self.feature_stats[(side, family, label, state)].add(bool(won), net_r)

    def observe(self, row: pd.Series) -> None:
        side = str(row.get("side", "")).upper()
        if side not in self.side_stats:
            return
        pnl = _numeric(row.get("pair_net_pnl"))
        if not np.isfinite(pnl):
            return
        net_r = _numeric(row.get("pair_net_r"))
        self.observe_tokens(side, pnl > 0, net_r, _evidence_tokens(row, side))

    def estimate(self, row: pd.Series, side: str) -> BayesianEstimate:
        side = str(side).upper()
        if side not in self.side_stats:
            raise ValueError(f"unsupported Bayesian side: {side}")

        baseline = self.side_stats[side]
        global_probability = (self.global_stats.wins + 1.0) / (self.global_stats.samples + 2.0)
        global_expected_r = (
            self.global_stats.net_r_sum / self.global_stats.samples
            if self.global_stats.samples
            else 0.0
        )
        # Side history is itself shrunk toward the all-trade baseline. This avoids
        # treating a short early streak on one side as a trustworthy 80-90% prior.
        prior_probability = (
            global_probability * self.prior_strength + baseline.wins
        ) / (self.prior_strength + baseline.samples)
        prior_expected_r = (
            global_expected_r * self.prior_strength + baseline.net_r_sum
        ) / (self.prior_strength + baseline.samples)
        prior_logit = _logit(prior_probability)

        family_rows: dict[str, list[tuple[float, int, float]]] = defaultdict(list)
        for family, label, state in _evidence_tokens(row, side):
            stats = self.feature_stats[(side, family, label, state)]
            if stats.samples <= 0:
                continue
            posterior = (
                prior_probability * self.prior_strength + stats.wins
            ) / (self.prior_strength + stats.samples)
            raw_lift = _logit(posterior) - prior_logit
            # A second reliability taper prevents dozens of tiny matching buckets
            # from accumulating into artificial certainty.
            reliability = stats.samples / (stats.samples + self.prior_strength)
            lift = raw_lift * reliability
            token_expected_r = (
                prior_expected_r * self.prior_strength + stats.net_r_sum
            ) / (self.prior_strength + stats.samples)
            family_rows[family].append(
                (lift, stats.samples, token_expected_r - prior_expected_r)
            )

        family_lifts: list[tuple[str, float]] = []
        family_support: list[int] = []
        expected_r_deltas: list[float] = []
        evidence_items = 0
        for family in EVIDENCE_FAMILIES:
            rows = family_rows.get(family, ())
            if not rows:
                continue
            evidence_items += len(rows)
            weights = np.array([math.sqrt(samples) for _, samples, _ in rows], dtype=float)
            lifts = np.array([lift for lift, _, _ in rows], dtype=float)
            deltas = np.array([delta for _, _, delta in rows], dtype=float)
            family_lift = float(np.average(lifts, weights=weights))
            family_lift = max(-FAMILY_LOG_ODDS_CAP, min(FAMILY_LOG_ODDS_CAP, family_lift))
            family_delta = float(np.average(deltas, weights=weights))
            support = int(median([samples for _, samples, _ in rows]))
            family_lifts.append((family, family_lift))
            family_support.append(support)
            expected_r_deltas.append(family_delta)

        combined_lift = sum(lift for _, lift in family_lifts)
        combined_lift = max(-TOTAL_LOG_ODDS_CAP, min(TOTAL_LOG_ODDS_CAP, combined_lift))
        probability = _sigmoid(prior_logit + combined_lift)

        evidence_families = len(family_lifts)
        context_samples = int(median(family_support)) if family_support else 0
        expected_r = prior_expected_r
        if expected_r_deltas:
            # Families are not independent, so average their R deltas instead of summing.
            expected_r += float(np.mean(expected_r_deltas))

        # Conservative diagnostic interval: use the median supporting sample count,
        # not the sum across correlated evidence families.
        effective_total = self.prior_strength + context_samples
        variance = probability * (1.0 - probability) / max(1.0, effective_total + 1.0)
        radius = 1.645 * math.sqrt(max(0.0, variance))

        return BayesianEstimate(
            side=side,
            probability=float(probability),
            prior_probability=float(prior_probability),
            expected_net_r=float(expected_r),
            context_samples=context_samples,
            side_samples=int(baseline.samples),
            lower_90=max(0.0, float(probability - radius)),
            upper_90=min(1.0, float(probability + radius)),
            confidence=_confidence(context_samples, evidence_families, baseline.samples),
            evidence_families=evidence_families,
            evidence_items=evidence_items,
            family_lifts=tuple(family_lifts),
        )


def _entry_column(trades: pd.DataFrame) -> str:
    for name in ("actual_entry_timestamp", "entry_time", "strategy_entry_time"):
        if name in trades.columns:
            return name
    raise ValueError("Bayesian research requires an entry timestamp column")


def _required_columns(trades: pd.DataFrame) -> None:
    required = {"side", "exit_time", "pair_net_pnl", "pair_net_r"}
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"Bayesian research missing trade columns: {missing}")
    _entry_column(trades)


def _threshold_column(threshold: float, min_context_samples: int) -> str:
    pct = int(round(float(threshold) * 100))
    return f"bayes_take_{pct}_min{int(min_context_samples)}"


def _lift_map(estimate: BayesianEstimate) -> dict[str, float]:
    return {family: float(lift) for family, lift in estimate.family_lifts}


def _top_evidence(estimate: BayesianEstimate, positive: bool) -> str:
    rows = [
        (family, lift)
        for family, lift in estimate.family_lifts
        if (lift > 0 if positive else lift < 0)
    ]
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return "; ".join(f"{family}:{lift:+.3f}" for family, lift in rows[:3])


def enrich_bayesian_trade_probabilities(
    trades: pd.DataFrame,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    min_context_samples: int = DEFAULT_MIN_CONTEXT_SAMPLES,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    min_evidence_families: int = DEFAULT_MIN_EVIDENCE_FAMILIES,
) -> pd.DataFrame:
    """Attach causal walk-forward LONG/SHORT probabilities to ``trades`` in place."""
    if min_context_samples < 1:
        raise ValueError("Bayesian minimum context samples must be positive")
    if min_evidence_families < 1:
        raise ValueError("Bayesian minimum evidence families must be positive")
    thresholds = tuple(float(value) for value in thresholds)
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Bayesian thresholds must be between 0 and 1")

    base_columns = {
        "bayes_model_version": pd.Series(dtype="string"),
        "bayes_long_probability": pd.Series(dtype="float64"),
        "bayes_short_probability": pd.Series(dtype="float64"),
        "bayes_actual_side_probability": pd.Series(dtype="float64"),
        "bayes_actual_side_prior_probability": pd.Series(dtype="float64"),
        "bayes_probability_edge_long_minus_short": pd.Series(dtype="float64"),
        "bayes_expected_net_r_long": pd.Series(dtype="float64"),
        "bayes_expected_net_r_short": pd.Series(dtype="float64"),
        "bayes_long_context_samples": pd.Series(dtype="int64"),
        "bayes_short_context_samples": pd.Series(dtype="int64"),
        "bayes_long_side_samples": pd.Series(dtype="int64"),
        "bayes_short_side_samples": pd.Series(dtype="int64"),
        "bayes_actual_context_samples": pd.Series(dtype="int64"),
        "bayes_actual_side_confidence": pd.Series(dtype="string"),
        "bayes_long_lower_90": pd.Series(dtype="float64"),
        "bayes_long_upper_90": pd.Series(dtype="float64"),
        "bayes_short_lower_90": pd.Series(dtype="float64"),
        "bayes_short_upper_90": pd.Series(dtype="float64"),
        "bayes_long_evidence_families": pd.Series(dtype="int64"),
        "bayes_short_evidence_families": pd.Series(dtype="int64"),
        "bayes_actual_evidence_families": pd.Series(dtype="int64"),
        "bayes_actual_evidence_items": pd.Series(dtype="int64"),
        "bayes_top_positive_evidence": pd.Series(dtype="string"),
        "bayes_top_negative_evidence": pd.Series(dtype="string"),
        "bayes_context_ready": pd.Series(dtype="bool"),
        "bayes_recommended_direction": pd.Series(dtype="string"),
        "bayes_recommendation_matches_actual": pd.Series(dtype="bool"),
    }
    for family in EVIDENCE_FAMILIES:
        base_columns[f"bayes_actual_{family}_log_odds_lift"] = pd.Series(dtype="float64")
    threshold_columns = {
        _threshold_column(value, min_context_samples): pd.Series(dtype="bool")
        for value in thresholds
    }
    if trades.empty:
        for name, values in {**base_columns, **threshold_columns}.items():
            trades[name] = values
        trades.attrs["bayesian_threshold_simulation"] = []
        return trades

    _required_columns(trades)
    entry_name = _entry_column(trades)
    entries = pd.to_datetime(trades[entry_name], utc=True, errors="coerce")
    exits = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    if entries.isna().any():
        raise ValueError("Bayesian research found a trade with an invalid entry timestamp")
    if exits.isna().any():
        raise ValueError("Bayesian research found a trade with an invalid exit timestamp")
    if (exits < entries).any():
        raise ValueError("Bayesian research found an exit timestamp before entry")

    model = BayesianTradeModel(prior_strength=prior_strength)
    pending: list[tuple[int, int, _PendingObservation]] = []
    order = np.argsort(entries.astype("int64").to_numpy(), kind="stable")
    scored: dict[int, dict[str, object]] = {}

    for sequence, position in enumerate(order):
        entry_ns = int(entries.iloc[position].value)
        while pending and pending[0][0] <= entry_ns:
            _exit_ns, _serial, completed = heapq.heappop(pending)
            model.observe_tokens(
                completed.side,
                completed.won,
                completed.net_r,
                completed.tokens,
            )

        row = trades.iloc[position]
        long_estimate = model.estimate(row, "LONG")
        short_estimate = model.estimate(row, "SHORT")
        actual_side = str(row.get("side", "")).upper()
        if actual_side not in {"LONG", "SHORT"}:
            raise ValueError(f"Bayesian research found unsupported trade side: {actual_side}")
        actual = long_estimate if actual_side == "LONG" else short_estimate
        ready = (
            actual.context_samples >= min_context_samples
            and actual.evidence_families >= min_evidence_families
        )

        long_ready = (
            long_estimate.context_samples >= min_context_samples
            and long_estimate.evidence_families >= min_evidence_families
        )
        short_ready = (
            short_estimate.context_samples >= min_context_samples
            and short_estimate.evidence_families >= min_evidence_families
        )
        if (
            long_ready
            and short_ready
            and abs(long_estimate.probability - short_estimate.probability) >= 0.05
            and max(long_estimate.probability, short_estimate.probability) >= 0.55
        ):
            recommendation = (
                "LONG"
                if long_estimate.probability > short_estimate.probability
                else "SHORT"
            )
        else:
            recommendation = "ABSTAIN"

        actual_lifts = _lift_map(actual)
        result: dict[str, object] = {
            "bayes_model_version": MODEL_VERSION,
            "bayes_long_probability": long_estimate.probability,
            "bayes_short_probability": short_estimate.probability,
            "bayes_actual_side_probability": actual.probability,
            "bayes_actual_side_prior_probability": actual.prior_probability,
            "bayes_probability_edge_long_minus_short": (
                long_estimate.probability - short_estimate.probability
            ),
            "bayes_expected_net_r_long": long_estimate.expected_net_r,
            "bayes_expected_net_r_short": short_estimate.expected_net_r,
            "bayes_long_context_samples": long_estimate.context_samples,
            "bayes_short_context_samples": short_estimate.context_samples,
            "bayes_long_side_samples": long_estimate.side_samples,
            "bayes_short_side_samples": short_estimate.side_samples,
            "bayes_actual_context_samples": actual.context_samples,
            "bayes_actual_side_confidence": actual.confidence,
            "bayes_long_lower_90": long_estimate.lower_90,
            "bayes_long_upper_90": long_estimate.upper_90,
            "bayes_short_lower_90": short_estimate.lower_90,
            "bayes_short_upper_90": short_estimate.upper_90,
            "bayes_long_evidence_families": long_estimate.evidence_families,
            "bayes_short_evidence_families": short_estimate.evidence_families,
            "bayes_actual_evidence_families": actual.evidence_families,
            "bayes_actual_evidence_items": actual.evidence_items,
            "bayes_top_positive_evidence": _top_evidence(actual, True),
            "bayes_top_negative_evidence": _top_evidence(actual, False),
            "bayes_context_ready": bool(ready),
            "bayes_recommended_direction": recommendation,
            "bayes_recommendation_matches_actual": (
                recommendation == actual_side if recommendation != "ABSTAIN" else False
            ),
        }
        for family in EVIDENCE_FAMILIES:
            result[f"bayes_actual_{family}_log_odds_lift"] = actual_lifts.get(family, 0.0)
        for threshold in thresholds:
            result[_threshold_column(threshold, min_context_samples)] = bool(
                ready and actual.probability >= threshold
            )
        scored[int(position)] = result

        pnl = _numeric(row.get("pair_net_pnl"))
        net_r = _numeric(row.get("pair_net_r"))
        observation = _PendingObservation(
            side=actual_side,
            won=bool(np.isfinite(pnl) and pnl > 0),
            net_r=net_r,
            tokens=_evidence_tokens(row, actual_side),
        )
        heapq.heappush(
            pending,
            (int(exits.iloc[position].value), sequence, observation),
        )

    names = list(next(iter(scored.values())))
    for name in names:
        trades[name] = [scored[position][name] for position in range(len(trades))]

    simulation = threshold_simulation(
        trades,
        thresholds=thresholds,
        min_context_samples=min_context_samples,
    )
    trades.attrs["bayesian_threshold_simulation"] = simulation.to_dict("records")
    return trades


def threshold_simulation(
    scored_trades: pd.DataFrame,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    min_context_samples: int = DEFAULT_MIN_CONTEXT_SAMPLES,
) -> pd.DataFrame:
    """Summarize take/skip results for causal posterior-probability thresholds."""
    thresholds = tuple(float(value) for value in thresholds)
    rows: list[dict[str, object]] = []
    total = len(scored_trades)
    for threshold in thresholds:
        name = _threshold_column(threshold, min_context_samples)
        if name not in scored_trades.columns:
            probability = pd.to_numeric(
                scored_trades.get(
                    "bayes_actual_side_probability",
                    pd.Series(index=scored_trades.index, dtype=float),
                ),
                errors="coerce",
            )
            samples = pd.to_numeric(
                scored_trades.get(
                    "bayes_actual_context_samples",
                    pd.Series(index=scored_trades.index, dtype=float),
                ),
                errors="coerce",
            )
            families = pd.to_numeric(
                scored_trades.get(
                    "bayes_actual_evidence_families",
                    pd.Series(
                        DEFAULT_MIN_EVIDENCE_FAMILIES,
                        index=scored_trades.index,
                        dtype=float,
                    ),
                ),
                errors="coerce",
            )
            mask = (
                (probability >= threshold)
                & (samples >= min_context_samples)
                & (families >= DEFAULT_MIN_EVIDENCE_FAMILIES)
            )
        else:
            mask = scored_trades[name].fillna(False).astype(bool)
        selected = scored_trades.loc[mask]
        pnl = pd.to_numeric(
            selected.get("pair_net_pnl", pd.Series(dtype=float)),
            errors="coerce",
        )
        net_r = pd.to_numeric(
            selected.get("pair_net_r", pd.Series(dtype=float)),
            errors="coerce",
        )
        rows.append(
            {
                "minimum_probability": threshold,
                "minimum_context_samples": int(min_context_samples),
                "trades": int(len(selected)),
                "retained_percentage": float(len(selected) / total) if total else 0.0,
                "wins": int((pnl > 0).sum()),
                "losses": int((pnl < 0).sum()),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
                "total_net_r": float(net_r.sum()) if len(net_r) else 0.0,
                "average_net_r": float(net_r.mean()) if len(net_r) else 0.0,
                "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
            }
        )
    return pd.DataFrame(rows)
