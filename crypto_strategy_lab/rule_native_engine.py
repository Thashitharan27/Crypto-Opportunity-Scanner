"""Rule-aware native Data Lake simulator extensions.

DI pressure, support/resistance, and lightweight futures research remain causal
prepared features. This layer only exposes those already-prepared values to the
generic Entry/Veto rule contract and neutralizes retired independent filtering
paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_strategy_lab.data_lake_production_engine import (
    DataLakeProductionBacktestEngine,
)
from crypto_strategy_lab.strategy_rule_model import CATEGORICAL_VALUE_CODES


DI_PRESSURE_STATE_CODES = CATEGORICAL_VALUE_CODES["DI_PRESSURE_STATE"]

_SR_CATEGORICAL_FIELDS = {
    "SR_NEAR_SUPPORT": "near_support",
    "SR_NEAR_RESISTANCE": "near_resistance",
    "SR_INSIDE_SUPPORT_ZONE": "inside_support_zone",
    "SR_INSIDE_RESISTANCE_ZONE": "inside_resistance_zone",
    "SR_SUPPORT_STATE": "support_state",
    "SR_RESISTANCE_STATE": "resistance_state",
    "SR_SUPPORT_HELD": "support_held",
    "SR_RESISTANCE_HELD": "resistance_held",
    "SR_TRADE_LOCATION_RATING": "trade_location_rating",
}
_SR_NUMERIC_FIELDS = {
    "SR_ROOM_IN_DIRECTION_ATR": "room_in_direction_atr",
    "SR_SUPPORT_DISTANCE_ATR": "nearest_support_distance_atr",
    "SR_RESISTANCE_DISTANCE_ATR": "nearest_resistance_distance_atr",
    "SR_SUPPORT_REJECTION_ATR": "support_rejection_atr",
    "SR_RESISTANCE_REJECTION_ATR": "resistance_rejection_atr",
}
_SR_RULE_INDICATORS = frozenset((*_SR_CATEGORICAL_FIELDS, *_SR_NUMERIC_FIELDS))

# These are lightweight source-native research blocks already prepared by the
# Data Lake service whenever validated local coverage exists. Values remain in
# their report/trade-list units so a rule threshold means exactly what the
# corresponding research column means.
_RESEARCH_NUMERIC_FIELDS = {
    "OI_CHANGE_PCT_5M": ("futures_positioning", "oi_change_pct_5m", 1.0),
    "OI_CHANGE_PCT_1H": ("futures_positioning", "oi_change_pct_1h", 1.0),
    "OI_CHANGE_PCT_24H": ("futures_positioning", "oi_change_pct_24h", 1.0),
    "OI_ZSCORE_7D": ("futures_positioning", "oi_zscore_7d", 1.0),
    "PRICE_CHANGE_PCT_1H": ("futures_positioning", "price_change_pct_1h", 1.0),
    "TOP_TRADER_ACCOUNT_BIAS": (
        "futures_positioning",
        "top_trader_account_bias",
        1.0,
    ),
    "TOP_TRADER_POSITION_BIAS": (
        "futures_positioning",
        "top_trader_position_bias",
        1.0,
    ),
    "GLOBAL_LONG_SHORT_ACCOUNT_BIAS": (
        "futures_positioning",
        "global_long_short_account_bias",
        1.0,
    ),
    "TAKER_LONG_SHORT_VOLUME_BIAS": (
        "futures_positioning",
        "taker_long_short_volume_bias",
        1.0,
    ),
    "FUNDING_RATE_BPS": ("funding_context", "funding_rate_bps", 1.0),
    "FUNDING_24H_SUM_BPS": ("funding_context", "funding_24h_sum_bps", 1.0),
    "FUNDING_CHANGE_BPS": ("funding_context", "funding_change", 10000.0),
    "FUNDING_3_EVENT_MEAN_BPS": (
        "funding_context",
        "funding_3_event_mean",
        10000.0,
    ),
    "FUNDING_ZSCORE_7D": ("funding_context", "funding_7d_zscore", 1.0),
    "MARK_INDEX_BASIS_BPS": ("basis_context", "mark_index_basis_bps", 1.0),
    "MARK_INDEX_BASIS_ZSCORE_7D": (
        "basis_context",
        "mark_index_basis_zscore_7d",
        1.0,
    ),
    "TRADE_MARK_BASIS_BPS": ("basis_context", "trade_mark_basis_bps", 1.0),
    "TRADE_INDEX_BASIS_BPS": ("basis_context", "trade_index_basis_bps", 1.0),
    "PREMIUM_INDEX_ZSCORE_7D": (
        "basis_context",
        "premium_index_zscore_7d",
        1.0,
    ),
    "TAKER_BUY_SELL_RATIO": (
        "taker_flow_context",
        "taker_buy_sell_ratio",
        1.0,
    ),
    "TAKER_DELTA_PCT": ("taker_flow_context", "taker_delta_pct", 1.0),
    "TAKER_DELTA_PCT_15M": (
        "taker_flow_context",
        "taker_delta_pct_15m",
        1.0,
    ),
    "TAKER_DELTA_PCT_1H": (
        "taker_flow_context",
        "taker_delta_pct_1h",
        1.0,
    ),
    "TAKER_FLOW_PERSISTENCE": (
        "taker_flow_context",
        "flow_persistence",
        1.0,
    ),
}
_RESEARCH_CATEGORICAL_FIELDS = {
    "OI_VS_PRICE_STATE_1H": ("futures_positioning", "oi_vs_price_state_1h"),
    "FUNDING_BIAS": ("funding_context", "funding_bias"),
    "FUNDING_EXTREME_POSITIVE": ("funding_context", "funding_extreme_positive"),
    "FUNDING_EXTREME_NEGATIVE": ("funding_context", "funding_extreme_negative"),
    "MARK_INDEX_BASIS_STATE": ("basis_context", "mark_index_basis_state"),
}
_RESEARCH_RULE_INDICATORS = frozenset(
    (*_RESEARCH_NUMERIC_FIELDS, *_RESEARCH_CATEGORICAL_FIELDS)
)


@dataclass(frozen=True, slots=True)
class NativeLatestEntryDecision:
    """Paper-safe view of the native Entry/Veto decision at one prepared row."""

    accepted: bool
    strategy_profile_key: str | None
    side: str | None
    reference_price: float
    detail: str


class RuleAwareDataLakeProductionBacktestEngine(DataLakeProductionBacktestEngine):
    """Current native runtime with prepared research evidence available to rules."""

    def evaluate_prepared_entry(self, index: int) -> NativeLatestEntryDecision:
        """Evaluate Entry/Veto semantics at one already-causal prepared row.

        This deliberately does not open a position, apply portfolio overlap
        behavior, or simulate an exit.  Callers own causal row selection.
        """
        if index < 0 or index >= len(self.close):
            raise IndexError("prepared strategy row is out of range")
        accepted, detail = self._strategy_profile_filter_result(index)
        context = self._profile_context(index)
        side = None
        if context is not None:
            _, side = self._resolved_profile_direction(index, context)
        return NativeLatestEntryDecision(
            accepted=bool(accepted),
            strategy_profile_key=context[2] if context else None,
            side=side,
            reference_price=float(self.close[index]),
            detail=str(detail),
        )

    @classmethod
    def from_prepared(cls, *args, **kwargs):
        """Keep the full prepared S/R snapshot available to Bayesian reporting.

        The production engine consumes the support/resistance block through its
        dedicated O(1) context reader and historically excluded it from generic
        research output attachment. Bayesian direction research benefits from
        having both LONG and SHORT causal S/R snapshots on the completed row, so
        expose the same already-prepared block for reporting without changing the
        S/R reader or any entry/exit decision.
        """
        engine = super().from_prepared(*args, **kwargs)
        prepared = args[0] if args else kwargs.get("prepared")
        if prepared is None:
            return engine
        sr_block = next(
            (
                block
                for block in prepared.research
                if block.name == "support_resistance"
            ),
            None,
        )
        if sr_block is None:
            return engine
        engine.research_features["support_resistance"] = sr_block
        existing = set(engine.research_output_columns)
        sr_columns = tuple(
            column for column in sr_block.values if column not in existing
        )
        engine.research_output_columns = (*engine.research_output_columns, *sr_columns)
        available_name = "support_resistance_feature_available_at"
        if available_name not in engine.research_feature_available_columns:
            engine.research_feature_available_columns = (
                *engine.research_feature_available_columns,
                available_name,
            )
        return engine

    def _build_result_row(self, p, row_kind, positions):
        """Publish stable aliases for entry-time fields consumed by research."""
        row = super()._build_result_row(p, row_kind, positions)
        aliases = {
            "atr_pct": "entry_atr_pct",
            "rsi": "entry_rsi",
            "momentum": "directional_momentum_return_at_entry",
            "bb_width_change": "bb_width_entry_5bar_change",
            "bb_width_change_pct": "bb_width_entry_5bar_change_pct",
            "mean_reversion_distance_atr": "mean_distance_atr",
        }
        for alias, source in aliases.items():
            if alias not in row and source in row:
                row[alias] = row[source]
        if "funding_change_bps" not in row:
            try:
                funding_change = float(row.get("funding_change"))
            except (TypeError, ValueError):
                funding_change = np.nan
            if np.isfinite(funding_change):
                row["funding_change_bps"] = funding_change * 10000.0
        return row

    def _di_pressure_filter_result(self, i):
        """Retired global pressure filter: Entry/Veto rules are authoritative."""
        del i
        return True, None

    def _should_reject_for_sr(self, i, direction, sr_context=None):
        """Retired S/R preset filter: Entry/Veto rules are authoritative."""
        del i, direction, sr_context
        return False, None

    def _prepared_pressure_value(self, i, direction, indicator):
        """Read directional/pressure rule evidence from already-prepared arrays."""
        if indicator == "DIRECTIONAL_DI":
            if direction == "LONG":
                value = float(self.plus_di_values[i])
            elif direction == "SHORT":
                value = float(self.minus_di_values[i])
            else:
                return np.nan
            return value if np.isfinite(value) else np.nan

        if indicator == "DI_SPREAD_CHANGE":
            value = float(self.di_pressure_spread_change[i])
            return value if np.isfinite(value) else np.nan

        if direction == "LONG":
            directional = self.long_directional_di_change
            opposing = self.long_opposing_di_change
            states = self.long_di_pressure_state
        elif direction == "SHORT":
            directional = self.short_directional_di_change
            opposing = self.short_opposing_di_change
            states = self.short_di_pressure_state
        else:
            return np.nan

        if indicator == "DIRECTIONAL_DI_CHANGE":
            value = float(directional[i])
            return value if np.isfinite(value) else np.nan
        if indicator == "OPPOSING_DI_CHANGE":
            value = float(opposing[i])
            return value if np.isfinite(value) else np.nan
        if indicator == "DI_PRESSURE_STATE":
            return DI_PRESSURE_STATE_CODES.get(str(states[i]).upper(), np.nan)
        raise KeyError(indicator)

    def _prepared_sr_context(self, i, direction):
        """Return one O(1) prepared S/R context and reuse it for all rules at the row."""
        if direction not in {"LONG", "SHORT"}:
            return None
        if not getattr(
            getattr(self, "config", None), "enable_support_resistance_analysis", False
        ):
            return None
        pending = getattr(self, "_pending_sr_context", None)
        if pending is not None and pending[0] == i and pending[1] == direction:
            return pending[2]
        context = self._analyze_support_resistance(i, direction)
        self._pending_sr_context = (i, direction, context)
        return context

    @staticmethod
    def _categorical_sr_value(indicator, raw):
        if hasattr(raw, "value"):
            raw = raw.value
        if isinstance(raw, (bool, np.bool_)):
            key = "TRUE" if bool(raw) else "FALSE"
        else:
            key = str(raw).upper()
        return CATEGORICAL_VALUE_CODES[indicator].get(key, np.nan)

    def _prepared_sr_value(self, i, direction, indicator):
        context = self._prepared_sr_context(i, direction)
        if context is None:
            return np.nan
        if indicator in _SR_CATEGORICAL_FIELDS:
            raw = getattr(context, _SR_CATEGORICAL_FIELDS[indicator])
            return self._categorical_sr_value(indicator, raw)
        if indicator in _SR_NUMERIC_FIELDS:
            raw = getattr(context, _SR_NUMERIC_FIELDS[indicator])
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return np.nan
            return value if np.isfinite(value) else np.nan
        raise KeyError(indicator)

    def _prepared_research_raw_value(self, i, feature_name, column):
        """Read one aligned research fact from DataFrame or native prepared block."""
        block = getattr(self, "research_features", {}).get(feature_name)
        if block is None or i < 0:
            return None
        if isinstance(block, pd.DataFrame):
            if i >= len(block) or column not in block.columns:
                return None
            return block.iloc[i][column]
        values = getattr(block, "values", None)
        if values is None or column not in values:
            return None
        series = values[column]
        if i >= len(series):
            return None
        return series[i]

    def _prepared_research_value(self, i, indicator):
        if indicator in _RESEARCH_NUMERIC_FIELDS:
            feature_name, column, scale = _RESEARCH_NUMERIC_FIELDS[indicator]
            raw = self._prepared_research_raw_value(i, feature_name, column)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return np.nan
            value *= scale
            return value if np.isfinite(value) else np.nan
        if indicator in _RESEARCH_CATEGORICAL_FIELDS:
            feature_name, column = _RESEARCH_CATEGORICAL_FIELDS[indicator]
            raw = self._prepared_research_raw_value(i, feature_name, column)
            if raw is None or raw is pd.NA:
                return np.nan
            if hasattr(raw, "value"):
                raw = raw.value
            if isinstance(raw, (bool, np.bool_)):
                key = "TRUE" if bool(raw) else "FALSE"
            else:
                try:
                    if bool(pd.isna(raw)):
                        return np.nan
                except (TypeError, ValueError):
                    return np.nan
                key = str(raw).upper()
            return CATEGORICAL_VALUE_CODES[indicator].get(key, np.nan)
        raise KeyError(indicator)

    def _strategy_profile_rule_value(self, i, direction, profile, indicator):
        if indicator == "ADX_CHANGE":
            if i <= 0 or not hasattr(self, "adx_values"):
                return np.nan
            current = float(self.adx_values[i])
            previous = float(self.adx_values[i - 1])
            if not np.isfinite(current) or not np.isfinite(previous):
                return np.nan
            return current - previous
        if indicator == "DIRECTIONAL_DI":
            return self._prepared_pressure_value(i, direction, indicator)
        if indicator in {
            "DI_PRESSURE_STATE",
            "DI_SPREAD_CHANGE",
            "DIRECTIONAL_DI_CHANGE",
            "OPPOSING_DI_CHANGE",
        } and hasattr(self, "di_pressure_spread_change"):
            return self._prepared_pressure_value(i, direction, indicator)
        if indicator in _SR_RULE_INDICATORS:
            return self._prepared_sr_value(i, direction, indicator)
        if indicator in _RESEARCH_RULE_INDICATORS:
            return self._prepared_research_value(i, indicator)
        return super()._strategy_profile_rule_value(i, direction, profile, indicator)

    def _strategy_profile_entry_rule_matches(self, i, direction, profile, rule):
        """Treat missing required research as a failed requirement, never a pass.

        Builder REQUIRED rules are compiled into REJECT rules that match the
        inverse of the desired condition. Returning ``True`` for missing evidence
        therefore rejects the candidate. VETO/FLIP rules keep fail-open matching:
        absent optional evidence cannot manufacture a veto or direction flip.
        """
        value = self._strategy_profile_rule_value(
            i, direction, profile, rule["indicator"]
        )
        if not np.isfinite(value):
            return str(rule.get("_builder_kind", "")).upper() == "REQUIRED"
        inside = float(rule["minimum"]) <= value <= float(rule["maximum"])
        return inside if rule.get("condition", "INSIDE") == "INSIDE" else not inside
