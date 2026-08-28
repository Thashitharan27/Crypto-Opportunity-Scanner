"""Optional support/resistance-aware take-profit logic.

This layer keeps the existing fixed-R strategy as the default and only changes
entries/targets when ``sr_take_profit_mode`` is explicitly set to
``SR_CAPPED_R``.
"""
from __future__ import annotations

import numpy as np

from crypto_strategy_lab.enhanced_engine import EnhancedBacktestEngine
from crypto_strategy_lab.trade import Side


class SRDynamicTPBacktestEngine(EnhancedBacktestEngine):
    """Cap a trade's final TP at the next opposing S/R level minus an R buffer."""

    def _effective_trade_direction(self, i: int) -> str | None:
        context = self._profile_context(i)
        if context is None:
            return None
        _regime, direction, _key, profile = context
        profile_filter_flip = bool(
            profile.entry_rules
            and self._strategy_profile_rule_group_match(
                i, direction, profile, "FLIP", profile.flip_rule_match_mode
            )
        )
        if profile.flip_direction or profile_filter_flip:
            return "SHORT" if direction == "LONG" else "LONG"
        return direction

    @staticmethod
    def _opposing_level_price(sr_context, direction: str):
        if sr_context is None:
            return None
        return (
            sr_context.nearest_resistance_price
            if direction == "LONG"
            else sr_context.nearest_support_price
        )

    def _expected_entry_price(self, indicator_i: int, execution_i: int | None, direction: str) -> float:
        if self.config.enable_daily_entry_schedule and execution_i is not None:
            raw = float(self.open[execution_i])
        else:
            raw = float(self.close[indicator_i])
        if direction == "LONG":
            return raw * (1.0 + self.config.slippage)
        return raw * (1.0 - self.config.slippage)

    def _sr_tp_filter_result(self, i: int, execution_i: int | None = None):
        if str(getattr(self.config, "sr_take_profit_mode", "FIXED_R")).upper() != "SR_CAPPED_R":
            return True, None

        context = self._profile_context(i)
        direction = self._effective_trade_direction(i)
        if context is None or direction not in ("LONG", "SHORT"):
            return True, None
        _regime, _original_direction, _key, profile = context

        sr_context = self._analyze_support_resistance(i, direction)
        level_price = self._opposing_level_price(sr_context, direction)
        if level_price is None or not np.isfinite(float(level_price)):
            policy = str(getattr(self.config, "sr_take_profit_no_level_policy", "USE_FIXED_TP")).upper()
            if policy == "REJECT_TRADE":
                return False, "SR_TP_NO_OPPOSING_LEVEL"
            return True, None

        r_unit = float(self.risk[i])
        stop_mult = float(profile.sl2_r if profile.partial_stop_enabled else profile.stop_loss_multiple)
        stop_distance = r_unit * stop_mult
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            return True, None

        entry = self._expected_entry_price(i, execution_i, direction)
        room_price = float(level_price) - entry if direction == "LONG" else entry - float(level_price)
        available_r = room_price / stop_distance
        buffer_r = float(getattr(self.config, "sr_take_profit_buffer_r", 0.20))
        maximum_r = float(getattr(self.config, "sr_take_profit_maximum_r", 3.0))
        minimum_r = float(getattr(self.config, "sr_take_profit_minimum_r", 1.5))
        target_r = min(maximum_r, available_r - buffer_r)
        if not np.isfinite(target_r) or target_r < minimum_r:
            return False, "SR_TP_INSUFFICIENT_ROOM"
        return True, None

    def _entry_filter_result(self, i, execution_i=None):
        passed, reason = super()._entry_filter_result(i, execution_i)
        if not passed:
            return passed, reason
        tp_passed, tp_reason = self._sr_tp_filter_result(i, execution_i)
        if not tp_passed:
            return False, tp_reason
        return True, reason

    def _apply_sr_capped_tp(self, indicator_i: int, pair) -> None:
        mode = str(getattr(self.config, "sr_take_profit_mode", "FIXED_R")).upper()
        positions = pair.positions()
        if not positions:
            return

        for pos in positions:
            direction = "LONG" if pos.side == Side.LONG else "SHORT"
            pos.sr_take_profit_mode = mode
            pos.sr_take_profit_applied = False
            pos.sr_take_profit_reason = "FIXED_R_BASELINE"
            pos.sr_take_profit_level_price = np.nan
            pos.sr_take_profit_available_r = np.nan
            pos.sr_take_profit_target_r = abs(float(pos.tp) - float(pos.entry_price)) / float(pos.risk)

            if mode != "SR_CAPPED_R":
                continue

            sr_context = self._analyze_support_resistance(indicator_i, direction)
            level_price = self._opposing_level_price(sr_context, direction)
            if level_price is None or not np.isfinite(float(level_price)):
                pos.sr_take_profit_reason = "NO_LEVEL_USE_FIXED_TP"
                continue

            risk_distance = float(pos.risk)
            if not np.isfinite(risk_distance) or risk_distance <= 0:
                pos.sr_take_profit_reason = "INVALID_RISK_USE_FIXED_TP"
                continue

            room_price = (
                float(level_price) - float(pos.entry_price)
                if direction == "LONG"
                else float(pos.entry_price) - float(level_price)
            )
            available_r = room_price / risk_distance
            buffer_r = float(getattr(self.config, "sr_take_profit_buffer_r", 0.20))
            maximum_r = float(getattr(self.config, "sr_take_profit_maximum_r", 3.0))
            minimum_r = float(getattr(self.config, "sr_take_profit_minimum_r", 1.5))
            existing_r = abs(float(pos.tp) - float(pos.entry_price)) / risk_distance
            target_r = min(existing_r, maximum_r, available_r - buffer_r)

            pos.sr_take_profit_level_price = float(level_price)
            pos.sr_take_profit_available_r = available_r
            pos.sr_take_profit_target_r = target_r
            if not np.isfinite(target_r) or target_r < minimum_r:
                # The pre-entry filter should normally catch this. If execution
                # slippage changes the exact R enough, preserve the original TP
                # rather than mutating an already-open trade to an invalid target.
                pos.sr_take_profit_reason = "POST_ENTRY_ROOM_BELOW_MIN_USE_FIXED_TP"
                continue

            side_sign = 1.0 if direction == "LONG" else -1.0
            pos.tp = float(pos.entry_price) + side_sign * target_r * risk_distance
            if getattr(pos, "partial_tp_enabled", False):
                pos.tp2_price = pos.tp
            pos.sr_take_profit_applied = target_r < existing_r - 1e-12
            pos.sr_take_profit_reason = "SR_CAPPED" if pos.sr_take_profit_applied else "FIXED_TARGET_INSIDE_SR_ROOM"

    def _open_pair(self, i, entry_filter_passed=True, entry_filter_reason="Strategy profile passed", schedule=None):
        before = len(self.active_pairs)
        result = super()._open_pair(i, entry_filter_passed, entry_filter_reason, schedule)
        if len(self.active_pairs) > before:
            indicator_i = schedule["indicator_index"] if schedule else i
            self._apply_sr_capped_tp(indicator_i, self.active_pairs[-1])
        return result

    def _build_result_row(self, p, row_kind, positions):
        row = super()._build_result_row(p, row_kind, positions)
        pos = positions[0] if positions else None
        row["sr_take_profit_mode"] = getattr(pos, "sr_take_profit_mode", str(getattr(self.config, "sr_take_profit_mode", "FIXED_R")))
        row["sr_take_profit_applied"] = bool(getattr(pos, "sr_take_profit_applied", False)) if pos is not None else False
        row["sr_take_profit_reason"] = getattr(pos, "sr_take_profit_reason", "") if pos is not None else ""
        row["sr_take_profit_level_price"] = getattr(pos, "sr_take_profit_level_price", np.nan) if pos is not None else np.nan
        row["sr_take_profit_available_r"] = getattr(pos, "sr_take_profit_available_r", np.nan) if pos is not None else np.nan
        row["sr_take_profit_target_r"] = getattr(pos, "sr_take_profit_target_r", np.nan) if pos is not None else np.nan
        row["sr_take_profit_maximum_r"] = float(getattr(self.config, "sr_take_profit_maximum_r", 3.0))
        row["sr_take_profit_minimum_r"] = float(getattr(self.config, "sr_take_profit_minimum_r", 1.5))
        row["sr_take_profit_buffer_r"] = float(getattr(self.config, "sr_take_profit_buffer_r", 0.20))
        row["sr_take_profit_no_level_policy"] = str(getattr(self.config, "sr_take_profit_no_level_policy", "USE_FIXED_TP"))
        return row
