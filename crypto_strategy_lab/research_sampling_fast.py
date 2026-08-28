"""Research-only batching for independent fixed-exit observations.

The portfolio simulator remains the semantic authority.  This mixin accelerates
only the strategy-research population when an active Position has no stateful
exit management.  For those simple fixed SL/TP observations, every overlapping
trade scans the same intrabar interval, so the interval can be sliced once and
first-hit candidates can be found in NumPy batches.  The actual winning event is
still executed by the mature timeout/exit helpers.

Anything stateful or unusual falls back to the normal Data Lake execution path:
break-even, trailing, partial exits, R-step trailing, ATR checkpoint extension,
missing/incomplete intrabars, alignment anomalies, and non-array windows.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from crypto_strategy_lab.trade import ExitSource, Side


class ResearchSamplingFastExitMixin:
    """Semantics-preserving batching used only by research sampling."""

    research_enable_batched_simple_exits = True
    research_simple_exit_batch_size = 2048

    @staticmethod
    def _research_simple_exit_eligible(position) -> bool:
        """Return True only when a position is a stateless fixed SL/TP trade."""
        return bool(
            position.is_open
            and not position.partial_sl_enabled
            and not position.partial_tp_enabled
            and not position.r_step_trailing_enabled
            and not position.atr_checkpoint_extension_enabled
            and not position.trailing_enabled
            and not position.be_enabled
            and position.profile_break_even_activation_r is None
            and position.be_active_after is None
        )

    def _research_bump_exit_stat(self, name: str, amount: int = 1) -> None:
        setattr(self, name, int(getattr(self, name, 0)) + int(amount))

    def research_exit_optimization_stats(self) -> dict[str, int | str | bool]:
        """Expose lightweight counters so long research runs are auditable."""
        return {
            "research_exit_kernel": "BATCHED_SIMPLE_INTRABAR_V1",
            "research_batched_simple_exit_enabled": bool(
                self.research_enable_batched_simple_exits
            ),
            "research_batched_simple_intervals": int(
                getattr(self, "research_batched_simple_intervals", 0)
            ),
            "research_batched_simple_position_intervals": int(
                getattr(self, "research_batched_simple_position_intervals", 0)
            ),
            "research_dynamic_exit_position_intervals": int(
                getattr(self, "research_dynamic_exit_position_intervals", 0)
            ),
            "research_batch_fallback_position_intervals": int(
                getattr(self, "research_batch_fallback_position_intervals", 0)
            ),
        }

    @staticmethod
    def _research_naive_ns(value) -> np.datetime64:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return np.datetime64(timestamp.to_datetime64(), "ns")

    @staticmethod
    def _research_window_arrays(window):
        """Return zero/low-copy arrays for either supported fast-window contract."""
        data = getattr(window, "data", None)
        start = getattr(window, "start", None)
        stop = getattr(window, "stop", None)
        if data is not None and start is not None and stop is not None:
            left = int(start)
            right = int(stop)
            return (
                left,
                np.asarray(data.timestamp[left:right], dtype="datetime64[ns]"),
                np.asarray(data.open[left:right], dtype=float),
                np.asarray(data.high[left:right], dtype=float),
                np.asarray(data.low[left:right], dtype=float),
            )

        left = getattr(window, "left", None)
        right = getattr(window, "right", None)
        values = getattr(window, "timestamp_values", None)
        unit = getattr(window, "timestamp_unit", None)
        opens = getattr(window, "opens", None)
        highs = getattr(window, "highs", None)
        lows = getattr(window, "lows", None)
        if any(
            value is None
            for value in (left, right, values, unit, opens, highs, lows)
        ):
            return None
        left = int(left)
        right = int(right)
        unit_ns = int(pd.Timedelta(1, unit=str(unit)).value)
        timestamps_ns = (
            np.asarray(values[left:right], dtype=np.int64) * unit_ns
        ).astype("datetime64[ns]")
        return (
            left,
            timestamps_ns,
            np.asarray(opens[left:right], dtype=float),
            np.asarray(highs[left:right], dtype=float),
            np.asarray(lows[left:right], dtype=float),
        )

    def _research_intrabar_max_timestamp(self):
        maximum = getattr(self.intrabar_data, "max_timestamp", None)
        if maximum is not None:
            return pd.Timestamp(maximum)
        timestamp_values = getattr(self.intrabar_data, "timestamp", None)
        max_method = getattr(timestamp_values, "max", None)
        if callable(max_method):
            return pd.Timestamp(max_method())
        return None

    def _research_complete_window(self, start, end):
        """Return batchable arrays only when the mature path sees a complete window."""
        minutes = int(self.config.intrabar_timeframe_minutes)
        if start.floor(f"{minutes}min") != start:
            return None
        fast_window = getattr(self.intrabar_data, "fast_window", None)
        if not callable(fast_window):
            return None
        window = fast_window(start, end)
        if window is None:
            return None

        expected = pd.Timedelta(minutes=minutes)
        gaps = window.gap_pairs(expected)
        incomplete = (
            window.empty
            or window.first_timestamp > start + expected
            or bool(gaps)
        )
        if incomplete:
            return None

        intrabar_max = self._research_intrabar_max_timestamp()
        if intrabar_max is not None and intrabar_max < end - expected:
            return None
        return self._research_window_arrays(window)

    def _research_scan_simple_group(self, pairs, i: int, start: pd.Timestamp) -> bool:
        """Batch one shared strategy interval and execute only each first event."""
        end = pd.Timestamp(self.times[i]) + self.entry_delta
        if start >= end:
            return False
        arrays = self._research_complete_window(start, end)
        if arrays is None:
            return False

        global_start, timestamps, opens, highs, lows = arrays
        row_count = len(timestamps)
        if row_count == 0:
            return False

        batch_size = max(1, int(self.research_simple_exit_batch_size))
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            positions = [position for _, position in batch]
            long_mask = np.fromiter(
                (position.side == Side.LONG for position in positions),
                dtype=bool,
                count=len(positions),
            )
            targets = np.fromiter(
                (float(position.tp) for position in positions),
                dtype=float,
                count=len(positions),
            )
            stops = np.fromiter(
                (float(position.sl) for position in positions),
                dtype=float,
                count=len(positions),
            )

            tp_hits = np.where(
                long_mask[:, None],
                highs[None, :] >= targets[:, None],
                lows[None, :] <= targets[:, None],
            )
            sl_hits = np.where(
                long_mask[:, None],
                lows[None, :] <= stops[:, None],
                highs[None, :] >= stops[:, None],
            )
            any_hit = tp_hits | sl_hits
            has_hit = any_hit.any(axis=1)
            first_hit = np.full(len(batch), row_count, dtype=np.int64)
            if has_hit.any():
                first_hit[has_hit] = np.argmax(any_hit[has_hit], axis=1)

            timeout_index = np.full(len(batch), row_count, dtype=np.int64)
            for local_index, (pair, position) in enumerate(batch):
                if not getattr(pair, "profile_timeout_enabled", False):
                    continue
                minutes = getattr(pair, "profile_timeout_minutes", None)
                if minutes is None:
                    continue
                timeout_at = pd.Timestamp(position.entry_time) + pd.Timedelta(
                    minutes=int(minutes)
                )
                timeout_index[local_index] = int(
                    np.searchsorted(
                        timestamps,
                        self._research_naive_ns(timeout_at),
                        side="left",
                    )
                )

            event_index = np.minimum(first_hit, timeout_index)
            for local_index, (pair, position) in enumerate(batch):
                event = int(event_index[local_index])
                if event >= row_count:
                    continue
                absolute_index = global_start + event
                timestamp = pd.Timestamp(timestamps[event])
                # The mature scanner checks timeout before TP/SL on each intrabar.
                if timeout_index[local_index] <= first_hit[local_index]:
                    self._maybe_timeout_position_at(
                        pair,
                        position,
                        absolute_index,
                        timestamp,
                        float(opens[event]),
                        ExitSource.INTRABAR,
                    )
                else:
                    # Preserve all same-bar ambiguity, slippage, fee and close logic
                    # by delegating the actual event to the existing helper.
                    self._maybe_exit_bar(
                        position,
                        absolute_index,
                        float(highs[event]),
                        float(lows[event]),
                        timestamp,
                        ExitSource.INTRABAR,
                    )
        return True

    def _update_positions_to_strategy_index(self, i):
        """Batch stateless research exits; delegate every other case unchanged."""
        if not self.research_enable_batched_simple_exits:
            return super()._update_positions_to_strategy_index(i)
        if not self.config.use_intrabar_data or self.intrabar_data is None:
            return super()._update_positions_to_strategy_index(i)
        if not callable(getattr(self.intrabar_data, "fast_window", None)):
            return super()._update_positions_to_strategy_index(i)

        strategy_time = pd.Timestamp(self.times[i])
        simple_groups = defaultdict(list)
        dynamic = []
        for pair in self.active_pairs:
            position = pair.position
            if i <= position.entry_index:
                continue
            if self._research_simple_exit_eligible(position):
                start = max(pd.Timestamp(position.entry_time), strategy_time)
                simple_groups[start].append((pair, position))
            else:
                dynamic.append((pair, position))

        for start, group in simple_groups.items():
            if self._research_scan_simple_group(group, i, start):
                self._research_bump_exit_stat("research_batched_simple_intervals")
                self._research_bump_exit_stat(
                    "research_batched_simple_position_intervals", len(group)
                )
            else:
                self._research_bump_exit_stat(
                    "research_batch_fallback_position_intervals", len(group)
                )
                for pair, position in group:
                    super()._scan_position_exit(pair, position, i)

        for pair, position in dynamic:
            self._research_bump_exit_stat("research_dynamic_exit_position_intervals")
            super()._scan_position_exit(pair, position, i)
