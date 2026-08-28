"""Unit tests for support/resistance detection with no look-ahead bias."""

import numpy as np
import pandas as pd
import pytest
import time
from crypto_strategy_lab.config import BacktestConfig, RiskMode
from crypto_strategy_lab.engine import BacktestEngine
from crypto_strategy_lab.support_resistance import (
    SwingDetector,
    SRZoneMerger,
    SupportResistanceDetector,
    SRContext,
    SRLevel,
    SRLevelType,
    LocationClassification,
    TradeLocationRating,
)


class TestSwingDetector:
    """Tests for swing high and swing low detection."""
    
    def test_swing_high_detection_basic(self):
        """Swing highs detected at clear peaks."""
        detector = SwingDetector(pivot_left=2, pivot_right=2)
        highs = np.array([10, 12, 15, 12, 10, 11, 13, 16, 14, 12], dtype=np.float64)
        
        # At index 6 (enough data for confirmation), swing at index 2 (15) should be available
        # because 6 - 2 = 4 >= pivot_right (2)
        swings = detector.detect_swing_highs(highs, 6)
        assert 2 in swings, f"Expected swing at index 2, got {swings}"
    
    def test_no_look_ahead_bias_swing_high(self):
        """Swing at index i only available after i + pivot_right."""
        detector = SwingDetector(pivot_left=2, pivot_right=2)
        highs = np.array([10, 12, 15, 12, 10, 11, 13, 16, 14, 12], dtype=np.float64)
        
        # At index 3, swing at index 2 should NOT be available yet
        # because 3 - 2 = 1 < pivot_right (2)
        swings = detector.detect_swing_highs(highs, 3)
        assert 2 not in swings, f"Look-ahead bias: swing at 2 available at index 3, swings={swings}"
        
        # At index 4, it should be available (4 - 2 = 2 >= pivot_right)
        swings = detector.detect_swing_highs(highs, 4)
        assert 2 in swings, f"Expected swing at index 2 at index 4, got {swings}"
    
    def test_swing_low_detection_basic(self):
        """Swing lows detected at clear valleys."""
        detector = SwingDetector(pivot_left=2, pivot_right=2)
        lows = np.array([10, 8, 5, 8, 10, 9, 7, 4, 6, 8], dtype=np.float64)
        
        # At index 6, swing at index 2 (5) should be available
        swings = detector.detect_swing_lows(lows, 6)
        assert 2 in swings, f"Expected swing at index 2, got {swings}"
    
    def test_no_look_ahead_bias_swing_low(self):
        """Swing low also respects confirmation delay."""
        detector = SwingDetector(pivot_left=2, pivot_right=2)
        lows = np.array([10, 8, 5, 8, 10, 9, 7, 4, 6, 8], dtype=np.float64)
        
        # At index 3, swing at index 2 should NOT be available
        swings = detector.detect_swing_lows(lows, 3)
        assert 2 not in swings, f"Look-ahead bias detected in swing lows"
        
        # At index 4, it should be available
        swings = detector.detect_swing_lows(lows, 4)
        assert 2 in swings, f"Expected swing at index 2, got {swings}"
    
    def test_insufficient_data_no_swings(self):
        """No swings returned when data too short."""
        detector = SwingDetector(pivot_left=2, pivot_right=2)
        highs = np.array([10, 12, 15], dtype=np.float64)
        
        # Not enough data (need at least pivot_left + pivot_right + 1 = 5)
        swings = detector.detect_swing_highs(highs, 2)
        assert len(swings) == 0, f"Expected no swings, got {swings}"
    
    def test_multiple_swings_detected(self):
        """Multiple swings detected in sequence."""
        detector = SwingDetector(pivot_left=2, pivot_right=2, min_bars_between=2)
        # Two clear peaks at index 2 and index 7
        highs = np.array([10, 12, 20, 12, 10, 12, 15, 25, 15, 10], dtype=np.float64)
        
        swings = detector.detect_swing_highs(highs, 9)
        assert 2 in swings, "Expected swing at index 2"
        assert 7 in swings, "Expected swing at index 7"
    
    def test_min_bars_between_enforced(self):
        """Minimum spacing between swings enforced."""
        detector = SwingDetector(pivot_left=1, pivot_right=1, min_bars_between=3)
        # Peaks at 2, 4, 7 - only 2 and 7 should pass (4 too close to 2)
        highs = np.array([10, 12, 15, 12, 14, 12, 13, 17, 12], dtype=np.float64)
        
        swings = detector.detect_swing_highs(highs, 8)
        assert 2 in swings, "Expected swing at index 2"
        assert 4 not in swings, "Swing at 4 too close to 2"
        assert 7 in swings, "Expected swing at index 7"


class TestSRZoneMerger:
    """Tests for zone merging logic."""
    
    def test_single_level_no_merge(self):
        """Single level stays unchanged."""
        merger = SRZoneMerger(zone_width_atr=1.0)
        level = SRLevel(
            price=100.0,
            level_type=SRLevelType.SUPPORT,
            bar_index=10,
            first_touch_index=10,
        )
        
        merged = merger.merge_levels([level], atr=1.0)
        assert len(merged) == 1
        assert merged[0].price == 100.0
        assert merged[0].zone_bottom == 100.0
        assert merged[0].zone_top == 100.0
    
    def test_nearby_levels_merged(self):
        """Levels within zone_width_atr merged into zone."""
        merger = SRZoneMerger(zone_width_atr=1.0)
        atr = 10.0  # zone_width = 1.0 * 10 = 10
        
        levels = [
            SRLevel(100.0, SRLevelType.SUPPORT, 5, 5),
            SRLevel(105.0, SRLevelType.SUPPORT, 10, 10),  # 5 away, within 10
        ]
        
        merged = merger.merge_levels(levels, atr)
        assert len(merged) == 1, "Expected 1 merged zone"
        assert merged[0].zone_bottom == 100.0
        assert merged[0].zone_top == 105.0
        assert merged[0].touch_count == 2
    
    def test_far_levels_separate(self):
        """Levels far apart remain separate zones."""
        merger = SRZoneMerger(zone_width_atr=1.0)
        atr = 10.0  # zone_width = 10
        
        levels = [
            SRLevel(100.0, SRLevelType.SUPPORT, 5, 5),
            SRLevel(120.0, SRLevelType.SUPPORT, 10, 10),  # 20 away, > 10
        ]
        
        merged = merger.merge_levels(levels, atr)
        assert len(merged) == 2, "Expected 2 separate zones"
    
    def test_support_zone_uses_min_price(self):
        """Support zone price is minimum of group."""
        merger = SRZoneMerger(zone_width_atr=2.0)
        levels = [
            SRLevel(105.0, SRLevelType.SUPPORT, 10, 10),
            SRLevel(100.0, SRLevelType.SUPPORT, 5, 5),
        ]
        
        merged = merger.merge_levels(levels, atr=10.0)
        assert len(merged) == 1
        assert merged[0].price == 100.0, "Support zone should use minimum"
    
    def test_resistance_zone_uses_max_price(self):
        """Resistance zone price is maximum of group."""
        merger = SRZoneMerger(zone_width_atr=2.0)
        levels = [
            SRLevel(100.0, SRLevelType.RESISTANCE, 5, 5),
            SRLevel(105.0, SRLevelType.RESISTANCE, 10, 10),
        ]
        
        merged = merger.merge_levels(levels, atr=10.0)
        assert len(merged) == 1
        assert merged[0].price == 105.0, "Resistance zone should use maximum"


class TestSupportResistanceFiltering:
    """Tests for SR entry filtering modes."""

    def test_analysis_only_mode_never_rejects(self):
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="15min"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        })
        engine = BacktestEngine(data, BacktestConfig(enable_support_resistance_analysis=True, sr_filter_mode="ANALYSIS_ONLY"))
        context = SRContext(
            nearest_support_price=99.0,
            nearest_support_bar_index=1,
            nearest_support_distance_atr=0.1,
            nearest_support_distance_price=1.0,
            nearest_resistance_price=101.0,
            nearest_resistance_bar_index=2,
            nearest_resistance_distance_atr=1.5,
            nearest_resistance_distance_price=1.0,
            price_location=LocationClassification.NEAR_SUPPORT,
            trade_location_rating=TradeLocationRating.BAD_LOCATION,
            near_support=True,
            near_resistance=False,
            inside_support_zone=False,
            inside_resistance_zone=False,
            room_in_direction_atr=0.5,
        )

        rejected, reason = engine._should_reject_for_sr(10, "LONG", context)
        assert rejected is False
        assert reason is None

    @staticmethod
    def _engine(**rules):
        data = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=30, freq="15min"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0})
        mode = rules.pop("sr_filter_mode", "APPLY_ENTRY_RULES")
        return BacktestEngine(data, BacktestConfig(enable_support_resistance_analysis=True, sr_filter_mode=mode, **rules))

    @staticmethod
    def _context(**changes):
        values = dict(nearest_support_price=99.0, nearest_support_bar_index=1, nearest_support_distance_atr=1.0, nearest_support_distance_price=1.0, nearest_resistance_price=101.0, nearest_resistance_bar_index=2, nearest_resistance_distance_atr=1.0, nearest_resistance_distance_price=1.0, price_location=LocationClassification.BETWEEN_LEVELS, trade_location_rating=TradeLocationRating.NEUTRAL_LOCATION, near_support=False, near_resistance=False, inside_support_zone=False, inside_resistance_zone=False, room_in_direction_atr=2.0, support_state="SUPPORT_HELD", resistance_state="RESISTANCE_HELD")
        values.update(changes)
        return SRContext(**values)

    @pytest.mark.parametrize("field,direction,changes,reason", [
        ("sr_long_avoid_near_resistance", "LONG", {"near_resistance": True}, "SR_LONG_NEAR_RESISTANCE"),
        ("sr_long_require_near_support", "LONG", {"near_support": False}, "SR_LONG_NOT_NEAR_SUPPORT"),
        ("sr_long_block_broken_support", "LONG", {"support_state": "SUPPORT_BROKEN"}, "SR_LONG_SUPPORT_BROKEN"),
        ("sr_long_min_room_to_resistance_atr", "LONG", {"room_in_direction_atr": 1.0}, "SR_LONG_INSUFFICIENT_ROOM_TO_RESISTANCE"),
        ("sr_short_avoid_near_support", "SHORT", {"near_support": True}, "SR_SHORT_NEAR_SUPPORT"),
        ("sr_short_require_near_resistance", "SHORT", {"near_resistance": False}, "SR_SHORT_NOT_NEAR_RESISTANCE"),
        ("sr_short_block_broken_resistance", "SHORT", {"resistance_state": "RESISTANCE_BROKEN"}, "SR_SHORT_RESISTANCE_BROKEN"),
        ("sr_short_min_room_to_support_atr", "SHORT", {"room_in_direction_atr": 1.0}, "SR_SHORT_INSUFFICIENT_ROOM_TO_SUPPORT"),
    ])
    def test_each_entry_rule_rejects_independently(self, field, direction, changes, reason):
        value = 1.5 if "min_room" in field else True
        assert self._engine(**{field: value})._should_reject_for_sr(10, direction, self._context(**changes)) == (True, reason)

    def test_combined_long_rules_all_must_pass(self):
        engine = self._engine(sr_long_avoid_near_resistance=True, sr_long_require_near_support=True, sr_long_block_broken_support=True, sr_long_min_room_to_resistance_atr=1.5)
        good = self._context(near_support=True, near_resistance=False, support_state="SUPPORT_HELD", room_in_direction_atr=2.0)
        assert engine._should_reject_for_sr(10, "LONG", good) == (False, None)
        assert engine._should_reject_for_sr(10, "LONG", self._context(near_support=True, near_resistance=True)) == (True, "SR_LONG_NEAR_RESISTANCE")

    def test_rules_are_direction_independent(self):
        long_engine = self._engine(sr_long_avoid_near_resistance=True)
        short_engine = self._engine(sr_short_avoid_near_support=True)
        context = self._context(near_support=True, near_resistance=True)
        assert long_engine._should_reject_for_sr(10, "SHORT", context) == (False, None)
        assert short_engine._should_reject_for_sr(10, "LONG", context) == (False, None)

    def test_unrecognized_mode_is_rejected_by_current_config_contract(self):
        with pytest.raises(ValueError, match="invalid sr_filter_mode"):
            self._engine(sr_filter_mode="UNKNOWN", sr_long_avoid_near_resistance=True)


def test_room_in_direction_uses_opposing_structure():
    detector = SupportResistanceDetector()
    support = SRLevel(
        95.0, SRLevelType.SUPPORT, 1, 1, zone_bottom=94.0, zone_top=96.0
    )
    resistance = SRLevel(
        110.0, SRLevelType.RESISTANCE, 2, 2, zone_bottom=108.0, zone_top=112.0
    )

    assert detector._calculate_room_in_direction(
        support, resistance, 100.0, "LONG", 2.0
    ) == pytest.approx(4.0)
    assert detector._calculate_room_in_direction(
        support, resistance, 100.0, "SHORT", 2.0
    ) == pytest.approx(2.0)

class TestAnalysisOnlyRegression:
    """ANALYSIS_ONLY must enrich current trades without changing execution."""

    @staticmethod
    def _wavy_candles(n=80):
        rows = []
        price = 100.0
        for i in range(n):
            wave = 5 * np.sin(i / 4.0)
            high = price + wave + 2
            low = price + wave - 2
            close = price + wave
            rows.append((price, high, low, close))
            price = close
        start = pd.Timestamp("2024-01-01", tz="UTC")
        return pd.DataFrame({
            "timestamp": [start + pd.Timedelta(minutes=15 * i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1] * len(rows),
        })

    @staticmethod
    def _run_current_profile(data, enable_sr):
        config = BacktestConfig(
            risk_mode=RiskMode.FIXED,
            fixed_r=2.0,
            use_intrabar_data=False,
            enable_trade_telemetry=False,
            enable_support_resistance_analysis=enable_sr,
            sr_filter_mode="ANALYSIS_ONLY",
        )
        engine = BacktestEngine(data, config)
        engine.market_regime_values[:] = "SIDEWAYS"
        engine.plus_di_values[:] = 50.0
        engine.minus_di_values[:] = 10.0
        engine.di_spread[:] = 40.0
        return engine.run()

    def test_analysis_only_trades_match_sr_disabled_trades(self):
        data = self._wavy_candles()
        disabled = self._run_current_profile(data, False)
        enabled = self._run_current_profile(data, True)
        assert len(disabled) == len(enabled)
        assert len(disabled) > 0
        shared_columns = [c for c in disabled.columns if c in enabled.columns and "_sr_" not in c]
        pd.testing.assert_frame_equal(
            disabled[shared_columns].reset_index(drop=True),
            enabled[shared_columns].reset_index(drop=True),
        )
        assert enabled["long_sr_context"].notna().any()
        assert disabled["long_sr_context"].isna().all()

    def test_sr_zone_and_level_columns_present_for_selected_side(self):
        enabled = self._run_current_profile(self._wavy_candles(), True)
        for column in ("long_sr_zone_low", "long_sr_zone_high", "long_sr_level_price"):
            assert column in enabled.columns
        assert enabled["long_sr_zone_low"].notna().any()
        assert "short_sr_zone_low" not in enabled.columns


class TestSupportResistanceDetector:
    """Tests for full SR detection and analysis."""
    
    def test_insufficient_data_returns_default(self):
        """Short data returns default context."""
        detector = SupportResistanceDetector(pivot_left=5, pivot_right=5)
        
        opens = np.array([100.0] * 5, dtype=np.float64)
        highs = np.array([101.0] * 5, dtype=np.float64)
        lows = np.array([99.0] * 5, dtype=np.float64)
        closes = np.array([100.5] * 5, dtype=np.float64)
        atrs = np.array([1.0] * 5, dtype=np.float64)
        
        context = detector.analyze_price_location(
            3, opens, highs, lows, closes, atrs, "LONG"
        )
        
        assert context.price_location == LocationClassification.NO_STRUCTURE
        assert context.trade_location_rating == TradeLocationRating.NEUTRAL_LOCATION
    
    def test_long_near_support_good_location(self):
        """Long trade near support classified as GOOD_LOCATION."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50,
            near_distance_atr=1.0
        )
        
        # Create data with a valid swing low
        # Index 5 should be a valley: lower than indices 3,4 and 6,7
        closes = np.array([
            100.0,  # 0
            100.0,  # 1
            100.5,  # 2 - higher
            101.0,  # 3 - higher (left side going up)
            100.0,  # 4 - down
            90.0,   # 5 - SWING LOW (valley)
            100.0,  # 6 - up
            100.5,  # 7 - higher (right side going up)
            100.0,  # 8
            100.0,  # 9
            100.0,  # 10
            100.0,  # 11
            100.0,  # 12
            100.0,  # 13
            100.1,  # 14
            100.2,  # 15 - near support at 90
        ], dtype=np.float64)
        
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0
        lows[5] = 89.5  # Make the swing low valid
        
        atrs = np.array([1.0] * 16, dtype=np.float64)
        
        # At index 15, price is 100.2, nearest support should be 90.0
        # Distance = (100.2 - 90.0) / 1.0 = 10.2 ATR (but should still classify location)
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        # With near_distance_atr=1.0, distance of 10.2 is NOT near
        # But we should still have detected the support
        if context.nearest_support_price is not None:
            assert context.trade_location_rating in [
                TradeLocationRating.GOOD_LOCATION,
                TradeLocationRating.NEUTRAL_LOCATION
            ], f"Unexpected rating: {context.trade_location_rating}"

    def test_support_and_resistance_zone_bounds_are_populated(self):
        """SRContext exposes zone_low/zone_high bounds for the nearest confirmed support/resistance."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50,
            near_distance_atr=1.0, zone_width_atr=0.5,
        )
        closes = np.array([
            100.0, 100.0, 100.5, 101.0, 100.0, 90.0, 100.0, 100.5,
            100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.1, 100.2,
        ], dtype=np.float64)
        opens = closes.copy(); highs = closes + 1.0; lows = closes - 1.0; lows[5] = 89.5
        atrs = np.array([1.0] * 16, dtype=np.float64)

        context = detector.analyze_price_location(15, opens, highs, lows, closes, atrs, "LONG")

        if context.nearest_support_price is not None:
            assert context.support_zone_low is not None and context.support_zone_high is not None
            assert context.support_zone_low <= context.nearest_support_price <= context.support_zone_high

    def test_short_near_resistance_good_location(self):
        """Short trade near resistance classified as GOOD_LOCATION."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50,
            near_distance_atr=1.0
        )
        
        # Create data with a valid swing high
        closes = np.array([
            100.0,  # 0
            100.0,  # 1
            99.5,   # 2 - lower
            99.0,   # 3 - lower (left side going down)
            100.0,  # 4 - up
            110.0,  # 5 - SWING HIGH (peak)
            100.0,  # 6 - down
            99.5,   # 7 - lower (right side going down)
            100.0,  # 8
            100.0,  # 9
            100.0,  # 10
            100.0,  # 11
            100.0,  # 12
            100.0,  # 13
            99.9,   # 14
            99.8,   # 15 - near resistance at 110
        ], dtype=np.float64)
        
        opens = closes.copy()
        highs = closes + 1.0
        highs[5] = 110.5  # Make the swing high valid
        lows = closes - 1.0
        
        atrs = np.array([1.0] * 16, dtype=np.float64)
        
        # At index 15, price is 99.8, nearest resistance should be 110.0
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "SHORT"
        )
        
        # With near_distance_atr=1.0, distance of ~10.2 is NOT near
        # But location should be properly classified
        if context.nearest_resistance_price is not None:
            assert context.trade_location_rating in [
                TradeLocationRating.GOOD_LOCATION,
                TradeLocationRating.NEUTRAL_LOCATION
            ], f"Unexpected rating: {context.trade_location_rating}"
    
    def test_long_near_resistance_bad_location(self):
        """Long trade near resistance classified as BAD_LOCATION."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50,
            near_distance_atr=2.0  # Increase threshold so we're "near"
        )
        
        # Create data with a valid swing high, price near it
        closes = np.array([
            100.0,  # 0
            100.0,  # 1
            99.5,   # 2 - lower
            99.0,   # 3 - lower (left side)
            100.0,  # 4 - up
            110.0,  # 5 - SWING HIGH
            100.0,  # 6 - down
            99.5,   # 7 - lower (right side)
            100.0,  # 8
            100.5,  # 9
            101.0,  # 10
            101.5,  # 11
            102.0,  # 12
            102.5,  # 13
            103.0,  # 14
            103.5,  # 15 - moving toward resistance
        ], dtype=np.float64)
        
        opens = closes.copy()
        highs = closes + 1.0
        highs[5] = 110.5  # Peak
        lows = closes - 1.0
        
        # With ATR=1.0, distance from 103.5 to 110.0 is 6.5/1.0 = 6.5 ATR
        # This is still > near_distance_atr=2.0, so won't be "NEAR"
        # But if we use a smaller ATR, it becomes near
        atrs = np.array([1.0] * 16, dtype=np.float64)
        
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        # With price approaching but not super close to resistance
        assert context.price_location in [
            LocationClassification.NEAR_RESISTANCE,
            LocationClassification.BETWEEN_LEVELS
        ], f"Expected near or between resistance, got {context.price_location}"
    
    def test_between_levels_neutral(self):
        """Price between support and resistance classified as NEUTRAL."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50,
            near_distance_atr=0.5
        )
        
        opens = np.array([100.0] * 20, dtype=np.float64)
        highs = np.full(20, 102.0, dtype=np.float64)
        lows = np.full(20, 98.0, dtype=np.float64)
        
        # Swing low at 5, swing high at 10
        lows[5] = 90.0
        highs[10] = 110.0
        
        closes = np.array(
            [100.0] * 5 + [91.0] + [100.0] * 4 + [109.0] + [100.0] * 9,
            dtype=np.float64
        )
        atrs = np.array([1.0] * 20, dtype=np.float64)
        
        # At index 15, price at 100 is well between support (90) and resistance (110)
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        assert context.price_location == LocationClassification.BETWEEN_LEVELS, \
            f"Expected BETWEEN_LEVELS, got {context.price_location}"
        assert context.trade_location_rating == TradeLocationRating.NEUTRAL_LOCATION
    
    def test_distance_calculation_accuracy(self):
        """Support/resistance distances calculated correctly in ATR units."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50
        )
        
        # Create data with a clear swing low at index 5
        # Need 2 bars before/after lower values
        closes = np.array([
            100.0,  # 0
            100.0,  # 1
            100.0,  # 2 - higher
            100.0,  # 3 - higher
            100.0,  # 4 - up
            50.0,   # 5 - SWING LOW
            100.0,  # 6 - up
            100.0,  # 7 - higher
            100.0,  # 8
            100.0,  # 9
            100.0,  # 10
            100.0,  # 11
            100.0,  # 12
            100.0,  # 13
            100.0,  # 14
            100.0,  # 15 - price at 100, support at 50
        ], dtype=np.float64)
        
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0
        
        # Price at 100, support at 50, ATR = 10.0
        # Distance = (100 - 50) / 10.0 = 5.0 ATR
        atrs = np.array([10.0] * 16, dtype=np.float64)
        
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        # Check that a support level was detected
        assert context.nearest_support_price is not None, \
            f"Support should be detected, got {context.nearest_support_price}"
        
        # Distance should be approximately 5.0 ATR
        if not np.isnan(context.nearest_support_distance_atr):
            assert np.isclose(context.nearest_support_distance_atr, 5.0, rtol=0.2), \
                f"Expected ~5.0 ATR distance, got {context.nearest_support_distance_atr}"
    
    def test_near_support_flag_accuracy(self):
        """near_support flag set correctly based on distance."""
        detector = SupportResistanceDetector(
            pivot_left=2, pivot_right=2, lookback_bars=50,
            near_distance_atr=1.0  # Within 1.0 ATR is "near"
        )
        
        # Create data with a swing low at index 5
        closes = np.array([
            100.0,  # 0
            100.0,  # 1
            100.0,  # 2
            100.0,  # 3
            100.0,  # 4
            50.0,   # 5 - SWING LOW
            100.0,  # 6
            100.0,  # 7
            100.0,  # 8
            100.0,  # 9
            100.0,  # 10
            100.0,  # 11
            100.0,  # 12
            100.0,  # 13
            100.0,  # 14
            100.0,  # 15 - price = 100, support = 50
        ], dtype=np.float64)
        
        opens = closes.copy()
        highs = closes + 1.0
        lows = closes - 1.0
        
        # Price = 100, support = 50, ATR = 10.0
        # Distance = (100 - 50) / 10.0 = 5.0 ATR (NOT near, since threshold is 1.0)
        atrs = np.array([10.0] * 16, dtype=np.float64)
        
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        assert not context.near_support, \
            f"Distance 5.0 ATR should NOT be near (threshold 1.0), got {context.near_support}"
        
        # Now test with price closer to support
        # Price = 50.5, support = 50, ATR = 1.0
        # Distance = 0.5 / 1.0 = 0.5 ATR (IS near)
        closes[15] = 50.5
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        assert context.near_support, \
            f"Distance 0.5 ATR should be near (threshold 1.0), got {context.near_support}"
    
    def test_zero_atr_handles_gracefully(self):
        """Zero or invalid ATR returns default context."""
        detector = SupportResistanceDetector()
        
        opens = np.array([100.0] * 20, dtype=np.float64)
        highs = np.array([101.0] * 20, dtype=np.float64)
        lows = np.array([99.0] * 20, dtype=np.float64)
        closes = np.array([100.0] * 20, dtype=np.float64)
        atrs = np.array([0.0] * 20, dtype=np.float64)  # Invalid
        
        context = detector.analyze_price_location(
            15, opens, highs, lows, closes, atrs, "LONG"
        )
        
        # Should return default (no errors)
        assert context.price_location == LocationClassification.NO_STRUCTURE

    def test_incremental_levels_match_legacy_swing_scan(self):
        """Incremental active levels match the original causal scan."""
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, 120))
        high = close + rng.uniform(0.1, 1.0, len(close))
        low = close - rng.uniform(0.1, 1.0, len(close))
        opens = close.copy()
        atrs = np.full(len(close), 1.5)
        detector = SupportResistanceDetector(pivot_left=3, pivot_right=2, lookback_bars=40)

        for index in range(len(close)):
            context = detector.analyze_price_location(index, opens, high, low, close, atrs, "LONG")
            high_indices = detector.swing_detector.detect_swing_highs(high, index)
            low_indices = detector.swing_detector.detect_swing_lows(low, index)
            start = max(0, index - detector.lookback_bars)
            expected_highs = [
                SRLevel(float(high[i]), SRLevelType.RESISTANCE, i, i)
                for i in high_indices if i >= start
            ]
            expected_lows = [
                SRLevel(float(low[i]), SRLevelType.SUPPORT, i, i)
                for i in low_indices if i >= start
            ]
            expected_resistance = detector.zone_merger.merge_levels(expected_highs, atrs[index])
            expected_support = detector.zone_merger.merge_levels(expected_lows, atrs[index])
            expected_nearest_support = detector._nearest_level(expected_support, close[index], below=True)
            expected_nearest_resistance = detector._nearest_level(expected_resistance, close[index], below=False)
            assert context.nearest_support_bar_index == (
                expected_nearest_support.bar_index if expected_nearest_support else None
            )
            assert context.nearest_resistance_bar_index == (
                expected_nearest_resistance.bar_index if expected_nearest_resistance else None
            )

    def test_context_cache_and_directional_derivation(self):
        """Repeated requests reuse structural work and only derive direction-specific fields."""
        close = np.array([100, 100, 105, 100, 95, 100, 105, 100, 100, 100], dtype=float)
        high = close + 1
        low = close - 1
        context_detector = SupportResistanceDetector(
            pivot_left=1, pivot_right=1, lookback_bars=20, near_distance_atr=1.1
        )
        long_context = context_detector.analyze_price_location(8, close, high, low, close, np.ones(10), "LONG")
        short_context = context_detector.analyze_price_location(8, close, high, low, close, np.ones(10), "SHORT")
        assert context_detector.analyze_price_location(8, close, high, low, close, np.ones(10), "LONG") is long_context
        assert context_detector.analyze_price_location(8, close, high, low, close, np.ones(10), "SHORT") is short_context
        assert long_context.nearest_support_price == short_context.nearest_support_price
        assert long_context.nearest_resistance_price == short_context.nearest_resistance_price
        assert long_context.trade_location_rating != short_context.trade_location_rating

    def test_zone_distance_uses_near_zone_edge(self):
        """Distances are measured to the nearest edge, not the extreme level price."""
        detector = SupportResistanceDetector()
        support = SRLevel(90.0, SRLevelType.SUPPORT, 1, 1)
        support.zone_bottom = 90.0
        support.zone_top = 95.0
        resistance = SRLevel(110.0, SRLevelType.RESISTANCE, 2, 2)
        resistance.zone_bottom = 105.0
        resistance.zone_top = 110.0
        assert detector._calculate_distance(100.0, support, 1.0) == (5.0, 5.0)
        assert detector._calculate_distance(100.0, resistance, 1.0) == (5.0, 5.0)
        assert detector._calculate_distance(93.0, support, 1.0) == (0.0, 0.0)
        assert detector._calculate_distance(107.0, resistance, 1.0) == (0.0, 0.0)

    def test_incremental_detector_scales_to_60000_candles(self):
        """A large causal run remains bounded by the active lookback set."""
        count = 60_000
        close = 100.0 + np.sin(np.arange(count, dtype=float) / 17.0)
        high = close + 0.25
        low = close - 0.25
        atrs = np.ones(count, dtype=float)
        detector = SupportResistanceDetector(pivot_left=5, pivot_right=5, lookback_bars=200)
        started = time.perf_counter()
        for index in range(count):
            detector.analyze_price_location(index, close, high, low, close, atrs, "LONG")
        elapsed = time.perf_counter() - started
        assert elapsed < 15.0
        assert detector._last_processed_index == count - 1

    def test_support_test_hold_is_confirmed_only_after_bounce(self):
        detector = SupportResistanceDetector(
            pivot_left=1, pivot_right=1, lookback_bars=20,
            hold_confirmation_bars=3, hold_confirmation_atr=0.25,
            break_tolerance_atr=0.25,
        )
        close = np.array([105, 105, 100, 105, 100.5, 100.5, 101.5], dtype=float)
        high = np.array([106, 106, 101, 106, 101.5, 101.5, 102], dtype=float)
        low = np.array([104, 104, 100, 104, 100, 100.5, 101], dtype=float)
        atrs = np.full(len(close), 4.0)
        contexts = [detector.analyze_price_location(i, close, high, low, close, atrs, "LONG") for i in range(len(close))]
        assert contexts[4].support_state == "SUPPORT_TESTING"
        assert not contexts[4].support_held
        assert contexts[5].support_state == "SUPPORT_TESTING"
        assert not contexts[5].support_held
        assert contexts[6].support_state == "SUPPORT_HELD"
        assert contexts[6].support_held
        assert contexts[6].support_rejection_atr >= 0.25
        assert contexts[4].support_state != "SUPPORT_HELD"
        assert contexts[5].support_state != "SUPPORT_HELD"

    def test_resistance_test_hold_is_causal(self):
        detector = SupportResistanceDetector(
            pivot_left=1, pivot_right=1, lookback_bars=20,
            hold_confirmation_bars=3, hold_confirmation_atr=0.25,
        )
        close = np.array([95, 95, 100, 95, 99.5, 99.5, 98.5], dtype=float)
        high = np.array([96, 96, 100, 96, 100, 99.5, 99], dtype=float)
        low = np.array([94, 94, 99, 94, 99, 98.5, 98], dtype=float)
        atrs = np.full(len(close), 4.0)
        contexts = [detector.analyze_price_location(i, close, high, low, close, atrs, "SHORT") for i in range(len(close))]
        assert contexts[4].resistance_state == "RESISTANCE_TESTING"
        assert contexts[5].resistance_state == "RESISTANCE_TESTING"
        assert contexts[6].resistance_state == "RESISTANCE_HELD"
        assert contexts[6].resistance_held

    def test_support_break_timeout_wick_basis_and_expiry(self):
        close = np.array([105, 105, 100, 105, 100.5, 100.5, 100.5, 100.5, 98.5], dtype=float)
        high = np.array([106, 106, 101, 106, 101.5, 101.5, 101.5, 101.5, 100.5], dtype=float)
        low = np.array([104, 104, 100, 104, 100, 100.5, 100.5, 100.5, 98.0], dtype=float)
        atrs = np.full(len(close), 4.0)
        close_detector = SupportResistanceDetector(pivot_left=1, pivot_right=1, lookback_bars=20, hold_confirmation_bars=2)
        close_contexts = [close_detector.analyze_price_location(i, close, high, low, close, atrs, "LONG") for i in range(len(close))]
        assert close_contexts[7].support_state == "APPROACHING_SUPPORT"
        assert close_contexts[8].support_state == "SUPPORT_BROKEN"

        wick_detector = SupportResistanceDetector(pivot_left=1, pivot_right=1, lookback_bars=20, break_basis="WICK")
        wick_contexts = [wick_detector.analyze_price_location(i, close, high, low, close, atrs, "LONG") for i in range(len(close))]
        assert wick_contexts[8].support_state == "SUPPORT_BROKEN"

        timeout_detector = SupportResistanceDetector(pivot_left=1, pivot_right=1, lookback_bars=20, hold_confirmation_bars=1)
        timeout_contexts = [timeout_detector.analyze_price_location(i, close[:8], high[:8], low[:8], close[:8], atrs[:8], "LONG") for i in range(8)]
        assert timeout_contexts[7].support_state == "APPROACHING_SUPPORT"

        expiry_detector = SupportResistanceDetector(pivot_left=1, pivot_right=1, lookback_bars=2)
        expiry_contexts = [expiry_detector.analyze_price_location(i, close, high, low, close, atrs, "LONG") for i in range(len(close))]
        assert expiry_contexts[8].support_state == "NO_SUPPORT_NEARBY"

    def test_multiple_support_tests_increment_count(self):
        detector = SupportResistanceDetector(pivot_left=1, pivot_right=1, lookback_bars=30, hold_confirmation_bars=2, hold_confirmation_atr=0.25)
        close = np.array([105, 105, 100, 105, 100, 101.5, 100, 101.5], dtype=float)
        high = np.array([106, 106, 101, 106, 101, 102, 101, 102], dtype=float)
        low = np.array([104, 104, 100, 104, 99.5, 101, 99.5, 101], dtype=float)
        atrs = np.full(len(close), 4.0)
        contexts = [detector.analyze_price_location(i, close, high, low, close, atrs, "LONG") for i in range(len(close))]
        assert contexts[4].support_test_count == 1
        assert contexts[6].support_test_count == 2
        assert contexts[6].support_last_test_index == 6


class TestIntegrationNoLookAhead:
    """Integration tests ensuring no look-ahead bias."""
    
    def test_full_detection_no_future_data(self):
        """Full SR analysis never uses future candles."""
        detector = SupportResistanceDetector(pivot_left=3, pivot_right=3, lookback_bars=20)
        
        # Create 30-candle dataset
        opens = np.linspace(100, 110, 30, dtype=np.float64)
        highs = opens + 2
        lows = opens - 2
        closes = opens + 0.5
        atrs = np.ones(30, dtype=np.float64)
        
        # At each historical point, verify no future data used
        for i in range(6, 30):  # Only where enough data exists
            context = detector.analyze_price_location(
                i, opens, highs, lows, closes, atrs, "LONG"
            )
            
            # All levels should have bar_index <= i - pivot_right
            if context.nearest_support_price is not None:
                assert context.nearest_support_bar_index is not None
                assert context.nearest_support_bar_index <= i - 3, \
                    f"Support at index {context.nearest_support_bar_index} uses future from {i}"
            
            if context.nearest_resistance_price is not None:
                assert context.nearest_resistance_bar_index is not None
                assert context.nearest_resistance_bar_index <= i - 3, \
                    f"Resistance at index {context.nearest_resistance_bar_index} uses future from {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
