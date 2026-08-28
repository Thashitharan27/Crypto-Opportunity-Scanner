# Support/Resistance Analysis

## Scope

Support/resistance (S/R) analysis detects confirmed swing structure without look-ahead,
stores an entry-time context snapshot on each position, and produces telemetry and analysis
reports. Detection is optional and is controlled by `enable_support_resistance_analysis`.

The implementation is split across:

- `crypto_strategy_lab/support_resistance.py`: swing detection, zone merging, interaction
  state, location classification, and directional room calculation.
- `crypto_strategy_lab/engine.py`: detector integration, entry rules, entry-time snapshots,
  and event labels.
- `crypto_strategy_lab/support_resistance_analysis.py`: CSV report generation.
- `crypto_strategy_lab/gui/main_window.py`: the single **Support & Resistance** configuration
  tab.
- `crypto_strategy_lab/gui/config_logic.py`: defaults, validation, serialization, and
  `BacktestConfig` construction.

## Filtering modes

Only two filtering modes are supported:

- `ANALYSIS_ONLY`: calculate and report S/R context without changing entry decisions.
- `APPLY_ENTRY_RULES`: apply the enabled direction-specific entry constraints.

An unrecognized value is invalid during GUI/config validation and is never interpreted as
another policy by the engine. Saved mode values are not normalized or migrated.

## Entry-rule configuration

These are the complete supported entry-rule fields:

### Long

- `sr_long_avoid_near_resistance`
- `sr_long_require_near_support`
- `sr_long_block_broken_support`
- `sr_long_min_room_to_resistance_atr`

### Short

- `sr_short_avoid_near_support`
- `sr_short_require_near_resistance`
- `sr_short_block_broken_resistance`
- `sr_short_min_room_to_support_atr`

Boolean rules are independent. Minimum-room rules are disabled at `0.0` and otherwise
reject an entry when the available room is below the configured ATR multiple.

## Directional room semantics

`room_in_direction_atr` measures the distance from the current price to the nearest
opposing zone boundary in the trade direction:

- For a **LONG**, it is `(resistance.zone_bottom - price) / ATR`.
- For a **SHORT**, it is `(price - support.zone_top) / ATR`.

It is `NaN` when ATR is unusable, the relevant opposing structure does not exist, or the
opposing zone is not ahead of the trade. Thus the long minimum-room rule always refers to
resistance room, while the short minimum-room rule always refers to support room.

## Event classification and reporting

Entry-time event labels remain telemetry/reporting data and are not filtering policies.
The engine can classify stored snapshots as:

- `NEAR_SUPPORT`
- `NEAR_RESISTANCE`
- `SUPPORT_BOUNCE`
- `RESISTANCE_BREAKOUT`
- `RESISTANCE_REJECTION`
- `SUPPORT_BREAKDOWN`
- `NO_NEARBY_SR`

Labels are derived from the position's stored entry snapshot rather than recalculating
structure after the trade. Reports include context, regime, distance, hold/rejection, and
test-count analyses.

## GUI

All S/R configuration is contained in the **Support & Resistance** tab. It contains:

1. Analysis enablement and policy selection.
2. Detection parameters.
3. Long and short entry-rule panels.
4. Confirmation and break-detection parameters.
5. A live summary of the effective rules.

The results area may display best/worst S/R contexts; this is report output rather than a
second configuration surface.

## Testing

S/R behavior is covered by:

- `tests/test_support_resistance.py` for causal detection, interaction state, filtering,
  analysis-only regression behavior, and directional room semantics.
- `tests/test_sr_reporting.py` for event labels and reports.
- `tests/test_gui_config_logic.py` for validation and config round trips.
- `tests/test_gui_main_window.py` for the single S/R tab and its two policy choices.
