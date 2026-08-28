# State Transition Research

`crypto_strategy_lab.state_transition_research` adds a research-only Markov/state-transition layer to Crypto Strategy Lab.

## Scope

This research layer deliberately does **not** alter entries, direction selection, stops, targets, sizing, Strategy Profile selection, or execution.

It answers three questions:

1. Given the current broad market state, what state historically came next?
2. How did actual backtest trades perform when grouped by **DI bucket × DI movement × volatility state**?
3. How did LONG and SHORT trades perform when the last fully completed 20-day regime was BULL, BEAR, or SIDEWAYS?

## Broad market state

The default state model is:

- `BULL`: trailing 20-day close return >= +5%
- `SIDEWAYS`: trailing 20-day close return between -5% and +5%
- `BEAR`: trailing 20-day close return <= -5%
- `UNKNOWN`: the 20-day lookback is not available yet

The output transition matrix includes probability and sample count. Rows whose current-state sample count is below the configured minimum are marked rather than silently trusted.

## Causal trade-state rule

Every trade uses only the **last fully completed daily state before the trade entry**.

A state dated `D` is considered available at `D + 1 day, 00:00 UTC`. A trade on day `D + 1` therefore cannot use the still-forming state for `D + 1`.

The per-trade expected-next regime is also causal. Its transition probabilities are built only from regime transitions that were already observable by the completed state date used for that trade. Future transitions from the backtest sample are not allowed to leak into the trade row.

## Trade alignment

Alignment between the completed regime and the actual final trade direction is classified as:

- `BULL + LONG` = `AGREE`
- `BULL + SHORT` = `COUNTER`
- `BEAR + SHORT` = `AGREE`
- `BEAR + LONG` = `COUNTER`
- `SIDEWAYS + LONG/SHORT` = `NEUTRAL`

The same mapping is used for `research_regime_transition_agreement`, but against the causal expected-next regime instead of the current completed regime.

## Volatility state

Daily volatility is the rolling standard deviation of one-day returns. The state is:

- `LOW`
- `NORMAL`
- `ELEVATED`

The low/high boundaries are rolling historical quantiles. They are shifted by one day before classification, so the current day's boundary never uses future observations.

## DI state

Trade rows use `directional_di` when available. Fallbacks are `max(plus_di, minus_di)` and finally `di_spread`.

Default buckets:

- 0-5
- 5-10
- 10-15
- 15-20
- 20-25
- 25-30
- 30+

DI movement uses `directional_di_change` when available, with fallbacks to existing DI-change telemetry. The default classification is:

- `RISING`: change > +0.5
- `STABLE`: change between -0.5 and +0.5
- `FALLING`: change < -0.5

## Generated files

Calling `generate_state_transition_reports(strategy_data, trades, run_dir)` creates:

`state_transition_research/daily_states.csv`

Daily close, trailing return, broad regime, rolling volatility, and volatility state.

`state_transition_research/regime_transition_matrix.csv`

Broad BULL/SIDEWAYS/BEAR transition probabilities and counts for the complete research sample.

`state_transition_research/volatility_transition_matrix.csv`

LOW/NORMAL/ELEVATED volatility transition probabilities and counts.

`state_transition_research/current_regime_probabilities.csv`

Only the broad transition probabilities applicable to the most recent regime state.

`state_transition_research/di_state_volatility_trade_performance.csv`

Trade count, wins, win rate, net R, average R, and minimum-sample flag for every observed DI bucket × DI movement × volatility state combination.

`state_transition_research/regime_direction_trade_performance.csv`

Six fixed rows: BULL/Bear/SIDEWAYS × LONG/SHORT. Each row contains trades, wins, losses, win rate, Net R, Avg R, and minimum-sample status. The regime assigned to each trade is the last fully completed daily regime before entry.

`state_transition_research/regime_alignment_trade_performance.csv`

AGREE, COUNTER, and NEUTRAL outcome summary with trades, wins, losses, win rate, Net R, Avg R, and minimum-sample status.

## `trade_list.csv` research telemetry

State-transition research runs overwrite the normal `trade_list.csv` export with the same trade rows plus these reporting-only fields:

- `research_regime_state`
- `research_regime_date`
- `research_regime_return_20d`
- `research_regime_trade_alignment`
- `research_regime_expected_next_state`
- `research_regime_expected_next_probability`
- `research_regime_transition_agreement`

The expected-next state and probability are calculated from transition history available at that time, not from future sample data.

## Example integration

```python
from crypto_strategy_lab.state_transition_research import generate_state_transition_reports

generate_state_transition_reports(data, trades, run_dir)
```

`data` is the strategy candle DataFrame used by the backtest; `trades` is the completed trade DataFrame.

## Research rules

These fields and reports are telemetry only. They must not be read by the entry engine or used to alter entry selection, trade direction, position sizing, TP, SL, or any other trading behavior unless a separate future strategy change is explicitly designed and walk-forward validated.
