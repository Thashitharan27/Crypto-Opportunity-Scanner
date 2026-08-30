# Task 10: PAPER scanner runtime

Task 10 adds a PAPER-only application layer around the existing live
`OPPORTUNITY_SCAN` service. It does not introduce another discovery, scoring,
Data Lake, or strategy implementation.

## Composition and data flow

`create_production_paper_scanner()` composes the Task 9 scanner application
service with the existing Data Lake backtest bundle loader and native
rule-aware production engine. Each cycle therefore follows the established
pipeline:

1. execute a `LIVE` opportunity scan through the application service;
2. consume that completed run's current final-candidate table;
3. load each candidate's published strategy request from the Data Lake;
4. choose the latest strategy row whose candle and required research inputs
   were available at the scan decision time;
5. invoke the native entry/profile/veto evaluation; and
6. durably record an accepted, non-duplicate synthetic PAPER entry.

The runtime deliberately has no rolling-candidate or position-retention policy.
Every cycle evaluates only the final candidates published by that cycle.

## Safety and durability

The signal identity is a SHA-256 digest of symbol, normalized strategy
interval, native strategy profile, final side, and authoritative signal-candle
timestamp. Before an entry is reported as emitted, the identity and PAPER
record are committed together by atomic state-file replacement. State schema
version or ledger/index corruption causes startup to fail closed rather than
resetting duplicate protection.

The state file and append-only JSONL audit log must be stored on durable local
storage. Configure them as different paths. Retention must remain longer than
the strategy stale-data limit so an otherwise eligible duplicate key cannot be
pruned.

Both discovery and strategy rows have explicit maximum ages. Future discovery
timestamps, incomplete candles, late required research features, and stale
rows are rejected without producing an entry. Transient public-data failures
have bounded retries; permanent errors fail the cycle, and stop-event waits
interrupt both backoff and scheduling.

## PAPER-only boundary

The production factory accepts scanner/Data Lake configuration only. It has no
broker, authenticated trading client, API key, order-submission callback,
position creator, or paper-to-live switch. A `PaperOpportunitySignal` is an
auditable synthetic record, not an exchange order.

## Focused verification

Run the deterministic Task 10 suite without Binance or any other network
dependency:

```bash
PYTHONPATH=. pytest -q tests/test_paper_scanner.py tests/test_rule_native_engine.py
```

