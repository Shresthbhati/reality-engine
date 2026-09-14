# Time Synchronization

Backlog: P1.6 — MISSING.

## Model

```
t_global = a · t_sensor + b
```

`a` = clock scale/drift, `b` = offset. Single-device streams: `a = 1`,
estimate `b`. GNSS/PPS-anchored streams: estimate both.

## Components

- `ClockModel(clock_id, a, b, uncertainty, method)` — one per sensor
  clock; immutable.
- `Timestamp(sensor_t, clock_id)` — original value always retained.
- `TimeAlignment` — applies a model; every synchronized sample carries:
  original timestamp · normalized timestamp · clock id · offset · drift
  · sync uncertainty · sync method.
- `SynchronizationDiagnostics` — offset estimate, drift estimate,
  residuals, confidence, rejected samples.

## Methods (in order of preference)

1. Hardware timestamps / shared clock (`a=1`, `b=0`, uncertainty from spec)
2. Known offset (manual `b`)
3. GNSS/PPS anchoring
4. Trigger synchronization
5. Signal correlation (audio/flash/event cross-correlation)
6. Optimization (bundle-adjust `a,b` against cross-stream constraints)

## Rules

- Original sensor timestamps are never overwritten or discarded.
- Never fabricate uncertainty: report the *estimate quality* only when a
  residual can be computed.
- Sync failure degrades honestly (samples flagged `UNSYNCHRONIZED`),
  never silently re-stamps.

## Tests

Synthetic offset (pure `b`) · synthetic drift (`a≠1`) · cross-stream
alignment · rejected-sample handling · round-trip provenance.

## Evidence to advance status

`ClockModel` + `TimeAlignment` implemented with diagnostics; offset and
drift synthetic tests; one cross-stream integration test.
