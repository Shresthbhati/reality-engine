# Reality Engine — Roadmap (Priority Order)

**This file is the authoritative WHAT.** Agents and sessions execute the
priorities in this order; they own the HOW. Do not reorder based on
preference. If a priority is blocked, resolve the dependency and return;
if the blocker requires a genuine product decision, record it as BLOCKED
in [`PENDING_IMPLEMENTATION.md`](PENDING_IMPLEMENTATION.md) (with options,
tradeoffs, recommendation) and continue with the next independent priority.

Status per item lives in [`PENDING_IMPLEMENTATION.md`](PENDING_IMPLEMENTATION.md).
Detailed specs for future subsystems live in [`../future/`](../future/).
Why each architectural choice was made lives in
[`../engineering/DESIGN_DECISIONS.md`](../engineering/DESIGN_DECISIONS.md).

---

## Execution order

| # | Priority | Outcome | Status |
|---|----------|---------|--------|
| P0.1–P0.20 | Mapping spine | Real photos → reconstruction → metric geometry → depth → perception → mesh → WorldIR → artifacts → viewer → `reality compile` | Mostly VERIFIED; gaps listed in backlog (MVS, acceptance test, registry coverage) |
| P1.4–P1.13 | Sensor + temporal foundation | RGB-D/IMU/GNSS as first-class evidence, time sync, VIO, cross-source registration, dense MVS + fusion | PARTIAL/MISSING — next after mapping-spine gaps |
| P2.14–P2.17 | Perception + identity | Multi-view identity, tracking, structure, materials | PARTIAL/MISSING |
| P3.18–P3.19 | Uncertainty + provenance | Covariance propagation; queryable lineage graph | PARTIAL |
| P4.20–P4.21 | Persistent world | WorldStore (SQLite), incremental compilation | MISSING |
| P6 | Reality Studio | Full client over WorldStore/WorldIR: evidence/provenance browsing, timeline, diff, measurement | FOUNDATION |
| P7.25–P7.29 | Domains + infra | GIS, robotics, large-world, packaging verification, CI | PARTIAL/MISSING |
| P8.30–P8.31 | Simulation | Physics → WorldIR write-back, advanced simulation | PARTIAL/deferred |

## Rules

1. **Dependency before breadth.** If P1.10 (registration) blocks P1.11
   (fusion), registration goes first regardless of personal preference.
2. **Verify before status.** No IMPLEMENTED/VERIFIED without execution
   evidence (see [`../engineering/VERIFICATION.md`](../engineering/VERIFICATION.md)).
3. **Never delete future items** because a partial interface exists.
4. **Deferred ≠ deleted.** P8 simulation systems stay unwired until the
   mapping spine and P4 persistence are real.
5. **Every session ends by updating** `.agent/EXECUTION_STATE.md` and
   `.agent/TASKS.yaml` so the next session resumes without
   rediscovery.
