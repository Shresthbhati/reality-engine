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
| P0.1–P0.20 | Mapping spine | Real photos → reconstruction → metric geometry → depth → perception → mesh → WorldIR → artifacts → viewer → `reality compile` | IMPLEMENTED; real-data validated (ledger P0-ROBUST-01/02 DONE, `tests/integration`, `tests/test_cli_compile.py`). Dense MVS now DONE (P6-01). Open per backlog: P0.15 relationships PARTIAL, P0.18 acceptance test not a committed automated test |
| P1.4–P1.13 | Sensor + temporal foundation | RGB-D/IMU/GNSS as first-class evidence, time sync, VIO, cross-source registration, dense MVS + fusion | DONE per ledger except two BLOCKED: P1.4 RGB-D device E2E (P1-03: contract verified on synthetic RGB-D, `tests/test_synthetic_rgbd.py`; no device) and P1.8 real VIO backend (P3-02: `tests/test_synthetic_vio.py`; no built backend). DONE: P1.5 (P1-01/02), P1.6 time sync (P2-01), P1.10 registration (P4-01, `tests/test_registration.py`), P1.11 dense MVS + fusion (P6-01/02, `tests/test_dense_*`, `tests/test_evidence_fusion.py`), P1.12/13 MeshData (P6-03) |
| P2.14–P2.17 | Perception + identity | Multi-view identity, tracking, structure, materials | P2.14 identity DONE (P7-01), P2.15 tracking DONE (P7-02, `tests/test_temporal_tracking.py`); P2.16 structure BLOCKED on real indoor data (P7-03: synthetic-verified `tests/test_synthetic_indoor.py`, known limits PT-06); P2.17 materials MISSING (no ledger task) |
| P3.18–P3.19 | Uncertainty + provenance | Covariance propagation; queryable lineage graph | DONE per ledger: P3.18 uncertainty (P10-02, `tests/test_uncertainty_propagation.py`), P3.19 provenance graph (P10-01, `tests/test_provenance_graph.py`) |
| P4.20–P4.21 | Persistent world | WorldStore (SQLite), incremental compilation | DONE per ledger: P4.20 WorldStore (P11-01; `worldstore/store.py`, `tests/test_world_store*.py`, `tests/test_detail_worldstore.py`), P4.21 incremental compilation (P12-01; `tests/test_incremental_world_e2e.py`, `tests/test_true_localized_incremental_acceptance.py`). Real-photo crash/restart/arbitration verified under PROD-01 |
| P6 | Reality Studio | Full client over WorldStore/WorldIR: evidence/provenance browsing, timeline, diff, measurement | IMPLEMENTED (no standalone ledger task): Next.js Studio in `frontend/` (evidence/provenance/per-version inspection, diff, status) backed by `apps/api`. Evidence: PROD-01 reconciliation 2026-10-02 (tsc 0, next build 0, contract tests, real-browser check), `tests/test_reality_studio_primary_journey.py`. Open: no Studio UI for uncertainty (`IMPLEMENTATION_STATUS.md`) |
| P7.25–P7.29 | Domains + infra | GIS, robotics, large-world, packaging verification, CI | P7.27 large world DONE (P13-01/02, `tests/test_spatial_tiling.py`, `tests/test_worldstore_tiles.py`); P7.28 packaging IMPLEMENTED (`tests/test_packaging_smoke.py`; clean-venv run, commit dc35a79); P7.29 CI IMPLEMENTED (`.github/workflows/ci.yml`, not executed in this review); P7.25 GIS / P7.26 robotics: only OPTIONAL_DEFERRED writers (P16-02, `tests/test_box_exporters.py`); city compiler P14-01 BLOCKED on real 3DCityDB |
| P8.30–P8.31 | Simulation | Physics → WorldIR write-back, advanced simulation | P8.30 generic world simulation DONE (P17-01, `tests/test_simulation_loop.py`; physics moved to `reality-engine-child` 2026-09-18); P8.31 advanced simulation deferred (no ledger task) |

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
