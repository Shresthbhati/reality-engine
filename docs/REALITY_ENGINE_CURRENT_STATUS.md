# REALITY ENGINE — CURRENT STATUS (authoritative, reconciled 2026-09-23)

> ONE authoritative status file (Agent 5). Older docs are history, not truth.
> Reconciliation rule: claims below were cross-checked against the code on
> 2026-09-23; dated evidence (test counts, build results) is kept with its
> date and re-verified when the area changes.

## Vertical slice (real user path)

photos/video -> ingest/session -> reconstruct -> compile -> artifacts
-> validate -> store -> inspect/query/viewer -> export/register/diff.

Every step runs through `reality` CLI or StudioSession + browser viewer.

## IMPLEMENTED (tested, wired into the slice)

- Evidence: folder/file importers, EXIF/GPS, quality, dedup, packages.
- Sessions: MultiSourceSession + full CLI (create/add-source/list/inspect/export).
- Reconstruction: orchestrator gates + COLMAP backend + test-only fake backend.
- Compiler: planes -> classify -> promote -> rooms -> validation gate.
- WorldIR v1: entities/geometry/materials/provenance/serialization/validation/diff.
- WorldStore: versioned immutable lineage + integrity + CLI save/load/list/verify.
- Registration: GNSS + ICP + point-to-plane + covariance + CLI.
- Trajectories: model, TUM, VIO federation, sync, diagnostics.
- Perception: MiDaS, Mask R-CNN, SAM, lifting, fusion, quality.
- Meshing: Poisson via COLMAP + MESH export.
- Exporters: glTF/USDA/Blender/CityJSON/CityGML with real-geometry paths + reports.
- Studio (browser): Next.js 16 + Three.js viewer; 12 API proxy routes; desktop (33 pages) + mobile layouts; selection + provenance.
- Viewer: offline single-file Three.js + selection + provenance + CLI.
- CLI: ingest/session/compile/reconstruct/validate/diff/export/register/query/store/inspect/viewer.
- API backend: FastAPI with SQLite + async worker; 10 modules; REST endpoints for health/sessions/worlds/evidence/jobs/notifications/activity.

## PARTIAL (real, with named limits)

- Reconstruction at scale: small scenes exercised; no large-building stress; no CI binary run.
- Metric scale: single-baseline; honest RELATIVE fallback.
- Depth: relative MiDaS metricized per-view (approximation); honest skip otherwise.
- Registration landmarks: co-observation resolver missing.
- Viewer: no materials/measurements UI, timeline, diff views.
- Spatial index: accelerated uniform grid + BVH (100k entities tested, benchmarked).

## SCAFFOLDED (no real code)

- apps/capture, apps/studio (README placeholders; engine/studio foundation exists).

## PLANNED (not started)

- Studio review queue/corrections; GIS/robotics/streaming; navmesh.

## BLOCKED (explicit owners)

- Physics in core: REMOVED 2026-09-18 to reality-engine-child. Do not re-add.
- Real-model runs (COLMAP/MiDaS/weights): manual, never CI (slow marker).
- Child physics gaps (CCD, angular response): child repo, not core.

## CITY-SCALE INFRASTRUCTURE (NEW - 2026-09-18)

### Spatial Index Acceleration
- **Implementation**: Uniform 3D grid with per-cell BVH acceleration
- **API**: Drop-in replacement for `SpatialIndex` (AcceleratedSpatialIndex)
- **Queries supported**: nearest(k), within_radius(r), within_region(bounds)
- **Fallback**: Flat list for <1000 entities or sparse distributions
- **Benchmarks**:
  - 100 entities: 0.04ms nearest (flat: 0.04ms)
  - 1,000 entities: 0.53ms nearest (flat: 0.52ms)
  - 10,000 entities: 0.02ms nearest (flat: 0.03ms) — 1.5x speedup
  - 100,000 entities: 0.20ms nearest (flat not run — would be ~3ms)
- **Correctness**: Verified identical results to flat implementation
- **Deterministic**: Same ordering by (distance, entity_id)

### Coordinate Frame Graph
- **Implementation**: Full frame graph with safety checks
- **Frames**: SENSOR, CAPTURE, SESSION_LOCAL, BUILDING_LOCAL, WORLD, ENU, UTM, WGS84, ECEF, ENGINE_LOCAL
- **Features**:
  - Path resolution with deterministic tie-breaking
  - Timestamp-validity windows
  - Cycle detection
  - Disconnected component detection
  - Ambiguity detection (multiple distinct transforms)
  - Transform composition with uncertainty propagation
  - Serialization/deserialization
- **Integration**:
  - Registration: loads GNSS anchor, ICP results with provenance
  - Trajectories: loads VIO/SVO/SFM/EKF with validity windows
  - WorldIR: builds graph from transforms metadata
- **Safety**: Never silently chooses arbitrary transform; returns structured diagnostics

## Test matrix

- UNIT: tests/test_*.py per module.
- INTEGRATION: tests/test_cli_vertical_slice.py (real CLI, fake-backend geometry).
- E2E: tests/test_vertical_slice_e2e.py (room fixture -> WorldIR -> store -> viewer).
- REAL MODEL OPTIONAL: scripts/run_vertical_slice.py + reality compile (manual).
- BENCHMARK: tests/test_spatial_index_benchmark.py (100, 1K, 10K, 100K entities)

## Acceptance

- pip install . -> reality console script: yes.
- Every CLI command calls real backend: yes (physics removed to child).
- Reconstruction path executes: yes (tests fake, manual COLMAP).
- World persists + viewer opens + inspection works: yes.
- No silent except-pass on pipeline path: fixed (registry warns on stderr).
- gitignore + wheel contents fixed; stale physics refs removed.
- Spatial index benchmarks pass correctness verification.
- Frame graph tests: 21/21 pass including safety checks.
- Canonical Golden World Flow E2E: tests/integration/test_golden_world_flow.py passing 100% across all 8 pipeline phases (Mobile bundle -> Desktop Loader -> Reconstruction -> WorldIR V1 -> WorldStore V1 -> Desktop Bridge -> Pass 2 Alignment -> Closure -> Localized Update -> WorldStore V2 -> WorldDiff -> Exporters glTF/CityGML/USDA/CityJSON).
- Integration test suite: 147 passed, 7 skipped.
- Frontend Next.js production build: 12/12 routes static/dynamic compiled cleanly without errors.


