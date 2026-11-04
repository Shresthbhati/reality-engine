# Implementation Status

Last reconciled: 2026-10-17 (Provenance and Uncertainty rows re-checked against .agent/TASKS.yaml P10-01/P10-02; other rows last reconciled 2026-09-23).
Status vocabulary is defined in `PENDING_IMPLEMENTATION.md`; transitions
require the evidence named there. This file states **what exists and was
verified**, not what is planned — see REALITY_ENGINE_CURRENT_STATUS.md for
the authoritative status and PENDING_IMPLEMENTATION for the backlog.

> **Note:** This file is now subsidiary to REALITY_ENGINE_CURRENT_STATUS.md.
> Where they disagree, the latter is authoritative.

## Baseline (recorded 2026-11-04, task 0.5)

Backend pytest suite (excluding tests/integration and tests/scenarios):
- **4350 passed, 38 failed, 10 skipped, 5 xfailed, 17 errors** in 27m40s (1660.02s)
- 17 errors are pre-existing `test_frontend_bridge_proxies_a_real_api_contract` setup failures (environment, not code)
- 38 failures are pre-existing (missing ML weights, torch-hub cache, stale expectations)
- Frontend build: **SUCCESS** (`npm --prefix frontend run build`, `.next/` produced)
- One-command launcher: **VERIFIED** (`launch_ui.bat --no-browser`, `/api/health` returns `{"status":"ok"}`)

## Canonical pipeline (vertical slice)

| Stage | Component | Status | Evidence |
|---|---|---|---|
| 1. Ingest | `evidence/ingest.py` | IMPLEMENTED | unit + E2E on real capture |
| 2. Reconstruction | `reconstruction/backend.py` (COLMAP) | IMPLEMENTED | 25/28 cameras, real capture |
| 2b. Backend seam | `VerticalSliceOptions.reconstruction_backend` | IMPLEMENTED | deterministic fake-backend CLI test |
| 3. Scale | `reconstruction/scale_estimator.py` | IMPLEMENTED | metric room capture (15.8 m extent) |
| 4. Frames | `evidence/build_frames.py` | IMPLEMENTED | suite |
| 5. Depth | `engine/pipeline/depth.py` (MiDaS) | IMPLEMENTED | real E2E; torch-hub env fragility noted |
| 5b. Depth→points | `reconstruction/depth_to_points.py` | IMPLEMENTED | deterministic unprojection tests |
| 5c. Fusion | `reconstruction/fusion/` | IMPLEMENTED | 121k-point fused cloud, real capture |
| 6. Meshing | `reconstruction/meshing/` | IMPLEMENTED | real Poisson mesh: 240k verts/482k faces; envelope filter cut 43k outlier pts |
| 7. Perception | `perception/` (SAM/SigLIP) | PARTIAL | per-image detections only; env-dependent |
| 8. Compile | `engine/pipeline/vertical_slice.py` → WorldIR | IMPLEMENTED | WorldIR 71 entities, metric |
| 9. Export | `exporters/` glTF/Blender/USD | IMPLEMENTED | real glTF: TRIANGLES mesh + POINTS entities, uint32 indices |

## Cross-cutting

| Capability | Status | Evidence |
|---|---|---|
| ArtifactStore (sha256, content-addressed) | IMPLEMENTED | roundtrip + hash tests |
| Mesh artifacts in WorldIR (MESH type) | IMPLEMENTED | pipeline writes `Geometry(type=MESH,data_uri,data_hash)` |
| glTF MESH payload export | IMPLEMENTED | real-data export verified |
| World package export/import | IMPLEMENTED | Versioned .zip package with checksum validation; export→import yields identical world (`tests/test_world_package_export_import_task11_2.py`, 4 passed) |
| Resumable chunked uploads | IMPLEMENTED | POST /api/uploads/resumable/{start,chunk,finalize}; resume from last acknowledged offset; content validated on finalize (`tests/test_resumable_upload.py`, 4 passed) |
| Viewer mesh layer | IMPLEMENTED | build_viewer + main.js mesh layer |
| CLI `reality compile` (one-command path) | IMPLEMENTED | `test_cli_compile.py` fake-backend E2E |
| CLI (session/source/orchestrator) | IMPLEMENTED | Full CLI with ingest/session/compile/reconstruct/validate/diff/export/register/query/store/inspect/viewer |
| Deterministic compiler (hash-stable outputs) | IMPLEMENTED | artifact hashes stable across runs |
| Multi-source session manifest | IMPLEMENTED | MultiSourceSession + full CLI |
<<<<<<< HEAD
| Provenance (per-artifact records) | IMPLEMENTED | Per-entity provenance enum + queryable provenance graph (ledger P10-01 DONE; `tests/test_provenance_graph.py`) |
| WorldStore / incremental compilation | IMPLEMENTED | Versioned immutable lineage + integrity + CLI save/load/list/verify |
=======
| Provenance (per-artifact records) | IMPLEMENTED | Per-entity provenance enum; queryable graph not yet built |
| WorldStore (file-backed) | IMPLEMENTED | Versioned immutable lineage + integrity + CLI save/load/list/verify; `tests/test_worldstore_tiles.py` |
| Incremental compilation | PARTIAL | WorldIR incremental update + localized recompile tested; product path still rebuilds from all evidence (CE-02) |
>>>>>>> 4027b5c
| Registration (GNSS/ICP/point-to-plane) | IMPLEMENTED | GNSS + ICP + point-to-plane + covariance + CLI |
| Trajectories (model/TUM/VIO) | IMPLEMENTED | Model, TUM, VIO federation, sync, diagnostics |
| Spatial index acceleration | IMPLEMENTED | Uniform grid + per-cell BVH (100k entities benchmarked) |
| Coordinate frame graph | IMPLEMENTED | Full frame graph with safety checks |
| **Reality Studio (browser UI)** | **IMPLEMENTED** | **Next.js 16 + Three.js viewer; 12 API proxy routes; desktop (33 pages) + mobile layouts; 33 frontend page components** |
| **API backend (FastAPI)** | **IMPLEMENTED** | **apps/api/ with 10 modules; 12 REST endpoints; SQLite + async worker** |
| **Exporters (gltf/blender/usda/cityjson/citygml)** | **IMPLEMENTED** | **13 exporter modules with real-geometry paths + reports** |
| **Competitive benchmark suite** | **IMPLEMENTED** | **13 benchmark modules** |
<<<<<<< HEAD
| CI | **IMPLEMENTED** | **.github/workflows/ci.yml — multi-Python matrix, wheel build, 28+ test files** |
| Uncertainty propagation | IMPLEMENTED | Ledger P10-02 DONE; `tests/test_uncertainty_propagation.py` |
| Uncertainty visualization in Studio | PARTIAL | Inspector "Uncertainty Model" panel (`frontend/src/components/inspector/AdaptiveInspector.tsx`) shows entity confidence (value, % and bar), the engine's `uncertainty` record (confidence/note), a "Low confidence" marker (< 0.5, header chip + panel), and an honest empty state. Data comes from the existing world API (`Entity.to_dict` already emits `confidence` and `uncertainty`; no API change). Spatial sigma/covariance shows only if `custom_properties.uncertainty_sigma_m` exists - no engine writer emits it yet, so that row is empty in practice. Not covered: covariance ellipsoids in the 3D viewport, propagated `Uncertain` quantities. Verified by tsc/lint/build/contract test only; not visually inspected against a live world. |

## Room reconstruction fidelity (Phase 7, 2026-10-31)

Plan: `docs/superpowers/plans/2026-10-30-room-reconstruction-fidelity.md`.
Baseline: `docs/reconstruction/ROOM_FIDELITY_BASELINE.md` (synthetic fixture only).

| Capability | Status | Verification |
|---|---|---|
| Capture quality checks, coverage map, hints (F1) | IMPLEMENTED | `tests/test_capture_quality.py`, progressive guidance units |
| Depth sidecars + metric scale from device depth (F2) | IMPLEMENTED (synthetic RGB-D) | `tests/test_depth_scale.py`, `tests/test_synthetic_rgbd.py`; physical device pending |
| CPU plane-sweep dense backend (F3) | PARTIAL | `tests/test_plane_sweep.py` (known depth, textureless refusal, pipeline stage); real-room coverage not measured |
| Single-image layout evidence (F4) | PARTIAL / BLOCKED on model choice | `tests/test_layout_evidence.py`; projection + fusion implemented, learned estimator not shipped |
| Lighting-normalised surface colour (F5) | IMPLEMENTED | `tests/test_surface_appearance.py` |
| User corrections as evidence (F6) | IMPLEMENTED | `tests/test_corrections.py` |
| Room joining through shared openings (F7) | IMPLEMENTED | `tests/test_room_join.py`, `tests/test_room_fidelity_fixture.py` |
| Floor plan SVG/DXF + room report incl. PDF (F8) | IMPLEMENTED (synthetic worlds) | `tests/test_floorplan_export.py`; no tape-measure comparison on a real room |
| Fidelity scorecard CI gate (F9) | IMPLEMENTED | `.github/workflows/ci.yml` + `scripts/room_fidelity_report.py --fail-on-regression` |
| Progressive preview + per-stage timings (F10) | PARTIAL | timings in progressive attempts and scorecard; no target until real-capture baseline exists |

### Structure / dimensions / appearance / viewer / UX (S, D, O, M, V, U; 2026-10-31)

SYNTHETIC FIXTURES ONLY. No real-room photos or tape measurements exist; nothing here is VERIFIED and nothing is
browser-verified. Backlog rows with remaining work: `PENDING_IMPLEMENTATION.md` ("Room fidelity plan items").

| Capability | Status | Verification |
|---|---|---|
| Phase 0 baseline + `fidelity-gate` CI job | PARTIAL | `scripts/room_fidelity_report.py`, `docs/reconstruction/ROOM_FIDELITY_BASELINE.md`; synthetic regressions only |
| Solid wall slabs, carved openings, Manhattan snap, inferred closing walls (S1-S4) | PARTIAL | `perception/architecture/wall_slabs.py`, `tests/test_wall_entities.py`; thickness is the labelled 0.10 m default |
| Dimension states MEASURED/RECONSTRUCTED/ESTIMATED/VERIFIED (D1) | PARTIAL | `world_ir/measure.py`, `tests/test_dimension_state.py` |
| Known-length scale + door-based scale suggestion, user accepts (D2/D2b) | PARTIAL | `tests/test_scale_suggestion.py`; accept path never run end to end |
| Dimension consistency + gates 2 % L/W/H, 6 cm openings (D3/D4) | EXPERIMENTAL | `tests/test_dimension_gates.py` (synthetic ground truth; real-room variant SKIPPED) |
| Learned door/window/furniture detection (O1/O2) | BLOCKED | detector choice needs the user; geometry-only opening host stays |
| Surface colour + texture atlas (M1/M2), material hypotheses (M3) | PARTIAL / EXPERIMENTAL | `tests/test_room_surfaces.py`, `tests/test_material_hypotheses.py`; hypotheses ESTIMATED, `asserted=False`, confidence <= 0.4, not wired to physics |
| Viewer room shell + atlas texture + evidence summary (V1/V2) | EXPERIMENTAL | frontend 171/171 contract tests, `tsc --noEmit` exit 0, `tests/test_api_world_artifacts.py`; no browser run |
| Capture guidance + guided flow (U1-U3) | PARTIAL | `flow-model.ts`; steps 4/5 always "needs action"; coverage routes never run on a stored world |
| CPU plane sweep exposed as API options (F3) | PARTIAL | `tests/test_api_dense_cpu.py` (8); no real job run with `dense_cpu=true` |
| Site composer (manual world placement on a metric grid) | FOUNDATION | being built; not verified; NOT automatic registration |
| Landmark / curved-structure reconstruction from one image | MISSING | planar rooms only; see PENDING_IMPLEMENTATION LAND-1 |

Observed this campaign: 44 tests in the freebuff-owned files; frontend 171/171 contract tests and `tsc --noEmit` exit 0;
19 tests in `test_api_dense_cpu.py` + `test_api_world_artifacts.py`; a backend agent reported 203 passed / 1 skipped / 0
failed over its suites. A clean FULL python suite result is NOT yet available (a run is in progress).

Last verified locally (2026-10-31): focused fidelity suite 47 passed;
compiler/geometry/multistorey regression tests 80 passed; broader
`-k` subset 779 passed, 8 pre-existing failures fixed by updating stale
expectations (slab entities are additive, room boundary is an inferred
BOX, glTF non-cube meshes may be POINTS or TRIANGLES); frontend
`tsc --noEmit` clean; frontend contract tests pass.
=======
| CI | PARTIAL | `.github/workflows/ci.yml` exists (matrix, targeted pytest, ruff/mypy non-blocking); green GitHub run and clean-venv wheel install not verified |
| Uncertainty propagation | PARTIAL | Schema fields exist; first-order propagation not implemented |
| **Uncertainty visualization in Studio** | **MISSING** | **No Studio UI for uncertainty** |
>>>>>>> 4027b5c

## Known environment constraints (this machine)

- COLMAP 4.2.0 **CPU-only**: `patch_match_stereo` (dense MVS) blocked;
  poisson/delaunay meshing work (used by stage 6).
- No open3d/trimesh/skimage installed; meshing stack is
  scipy/subprocess-based by design.
- SAM torch-hub cache failure is a **pre-existing environment
  breakage** (1 failing test when perception tests included); not
  caused by current work, tracked in PENDING_IMPLEMENTATION.

## Not implemented (intentionally)

See `PENDING_IMPLEMENTATION.md` for the authoritative backlog with
per-item evidence requirements. Deferred domains (GIS, robotics,
large-world, advanced physics) have specs under `docs/future/`.
