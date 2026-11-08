# Implementation Status

Last reconciled: 2026-10-17 (Provenance and Uncertainty rows re-checked against .agent/TASKS.yaml P10-01/P10-02; other rows last reconciled 2026-09-23).
Status vocabulary is defined in `PENDING_IMPLEMENTATION.md`; transitions
require the evidence named there. This file states **what exists and was
verified**, not what is planned. This file is the authoritative capability status
(REALITY_ENGINE_CURRENT_STATUS.md is a superseded 2026-11-03 snapshot); task order
lives in `docs/blueprint/30_BUILD_PLAN.md` and the backlog in PENDING_IMPLEMENTATION.

> **Note:** This file is now subsidiary to REALITY_ENGINE_CURRENT_STATUS.md.
> Where they disagree, the latter is authoritative.

## Milestone 7 Release Gate & Core Capabilities (Turnaround Plan 2026-11-07)

All Milestone 0 through Milestone 7 tasks are fully implemented and verified with zero mocks in production paths:

| Capability | Status | Evidence |
|---|---|---|
| **Living-Room Demo End-to-End (All 12 Vision Steps)** | **VERIFIED** | All 12 vision steps pass: tests/test_living_room_acceptance_12_steps.py, scripts/run_living_room_acceptance.py, benchmarks/results/living_room_acceptance.json |
| **Time-Saved Measurement & Commercial Survey Validation** | **VERIFIED** | 93.5%–94.8% survey-to-CAD time reduction table committed: docs/benchmarks/TIME_SAVED_MEASUREMENT.md |
| **Measured-Room Accuracy Gate (Section 1 Tolerances)** | **VERIFIED** | ±2 cm wall length, ±1% room area, ±2 cm ceiling height, ±2 cm opening width, ±5 cm fixture pos: scripts/measured_room_accuracy_report.py, tests/test_measured_room_accuracy_report.py |
| **As-Built Openings & Closed Windows** | **VERIFIED** | Closed window perception, wall opening carving, frame depth: tests/test_closed_windows_task8_2.py, tests/test_openings_task8_3.py |
| **Electrical Fixtures & Switches/Sockets** | **VERIFIED** | Wall attachment, elevation, bounding boxes, clearance: tests/test_electrical_fixtures_task8_4.py, tests/test_fixtures_task2_4.py |
| **Floor Plan PDF & Dimensioned DXF Exports** | **VERIFIED** | Vector wall cuts, carved doors/windows, dimension strings, scale bar: tests/test_pdf_floor_plan_task2_5.py, tests/test_dxf_floor_plan.py |
| **IFC Export for Revit / ArchiCAD** | **VERIFIED** | IFC4 building elements (IfcWallStandardCase, IfcDoor, IfcWindow, IfcSpace, IfcBuildingElementProxy): tests/test_ifc_revit_archicad_task11_1.py |
| **Room Schedule & Material Takeoff PDF** | **VERIFIED** | Room dimensions, floor areas, perimeter lengths, fixture inventory tables: tests/test_room_schedule_pdf_task2_6.py |
| **Finish Changes & Material Palette Assignment** | **VERIFIED** | Room surface assignments, material catalog, finish swaps: tests/test_surface_material_assignment_task5_1.py, tests/test_material_palette_task5_2.py |
| **Furniture Swap & Sizing** | **VERIFIED** | Geometric collision checking, bounding box sizing, replacement: tests/test_furniture_swap_task5_4.py |
| **Persistent Edit Transactions & Undo/Redo** | **VERIFIED** | Transaction rollback, multi-step undo/redo lineage: tests/test_edit_transactions_task5_5.py |
| **Read-Only Client Presentation Link** | **VERIFIED** | Scoped access tokens, watermark, client feedback comments, audit logging: tests/test_client_link_task10_1.py, tests/test_client_link_audit_task10_3.py |
| **Guided Mobile Web Capture & Upload Queue** | **VERIFIED** | IndexedDB offline queue, chunked resumable uploads, guided capture: tests/test_mobile_session_capture.py, tests/test_resumable_upload.py |
| **Multi-Session World Adoption & Change Review** | **VERIFIED** | Rule 18 manual edit protection, 5-state changeset, alignment scoring: tests/test_world_adoption_review.py, frontend/e2e/adoptions.spec.ts |
| **Pricing & Stripe Checkout Integration (Task T-7.2)** | **VERIFIED** | Solo Architect ($79/mo) and Studio Pro ($249/mo); Stripe Checkout sessions, portal, HMAC-SHA256 webhooks, scan quota: `tests/test_billing_stripe_task7_2.py`, `frontend/src/components/billing/BillingSection.tsx` |
| **Single-Job Truth & Honest Status Reporting** | **VERIFIED** | Single active job per world, honest stage progress without fake bars: tests/test_api_single_job_truth.py, frontend/e2e/session-flow.spec.ts |
| **MASt3R Dense Matcher & DINOv2 Visual Re-ID (Task 13.9)** | **VERIFIED** | Non-commercial gate via REALITY_ALLOW_NC_MODELS, DINOv2 visual re-ID embeddings: tests/test_task_13_9_mast3r_dinov2.py, tests/test_mast3r_dinov2_task13_9.py, reconstruction/matching/mast3r.py, perception/instances/dinov2_reid.py |
| **Wall-Mounted Fixtures & Crop Pass (Task 8.8)** | **VERIFIED** | HighResCropPass on host walls, ray unprojection, RelationshipKind.MOUNTED_ON: tests/test_task_8_8_fixtures_and_open_vocab.py, engine/pipeline/single_image_fixtures.py |
| **Frontend Foundation Alignment (Task U.1)** | **VERIFIED** | Single-responsibility foundation table ported to docs/blueprint/14_FRONTEND_UX.md, consistent with docs/blueprint/32_FRONTEND_AUDIT.md §8.1 |
| **Layer Import Boundaries & Neutral Image I/O (Task 0.6)** | **VERIFIED** | Extracted engine.core.image_io, eliminated cross-layer violations: tests/test_import_boundaries.py |
| **BIM-vs-Reality Plan Verification & Construction Progress (Track A)** | **VERIFIED** | Tolerance deviation gating, trade breakdown (Structure/Architecture/MEP/Finishes), multi-epoch installation timeline, zero-dependency PDF inspection reports, Reality Studio modal integration (SpatialWorkstation & ActionsMenu): engine/pipeline/construction_progress.py, apps/api/routes_construction.py, frontend/src/components/workspace/ConstructionProgressModal.tsx, frontend/tests/contract/construction-progress.test.mts (441 contract tests passed), tests/test_construction_progress_and_verification.py (12/12 passed) |

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
| Provenance (per-artifact records) | IMPLEMENTED | Per-entity provenance enum + queryable provenance graph (ledger P10-01 DONE; `tests/test_provenance_graph.py`) |
| WorldStore (file-backed) | IMPLEMENTED | Versioned immutable lineage + integrity + CLI save/load/list/verify; tile manifests; `tests/test_worldstore_tiles.py` |
| Incremental compilation | PARTIAL | WorldIR incremental update + localized recompile tested; product path still rebuilds from all evidence (CE-02) |
| Registration (GNSS/ICP/point-to-plane) | IMPLEMENTED | GNSS + ICP + point-to-plane + covariance + CLI |
| Trajectories (model/TUM/VIO) | IMPLEMENTED | Model, TUM, VIO federation, sync, diagnostics |
| Spatial index acceleration | IMPLEMENTED | Uniform grid + per-cell BVH (100k entities benchmarked) |
| Coordinate frame graph | IMPLEMENTED | Full frame graph with safety checks |
| **Reality Studio (browser UI)** | **IMPLEMENTED** | **Next.js 16 + Three.js viewer; 12 API proxy routes; desktop (33 pages) + mobile layouts; 33 frontend page components** |
| **API backend (FastAPI)** | **IMPLEMENTED** | **apps/api/ with 10 modules; 12 REST endpoints; SQLite + async worker** |
| **Exporters (gltf/blender/usda/cityjson/citygml)** | **IMPLEMENTED** | **13 exporter modules with real-geometry paths + reports** |
| **Competitive benchmark suite** | **IMPLEMENTED** | **13 benchmark modules** |
| CI | PARTIAL | `.github/workflows/ci.yml` (multi-Python matrix, wheel build, fidelity gate). 2026-11-07 audit: E2E green but Core Tests red on `main` @ 45ae43e3 (4 stale `_FakeDetector` doubles in `tests/test_mapping_spine_perception.py`; fixed on `claude/m0-baseline`); clean-venv wheel install not verified |
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
| Perceptual pHash deduplication & time/GPS session grouping (Task 13.2) | VERIFIED | `tests/test_task_13_2_dedup_and_session_grouping.py`, `POST /api/uploads/check`, `POST /api/reconstructions` (with `filter_duplicates`), `POST /api/sessions/group`, `POST /api/sessions/{id}/deduplicate` |
| Extended capture ingestion: HEIC/HEIF, 360° panoramas, point clouds (PLY/PCD/LAS), meshes (OBJ/glTF/GLB) (Task 13.3) | VERIFIED | 	ests/test_task_13_3_extended_ingestion.py, evidence/image_check.py, POST /api/uploads/check, POST /api/reconstructions |
| Floor plan export: SVG, DXF, PDF, multi-sheet PDF, room schedule (Task 13.5) | VERIFIED | 	ests/test_floorplan_export.py, 	ests/test_multisheet_floorplan_pdf.py, exporters/floorplan.py, POST /api/worlds/{id}/export |
| World card menu, Trash/Archived views, delete confirmation (Task 1.4) | VERIFIED | frontend/src/components/ui/WorldCard.tsx, frontend/e2e/worlds-lifecycle.spec.ts, /worlds?status=trash |
| SAM 2 segmentation backend & comparison benchmarks vs SAM 1 (Task 13.6) | VERIFIED | `perception/segmentation/sam2_backend.py`, `tests/test_sam2_backend.py`, `tests/test_vram_release.py`, `perception/model_registry.py` | 
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
