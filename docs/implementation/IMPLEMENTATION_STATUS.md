# Implementation Status

Last reconciled: 2026-09-23 (cross-checked against REALITY_ENGINE_CURRENT_STATUS.md and .agent/TASKS.yaml).
Status vocabulary is defined in `PENDING_IMPLEMENTATION.md`; transitions
require the evidence named there. This file states **what exists and was
verified**, not what is planned — see REALITY_ENGINE_CURRENT_STATUS.md for
the authoritative status and PENDING_IMPLEMENTATION for the backlog.

> **Note:** This file is now subsidiary to REALITY_ENGINE_CURRENT_STATUS.md.
> Where they disagree, the latter is authoritative.

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
| Viewer mesh layer | IMPLEMENTED | build_viewer + main.js mesh layer |
| CLI `reality compile` (one-command path) | IMPLEMENTED | `test_cli_compile.py` fake-backend E2E |
| CLI (session/source/orchestrator) | IMPLEMENTED | Full CLI with ingest/session/compile/reconstruct/validate/diff/export/register/query/store/inspect/viewer |
| Deterministic compiler (hash-stable outputs) | IMPLEMENTED | artifact hashes stable across runs |
| Multi-source session manifest | IMPLEMENTED | MultiSourceSession + full CLI |
| Provenance (per-artifact records) | IMPLEMENTED | Per-entity provenance enum; queryable graph not yet built |
| WorldStore / incremental compilation | IMPLEMENTED | Versioned immutable lineage + integrity + CLI save/load/list/verify |
| Registration (GNSS/ICP/point-to-plane) | IMPLEMENTED | GNSS + ICP + point-to-plane + covariance + CLI |
| Trajectories (model/TUM/VIO) | IMPLEMENTED | Model, TUM, VIO federation, sync, diagnostics |
| Spatial index acceleration | IMPLEMENTED | Uniform grid + per-cell BVH (100k entities benchmarked) |
| Coordinate frame graph | IMPLEMENTED | Full frame graph with safety checks |
| **Reality Studio (browser UI)** | **IMPLEMENTED** | **Next.js 16 + Three.js viewer; 12 API proxy routes; desktop (33 pages) + mobile layouts; 33 frontend page components** |
| **API backend (FastAPI)** | **IMPLEMENTED** | **apps/api/ with 10 modules; 12 REST endpoints; SQLite + async worker** |
| **Exporters (gltf/blender/usda/cityjson/citygml)** | **IMPLEMENTED** | **13 exporter modules with real-geometry paths + reports** |
| **Competitive benchmark suite** | **IMPLEMENTED** | **13 benchmark modules** |
| CI | **IMPLEMENTED** | **.github/workflows/ci.yml — multi-Python matrix, wheel build, 28+ test files** |
| Uncertainty propagation | PARTIAL | Schema fields exist; first-order propagation not implemented |
| **Uncertainty visualization in Studio** | **MISSING** | **No Studio UI for uncertainty** |

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
