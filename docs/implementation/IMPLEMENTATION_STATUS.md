# Implementation Status

Last reconciled: 2026-09-14 (mapping campaign; PRs #14, #16, #17, #18, #19).
Status vocabulary is defined in `PENDING_IMPLEMENTATION.md`; transitions
require the evidence named there. This file states **what exists and was
verified**, not what is planned — see ROADMAP for sequence and
PENDING_IMPLEMENTATION for the backlog.

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
| CLI (session/source/orchestrator) | PARTIAL | orchestrator→compile only |
| Deterministic compiler (hash-stable outputs) | IMPLEMENTED | artifact hashes stable across runs |
| Multi-source session manifest | FOUNDATION | manifest schema; no cross-source alignment |
| Provenance (per-artifact records) | FOUNDATION | no queryable graph |
| Uncertainty model | MISSING | ad hoc per-stage stats only |
| WorldStore / incremental compilation | MISSING | per-run JSON exports only |
| CI | MISSING | no workflows in repo (verified 2026-09-14) |

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
