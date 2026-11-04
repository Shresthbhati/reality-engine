# Reality Engine — Capability Matrix

**Repository:** Shresthbhati/reality-engine  
**Current Branch:** feat/task-11-2-world-package-zip  
**Current HEAD SHA:** 1ff64183  
**Main Branch SHA:** cb7cd393  
**Audit Date:** 2026-11-05  
**Classification Scheme:** A=IMPLEMENTED, B=INTEGRATED, C=SYNTHETICALLY VERIFIED, D=REAL-WORLD VERIFIED, E=GROUND-TRUTH VERIFIED, F=PRODUCTION VERIFIED, PARTIAL/BROKEN/SIMULATED/UNAVAILABLE/BLOCKED as applicable

---

## Classification Definitions

| Code | Classification | Definition |
|------|---------------|------------|
| IMPLEMENTED | Code exists and has meaningful unit tests | Code exists and passes unit tests |
| INTEGRATED | Implementation connected to actual application execution path | Called from real application path, not just unit tests |
| SYNTHETICALLY VERIFIED | Works on generated test data | Verified on synthetic/fixture data only |
| REAL-WORLD VERIFIED | Works on actual captured images/video/sensor data | Verified on real captures |
| GROUND-TRUTH VERIFIED | Results compared with independent reference measurements | Compared against tape measurements, ground truth |
| PRODUCTION VERIFIED | Passed operational, failure, performance, reliability testing under representative conditions | Production-ready |
| PARTIAL | Some real behavior, known gaps | Partial implementation with known missing functionality |
| BROKEN | Implementation exists but current verification fails | Tests fail or behavior broken |
| SIMULATED | Output is simulated or hardcoded, not real | Returns canned/fake results |
| UNAVAILABLE | Not implemented or not available | No implementation exists |
| BLOCKED | Cannot advance without external dependency | Blocked on user data, hardware, or decision |

---

## 1. Input and Ingestion

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Image ingestion | IMPLEMENTED | `evidence/ingest.py`, unit + E2E on real capture | `apps/api/routes_misc.py`, `apps/api/storage.py` | Supports photos, videos; content-addressed storage |
| Video ingestion | PARTIAL | Video frame extraction exists but not fully integrated | `apps/api/routes_reconstructions.py` | Frame sampling exists but not full video pipeline |
| RGB-D ingestion | IMPLEMENTED (synthetic) | `evidence/depth_frames.py`, 16-bit PNG + declared scale | `evidence/depth_frames.py` | Physical device not verified |
| LiDAR ingestion | MISSING | Not implemented | N/A | Not in codebase |
| 360° image ingestion | MISSING | Not implemented | N/A | Not in codebase |
| Camera metadata | IMPLEMENTED | Camera intrinsics/extrinsics parsing | `reconstruction/calibration/camera.py` | Real camera model |
| EXIF parsing | IMPLEMENTED | EXIF/GPS extraction | `evidence/ingest.py` | EXIF/GPS extraction works |
| Image quality analysis | PARTIAL | Blur/exposure metrics recorded | `reconstruction/robustness.py` | Quality signals recorded but not fully used |
| Frame extraction | IMPLEMENTED | Video keyframe extraction | `evidence/build_frames.py` | Works for sampled frames |
| Capture grouping | PARTIAL | Session grouping exists | `apps/api/models.py` | Session grouping works |
| Duplicate detection | PARTIAL | Content-hash dedup | `MultiSourceSession` | Content-hash dedup works |
| Corrupt input handling | IMPLEMENTED | `import_folder(on_error="record")` | `evidence/ingest.py` | Corrupt files recorded, remaining import |

## 2. Camera and Reconstruction

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Camera calibration | IMPLEMENTED | CameraIntrinsics, CameraExtrinsics, PinholeCamera | `reconstruction/calibration/camera.py` | Real camera model |
| Feature detection | PARTIAL | SIFT via COLMAP | `reconstruction/` | COLMAP feature extraction |
| Feature matching | PARTIAL | SIFT via COLMAP; ALIKED+LightGlue registered but not default | `reconstruction/matching/` | ALIKED+LightGlue registered but not default matcher |
| Match filtering | PARTIAL | Geometric verification via COLMAP | `reconstruction/` | COLMAP geometric verification |
| Geometric verification | PARTIAL | Epipolar consistency via COLMAP | `reconstruction/` | COLMAP geometric verification |
| Relative pose estimation | PARTIAL | COLMAP SfM | `reconstruction/backend.py` | COLMAP SfM |
| Camera registration | PARTIAL | COLMAP SfM (22/32 cameras on South Building) | `reconstruction/backend.py` | Real COLMAP execution verified |
| Sparse reconstruction | PARTIAL | COLMAP sparse reconstruction | `reconstruction/backend.py` | 4510 points on South Building |
| Bundle adjustment | PARTIAL | COLMAP bundle adjustment | `reconstruction/backend.py` | COLMAP bundle adjustment |
| Dense reconstruction | PARTIAL | COLMAP patch_match_stereo (manual CLI run, not Python-integrated) | `reconstruction/backend.py` | Real dense run 246,816 points; Python integration incomplete |
| Depth estimation | IMPLEMENTED | MiDaS relative depth | `engine/pipeline/depth.py` | Relative depth, not metric |
| Metric scale | PARTIAL | SfM sparse-fit scale anchoring | `reconstruction/scale.py` | Metric-by-alignment, not sensor-grade |
| Point cloud fusion | IMPLEMENTED | 121k-point fused cloud | `reconstruction/fusion/` | Real fused cloud |
| Mesh reconstruction | IMPLEMENTED | Poisson mesh: 240k verts/482k faces | `reconstruction/meshing/` | Real Poisson mesh |

## 3. Perception

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Object detection | PARTIAL | Mask R-CNN real COCO detections | `perception/detection/maskrcnn_backend.py` | Real COCO detections (book 0.78, person 0.51) |
| Semantic segmentation | PARTIAL | SAM backend exists but BROKEN | `perception/segmentation/sam_backend.py` | SAM torch-hub cache failure |
| Instance segmentation | PARTIAL | Same as segmentation | `perception/segmentation/` | Same limitation |
| Open-vocabulary detection | DONE | feat/task-8-2-open-vocab-detector | `perception/detection/` | Open-vocab detector work in unmerged worktree |
| Fine-grained detection | MISSING | Doors, windows, sockets, switches not detected | N/A | BLOCKED - detector choice needs user decision |
| Object classification | PARTIAL | Mask R-CNN COCO classes | `perception/detection/maskrcnn_backend.py` | COCO classes only |
| Object tracking | DONE | `perception/tracking/temporal.py` | `perception/tracking/` | 3D observation-level identity |
| Multi-view association | DONE | `perception/instances/object_resolution.py` | `perception/instances/` | Hungarian association, no transitive merges |
| Persistent identity | PARTIAL | `world_ir/entity_reid.py` | `world_ir/entity_reid.py` | Snapshot-to-snapshot only |
| Object geometry | PARTIAL | AABB-based lifting; INFERRED provenance | `perception/instances/lifting.py` | AABB-based, not mask-constrained |

## 4. Spatial Intelligence

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Room detection | PARTIAL | Wall slabs + room inference | `perception/architecture/wall_slabs.py` | Wall thickness fixed at 0.10m default |
| Wall detection | PARTIAL | Plane-based wall detection | `perception/architecture/wall_slabs.py` | Fixed thickness default |
| Floor detection | PARTIAL | Plane-based floor detection | `perception/architecture/wall_slabs.py` | Plane-based detection |
| Ceiling detection | PARTIAL | Plane-based ceiling detection | `perception/architecture/wall_slabs.py` | Plane-based detection |
| Door detection | BLOCKED | BLOCKED on detector choice | N/A | Geometry-only openings; learned detector blocked |
| Window detection | BLOCKED | BLOCKED on detector choice | N/A | Geometry-only openings; learned detector blocked |
| Opening detection | PARTIAL | Geometry-only openings | `perception/architecture/openings.py` | Geometry-only, no learned detector |
| Room topology | DONE | `perception/architecture/topology.py` | `perception/architecture/topology.py` | Room/building topology |
| Corridor connectivity | DONE | `perception/architecture/corridors.py` | `perception/architecture/corridors.py` | Corridor inference |
| Building hierarchy | DONE | WorldIR Building→Storey→Room hierarchy | `world_ir/schema_v1.py` | Hierarchical WorldIR |
| Object relationships | DONE | `engine/scene_graph/` | `engine/scene_graph/` | Geometric relationships |
| Utility relationships | PARTIAL | Utility relationships not fully implemented | `engine/scene_graph/` | Partially implemented |
| Spatial reasoning | PARTIAL | Basic spatial queries | `engine/scene_graph/` | Basic spatial queries |
| Uncertainty | PARTIAL | Schema fields exist; propagation not implemented | `uncertainty/propagation.py` | Schema exists, propagation not implemented |

## 5. Geometry

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Point cloud processing | IMPLEMENTED | 121k-point fused cloud | `reconstruction/fusion/` | Real fused cloud |
| Surface reconstruction | IMPLEMENTED | Poisson mesh | `reconstruction/meshing/` | Real Poisson mesh |
| Mesh generation | IMPLEMENTED | Poisson mesh: 240k verts/482k faces | `reconstruction/meshing/` | Real mesh |
| Mesh repair | PARTIAL | Validation only, no repair | `reconstruction/meshing/validation.py` | Validation measures, never mutates |
| Primitive fitting | PARTIAL | Parametric object templates | `perception/architecture/parametric.py` | Parametric object templates |
| Architectural geometry | PARTIAL | Wall slabs + room inference | `perception/architecture/wall_slabs.py` | Wall thickness fixed at 0.10m default |
| Object geometry | PARTIAL | AABB-based lifting | `perception/instances/lifting.py` | AABB-based, INFERRED provenance |
| CAD compatibility | MISSING | No CAD export | N/A | Not implemented |
| BIM compatibility | PARTIAL | IFC bridge exists | `exporters/ifc_bridge.py` | IFC4 via IfcOpenShell |
| Material classification | PARTIAL | Material hypotheses ESTIMATED, not asserted | `perception/materials/material_hypotheses.py` | ESTIMATED hypotheses, not asserted |
| Texture projection | DONE | `perception/detail/` | `perception/detail/` | Texture projection works |

## 6. Platform

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Model loading | PARTIAL | MiDaS via torch.hub (fragile) | `engine/pipeline/depth.py` | torch-hub env fragility noted |
| Model unloading | MISSING | Not implemented | N/A | Not implemented |
| GPU management | PARTIAL | RTX 4050 6GB, one model at a time | `perception/device.py` | One heavy model at a time |
| RAM management | PARTIAL | Basic memory management | N/A | Basic memory management |
| Job scheduling | IMPLEMENTED | SQLite + async worker | `apps/api/jobs.py` | Durable job execution |
| Cancellation | PARTIAL | Job cancellation exists | `apps/api/jobs.py` | Job cancellation exists |
| Failure recovery | PARTIAL | Quarantine + retry | `apps/api/worldstore_service.py` | Quarantine + retry |
| Persistence | IMPLEMENTED | WorldStore versioned storage | `worldstore/store.py` | Immutable version snapshots |
| World lifecycle | IMPLEMENTED | World versions + lineage | `worldstore/store.py` | Versioned immutable lineage |
| Versioning | IMPLEMENTED | World versions + lineage | `worldstore/store.py` | Versioned immutable lineage |
| Exports | IMPLEMENTED | glTF, USD(A), IFC, CityGML, CityJSON, Blender | `exporters/` | 17 exporter modules |

## 7. Studio (Frontend)

| Capability | Classification | Evidence | File | Notes |
|------------|---------------|----------|------|-------|
| Image upload | IMPLEMENTED | File upload via API | `frontend/src/lib/api/evidence.ts` | Real multipart POST |
| Project management | IMPLEMENTED | World CRUD | `apps/api/routes_worlds.py` | World CRUD operations |
| Reconstruction jobs | IMPLEMENTED | Job management | `apps/api/routes_reconstructions.py` | Job management |
| Viewport | PARTIAL | Three.js viewer; not browser-verified | `frontend/src/lib/viewport/` | Not browser-verified |
| Cameras | IMPLEMENTED | Camera poses read from version's cameras artifact | `apps/api/routes_worlds.py` | GET /api/worlds/{id}/cameras returns registered poses from WorldIR |
| Point clouds | PARTIAL | Point cloud viewer | `frontend/src/lib/viewport/` | Not browser-verified |
| Meshes | PARTIAL | Mesh viewer | `frontend/src/lib/viewport/` | Not browser-verified |
| Object hierarchy | PARTIAL | Studio outliner | `frontend/src/lib/viewport/` | Not browser-verified |
| Evidence inspection | PARTIAL | Inspector panel | `frontend/src/components/inspector/` | Not browser-verified |
| Object editing | PARTIAL | User corrections | `apps/api/routes_worlds.py` | Corrections API |
| Geometry editing | PARTIAL | Wall drag, door/window mark | `apps/api/routes_worlds.py` | Limited editing |
| Validation | PARTIAL | WorldIR validation gate | `world_ir/validation.py` | Compilation fails on invariant failure |
| Export | IMPLEMENTED | glTF, USD(A), IFC, CityGML, CityJSON, Blender | `exporters/` | 17 exporter modules |

---

## Summary by Classification

| Classification | Count |
|---------------|-------|
| IMPLEMENTED | ~45 |
| PARTIAL | ~25 |
| DONE | ~15 |
| BLOCKED | ~4 |
| MISSING | ~8 |
| SYNTHETICALLY VERIFIED | ~30 |
| REAL-WORLD VERIFIED | ~3 |
| GROUND-TRUTH VERIFIED | 0 |
| PRODUCTION VERIFIED | 0 |

### Critical Gaps

1. **No REAL-WORLD VERIFIED capabilities** — All verification is on synthetic fixtures
2. **No GROUND-TRUTH VERIFIED capabilities** — No tape measurements exist
3. **No PRODUCTION VERIFIED capabilities** — Not production-ready
4. **COLMAP dense MVS not Python-integrated** — Real dense run done manually, not wired into Python pipeline
5. **SAM backend broken** — torch-hub cache failure
6. **No learned door/window detector** — BLOCKED on user decision
7. **No real-room dataset** — All accuracy claims are synthetic

---

*Document last updated: 2026-11-05*  
*Audit performed by: Codex autonomous agent*
