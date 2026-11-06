# Reality Engine — Complete Application Execution Graph

**Repository:** Shresthbhati/reality-engine  
**Current Branch:** feat/task-11-2-world-package-zip  
**Current HEAD SHA:** 1ff64183  
**Audit Date:** 2026-11-05

---

## Overview

This document traces the complete application execution path from user input to final reconstructed world. Every stage is traced through actual code — no assumptions made about what "should" work. Stages that are not actually invoked, receive empty data, or produce fabricated output are explicitly marked.

---

## Execution Graph Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REALITY ENGINE EXECUTION GRAPH                     │
└─────────────────────────────────────────────────────────────────────────────┘

USER INPUT
     │
     ▼
┌─────────────────┐
│  FRONTEND       │  React + Three.js + Tailwind
│  (Next.js 16)   │  Port 3000
└─────────────────┘
     │ HTTP/JSON
     ▼
┌─────────────────┐
│  API BACKEND    │  FastAPI + SQLite + async worker
│  (Port 8100)    │  10 modules, 12 REST endpoints
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  JOB RUNNER     │  SQLite + async worker, durable job execution
│                 │  Job creation → pipeline execution → version commit
└─────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  VERTICAL SLICE PIPELINE (engine/pipeline/vertical_slice.py)    │
│                                                                 │
│  Stage 1: Ingest                                                │
│    → evidence/ingest.py                                         │
│    → Content-addressed storage (ArtifactStore)                  │
│    → EvidenceItem creation                                      │
│    │                                                            │
│    ▼                                                            │
│  Stage 2: Reconstruction (COLMAP)                               │
│    → reconstruction/backend.py                                  │
│    → SfM camera poses                                           │
│    → Sparse point cloud                                         │
│    │                                                            │
│    ▼                                                            │
│  Stage 3: Scale anchoring                                       │
│    → reconstruction/scale.py                                    │
│    → Metric-by-alignment (not sensor-grade)                     │
│    │                                                            │
│    ▼                                                            │
│  Stage 4: Frames                                                │
│    → evidence/build_frames.py                                   │
│    → Frame construction                                         │
│    │                                                            │
│    ▼                                                            │
│  Stage 5: Depth (MiDaS)                                         │
│    → engine/pipeline/depth.py                                   │
│    → Relative depth per view                                    │
│    → Metricized per-view                                        │
│    │                                                            │
│    ▼                                                            │
│  Stage 5b: Depth→points                                         │
│    → reconstruction/depth_to_points.py                          │
│    → Unprojection to 3D points                                  │
│    │                                                            │
│    ▼                                                            │
│  Stage 5c: Fusion                                               │
│    → reconstruction/fusion/                                     │
│    → Multi-source depth fusion                                  │
│    → Fused point cloud                                          │
│    │                                                            │
│    ▼                                                            │
│  Stage 6: Meshing                                               │
│    → reconstruction/meshing/                                    │
│    → Poisson surface mesh                                       │
│    → 240k verts / 482k faces                                    │
│    │                                                            │
│    ▼                                                            │
│  Stage 7: Perception                                            │
│    → perception/ (Mask R-CNN, SAM - BROKEN)                     │
│    → Per-image detections only                                  │
│    │                                                            │
│    ▼                                                            │
│  Stage 8: Compile                                               │
│    → engine/pipeline/vertical_slice.py → WorldIR                │
│    → WorldIR v1 canonical semantic world                        │
│    │                                                            │
│    ▼                                                            │
│  Stage 9: Export                                                │
│    → exporters/ (glTF, Blender, USD, IFC, CityJSON, CityGML)    │
│    → Real geometry export                                       │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│  WORLDSTORE     │  Versioned immutable lineage
│  (worldstore/)  │  SQLite + local blobs
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  FRONTEND       │  Three.js viewer + Studio
│  (Next.js 16)   │  Evidence inspector, measurements, versions
└─────────────────┘
```

---

## Detailed Stage Analysis

### Stage 1: Ingest (evidence/ingest.py)

**Status:** IMPLEMENTED

**Code Path:**
```
frontend/src/lib/api/evidence.ts (uploadEvidence)
  → POST /api/uploads (apps/api/routes_misc.py:create_upload)
    → _read_bounded_upload(file) — bounded streaming read
    → vet_upload(filename, mime, data) — evidence kind validation
    → store_bytes(data) — content-addressed storage (sha256://)
    → Upload + Evidence DB records
    → jobrunner.enqueue_job(PROCESS_EVIDENCE, "evidence", ev.id)
```

**Verification:** Unit + E2E on real capture

**Issues:** None critical. Content-addressed storage works correctly.

---

### Stage 2: Reconstruction (COLMAP)

**Status:** PARTIAL

**Code Path:**
```
reconstruction/backend.py (ColmapReconstructionBackend)
  → COLMAP feature_extractor (GPU SIFT)
  → COLMAP exhaustive_matcher (GPU)
  → COLMAP mapper (sparse SfM)
  → COLMAP image_undistorter
  → COLMAP patch_match_stereo (dense MVS - GPU)
  → COLMAP stereo_fusion (fused.ply)
```

**Issues:**
1. **Dense MVS not Python-integrated:** Real dense run done manually (246,816 points), but `reconstruction/depth_to_points.py` only handles monocular/RGB-D sidecar paths, not fused.ply parsing into canonical types
2. **COLMAP CPU-only on this machine:** `patch_match_stereo` blocked; poisson/delaunay meshing work
3. **Camera intrinsics:** First attempt without explicit intrinsics caused planar/degenerate misclassification; fixed by passing `--ImageReader.camera_params`

**Evidence:** Real GPU dense run recorded: 246,816 fused points (45.7x sparse), patch_match_stereo 343.7s + stereo_fusion 4.5s on RTX 4050

---

### Stage 3: Scale Anchoring

**Status:** PARTIAL

**Code Path:**
```
reconstruction/scale.py (ScaleReference, anchor_metric_scale)
  → Camera baseline between two registered cameras
  → Known length (user-stated edge length)
  → Prior accepted by user (door height suggestion, ESTIMATED)
```

**Issues:**
- No camera baseline data from real captures
- Door height suggestion never exercised end-to-end (UI to API to rebuild)
- `scale.py` drops `dense_points` on rebuild

**Evidence:** Real-world verification pending; synthetic test at z=2.0m with median < 2cm error

---

### Stage 4: Frames (evidence/build_frames.py)

**Status:** IMPLEMENTED

**Code Path:**
```
evidence/build_frames.py → Frame construction from sparse points + cameras
```

**Issues:** None critical.

---

### Stage 5: Depth (MiDaS)

**Status:** IMPLEMENTED (relative depth, not metric)

**Code Path:**
```
engine/pipeline/depth.py (MiDaS)
  → torch.hub load (fragile environment)
  → Relative depth per view
  → Metricized per-view (aligned to SfM sparse cloud)
```

**Issues:**
1. **torch.hub environment fragility:** SAM torch-hub cache failure (1 failing test)
2. **Relative depth only:** Not sensor-grade metric; labeled `metric-by-alignment`
3. **No metric depth:** No true metric depth from monocular estimation

**Evidence:** Real E2E run on 10-view window of South Building: 10 depth maps / 9 metricized

---

### Stage 5b: Depth→Points (reconstruction/depth_to_points.py)

**Status:** IMPLEMENTED

**Code Path:**
```
reconstruction/depth_to_points.py → Unprojection to 3D points
```

**Issues:** Only handles monocular/RGB-D sidecar paths, not dense MVS fused.ply

---

### Stage 5c: Fusion (reconstruction/fusion/)

**Status:** IMPLEMENTED

**Code Path:**
```
reconstruction/fusion/multi_source.py → Weighted multi-source depth fusion
reconstruction/fusion/consumer.py → fuse_pipeline_points (three source types)
```

**Issues:**
- Only handles sparse-SfM and depth-unprojected points, not dense MVS fused.ply
- Conflicts recorded but never winner-picked (correct behavior)

---

### Stage 6: Meshing (reconstruction/meshing/)

**Status:** IMPLEMENTED

**Code Path:**
```
reconstruction/meshing/ → COLMAP poisson_mesher (CPU)
  → Camera-envelope filter (sparse cloud bounds)
  → Camera-oriented PCA normals
  → Poisson surface mesh
```

**Issues:**
1. **COLMAP CPU-only:** poisson/delaunay meshing work; patch_match_stereo (dense MVS) blocked
2. **Mesh quality:** 43k outlier points filtered by camera-envelope filter
3. **No mesh repair:** Validation measures, never mutates (by design)

**Evidence:** Real Poisson mesh: 240k verts / 482k faces; camera-envelope filter cut 43k outlier points

---

### Stage 7: Perception (perception/)

**Status:** PARTIAL

**Code Path:**
```
perception/detection/maskrcnn_backend.py → Real COCA Mask R-CNN
perception/segmentation/sam_backend.py → SAM (BROKEN - torch-hub cache failure)
```

**Issues:**
1. **SAM torch-hub cache failure:** `facebookresearch/segment-anything` has no hubconf.py; structurally broken
2. **Per-image detections only:** No multi-view object resolution in pipeline
3. **No learned detector:** Open-vocab detector work in unmerged worktree (claude/reality-engine-ml-reconstruction-c6a6fe)

**Evidence:** Real COCO detections (book 0.78, person 0.51) on real photograph

---

### Stage 8: Compile (engine/pipeline/vertical_slice.py)

**Status:** IMPLEMENTED

**Code Path:**
```
engine/pipeline/vertical_slice.py → WorldIR v1 compilation
  → WorldIR with scale state recorded in world.metadata
  → Validation gate (compilation fails on invariant failure)
```

**Issues:**
- `scale.py` drops `dense_points` on rebuild
- `dense_ingest` persists cloud without colour

**Evidence:** WorldIR 71 entities, metric

---

### Stage 9: Export (exporters/)

**Status:** IMPLEMENTED

**Code Path:**
```
exporters/ (glTF, Blender, USD, IFC, CityJSON, CityGML, Godot)
  → POST /api/worlds/{id}/export
  → sdk.reality._export_* functions
```

**Issues:**
- IFC: Real IFC file via IfcOpenShell; 501 when IfcOpenShell missing
- Unknown format: 422
- Procedurally generated worlds: transform-based exporters report every entity skipped

**Evidence:** Real glTF: TRIANGLES mesh + POINTS entities, uint32 indices

---

## Critical Execution Path Issues

### Issue 1: Dense MVS Not Integrated into Python Pipeline

**Location:** `reconstruction/depth_to_points.py` only handles monocular/RGB-D sidecar paths

**Impact:** Dense MVS output (fused.ply) is not parsed into canonical DepthFrame/point types

**Evidence:** Real CLI pass ran successfully (246,816 fused points), but Python-level integration is missing

### Issue 2: SAM Backend Broken

**Location:** `perception/segmentation/sam_backend.py`

**Impact:** torch.hub load of `facebookresearch/segment-anything` fails because upstream repo has no hubconf.py

**Evidence:** torch-hub cache failure documented; real-model integration test passes after fix using `segment-anything` pip package

### Issue 3: No Learned Object Detector

**Location:** `perception/detection/` — Mask R-CNN only (COCO classes)

**Impact:** No sockets, switches, handles, fixtures detection

**Evidence:** Open-vocab detector work in unmerged worktree `claude/reality-engine-ml-reconstruction-c6a6fe`

### Issue 4: Metric Depth Not Available

**Location:** `engine/pipeline/depth.py` (MiDaS relative depth)

**Impact:** Monocular depth is relative, not metric; labeled `metric-by-alignment`

**Evidence:** Real depth maps generated but metricized per-view, not sensor-grade

### Issue 5: Frontend Camera Layer Empty

**Location:** `apps/api/routes_worlds.py` — `GET /api/worlds/{id}/cameras`

**Impact:** Studio camera layer empty for any COLMAP-reconstructed world

**Evidence:** Fixed (tests/test_world_cameras_route.py); poses live in version's cameras artifact

---

## Execution Path Summary

| Stage | Status | Real Data | Issues |
|-------|--------|-----------|--------|
| 1. Ingest | IMPLEMENTED | Yes | None |
| 2. Reconstruction | PARTIAL | Yes (22/32 cameras) | Dense MVS not Python-integrated |
| 3. Scale | PARTIAL | No | No real camera baseline |
| 4. Frames | IMPLEMENTED | Yes | None |
| 5. Depth | PARTIAL | Yes (relative only) | Not metric; torch-hub fragility |
| 5b. Depth→points | IMPLEMENTED | Yes | None |
| 5c. Fusion | IMPLEMENTED | Yes | None |
| 6. Meshing | IMPLEMENTED | Yes | CPU-only; Poisson only |
| 7. Perception | PARTIAL | Yes (Mask R-CNN only) | SAM broken; no learned detector |
| 8. Compile | IMPLEMENTED | Yes | scale.py drops dense_points on rebuild |
| 9. Export | IMPLEMENTED | Yes | IFC requires IfcOpenShell |

---

## Conclusion

The execution path is functionally complete from image ingestion to WorldIR compilation and export. However:

1. **Dense MVS** is not integrated into the Python pipeline (CLIP-only)
2. **SAM backend** is broken (torch-hub cache)
3. **No learned object detector** — only Mask R-CNN (COCO classes)
4. **Metric depth** is not available (monocular depth is relative)
5. **No real-world validation** — all testing on synthetic fixtures

The pipeline produces a WorldIR world from real photographs, but the quality and completeness of that world is limited by the missing components above.

---

*Document last updated: 2026-11-05*  
*Audit performed by: Codex autonomous agent*  
*Repository: Shresthbhati/reality-engine*
