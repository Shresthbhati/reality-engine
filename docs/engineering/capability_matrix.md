# Capability matrix

Verified = the row's unit tests passed in this run; it is not accuracy on real photographs.

Live roots: apps.api.main, apps.api.jobs. Generated 2026-11-20T00:37:37+0530.

| Capability | Status | Verified | Modules (reachable from API / total) | Tests passed/failed | Real-data evidence |
|---|---|---|---|---|---|
| Image ingestion | Existing | yes (unit tests) | 4 / 4 | 37/0 | - |
| Video ingestion | Partial | yes (unit tests) | 0 / 2 | 16/0 | - |
| Image quality analysis | Existing | yes (unit tests) | 2 / 2 | 40/0 | - |
| Camera calibration | Partial | yes (unit tests) | 2 / 3 | 79/0 | - |
| Object detection | Existing | yes (unit tests) | 3 / 3 | 31/0 | Grounding DINO coarse pass inspected on one 1672x941 interior: vents, AC, doors, windows, shelves, plants, picture frame boxed sensibly. |
| Instance segmentation | Partial | yes (unit tests) | 0 / 3 | 7/0 | - |
| Small-object detection | Partial | yes (unit tests) | 3 / 4 | 16/0 | Tiled pass on 10 images: low precision (railings as 'visible wiring', texture as 'drain'); 678 unreviewed drafts, no ground truth. |
| Object tracking | Partial | yes (unit tests) | 0 / 3 | 29/0 | - |
| Depth estimation | Existing | yes (unit tests) | 3 / 3 | 25/0 | DPT_Hybrid needs a scale per view; a metric model (Depth Anything V2 Metric-Indoor Small) fits SfM better (0.117 vs 0.199 relative residual) with ONE scale, but did not change floor/ceiling structure in a controlled A/B (docs/engineering/TECHNOLOGY_EVALUATION.md 3a). |
| Metric-3D evidence producers | Partial | yes (unit tests) | 3 / 4 | 66/0 | Executed with REAL models on 3 South Building photos (exterior, no ground truth): MiDaS DPT_Hybrid (dpt_hybrid_384.pt, sha256 501f0c75...) and Depth Anything V2 Small (sha256 3152477c...) each emitted 3 valid relative-depth records; raw 768x1024 float32 arrays reloaded byte-exact from disk. Sparse-SfM lifter on the 32-photo set: COLMAP registered 22/32 images (4,506 points), OWL-ViT made 89 detections, 23 regions lifted to sfm_gauge records, 17 rejected for too few points. Gauge units only: nothing here is metres, and no interior or physical ground truth was used. |
| Metric scale calibration | Partial | yes (unit tests) | 0 / 1 | 34/0 | No physical anchor has ever been applied to a real reconstruction; South Building stays COLMAP_SFM_UNITS. Anchor acceptance is verified on synthetic scenes only. |
| Benchmark manifests (BLOCKED when ground truth is missing) | Partial | yes (unit tests) | 0 / 1 | 16/0 | No furnished-interior benchmark exists in the repository: every interior metric is BLOCKED (no scenes, no independent measurements). |
| SfM | Existing | yes (unit tests) | 3 / 3 | 22/0 | 10 photos: 9/10 registered after seed pinning; sparse geometry differed between runs (995 vs 1,423 points). |
| MVS | Existing | yes (unit tests) | 3 / 3 | 31/0 | - |
| Point-cloud fusion | Partial | yes (unit tests) | 2 / 3 | 25/0 | - |
| Mesh reconstruction | Partial | yes (unit tests) | 2 / 3 | 55/0 | - |
| Multi-view identity | Existing | yes (unit tests) | 3 / 3 | 60/0 | - |
| Room segmentation | Existing | yes (unit tests) | 3 / 3 | 34/0 | Floor/wall consolidation cut replayed wall counts 25->8 and 21->6; footprint remains a ragged polygon. |
| Architectural topology | Partial | yes (unit tests) | 5 / 7 | 24/0 | - |
| Material recognition | Existing | yes (unit tests) | 3 / 3 | 17/0 | - |
| Uncertainty estimation | Existing | yes (unit tests) | 4 / 5 | 58/0 | - |
| WorldIR | Existing | yes (unit tests) | 27 / 32 | 27/0 | - |
| Accuracy validation | Existing | yes (unit tests) | 2 / 6 | 66/0 | - |
| Active capture guidance | Existing | yes (unit tests) | 2 / 2 | 7/0 | - |
| Export and persistence | Existing | yes (unit tests) | 4 / 4 | 48/0 | - |

## Implemented but not reachable from the API

* Video ingestion: reconstruction/video_frames.py, evidence/video.py
* Camera calibration: trajectories/interpolation.py
* Instance segmentation: perception/segmentation/sam2_backend.py, perception/segmentation/sam_backend.py, perception/detection/masks.py
* Small-object detection: perception/fine_grained/estimators.py
* Object tracking: perception/instances/track_backend.py, perception/tracking
* Metric-3D evidence producers: engine/multiengine/adapters/metric_3d_lifter.py
* Metric scale calibration: engine/multiengine/depth_scale_alignment.py
* Benchmark manifests (BLOCKED when ground truth is missing): engine/multiengine/benchmark_manifest.py
* Point-cloud fusion: reconstruction/fusion/robust_fusion.py
* Mesh reconstruction: reconstruction/meshing/room_mesh.py
* Architectural topology: perception/architecture/split_level.py, perception/architecture/multi_storey_validator.py
