# Capability matrix

Verified = the row's unit tests passed in this run; it is not accuracy on real photographs.

Live roots: apps.api.main, apps.api.jobs. Generated 2026-11-15T11:25:49+0530.

| Capability | Status | Verified | Modules (reachable from API / total) | Tests passed/failed | Real-data evidence |
|---|---|---|---|---|---|
| Image ingestion | Existing | yes (unit tests) | 4 / 4 | 37/0 | - |
| Video ingestion | Partial | yes (unit tests) | 0 / 2 | 16/0 | - |
| Image quality analysis | Partial | FAILING | 2 / 2 | 36/1 | - |
| Camera calibration | Existing | yes (unit tests) | 2 / 2 | 73/0 | - |
| Object detection | Existing | yes (unit tests) | 3 / 3 | 25/0 | Grounding DINO coarse pass inspected on one 1672x941 interior: vents, AC, doors, windows, shelves, plants, picture frame boxed sensibly. |
| Instance segmentation | Partial | yes (unit tests) | 0 / 3 | 7/0 | - |
| Small-object detection | Partial | yes (unit tests) | 2 / 3 | 12/0 | Tiled pass on 10 images: low precision (railings as 'visible wiring', texture as 'drain'); 678 unreviewed drafts, no ground truth. |
| Object tracking | Partial | yes (unit tests) | 0 / 3 | 29/0 | - |
| Depth estimation | Existing | yes (unit tests) | 3 / 3 | 25/0 | DPT_Hybrid needs a scale per view; a metric model (Depth Anything V2 Metric-Indoor Small) fits SfM better (0.117 vs 0.199 relative residual) with ONE scale, but did not change floor/ceiling structure in a controlled A/B (docs/engineering/TECHNOLOGY_EVALUATION.md 3a). |
| SfM | Partial | FAILING | 2 / 3 | 21/1 | 10 photos: 9/10 registered after seed pinning; sparse geometry differed between runs (995 vs 1,423 points). |
| MVS | Partial | FAILING | 2 / 3 | 27/4 | - |
| Point-cloud fusion | Existing | yes (unit tests) | 2 / 2 | 16/0 | - |
| Mesh reconstruction | Existing | yes (unit tests) | 2 / 2 | 51/0 | - |
| Multi-view identity | Existing | yes (unit tests) | 3 / 3 | 59/0 | - |
| Room segmentation | Existing | yes (unit tests) | 3 / 3 | 32/0 | Floor/wall consolidation cut replayed wall counts 25->8 and 21->6; footprint remains a ragged polygon. |
| Architectural topology | Existing | yes (unit tests) | 4 / 4 | 18/0 | - |
| Material recognition | Existing | yes (unit tests) | 3 / 3 | 17/0 | - |
| Uncertainty estimation | Partial | yes (unit tests) | 3 / 5 | 58/0 | - |
| WorldIR | Existing | yes (unit tests) | 24 / 27 | 27/0 | - |
| Accuracy validation | Existing | yes (unit tests) | 2 / 5 | 62/0 | - |
| Active capture guidance | Partial | yes (unit tests) | 1 / 2 | 7/0 | - |
| Export and persistence | Existing | yes (unit tests) | 4 / 4 | 40/0 | - |

## Implemented but not reachable from the API

* Video ingestion: reconstruction/video_frames.py, evidence/video.py
* Instance segmentation: perception/segmentation/sam2_backend.py, perception/segmentation/sam_backend.py, perception/detection/masks.py
* Small-object detection: perception/entities/small_objects.py
* Object tracking: perception/instances/track_backend.py, perception/tracking
* SfM: reconstruction/sfm/core_pipeline.py
* MVS: reconstruction/mvs/dense_cloud.py
* Uncertainty estimation: engine/quality/report.py
* Active capture guidance: reconstruction/completeness.py

## Failing tests (this run)

* Image quality analysis: tests/test_capture_quality.py::test_api_frame_picker_keeps_contract_and_falls_back
* SfM: tests/test_colmap_session.py::test_a_workspace_path_too_long_for_native_colmap_fails_early_with_the_real_cause
* MVS: tests/test_colmap_backend_dense.py::test_dense_mvs_attaches_fused_points_and_report, tests/test_colmap_backend_dense.py::test_dense_without_store_records_honest_no_artifact, tests/test_colmap_backend_dense.py::test_dense_stage_failure_raises_never_sparse_success, tests/test_colmap_backend_dense.py::test_dense_refused_when_binary_not_dense_capable
