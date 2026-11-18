# Capability matrix

Verified = the row's unit tests passed in this run; it is not accuracy on real photographs.

Live roots: apps.api.main, apps.api.jobs. Generated 2026-11-18T21:37:58+0530.

| Capability | Status | Verified | Modules (reachable from API / total) | Tests passed/failed | Real-data evidence |
|---|---|---|---|---|---|
| Image ingestion | Partial | not run | 4 / 4 | 0/0 | - |
| Video ingestion | Partial | not run | 0 / 2 | 0/0 | - |
| Image quality analysis | Partial | not run | 2 / 2 | 0/0 | - |
| Camera calibration | Partial | not run | 2 / 2 | 0/0 | - |
| Object detection | Partial | not run | 3 / 3 | 0/0 | Grounding DINO coarse pass inspected on one 1672x941 interior: vents, AC, doors, windows, shelves, plants, picture frame boxed sensibly. |
| Instance segmentation | Partial | not run | 0 / 3 | 0/0 | - |
| Small-object detection | Partial | not run | 3 / 3 | 0/0 | Tiled pass on 10 images: low precision (railings as 'visible wiring', texture as 'drain'); 678 unreviewed drafts, no ground truth. |
| Object tracking | Partial | not run | 0 / 3 | 0/0 | - |
| Depth estimation | Partial | not run | 3 / 3 | 0/0 | DPT_Hybrid needs a scale per view; a metric model (Depth Anything V2 Metric-Indoor Small) fits SfM better (0.117 vs 0.199 relative residual) with ONE scale, but did not change floor/ceiling structure in a controlled A/B (docs/engineering/TECHNOLOGY_EVALUATION.md 3a). |
| SfM | Partial | not run | 3 / 3 | 0/0 | 10 photos: 9/10 registered after seed pinning; sparse geometry differed between runs (995 vs 1,423 points). |
| MVS | Partial | not run | 3 / 3 | 0/0 | - |
| Point-cloud fusion | Partial | not run | 2 / 2 | 0/0 | - |
| Mesh reconstruction | Partial | not run | 2 / 2 | 0/0 | - |
| Multi-view identity | Partial | not run | 3 / 3 | 0/0 | - |
| Room segmentation | Partial | not run | 3 / 3 | 0/0 | Floor/wall consolidation cut replayed wall counts 25->8 and 21->6; footprint remains a ragged polygon. |
| Architectural topology | Partial | not run | 4 / 4 | 0/0 | - |
| Material recognition | Partial | not run | 3 / 3 | 0/0 | - |
| Uncertainty estimation | Partial | not run | 4 / 5 | 0/0 | - |
| WorldIR | Partial | not run | 26 / 31 | 0/0 | - |
| Accuracy validation | Partial | not run | 2 / 5 | 0/0 | - |
| Active capture guidance | Partial | not run | 2 / 2 | 0/0 | - |
| Export and persistence | Partial | not run | 4 / 4 | 0/0 | - |

## Implemented but not reachable from the API

* Video ingestion: reconstruction/video_frames.py, evidence/video.py
* Instance segmentation: perception/segmentation/sam2_backend.py, perception/segmentation/sam_backend.py, perception/detection/masks.py
* Object tracking: perception/instances/track_backend.py, perception/tracking
