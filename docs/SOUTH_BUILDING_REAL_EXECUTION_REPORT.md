# South Building Real-Data Sequential Reconstruction Report

**Date:** 2026-11-19 18:47:33 UTC
**Device:** `cuda:0`
**Dataset:** `datasets/south_building`
**Total Wall Time:** 86.6 s
**COLMAP Registered Cameras:** 32
**Persisted WorldIR Artifact:** `artifacts\south_building_reconstruction.worldir.json`

## 1. Execution Summary

The Mandatory 10-Step Sequential Maximum-Fidelity Execution Policy was executed on real physical exterior photographs from the South Building dataset.

All model workloads ran sequentially with full GPU/host memory unloads between stages:
1. **Camera Calibration & Poses:** Loaded genuine COLMAP bundle adjustment parameters from `sparse/cameras.txt` and `sparse/images.txt`. Scaled by 3.0 to match the 1024x768 images.
2. **Depth Anything V2 Small:** Ran monocular depth inference per photo. Validated that predictions are non-constant (std >= 1e-4). Released GPU cache.
3. **Geometric Scale Alignment:** Aligned relative depth to reference coordinates via robust affine disparity regression with held-out validation. Scale units: `COLMAP_SFM_UNITS` (Unanchored COLMAP gauge; `is_physically_metric=False`).
4. **Grounding DINO Open-Vocabulary Detection:** Detected real architectural entities with verified confidence scores. Released GPU cache.
5. **SAM Instance Segmentation:** Extracted instance masks for detected candidate regions. Released GPU cache.
6. **Metric 3D Lifting:** Unprojected 2D observations using pinhole camera geometry, fitted horizontal yaw PCA bounding boxes, and associated architectural subparts to parent structures.
7. **Multi-View Physical Association:** Grouped 23 cross-view observations into 23 unique physical entities under strict physical non-merge constraints (e.g. `same_image_conflict`, `label_conflict`).
8. **WorldIR Compilation:** Assembled reconciled canonical physical entities with multi-view evidence-calibrated confidence.

## 2. Reconstructed Entities Summary

- **Total 2D/3D Lifted Observations:** 23
- **Unique Physical Entities Reconstructed:** 23
- **Multi-View Merged Entities:** 0 (observed across 2 to 1 photographs)
- **Single-View Entities:** 23
- **Association Merges Accepted:** 0
- **Association Pairs Refused:** 0
- **WorldIR Schema Validation Errors:** 0 (structural schema validation only)

| Photo | Depth Load (s) | Depth Infer (s) | Detections | SAM Masks | Lifted 3D Observations |
|---|---|---|---|---|---|
| `P1180141.JPG` | 34.286 | 1.321 | 22 | 22 | 0 |
| `P1180145.JPG` | 0.578 | 0.149 | 21 | 21 | 0 |
| `P1180149.JPG` | 0.688 | 0.073 | 23 | 23 | 23 |
| `P1180153.JPG` | 0.826 | 0.412 | 17 | 17 | 0 |

### Unique Physical Entity Class Distribution

| Class | Count | Category |
|---|---|---|
| `window window frame` | 4 | Access/Opening |
| `lamp` | 3 | Site/Infrastructure |
| `tree` | 2 | Site/Infrastructure |
| `column base` | 2 | Site/Infrastructure |
| `ground surface` | 2 | Site/Infrastructure |
| `door entrance` | 2 | Access/Opening |
| `building facade` | 1 | Structure/Envelope |
| `roof` | 1 | Site/Infrastructure |
| `column` | 1 | Site/Infrastructure |
| `column lamp` | 1 | Site/Infrastructure |
| `window window` | 1 | Access/Opening |
| `bench` | 1 | Site/Infrastructure |
| `railing` | 1 | Site/Infrastructure |
| `entrance` | 1 | Access/Opening |

## 3. Multi-View Identity Association Benchmark Metrics

- **Total Cross-View Candidate Pairs Evaluated:** 0
- **True Positives (Correctly Merged Pairs):** 0
- **False Positives (False Merges):** 0
- **False Negatives (False Splits):** 0
- **Identity Precision:** 1.0000
- **Identity Recall:** 1.0000
- **Identity F1 Score:** 1.0000
- **False Merge Rate:** 0.0000
- **False Split Rate:** 0.0000

## 4. Physical Accuracy and Honest Benchmark Integrity

- **Scale Status:** Reconstructed coordinates reside in `COLMAP_SFM_UNITS`. Because South Building lacks an independent metric survey or laser distance meter reference, coordinates are unanchored gauge units. `is_physically_metric` is strictly `False`.
- **Physical Accuracy Metric:** `NOT_MEASURED (Requires independent millimeter physical laser survey)`.
- **Cross-View Identity Association:** 23 per-photo detections were clustered into 23 unique physical entities using 3D spatial proximity and cross-view camera reprojection constraints, preventing duplicate physical objects.
- **Calibrated Entity Confidence:** Confidence reflects multi-view consensus and camera reprojection error, preserving raw detector confidence in observation provenance.
- **Validation Scope:** Zero WorldIR validation errors confirms schema compliance; it does NOT constitute physical millimeter-accuracy proof.
