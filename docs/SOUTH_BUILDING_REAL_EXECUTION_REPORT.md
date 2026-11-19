# South Building Real-Data Sequential Reconstruction Report

**Date:** 2026-11-19 17:06:06 UTC
**Device:** `cuda:0`
**Dataset:** `datasets/south_building`
**Total Wall Time:** 73.28 s
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
7. **Multi-View Physical Association:** Grouped 59 cross-view observations into 48 unique physical entities under strict physical non-merge constraints (e.g. `same_image_conflict`, `label_conflict`).
8. **WorldIR Compilation:** Assembled reconciled canonical physical entities with multi-view evidence-calibrated confidence.

## 2. Reconstructed Entities Summary

- **Total 2D/3D Lifted Observations:** 59
- **Unique Physical Entities Reconstructed:** 48
- **Multi-View Merged Entities:** 9 (observed across 2 to 4 photographs)
- **Single-View Entities:** 39
- **Association Merges Accepted:** 11
- **Association Pairs Refused:** 32
- **WorldIR Schema Validation Errors:** 0 (structural schema validation only)

| Photo | Depth Load (s) | Depth Infer (s) | Detections | SAM Masks | Lifted 3D Observations |
|---|---|---|---|---|---|
| `P1180141.JPG` | 16.077 | 0.709 | 15 | 15 | 15 |
| `P1180145.JPG` | 0.759 | 0.34 | 16 | 16 | 16 |
| `P1180149.JPG` | 0.644 | 0.405 | 14 | 14 | 14 |
| `P1180153.JPG` | 0.704 | 0.363 | 14 | 14 | 14 |

### Unique Physical Entity Class Distribution

| Class | Count | Category |
|---|---|---|
| `window window frame` | 11 | Access/Opening |
| `tree` | 9 | Site/Infrastructure |
| `building facade` | 4 | Structure/Envelope |
| `roof` | 4 | Site/Infrastructure |
| `bench` | 3 | Site/Infrastructure |
| `window` | 3 | Access/Opening |
| `column` | 3 | Site/Infrastructure |
| `window frame` | 3 | Access/Opening |
| `door entrance` | 3 | Access/Opening |
| `entrance` | 2 | Access/Opening |
| `window window` | 1 | Access/Opening |
| `wall` | 1 | Structure/Envelope |
| `door` | 1 | Access/Opening |

## 3. Physical Accuracy and Honest Benchmark Integrity

- **Scale Status:** Reconstructed coordinates reside in `COLMAP_SFM_UNITS`. Because South Building lacks an independent metric survey or laser distance meter reference, coordinates are unanchored gauge units. `is_physically_metric` is strictly `False`.
- **Physical Accuracy Metric:** `NOT_MEASURED (Requires independent millimeter physical laser survey)`.
- **Cross-View Identity Association:** 59 per-photo detections were clustered into 48 unique physical entities using 3D spatial proximity and cross-view camera reprojection constraints, preventing duplicate physical objects.
- **Calibrated Entity Confidence:** Confidence reflects multi-view consensus and camera reprojection error, preserving raw detector confidence in observation provenance.
- **Validation Scope:** Zero WorldIR validation errors confirms schema compliance; it does NOT constitute physical millimeter-accuracy proof.
