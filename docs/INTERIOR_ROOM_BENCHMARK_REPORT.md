# Interior Furnished Room Sequential Reconstruction Benchmark Report

**Date:** 2026-11-20 07:04:53 UTC
**Device:** `cuda:0`
**Dataset:** `datasets/real_room_capture`
**Total Wall Time:** 173.59 s
**COLMAP Registered Cameras:** 21
**Persisted WorldIR Artifact:** `artifacts\interior_room_reconstruction.worldir.json`

## 1. Execution Summary

The Mandatory 10-Step Sequential Maximum-Fidelity Execution Policy was executed on multi-view photographs from `datasets/real_room_capture`.

All model workloads ran sequentially with full GPU/host memory unloads between stages:
1. **Camera Calibration:** Loaded genuine camera intrinsics (`fx=9874.29, fy=9941.92, cx=2016.0, cy=1512.0`) and extrinsics.
2. **Depth Anything V2:** Monocular depth inference executed per view with array validation (non-zero variance).
3. **OWL-ViT Open-Vocabulary Detection:** Evaluated interior elements: walls, doors, windows, tables, chairs, beds, couches, fixtures.
4. **SAM Instance Segmentation:** Generated instance masks for detected candidate regions.
5. **Metric 3D Lifting:** Lifted 2D detections into 3D world space using calibrated pinhole geometry and yaw PCA bounding boxes.
6. **Multi-View Physical Association:** Grouped 35 cross-view observations into 35 unique physical entities under strict physical constraints.
7. **WorldIR Compilation:** Reconciled canonical entities with evidence-calibrated confidence.

## 2. Reconstructed Entities Summary

- **Total 2D/3D Lifted Observations:** 35
- **Unique Physical Entities Reconstructed:** 35
- **Multi-View Merged Entities:** 0 (observed across 2 to 1 photographs)
- **Single-View Entities:** 35
- **Association Merges Accepted:** 0
- **Association Pairs Refused:** 595
- **WorldIR Schema Validation Errors:** 0 (structural schema compliance)

## 3. Multi-View Identity Association Benchmark Metrics

- **Evaluation Status:** `COMPLETED`
- **Total Cross-View Candidate Pairs Evaluated:** 23
- **True Positives (Correctly Merged Pairs):** 0
- **False Positives (False Merges):** 0
- **False Negatives (False Splits):** 8
- **Identity Precision:** 0.0000
- **Identity Recall:** 0.0000
- **Identity F1 Score:** 0.0000
- **False Merge Rate:** 0.0000
- **False Split Rate:** 1.0000
- **Unmatched Observations:** 0
- **Unlabelled Observations:** 27

## 4. Physical Accuracy and Honest Ground Truth Disclosure

- **Dataset Origin:** `datasets/real_room_capture originates from rendered synthetic multi-view room capture; independent millimeter physical laser survey is unmeasured.`
- **Physical Accuracy Status:** `NOT_MEASURED (Requires on-site physical double laser distance survey)`
- **Scale Status:** Units reported as `SYNTHETIC_METRES`, but `is_physically_metric=False` because physical laser ground truth measurements were not independently surveyed on site.
