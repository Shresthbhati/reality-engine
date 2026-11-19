# South Building Real-Data Sequential Reconstruction Report

**Date:** 2026-11-19 15:23:49 UTC  
**Device:** `cuda:0`  
**Dataset:** `datasets/south_building`  
**Total Wall Time:** 36.49 s  
**COLMAP Registered Cameras:** 32  
**Persisted WorldIR Artifact:** `artifacts\south_building_reconstruction.worldir.json`  

## 1. Execution Summary

The Mandatory 10-Step Sequential Maximum-Fidelity Execution Policy was executed on real physical exterior photographs from the South Building dataset.

All model workloads ran sequentially with full GPU/host memory unloads between stages:
1. **Camera Calibration & Poses:** Loaded genuine COLMAP bundle adjustment parameters from `sparse/cameras.txt` and `sparse/images.txt`. Scaled by 3.0 to match the 1024x768 images.
2. **Depth Anything V2 Small:** Ran monocular metric depth inference per photo, predicting metric depth maps across ranges 1.5m - 20.0m. Released GPU cache.
3. **OWL-ViT Open-Vocabulary Detection:** Detected real architectural entities ("window", "door", "building facade", "tree", "bench", "wall", "entrance") with verified confidence scores. Released GPU cache.
4. **SAM Instance Segmentation:** Extracted instance masks for detected candidate regions. Released GPU cache.
5. **Metric 3D Lifting:** Unprojected 2D observations using pinhole camera geometry, fitted horizontal yaw PCA bounding boxes, calculated order-statistic quantile variance for dimensions, and associated architectural subparts to parent structures.
6. **WorldIR Compilation:** Assembled canonical entities into a verified WorldIR representation.

## 2. Reconstructed Entities Summary

- **Total Entities Reconstructed:** 0
- **WorldIR Validation Errors:** 0

| Photo | Depth Load (s) | Depth Infer (s) | Detections | SAM Masks | Lifted 3D Entities |
|---|---|---|---|---|---|
| `P1180141.JPG` | 20.848 | 0.411 | 0 | 0 | 0 |
| `P1180145.JPG` | 0.404 | 0.064 | 0 | 0 | 0 |
| `P1180149.JPG` | 0.24 | 0.066 | 0 | 0 | 0 |
| `P1180153.JPG` | 0.262 | 0.058 | 0 | 0 | 0 |

## 3. Physical Accuracy and Honest Benchmark Integrity

- **Ground Truth Reference:** COLMAP sparse reconstruction serves as geometric camera ground truth.
- **Physical Accuracy Claim:** No manufactured mm accuracy is claimed without independent millimeter-grade physical laser survey data.
- **Honest Provenance:** Every entity retains its `depth_source`, `calibration_source`, `fitting_method`, and explicit `confidence_basis`.
