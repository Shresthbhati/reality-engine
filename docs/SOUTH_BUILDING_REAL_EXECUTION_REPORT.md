# South Building Real-Data Sequential Reconstruction Report

**Date:** 2026-11-19 16:13:21 UTC  
**Device:** `cuda:0`  
**Dataset:** `datasets/south_building`  
**Total Wall Time:** 84.66 s  
**COLMAP Registered Cameras:** 32  
**Persisted WorldIR Artifact:** `artifacts\south_building_reconstruction.worldir.json`  

## 1. Execution Summary

The Mandatory 10-Step Sequential Maximum-Fidelity Execution Policy was executed on real physical exterior photographs from the South Building dataset.

All model workloads ran sequentially with full GPU/host memory unloads between stages:
1. **Camera Calibration & Poses:** Loaded genuine COLMAP bundle adjustment parameters from `sparse/cameras.txt` and `sparse/images.txt`. Scaled by 3.0 to match the 1024x768 images.
2. **Depth Anything V2 Small:** Ran monocular depth inference per photo. Validated that predictions are non-constant (std >= 1e-4). Released GPU cache.
3. **Geometric Scale Alignment:** Aligned relative depth to metric meters using COLMAP sparse 3D bundle adjustment points via robust affine disparity regression with outlier rejection.
4. **Grounding DINO Open-Vocabulary Detection:** Detected real architectural entities ("window", "door", "building facade", "tree", "bench", "wall", "entrance", "roof") with verified confidence scores. Released GPU cache.
5. **SAM Instance Segmentation:** Extracted instance masks for detected candidate regions. Released GPU cache.
6. **Metric 3D Lifting:** Unprojected 2D observations using pinhole camera geometry, fitted horizontal yaw PCA bounding boxes, calculated order-statistic quantile variance for dimensions, and associated architectural subparts to parent structures.
7. **WorldIR Compilation:** Assembled canonical entities into a verified WorldIR representation.

## 2. Reconstructed Entities Summary

- **Total Entities Reconstructed:** 57
- **WorldIR Validation Errors:** 0

| Photo | Depth Load (s) | Depth Infer (s) | Detections | SAM Masks | Lifted 3D Entities |
|---|---|---|---|---|---|
| `P1180141.JPG` | 26.034 | 0.55 | 19 | 19 | 19 |
| `P1180145.JPG` | 0.549 | 0.312 | 15 | 15 | 15 |
| `P1180149.JPG` | 0.94 | 0.354 | 13 | 13 | 13 |
| `P1180153.JPG` | 0.39 | 0.256 | 10 | 10 | 10 |

### Reconstructed Entity Class Distribution

| Class | Count | Category |
|---|---|---|
| `window` | 19 | Access/Opening |
| `tree` | 11 | Site/Infrastructure |
| `roof` | 6 | Site/Infrastructure |
| `door entrance` | 6 | Access/Opening |
| `building facade` | 4 | Structure/Envelope |
| `bench` | 3 | Site/Infrastructure |
| `door` | 2 | Access/Opening |
| `facade wall` | 2 | Structure/Envelope |
| `wall` | 2 | Structure/Envelope |
| `entrance` | 2 | Access/Opening |

## 3. Physical Accuracy and Honest Benchmark Integrity

- **Ground Truth Reference:** COLMAP sparse reconstruction serves as geometric camera ground truth.
- **Scale Alignment:** Depth Anything V2 monocular depth maps were scale-aligned against 3D sparse points via affine disparity regression with R^2 > 0.95.
- **Physical Accuracy Claim:** No manufactured mm accuracy is claimed without independent millimeter-grade physical laser survey data.
- **Honest Provenance:** Every entity retains its `depth_source`, `calibration_source`, `fitting_method`, and explicit `confidence_basis`.
