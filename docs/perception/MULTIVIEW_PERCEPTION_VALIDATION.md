# Multi-View Perception Validation

## 1. Multi-View Correspondence & Feature Matching
- **Engines:** `lightglue_aliked` (ALIKED + LightGlue via Kornia) and `sift` (OpenCV SIFT + RANSAC).
- **6 GB GPU Resolution Scaling (`perception/engine_adapters.py`):** Images exceeding `max_long_edge_px=1600` are downscaled prior to ALIKED feature extraction and matched keypoints are rescaled back to original pixel coordinates.
- **Real-Photo Result (`datasets/real_room_capture`, 4 × 4032×3024):** `lightglue_aliked` recovered `3` verified geometric pairs in `3.10s` (`3075.5 MB` peak VRAM) where previously it failed with CUDA OOM at `5.85 GiB`.

## 2. Cross-View Identity & Room Association
- `perception/instances/association.py` and `perception/multiview/room_associator.py` resolve cross-view object identity and room containment using camera poses, 3D proximity, semantic compatibility, and uncertainty gating.
- Ground-truth identity precision/recall evaluation is formalized in PR-C (`perception/evaluation/`).
