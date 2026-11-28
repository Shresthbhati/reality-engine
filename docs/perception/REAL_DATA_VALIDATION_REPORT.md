# Real-Data End-to-End Perception Validation Report (Phases 7–10)

**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), PyTorch `2.14.1+cu130` (`cuda: True`).  
**Script:** `scripts/run_perception_real_validation.py`  
**Note:** `datasets/real_room_capture` and `datasets/south_building` do not include 2D/3D per-object bounding-box ground truth, so per-class detection counts below are **unscored operational/detection runs** (never claimed as precision/recall).

---

## 1. Indoor Validation (`datasets/real_room_capture`, 4 × 4032×3024 photos, `domain="indoor"`, `tile_scale=2`)

- **Total Wall Time:** `286.25s`
- **Total Observations:** `195` (`92` detection, `92` segmentation, `8` depth, `3` SIFT matching)
- **Masks Persisted (`MaskStore`):** `92`
- **Detections by Class:** `architecture/wall` (24), `architecture/floor` (14), `architecture/ceiling` (11), `fixtures/electrical/socket` (8), `architecture/window` (6), `architecture/door` (5), `electrical.junction_box` (5), `architecture/passage` (4), `architecture/skirting` (3), `architecture/column` (3), `circulation.railing.glass` (3), `electrical.switchboard.module.blank` (3), `fixtures/electrical/switch` (2), `opening.opening` (1).
- **Small-Object Detections (18 total):** `fixtures/electrical/socket` (8), `electrical.junction_box` (5), `electrical.switchboard.module.blank` (3), `fixtures/electrical/switch` (2).

| Engine | Status | Device | Wall Time (s) | Peak VRAM (MB) | Observations | Notes / Failures |
|---|---|---|---:|---:|---:|---|
| `lightglue_aliked` | `invalid_output` | `cuda` | 38.87 | — | 0 | CUDA OOM (`5.85 GiB` requested on 4032×3024); adapter returned empty `"succeeded"` caught by orchestrator |
| `grounding_dino` | `succeeded` | `cuda` | 40.62 | 1,848.3 | 92 | 2×2 tiled inference |
| `sift` | `succeeded` | `cuda` | 6.86 | 8.1 | 3 | 3 verified pairs |
| `sam2` | `succeeded` | `cuda` | 161.99 | 5,543.4 | 92 | Exceeded 5,200 MB safe budget at 4032×3024 with 92 prompts |
| `depth_anything` | `succeeded` | `cuda` | 6.05 | 312.6 | 4 | Relative depth |
| `maskrcnn` | `invalid_output` | `cuda` | 3.01 | — | 0 | 0 COCO detections above threshold on interior wall views |
| `midas` | `succeeded` | `cuda` | 7.90 | 715.0 | 4 | Relative inverse depth |

---

## 2. Outdoor Validation (`datasets/south_building/images`, 4 photos, `domain="outdoor"`, `tile_scale=2`)

- **Total Wall Time:** `70.40s`
- **Total Observations:** `222` (`104` detection, `104` segmentation, `8` depth, `6` matching)
- **Masks Persisted (`MaskStore`):** `103`
- **Top Outdoor Classes:** `signage.pole` (13), `exterior.facade` (10), `road.footpath` (8), `street.bin` (7), `street.bench` (6), `building` (6), `utility.transformer` (6), `structural.beam` (5), `street.lamppost` (5), `utility.meter` (5), `decor/lighting/lamp` (4), `signage.sign` (4), `exterior.chimney` (3), `road.drain` (3), `exterior.roof` (3), `street.postbox` (2), `exterior.balcony` (2), `structural.arch` (2), `architecture/column` (2).
- **Small-Object Detections (9 total):** `utility.meter` (5), `road.drain` (3), `street.lamppost.fixture` (1).

| Engine | Status | Device | Wall Time (s) | Peak VRAM (MB) | Observations |
|---|---|---|---:|---:|---:|
| `lightglue_aliked` | `succeeded` | `cuda` | 2.29 | 1,444.7 | 3 |
| `grounding_dino` | `succeeded` | `cuda` | 44.27 | 1,927.2 | 103 |
| `sift` | `succeeded` | `cuda` | 1.09 | 89.6 | 3 |
| `sam2` | `succeeded` | `cuda` | 12.26 | 1,569.8 | 103 |
| `depth_anything` | `succeeded` | `cuda` | 1.93 | 329.7 | 4 |
| `maskrcnn` | `succeeded` | `cuda` | 2.31 | 1,553.5 | 2 |
| `midas` | `succeeded` | `cuda` | 5.53 | 716.2 | 4 |

---

## 3. Error Analysis & Known Limitations (Phase 9)

1. **High-Res OOM on `lightglue_aliked`:** On 4032×3024 (`12 MP`) indoor images, ALIKED/LightGlue attempts a `5.85 GiB` allocation and OOMs on 6 GB GPUs without input downscaling.
2. **SAM2 VRAM Spike on High-Res + High Box Count:** `sam2` reached `5,543 MB` on 4032×3024 indoor images with 92 boxes (`162s`), whereas on `south_building` it used `1,570 MB` (`12.3s`).
3. **`MultiEngineReport.vram_peak_mb`:** Reported `89.6 MB` (last wave) instead of `max(e.peak_vram_mb)` across all executed engines.
4. **Unvalidated Per-Class Accuracy:** Neither `real_room_capture` nor `south_building` has 2D/3D object bounding-box ground truth; precision/recall on real captures requires annotated CVAT/COCO review exports (`world_review_coco`).
