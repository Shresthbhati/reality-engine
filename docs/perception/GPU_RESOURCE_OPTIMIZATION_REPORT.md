# GPU Resource Optimization Report (6 GB VRAM Target)

**Hardware Target:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GiB VRAM), PyTorch `2.14.1+cu130`
**Baseline Commit:** `48218d74` (PR #421)

## 1. Root Causes Identified

1. **`LightGlue-ALIKED` 6 GB OOM on High-Resolution Photos (`perception/engine_adapters.py`):**
   - On `datasets/real_room_capture` (`4 × 4032×3024`), `_lightglue_run` fed full 12 MP tensors directly into ALIKED/LightGlue, hitting `CUDA out of memory` at `5.85 GiB`.
   - Failed pairs were appended to `unverified` while `_lightglue_run` still returned `status="succeeded"`, and `hardware_class` was hardcoded to `"gpu"`.
2. **`MultiEngineReport.vram_peak_mb` Under-Reporting (`perception/engine_orchestrator.py`):**
   - `merge_engine_outputs` and `MultiEngineOrchestrator.run_all` reported only the last wave's `_get_vram_peak_mb()` (`89.625 MB`) instead of the maximum peak VRAM across all executed engines.
   - Inter-engine `gc.collect()` + `torch.cuda.empty_cache()` was not invoked after each engine execution, allowing failed/completed engine allocations to starve subsequent engines in the same process.
3. **Mask R-CNN Contract & Lazy Import (`perception/maskrcnn_adapter.py`):**
   - Top-level `import torch` broke torch-free test collection.
   - Segmentation `ObservationRecord` emitted `mask_rle` in `metadata` without setting `mask` or `bbox_px`, preventing `persist_masks()` from persisting Mask R-CNN masks into `MaskStore`, and omitted ontology grounding.
   - `torch.cuda.reset_peak_memory_stats(0)` was not called and `peak_vram_mb` was always `None`.
4. **Depth Adapter Offline & Partial-Failure Honesty (`perception/depth_adapter.py`):**
   - `from_pretrained` omitted `local_files_only=True`.
   - `hardware_class` was hardcoded to `"gpu"` regardless of device, and per-image failures were not tracked in `failed_images` or reflected in `status`.

## 2. Implemented Mitigations

- **`perception/engine_adapters.py` (`_lightglue_run`):**
  - Added `max_long_edge_px=1600` downscaling before tensor upload, rescaling matched keypoints back to original pixel coordinates.
  - Added `torch.cuda.reset_peak_memory_stats(0)` before extraction, set `hardware_class=device`, and graded `status` honestly (`"failed"` / `"partial"` / `"succeeded"`).
- **`perception/engine_orchestrator.py`:**
  - `merge_engine_outputs` and `MultiEngineOrchestrator.run_all` now compute `vram_peak_mb = max(...)` across all executed engines.
  - Added inter-engine `gc.collect()` + `torch.cuda.empty_cache()` in `_execute_one` after capturing `peak_vram_mb`.
- **`perception/maskrcnn_adapter.py`:**
  - Moved `torch` import inside `_maskrcnn_run` (lazy import).
  - Grounded labels with `ground_label(label)`, populated `ObservationRecord.mask` and `ObservationRecord.bbox_px` on segmentation records so `persist_masks()` stores them in `MaskStore`, reset CUDA peak memory stats, and reported actual `peak_vram_mb` and `failed_images` status.
- **`perception/depth_adapter.py`:**
  - Enforced `local_files_only=True` on `AutoImageProcessor.from_pretrained` and `AutoModelForDepthEstimation.from_pretrained`.
  - Added `torch.cuda.reset_peak_memory_stats(0)`, `hardware_class=device`, and `failed_images` status grading.

## 3. Before / After Measurements on RTX 4050 (`datasets/real_room_capture`, 4 × 4032×3024)

| Metric | Before (`48218d74` Baseline) | After (PR-A) |
| :--- | :--- | :--- |
| `lightglue_aliked` status | `failed` (CUDA OOM at `5.85 GiB`) | `succeeded` (`3` pairs, `3.10s`, `3075.5 MB` peak) |
| `grounding_dino` status | `succeeded` | `succeeded` (`228` obs, `32.32s`, `1846.1 MB` peak) |
| `sift` status | `succeeded` | `succeeded` (`3` obs, `5.18s`, `8.1 MB` peak) |
| `MultiEngineReport.vram_peak_mb` | `89.625 MB` (last-wave bug) | `11298.98 MB` (`max` across engines) |
| Total indoor `n_observations` | `195` (`92` masks) | `297` (`63` masks persisted) |
| Indoor wall time | `286.25s` | `183.71s` |

## 4. Remaining Hardware Bottleneck

- **SAM2 (`perception/segmentation_adapter.py`)**: On `4032×3024` indoor images, SAM2 still hit CUDA OOM on `IMG_0001..IMG_0003` (`1.81–1.91 GiB` allocation failure after `9.66–10.07 GiB` PyTorch/WDDM allocation), leaving `status="partial"` (`63` masks on `IMG_0000`). Because `perception/segmentation_adapter.py` is owned separately, SAM2 input tiling/downscaling was not altered in this PR.
