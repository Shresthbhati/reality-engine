# Candidate Model Comparison & Selection Benchmark

Follows Cross-Agent Protocol Rules 5, 6, 7, and 10 (`perception/evaluation/candidate_model_benchmark.py`).

## 1. Candidate Models, Checkpoints, Licenses & Hardware Requirements

| Model ID | Task | Checkpoint | License | Min VRAM |
| :--- | :--- | :--- | :--- | :--- |
| `grounding_dino` | Open-vocab & tiled small-object detection | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | ~2048 MB |
| `maskrcnn` | Closed-vocab detection & instance masks | `torchvision/maskrcnn_resnet50_fpn_v1` | BSD-3 / MIT | ~4000 MB |
| `florence2_ocr` | Nameplate/panel OCR (`<OCR>`) | `microsoft/Florence-2-base` | MIT | ~2048 MB |
| `lightglue_aliked` | Learned keypoint extraction & matching | `kornia/ALIKED-n16+LightGlue` | Apache-2.0 | ~1024 MB |
| `sift` | Classical keypoint matching | `opencv/SIFT+RANSAC` | Apache-2.0 | 0 MB (CPU) |
| `depth_anything` | Monocular relative depth | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | ~2048 MB |
| `midas` | Secondary monocular relative depth | `Intel/dpt-hybrid-midas` | MIT | ~2000 MB |

## 2. Measured RTX 4050 Baseline (`datasets/real_room_capture`, 4 × 4032×3024)

| Engine | Status | Wall Time | Peak VRAM | Observations |
| :--- | :--- | :--- | :--- | :--- |
| `lightglue_aliked` | `succeeded` | `3.10s` | `3075.5 MB` | `3` pairs |
| `grounding_dino` | `succeeded` | `32.32s` | `1846.1 MB` | `228` detections |
| `sift` | `succeeded` | `5.18s` | `8.1 MB` | `3` pairs |

## 3. Disagreement & Failure Policy
- **No silent fallbacks:** Unavailable or failed candidates are recorded with `status="failed"` / `"unavailable"` and their exact exception/reason.
- **Disagreement preservation:** `find_model_disagreements()` preserves both hypotheses whenever two models overlap (`IoU >= 0.5`) with different labels instead of averaging or dropping one silently.
- **Class-aware NMS:** `perception/detection/ontology_grounding.py:nms(..., labels=...)` only suppresses overlapping boxes that share the same normalized label, preserving adjacent small components (`switch` inside `switchboard`, `door_handle` on `door`).
