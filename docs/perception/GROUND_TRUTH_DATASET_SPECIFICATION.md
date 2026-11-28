# Ground-Truth Dataset Specification

## 1. Schema (`datasets/gt_perception_benchmark/annotations.json`)
Extends standard COCO detection/segmentation JSON with:
- `annotations[].track_id`: Canonical physical instance ID shared across multiple camera views (`indoor_view_0.jpg`, `indoor_view_1.jpg`).
- `annotations[].is_small`: Boolean flag for small architectural/MEP details (`< 32×32` px or `< 15 cm` physical extent: switches, sockets, door handles).
- `relationships[]`: Directed scene-graph ground truth (`source_track_id`, `target_track_id`, `kind`).

## 2. Benchmark Metrics (`perception/evaluation/gt_benchmark.py`)
- **Detection:** COCO `AP@0.50`, `AP@0.50:0.95` (10 IoU thresholds `0.50..0.95`), per-class AP, and `small_object_recall@0.50`.
- **Segmentation:** `mean_mask_iou` over matched predictions carrying masks.
- **Multi-View Identity:** Pairwise same-track `precision`, `recall`, `F1`, and `id_switches`.
- **Scene Graph:** Directed relationship `precision`, `recall`, `F1`.

## 3. Separation from Unannotated Real-Photo Runs
- Real captures (`datasets/real_room_capture`, `datasets/south_building/images`) without human-labeled COCO GT report only execution counts, runtime, VRAM, and persisted masks — never fabricated AP/precision/recall.
