# Expanded Perception & Production Connection Benchmark Results

**Base Commit:** `f5ac660e` (`origin/main`)
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GiB VRAM)

## 1. Expanded Ground-Truth Dataset (`datasets/gt_perception_benchmark/annotations.json`)
- **64** COCO + `track_id` annotations across **8** multi-view images (4 indoor, 4 outdoor) and **22** categories:
  - Indoor & MEP/Hardware: `wall`, `door`, `switch`, `socket`, `plug`, `door_handle`, `hinge`, `vent`, `light_fixture`, `pipe`, `utility_meter`, `plant`, `chair`
  - Outdoor & Infrastructure: `building_facade`, `road`, `footpath`, `kerb`, `lamppost`, `traffic_sign`, `drain`, `manhole`, `tree`
- **10** directed `relationships[]` (`attached_to`, `part_of`, `adjacent_to`)
- **4** `ocr_regions[]` (`utility_meter`, `traffic_sign`)
- **Condition Coverage:** `normal`, `low_light`, `glare`, `reflective`, `motion_blur`, `occluded`, `wide_baseline`, `is_small`.

## 2. Production Connection on Real Photo (`datasets/real_room_capture/IMG_0000.jpg`, RTX 4050)
Connected `grounding_dino` (`tile_scale=2`, `query_budget=40`, `small_share=0.4`) directly to `connect_attributes_and_scene_graph()`:
- `grounding_dino` status: `succeeded` (`1847.66 MB` peak VRAM, `63` `ObservationRecord`s)
- `extract_observed_attributes`: `63` `EntityAttributeBundle`s, **`320`** `observed` attributes (`0` fabricated priors)
- `build_scene_graph_from_observations`: **`216`** `observed` scene-graph edges, `0` `inferred` priors

## 3. Per-Class & OCR Evaluation (`perception/evaluation/production_eval.py`)
- `evaluate_per_class_metrics()` computes per-class Precision, Recall, F1, AP@0.50, AP@0.50:0.95, and mean center localization error (px).
- `evaluate_ocr_candidates()` computes exact match, normalized match, and CER, reporting `status="unavailable"` honestly when a candidate checkpoint (e.g. `Florence-2`) is not cached locally.
- Verified by `tests/test_perception_expanded_eval.py` (3/3 passing) and `ruff check` clean.
