# Perception Benchmark Specification (Phases 5, 7, 8, 9)

**Scope:** Independent, leakage-free evaluation of detection, small-object recognition, segmentation, multi-view identity, geometry, spatial relationships, and operational resource usage across **Indoor** and **Outdoor** capability profiles.

---

## 1. Metric Definitions (Separate by Task — Never Collapsed)

1. **Detection (`perception/evaluation/small_object_benchmark.py`, `benchmarks/annotation_review.py`):**
   - Class-aware IoU matching (`IoU >= 0.40` / `0.50` AND `normalized_pred_label == normalized_gt_label`).
   - Metrics: Precision, Recall, F1, per-class breakdown (`count`, `detected`, `recall`), and small-object vs large-object stratification.
2. **Segmentation (`perception/mask_persistence.py`, `world_ir/mask_store.py`):**
   - IoU, Dice score, mask pixel area fidelity, and RLE round-trip integrity.
3. **Multi-View Identity (`engine/multiengine/multi_view_association.py`):**
   - `evaluate_association_benchmark()`: Association Precision, Recall, F1, False Merge Rate, False Split Rate, and occlusion/repeated-structure/missing-calibration slices.
   - Reference dataset: `datasets/real_room_capture/ground_truth_identities.json` and `datasets/scene_disjoint_evaluation/ground_truth_identities.json`.
4. **Geometry (`benchmarks/measured_rooms/`):**
   - Point-to-point / point-to-plane error, surface distance, completeness, and scale error.
   - **Representation Hierarchy (Phase 5):** `(1) Observed 2D region -> (2) Camera-space observation -> (3) Approximate 3D location -> (4) Geometric primitive -> (5) Reconstructed surface -> (6) Detailed geometry`. Metric accuracy is never claimed without calibration or laser-survey ground truth.
5. **Spatial Relationships (`world_ir/relations.py`):**
   - Relationship precision/recall and `validate_scene_graph_consistency()` violation rate (dangling refs, attachment gap > `0.25m`, contradictory parents).
6. **Operational Performance:**
   - Wall-clock runtime (s), per-engine Peak VRAM (MB, reset via `torch.cuda.reset_peak_memory_stats(0)`), CPU RSS, images/minute, failure rate, and offline reproducibility (`local_files_only=True`).

---

## 2. Indoor vs Outdoor Capability Profiles & Unsupported Conditions (Phase 7)

| Profile | Datasets | Supported / Partially Supported | Explicitly Unsupported Conditions |
|---|---|---|---|
| **Indoor** | `datasets/real_room_capture/`, `datasets/room_capture/`, `benchmarks/measured_rooms/room_01_living_studio/` | Rooms, walls, floors, ceilings, doors/windows (partial), furniture, tiled small electrical/architectural prompts | Night/low-light scenes, severe motion blur, mirror/glass reflections, heavy occlusion, textureless white walls without SfM features |
| **Outdoor** | `datasets/south_building/`, `datasets/city_osm/` | Facades, trees, signs, lampposts, roads/footpaths (2D open-vocab; 3D outdoor geometry partial) | Wide outdoor baselines, dense foliage/canopy 3D meshing, repetitive facades, night scenes |

---

## 3. Dataset Splits & Ground-Truth Policy

- **No Leakage:** Tuning/development uses `datasets/room_capture/`; held-out evaluation uses `datasets/real_room_capture/`, `datasets/south_building/`, and `datasets/scene_disjoint_evaluation/`.
- **Unscored Smoke vs Ground Truth:** Runs on datasets without 3D object annotations (e.g. `datasets/south_building` object boxes) are labelled **unscored operational/detection runs**, never accuracy claims. Synthetic data is never labelled `physical`.
