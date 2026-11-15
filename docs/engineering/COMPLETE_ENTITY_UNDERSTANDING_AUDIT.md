# Complete entity understanding: audit and plan

Requirement: detect, identify, count, locate and reconstruct every observable entity in the test photographs, keep
identities across views, and persist one consistent world. Evaluation contract: `benchmarks/reality_benchmark/`.

Every statement below is labelled **observed** (read in the code, the environment, or measured in a run this session)
or **not checked**. Nothing here is a measured detection result: there are **no annotations** of the test images, so no
precision, recall, F1 or mAP has been measured.

## 1. Implemented and wired into the live job path (observed)

`apps/api/jobs.py` -> `engine/pipeline/progressive.py` -> `vertical_slice` stages.

| Capability | Where | Notes |
|---|---|---|
| Sparse SfM, camera poses | `reconstruction/backend/colmap_backend.py` | seed pinned, retried on weak registration |
| Monocular depth, metricised per view | `engine/pipeline/vertical_slice.py::_depth_stage`, DPT_Hybrid | scale-only alignment (see bottlenecks) |
| Open-vocabulary detection | `perception/detection/grounded_scene.py` (Grounding DINO tiny, 19 prompts) | classes limited to the prompt list |
| Box -> object lift, cross-view association | `perception/instances/{lifting,association,object_resolution}.py` | Mahalanobis gate, epipolar check, appearance term |
| Planes, roles, consolidation | `perception/geometry/{planes,orientation}.py` | floors/ceilings/walls merged (committed) |
| Room footprint, layout, wall/floor/ceiling slabs | `perception/architecture/{room_semantics,room_layout,wall_slabs}.py` | footprint from plan-view occupancy |
| Material classification (12 classes, PBR) | `engine/compiler/world_compiler.py` -> `perception/materials/{material_hypotheses,classifier,room_surfaces}.py` | **room surfaces only** (estimated hypotheses); objects get none |
| Validation, versioned store | `world_ir/validation.py`, `worldstore` | invariants incl. duplicate entities |

## 2. Implemented but NOT on the live path (observed: no importer in `engine/` or `apps/api/`)

| Capability | Where | Evidence |
|---|---|---|
| Entity taxonomy (sockets, switches, plumbing, HVAC ...) | `perception/entities/taxonomy.py` | its own docstring: "registered, not supported" |
| Small-object coarse-to-fine tiling pass | `perception/entities/small_objects.py` | imported only by `perception/fine_grained/pipeline.py` |
| Switchboard module decomposition | `perception/entities/switchboard.py` | imported by `engine/pipeline/mep_topology.py` only |
| Fixture-level adaptive inspection | `perception/detail/adaptive_inspection.py` | no importer |
| Fine-grained perception pipeline | `perception/fine_grained/` | used by `engine/orchestrator/universal_pipeline.py` (SDK path), not by the API job path |
| DINOv2 re-identification | `perception/instances/dinov2_reid.py` | exported from `instances/__init__` only |
| SAM / SAM 2 masks | `perception/segmentation/{sam,sam2}_backend.py` | `segment_anything` and `sam2` packages are **not installed**; no SAM weights in the caches |

## 3. Missing (observed)

* **No per-image entity inventory anywhere** (`grep inventory` finds nothing in perception/engine/evidence/apps/world_ir).
  Promoted furniture entities keep `n_views` and `evidence_ids` only (`perception/instances/furniture_from_detections.py`);
  the per-view box, mask and score are not persisted in WorldIR, so an inventory cannot be rebuilt from a stored world.
* **Annotations of the 15 test images** (boxes, instance identities across views): none exist. Required for every metric.
* Part-level classes (cushions, legs, drawers, individual steps), instance counting, occlusion state: not produced by the live path.
* Mask-based metrics (instance mIoU, boundary F1): no masks on the live path.
* Camera-pose, Chamfer and dimension metrics: need surveyed truth; none exists for these images.

## 4. Bottlenecks measured this session

* One full run of 10 photos takes ~13 minutes (Grounding DINO on 19 prompts per photo dominates).
* COLMAP geometry differs between runs on the same photos (995 vs 1,423 sparse points; scene sometimes entirely above the
  cameras), so multi-view results are not reproducible even though registration (9/10) is.
* Per-view depth (DPT_Hybrid, a scale per view) fits the sparse cloud to a relative residual of 0.199; a metric model fits it
  better (0.117) with one scale. Neither changed how many "floor" levels the compile finds (about 950 floor-role plane
  candidates before consolidation in every variant): those are mostly furniture tops and steps labelled floor because their
  normals point up, not a depth artifact (TECHNOLOGY_EVALUATION 3a). An earlier affine-fit experiment was reverted.
* GPU is an RTX 4050 with 6 GB.
* Missing libraries: `segment_anything`, `sam2`, `open3d`, `pycolmap`, `ultralytics`.

## 5. Reusable technology already present (observed in caches/code)

Grounding DINO tiny, OWL-ViT, Mask R-CNN, Depth-Anything-V2 (small and metric-indoor small), DPT_Hybrid, MiDaS small,
DINOv2 ViT-S/14, UPerNet ConvNeXt-tiny, ALIKED/DISK + LightGlue weights, trimesh, OpenCV, torch, transformers, timm.

## 6. Built in this change

`benchmarks/reality_benchmark/inventory.py`: the per-image inventory contract (mandatory fields; an unmeasured field must be
`None` with a stated reason, never silently dropped) and an evaluator that scores any inventory against annotations:
precision, recall, F1, mAP@50, mAP@50:95, classification accuracy, duplicate-entity rate, cross-view identity,
merge errors, and per-group small-object precision/recall/count accuracy, emitted as benchmark `metric_results`.
Verified on synthetic annotations only (`tests/test_reality_inventory.py`).

## 7. Plan, in dependency order

1. **Annotations** of the 15 images (boxes + cross-view instance ids). Human-reviewed; a model may pre-label but its labels are
   not truth until reviewed. Blocks every measurement.
2. **Persist per-view observations** (box, mask, score, label, view id) through promotion so `inventory_from_world` can exist.
3. **Wire the small-object pass and taxonomy prompts** into the live perception stage (tile only wall regions; budget the time).
4. **Masks**: install SAM 2 and its weights (needs approval for the download), prompt it with the boxes.
5. **Re-identification**: feed DINOv2 embeddings into the association resolver.
6. **Depth consistency**: cross-view depth agreement before dense points are trusted; revisit the affine fit with it.
7. **Run the benchmark** on the annotated images and report measured results, including failures.
