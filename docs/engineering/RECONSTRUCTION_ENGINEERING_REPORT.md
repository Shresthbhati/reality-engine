# Reconstruction Engineering Report

Branch `claude/reconstruction-accuracy-engineering-d0cf62`. Scope: make existing reconstruction more honest and measurable; no new pipeline.

## Defect audit (B1-B10 + found)

| # | Defect | Result |
|---|---|---|
| B1 | Relative/SfM coordinates treated as metres in evidence association (`MultiViewEntityObservation.scale_units` defaulted to `"METRES"`; South Building benchmark never set it; metre-constant tolerances 0.35/0.40/0.5 applied to gauge units) | **Reproduced, fixed.** Default is `UNSPECIFIED`; mixed units refused (`mixed_scale_units`); metre constants only for `METRES`, gauge units use extent-relative tolerance. |
| B2 | Sim(3)-fitted residuals presented as accuracy (`evaluate_absolute_poses`) | **Reproduced, fixed.** Report states `residual_is_independent_metric_accuracy: False`; `metric_*` rigid-alignment keys only when the estimate declares metric scale. |
| B3 | Low per-pose error with inconsistent pose-graph cycles | Diagnostics already exist (`perception/camera_pose_graph.py`); wrapped as `pose_graph_cycle_error` metric. Absolute poses are cycle-consistent by construction, so the metric needs pairwise edges (refused otherwise). |
| B4 | Disconnected / wide-baseline sets | **Not changed.** Observed: 25/32 registered; unregistered images are reported, not placed. Relative placement of components remains unknown. |
| B5 | Single image assumes camera / level | Verified already disclosed (`assumed` camera, 60 deg HFOV, level, confidence ceiling, recorded in metadata). Gravity not estimated. |
| B6 | Dense MVS not in canonical path | `dense_auto` is on in the job path; **found a real defect in its gate** (below). Dense genuinely ran in the 32-image benchmark. |
| B7 | Benchmarks disabled depth/perception/mesh/dense | New `scripts/real_photo_production_benchmark.py` runs the production path with all stages enabled. |
| B8 | No independent ground truth | **Unfixable here.** Reference is a COLMAP model of the same photos; `independent_ground_truth: false` recorded. |
| B9 | Scale-state vocabulary inconsistent | **Not unified.** At least four vocabularies coexist (`metric3d.ScaleState`, `depth_scale_alignment` strings, `metric_lifting` states, `world_scale.scale_kind`). Needs an owner decision. |
| B10 | Timeout/cache invalid outcomes | COLMAP timeouts already raise a typed `ReconstructionStepError`. **Found stale-cache defect:** depth cache key ignored output convention, so pre-`156f3e36` maps would be served as inverse-depth. Fixed with `DEPTH_CONVENTION`. |
| new | Dense gate counted monocular depth points as structure (median track length 1, structure inflated 5k -> 47k) | **Fixed** (`sparse_points`). Every earlier real run reported "median structure point seen by 1 camera". |

## Quality framework
`reconstruction/quality_metrics.py`: 10 metrics (registration rate, reprojection error, pose-graph cycle error, scale validity, point-cloud completeness, depth consistency with occlusion exclusion, mesh integrity incl. duplicate triangles and winding, surface coverage, uncertainty calibration, cross-view consistency). Each has formula, inputs, limitations in `METRICS` and a test; uncomputable inputs return `value: None` + reason. Reference-based metrics refuse unless `same_frame=True`.

## Real-photo results (South Building, 1024x768, COLMAP 4.2.0 CUDA, RTX 4050)
Files: `docs/benchmarks/real_photo_production_south_building_{12,32,32_dense}.json`.

| Run | Registered | Dense | Points | Completeness / precision* |
|---|---|---|---|---|
| 12 images | 3/12 | refused (gate: 25%, 77 pts) | 77 | 0.29 / 1.00 |
| 32 images, old gate | 25/32 | refused (78% < 80%, track=1 defect) | 47,514 | 0.992 / 0.9998 |
| 32 images, fixed gate + operator override `REALITY_DENSE_MIN_REGISTERED_FRACTION=0.75` | 25/32 | **ran, 126,799 dense points** | 186,527 | 0.995 / 0.997 |

\*After Sim(3) fit to the reference (scale fitted); tau = 2% of reference bbox diagonal in gauge units. Shape agreement only, **not metric accuracy**; scale state is `relative` in all runs. Pose: median rotation error 0.11 deg over 25 cameras (one camera 113 deg off - unresolved outlier). Median point-to-reference distance 0.157 -> 0.016 after dense, measured on a stride subsample (18,653 of 186,527) with max 46 units (outliers): more points are not claimed as more accurate; the distance change is reported as measured.
Depth ran in the 32 runs; perception and mesh stages were `skipped` (reasons not yet investigated). 0.75 is an operator override of an EXPERIMENTAL threshold, not a tuned default.

## Test evidence
- `tests/test_multi_view_scale_units.py` (4), `test_multi_view_association.py` + `test_task_13_9_mast3r_dinov2.py`: 28 passed.
- `tests/test_pose_evaluation_honesty.py` (3) + `test_pose_evaluation.py` + scenarios: 33 passed, 5 skipped.
- `tests/test_reconstruction_quality_metrics.py`: 11 passed.
- `tests/test_depth_cache.py`: 5 passed. `tests/test_dense_gate.py`: 11 passed.
- Not run: full suite (see the separate CI-recovery session).

## Changed-file matrix
| File | Change |
|---|---|
| `engine/multiengine/multi_view_association.py` | unit-honest defaults/tolerances, mixed-unit refusal |
| `reconstruction/evaluation.py` | honesty keys + rigid metric block |
| `reconstruction/quality_metrics.py` (new) | metric registry |
| `engine/multiengine/depth_cache.py` | `DEPTH_CONVENTION` in key |
| `engine/pipeline/dense_gate.py` | count triangulated points only |
| `scripts/stage_report.py` | exposes `LAST_PROG` |
| `scripts/real_photo_production_benchmark.py` (new) | production-path benchmark + scoring |
| tests: `test_multi_view_scale_units`, `test_pose_evaluation_honesty`, `test_reconstruction_quality_metrics`, `test_depth_cache`, `test_dense_gate` | regressions |

## Remaining defects
1. Scale-state vocabulary fragmentation (B9) - needs owner decision before unifying.
2. Disconnected components have no relative placement (B4).
3. 113 deg outlier camera in the 32-image run; not diagnosed.
4. Perception and mesh stages skipped in real runs; cause not investigated.
5. Dense gate thresholds (80% registration) experimental; 78% registration is borderline.
6. No independently measured ground truth anywhere (B8); metric accuracy cannot be claimed.
7. Pose-graph cycle check needs pairwise edges wired into the benchmark.
8. Large files (`colmap_backend.py` 1121 lines, `robust_fusion.py` 801) not refactored: no behaviour-preserving split was justified in this pass.
9. PRs: none opened (local commit only, per hold-push rule).
