# Day 4–6 real-geometry measurements: corrected pair geometry, reference-rotation comparison, pose graph

Date: 2026-11-26. Machine: RTX 4050 Laptop GPU, 6141 MiB total VRAM, Python 3.14.7,
torch 2.14.0+cu130, COLMAP 4.2.0 (CUDA build), pycolmap 4.2.1.

This document records what was actually executed and measured. It supersedes the
2026-11-25 version of the same file, which reported a 21.37° median camera-cycle
disagreement and explained it as planar degeneracy. **That conclusion was wrong**, for
two reasons that are both reproduced below. The only quantitative geometry reference
here is the COLMAP model shipped with the dataset (`datasets/south_building/sparse`), not
a survey, so nothing below is a claim of survey accuracy or metric scale.

## 0. Reproduce

```bash
python scripts/run_multi_engine_real_capture.py --out datasets/south_building/runs/multi_engine_real
python scripts/run_south_building_e2e.py --gpu          # independent COLMAP registration
python -m pytest tests/test_two_view_geometry.py tests/test_camera_pose_graph.py -q
```

Report: `datasets/south_building/runs/multi_engine_real/report.json` (the referenced
`.../evidence-world/` directory is the run's Evidence Matrix + WorldStore, generated and
not committed). Dataset: `datasets/south_building` — 32 real photographs, Panasonic
DMC-TZ3, resized to 1024×768. Its manifest does not say synthetic; `datasets/real_room_capture`
is still refused by the script's own guard (`is_physically_surveyed: false`).

## 1. Two defects in the verification (why the 21° figure was wrong)

### 1.1 The cycle residual multiplied its rotations in the wrong order

`_camera_graph` computed `angle(R_ab @ R_bc @ R_ac.T)`. The closure is
`R_bc @ R_ab @ R_ac.T`: composing a→b then b→c must reproduce a→c, and rotation
composition does not commute. With three distinct random consistent rotations the old
formula returns 100.35°, 137.87°, 110.46°; the correct closure returns 0.0°.

The previous version of this document claimed the metric "was sanity-checked on synthetic
cameras with known relative rotations, where it returns 0.0°". That check is not in the
tree and is not reproducible: any test with distinct rotations fails. A regression test
now pins the closure
(`tests/test_camera_pose_graph.py::test_cycle_residual_is_zero_for_consistent_poses`).

### 1.2 The intrinsics were the reference model's, applied to resized images

`sparse/cameras.txt` calibrates the ORIGINAL frames: `SIMPLE_RADIAL 3072×2304
f=2559.68, cx=1536, cy=1152, k1=-0.0205`. The committed photographs are 1024×768
derivatives. Name matching alone put the principal point at (1536, 1152) — outside a
1024×768 image — so every `K` was wrong, every `undistortPoints` normalisation was wrong,
and therefore every essential matrix and every recovered rotation was wrong, while the
pixel-space Sampson residual stayed at 0.13 px and looked excellent. The pair rotation
error against the reference model before this fix was a median **55.1°** (0 of 59 pairs
within 2°).

Fixes: each image's camera is rescaled to the image's actual size (exact for a pure
resize), the `SIMPLE_RADIAL`/`RADIAL`/`OPENCV` distortion coefficients are parsed and
passed to `undistortPoints`, and each pair record states which calibration it used
(`relative_pose.calibration`). The distortion term is not cosmetic: on the 31 stride-1
pairs, supplying `k1` moved the median rotation error against the reference from 0.84° to
0.65° and cut the coherent rotation bias (mean error log-vector 0.70° → 0.29°).

## 2. Pair strategy

`sequential_skip2`: strides 1 and 2 over the sorted image list = 61 pairs, chosen without
consulting the reference model. Against the reference co-visibility graph (≥ 50 shared 3D
points, 94 pairs) the engine graph covers 39 pairs (41.5%). 20 verified pairs are not
co-visible in the reference model at all; the other 74 reference pairs were never
requested, so coverage is bounded by the pair strategy, not only by matching quality.

## 3. Registration and geometric verification (real measurements)

Matching engines through `MultiEngineOrchestrator.run_wave`, 61 pairs, 26.5 s total:

| engine | status | runtime | VRAM peak (allocated) | pairs verified | median inliers | median inlier ratio | median Sampson residual |
|---|---|---|---|---|---|---|---|
| lightglue_aliked (ALIKED n16 + LightGlue, CUDA) | succeeded | 16.03 s | 1373.4 MB | 61/61 | 385 | 0.572 | 0.128 px |
| sift (OpenCV, ratio 0.75 + RANSAC F) | succeeded | 10.07 s | 8.1 MB | 58/61 | 17 | 0.443 | 0.038 px |

Camera graph from verified pairs with ≥ 15 inliers: 32 nodes, 59 edges, 1 connected
component, no isolated or degree-1 cameras.

### 3.1 Estimated pair rotations vs the reference model (the decisive measurement)

A relative rotation does not depend on the world gauge, so the reference model can score
every pair directly: `R_est @ R_ref.T` with `R_ref = R_b @ R_a.T` from the reference
poses (`perception/camera_pose_graph.py::compare_pair_rotations`).

| measurement | value |
|---|---|
| pairs compared | 59 / 59 |
| median rotation error | **0.80°** |
| mean rotation error | 10.53° |
| p90 | 18.68° |
| p95 | 90.03° |
| max | 176.42° |
| pairs within 2° | 48 |
| pairs beyond 30° | 6 |
| median translation direction error (in camera-a frame) | 0.68° |
| p95 translation direction error | 82.87° |

So the pair geometry is **good, not degenerate**: 48 of 59 pairs reproduce the reference
relative rotation to within 2°. The six exceptions (inliers in brackets) are
0225–0308 (52, 176.4°), 0193–0201 (25, 99.7°), 0221–0308 (47, 95.6°), 0181–0189 (66,
89.4°), 0308–0316 (73, 58.9°), 0153–0161 (18, 42.4°). These are single pairs where the
essential-matrix decomposition returned the wrong branch or the correspondences are wrong;
they are not a dataset-wide property. Degeneracy is localised, and the cases with 250+
inliers in the second capture sequence are fine (see §4).

### 3.2 Corrected cycle consistency

| statistic | before (wrong closure) | after (correct closure) |
|---|---|---|
| median rotation residual | 21.37° | **1.26°** |
| p90 | 111.93° | 88.97° |
| max | 155.98° | 176.73° |
| triangles over 5° | 27 / 28 | **7 / 28** |

The 7 remaining over-threshold triangles are exactly the triangles that contain the six
wrong pairs above (worst: 0225–0308–0312 at 176.73°); the other 21 triangles are
consistent. The dataset is not globally degenerate — it has a handful of bad pair
estimates.

## 4. Pose graph over the verified pairs

`perception/camera_pose_graph.py::solve_pose_graph`: maximum-weight spanning forest
initialisation (Kruskal by RANSAC inliers), IRLS rotation averaging on SO(3) (right
perturbations, Huber 1°, one pinned root per connected component), edges whose residual
exceeds 5° are rejected with their reason and the fit is redone on the accepted edges, and
bridge edges (no cycle through them) are reported because a part hinged on a single edge
can be rotated about it without changing any residual.

| measurement | value |
|---|---|
| nodes / edges given | 32 / 59 |
| fitting passes | 2 |
| edges accepted | 52 |
| edges rejected | 7 |
| residual of accepted edges (median / p90 / max) | 0.27° / 1.21° / 3.30° |
| connected components | 1 |
| bridge edges | 4 |
| rotation-observable cameras (at least one accepted edge on a cycle) | **30** |
| rotation-unobservable cameras | **2** (P1180308, P1180332 — every incident edge is a bridge) |
| **per-camera rotation error, sequence 0141–0225 (22 cameras)** | **1.77° – 8.81°, median 6.4°** |
| per-camera rotation error, sequence 0308–0344 (10 cameras) | 167.03° – 169.37° |
| per-camera rotation error, all 32 (median / mean / p90 / p95 / max) | 7.27° / 56.40° / 168.08° / 168.84° / 169.37° |
| per-camera rotation error, observable cameras only (30) | median 6.83°, mean 49.00°, p95 168.88°, max 169.37° |
| cameras within 2° / within 5° | 3 / 7 |

Rejected edges (residual from the averaged solution, inliers):

| pair | residual | inliers |
|---|---|---|
| 0225–0312 | 176.85° | 16 |
| 0193–0201 | 100.98° | 25 |
| 0221–0308 | 89.06° | 47 |
| 0181–0189 | 88.83° | 66 |
| 0308–0316 | 58.25° | 73 |
| 0153–0161 | 41.19° | 18 |
| 0332–0340 | 14.81° | 40 |

Two independent criteria agree: every one of the six pairs the reference model calls wrong
(§3.1) is rejected or is a bridge, and the 14.81° edge (0332–0340, 40 inliers, 12.7° vs the
reference) is rejected as well.

### 4.1 Why the second sequence sits 168° away — and why that is not an accuracy result

The 0308–0344 sequence is internally excellent: its pairs are 0.3–1.7° from the reference.
But its ONLY verified links to the first sequence are 0225–0308 (52 inliers, 176.4° wrong)
and 0221–0308 (47 inliers, rejected); 0225–0312 (16 inliers) is also wrong. So the whole
sequence hangs on **one bridge** (0225–0308) that the reference model says is wrong, and a
rigid rotation of the sequence about that bridge changes no accepted residual. The reported
168° is therefore an unobservable degree of freedom of this pair set, not a measured
accuracy. The bridge is listed in `pose_graph.bridges` and named in
`reference_comparison.bridge_edges`, and no registration claim is made for that sequence.

### 4.2 Is the 6.4° of the first sequence consistent with 0.80° pairs?

No — it is larger than independent noise explains. Monte-Carlo over the same graph with
exact reference rotations perturbed by independent 0.8° per pair gives 1.29° per camera
(1.5° per pair → 2.43°). The real 6.4° means the pair errors are *correlated*: a small
coherent bias per pair accumulates along the 22-camera chain. That is why COLMAP's global
bundle adjustment over 3D points reaches 0.209° median rotation error on 22 cameras
(§5) while rotation-only averaging over two-view pairs does not. A rotation-only pose
graph is not a substitute for bundle adjustment, and none is claimed here.

### 4.3 Camera centres — withheld, because the translation solution is not physical

Translation averaging uses the measured unit translations (`c_b - c_a = s_e d_e`, one
unknown baseline per edge, bounded non-negative, one baseline fixed as the gauge). On this
capture the solve reports `status = "ambiguous"`: one baseline is pinned at zero, which
folds two cameras onto one point, and the maximum direction residual is 0.99. Per the rule
that centre accuracy must not be reported when the translation solution is unobservable or
ambiguous, `reference_comparison.centre_error_model_units` is
`{"status": "unavailable"}` — **no centre error is claimed for this dataset at all**. The
earlier unconstrained version reproduced coincident cameras even with EXACT reference
directions (10 pairs at one point, pairwise distance ratios 1.05–14.4 against the truth),
so the direction-only translation path is a known, documented gap rather than a measurement.

What the solver can and cannot see, measured on synthetic scenes (`§4.4`): zero and negative
baselines, layout rank < 3, and coincident cameras are detected and reported; a globally
mirrored layout (all bearings negated) is a *consistent* solution of the direction
constraints and cannot be caught by residuals — only the reference comparison or a
cheirality/3D-point check can, and no such check exists yet.

### 4.4 What the pose graph is verified against (synthetic, noise-free)

`tests/test_camera_pose_graph.py`, 30 tests, all passing. The ones that pin conventions and
degeneracies rather than happy paths:

| test | what it pins |
|---|---|
| `test_relative_rotation_composes_world_to_camera_poses` | `R_b = R_ab R_a`, det = +1, orthonormal |
| `test_camera_centre_and_translation_follow_the_world_to_camera_convention` | `t = -R c`; pair `t = t_b - R_b R_a^T t_a` (not `t_b - t_a` — different frames) |
| `test_relative_pose_inversion_transposes_the_rotation_and_flips_the_baseline` | inverse edge is `(R^T, -R^T t)` |
| `test_longer_loops_close_only_with_the_closure_product_order` | 5-cycle closes at 0.0° in the right order, > 20° in the wrong one |
| `test_randomised_graphs_with_known_ground_truth` (5 seeds) | 12 cameras, random spanning tree + redundant edges: rotations and centres exact (< 1e-4° / < 1e-3) |
| `test_a_reflection_edge_is_rejected_not_averaged` | det = -1 input is rejected with its reason, never orthonormalised into a fake rotation |
| `test_a_rotation_rounded_to_six_decimals_is_still_a_rotation` | serialised rounding (det 0.999999) is not treated as a defect — a 1e-6 threshold wrongly dropped five real edges |
| `test_a_consistent_frame_mistake_is_invisible_to_cycles_and_caught_by_the_reference` | a graph given entirely in the inverse convention is self-consistent (0° residuals) and is only exposed by the reference comparison |
| `test_a_planar_camera_layout_withholds_the_centre_accuracy` | rank-2 layout → translation ambiguous, centre error withheld, rotations still scored |
| `test_a_flipped_bearing_convention_produces_a_mirrored_layout` | mirrored bearings give a mirrored (consistent) layout — documented limit, centre error exposes it |
| `test_a_camera_hinged_on_one_bridge_is_named_unobservable` | cameras whose only edges are bridges are named unobservable |
| `test_translation_baselines_are_positive_on_a_noise_free_scene` | every baseline > 0, direction residual < 1e-6, scale reported as a gauge |

Rotation uncertainty is reported separately from translation uncertainty:
`pose_graph.rotation_uncertainty_deg` (per camera: RMS of incident accepted-edge residuals
/ sqrt(deg-1)) and `pose_graph.translation.direction_residual` plus the ambiguity status.
Neither is a confidence interval, and neither is used as an accuracy claim.

## 5. Independent COLMAP registration for comparison

`scripts/run_south_building_e2e.py --gpu` (SIFT + exhaustive matching + mapper, 37.4 s,
unchanged from the earlier run): 22/32 cameras registered, rotation error vs the reference
median 0.209° (max 0.504°) after gauge alignment, camera-centre error median 0.0118, gauge
scale ratio 1.0752, 5463 sparse points, classification `partial`. Scale is COLMAP's own
gauge; no metric anchor exists for this dataset, so none of these numbers is in metres.

## 6. Production multi-engine pass, world persistence and fresh-process reload

`apps.api.evidence_world_stage.run_for_job` on the same 32 images, `detection="legacy"`,
72.1 s for the whole job:

| engine | status | runtime | VRAM peak (allocated) | observations |
|---|---|---|---|---|
| grounding_dino (tiny) | succeeded | 17.87 s | 1816.5 MB | 474 |
| sam2 (sam2.1-hiera-small) | succeeded | 10.11 s | 1567.0 MB | 474 |
| lightglue_aliked | succeeded | 7.28 s | 1368.1 MB | 31 |
| sift | succeeded | 4.47 s | 8.1 MB | 31 |
| depth_anything (Depth Anything V2 small) | succeeded | 3.37 s | 312.5 MB | 32 |

Evidence Matrix: 1042 observations imported, 5 run records, `calibrated_confidence: none`.
World: `persisted`, version `v-7b0509f7df443c51`, 1024 entities, `reload_verified = true`,
job `status = partial` (whole-image detectors rather than the tiled pipeline).

Fresh-process reload: a new Python process (pid 25060) re-opened the WorldStore,
`verify_version` returned no problems, and `world.metadata["multi_engine_evidence"]`
matched `multi_engine_evidence.json` including per-engine statuses (all five `succeeded`)
and content ids (474 / 474 / 32 / 31 / 31). The evidence lineage survives a process
restart.

## 7. Timeout → device hand-over

`_call_with_timeout` used to abandon a timed-out call in a daemon thread and return at
once, so the next engine could start while the abandoned one still held its CUDA
allocations. Now the abandoned thread is **drained** for `EngineSpec.drain_after_timeout_s`
(default 60 s) before the next engine starts, `wall_time_s` includes the drain wait,
provenance records `timeout_s` / `drain_wait_s` / `abandoned_thread_drained` /
`device_release`, and if the drain expires the GPU is **quarantined** (further GPU engines
report `unavailable` with `reason = "gpu_quarantined"`) until the holder exits. Tests:
`tests/test_unified_engine_orchestration.py`.

Also fixed earlier and still in force: ALIKED/LightGlue are loaded once per run instead of
per pair, both list- and tensor-shaped kornia `matches`/`scores` are handled, and models
are released before the adapter returns.

## 8. Day 5 (depth and metric scale) — status

Depth Anything V2 ran on all 32 real photographs (3.58 s, 312.5 MB peak, 32 observations)
with `scale_state = relative`. There is no ground-truth depth and no metric anchor for
this dataset, so no AbsRel/RMSE/δ metrics exist to compute and no metric scale was
recovered or promoted. Depth was not lifted to 3D in this run; that integration is still
not validated.

## 9. Day 6 (architectural and object perception) — status

Grounding DINO (474 detections) and SAM 2 (474 masks) executed and reached the canonical
matrix and world. There is no surveyed ground truth for this building, so entity
correctness is unmeasured; the opening/host-wall representation, cross-view identity
association and architectural topology tests are **not** implemented.

## 10. What is not done

* Bundle adjustment over 3D points (the only path to COLMAP-level rotation accuracy; the
  6.4° rotation-only figure above is the measured consequence of skipping it).
* Camera-centre placement accuracy: the translation solve is weak and unvalidated (§4.3).
* Wider pair strategy: 41.5% reference co-visibility coverage, and the second sequence has
  a single (wrong) bridge to the first — more pairs would give it a verifiable cycle.
* Depth-to-geometry integration, metric scale, opening representation, architectural
  topology.
* Day 7 and later: not started.
