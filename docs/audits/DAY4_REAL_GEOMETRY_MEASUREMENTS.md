# Day 4–6 real-geometry measurements (multi-engine run, lineage reload, timeout hand-over)

Date: 2026-11-25. Machine: RTX 4050 Laptop GPU, 6141 MiB total VRAM, Python 3.14.7,
torch 2.14.0+cu130, COLMAP 4.2.0 (CUDA build), pycolmap 4.2.1.

This document records what was actually executed and measured. It does **not** claim
reconstruction accuracy: the only quantitative geometry check against an independent
reference is the COLMAP registration comparison in §2, and that reference is a model
shipped with the dataset, not a survey.

## 0. Reproduce

```bash
python scripts/run_multi_engine_real_capture.py --out datasets/south_building/runs/multi_engine_real
python scripts/run_south_building_e2e.py --gpu
```

Report: `datasets/south_building/runs/multi_engine_real/report.json`
(referenced directory `.../evidence-world/` is the run's Evidence Matrix + WorldStore,
generated and not committed).

Dataset: `datasets/south_building` — 32 real photographs, Panasonic DMC-TZ3, resized to
1024×768 (the manifest's `derivative_recipe`). Its manifest does **not** say synthetic.
`datasets/real_room_capture` was rejected by the script's own guard: its manifest says
`"dataset": "synthetic_room_capture"` and `is_physically_surveyed: false`. It is a fixture,
not a real room capture, so it was not used as real evidence.

Camera intrinsics: `datasets/south_building/sparse/cameras.txt` (SIMPLE_RADIAL, 3072×2304,
f = 2559.68 px), matched per image by name; all 32 committed images are in that model.

## 1. Pair strategy

`sequential_skip2`: strides 1 and 2 over the sorted image list = 61 pairs. Chosen without
consulting the reference model (content only). Overlap necessarily falls with stride; the
geometric verification then decides which pairs are real. Against the reference model's
co-visibility (≥ 50 shared 3D points, 94 pairs) the engine graph covered 39 pairs
(41.5%). Of the 59 verified pairs, 20 are not co-visible in the reference model at all, and
the other 74 reference pairs were never requested by the two strides — so this coverage is
bounded by the pair strategy, not only by matching quality.

## 2. Registration and geometric verification (real measurements)

Matching engines, through `MultiEngineOrchestrator` (`run_wave`, 61 pairs, 23.9 s total):

| engine | status | runtime | VRAM peak (allocated) | pairs verified | median inliers | median inlier ratio | median Sampson residual |
|---|---|---|---|---|---|---|---|
| lightglue_aliked (ALIKED n16 + LightGlue, CUDA) | succeeded | 14.71 s | 1373.4 MB | 61/61 | 385 | 0.572 | 0.128 px |
| sift (OpenCV, ratio 0.75 + RANSAC F) | succeeded | 9.09 s | 8.1 MB | 58/61 | 17 | 0.443 | 0.038 px |

Every verified pair carries keypoint counts, putative matches, RANSAC inliers, inlier
ratio, Sampson residual median/p90, and (because intrinsics are known) the essential matrix
and the recovered relative rotation + unit translation direction. Confidence is the measured
inlier ratio (SIFT) or LightGlue's own mean match score, both labelled
`calibration_quality = "unverified"`; neither is a probability of correctness.

Camera graph from verified pairs with ≥ 15 inliers: 32 nodes, 59 edges, **1 connected
component**, no isolated cameras, no degree-1 cameras.

Cycle consistency (28 triangles with three estimated pair rotations):

| statistic | value |
|---|---|
| median rotation residual | 21.37° |
| p90 | 111.93° |
| max | 155.98° |
| triangles over 5° | 27 / 28 |

The metric was sanity-checked on synthetic cameras with known relative rotations, where it
returns 0.0°, so this is a property of the data, not of the comparison. **Conclusion:** the
pair graph is connected and each pair is individually verified, but the pairwise essential
matrices do **not** agree around loops. This dataset's wide-baseline façade views are close
to planar, where the essential matrix is degenerate and `recoverPose` returns a rotation that
is locally consistent but not globally. A pose graph / bundle adjustment over these pairs is
still required, and the engine pair poses must not be presented as camera registration.

Independent registration for comparison (`scripts/run_south_building_e2e.py --gpu`, real
COLMAP SIFT + exhaustive matching + mapper, 37.4 s):

| measurement | value |
|---|---|
| cameras registered | 22 / 32 |
| rotation error vs reference (after Horn gauge alignment) | median 0.209°, mean 0.207°, max 0.504° |
| camera-centre error (model units, after gauge alignment) | median 0.0118, mean 0.0120, max 0.0327 |
| gauge scale ratio | 1.0752 |
| sparse points | 5463 |
| point-to-reference distance | median 0.0046, mean 0.0253, p90 0.0161, max 36.1 (model units) |
| run classification | `partial` — `registration_status=partial: result usable but incomplete` |

Scale is COLMAP's own gauge: no metric anchor exists for this dataset, so none of these
numbers is in metres.

## 3. Production multi-engine pass, world persistence and fresh-process reload

`apps.api.evidence_world_stage.run_for_job` on the same 32 images, `detection="legacy"`,
72.0 s for the whole job (flow + multi-engine pass + matrix import + world persist):

| engine | status | runtime | VRAM peak (allocated) | observations |
|---|---|---|---|---|
| grounding_dino (tiny) | succeeded | 18.02 s | 1816.5 MB | 474 |
| sam2 (sam2.1-hiera-small) | succeeded | 10.41 s | 1567.0 MB | 474 |
| lightglue_aliked | succeeded | 7.36 s | 1368.1 MB | 31 |
| sift | succeeded | 4.55 s | 8.1 MB | 31 |
| depth_anything (Depth Anything V2 small) | succeeded | 3.47 s | 312.5 MB | 32 |

Engine evidence imported into the run's Evidence Matrix: 1042 observations, 5 run records,
0 rejected, `calibrated_confidence: none` (nothing is calibrated). Observations are keyed by
content, and the lineage block lists 474 dino / 474 sam2 / 32 depth / 31 + 31 matching
content ids per engine.

World: `status = persisted`, 1024 entities, version `v-e1d516c7d90fb5bf`,
`reload_verified = true`; job `status = partial` (the detection stage ran the whole-image
detectors rather than the tiled pipeline).

Fresh-process reload (a new Python process, not the writer): re-opened the WorldStore,
`verify_version` returned no problems, loaded the version, and
`world.metadata["multi_engine_evidence"]` matched `multi_engine_evidence.json`
byte-for-byte in content, including the per-engine statuses and observation content ids.
The world's evidence lineage therefore survives a process restart.

## 4. Timeout → device hand-over (the fix in this change)

Before: `_call_with_timeout` abandoned a timed-out call in a daemon thread and returned
immediately, so the next engine could start while the abandoned one still held its CUDA
allocations — two engines on one 6 GB card, and a VRAM peak attributed to neither.

Now (`perception/engine_orchestrator.py`):

* the abandoned call's thread is returned and **drained** for
  `EngineSpec.drain_after_timeout_s` (default 60 s) before the next engine starts;
* `EngineExecution.wall_time_s` includes the drain wait, and `provenance` records
  `timeout_s`, `drain_wait_s`, `abandoned_thread_drained` and a `device_release` sentence;
* if the call is still running when the drain expires, the GPU is **quarantined**: further
  GPU engines are reported `unavailable` with `reason = "gpu_quarantined"` naming the
  holder, instead of being started next to it. The quarantine clears once the holder exits.

Tests: `tests/test_unified_engine_orchestration.py`
(`test_an_abandoned_engine_is_drained_before_the_next_engine_starts`,
`test_an_engine_that_outlives_the_drain_wait_keeps_the_gpu_quarantined`).

Also fixed in this change: LightGlue and ALIKED were constructed **inside the pair loop**,
reloading weights for every pair; both are now loaded once per run, and the models are
released before the adapter returns. kornia returns `matches`/`scores` as lists of per-image
tensors, which the previous code turned into a `torch.as_tensor` error on every pair; both
shapes are now handled.

## 5. Day 5 (depth and metric scale) — status

Depth Anything V2 executed on all 32 real photographs (3.47 s, 312.5 MB peak, 32
observations) and its observations reached the canonical matrix with
`scale_state = relative`. There is **no ground-truth depth** for this dataset and no metric
anchor, so:

* no AbsRel / RMSE / δ metrics were computed — there is nothing to score against;
* no metric scale was recovered or promoted; no camera height, object dimension or measured
  distance exists in the repo for this capture;
* depth was not lifted to 3D in this run (the production path needs registered cameras plus
  sparse points, which the COLMAP step provides — that integration is **not** validated here).

So Day 5 remains unimplemented beyond a real relative-depth execution with provenance.

## 6. Day 6 (architectural and object perception) — status

Grounding DINO produced 474 product-level detections and SAM 2 produced 474 masks on the 32
real photographs; both are imported as image-frame observations with provenance. Those
observations feed canonical entities (1024 entities persisted), but:

* there is no surveyed ground truth for this building, so entity correctness is unmeasured;
* the opening/host-wall representation is **not** implemented: openings are still plain
  detections and are not promoted to plan-view structural lines, so the known
  opening-promotion gap remains open;
* cross-view object identity association and architectural topology tests were **not**
  implemented in this change.

## 7. What is not done (Days 4–6 remainder)

* Pose graph / bundle adjustment over the engine pair graph (the cycle inconsistency above
  is the reason it is needed, and it is not built).
* Depth-to-geometry integration validation (camera conventions, units, outlier rejection,
  normals, duplicate geometry, cross-view disagreement).
* Metric-scale recovery from any independent reference (blocked: no metric reference data).
* Opening representation, cross-view identity hypotheses, architectural topology.
* Day 7 and later: not started.
