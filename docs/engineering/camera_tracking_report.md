# Camera tracking: contract, SLAM/VIO compatibility, measured pose benchmark

Owner: camera-motion / tracking workstream. Status: **contract + validation + benchmark landed; no SLAM/VIO adapter
(not justified yet, see §3).** This is an engineering report, not a roadmap; priorities stay in `RELEASE.md`.

## 1. Contract (pose evidence)

| Item | Definition |
|---|---|
| Input | images (photo set or ordered video keyframes); optional `imu` evidence (§4) |
| Output | `{image_name: (R 3x3, t 3)}`, **world→camera**, `x_cam = R X + t` (same as `perception/camera_pose_graph.py`) |
| Frame | arbitrary world gauge per reconstruction; never compared without alignment |
| Units / scale | every pose set carries `scale_state ∈ {metric, relative}`; `relative` is the default for image-only. No metric scale is assumed anywhere in `perception/pose_validation.py` |
| Uncertainty | none produced by COLMAP backends today (**gap**); validation reports measured error vs a reference, not a covariance |
| Provenance | backend + version, commit SHA, image ids, config, runtime, peak VRAM (recorded by `scripts/pose_benchmark.py`) |
| Hardware | RTX 4050 6 GB Windows 11 target; CPU fallback must be reported as such |
| Failure | unavailable ⇒ report unavailable (`refused`/`unavailable`), never a substitute pose. Unregistered images are listed by name |
| Consumer | existing COLMAP backend / `engine/pipeline/progressive.py` ← poses; WorldIR evidence contract owned by Kilo, integration registry by OpenCode (not touched here) |

## 2. What was added (preserves all existing behaviour)

* `perception/pose_validation.py`
  * `evaluate_trajectory`: ATE (centres) + rotation error after **robust** (RANSAC + inlier refit) Sim(3), or rigid
    SE(3) only when *both* sides are metric; alignment-free RPE for ordered sequences; named outliers.
    The existing `reconstruction.evaluation.evaluate_absolute_poses` is plain least squares, so one wrong camera shifts
    the gauge and makes good cameras look bad; it is left unchanged.
  * `validate_poses`: reference-free: non-finite / non-rotation poses, coincident centres, disconnected components
    (from supplied edges), per-frame rotation jumps and trajectory discontinuities (MAD + 3×median floor).
* `scripts/pose_benchmark.py`: same images / SIFT / exhaustive matches, mapper varied; writes a full-provenance JSON.
* `tests/test_pose_validation.py` (9 tests, synthetic ring with an injected 113° rotation outlier, position outlier,
  gauge-recovery, scale-state, refusal, reflection, disconnected chain).

## 3. SLAM / VIO compatibility (research, NOT benchmarked here)

Only the two COLMAP rows were measured. Everything else is from prior knowledge and **must be re-verified (licence,
version, VRAM) before any adoption**; no claim below is an observed result.

| Candidate | Licence (recall) | Win/Linux | 6 GB fit (recall) | Verdict |
|---|---|---|---|---|
| COLMAP incremental `mapper` (4.2.0 CUDA) | BSD | both, **measured** | 0.84 GB peak delta, measured | baseline, in use |
| COLMAP `global_mapper` (GLOMAP-style) | BSD | both, **measured** | same | measured: see §5, silent failure |
| ORB-SLAM3 (mono / mono-inertial, loop closure, relocalisation) | GPLv3 | Linux-first, no official Windows | CPU | out-of-process research tool only; GPL conflicts with SaaS distribution |
| RTAB-Map (loop closure, graph optimisation) | BSD-3 core | both | CPU/GPU | RGB-D/stereo oriented; weak for mono photo/video |
| DROID-SLAM | BSD-3 | Linux | large (official guidance ≥ ~11 GB) | does not fit 6 GB at default |
| DPVO | MIT-style | Linux | reported much lower than DROID | first learned candidate worth a benchmark |
| MASt3R-SLAM / MASt3R | non-commercial | Linux | n/a | licence conflicts with the commercial arch/interior SaaS wedge |
| hloc + LightGlue (relocalisation) | Apache-2.0 | both | fits | candidate for relocalising unregistered images |
| Basalt / OpenVINS / Kimera-VIO (VIO) | BSD / GPLv3 / BSD | Linux | CPU | only relevant once IMU data exists |
| GTSAM / Ceres pose-graph optimisation | BSD | both | CPU | scipy robust solver already in `camera_pose_graph.py` |

**Decision:** no SLAM adapter now. Reasons: no video+IMU dataset or reference exists locally; the photo/COLMAP path works
and is the customer wedge; and the rule "do not claim SLAM integration until its poses demonstrably change the final
reconstruction" cannot be satisfied without a candidate that has been run end-to-end. Next benchmark candidates, in
order: DPVO (video), `hloc` relocalisation (unregistered images), Basalt (only with IMU).

## 4. Optional IMU interface (contract only, nothing implemented)

`imu` is `None` by default and photo-only reconstruction must never require it. When present:
`{samples: [(t_s, gyro xyz rad/s, accel xyz m/s²)], T_cam_imu (4x4), time_offset_s ± sigma, gyro/accel noise density,
timebase: "camera_clock"|"imu_clock"}`. IMU would supply metric scale and gravity direction (⇒ `scale_state: metric`)
and must be rejected, not guessed, if the time offset is not estimated.

## 5. Measured benchmark: South Building, 32 images (`benchmarks/results/pose_benchmark_south_building.json`)

Commit `156f3e36` (+ working tree), COLMAP 4.2.0 CUDA, RTX 4050 6 GB, SIFT + exhaustive matching, reference = the
COLMAP model shipped with the dataset (**not independent ground truth; relative scale**; units are reference units).

| | incremental `mapper` | `global_mapper` |
|---|---|---|
| registered | **22 / 32** (10 unregistered) | 32 / 32 |
| mapper runtime | 26.3 s | 14.2 s |
| alignment inliers (robust Sim3) | 22 / 22 | **15 / 32** |
| ATE rmse, inliers | 0.0215 | 0.0185 |
| ATE rmse, all registered | 0.0215 | **16.05** |
| rotation err median / max (all) | 0.14° / 0.67° | **26.5° / 177.1°** |
| RPE rot median | 0.19° | 0.26° (max 160°) |
| outlier cameras | 0 | **17** (blocks at ≈177°, ≈90°, plus 2 singles) |

Peak GPU memory 1032 MiB incl. 191 MiB baseline (≈0.84 GB).

Findings (observed):
1. `global_mapper` registers every image but **17/32 cameras are wrong**, in coherent blocks (≈177° and ≈90°): a
   silent failure that the registered-image count hides. Local consistency is fine (RPE median 0.26°); the error is
   in the global gauge of the blocks. It must not be selected on registration count.
2. Incremental registers fewer images but those are accurate (sub-degree). The reference-free `validate_poses` flagged
   2 trajectory discontinuities in each model (photo-set filename order, so indicative only).
3. The 113° outlier mentioned in the brief was **not reproduced** in either run; the failure modes seen here are
   unregistered images (incremental) and block-flipped rotations (global).
4. Robust alignment is what makes (1) legible: with plain least squares the 15 good cameras would be dragged by the 17 bad.

## 6. Not done / limitations

* WS1 learned SLAM, WS2 VIO: research table only; no adapter, no run.
* WS3 video: `reconstruction/video_frames.py` already implements blur/near-duplicate/over-budget keyframing and is
  wired into `apps/api` and the facade; verified on synthetic video only. **No real video exists in the repo**, so no
  video-to-reconstruction benchmark was run. Adjacent-frame dependence is not yet down-weighted in matching.
* No per-pose uncertainty from the backends; scale remains `relative`.
* WS5 native/CI reliability is owned by the CI-recovery session (`claude/laughing-colden-c781cf`); nothing duplicated here.
* Reference is COLMAP itself; agreement ≠ accuracy.
