# Reconstruction audit: what the pipeline genuinely does (2026-10-13)

Method: traced the real execution path in code, then ran it on real photographs on this machine (Windows 11, RTX 4050
6 GB, COLMAP 4.2.0 built with CUDA, CPU-only PyTorch). Every number below was measured in this session unless it cites a
recorded earlier run. Nothing here comes from a README.

## 1. Executable truth

**The path is real end to end.** Photographs go through: input inspection (`evidence/image_check.py`: decode, EXIF,
evidence class) and admission (`reconstruction/robustness.py`) to COLMAP (`reconstruction/backend/colmap_backend.py`: SIFT
with affine shape and domain-size pooling, exhaustive matching, mapper, intrinsics pinning, sub-model merge) to metric
scale anchoring (only from distances the user measured) to MiDaS depth, perception, meshing, WorldIR compile and validation
gate, then a WorldStore commit with an adoption protocol, artifacts (point cloud, camera poses) and six exports.

**No canned scene on the production path.** Searching `apps/api`, `engine/pipeline`, `reconstruction` and `perception`
for `synthetic`, `fabricat`, `placeholder`, `mock`, `fake`, `canned` finds only comments that disclaim fabrication and the
`REALITY_TEST_BACKEND` test seam. The one fallback that does produce geometry without multi-view registration is the
single-image bootstrap (level 0, MiDaS depth): it is labelled a hypothesis, relative scale, `registration_status=partial`,
never `SUCCESS`.

**Test seam caveat.** `REALITY_TEST_BACKEND` disables the learned stages only when the injected backend registers cameras.
A backend that returns nothing falls through to the single-image bootstrap, which loads the real MiDaS model on any
decodable photograph. Tests must use a backend that registers (fixed in `tests/test_job_progress.py`).

## 2. Baseline on real photographs

Dataset: committed South Building subset, 32 real photographs (every 4th of COLMAP's 128-photo example set), with the
reference reconstruction shipped alongside (`datasets/south_building/sparse`).

| Run | Photos | Registered | Points | Mean reprojection error | Wall time |
|---|---|---|---|---|---|
| Product path, this session (CPU contention) | 32 | **22 (69%)** | 5,397 | 0.297 px | 1,001 s |
| Recorded GPU run (2026-09-21) | 32 | 22 | 5,402 | n/a | 620 s |
| Product path, first 10 photographs | 10 | 3 | 114 | 0.225 px | about 75 s |
| Product path, first 8 photographs | 8 | 0 | n/a | n/a | about 80 s |

The 22-of-32 result reproduces exactly across two machines and runs, so it is a property of the data, not noise. Recorded
accuracy of those 22 against the reference (earlier run, through `reconstruction/evaluation.py`): rotation error median
0.18 deg (max 0.38 deg), camera-centre error median 1.1% of scene scale, after Sim(3) alignment.

Why only 22: the reference registers all 32 because it was solved with the full 128 photographs, whose in-between photos
bridge the gaps. A stride-4 subset has weakly linked stretches. Registered photos form three contiguous runs (141 to 153,
185 to 225, 308 to 332).

## 3. Hypotheses tested, with results

| Hypothesis | Test | Result |
|---|---|---|
| A pre-upload overlap check can predict which photos will register | All-pairs verified-match graph (SIFT, ratio test, RANSAC) on the 32 photos, components at thresholds 15 to 150, compared with COLMAP's registered set | **Refuted.** At 15 to 30 matches the graph calls all 32 one group; at 50 the largest group is 28 (18 of the 22 registered). No single match-count threshold predicts registration, so no "predicted registrable" claim is made. |
| Relaxing COLMAP's registration thresholds registers more photos safely | `image_registrator` with min inliers 12, inlier ratio 0.10, max error 16 on a copy of the workspace, then triangulate and bundle adjust | **Refuted.** 25 registered instead of 22, but the three new cameras have 10, 12 and 15 triangulated observations against a median of 663 for the rest: unsupported poses. Defaults kept. Agrees with the repo's own earlier finding that guided matching reached 31 of 32 cameras but warped the model (rotation error 24.6 deg vs 0.15 deg). |
| Dense geometry is available on this machine | Check GPU, COLMAP build, recorded runs | **Available.** RTX 4050, COLMAP with CUDA; a recorded run fused 246,816 dense points (patch-match about 344 s). |
| The product path produces dense geometry for the real dataset | Read the 32-photo report | **No.** `engine/pipeline/dense_gate.py` skipped dense: "22 of 32 photographs registered (69%; needs >= 80%)". The threshold is documented as EXPERIMENTAL. |

## 4. Implemented in this session

| File | Change | Evidence |
|---|---|---|
| `reconstruction/diagnostics.py` (new) | Per-camera support (triangulated observations, mean point confidence) and structured findings: `UNREGISTERED_IMAGE`, `WEAKLY_SUPPORTED_CAMERA`, `LOW_REGISTRATION`, `NON_FINITE_GEOMETRY`, `DANGLING_OBSERVATION`, `UNKNOWN_CAMERA`; each with measurement, threshold, severity, `safe_to_auto_correct`, recommended operation | `tests/test_reconstruction_diagnostics.py` (7 tests, including the real reference model: median 2,634 observations per camera, minimum 396) |
| `engine/pipeline/vertical_slice.py`, `apps/api/jobs.py` | Diagnostics recorded in the stage facts and surfaced in every version report as `stages.reconstruction.diagnostics`; analysis can never fail or change a build | `tests/test_place_api.py::test_job_report_carries_reconstruction_diagnostics` |
| `engine/pipeline/dense_gate.py` | Registration threshold configurable with `REALITY_DENSE_MIN_REGISTERED_FRACTION` (default unchanged at 0.8; invalid values fall back) | `tests/test_dense_gate.py` (10 pass) |
| `tests/test_job_progress.py` | Slow test backend now registers cameras, so it stays offline and fast | test goes from a 237 s failure to 23 s |

## 5. Experiments on the 32-photo subset (product path, one variable changed at a time)

Machine: Windows 11, RTX 4050 6 GB, COLMAP 4.2.0 (CUDA), Python 3.14. Input: the 32 committed photographs (per-image
sha256 in `datasets/south_building/MANIFEST.json`). The only variable between baseline and candidates is
`REALITY_DENSE_MIN_REGISTERED_FRACTION` (default 0.8, candidates 0.6) plus `JOB_TIMEOUT_SECONDS=3600`. The reference
model is read only by `scripts/evaluate_session_model.py`, after a run; the product path never sees it.

### 5.1 What happened to each run (VERIFIED from job records and workspaces)

| Run | Config | Terminal state | Notes |
|---|---|---|---|
| baseline (earlier this session) | gate 0.8 | completed, **dense skipped** ("22 of 32 registered (69%; needs >= 80%)") | 22/32, 5,397 points, 0.297 px, 1,001 s under CPU contention |
| `expdense` | gate 0.6, 1800 s limit | **FAILED**: `job handler timed out after 1800s` | CPU contention; explicit failure, no world adopted |
| `expdense2` | gate 0.6 | left `running`, then interrupted | machine sleep/stop; no result, not counted |
| `expdense3` job_ae5d966b0ae4 | gate 0.6, 3600 s | **partial**, 12 m 57 s | candidate A below |
| `expdense4` job_fbf216bd0ee0 | gate 0.6, 3600 s | **partial**, 11 m 44 s | candidate B below (identical config, repeat) |

Note: a few seconds of unit tests ran while A and B were in progress; neither run was disturbed by sleep or contention.

### 5.2 Baseline vs candidates (VERIFIED, `scripts/evaluate_session_model.py`; reference scene diagonal 5.81)

| | baseline (gate 0.8) | candidate A: expdense3 | candidate B: expdense4 |
|---|---|---|---|
| registered / unregistered | 22 / 10 | **25 / 7 (3 of the 25 are wrong)** | 22 / 10 |
| sparse points | 5,397 | 5,466 | 5,395 |
| pose error, median rot / max rot (deg) | not rerun | 0.20 / **128.7** | 0.185 / 0.51 |
| camera-centre error, median / max | not rerun | 0.0138 / **5.32** | 0.0129 / 0.049 |
| outlier cameras | none recorded | P1180173, 177, 181 (104 to 129 deg off) | none |
| dense executed | no (gate) | yes | yes |
| patch-match / fusion | n/a | 393.6 s / 5.1 s | 324.9 s / 3.9 s |
| fused dense points | 0 | 257,823 (25 source images) | 254,560 |
| dense judge verdict | n/a | **not recorded** (report gap, fixed; see 8) | accepted: 98% of sparse structure reproduced, density x47.2, 16% new surface |
| dense distance to reference points: median / p90 / p99 | n/a | 0.0134 / 0.031 / 0.16 | 0.0126 / 0.029 / 0.063 |
| dense share within 1% / 5% of scene diagonal | n/a | 97.1% / 100% | 98.7% / 100.0% |
| reference points with a dense point within 2% of diagonal | n/a | 66% | 67% |
| wall time | 1,001 s (contended) | 777 s | 704 s |

Reading the table honestly:

1. **Registration is not deterministic.** The same photographs, code and configuration gave 22 correct cameras in
   three runs and 25 cameras (3 misplaced) in one. "25 registered" looked like an improvement over 22 and was not: the
   extra three were 104 to 129 degrees wrong. Registered count alone is not a quality measure.
2. **The three wrong cameras were invisible to every per-camera measure**: 76 to 78 triangulated observations (median
   664 for the rest) and 0.25 to 0.27 px reprojection error, the same as good cameras. The only reference-free signal
   found is structural: they share 55 to 72 points with each other but at most 11 with any other camera.
3. **Dense reconstruction works on this machine and its output is geometrically consistent with the reference.**
   Distances are to the reference's sparse points, a proxy and not a surface error; coverage is about two thirds of the
   reference structure, not complete.
4. A higher dense point count is not the evidence. The evidence is the judge verdict (B), the reference distances, and
   that A's dense cloud did not visibly degrade despite three misplaced cameras (why is UNVERIFIED; patch-match uses
   neighbouring views, so a mislocated camera plausibly fails consistency and is filtered, but that was not proven).
5. **The default stays 0.8.** Two dense runs on one dataset are not enough to move a production threshold, and the
   gate's registration fraction is itself unreliable (point 1). A better gate would count only cameras that survive the
   tie-group check (22 of 32 = 69% in both A and B, versus a raw 78% in A): recorded as the recommended review item.

## 6. Root cause of the unregistered cameras (VERIFIED from COLMAP's own match database)

The unregistered photographs are not scattered; they are one **island** of consecutive frames
(P1180157 to P1180181, and P1180336 to P1180344) that is strongly connected internally and only thinly connected to
the registered main body.

| Evidence (run B) | Measured |
|---|---|
| strongest pairs inside the island | 165-344: 2,324 inliers; 161-340: 727; 169-165: 506; 177-181: 1,180 |
| links from the island to registered cameras at the bridges | 157-153: 33; 336-153: 37; 336-149: 17; 181-185: 47; 177-316: 23 |
| keypoints per island photograph | 7.4k to 12.3k (same range as registered photographs) |
| registered partners with >= 15 inliers | 0 for 161, 165, 169, 340, 344 (island core); 5 to 6 only for 173, 177, 181 |

So the failure is **graph connectivity**, not image quality: stride-4 sampling leaves the 153 to 157 and 181 to 185
gaps with only 17 to 47 verified inliers, below what COLMAP needs to place a camera absolutely. The reference registers
all 32 because it was solved with the 96 photographs this subset drops. In run A the mapper placed part of the island
(173, 177, 181) through those thin bridges and placed it wrongly; in run B it correctly placed none of it.

Not examined (UNVERIFIED): repetitive-texture ambiguity as the cause of the wrong placement in run A, camera calibration
residuals, per-image blur. No single match-count threshold predicts registration (section 3); production thresholds
were not lowered.

## 7. Correction loop (IMPLEMENTED and tested; recommendation-only in the product)

`reconstruction/correction.py` builds on `reconstruction/diagnostics.py`:

1. Reads structured findings; maps only `safe_to_auto_correct` findings that have a registered executable action.
2. Records `rationale` and `expected_effect` per attempt.
3. Bounded: `Budget(max_attempts=2, max_seconds=60)`, checked before each action.
4. The baseline object is never modified; candidates are new objects.
5. Candidates are scored with the same `measure()` as the baseline (diagnostic counts, registered cameras, points,
   finite check, and the reference pose evaluation when a reference is supplied).
6. Rejected when: non-finite geometry, more dangling/unknown references, more cameras lost than declared, fewer than
   90% of points kept, any increase in weak or bridged findings, reference rotation/centre error regressing, or no
   measured improvement.
7. A candidate that raises is FAILED and recorded, never propagated.
8. Unregistered photographs become an **evidence request** (which images, what to photograph), never a fix.

Executable actions: `EXCLUDE_WEAK_CAMERAS`, `EXCLUDE_BRIDGED_GROUP`. Nothing else is attempted.

New diagnostic `WEAKLY_BRIDGED_GROUP`: cameras tied to the rest by fewer than `BRIDGE_MIN_SHARED_POINTS` = 30 shared
points per pair (EXPERIMENTAL). Derived from one observed failure and checked against the committed 32-camera reference
(one group up to a threshold of 50). It also fixed a defect: dense points, which cite every camera, had been counted as
camera support, making each camera look supported by about 258,000 points.

**Real-data result (VERIFIED; `tests/test_reconstruction_correction.py`, fixture
`tests/fixtures/south_building_stride4_misregistered.json.gz`, the actual model from run A):** the loop detects the
group, accepts `EXCLUDE_BRIDGED_GROUP`, removes exactly P1180173/177/181, and against the committed reference changes:

| | before | after |
|---|---|---|
| max rotation error | 128.66 deg | 1.06 deg |
| max camera-centre error | 5.32 | 0.048 |
| median rotation / centre error | 0.202 / 0.01375 | 0.201 / 0.01367 |
| common cameras | 25 | 22 |

This corrects the registered set by removing wrong cameras; it adds no geometry. It is **not applied by the product**:
the version report carries `stages.reconstruction.correction_plan` with `applied: false`. Dense output of a corrected
model must be recomputed; the action discards stale dense points rather than keeping them.

## 8. Lifecycle and persistence

* A timeout fails the job explicitly (run `expdense`: `job handler timed out after 1800s`); nothing is adopted.
* Diagnostics failure still cannot fail a build, but is now graded: a diagnostics error or any error-severity finding
  (for example `WEAKLY_BRIDGED_GROUP`, non-finite geometry) makes the outcome **partial**, never `succeeded`.
* `dense_gate` and `dense_mvs` (criteria, stage timings, judge verdict) are now in the version report
  (`stages.dense_gate`, `stages.dense_mvs`); before this change run A's judge verdict was unrecoverable from the API.
* Sparse and dense stay separate: diagnostics count sparse points only (`dense_points` reported separately).

## 9. Status, blockers and next work

| Item | Status | Evidence |
|---|---|---|
| Baseline preserved and reproducible | VERIFIED | section 5; `datasets/south_building/runs`, `scripts/evaluate_session_model.py` |
| Experiment A (32-photo) | VERIFIED | baseline plus two uncontended candidate runs |
| Experiment B (full 128 photographs) | **BLOCKED** | the other 96 photographs and their reference are not in the repository; `scripts/fetch_south_building.py` downloads the pinned zip (sha256 in the manifest). Needs the user's approval to download |
| Experiment C (32 + intermediates near failures) | **BLOCKED** (same) | intermediates needed: P11801{58-60,62-64,66-68,70-72} and P11803{37-39,41-43,45-47} |
| Root cause of unregistered cameras | VERIFIED (connectivity); repetitive texture UNVERIFIED | section 6 |
| Dense on this dataset | VERIFIED (two runs) | section 5 |
| Dense default gate 0.8 | unchanged | needs more data; recommend counting tie-group-verified cameras |
| Correction loop | IMPLEMENTED; verified offline on one real model | section 7 |
| Correction loop changing the build | NOT DONE (deliberately) | needs a dense re-run path and more than one real example |
| Tie threshold (30) generality | UNVERIFIED beyond one dataset | needs experiments B/C and other captures |
| Run-to-run registration variance | MEASURED (22 x3, 25 x1); cause UNVERIFIED | needs repeated runs with thread control |

### Agent instructions (next session)

1. With the user's approval, `python scripts/fetch_south_building.py download`; run experiment B (all 128) and C (add the
   listed intermediates), one at a time, evaluating each with `scripts/evaluate_session_model.py`.
2. Repeat the 32-photo candidate 3 more times to estimate the registration variance; if the wrong-group failure recurs,
   test `colmap mapper --Mapper.num_threads 1` as a single-variable change.
3. Only after B/C: decide whether the dense gate should count tie-group-verified cameras, and whether to apply
   `correction_plan` automatically (requires a dense re-run on the corrected model, judged by `dense_judge`).
4. Do not lower registration thresholds to raise camera counts (section 3 refuted it).

Acceptance criteria: B/C records exist with image hashes, config, versions, runtime and evaluation; the bridge threshold
is confirmed or revised with evidence; `pytest tests/test_reconstruction_correction.py
tests/test_reconstruction_diagnostics.py tests/test_dense_gate.py` stays green.
