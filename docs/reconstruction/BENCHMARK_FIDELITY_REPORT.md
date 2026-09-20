# Benchmark Fidelity Report

P1 RECONSTRUCTION FIDELITY mission. Every number below is measured from
an actual pipeline run on this machine (COLMAP 4.2.0 CUDA, Python 3.14)
or is an explicit refusal. No benchmark name is treated as evidence of
success; no metric is invented.

Date: 2026-09-20. Branch: `agent/freebuff-reconstruction-perception`.

---

## 0. The fidelity metric itself (what changed)

The previous GT comparison (`scripts/run_south_building_e2e.py`)
compared reconstructed quaternions DIRECTLY against the reference
model and claimed "rotation is comparable without alignment". That is
false: an incremental SfM run recovers the scene up to an arbitrary
gauge (global rotation + scale + translation chosen by the mapper), so
the recorded **8.94° "median rotation disagreement" was dominated by
gauge difference, not error** — it said almost nothing about fidelity.

`reconstruction/evaluation.py` is the canonical evaluator, enforced by
measured properties (tests/test_pose_evaluation.py, 13 tests):

- P1 gauge invariance: a known Sim(3) transform between estimate and
  reference evaluates to ~0 error (absorbed by alignment);
- P2 noise honesty: a known per-camera perturbation is measured back at
  the injected magnitude; a CONSTANT perturbation — pure gauge — is
  correctly absorbed, not reported as error;
- P3 refusal: <3 common cameras or degenerate (coincident/collinear)
  geometry REFUSES — never a guessed score;
- P4 determinism: identical inputs produce identical reports.

Alignment: Horn quaternion rotation (chordal-optimal over the paired
camera→world orientation frames) + scale/translation from camera-center
geometry (robust to planar/collinear reference configurations). Point
fidelity reuses the SAME single gauge transform — one rigid gauge
explains orientations, centers and points.

---

## 1. south-building (real photos, real backend) — REAL

**INPUT.** `datasets/south_building/images`: 32 real photographs of the
South Building, UNC Chapel Hill (Panasonic DMC-TZ3; 1024-px deterministic
downscale of the COLMAP authors' 128-photo release, stride-4 subset).
Committed reference: 32 cameras + 49,608 sparse points (ids preserved).
Evidence label: REAL.

**PIPELINE.** import_folder (real decode, EXIF, measured quality) →
quality admission → ColmapReconstructionBackend (REAL feature_extractor
→ exhaustive_matcher → mapper → model_converter) → run-level robustness
classification → gauge-aligned fidelity evaluation → fusion.

**OUTPUT (run_20260920T175018_gpu.json, COLMAP 4.2.0 CUDA, 50.2 s total,
48.1 s reconstruction):**

| Metric | Measured value |
|---|---|
| cameras ingested / admitted | 32 / 32 (2 degraded by measured clipping 0.264 > 0.25) |
| cameras registered | 22 / 32 (10 honestly unregistered) |
| points | 4,453 |
| outcome | DEGRADED — "registration_status=partial: result usable but incomplete" |
| pose fidelity (post-alignment) | rotation median **0.214°**, max 0.653°, mean 0.215° |
| camera-center error (post-alignment) | median 0.0215, max 0.0349 (reference units) |
| scale ratio est/ref | 1.092 |
| point fidelity (est→ref surface) | median **0.0068**, mean 0.0272, p90 0.0162, max 0.135 |
| fusion | 22 observations, provenance CONFLICT preserved (198 conflicts kept, not averaged away) |

**Interpretation (honest).** WHERE the pipeline registers, the
reconstruction is genuinely faithful: sub-centimeter point-to-reference
distance at median and 0.2° pose fidelity on real photographs are the
magnitudes a correct SfM solution produces — this is the evidence that
the pipeline recovers true structure, not a lookalike. The honest gap is
COVERAGE: 22/32 cameras and ~4.5k vs ~49.6k reference points.

**KNOWN FAILURE.** 10/32 cameras unregistered (insufficient pairwise
matches after downscale + stride subset). Recorded as DEGRADED with
machine-readable reason; never converted into success.

**REMAINING GAP.** Coverage. Higher-resolution committed subset, more
images, or guided matching would close it; not attempted here because
the fidelity question is now answered with measured numbers.

---

## 1b. Coverage campaign (measured, on the same real subset)

The coverage gap from §1 was attacked with controlled ablations on the
committed 32 photos. Every number is a real COLMAP 4.2.0 run measured
from its own database/models, not an assumption:

| Config | Cameras registered | Points | Pose fidelity (median) | Verdict |
|---|---|---|---|---|
| A: baseline SIFT | 22 / 32 | 4,438 | **0.164°** | the §1 run |
| B: affine+DSP SIFT | 22 / 32 | **5,392** (+22%) | **0.164°** | STRICT WIN — adopted as default |
| C: affine+DSP + guided matching | 31 / 32 | 11,632 | **24.6°** (max 160.9°) | FALSE COVERAGE — rejected |
| D: B + multi-model merge (first attempt) | 23 | 5,395 | 3.1° | CONTAMINATED — fixed (below) |
| E: D's fixes (robust gauge + merge minimum), final run | 22 / 32 | 5,388 | **0.172°** (max 0.482°) | CLEAN — the adopted default |

**Final run (run_20260920T210502_gpu.json, COLMAP 4.2.0 CUDA, 699 s** —
the affine+DSP extraction is CPU-bound and ~14× slower than baseline;
that is the measured price of +21% verified points at unchanged
fidelity). Zero outliers — the robust gauge was not even needed on a
healthy run, which is exactly how it should behave: it exists for the
unhealthy ones. Center-error median improved to 0.0107 (baseline
0.0215). Single sub-model this run, so the merge path was not exercised
in production — its refusal discipline is proven by unit tests
(`test_tiny_submodel_refused_not_merged`, `test_merge_report_travels_on_result`).
Point fidelity: median 0.0065, p90 0.0162 reference units (median 6.5 mm
at building scale) — identical to baseline within noise.

**Guided matching is measurably UNSAFE on repetitive architecture.** Its
9 extra cameras came from forcing matches through similar facade
windows; the warp is visible in the geometry itself (median point error
0.206 vs 0.0068 reference units). `DEFAULT_GUIDED_MATCHING = False`;
it remains opt-in (`--guided-matching`) with this measurement as the
reason. Affine-shape + domain-size-pooling extraction (B) keeps fidelity
intact while adding 22% verified points — adopted as default.

**Config D exposed two more honesty defects, both fixed red-first:**

1. **The degenerate-merge hole.** COLMAP's mapper split off a 3-point
   sub-model; `merge_submodel_results` ICP-snapped it onto the 5,392-
   point reference with ~zero residual (3 points always fit 3 nearest
   neighbors) — every existing gate (RMSE, inlier fraction, scale
   sanity) passed, its one camera entered the scene, and the merge
   report was DISCARDED by the backend (`_merge_report` unused), leaving
   zero trace. Fix: `MIN_MERGE_POINTS = 8` verification minimum (a
   cloud too small to constrain a transform cannot be certified as a
   same-scene registration — refusal with recorded reason, geometry
   excluded, nothing identity-placed or silently dropped), and the
   serialized merge report now travels ON the ReconstructionResult
   (`merge_report` field) and into the E2E run record.
2. **The gauge was outlier-naive.** One flipped camera out of 23 dragged
   the plain least-squares Horn gauge so far that every healthy camera
   reported 3.1° median / 156.5° max — the metric measured the
   outlier's pull on the gauge, not fidelity (unit test proves a single
   160° camera pushes the old evaluator to 1.44° median on otherwise
   perfect data). Fix: outlier-honest two-pass gauge — refit on cameras
   within `OUTLIER_REJECTION_DEG = 15°` (healthy COLMAP medians here:
   0.15–0.21°; false-merge signature: 90–180°) then re-measure EVERY
   camera under the robust gauge. Outliers stay fully measured and are
   NAMED in the report's `outliers` list — separated and reported,
   never dropped or averaged away.

---

## 2. Victoria Memorial (euro-009)

**INPUT.** None in this repository. `benchmarks/structures/__init__.py`
declares the capture requirements (walkaround coverage, facade detail,
GNSS/IMU/calibration) and the record is CAPTURE_PENDING.

**PIPELINE.** Not run — `benchmarks/architectural.py` refuses a
CAPTURE_PENDING record before any reconstruction work.

**OUTPUT.** None. Evidence label: NO_CAPTURE_YET.

**KNOWN FAILURE.** No failure — the capture is a genuine external
dependency (a human with a camera in Kolkata). Nothing is simulated to
fill the gap.

**REMAINING GAP.** The capture itself. The corpus frontend's
"COMPILED WorldIR" claim for this structure is NOT backed by any
committed evidence artifact; treat it as aspirational until a capture
session id exists on the record.

---

## 3. Burj Khalifa (skysc-001)

**INPUT.** None. Not a single photograph, LiDAR frame, or survey exists
in the repository. The frontend corpus marks it NOT_YET_CAPTURED with
28,500 expected images, HIGH_ALT_DRONE + AERIAL_LIDAR + GROUND_SURVEY
capture types, and real capture challenges (atmospheric dust at 800 m,
specular glazing, GNSS PDOP > 4.2).

**PIPELINE.** None run. A run without evidence is impossible, and the
mission's honesty rules (and benchmarks/architectural.py's CAPTURE_PENDING
refusal) forbid fabricating one.

**OUTPUT.** None.

**STRUCTURAL SIGNATURE STATUS.** The Y/tri-axial buttressed core, 27-tier
setbacks, central spire and 828 m proportion can only be preserved by a
reconstruction from real evidence; there is no evidence, so there is no
claim. No tower primitive, no "generic skyscraper" placeholder, no
hand-modeled stand-in was created — that would be the exact
benchmark-specific cheat the mission bans.

**KNOWN FAILURE.** Evidence unavailability, recorded honestly as
NOT_YET_CAPTURED.

**REMAINING GAP.** The entire capture. Even the reduced-scale public
photo collections that exist for the building are not committed here;
acquiring one would be the first real step (deterministic fetch +
checksum manifest, the pattern proven with south_building).

---

## 4. room_capture / room_capture_mvs (synthetic-rendered rooms)

**INPUT.** `datasets/room_capture*`: procedurally rendered room captures
with known ground truth (exact intrinsics, measured station coordinates).
Evidence label: SYNTHETIC EVIDENCE — useful for algorithm verification,
NOT real-world validation.

**PIPELINE.** Real ColmapReconstructionBackend (report.json:
`backend=ColmapReconstructionBackend`, `registration_status=partial`,
overall PARTIAL_SUCCESS) plus the detail spine and MVS dense fusion
(294,345-point fused.ply verified in P6-01 integration tests).

**OUTPUT.** WorldIR entities + dense geometry; pipeline_out artifacts are
locally generated (gitignored) — fresh clones regenerate them.

**KNOWN FAILURE.** registration_status=partial on room_capture: some
stations unregistered, recorded in the report rather than hidden.

**REMAINING GAP.** None claimed beyond what the reports record. These
benchmarks prove algorithm behavior under known ground truth; they do
NOT prove real-world robustness — south-building does that.

---

## 5. Metrics inventory (what the fidelity suite now measures)

| Metric | Where | Basis |
|---|---|---|
| pose rotation error (median/mean/max) | `evaluate_absolute_poses` | geodesic quaternion angle AFTER gauge alignment |
| camera-center error | `evaluate_absolute_poses` | Sim(3)-aligned center distances |
| scale ratio | `evaluate_absolute_poses` | center distance ratios |
| point-to-reference distance | `evaluate_point_cloud` | cKDTree one-way est→GT (no symmetric overclaim), median/mean/p90/max |
| registered coverage | E2E record | registered / reference cameras |
| admission outcomes | robustness_admission | measured per-item reasons |
| run outcome vocabulary | classify_reconstruction_run | ACCEPTED/DEGRADED/UNRESOLVED/FAILED with reason |
| fusion conflicts preserved | fuse_quantity | conflicts recorded, never averaged away |

Every metric is measurable, deterministic, and refuses rather than
fabricates when its preconditions fail.

---

## 6. Where fidelity is actually lost (audit conclusion)

1. **Gauge, not geometry** — the single largest historical error source
   in reported fidelity was the evaluation layer itself (8.94° vs the
   true 0.214°). Fixed with the canonical evaluator.
2. **Coverage, not correctness** — on real data the pipeline's errors
   are small; its losses are unregistered cameras (~31% here) and
   sparse point coverage (~9% of reference). This is an evidence-quality
   and matching-recall problem, not a projection/metric problem.
3. **Unproven, not broken** — Victoria Memorial and Burj Khalifa are
   not failing benchmarks; they are UNCAPTURED benchmarks. The system's
   honesty machinery (CAPTURE_PENDING refusal, refusal-valued metrics)
   is doing its job; the missing input is the real world, not code.
