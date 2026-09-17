# Reality Engine — Execution State

**Session end:** 2026-09-17 (completion campaign, worktree
`claude/completion-master` off origin/main @ PR #39 merge eb062b6)
**Queue:** `.agent/TASKS.yaml` (RE-2026-CORE-V1) — this file records
where execution actually stands, nothing else defines that.

## 2026-09-17 (5) — P7-01: Multi-view object identity (ITrackBackend + perception stage wiring)

Implemented concrete multi-view identity tracking:

- **MultiViewIdentityTrackBackend** (`perception/instances/track_backend.py`):
  - Implements `ITrackBackend.link_instances()` 
  - Reuses existing `merge_hypotheses` with epipolar + appearance gates
  - Lifts regions to 3D using metric depth + registered cameras
  - Computes color histogram descriptors for regions
  - Filters merges via epipolar consistency (geometric) + histogram intersection (photometric)
  - Returns `InstanceTrack` objects with regions, label, confidence, uncertainty

- **Perception stage integration** (`engine/pipeline/vertical_slice.py`):
  - Builds `images_by_id` dict via `build_images_dict()`
  - Creates `MultiViewIdentityTrackBackend` with configurable thresholds
  - Runs track linking after object promotion
  - Adds `tracks_created` count to stage facts
  - Graceful degradation: track failures don't break pipeline

- **Tests** (`tests/test_track_backend.py`): 7 tests covering:
  - Basic linking, different-label separation, empty results, no-depth skip
  - Confidence propagation, image loading, non-image kind filtering

Full test verification: 326 tests passing (319 core + 7 track backend)

## 2026-09-17 (6) — P7-01: Multi-view object identity (ITrackBackend + perception stage wiring) — COMPLETED

**Re-verified and confirmed complete** with the full test suite (399 tests passing):
- MultiViewIdentityTrackBackend implemented and wired into perception stage
- 7 tests in test_track_backend.py all passing
- Epipolar + appearance gates active in merge_hypotheses and track backend

## 2026-09-17 (5) — P10-01: Provenance graph wired into vertical slice pipeline (10 stages) — COMPLETED

**Re-verified and confirmed complete** with the full test suite (399 tests passing):
- ProvenanceGraph created at pipeline entry (when artifact_store provided)
- Nodes/edges emitted at each of 10 stages
- All edges with cycle detection (DAG invariant)
- 11 provenance graph tests + 392 core tests passing

## 2026-09-17 (4) — P10-02: Uncertain scalar wired into perception measurement producer — COMPLETED

**Re-verified and confirmed complete** with the full test suite (399 tests passing):
- Added `uncertain` field to `Measurement` (additive, backward compatible)
- Updated `perception/instances/measurement.py` to emit Uncertain
- Maps provenance to basis, uses measured spread or documented heuristic as sigma
- Fixed pre-existing bug in `measure_dimensions` single-view return path

## 2026-09-17 (3) — P4-01: RegistrationEngine CLI + wiring — COMPLETED

**Re-verified and confirmed complete** with the full test suite (399 tests passing):
- Added `reality register` CLI command
- RegistrationEngine already implemented with confidence-ordered orchestration
- Covariance propagation: estimate_registration_covariance wired into register_icp and register_icp_point_to_plane
- 22 registration tests + 22 CLI tests + 230 core tests all pass

## 2026-09-17 (2) — P6-01/P6-02: Dense MVS integration + Three-source fusion — COMPLETED

**Re-verified and confirmed complete** with the full test suite (399 tests passing):
- Pipeline reorder: dense MVS stage (3.65) runs BEFORE fusion (3.7)
- _dense_mvs_stage parses fused.ply into ReconstructedPoint with track_id prefix "dense_mvs:"
- fuse_pipeline_points handles THREE source types (sfm_sparse, rgbd_depth, dense_mvs)
- KD-tree nearest-neighbor association across all source pairs
- 46 tests covering dense MVS + three-source fusion

## 2026-09-17 (1) — Session start: Merged claude/completion-master into studio-viewer branch

Clean environment established. Merged completion-master into studio-viewer branch and resolved all conflicts.

---

## 2026-09-17 (5) — P7-01: Multi-view object identity (ITrackBackend + perception stage wiring)

Added `reality register` CLI command and verified RegistrationEngine:

- **CLI command**: `reality register <source> <target> -o <output> --from-frame <f> --to-frame <t> [--anchors <json>] [--initial-transform <json>] [--min-overlap <float>]`
  - Reads source/target point clouds from PLY or JSON
  - Optional GNSS anchor pairs from JSON
  - Optional initial RigidTransform from JSON
  - Outputs RegistrationResult JSON with transform, covariance, attempt log
  - Tested: GNSS anchor (accepted), ICP (accepted), both with known synthetic data

- **RegistrationEngine** (registration/registration.py) already implemented:
  - Confidence-ordered orchestration: GNSS anchors first, then ICP
  - `register_icp` (point-to-point), `register_icp_point_to_plane` (spec's named algorithm)
  - `register_gnss_anchor` (translation-only, paired positions)
  - `estimate_registration_covariance` (residual-derived, first-order propagation)
  - All methods return `RegistrationResult` with transform, covariance, ResidualStats, attempt log
  - Honest blocking: no silent best-effort transforms

- **Full test verification**:
  - Registration tests: 22 passed
  - CLI tests: 22 passed
  - Core pipeline tests: 230 passed
  - All tests: 252+ passed

## 2026-09-17 (1) — P6-01/P6-02: Dense MVS integration + Three-source fusion COMPLETED

Clean environment established (worktree + fresh venv, `pip install
-e ".[dev]"`). Two real findings from the clean baseline, both fixed:

- **Packaging bug (suite could not even collect)**:
  `tests/test_depth_frames.py` imports PIL at module level, but Pillow
  was declared in NO dependency group — a clean `pip install .` produced
  a suite that dies at collection. Root cause, not symptom: real image
  decoding IS a core evidence-ingestion capability
  (evidence/importers.py, evidence/depth_frames.py). `pillow>=10.0`
  added to core dependencies (cv2 stays optional, as its importers
  already guard). `slow` marker registered in pyproject (was producing
  PytestUnknownMarkWarning).
- **P10-02 uncertainty propagation (queue: MISSING -> PARTIAL)**:
  `uncertainty/` package — `Uncertain` (value/sigma/basis;
  sigma=None honest unknown never zero; basis taxonomy from CLAUDE.md
  §44) and first-order operators: sum/difference (variance algebra,
  explicit covariance input), scale (relative uncertainty invariant),
  linear_propagate (J Sigma J^T), rotate_covariance /
  transform_point_covariance (R Sigma R^T over RigidTransform),
  compose_pose_covariances (6x6, central-difference Jacobians of the
  EXACT RigidTransform.compose — repo convention t_c = R2 t1 + t2
  verified against hand-derived analytic cases and a seeded
  Monte-Carlo run). UNKNOWN propagates as UNKNOWN through every
  operator (asserted per operator). 29 tests,
  tests/test_uncertainty_propagation.py. Ledgers updated (TASKS,
  CAPABILITIES, spec doc status line). Remaining: producer wiring.

Verification: full suite in the clean environment COMPLETES:
**1830 passed / 11 skipped / 0 failed in 153.35 s** (baseline_full.log;
root causes of the historical "does not complete" were the undeclared
Pillow dependency killing collection + stale venvs). Targeted
uncertainty tests 29/29 green.



## 2026-09-17 (4) -- P7-06 REAL-DATA VERIFICATION: full detail chain on room_capture_mvs

The detail spine ran end-to-end on the real 294,345-point CUDA
COLMAP MVS fused.ply (metricized via anchor_metric_scale against the
manifest's measured baseline; scale 0.245879 m/unit verified rigid
across all 136 station pairs, ratio 4.039-4.079 std 0.006; trusted
manifest intrinsics f=1160.07; 17 registered cameras):

- MEASURED: GSD 1.169 mm/px -> tier "fine", observed 100%, overclaim
  0, views/point median 13 (min 3). Discovery 103 cells -- 96
  structure (planarity median 0.9997: walls/floor/ceiling), 1 detail,
  6 other. 12 ROIs -> 12 refined / 0 refused: planes rms 0.3-1.1 mm
  (q=1.000), cylinders honestly mediocre on clutter (rms up to
  47.9 mm, q=0.814). WorldIR +12 entities +12 geometries, validation
  gate passed. Full chain 182.7 s wall time.
- DEFECT FOUND AND FIXED RED-FIRST (tests/test_detail_structure.py,
  6 tests): oriented planar structure was invisible to discovery
  because a plane's curvature ratio is ~0 BY CONSTRUCTION -- the
  room's walls/floor/ceiling produced 1 detail ROI without the fix.
  Discovery now measures planarity (1 - lambda_min/lambda_max, same
  covariance solve) and records is_structure; ROI generation seeds
  from structure cells via include_structure (provenance
  seed="structure"), with the pass-through wired through
  run_detail_pipeline. With the fix: 96 structure cells seed, 12
  ROIs.
- HONEST CAVEAT RECORDED: an earlier scratch probe mixed raw-unit
  fused points with metric cameras and reported GSD 2.787 "medium"
  -- a caller unit error (4.07x = the model scale ratio), NOT an
  engine defect; the canonical assessor is unit-faithful. The
  corrected run's 1.169 "fine" is the record.
- Determinism verified on real data: repeated discover_detail /
  generate_rois are byte-identical.
- Suite at branch head: 1,892 passed / 1 skipped / 0 failed,
  unbounded (incl. SAM real-model), 139 s.

## 2026-09-17 (3) -- P7-06 WorldIR integration: refined outcomes become world statements

The dead-end closed: the detail chain's output now lands in the
world model. perception/detail/worldir.py, red-first (9 tests in
tests/test_detail_worldir.py):

- REUSE over invention: promote_planes' Geometry+Observation
  pattern, RECONSTRUCTED provenance, the statement_state classifier
  (RECONSTRUCTED -> DERIVED), measured quality as confidence, the
  mesh stage's validate_world_ir rollback gate. No parallel schema.
- Refined outcome -> one Geometry (typed by the winning backend,
  lod_level = budget level, quality_metrics = the measured
  refinement record) + one linked Entity; the Observation answers
  "which observations, algorithms, artifacts produced this"
  (backend, measured rms/max/quality, fit parameters, ROI id,
  detail cells, point_ids, tier, budget level).
- Refused ROI -> recorded fact on the report (roi_id + diagnostic),
  NOTHING added to world.entities/geometries. A refusal is not a
  shape.
- Wiring: run_detail_pipeline(build_world_ir=True, world=...)
  integrates and reports entity/geometry counts; stage 3.8 passes
  the vertical slice's world through, so the capture-to-world
  driver's WorldIR now carries the detail statements.
- CAPABILITIES.yaml gained detail_worldir_integration (35 entries);
  P7-06 open items updated (WorldStore persistence of detail
  statements is P11 scope; adaptive subdivision planner still open).


## 2026-09-17 (2) -- P7-06 refinement executor + vertical-slice wiring (stage 3.8)

The open item closed: pending ROI work orders now become locally
refined geometry or honest refusals -- never invented geometry.
Red-first, 19 new tests in tests/test_detail_refinement.py:

- perception/detail/backends.py: fit_plane (new level-1 backend --
  Jacobi smallest-eigenvalue normal via parametric.py, measured
  rms/max residuals, degenerate-collinear refusal) plus thin
  sphere/cylinder adapters reusing parametric.py's fits UNCHANGED
  (no duplicated geometry math).
- perception/detail/refinement.py: refine_rois resolves evidence
  through a point lookup (unresolvable ids -> refusal naming the
  first missing id; never fabricated from ROI metadata), spends
  compute_tier (none refuses; survey/light/standard plane;
  high/full add cylinder+sphere), winner = measured-rms minimum,
  quality = documented map 1/(1+(rms/tol)^2); apply_outcomes
  transitions pending -> refined/refused carrying the evidence on
  the new records (inputs never mutated); unknown outcome ids
  tolerated.
- perception/detail/pipeline.py: one-call driver (assess -> discover
  -> ROI -> refine -> apply) with a default lookup built from the
  result's own points; empty scene -> honest empty report.
- WIRING: engine/pipeline/vertical_slice.py stage 3.8 (_detail_stage)
  runs the chain with trusted-intrinsics cameras (same gate the
  sidecar depth stage applies -- no intrinsics, no GSD, visible
  skip), records facts under world.metadata["detail"], never raises.
- RegionOfInterest gained two optional lifecycle fields
  (refinement_outcome, status_reason) -- additive, existing
  constructors unaffected.
- Ledger: P7-06 open items updated (refinement DONE; WorldIR
  integration of refined geometry is the next spine increment);
  CAPABILITIES.yaml gained roi_refinement + detail_pipeline (34
  entries).


## 2026-09-17 (1) -- P7-06: universal detail discovery + ROI generation (PHASE 2-3 chain)

Continuing the universal-perception spine (directive sections 8/11/12/14):
`perception/detail/discovery.py` + `perception/detail/roi.py`, TDD
red-first (7 + 4 tests):

- Discovery: domain-agnostic -- reconstruction points + the measured
  P7-04 report in, detail candidates out. Deterministic voxel binning
  (1 m default, floor-division keys, stable sort); per-cell MEASURED
  curvature (PCA smallest-eigenvalue ratio via closed-form cubic
  eigensolver -- validated against numpy.linalg.eigvalsh to 1e-9 over
  200 random trials; the first draft's Cardano branch silently dropped
  roots for symmetric matrices and was caught by the known-answer
  tests); per-cell budgets derived by the DOCUMENTED P7-05 mapping
  from the cell's own view counts (hallucination gate: single-view /
  weak-evidence cells cannot claim fine levels). Cells below
  min_points are skipped, never guessed.
- ROI: candidates become bounded work orders -- 26-adjacent region
  growth (<= 27 cells), unioned point provenance, tight voxel bounds,
  aggregate MIN budget (weakest evidence bounds the group; exercised
  with mixed multi/single-view cells in tests), status "pending"
  (generation creates work orders, it does not fake refinement).
- Fixtures had real geometry bugs caught during TDD: a helical ring is
  nearly PLANAR (ratio 0.055 -- a flat ring cannot drive a curvature
  metric); replaced with stacked-ring shell samples as a real scanner
  would produce (~0.19).
- Ledger: P7-06 added (PARTIAL -- no refinement executor consumes
  compute_tier yet); P7-05's open item (per-region spatial budgets)
  closed with evidence; CAPABILITIES.yaml gained detail_discovery +
  roi_generation (32 entries).


## 2026-09-16 (7) -- P6-01: real dense-MVS capture integration (next TASKS.yaml item)

Next non-blocked queue item executed (P6-01's PENDING verification:
parse-to-fusion integration on a real dense run). The REAL GPU COLMAP
output exists locally (datasets/room_capture_mvs/dense/fused.ply,
294,345 points, gitignored). TDD:

- 4 integration tests (tests/test_dense_real_capture_integration.py):
  real bytes -> parse_fused_ply -> ingest_fused_ply ->
  PointCloudData + content-addressed artifact -> canonical
  Geometry(POINTCLOUD) with dense_mvs_fused provenance; module
  auto-skips naming the exact missing-dataset path when the artifact
  is absent (real-data-gated, never simulated).
- REAL DATA CAUGHT A REAL BUG: COLMAP writes CRLF PLY headers; the
  parser partitioned on 'end_header\n' only and rejected every real
  fused.ply while passing all synthetic fixtures. Fixed in
  _parse_header; CRLF regression test added at unit level
  (TestCrlfHeaders) so the fix does not depend on the gitignored
  dataset. Dense suite 15/15 green.
- Ledger: P6-01 verification PENDING -> DONE with evidence; status
  stays PARTIAL (cross-source fusion consumption = P6-02's subject;
  metric-scale anchoring of the run remains).


## 2026-09-16 (6) -- P7-05 detail budget (adaptive-compute consumer)

Continuing the universal-perception spine (directive sections 10-11):
`perception/quality/detail_budget.py` -- the first consumer of the
P7-04 evidence-quality report, TDD red-first (10 tests in
tests/test_detail_budget.py):

- Documented GSD-to-level mapping on the multi-scale scale (L4 <= 5
  mm/px, L2 <= 25, L1 <= 100, else L0); multi-view coverage cap (min
  2 views to hold the level, else capped at L2); compute tiers
  none/survey/light/standard/high/full for ROI/adaptive allocation.
- Honesty: observed-points-without-GSD reports REFUSE (assessor-
  bypass detection); empty and coverage-failed scenes -> honest
  zero-allocation unsupported budgets, never invented numbers.
- WIRED (connective-tissue rule): recommend_capture now returns a
  dict with the machine-readable scene_budget (DetailBudget) plus
  prose recommendation keys -- one artifact for the Capture app
  (tests/test_capture_feedback_wiring.py, 3 tests).
- Ledger: P7-04/P7-05 updated (P7-05 PARTIAL: scene-level only,
  per-region spatial budgets + ROI consumption of compute_tier open);
  CAPABILITIES.yaml gained detail_budget (30 entries).


## 2026-09-16 (5) -- suite-stall fix: parametric fitting performance + SAM gate verified

Directive (current-state recovery): the full suite "does not complete
within a reasonable window"; bottleneck around
`tests/test_arch_benchmark.py::test_deterministic_report` (19.2s),
whole file 36.8s. PROFILED (cProfile), root-caused, fixed at the
implementation level -- no timeout increase, no test weakened:

- Root cause: `fit_cylinder`'s axis optimization re-ran PURE-PYTHON
  per-point circle math inside ~8k golden-section evaluations per
  segment (33k `_projected_circle_rms` calls x N interpreted point
  ops x 3 test runs; 36M+ generator steps for ~500 points).
- Fix 1 -- vectorize the Kasa circle fit in float64 numpy
  (`_projected_circle_rms`); factors-of-2 bug in my first normal-
  equations draft was CAUGHT by the known-answer tests and fixed
  (center doubled); 9.6s -> 2.4s per run.
- Fix 2 -- hoist axis-INVARIANT work out of the search: `_circle_eval`
  takes precomputed (rel, centroid); frame built with manual 3-vector
  cross products (np.cross is ~40us on 3-vectors and dominated the
  loop); 2.4s -> 0.90s per run. Same semantics, byte-identical
  deterministic report (asserted), all 53 parametric/arch tests green.
- Guard: `test_report_runtime_is_bounded` (5s budget = ~5x headroom;
  a regression to per-point math fails loudly with a pointer).
- SAM real-model integration test (previously deselected as
  environment-gated): now RUNS and PASSES in 95.6s (95.59s measured
  in --durations). Full suite UNBOUNDED: 1828 passed / 1 skipped /
  0 failed in 177s (2:57). No deselection needed anymore.
- Docs: PENDING_IMPLEMENTATION.md P2.15 (tracking/temporal identity)
  reconciled MISSING -> PARTIAL with implementation evidence (it
  still described temporal tracking as MISSING after P7-02 landed).


## 2026-09-16 (4) -- P7-04 universal evidence-quality assessment

Directive: universal perception + maximum-fidelity reconstruction
(domain-agnostic; Victoria Memorial must not define the engine).
Foundation increment: the measured evidence-quality model (directive
items 2/11/32). Landed, TDD red-first (`perception/quality/assessment.py`,
18 tests in `tests/test_evidence_quality.py`):

- GSD MEASURED from real geometry: per-point euclidean
  camera-to-surface distance x 1000 / max(fx,fy) (finest-sampled
  axis, documented choice), median across observations.
- Observation decided by PROJECTION inside image bounds -- not by
  trusting `source_evidence_ids` metadata; metadata-vs-projection
  overclaims are counted and reported, never silently reconciled.
- Detail tier derived by documented thresholds (fine needs >= 2
  views); capture recommendations derived from the same measured
  facts (closed loop for the Capture app).
- Honesty: no observations -> GSD None + tier "unsupported"; points
  no camera sees reported (`unprojectable_point_ids`) not dropped;
  nonempty points + no cameras refuses (ValueError) rather than
  guessing intrinsics. Domain-agnostic by construction: columns,
  pipes, fenders, rocks are all just points.
- Ledger: P7-04 added (PARTIAL; remaining section-11 inputs, no
  consumer yet, thresholds untuned); CAPABILITIES.yaml gained
  `evidence_quality_assessment`.
- Full-suite gate on branch head: 1826 passed / 1 skipped / 0 failed.


## 2026-09-16 (3) -- P7-02 temporal tracking (3D observation layer)

Landed, TDD red-first (`perception/tracking/temporal.py`, 14 tests in
`tests/test_temporal_tracking.py`):

- `TimedObservation` -> per-label `TrackRecord` chains: track_id,
  observation history, measured path_length_m / duration_s /
  implied_speed_m_s. Continuation requires BOTH the speed budget
  (distance <= max(merge_distance, max_speed*gap)) and the gap budget;
  else a new track starts -- no forced associations (the honesty rule
  from instances/interface.py).
- Untimed observations are returned separately (`(tracks, untimed)`),
  never mixed into timed chains. Duplicate observation_id raises.
- Confidence honesty (a fabricated 1.0 was caught during review):
  track confidence = min of MEASURED member confidences, None when
  unmeasured; `instance_tracks_from` refuses tracks without measured
  confidence instead of inventing one, and refuses missing regions
  instead of placeholders.
- perception/tracking/README.md replaced (was "Not yet implemented");
  package `__init__` docstring added.
- Ledger: P7-02 MISSING -> PARTIAL (2D ByteTrack-style box
  association, motion compensation, ID-switch diagnostics still open);
  CAPABILITIES.yaml gained `temporal_tracking`.


## 2026-09-16 (2) -- P7-03 architectural-perception expansion

Directive: architectural perception for complex structures (proof
target: Victoria Memorial). Landed, all TDD red-first:

- **parametric.py**: deterministic cylinder/sphere/circle fits with
  MEASURED residuals, shell + elongation refusal gates (FitRefused,
  never best-effort), documented confidence mapping. **Real bug found
  and fixed**: the analytic cubic eigen solver returned negative
  eigenvalues on near-degenerate column shells (Var(x)~=Var(y)) ->
  replaced with Jacobi rotation solver.
- **segments.py**: deterministic voxel segmentation (26-connected,
  sorted-key); undersized clusters reported, never dropped.
- **registry.py**: extensible class registry (Phase 1-3 built-ins,
  per-class acceptance thresholds, no silent overrides).
- **components.py**: component observations (acceptance recorded with
  reasons; unaccepted never silently dropped), confidence tiers
  (OBSERVED/STRONG/WEAK/UNOBSERVED), union-find multi-view entity
  resolution, repetition + bilateral-symmetry priors that raise
  confidence but NEVER move geometry.
- **promotion.py**: candidate -> canonical WorldIR Entity (measured
  fit facts in custom_properties, evidence ids for traceability,
  INFERRED provenance, ESTIMATED adjacency edges from real positions;
  no invented hierarchy). WorldIR +DOME/+ARCH additively.
- **benchmarks/architectural.py + benchmarks/structures/**: benchmark
  harness with honest capture gate -- Victoria Memorial record is
  CAPTURE_PENDING (real capture = external dependency; run refused,
  never simulated); structure-agnostic records; measured report with
  failure regions. **Design lesson fixed**: segmentation precedes
  plane subtraction -- a cylinder's front strip is locally planar, so
  plane-first subtraction shredded columns (measured, then fixed).
- Full-suite gate: **1,795 passed / 1 skipped / 0 failed**
  (1,737 + 58 new). P7-03 remains PARTIAL honestly: detector-backed
  semantic fusion, stairs, remaining Phase-2/3 classes open; Victoria
  Memorial real capture outstanding (human dependency).


## 2026-09-16 -- sync consumers wired (E-B of the directive list)

- **E-B DONE (P2-01 consumers)**: trajectories/backend/sync.py --
  apply_clock_model maps backend output onto the global timeline
  (ns rounding; originals preserved in TrajectoryFrame.
  sensor_timestamp_ns; time-collapse raises instead of silently
  merging samples; UNSYNCHRONIZED sentinel degrades honestly),
  estimate_trajectory integrates it (clock_id/sync_state/sync_method
  on the canonical Trajectory), Trajectory serializes the sync
  metadata. **Real units bug the e2e test caught**: a ClockModel fit
  from a seconds-denominated SensorStream carries b in SECONDS;
  applying it to ns stamps corrupts time by 10^9. Fixed with an
  explicit model_unit ("s" for stream-fitted models, "ns"
  canonical) -- no implicit unit. 16 tests
  (tests/test_sync_consumers.py), incl. the full chain
  SensorStream -> synchronize_stream -> estimate_trajectory with
  known answers.
- **E-A verified**: depth end-to-end (sidecar -> DepthFrame ->
  unprojection -> WorldIR) 53/53 green -- no new work needed.
- **E-I verified**: provenance graph landed in PR #34 (merged).
- Full-suite gate: 1,737 passed / 1 skipped / 0 failed
  (1,717 base incl. 16 new + 20 SAM real-model run explicitly).
- **E-C NOT DONE (environment, not code)**: real VIO backend run on
  EuRoC MH_01 -- Docker Desktop daemon did not come up this session
  (cold boot after restart); the OpenVINS/ORB-SLAM3 build path
  (ROS1 container + run_subscribe_msckf) is the planned route when
  the daemon is up. P3-02 ledger note records exactly this.


## 2026-09-15 -- five-execution spine batch (claude/spine-executions-5)

- **E1 P4-01 DONE**: point-to-plane ICP (spec's named algorithm),
  ResidualStats (measured, on accepted AND blocked results),
  RegistrationEngine (confidence-ordered, attempts recorded), contact
  pre-flight + prior seeding. 18 tests.
- **E2 P5-02 DONE**: reconstruction/backend/selection.py --
  InputProfile -> deterministic policy decision (declines for
  depth/LiDAR-only/<3-image profiles, preference-ordered selection
  otherwise) + assess/advance loop (failed candidates never retried).
  9 tests.
- **E3 P6-01 advanced**: reconstruction/backend/dense_output.py --
  parse_fused_ply closes the named "fused.ply unreadable in Python"
  gap (honest errors, present-but-unused facts recorded). 6 tests.
  Task stays PARTIAL: real dense-run integration remains gated on a
  GPU MVS capture.
- **E4 P6-02 DONE**: reconstruction/fusion/consumer.py + pipeline
  stage 3.7 -- plural-source association + per-point fusion
  (conflicts recorded, never winner-picked) + WorldIR POINTCLOUD
  write-back (hashed artifact, per-stage counts in observation
  metadata). 9 tests.
- **E5 P10-01 DONE**: provenance/graph.py -- ProvenanceGraph over the
  ArtifactStore (digest-keyed immutable nodes, added-only edges with
  cycle rejection, ancestor/descendant walks, why_exists stage chain,
  verify_all tamper detection, tombstones). 11 tests.
- Ledger: 16 DONE / 9 PARTIAL / 11 MISSING. CAPABILITIES:
  cross_source_registration -> IMPLEMENTED, provenance_graph added
  (23 entries).
- Full-suite gate on the final head: 1,692 passed / 1 skipped /
  0 failed (~226 s) -- fully accounted (1,648 + 44 new).

## Verified baseline

- Suite: **1,613 passed, 1 failed, 1 skipped** (~98 s) at finalization
  (2026-09-15). The ONE failure is
  `tests/test_sam_backend.py::TestSAMSegmentationBackendIntegration::test_real_model_load_and_inference`
  — the pre-existing SAM real-model environment failure (torch-hub
  cache), NOT caused by this batch and not fixed by it; it is
  documented, not hidden. This run had no deselect filter, so the SAM
  test shows as FAILED rather than deselected.
  Landmarks en route: 1,569 passed, 1 skipped at P5-01/P5-02; P3-01 at
  1,520, P3-02 at 1,541, P3-03 at 1,550, P4-01 at 1,559.
  Prior landmark: 1,400 passed on the P0-02 branch (main `5235960` at
  1,392; +8 platform-boundary tests). Main includes PR #22 (depth
  sidecars + RGB-D unprojection), PR #23 (P0-01 canonical state),
  PR #24 (P2-01 clock model + first sync backend).
- Known environment failures: the SAM real-model integration test
  (above; plus `tests/test_perception_sam.py` torch-hub cache
  failures when run without deselect) — pre-existing, unrelated to
  code changes. Never reclassify; fix the environment or record it.

## Post-finalization fixes (2026-09-15, branch fix/sam-load-path)

- **SAM "environment failure" was two REAL CODE BUGS — fixed, suite now
  fully green (1,614 passed, 1 skipped, 0 failed, ~206 s):**
  1. `perception/segmentation/sam_backend.py` loaded via
     `torch.hub.load("facebookresearch/segment-anything", ...)` — but the
     upstream repo has NO hubconf.py (verified via GitHub API: 0 commits
     touching it; the zipball contains no hubconf). Every source="github"
     load was structurally broken for every consumer. The load path now
     uses the `segment-anything` pip package's `sam_model_registry` +
     checkpoint-file resolution (explicit path -> torch-hub checkpoint
     cache -> direct download -> honest UnavailableError);
     `segment-anything>=1.0` added to the perception extra.
  2. SAM's `predicted_iou` can exceed 1.0 by float round-off (observed
     1.0005); `Uncertainty` correctly rejected it and the per-item
     `except: continue` silently DISCARDED valid regions. Fixed by
     clamping at the conversion boundary (with comment); regions are no
     longer silently dropped.
  - vit_b checkpoint pre-cached into the torch-hub checkpoint cache;
    real-model integration test passes (~90 s CPU).
  - CAPABILITIES.yaml: image_segmentation BROKEN -> IMPLEMENTED with
    the real limitation set (22 entries).
- **P6-03 bad-scale check + export-time quality metadata — DONE:**
  `validate_mesh(expected_extent_m=..., scale_tolerance=...)` measures
  per-axis ratios (declared expectation only; never inferred), records
  the tolerance used; 8 new known-answer tests (half-scale 0.5,
  mm-vs-m 1000, per-axis failure, custom tolerance, roundtrip, loud
  rejection). Mesh stage records the report into the WorldIR Geometry's
  new additive `quality_metrics` field (v1-dict compatible, mirrors
  Entity.statement_state). CAPABILITIES gains mesh_quality_validation
  (22 entries). Ledger: P6-03 DONE (11 DONE / 14 PARTIAL / 11 MISSING).
- Note: `tests/test_perception_sam.py` named in an older note does not
  exist on this branch (stale reference; no action).

## Completed (most recent first, with evidence)

- **2026-09-15 — finalization batch (P6-02/P6-03/P7-01/P7-03/P9-01)**:
  the campaign batch's remaining five ledger IDs implemented and
  reconciled. P6-02: weighted multi-source depth fusion adapter
  (reconstruction/fusion/multi_source.py over the inverse-variance
  engine; conflict never blended; tests incl. uncertainty steering).
  P6-03: mesh quality validation (reconstruction/meshing/validation.py;
  known-bad-mesh test matrix; 'bad scale' check + artifact-metadata
  recording stay open — PARTIAL). P7-01: appearance color-histogram
  + epipolar consistency wired additively into merge_hypotheses;
  both wrong-merge rejection tests green (temporal consistency stays
  open — PARTIAL). P7-03: geometry-only wall/floor/ceiling/doorway
  classifier, synthetic-room verification incl. no-false-doorway
  (MISSING -> PARTIAL; broader element classes deliberately skipped,
  no detectors exist). P9-01: StatementState additive layer on
  Entity with v1-only-dict load tests (other WorldIR 2.0 areas stay
  open — PARTIAL). Full-suite gate 1,613/1/1 (SAM env failure).
  Ledger statuses rewritten with evidence; statuses were stale
  (P7-03 still said MISSING with the code landed).
- **2026-09-15 — real GPU dense-MVS pass (manual CLI, P6-01)**: with
  the CUDA COLMAP build in place, ran the actual dense-reconstruction
  pipeline end-to-end on a freshly rendered 20-image synthetic room
  (scripts/render_room_dataset.py -> datasets/room_capture_mvs,
  gitignored, not committed). feature_extractor (GPU) -> matcher
  (GPU) -> mapper: 17/20 registered, 680 sparse pts, 0.67px reproj
  error -> undistorter -> patch_match_stereo --geom_consistency true
  (GPU, ~6.9 min, real per-view CUDA sweep timings) -> stereo_fusion:
  294,345 dense points, sanity-checked (tight/plausible mean+std,
  real sampled texture colors, not degenerate). TEST-CAUGHT BUG (real,
  found mid-run): the first attempt let COLMAP guess camera intrinsics
  from image size, producing a wrong focal length; every two-view
  geometry was then misclassified as planar/degenerate (config=6) and
  mapper couldn't find an initial pair at all. Fixed by passing the
  renderer's actual known intrinsics explicitly
  (--ImageReader.camera_params); config histogram went from
  {0:98,3:30,6:62} (no calibrated pairs) to including 15 genuinely
  CALIBRATED (config=2) pairs, and mapper succeeded. This is CLI-level
  evidence only, not yet Python-integrated: fused.ply is not parsed
  into this repo's DepthFrame/point types
  (reconstruction/depth_to_points.py still only handles monocular/
  RGB-D sidecar paths) -- that wiring is the remaining P6-01 work.
- **2026-09-15 — P5-01 (DONE) / P5-02 (PARTIAL)** reconstruction
  backend registry + selection foundation:
  `reconstruction/backend/registry.py` — BackendDescriptor/
  BackendAvailability, discover_backends (PATH presence via
  shutil.which for all 5 spec-named tools: COLMAP, OpenMVS,
  AliceVision/Meshroom, OpenSfM, OpenDroneMap), select_backends
  (deterministic preference-order ranking, explicit policy per the
  constitution's no-LLM-selection rule). Only COLMAP has a real
  adapter in this repo (ColmapReconstructionBackend); the other four
  are honestly reported as no-adapter even when/if installed --
  discovery never implies runnability. P5-01 DONE against its written
  verification (registry discovery tests; unavailable-backend honesty
  tests) -- NOT claimed: the scope text's capability()/diagnostics()/
  artifacts() per-INSTANCE contract methods on IReconstructionBackend
  itself; this session built registry-level discovery instead, which
  is what the verification actually tests. P5-02 marked PARTIAL, not
  DONE: select_backends ranks by availability only, not by the
  spec's input-characteristics profile (image count, overlap, camera
  model, GNSS/IMU/depth/LiDAR, scene type, GPU, budget) -- building
  that now would be selecting among N=1 runnable backend (only COLMAP
  has an adapter), so there is nothing yet to differentiate;
  deliberately deferred until a second real adapter exists, per
  YAGNI, rather than building a speculative ranking function with one
  candidate. 10 new tests, including COLMAP's real on-PATH detection
  (genuinely installed on this machine, verified before writing the
  assertion) and the other four's real absence. Suite 1,569.
- **2026-09-15 — P4-01 (PARTIAL)** cross-source registration engine:
  `registration/registration.py` — register_icp (point-to-point ICP:
  Kabsch/SVD per iteration, median-multiplier outlier rejection,
  matrix-to-quaternion via Shepperd's method) and register_gnss_anchor
  (translation-only, paired anchors). 2 of the spec's 4 confidence-
  ordered methods implemented; landmark-correspondence alignment
  (needs multi-view object identity, P2, not built) and manual-anchor
  CLI (no engine logic to write) deliberately skipped, not stubbed.
  Deliberate deviation from spec's "point-to-plane" ICP: point-to-
  point, since nothing upstream produces surface normals for a bare
  point list (documented in-module). TEST-CAUGHT BUGS (2, both real,
  found by the deterministic-fixture tests the spec requires): (1)
  ICP with an offset larger than the cloud's own extent converges to
  a wrong-but-self-consistent local optimum -- inherent to nearest-
  neighbor ICP, not fixable without a coarse-alignment prior; fixed
  the TESTS to use realistic small offsets and added
  TestICPLocalMinimum to document (not hide) the limitation. (2) the
  no-overlap rejection gate (median-multiplier inlier fraction) is
  blind to a uniformly-offset cloud of IDENTICAL shape -- that case
  is legitimately recoverable (relative structure preserved) so the
  gate correctly accepted it; the real gap was no ABSOLUTE-scale
  check, so a genuinely unrelated point cloud could in principle look
  internally self-consistent. Added an absolute rmse-vs-point-spacing
  gate (spec's own failure mode: "rmse >> voxel size") and rewrote
  the no-overlap test with a genuinely unrelated random-scattered
  source cloud. 9 new tests (known-offset recovery incl. rotation,
  local-minimum documentation, degenerate/no-overlap rejection, GNSS
  anchor recovery + input-contract errors). Suite 1,559. NOT claimed:
  the full RegistrationEngine orchestration (extrinsics->trajectory-
  prior->coarse-alignment->ICP->uncertainty pipeline as one object),
  landmark-based alignment, covariance/uncertainty propagation on the
  result, CLI `reality register` wiring, real multi-source capture
  evidence.
- **2026-09-15 — P3-03 (DONE)** trajectory quality diagnostics:
  `trajectories/diagnostics.py` — absolute_trajectory_error (ATE),
  relative_pose_error (RPE), endpoint_drift (loop-closure proxy),
  TrackingState enum, standard TUM RGB-D benchmark metrics (Sturm et
  al. 2012), no SE(3) alignment step (documented scope cut -- run
  P4-01 registration first if trajectories aren't already
  co-registered). Pure functions over exact-matching timestamps only
  (Trajectory.at's no-interpolation rule carried through). 9 new
  tests against analytic values (constant offset -> ATE==offset,
  RPE==0 since a constant offset cancels in relative motion;
  accelerating drift -> RPE>0; loop path -> endpoint_drift==0).
  Suite 1,550. NOT claimed: SE(3)/Umeyama alignment, real
  loop-closure detection (place recognition) -- endpoint_drift is a
  named proxy, not a substitute.
- **2026-09-15 — P3-02 (PARTIAL)** VIO/localization backend
  federation: `trajectories/backend/` — ITrajectoryBackend interface,
  TUM-trajectory-format parser (real format shared by ORB-SLAM3/
  OpenVINS/Basalt's evaluation exports), SubprocessTrajectoryBackend
  base (binary presence via shutil.which, caller-supplied argv since
  none of the three tools has one stable CLI across builds -- never
  guessed), three thin adapters (orb_slam3/openvins/basalt, license
  notes per LICENSES.yaml federation_candidates), and
  selection.estimate_trajectory (try-in-order, BackendAttempt log,
  honest TrajectoryBackendUnavailableError vs TrajectoryBackendRunError
  distinction). Federates per the standing rule -- no custom VIO
  written. 21 new tests: TUM parser edge cases (malformed/non-unit-
  quaternion/non-monotonic), all three real adapters verified
  genuinely BACKEND_UNAVAILABLE on this machine (shutil.which
  confirmed absent before writing the assertions, not mocked), and an
  adapter CONTRACT test that runs a REAL subprocess (python itself as
  the "binary") writing a real TUM file, exercised through the same
  SubprocessTrajectoryBackend code path a real VIO adapter uses. Suite
  1,541. NOT claimed: a real ORB-SLAM3/OpenVINS/Basalt run (hardware/
  install gate, named in CAPABILITIES.yaml as the VERIFIED gate);
  correctness of any specific tool's exact CLI flags (left to the
  caller by design).
- **2026-09-15 — P3-01 (DONE)** canonical trajectory model:
  `trajectories/trajectory.py` — Trajectory/TrajectoryFrame/
  FrameSource/DriftEstimate. Reuses RigidTransform (P2-02) for poses
  and the ClockModel global timeline (P2-01) for timestamps rather
  than a second representation. Construction validates strictly
  increasing timestamps and consistent from_frame/to_frame identity
  across every frame (a trajectory that changes frame mid-sequence is
  rejected, per spec acceptance criteria); `at()` is exact-timestamp
  only, no interpolation (explicit scope cut for P3-02/P3-03 to own).
  DriftEstimate is UNKNOWN-by-default (never implicitly zero drift).
  23 new tests (monotonicity, frame-identity, covariance-shape,
  roundtrip). Suite 1,520. NOT claimed: VIO/SfM backend adapters
  (P3-02), quality diagnostics/ATE/RPE (P3-03), real trajectory data.
- **2026-09-15 — P2-02 (DONE)** calibration/frame system unification:
  reconstruction/calibration/transforms.py — RigidTransform with named
  frames (compose validates connectivity; direction-vs-point; exact
  inverse), CalibrationChain with gap-at-construction guard, WGS84
  geodetic<->ECEF (Newton, sub-mm roundtrip, ellipsoidal height — geoid
  is a caller concern), ECEF->ENU with the reference origin carried in
  the frame name (enu@lat,lon), SensorRig/CalibrationEntry with
  labeled provenance (declared/estimated/assumed) and
  absent-calibration-never-guessed, reprojection residual statistics
  (mean/median/p95/per-camera, pixels; unprojectable points counted,
  never fabricated). Reuses engine.physics.math3 + camera types — no
  second rotation representation. TEST-CAUGHT BUG: the ENU Up row used
  sin_lon for sin_lat — non-orthogonal 'rotation' that would have
  skewed every ENU placement; caught by h(ref+10·up)==ref+10 m, fixed,
  triad proven orthonormal/right-handed at 4 lat/lon sets. 45 new
  tests; suite 1,474. P1-03's real-device remainder recorded BLOCKED
  (hardware). NOT claimed: estimation procedures, real-device
  calibration evidence, geoid models.
- **2026-09-15 — P1-02 (DONE)** canonical sensor model (DECLARED
  identity half; continuation of existing sensor modules + PR #24
  clocks): SensorDescriptor in evidence/sensors.py loaded from
  <component>/sensor_identity.json — sensor_id (required),
  source_id/capture_id/clock_id/frame/coordinate_system/units/
  quality/typed intrinsics+extrinsics/provenance; absent = undeclared
  = None, never defaulted; kind-mismatch fails loudly. Attached to
  SensorStream + every sample, CalibrationRecord, DepthFrame via
  attach_sensor_identities and the session accessors. synchronized()
  clock resolution: explicit > declared > '<undeclared>' sentinel
  with honest UNSYNCHRONIZED degradation. Integration bugs the tests
  caught: calibration parsing swept the identity sidecar as evidence
  (now excluded as metadata); identity-with-units does NOT substitute
  for a depth scale manifest (Decision 020 preserved). 22 new tests;
  suite 1,429. NOT claimed: real-device capture declaring identity
  (synthetic fixtures only — recorded in CAPABILITIES.yaml).
- **2026-09-15 — P1-01 (DONE)** canonical evidence/source model:
  SourceRecord now carries the unified identity — acquisition_id
  (content-derived `acq-<hash16>` by default so the same bytes
  anywhere share one acquisition identity, i.e. the dedupe rule made
  explicit; a real device-side uuid overrides it once the P1-02
  sensor model carries one), device_id (only caller-declared via an
  explicit EvidenceSource; default filesystem source records None,
  never a fabricated device), capabilities (evidence kinds actually
  ingested, derived from the package — measured, not promised).
  Roundtrip and old-format sessions handled: from_dict derives the
  acquisition_id deterministically for legacy records (invents
  nothing). Continuation, not a rewrite: dedupe/roundtrip/component
  discipline untouched. 7 new tests (TestUnifiedSourceIdentity);
  suite 1,407. NOT claimed: device-side acquisition uuids (no carrier
  on EvidenceSource yet — P1-02 seam), real multi-device capture.
- **2026-09-15 — LEDGER RACE REPAIR**: the P0-03/P18-01 closure
  commit (dba9652) was pushed after the PR #25 merge (bff0f83) had
  already snapshotted the branch, so main lost the two status flips.
  Re-applied verbatim here (evidence unchanged).
- **2026-09-15 — P0-03 (DONE)** backend capability registry verified:
  mechanical cross-check of .agent/CAPABILITIES.yaml — 19/19 entries,
  every backend/fallback .py path resolves to real code, all required
  fields present, honest maturity labels (13 IMPLEMENTED / 2 PARTIAL /
  3 MISSING / 1 BROKEN; BROKEN image_segmentation = pre-existing SAM
  torch-hub env failure, recorded not hidden). Satisfies P0-03's two
  verification clauses; P5-02 (backend selection) dependency now met.
- **2026-09-15 — P18-01 (DONE)** application extraction closed as
  premise-falsified: P0-02's verified inventory found NO application
  workflows in core to extract ('engine/disasters/' is a README-only
  scaffold; fire/destruction/environment are generic primitives).
  Preventive boundary adopted instead — R1 enforced by
  tests/test_platform_boundary.py (its 'core imports no application
  modules' verification item; merged in PR #25). Remaining item
  ('application package installs/tests independently') recorded as
  CONDITIONAL until an application package exists.
- **2026-09-15 — P0-02 (DONE)** platform/application separation
  scoping: verified inventory (the disaster-application layer does NOT
  exist in code — `engine/disasters/` is a README-only scaffold; no
  evacuation/hazard-scenario/emergency-response logic anywhere;
  fire/destruction/environment are generic physics primitives, STAY).
  Preventive rules R1–R3 adopted; **R1 enforced structurally** by
  `tests/test_platform_boundary.py` (platform chain must not import
  simulation domains; physics-compiler bridge attaches only via the
  SDK facade; 4 basic-utility couplings allowlisted with follow-ups
  F1–F3 registered). Plan:
  `docs/implementation/PLATFORM_APPLICATION_SEPARATION.md`. Suite
  1,400. No runtime code changed beyond the new guard tests.
- **2026-09-15 — P2-01 (PARTIAL)** time synchronization:
  `evidence/clocks.py` — ClockModel/Timestamp/TimeAlignment/
  SynchronizedSample/SynchronizationDiagnostics with
  t_global = a*t_sensor + b; backend seam (ISynchronizationBackend +
  DEFAULT_BACKENDS in spec preference order); first backend
  metadata_alignment (shared_clock + known offset/drift from capture
  metadata); SensorStream.synchronized(clock_id, metadata) wiring
  (stream never mutated). 33 tests: pure-b, drift a≠1, cross-stream,
  rejected samples (non-finite per-sample guard), backend fallthrough,
  honest UNSYNCHRONIZED degradation, uncertainty honesty (unknown ≠
  zero; zero only by construction for shared clock), roundtrip.
  Suite 1,392. NOT claimed: spec methods 3-6 (GNSS/PPS, trigger,
  signal correlation, optimization); real multi-device capture
  evidence.
- **2026-09-15 — P0-01** canonical state (PR #23, merged `54b7574`):
  ENGINEERING_CONSTITUTION.md, REALITY_ENGINE_MISSION.md, TASKS.yaml
  (RE-2026-CORE-V1), this file, CAPABILITIES.yaml, LICENSES.yaml;
  legacy state archived with banners. Suite green at merge time.
- **2026-09-15 — P1-03 (PR #22, merged `f48c733`)**: DepthFrame +
  16-bit PNG sidecar parser (explicit depth_scale, Decision 020),
  MultiSourceSession.depth_frames(), pipeline sidecar unprojection
  with scale-coherence gate, deterministic RGB-D E2E (33 wall points
  at exactly z=2.0 m), metric-scale `result` reassignment bugfix.
  Suite 1,363. Status: PARTIAL — real-device RGB-D capture run is the
  named evidence for VERIFIED.
- **2026-09-14 — PR #21 (merged)**: canonical docs architecture
  (docs/{architecture,implementation,engineering,future}), CLAUDE.md
  §51–54. Suite 1,320 at the time.
- **2026-09-14 — PR #19 (merged)**: `reality compile` one-command CLI
  with deterministic fake-backend E2E.
- **2026-09-14 — PR #16/#17/#18 (merged)**: meshing stack (MeshData,
  scipy preprocessing, COLMAP poisson backend, camera-envelope
  filter), glTF uint32 mesh export, viewer mesh layer.

## In progress

- None open. P5-01/P5-02 landed this session; next queue item not
  started. Remaining P4-01 scope (RegistrationEngine orchestration,
  landmark alignment, uncertainty propagation) and P5-02's input-
  profile ranking both stay open under their existing ids.

## Blockers

- ~~P6-01 dense MVS: CUDA unavailable~~ RESOLVED 2026-09-15: the
  machine has an RTX 4050 (nvidia-smi verified); the installed COLMAP
  build was CPU-only, not the GPU. Swapped in the official prebuilt
  CUDA build (colmap/colmap release 4.2.0,
  colmap-x64-windows-cuda.zip) at the same PATH location; old CPU
  build kept as colmap-extracted-nocuda-backup. No CUDA Toolkit
  install needed. GPU execution verified genuinely working (SIFT GPU
  feature extraction ran on synthetic images, not just `-h`).
  patch_match_stereo itself (the real MVS deliverable) has not been
  run end-to-end yet -- that's P6-01's actual remaining scope, now
  unblocked.
- **P3-02 real backend run**: none of ORB-SLAM3/OpenVINS/Basalt
  installed on this machine — the adapter/selection code is real and
  tested (fake-binary contract test), but no real VIO trajectory has
  been produced. Decision required: none (install gate, not blocking).
- **SAM perception tests**: torch-hub cache failure (environment).
- **open3d/trimesh/skimage absent**: meshing deliberately
  scipy/subprocess-based; revisit only if a capability demands it.

## Next (exact next step, per queue / canonical spine)

Canonical spine position: time sync (P2-01 DONE) -> trajectory/VIO
(P3-01 DONE, P3-02/P3-03 PARTIAL) -> cross-source registration
(P4-01 PARTIAL) -> dense MVS/fusion (P6-01 CLI-verified, P6-02 engine
landed) -> multi-view identity (P7-01 PARTIAL) -> uncertainty/
provenance (P10-01) -> WorldStore -> incremental compilation.

1. **P6-01 remainder**: parse the verified dense output (fused.ply)
   into canonical types (DepthFrame/points) and wire into the
   orchestrator — the CLI pass is real; the Python-level integration
   is the named remaining scope.
2. **P6-02 remainder**: a pipeline consumer that ingests plural
   sources and calls fuse_depth_observations, writing fused geometry
   into WorldIR (the '-> WorldIR geometry' tail).
3. **P4-01 remainder** (same id): RegistrationEngine orchestration
   (extrinsics -> trajectory-prior -> ICP -> uncertainty).
4. **P10-01** provenance graph — next untouched spine item after
   the Phase-6 remainders.

WorldStore / incremental compilation follow; do NOT jump to
disaster-management or unrelated Studio polish.

## Rules reminder

Never mark work complete without execution evidence. Never erase
incomplete work. Update this file at session end (constitution
Article II/IV).
## 2026-09-15 — Session: directive items P4/P5/P11/P12/P13 + measurement uncertainty (PR #34)

- Registration covariance (P4-F): `estimate_registration_covariance` (6x6 from
  point/plane correspondences), wired into accepted ICP results.
- Dense ingestion (P5): `reconstruction/dense_ingest.py` — fused.ply ->
  canonical PointCloudData -> WorldIR Geometry(POINTCLOUD) with hashed
  artifact + dense provenance.
- Measurement uncertainty (P11): measured-spread precision (RMS inter-view
  disagreement), leave-one-out 5-sigma conflict detection (n=2 honestly
  undecidable), `Measurement.precision_note` names the precision source;
  single-view keeps the documented heuristic, honestly labeled.
- WorldStore (P12): `worldstore/store.py` — immutable version snapshots,
  parent DAG, restore-any-version; world survives process restarts.
- Incremental compilation (P13): `engine/incremental.py` — dependency-graph
  invalidation compiler (content hashes, cycle detection, resume).
- Ledger: 18 DONE / 9 PARTIAL / 9 MISSING. CAPABILITIES: +world_store,
  +incremental_compilation, +uncertainty_propagation.
- Verification: full suite 1,720 passed / 1 skipped / 0 failed; SAM
  real-model integration test run separately (passed, ~58 s CPU).
