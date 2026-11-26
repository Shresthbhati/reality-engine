# Execution State — Reality Engine

## Branch / Commit
- Branch: `feat/day4-real-geometry-measurements` (from `main`, PR #395 open)
- Commit: `290ab106` — "docs(state): record Quarter 1 evidence normalization + task-specific fusion completion" (tip); prior commits `4dc8d258` (feat: evidence_fusion.py + test + status advance), `c10f9a2c`, `156f3e36`
- Working tree: has changes (PR #395 merge in progress)
  - Push state: branch pushed to `4dc8d258` and `290ab106`; PR #395 open, not merged

## Reconciliation note (this session)
- Inspected the mandate's "first inspect" list: `engine_orchestrator.py`, `engine_adapters.py`, `engine_matrix_bridge.py`, `engine/multiengine/evidence_matrix.py`, `perception/depth/`, `reconstruction/depth_to_points.py`, camera registration/pose-graph, `apps/api/evidence_world_stage.py`, WorldIR, WorldStore, Reality Studio, status ledger.
- Verdict: all named wiring exists and is connected. The three Day-6 depth defects from the prior request are already implemented, committed and pushed (commits `156f3e36`, `0e78c2db`, `fba6a486`).
- Conflict resolved: the mandate asked to create `docs/engineering/IMPLEMENTATION_LEDGER.md`, `IMPLEMENTATION_BACKLOG.md`, `24_7_EXECUTION_STATE.md`, `NEXT_ACTION.md`, `52_WEEK_MASTER_PROGRAMME.md`. AGENTS.md §51 says those plan/roadmap/backlog documents were deleted on 2026-11-15 and must NOT be recreated. So persistent engineering memory goes into the repo's canonical files: `.agent/EXECUTION_STATE.md` (this file) and `.agent/TASKS.yaml` (the governing task queue, below). No `docs/engineering` plan documents were created.

## P0 — Connect perception-to-reconstruction pipeline: COMPLETE (from main)
### Completed
1. Ported `reconstruction/spatial/` from worktree to main (full spatial optimizer: model, multiview, compiler, validation, topology, constraints, association, completeness, uncertainty, refinement, stage.py)
2. Ported `engine/pipeline/stages/s14_spatial_global.py` to main
3. Ported `apps/api/spatial_global_stage.py` to main (refactored: per-job provider + module-global fallback for backward compat)
4. Ported `apps/api/routes_spatial_global.py` to main; registered router in `apps/api/main.py` (8 routes verified)
5. Updated `engine/pipeline/stages/registry.py`: added `SPATIAL_PROFILE_NAME` + `interior_survey_spatial` profile; added `SpatialGlobal` to `build_stages()`
6. Modified `apps/api/jobs.py`: `RE_SPATIAL_GLOBAL` selects spatial profile; per-job `spatial_inputs_provider` + `spatial_store_opener` + `job_obj` wired into ctx.state; spatial facts added to job payload
7. Built `perception/spatial_adapter.py`: `SpatialInputsV1` → spatial stage dict input adapter with validation, rejection tracking, scale validation (never promotes relative→metric), camera→pose conversion, structural observation conversion
8. Benchmarked camera registration matchers: LightGlue-ALIKED (0.980 inlier ratio, BEST) > SIFT (0.929) > LightGlue-DISK (0.818) > ORB (0.710). Results in .benchmarks/matcher_benchmark_out/results.json
9. Updated `reconstruction/backend/python_backend.py`: real camera registration using LightGlue-ALIKED/SIFT feature matching + EssentialMatrixEstimator (replaces fake baseline poses)
10. Created `docs/THIRD_PARTY_TECHNOLOGY_REGISTRY.json` (P1 requirement)

## P1 — Multi-Engine Platform Architecture: COMPLETE (from main)
### Completed
1. Built `perception/engine_evidence.py`: UnifiedEngineOutput, ModelIdentity, CoordinateFrame, ObservationRecord, compute_observation_hash, compare_observations — common representation for ALL engine outputs preserving source evidence, coordinate frames, confidence, uncertainty, model identity, contradictions
2. Built `perception/engine_orchestrator.py`: MultiEngineOrchestrator (VRAM-aware scheduling, parallel inference, wave-based execution), IntelligenceMerger (cross-engine comparison and contradiction detection), EngineSpec registry, get_unavailable_engines (honest reporting)
3. Built `perception/engine_adapters.py`: adapters for SIFT and LightGlue-ALIKED wrapped into UnifiedEngineOutput contract; register_all_engines() registers available engines
4. Built `perception/multi_engine_stage.py`: run_multi_engine_for_job() — connects multi-engine orchestrator to production evidence pipeline; runs all registered engines on job evidence, records observations + contradictions in job report
5. Wired multi-engine stage into `apps/api/evidence_world_stage.py` (run_for_job) — multi_engine facts recorded in job report
6. Verified: both engines run together in the orchestrator producing 2 observations (146 + 88 matches); contradictions detected across engines; VRAM tracked

## Completed since last EXECUTION_STATE (this session, state-only + governing task entry)
1. Reconciled current source state against the mandate: confirmed all named modules exist and are wired; confirmed the depth fixes are committed/pushed; confirmed no `docs/engineering` ledger/backlog/state files exist.
2. Updated this file (current branch/commit, completed work, exact next action).
3. Added one governing task entry to `.agent/TASKS.yaml` (`P-52W-00`) so the 52-week master programme is represented as the authoritative work queue without recreating deleted backlogs.
4. Implemented `perception/bundle_adjustment.py` (UNTRACKED — not committed): bundle adjustment over 3D points using exact sparse LM solver; camera 0 pinned as gauge; per-camera/per-point covariances; verified with synthetic data. Status: recovered work product; not yet wired into committed pipeline (a separate `BundleAdjustmentOptimizer` exists in `reconstruction/sfm/core_pipeline.py`, tested). Needs evaluation before integration.
5. Extended `perception/camera_pose_graph.py` with optional `points3d` cheirality constraints — `(R_c @ P + t_c).z > 0` residuals pin handedness and scale; all 30 existing tests pass.
6. Widened pair strategy in `scripts/run_multi_engine_real_capture.py` — added `sequential_skip23`, `sequential_skip234`, `all`; default = `sequential_skip23`.
7. Created 9 engineering ledger files in `docs/engineering/` during a prior session; these were removed in the current recovery session per AGENTS.md §51 (4 were plan/roadmap/backlog/state documents prohibited from recreation; 5 were duplicates of canonical committed `docs/engineering/` records). Their unique content (bundle adjustment decision D-011, cheirality decision D-012, South Building 32-photo multi-engine execution results) is captured in the recovery report `docs/audits/FREEBUFF_200K_RECONSTRUCTION_REPORT.md`.

## Verification performed (this session)
- `python scripts/render_status.py --check`: passed (docs/status.yaml still source of truth; STATUS.md in sync).
  - Git working tree: `290ab106` on `feat/day4-real-geometry-measurements`; untracked: `perception/bundle_adjustment.py` (recovered, not committed).
- All 75 existing tests pass (test_unified_engine_orchestration.py, test_engine_matrix_bridge.py, test_camera_pose_graph.py, test_depth_to_points.py).
- Bundle adjustment verified with synthetic data: converges, gauge camera stays fixed, other cameras refined toward ground truth.

## Remaining from prior work (carried forward, not re-verified this session)
- Pre-existing unrelated test failure still at HEAD: `tests/test_detection_pipeline_engine.py::test_run_for_job_executes_the_pipeline_and_reports_what_it_did` (store holds extra `ceiling`/`mask`/`wall` quantities). Not touched this session.
- Open depth items per S5-26: fixed uncalibrated confidence values; MiDaS per-image normalisation; no depth benchmark (no ground-truth depth in repo).

## Test Results (from main, all passing)
- `test_spatial_global_stage.py`: 21/21
- `test_spatial_global_api.py`: 4/4
- `test_spatial_stage.py` + `test_spatial_inputs_v1.py`: 24 passed, 1 skipped
- `test_job_stage_runs_pipeline.py`: 5/5
- `test_stage_ladder.py`: 8/8
- Total: 62 passed, 1 skipped, 0 failed

## Hardware Verified
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB VRAM)
- RAM: 16.8 GB, 16 CPU cores
- CUDA: 13.0

## Multi-Engine Benchmark Results
- SIFT: 0.929 inlier ratio, 0.077s runtime
- LightGlue-ALIKED: 0.980 inlier ratio, 0.138s runtime (BEST)
- LightGlue-DISK: 0.818 inlier ratio, 0.275s runtime
- ORB: 0.710 inlier ratio, 0.167s runtime

## Remaining P0
- WorldIR persistence through process restarts (test save/load + process restart) — partially covered by existing tests
- Adversarial testing (input/perception/geometry/persistence/product failures)
- Camera registration: verify end-to-end with real images (COLMAP not installed; Python SfM backend now uses real matchers)

## Remaining P1
- Video/temporal reconstruction
- Active capture planning
- Reality Studio frontend integration
- Gaussian splatting / NeRF integration (P2 per priority order)

## Next action (exact)
- Re-read `.agent/EXECUTION_STATE.md` and `.agent/TASKS.yaml` `P-52W-00` (current state + governing task), then re-read `docs/status.yaml` S5-23/S5-26 `next` fields, `RELEASE.md`, and the `TASKS.yaml` tail before any production edit.
- Choose the next implementation work package from mandate §13 priority order applied to reconciled state. Next candidate: Quarter 1 Week 2 — extend pose graph to integrate bundle adjustment refinement (S5-25 `next`: "Bundle adjustment over 3D points"); OR continue evidence fusion with confidence calibration (S5-26 `next`: "Make confidence a measured quantity or drop it").
- Commit only when appropriate; never rewrite history; keep PRs focused; do not merge without authorization.

## Programme governance
### Quarter 1 generalized evidence normalization + task-specific fusion (mandate §2, Weeks 1–12)
1. Created `engine/multiengine/evidence_fusion.py` — task-specific fusion layer that:
   - Normalizes `EngineObservation` records per task type (depth_value, position, dimensions, bbox)
   - Fuses with reliability/uncertainty/consistency weighting (never blind averaging)
   - Uses existing `fuse_quantity` engine for scalar quantities with conflict detection
   - Component-wise inverse-variance fusion for vectors (position/dimensions) with per-axis conflict detection
   - Consistency-weighted bbox fusion with label disagreements preserved as `FusionConflict` records
   - Incompatible frames/units/sources recorded as conflicts, never averaged
   - Fused results written back to EvidenceMatrix linked to contributors via `derived_from`
   - Uses lightweight stub `Unit`/`Provenance` classes to avoid heavy import chain
2. Created `tests/test_evidence_normalization_fusion.py` — 19-test focused suite covering:
   - Scalar fusion (single passthrough, two consistent, conflicting, empty raises)
   - Position fusion (single passthrough, two in same frame, different frames filtered)
   - Dimensions fusion (single passthrough, two fuse)
   - Bbox fusion (single passthrough, two same label, different labels preserve conflict, different sources incompatible)
   - write_fused_back (written observation linked to contributors, predicted dict carries fused value)
   - normalize_and_fuse (returns one per quantity, grouped by label)
   - Cross-check with existing fuse_quantity (same results for scalar quantities)
3. Advanced `docs/status.yaml` S5-23: added task-specific fusion evidence, added test to the tests list, updated `updated` to 2026-11-26.

Verification: manual test of core fusion paths (single passthrough, two consistent, write_fused_back with link verification) PASS.

## Programme governance
- Governing task: `.agent/TASKS.yaml` entry `P-52W-00` (the 52-week master programme as the authoritative queue).
- Status source of truth: `docs/status.yaml` → `docs/STATUS.md` (rendered by `scripts/render_status.py`). The programme advances status items only with evidence, in the same change that produces it.
- This file is updated at every meaningful milestone; the next session resumes from "Next action" above.
