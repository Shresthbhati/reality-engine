# Execution State — Reality Engine

## Branch / Commit
- Branch: `feat/day4-real-geometry-measurements` (from `main`, PR #395 MERGED)
- Commit: `4c1c38d2` — "docs(recovery): add BA consolidation analysis (§4.3) and ops stash comparison (§4.1)" (tip); session commits `8aa06d06` (report + ledger final update), `55f78109` (occlusion tests), `59ca5a39` (report + ledger update), `36050335` (occlusion module), `5a51204a` (evidence_fusion.py refactor), `79c33276` (initial recovery report); prior `290ab106`, `4dc8d258`, `c10f9a2c`, `156f3e36`
- Working tree: clean except unrelated untracked files (finance `.claude`/`.codex` agents, `.github/agents/`, `perception/bundle_adjustment.py` recovered-but-untracked, plus KiloCode docs `KILOCODE_200K_ENGINEERING_REPORT.md` + `KILOCODE_DELIVERY_LEDGER.yaml`)
  - Push state: branch pushed to `4c1c38d2`; remote matches local; PR #395 MERGED (per gh; remote branch ref confirms 4c1c38d2 landed)

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
4. Implemented `perception/bundle_adjustment.py` (UNTRACKED — not committed): bundle adjustment over 3D points using exact sparse LM solver; camera 0 pinned as gauge; per-camera/per-point covariances. Status: recovered work product; NOT committed (evaluated this session — see §Comparison verdicts below).
5. Ported stash@{6} upload dedup into `apps/api/routes_misc.py::create_upload` (commit `2e15cd44`): byte-identical re-uploads reuse the existing Evidence row + artifact, return `deduplicated=true` with the existing evidence_id, avoid a duplicate artifact/evidence/processing-job. Test `test_upload_deduplicates_byte_identical_content` passes. This is the one genuinely-improving, non-conflicting piece from stash@{6}; the `_run_reconstruct_session` rewrite was NOT ported (CONFLICTING — current branch's stage-segmented implementation is more complete + tested).
6. Extended `perception/camera_pose_graph.py` with optional `points3d` cheirality constraints — `(R_c @ P + t_c).z > 0` residuals pin handedness and scale; all 30 existing tests pass.
7. Comparison-verdict items resolved this session (all three): (a) upload dedup [PORTED — 2e15cd44 + test passes]; (b) BA consolidation [HANDOFF to Claude — geometry owner; Option A/B/C decision is Claude's, not mine, since I must not create a second canonical geometry pipeline; handoff at docs/audits/BUNDLE_ADJUSTMENT_CLAUDE_HANDOFF.md]; (c) ops stash piecemeal [upload dedup done; remaining piece (heartbeat reclamation) HANDOFF to OpenCode — backend recovery owner; handoff at docs/audits/STASH6_OPENCODE_HANDOFF.md; do NOT apply stash@{6}; do NOT drop it]. Six-agent branch inspection = AUDITING — suspended per new directive (stop auditing, start delivering).
6. Widened pair strategy in `scripts/run_multi_engine_real_capture.py` — added `sequential_skip23`, `sequential_skip234`, `all`; default = `sequential_skip23`.
7. Created 9 engineering ledger files in `docs/engineering/` during a prior session; these were removed in the current recovery session per AGENTS.md §51 (4 were plan/roadmap/backlog/state documents prohibited from recreation; 5 were duplicates of canonical committed `docs/engineering/` records). Their unique content (bundle adjustment decision D-011, cheirality decision D-012, South Building 32-photo multi-engine execution results) is captured in the recovery report `docs/audits/FREEBUFF_200K_RECONSTRUCTION_REPORT.md`.

## Comparison verdicts (performed 2026-11-26)

### BA consolidation (perception/bundle_adjustment.py vs reconstruction/sfm/core_pipeline.py::BundleAdjustmentOptimizer)
- Finding: NOT equivalent implementations. Untracked file is the correct, rigorous sparse-LM bundle adjustment (6-DOF cameras + 3-DOF points, analytic sparse Jacobian, gauge camera 0, Cauchy loss, per-camera/per-point covariances, full BundleAdjustmentResult). Committed BundleAdjustmentOptimizer is a weaker coordinate-descent point refiner (point positions only, rotations fixed, heuristic backprojection-ray Jacobian, no gauge, no covariances, Huber weighting post-hoc not in normal equations) mislabeled as BA.
- Decision: do NOT commit as-is (untested, not integrated). Do NOT leave competing implementations. Recommended: Option A (integrate rigorous BA into canonical pipeline with tests + adapter from core_pipeline data structures to CameraParams/CameraObservation/Point3D), Option B (document distinction, track wiring task), or Option C (archive with explanation). Blocked by: no tests, no integration, no adapter.
- Full analysis: `docs/audits/FREEBUFF_200K_RECONSTRUCTION_REPORT.md` §4.3; ledger: `docs/engineering/FREEBUFF_DELIVERY_LEDGER.yaml` section `Comparison verdicts (this session)`.

### Ops stash (stash@{6} vs current branch) — evaluated 2026-11-26
- Verdict: CONFLICTING, not complementary. `_run_reconstruct_session` is a different architecture (stash: COLMAP backend + ReconstructionOrchestrator + reality.compile_world_from_reconstruction + WorldStore.save_version direct; current: stage-segmented vertical slice via run_progressive + vertical_slice + worldstore_service.commit_version + manifest grading). Same function signature, different body = direct conflict. Current branch's implementation is more complete and tested.
- routes_worlds new routes (worldir/cameras/points/diff): SUPERSEDED — current already has these + full CAS-validated commit endpoint with version integrity, duplicate detection, four-way commit storm tests, space-graph.
- upload dedup (routes_misc.py create_upload): IMPROVEMENT — current create_upload has no dedup; stash's checksum-based dedup is a genuine enhancement; touches same function; port selectively.
- verify_artifact in storage.py: CONFLICTS — current storage.py already re-exports verify_artifact from engine.pipeline.stages.content_store.
- graceful shutdown (main.py, 30s): SUBSET — current has 10s + shield + db dispose timeout (more defensive).
- reclaim_abandoned_jobs (heartbeat-based): ALTERNATIVE — current uses proc-based recover_orphaned_jobs at startup; blocked by _run_reconstruct_session conflict from applying cleanly.
- cli/main.py incremental update: INDEPENDENT — separate CLI concern; evaluate separately.
- .obsidian/*: IGNORE — editor state.
- test_application_api.py expansion: OBSOLETE — current test suite covers the chain.
- Decision: do NOT apply stash@{6} to this branch. Do NOT drop it. Port piecemeal if valuable (upload dedup, heartbeat reclamation as alternative to proc-based recovery). Do NOT apply the _run_reconstruct_session rewrite.
- Full analysis: `docs/audits/FREEBUFF_200K_RECONSTRUCTION_REPORT.md` §4.1; ledger: `docs/engineering/FREEBUFF_DELIVERY_LEDGER.yaml` section `Comparison verdicts (this session)`.

## KiloCode session 2026-11-26 (independent production engineering + verification)
### Completed
1. Independent verification of `engine/multiengine/evidence_fusion.py`: found and fixed 4 real defects (duplicate `Provenance`/`Unit` stub class type-incompatible with `reconstruction/fusion/fusion.py`; `write_fused_back` linked before storing; `_fuse_bboxes` dropped cross-source bboxes instead of recording incompatibility; `_fuse_scalar_quantity` passed `None` confidence into `FusableObservation`). All 19 fusion tests now pass (8 were failing pre-fix).
2. Authored object-level occlusion reasoning + object permanence module `perception/spatial_intelligence/occlusion.py` (407 lines): `OcclusionGraph`, `OcclusionRelation`, `compute_object_occlusions`, `merge_graphs`, `transitive_occlusion_chain`, `occluded_region_of`, `object_permanence_in_view`. All 13 tests in `tests/test_spatial_occlusion.py` pass.
3. Authored `tests/test_spatial_occlusion.py` (262 lines): occlusion graph construction, transitive chains, occluded regions, object permanence.
4. Created KiloCode delivery documents per Workstream 12:
   - `docs/audits/KILOCODE_200K_ENGINEERING_REPORT.md`
   - `docs/engineering/KILOCODE_DELIVERY_LEDGER.yaml`
5. Updated `.agent/EXECUTION_STATE.md` and `.agent/TASKS.yaml` (this session).

### Verification performed (KiloCode session)
- `tests/test_evidence_normalization_fusion.py`: 19 passed (8 previously failing, now fixed).
- `tests/test_spatial_occlusion.py`: 13 passed (verified this session via background process mode).
- Focused regression suite (pose graph, depth backend, matrix bridge, orchestration, matrix executor, spatial reasoning, scene graph, depth consistency): all pass.
- Combined focused suite: 138 passed, 0 failed.
- No new external dependencies added; existing Python/numpy/scipy/pytest stack only.

### KiloCode role
Independent production engineering contributor, spatial-intelligence author, new-capability implementer, cross-system optimizer, and independent verification authority — defects found during verification were repaired, not merely reported.


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
- Resume from `.agent/TASKS.yaml` `P-52W-00` and this file's KiloCode session section.
- Next implementation candidates: evaluate/commit `perception/bundle_adjustment.py` (untracked recovered work; evaluation done — decision deferred to Option A/B/C with tests); extend pose graph to integrate bundle adjustment refinement (S5-25 next); continue evidence fusion confidence calibration (S5-26 next: "Make confidence a measured quantity or drop it").
- Two comparisons complete this session: BA consolidation (§4.3 of report) — untracked file is superior algorithm but untested; ops stash (§4.1 of report) — CONFLICTING, not applied, do not drop.
- Commit only when appropriate; never rewrite history; keep PRs focused; do not merge without authorization.
- Coordinate with Cline/OpenCode on shared contracts before changing them (no agent channel; manual).

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

## 1M backend wave-1 (2026-11-26)
- Branch re1m/feature-flags PR#402; re1m/event-persistence PR#403; re1m/api-keys PR#404; re1m/delivery-ledger-1m PR#405.
- Verified: feature_flags 12; event_store 10 + event regression 107/1; api_keys 9. Not merged. 1M LOC target NOT met; no filler.

## KiloCode session — continued (Phase 3 golden E2E + Phase 5 benchmark + verification)
### Completed
1. Authored golden vertical-slice end-to-end acceptance test `tests/integration/test_golden_e2e_workflow.py` (5 tests): real images -> EvidenceItems -> vertical_slice pipeline (deterministic test backend) -> WorldIR -> WorldStore persistence (save -> read-back verify -> load -> hash integrity) -> WorldDiff (self-diff empty, cross-version diff detects changes) -> export (glTF + PLY) -> validation -> rejected unsupported export format -> degraded optional stage recorded honestly. All 5 tests pass.
2. Authored persistence integrity benchmark `benchmarks/persistence_integrity_bench.py`: save->verify->load->hash integrity (25 iters), version-chain lineage (40 versions), concurrent save/load safety (8 writers x 10 versions = 80 versions, no corruption). `all_passed: True`.
3. Independent verification (regression): ran focused suites across the areas touched:
   - tests/test_evidence_world_persistence.py, test_failure_modes_worldstore.py, test_evidence_fusion.py, test_fusion_pipeline.py, test_evidence_normalization_fusion.py: 57 passed, 0 failed.
   - tests/test_spatial_reasoning.py, test_job_stage_runs_pipeline.py, test_detail_worldstore.py: 11 passed, 0 failed.
   - Combined focused regression suites: 206 passed, 0 failed.
   - Pre-existing unrelated failure `tests/integration/test_acceptance_p018.py::test_relative_run_real_photos` (needs real photos + COLMAP; environment-dependent) — untouched, not caused by this session.

### New files this session (untracked, ready to commit/PR)
- `tests/integration/test_golden_e2e_workflow.py`
- `benchmarks/persistence_integrity_bench.py`

### No new external dependencies; existing Python/numpy/scipy/pytest/PIL/scipy stack only.
