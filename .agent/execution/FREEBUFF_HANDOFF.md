# Freebuff Handoff — Reality Engine

Written: 2026-09-15, end of the Claude execution campaign batch.
Audience: the next agent (Freebuff + GLM-5.3). Everything below is
verified repository reality, not intention. If a statement here
contradicts the repository, the repository wins — re-inspect, then fix
this file.

## UPDATE 2026-09-16f — P7-05 detail budget consumer (newest)

- `perception/quality/detail_budget.py`: first consumer of the P7-04
  measured evidence-quality report — documented GSD→level mapping
  (L0–L4 multi-scale), multi-view coverage cap, compute tiers
  none→full for adaptive allocation. Refuses assessor-bypass reports;
  empty/coverage-failed scenes → honest zero-allocation unsupported
  budget. 10 tests red-first (tests/test_detail_budget.py).
- WIRED: `recommend_capture` returns a dict with machine-readable
  `scene_budget` (DetailBudget) + prose keys (wiring tests in
  tests/test_capture_feedback_wiring.py).
- Ledger P7-05 added (PARTIAL: scene-level only; per-region spatial
  budgets + ROI consumption of compute_tier open).

## UPDATE 2026-09-16e — suite-stall fixed + SAM gate verified

- The full suite now COMPLETES unbounded: **1828 passed / 1 skipped /
  0 failed in ~177s** — including the SAM real-model integration test
  (95.6s), which previous passes deselected as environment-gated; it
  now runs and passes. No test was weakened or removed.
- Root cause of the old stall (profiled): `fit_cylinder` re-ran
  pure-Python per-point circle math inside ~8k golden-section
  evaluations per segment. Fixed in `perception/architecture/
  parametric.py` (float64-numpy Kasa fit + hoisted axis-invariant
  work + manual cross products): 9.6s → 0.90s per benchmark run,
  byte-identical deterministic report. Regression guard:
  `test_report_runtime_is_bounded` (5s budget).
- PENDING_IMPLEMENTATION.md P2.15 reconciled (temporal tracking was
  still "MISSING" there after P7-02 landed) → PARTIAL with evidence.
- Deselection convention is now obsolete: run the plain suite.

## UPDATE 2026-09-16d — P7-04 universal evidence-quality assessment

Directive: universal perception + maximum-fidelity reconstruction —
the engine must work on ANY scene (objects/vehicles/machines/forts/
streets), not be defined by Victoria Memorial. Foundation increment
landed, TDD red-first: `perception/quality/assessment.py` (18 tests
in `tests/test_evidence_quality.py`, full suite 1826 passed /
1 skipped / 0 failed):

- `assess_evidence_quality(ReconstructionResult, cameras) ->
  EvidenceQualityReport`: measured per-point GSD (euclidean
  camera-to-surface distance x 1000 / max(fx,fy), median across
  observations), observation by projection inside image bounds (NOT
  trusting `source_evidence_ids`; overclaims counted + reported),
  documented detail-tier thresholds, per-point view counts,
  unprojectable points reported not dropped.
- `recommend_capture(report)`: measured-fact-derived capture
  recommendations (close-range / overlapping views / coverage gaps)
  — the closed loop for the Capture app.
- Honesty: no observations -> GSD None + "unsupported"; nonempty
  points + no cameras refuses (ValueError) rather than guessing
  intrinsics. Domain-agnostic by construction.
- Ledger P7-04 added (PARTIAL: remaining section-11 inputs — view
  angle, focus/blur, texture richness, depth confidence, occlusion
  beyond projection bounds — no detail-budget/adaptive-compute
  consumer yet, thresholds untuned); CAPABILITIES.yaml gained
  `evidence_quality_assessment`.
- Next universal-perception increments on the spine: the section-11
  detail-budget consumer (adaptive compute), generic repeated-
  pattern / thin-structure / small-object perception (directive
  sections 19–23), and the ROI system (section 9).

## UPDATE 2026-09-16c — P7-02 temporal tracking

- `perception/tracking/temporal.py`: 3D observation-level identity
  across time (TrackRecords with measured path/duration/implied speed;
  speed+gap gated continuation; untimed observations returned
  separately; confidence = min of measured member confidences, refusal
  to fabricate). 14 tests; no regressions in the identity/lifting
  neighbors (48 passed).
- PR #36 (P7-03) was merged; this branch `claude/p7-02-temporal-tracking`
  carries the P7-02 work on top of updated main.
- Next natural priorities: 2D ByteTrack-style detection association
  (spec TRACKING.md), detector-backed semantic fusion into the
  architecture registry (P7-03 open half), E-C alternative via WSL
  (ROS1 Noetic) for the real VIO run, P8-01/P8-02 room/building graphs.

## UPDATE 2026-09-16b — architectural perception expansion (newest)

- P7-03 expansion landed: parametric perception (cylinder/sphere/
  circle fits, measured residuals, honest FitRefused), deterministic
  voxel segmentation, extensible class registry, component
  observations + confidence tiers + multi-view entity resolution,
  repetition/symmetry priors (support-only), WorldIR +DOME/+ARCH
  (additive), promotion + adjacency graph, benchmark harness with an
  honest capture gate. 58 new tests; full gate 1,795/1/0.
- **Victoria Memorial benchmark record exists and is CAPTURE_PENDING**:
  the real photo capture is a human/external dependency. The harness
  refuses to run it until then — do not simulate a capture.
- Two real bugs the fixtures caught, both fixed: analytic cubic
  eigen solver -> Jacobi (negative eigenvalues on near-degenerate
  shells); plane-subtraction-before-segmentation shredded curved
  structures (segment first, classify after).
- Next natural priorities: detector-backed semantic fusion into
  component hypotheses (P7-03 remainder), stairs (Phase 1), P8-01
  openings/relationships topology, P9-01 remainder, real capture
  planning for Victoria Memorial.

## UPDATE 2026-09-16 (read this first — newer than everything below)

- PRs #30–#34 are MERGED; main carries all campaign work through the
  P4/P5/P11/P12/P13 batch (registration engine + covariance, backend
  selection policy, fused.ply parser, fusion consumer + WorldIR
  write-back, provenance graph, dense ingest, WorldStore,
  incremental compilation, measurement uncertainty).
- New this session: **time-sync consumers wired**
  (`trajectories/backend/sync.py` + `estimate_trajectory(clock_model=...)`);
  canonical Trajectory now carries clock_id/sync_state/sync_method and
  preserves original sensor stamps. Units are EXPLICIT: stream-fitted
  ClockModels are in seconds (`model_unit="s"`), trajectories in ns —
  the e2e test caught the 10^9 corruption bug before it could ship.
  16 tests in `tests/test_sync_consumers.py`.
- Gate on the final head: 1,737 passed / 1 skipped / 0 failed
  (SAM real-model included, passing).
- **Next priority: E-C — real VIO backend run** (P3-02 remainder):
  Docker Desktop daemon was DOWN all session (cold boot after restart;
  launch attempted, npipe never came up). Planned route: ROS1 Noetic
  container, OpenVINS `run_subscribe_msckf` (upstream's real-image
  path — `run_simulation` is synthetic-only, verified from source),
  EuRoC MH_01 bag → TUM → existing parser → canonical Trajectory →
  ATE. The C++ toolchain (cmake 4.2.1, VS) is present as a fallback.
- Canonical queue/status: `.agent/TASKS.yaml` +
  `.agent/EXECUTION_STATE.md` (dated entries at top).

## Canonical state (single source of truth — do not duplicate)

- Queue: `.agent/TASKS.yaml` (36 tasks; statuses evidence-backed).
- Execution log: `.agent/EXECUTION_STATE.md` (what ran, what passed,
  what is NOT claimed).
- Capability registry: `.agent/CAPABILITIES.yaml` (21 entries,
  backend/limitation/fallback each).
- Governing docs: `CLAUDE.md` (constitution),
  `.agent/REALITY_ENGINE_MISSION.md`,
  `.agent/ENGINEERING_CONSTITUTION.md`.
- This directory (`.agent/execution/`) is a handoff pointer, not a
  competing roadmap. Do not fork the queue.

## Current repository state

- Campaign branch: `claude/reality-engine-architecture-ed811e`
  (worktree `.claude/worktrees/reality-engine-architecture-ed811e`).
  It already contains merged PRs through #29 (calibration, sensor
  model, clocks, canonical state, RGB-D pipeline).
- On top of that base, the finalization batch was committed on this
  branch (see "Last Claude PR" below): trajectories/
  (P3-01/02/03), registration/ (P4-01 partial), backend registry
  (P5-01/02), real GPU dense-MVS CLI pass (P6-01, evidence in
  EXECUTION_STATE), multi-source fusion (P6-02), mesh validation
  (P6-03), appearance+epipolar identity (P7-01), architectural
  perception (P7-03), WorldIR  statement states (P9-01).
- `datasets/*` are gitignored (rendered fixtures, machine-specific); (rendered fixtures, machine-specific);
  never commit them.

## Test status (exact, observed)

## Last Claude PR (finalization batch)

- PR: https://github.com/Shresthbhati/reality-engine/pull/30
- Branch: `claude/reality-engine-architecture-ed811e`
- Head commit: `4e188b5` — 38 files, +4,259/−77; base = merged PR #29
  (main `f86cb2d`). This commit is the intended final state of the
  batch; nothing is left uncommitted on the branch (only gitignored
  junk dirs: `MagicMock/`, `.serena/`, `.claude/`).

## Test status (exact, observed)

- **Latest (2026-09-15, P2-01 pass on `claude/p2-01-sync-backends`):
  1,648 passed, 1 skipped, 0 failed (~122 s).**
- Historical: 1,613 passed, 1 failed, 1 skipped at the finalization PR
  (PR #30, head 63b9b7b); the ONE failure was
  `tests/test_sam_backend.py::TestSAMSegmentationBackendIntegration::test_real_model_load_and_inference`.
- **Fixed on `fix/sam-load-path` (merged as PR #31): 1,614 passed,
  1 skipped, 0 failed (~206 s) — first fully green gate.** The SAM failure was
  diagnosed as two real code bugs, not the environment: (1) the backend
  used torch.hub.load on a repo that has NO hubconf.py (verified
  upstream, 0 commits touching it) — replaced with the segment-anything
  pip package + checkpoint-file resolution; (2) SAM's predicted_iou can
  exceed 1.0 (observed 1.0005), which Uncertainty rejected and the
  per-item except silently swallowed, discarding valid regions — clamped
  at the conversion boundary. vit_b checkpoint is pre-cached in
  ~/.cache/torch/hub/checkpoints/.

## Known failures

- **None in the suite.** The SAM real-model failure is FIXED on
  `fix/sam-load-path` (root causes were code, not environment — see
  Test status). Do not reintroduce torch.hub.load for SAM.
- Historical note for audits: earlier runs showed either
  `1 failed` (no filter) or `20 deselected` (with deselect filter) —
  both were the same SAM load-path bug, now fixed.

## Completed priorities (exact IDs — do NOT redo)

Derived mechanically from `.agent/TASKS.yaml` (36 tasks: 12 DONE /
13 PARTIAL / 11 MISSING after the P2-01 pass), not from prose:

DONE: P0-01, P0-02, P0-03, P1-01, P1-02, P2-01 (all six sync methods
landed 2026-09-15), P2-02, P3-01, P3-03, P5-01, P6-03 (bad-scale
check + export-time quality metadata), P18-01.
PARTIAL (implemented, named remainder in ledger `open:`/`verification:`
fields): P1-03, P3-02, P4-01, P5-02, P6-01, P6-02,
P7-01, P7-03, P8-01, P9-01, P10-01, P16-01, P17-01.
MISSING (not started): P7-02, P8-02, P8-03, P10-02, P11-01, P12-01,
P13-01, P13-02, P14-01, P15-01, P19-01.
Hardware/availability blockers inside PARTIAL: P1-03 (real-device
RGB-D capture — hardware), P3-02 real backend runs (ORB-SLAM3/
OpenVINS/Basalt binaries not installed; TUM-format subprocess
adapters are landed and honest about absence).

## Next priority (strict order — Priority Authority, spine order)

The canonical spine is TIME SYNC -> TRAJECTORY/VIO -> REGISTRATION ->
DENSE MVS/FUSION -> MULTI-VIEW IDENTITY -> UNCERTAINTY/PROVENANCE ->
WORLDSTORE -> INCREMENTAL COMPILATION. TRAJECTORY/VIO is landed
(model/diagnostics DONE, adapters PARTIAL on binary availability), so
the spine head is TIME SYNCHRONIZATION:

1. ~~P2-01 remainder (spine head)~~ **DONE** (2026-09-15): all six
   spec methods (metadata, GNSS/PPS, trigger, signal correlation,
   optimization) with deterministic known-answer fixtures; 56 clocks
   tests; full suite 1,648 green. Downstream consumers wire the seam
   in their own tasks.
2. ~~P4-01 remainder~~ **DONE** (2026-09-15 five-execution batch:
   point-to-plane ICP, RegistrationEngine, ResidualStats; covariance
   propagation and the landmark method remain PENDING in the ledger).
   Spine head now: P3-02 real VIO backend runs (binary availability)
   and P6-01's gated real-MVS integration; next implementable:
   P7-02 temporal tracking or P8-01 room graph.
3. **P6-01/P6-02 remainders**: parse verified dense output (fused.ply)
   into canonical types; pipeline consumer calling
   fuse_depth_observations -> WorldIR geometry write-back.
4. **P10-01 remainder**: queryable provenance graph +
   `reality provenance` query (next untouched spine item after the
   above).

Do NOT jump to disaster-management work or random Studio polish. Read
each task's `spec:` pointer and `open:` field in TASKS.yaml before
starting.

## Strategic objective

Reality Compiler + WorldOS: capture -> evidence -> geometry ->
WorldIR 2.0 -> compiled, simulated, exportable world — with
provenance and honest uncertainty on every statement.

## Execution rules for the next agent

inspect -> implement -> execute -> test -> debug -> verify ->
update `.agent/EXECUTION_STATE.md` + `.agent/TASKS.yaml` honestly ->
continue to the next priority without waiting for permission.
- Verify claims against executable code, never documents or old
  statuses (statuses here were stale once already — P7-03 said
  MISSING while the code existed).
- Never fabricate: no placeholder backends, no invented metrics, no
  synthetic evidence labeled real. An unavailable backend/asset is
  recorded as unavailable.
- Merge discipline: verify the branch head is the intended final
  commit BEFORE merging, and verify from main AFTER merging that
  ledger flips actually carried (a merge race lost work once —
  see EXECUTION_STATE history).
- Full-suite gate before every PR; report exact numbers including
  failures; never claim "all tests pass" while the SAM failure
  persists.
- Keep scope tight; smallest correct change; no silent redesign.

## Capability usage

Use what the environment actually exposes; discover before loading:
- Skills/agents/plugins/MCP tools relevant to the task (do not load
  everything blindly).
- The `shres` skills/agents repository where exposed as a capability
  source.
- Terminal + repository tools aggressively; token/context-saving
  capabilities when available; Ponytail if exposed.
- Specialist delegation (graphics, geometry, testing) where the
  environment provides it; integration responsibility stays with the
  principal agent.

## Do-not-duplicate list

Everything in "Completed priorities" has landed code + tests + ledger
evidence. Do not reimplement from stale documents. Source of truth:
source code, current tests, current TASKS.yaml, current
EXECUTION_STATE.md, git log, the PR diff.
