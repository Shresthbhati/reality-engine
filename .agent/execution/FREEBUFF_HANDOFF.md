# Freebuff Handoff — Reality Engine

Written: 2026-09-15, end of the Claude execution campaign batch.
Audience: the next agent (Freebuff + GLM-5.3). Everything below is
verified repository reality, not intention. If a statement here
contradicts the repository, the repository wins — re-inspect, then fix
this file.

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

- **1,613 passed, 1 failed, 1 skipped** at the finalization PR (PR #30,
  head 63b9b7b). The ONE failure was
  `tests/test_sam_backend.py::TestSAMSegmentationBackendIntegration::test_real_model_load_and_inference`.
- **Superseded on `fix/sam-load-path` (2026-09-15): 1,614 passed,
  1 skipped, 0 failed (~206 s) — fully green.** The SAM failure was
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

Derived mechanically from `.agent/TASKS.yaml` (36 tasks: 10 DONE /
15 PARTIAL / 11 MISSING), not from prose:

DONE: P0-01, P0-02, P0-03, P1-01, P1-02, P2-02, P3-01, P3-03,
P5-01, P6-03 (bad-scale check + export-time quality metadata landed
2026-09-15), P18-01.
PARTIAL (implemented, named remainder in ledger `open:`/`verification:`
fields): P1-03, P2-01, P3-02, P4-01, P5-02, P6-01, P6-02,
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

1. **P2-01 remainder (spine head)**: synchronization backends 3-6
   per `docs/future/synchronization/TIME_SYNCHRONIZATION.md` — GNSS/
   PPS, cross-correlation, trajectory correlation, optimization-based
   alignment; each with its own deterministic fixture (the ledger's
   `PENDING` verification items).
2. **P4-01 remainder**: point-to-plane + symmetric ICP and
   RegistrationEngine orchestration (extrinsics -> trajectory prior
   -> ICP -> uncertainty).
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
