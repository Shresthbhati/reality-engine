<!--
ARCHIVED 2026-09-15 (P0-01 canonical-state consolidation).
SUPERSEDED: this file is point-in-time history and is NOT current truth.
Canonical files now: .agent/ENGINEERING_CONSTITUTION.md,
.agent/REALITY_ENGINE_MISSION.md, .agent/TASKS.yaml,
.agent/EXECUTION_STATE.md, .agent/CAPABILITIES.yaml, .agent/LICENSES.yaml.
Do not extend or edit this file; read it only for history.
-->

# Reality Engine — Autonomous Execution State

**Updated:** 2026-09-14 (docs-canon session: canonical documentation
architecture + operating-rule additions; PR chain #14/#16/#17/#18/#19 all
merged; `reality compile` CLI landed via PR #19)

## Mission

Turn Reality Engine V11 into a coherent evidence-grounded
reality-to-digital-world compiler and persistent world platform.

## Priority order (authoritative — see docs/implementation/ROADMAP.md)

- P0 mapping spine: **DONE** (SfM → scale → frames → depth → fusion →
  mesh → WorldIR → glTF; PRs #14/#16; one-command `reality compile`,
  PR #19)
- P1 sensor/temporal: IN PROGRESS — sensor sidecar parsing (IMU/GNSS/
  telemetry/calibration) landed on main by parallel session; remaining:
  depth sidecar ingestion (BLOCKED: format decision now recorded as
  D001, 16-bit PNG first → unblocked), time synchronization, VIO,
  cross-source registration
- P2 perception identity: NOT STARTED (specs written)
- P3 uncertainty/provenance: NOT STARTED (specs written)
- P4 WorldStore: NOT STARTED (specs written)
- P5 GIS/robotics/large-world: DEFERRED (specs written)
- P6 Studio beyond viewer: NOT STARTED (spec written)
- P7 physics beyond foundation: NOT STARTED (spec written)
- P8 CI/release: NOT STARTED (verified: no workflows exist)

## Completed (verified)

- 2026-09-14, PR #16: MeshData + deterministic PLY I/O; scipy-based
  preprocessing (voxel downsample, outlier filter, camera-oriented
  PCA normals); COLMAP poisson_mesher backend (auto depth/trim
  adaptation, verified 9k verts from 3k pts, 75k verts from 40k pts
  in 1.6 s); pipeline stage 6 with ArtifactStore; WorldIR MESH
  geometry; glTF real-mesh export (uint32 indices after real-data
  uint16 overflow catch); camera-envelope filter from sparse SfM
  points (cut 43,445 depth-noise points; 18 km garbage extent →
  honest 15.8 m room extent). Real E2E: 240,735 verts / 482,142
  faces. Suite: 1,294 passed.
- 2026-09-14, PR #19: `reality compile ./capture` one-command path;
  shared dataset loader + artifact writer modules; injectable
  reconstruction backend seam (`REALITY_TEST_BACKEND` env, test-only);
  deterministic fake-backend E2E test. Suite: 1,279 passed (SAM env
  failure excluded, pre-existing).
- Parallel sessions (merged main): sensor sidecar parsing
  (IMU/GNSS/telemetry/calibration), mesh-gate fix, suite at 1,339.

## In progress

- docs-canon branch (this session): canonical docs architecture —
  architecture/, implementation/, engineering/, future/ (18 specs),
  CLAUDE.md sections 51–54, this state file, task ledger.

## Next (in priority order)

1. Land docs-canon PR; reconcile stale root-level docs pointers.
2. P1 depth sidecar ingestion (16-bit PNG first, per D001; feed the
   existing DEPTH_FUSION contract).
3. P1 time synchronization (spec written).
4. P1 VIO backend slot (spec written; ORB-SLAM3/VINS subprocess).
5. P1 cross-source registration (spec written; ICP + GNSS anchors).
6. P8 minimal CI (unit + integration + deterministic CLI test; no
   model downloads).

## Known blockers

- Dense MVS (`patch_match_stereo`): CUDA required, machine is
  CPU-only COLMAP 4.2.0. Backend slot documented, honest skip.
- SAM torch-hub cache failure: pre-existing environment breakage
  (1 test); unrelated to current work; perception stage keeps honest
  BACKEND_UNAVAILABLE behavior.
- open3d/trimesh/skimage absent: meshing deliberately scipy-based.

## Active failures

- None open. (SAM env failure tracked above as environment, not code.)

## Last verification

- docs-canon worktree: full suite `python -m pytest` — to be run
  before PR (docs-only changes + CLAUDE.md + .agent files; code
  untouched, but suite run is the gate anyway).
- Prior gates this campaign: see Completed entries (1,294 / 1,279 /
  1,339 passed with the SAM env failure excluded).

## Rules

Never mark work complete without execution evidence.
Never erase incomplete work.
Never replace scientific functionality with placeholders.
