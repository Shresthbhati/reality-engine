# Cline execution plan: finish integrating the technology ecosystem into Reality Engine

Read this whole file first. It is self-contained. Work the tasks **in order**; each has a goal, the files involved,
concrete steps, and an acceptance test that must pass before you move on. Record every task's real outcome in
`docs/research/IMPLEMENTATION_LOG.md` (append a section; never rewrite history there).

## 0. Ground rules (non-negotiable)

1. **Do not commit, push, reset, clean, stash or delete** anything. Report final `git status` at every stopping point.
2. **Never claim what you did not run.** Quote the exact command and its real result. If a check cannot run, write
   `NOT VERIFIED` and why. Do not edit a test to make it pass unless the test encodes a defect, and then say so.
3. **Stop and ask the user** (do not choose) when: a task says DECISION REQUIRED; a change needs a paid service,
   a large download (>500 MB), model weights, or installing something that edits agent/IDE configuration (this
   includes `npx rea-agents setup`); a licence is unclear for something that would be distributed.
4. Keep diffs small and in the existing style. Prefer extending existing modules. No new dependency unless a task
   says so. Optional heavy dependencies must fail with a clear message, never a stack trace.
5. Respect `CLAUDE.md` section 51/52: the roadmap in `docs/implementation/ROADMAP.md` and the registry in
   `docs/implementation/PENDING_IMPLEMENTATION.md` are authoritative; update a status only with evidence.

## 1. Environment (verified on this machine)

| Item | Fact |
|---|---|
| Repo (git worktree) | `C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.claude\worktrees\reality-engine-scene-identity-586ed8`, branch `claude/reality-engine-capability-fusion-94924d` |
| Python / Node | Python 3.14.7, Node 26.8.1, Windows 11, Git Bash + PowerShell |
| GPU | RTX 4050 6 GB, but **PyTorch is a CPU build** (`2.14.0+cpu`); COLMAP 4.2.0 is CPU-only (no `patch_match_stereo`) |
| Not installed | GTSAM (no cp314 wheel, verified), Open3D, Rust/cargo (so `cjval` cannot be built), IfcOpenShell (status unchecked) |
| Research clones | `C:\Users\shres\codes-research\` (outside OneDrive); shas in `MANIFEST.csv`. Shallow clones; do not copy code from them without a licence review. |
| Datasets | NOT in this worktree. Real photos live in the main checkout: `C:\Users\shres\OneDrive\Desktop\codes\reality-engine\datasets\room_capture\images\IMG_*.jpg` |
| Frontend dev server | Turbopack crashes in this worktree (symlinked `frontend/node_modules`). Use the webpack config: `.claude/launch.json` -> `reality-studio-webpack` (port 3011). API config `reality-api` writes into `./data/verify_*`; prefer your own temp dirs via env `DATABASE_URL`, `STORAGE_ROOT`, `WORLDSTORE_ROOT`, `CORS_ORIGINS=http://localhost:3011`. |
| Test runner | `python -m pytest <files> -q --no-header -p no:cacheprovider`. Full suite takes ~26 min; run it in the background and only at milestones. Frontend: `cd frontend && npm run test:contract`, `npx tsc --noEmit -p .`, `npx eslint <paths>`. |
| Shell gotcha | Scripts that `cd` leave the working directory changed; use absolute paths. |

## 2. What already exists (do not redo; verify before extending)

| Capability | Where | Evidence |
|---|---|---|
| 3D Tiles 1.1 export (box content, 1 LOD, optional ENU->ECEF root) | `exporters/tiles3d/exporter.py`, registered in `exporters/profiles.py`; CLI `export --format 3dtiles` | `tests/test_tiles3d_profile.py` (validates `tileset.json` against the official schema in `codes-research/3d-tiles`) |
| Export of procedural worlds (was 0 of 35 entities in 11 of 14 exporters) | `exporters/boxes.py::with_explicit_placement`, applied in `sdk/reality.py` | `tests/test_procedural_export.py` |
| CityJSON `MultiSurface` nesting fix | `exporters/cityjson/exporter.py` | `tests/test_cityjson_exporter.py` (official 2.0.2 schema oracle) |
| SAM diagnostics + no silent download | `perception/segmentation/sam_backend.py` | `tests/test_sam_backend.py` (23 pass; one test runs the real vit_b model, ~90 s) |
| Robust SE(3) pose graph (scipy backend), likelihood-ratio outlier test, ambiguity reporting | `registration/pose_graph.py`, CLI `pose-graph` | `tests/test_pose_graph.py` |
| Pose graph wired into cross-session registration + WorldIR record | `registration/cross_session.py` (`refine_session_chain`, `record_cross_session_registration`) | `tests/test_cross_session_pose_graph.py` |
| API: `POST /api/registration/pose-graph`, `/session-alignments` | `apps/api/routes_registration.py` | `tests/test_registration_api.py` |
| Evidence-backed change proposals (assess, accept as NEW version, override, reject) | `worldstore/proposals.py`, `apps/api/routes_proposals.py` | `tests/test_world_proposals.py`, `tests/test_proposals_api.py` |
| Studio UI for proposals | `frontend/src/components/proposals/ProposalsPanel.tsx`, route `frontend/src/app/(desktop)/worlds/[id]/proposals/page.tsx`, `lib/api/proposals.ts`, `lib/proposals/proposal-model.ts` | contract tests `frontend/tests/contract/proposal-model.test.mts`; manually driven in a browser |
| Research docs | `docs/research/` (`capability-fusion-matrix.csv`, `REPOSITORY_INVENTORY.md`, `INTEGRATION_DECISIONS.md`, `KNOWN_LIMITATIONS.md`, `IMPLEMENTATION_LOG.md`) | read them first |

Baseline test result before any of this work: **6 failed / 3159 passed**. Fixed: 2 colmap-merge tests, 1 packaging test.
**Still open: 2 room-geometry tests (Task 1) and 1 flaky concurrency test (Task 2).**

## 3. Tasks

### Task 1 - Resolve the 2 failing room-geometry tests  (DECISION REQUIRED, investigate first)
- Failing: `tests/test_geometry_artifacts.py::TestPromoteRoomsWritesRealGeometry::{test_compiled_room_geometry_carries_a_resolvable_data_uri, test_omitting_artifact_store_leaves_room_geometry_ids_empty}`.
- Symptom: compiled ROOM entities now carry an INFERRED BOX geometry `geom-room-plane-NNN` with no `data_uri`; the tests expect a
  boundary point cloud (and `[]` when no artifact store). Probable cause: commit `489272b` ("true room footprint").
- Steps: `git log -p --follow -- engine/compiler/world_compiler.py evidence/promote_rooms.py | head`, read
  `docs/implementation/SCENE_IDENTITY.md` and `docs/DECISIONS.md`, find which contract is documented. Then write a
  **short options note** (what each side breaks, who consumes `geometry_ids` of rooms: exporters, validation, viewer).
- **Do not change code or tests until the user picks.** After the decision, implement it and make both tests plus
  `tests/test_interior_compiler.py tests/test_room_inference.py tests/test_world_physics_compiler.py` pass.
- Done when: both tests green, the note is in the log, no other test regressed.

### Task 2 - Root-cause the flaky WorldStore concurrency test
- `tests/test_world_store_concurrency.py::TestConcurrentThreadWriters::test_no_lost_writes_under_thread_contention`
  failed once in the 26-minute full run and passed alone and in two later sweeps.
- Steps: run it 30x in a loop under CPU load (run another pytest in parallel). If it reproduces, read
  `worldstore/store.py::_SequenceLock` and `save_version`; the sequence read-modify-write is the suspect.
- Done when: either a deterministic reproduction + fix + the test passing 30/30 under load, or a written conclusion
  "not reproducible in N runs under conditions X" (then leave it and say so; do not claim it fixed).

### Task 3 - Playwright e2e for the proposals page
- Existing e2e setup: `frontend/playwright.config.ts`, `frontend/e2e/`. Read `e2e/world.spec.ts` for how they start the
  stack. Write `frontend/e2e/proposals.spec.ts` covering: create a proposal from pasted WorldIR JSON, conflict tick ->
  reason required -> accept, reject with reason, stale 409 text.
- Seed data: reuse the pattern in the throwaway script described in `IMPLEMENTATION_LOG.md` "Batch F" (world + base
  version via `worldstore_service.commit_version`; the helper builders are in `tests/test_world_proposals.py`).
- Done when: `npx playwright test e2e/proposals.spec.ts` passes, and you ran it twice. Capture a screenshot artifact
  if the browser can draw; otherwise say it could not.

### Task 4 - Persist and apply cross-session registration
- Gap: `record_cross_session_registration(world, report)` only writes `world.metadata["registration"]["cross_session"]`;
  no API route calls it, and nothing applies the transforms to geometry.
- Steps: add `POST /api/worlds/{id}/registration/cross-session` that takes the same body as
  `/api/registration/session-alignments`, runs `refine_session_chain`, records the result on a **candidate** world and
  returns a change **proposal** (reuse `propose_update`) - never mutate HEAD directly.
- Done when: an API test (pattern: `tests/test_proposals_api.py`) shows alignments -> proposal -> accept -> new version
  whose `metadata.registration.cross_session.provenance == "registration:pose_graph"`.

### Task 5 - Harden the ICP acceptance gate (measured defect)
- Measured: `registration/cross_session.py::align_session` ACCEPTED a C->A alignment that was 4.29 m wrong
  (rmse 0.55 m, inlier fraction 1.0) on a ~10 m scene while A/B and B/C were exact
  (`tests/test_cross_session_pose_graph.py::test_all_pairs_refine_runs_real_icp_end_to_end` documents it).
- Steps: find the acceptance gates in `register_icp` / `align_session` (`registration/registration.py`); add a
  **scale-relative** rmse gate (rmse as a fraction of the cloud's bounding-box diagonal) and, if cheap, a
  symmetric (reverse-direction) consistency check. Keep every existing cross-session test green.
- Done when: the C->A alignment in that fixture is **refused with a reason** (or corrected), the e2e test above is
  updated to assert the new truthful outcome (all pairs exact -> `refinement.status == "refined"`), and
  `tests/test_cross_session_registration.py tests/test_colmap_coverage.py tests/test_cross_session_pose_graph.py` pass.
  Report the threshold you chose and the data that justifies it.

### Task 6 - 3D Tiles beyond boxes
- Today: box content, single LOD. Required next steps, in order: (1) **resolve the up-axis inconsistency**:
  `exporters/boxes.py` says WorldIR is Z-up, `exporters/godot/profile.py` says Y-up - find what the procedural grammar
  and the viewer really use (`procedural/*.py`, `frontend/src/lib/viewport/three-scene.ts`) and document it in
  `docs/research/KNOWN_LIMITATIONS.md`; DECISION REQUIRED if the two profiles must change; (2) export MESH and
  POINTCLOUD geometry through the existing artifact store (`world_ir/geometry_data.py`, see `exporters/gltf/exporter.py`
  `_real_triangles_mesh`) instead of boxes; (3) a geometric-error ladder (parent tiles = merged coarser content);
  (4) a reproducible viewer demo: a static HTML page under `docs/research/demo/` loading CesiumJS **from an exact
  pinned CDN version** that opens the exported folder (no secrets, no Cesium ion token required: use a
  `Cesium3DTileset.fromUrl` with a local server and `EllipsoidTerrainProvider`).
- Done when: schema test still passes, new tests cover mesh content and the LOD ladder, and the demo loads in the
  browser pane (report what you saw; if CesiumJS cannot render offline, say NOT VERIFIED).

### Task 7 - Interoperability oracles for the remaining exporters
- CityJSON: **DECISION REQUIRED** - exporter declares version "1.1", official schemas in
  `codes-research/cityjson-specs` are 2.0.2. Options: bump to 2.0 (schema-clean) vs keep 1.1 and validate against the
  1.1 schema from the specs repo history (`git -C codes-research/cityjson-specs log --oneline -- schemas`). Ask.
- CityGML / 3DCityDB: find the official XSDs (do not guess URLs; check the licence), validate with `lxml` if installed,
  else skip with a message. IFC: if `python -c "import ifcopenshell"` works, validate the produced IFC by reopening it
  and counting entities; if not, say NOT VERIFIED and keep the optional-import behaviour.
- Done when: each format has either an oracle test or an explicit recorded reason it cannot.

### Task 8 - Perception providers
- Goal: one provider interface returning depth (with scale assumptions), masks, labels, confidence, provenance,
  runtime. Start by reading `perception/depth/`, `perception/segmentation/interface.py`, `perception/availability.py`,
  `perception/model_registry.py`.
- Depth Anything V2 vs MiDaS: **DECISION REQUIRED before downloading weights** (size, licence: check
  `codes-research/Depth-Anything-V2/LICENSE` and the per-checkpoint licences; the small model is Apache-2.0, larger
  ones are CC-BY-NC). Benchmark on `datasets/room_capture` (main checkout) with the SfM sparse-fit scale alignment the
  repo already uses; report AbsRel/RMSE against the COLMAP sparse points, runtime on CPU, and label relative depth as
  relative unless scale-aligned. **Do not install a CUDA torch build without asking** (it changes the environment).
- Done when: a provider-agnostic wrapper exists with tests that run without weights (stubs) plus one marked real-model test.

### Task 9 - Open3D (only if installable) and the COLMAP baseline
- First check: `python -m pip download open3d --no-deps -d <tmp>`. If there is no cp314 wheel, **stop this task** and
  record it. If installable: add an optional provider for voxel downsample, statistical outlier removal, point-to-plane
  ICP and Poisson meshing behind the existing registration/meshing interfaces (`registration/registration.py`,
  `reconstruction/meshing/`), compare against the current implementations on the same inputs (RMSE, runtime) and
  **replace nothing** unless the comparison shows a clear win. COLMAP stays the SfM baseline.

### Task 10 - Complete the proposals pipeline
- `worldstore/proposals.py` known limits (see its docstring): materials/surfaces of applied entities are not copied;
  relationship edges from other entities to added/removed ones are not rewritten; no endpoint compiles new evidence
  into a candidate. Fix the first two with tests (a proposal that adds an entity with a material must apply cleanly or be
  refused with a precise reason), then add `POST /api/worlds/{id}/proposals/from-session/{session_id}` only if the
  existing reconstruction pipeline exposes a function that returns a WorldIR without committing (check
  `apps/api/jobs.py` and `engine/pipeline/`); otherwise record why not.

### Task 11 - Documentation and licences
- Write `docs/research/TECHNOLOGY_DECOMPOSITION.md` and `docs/research/REALITY_ENGINE_TARGET_ARCHITECTURE.md` from the
  real state (no generic boilerplate; every claim points to a file or test).
- Read the actual LICENSE texts of the cloned repos and fill the licence column in
  `capability-fusion-matrix.csv` and `INTEGRATION_DECISIONS.md` (currently "file present, text unreviewed"). Flag:
  ORB_SLAM3 (GPL family - research reading only), IfcOpenShell (LGPL - keep optional), Depth-Anything-V2 (per-checkpoint).
- Reconcile `docs/implementation/PENDING_IMPLEMENTATION.md` (last reconciled 2026-09-14; stale: SAM is not broken,
  exporters exist) using the evidence in the log.

### Task 12 - REA (only with explicit user approval)
REA (`codes-research/rea`, MIT) analyses built artifacts, not source. Do not run `npx rea-agents setup` (edits agent
config) or any `npx` that downloads code until the user approves in chat. If approved, the one useful target is the
local COLMAP binary (`C:\Users\shres\tools\colmap-extracted\bin\colmap.exe`) to confirm build flags (no CUDA); native
analysis needs Ghidra/Hopper/IDA - check what is installed first and report if none.

## 4. Reporting format at every stopping point
1. Tasks completed (with the exact test commands and their results) / partially done / blocked (exact reason).
2. Files changed (`git status --short`), no commits.
3. New defects found, with evidence.
4. Questions that need the user (DECISION REQUIRED items), each with options and your recommendation.
5. The next task you would start.
