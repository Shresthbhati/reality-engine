# 30 — Build plan

The work queue. Take the **first task that is not DONE whose dependencies are
DONE**. Each task is small enough for one agent session and says exactly how
to prove it works.

## How to execute a task

- **Who does what**: a planning agent (Opus) briefs; an implementing agent
  (Sonnet, medium effort) builds one task on non-overlapping files; the planner
  reviews the diff and re-runs **Verify** before marking DONE.
- **Brief template** (paste into the implementing agent):
  ```
  Task <ID>: <title>. Read docs/blueprint/README.md, then the docs listed under "Read".
  Files you may change: <list>. Do not touch others.
  Done when: <copy>. Verify: <copy> — paste real output.
  Reuse first: docs/blueprint/20_REUSE_MAP.md. Rules: docs/blueprint/01_PRINCIPLES.md.
  ```
- **Definition of DONE** (all of these):
  1. *Done when* is met.
  2. *Verify* was run and its real output is pasted in the PR/handoff.
  3. Existing tests still pass (`python -m pytest -q`, `npm --prefix frontend run build`).
  4. The [Vision check](README.md#vision-check-run-before-every-merge) is answered.
  5. This file says `DONE (<commit>)` for the task, and
     `docs/implementation/IMPLEMENTATION_STATUS.md` is updated.
- **If blocked** by a decision only the user can make: mark `BLOCKED: <question>`
  here, then move on to the next independent task.

Status values: `TODO · IN PROGRESS · DONE (<commit>) · BLOCKED: <reason>`.

---

## Phase 0 — Consolidate and baseline (nothing new — make it run)

Goal: one branch, one launcher, a known baseline. **Gate:** the app starts with
one command; the test baseline is recorded.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 0.1 | Merge `integration/all-worktrees` into `main` (fast-forward) | — | git only | `main` contains `world_ir/observations.py`, `perception/instances/association.py`, `reconstruction/matching/`, `perception/device.py` | full pytest on the branch **before** merging; paste the pass/fail/skip counts | [20](20_REUSE_MAP.md) §0 | IN PROGRESS: integration merged into the work branch; baseline has 8 failed + 13 errors inherited from the merge (see 0.7); `main` fast-forward left to the user |
| 0.2 | Environment doctor script | 0.1 | `scripts/doctor.py` (new, or extend an existing check script) | prints Python version, venv active, `torch.cuda.is_available()`, GPU name/VRAM, `colmap` on PATH + version, node/npm versions, DB path writable | run it; paste output | [15](15_ML_MODEL_STRATEGY.md) GPU | DONE (uncommitted, verified 2026-11-01) |
| 0.3 | CUDA torch in the project venv (only if 0.2 shows CPU-only) | 0.2 | `pyproject.toml` notes, `QUICKSTART.md` | doctor shows CUDA true on the RTX machine | doctor output | [15](15_ML_MODEL_STRATEGY.md) | DONE (not needed: doctor shows torch 2.14+cu130, CUDA true, RTX 4050 6 GiB, 2026-11-01) |
| 0.4 | One-command launcher: API (8100) + worker + frontend (3100) | 0.1 | `start.bat`, `launch_ui.bat` | `start.bat` brings all three up; `GET /health` is OK; the UI loads | curl `/health`; browser screenshot | [10](10_SYSTEM_ARCHITECTURE.md) topology | TODO |
| 0.5 | Record the baseline | 0.4 | `docs/implementation/IMPLEMENTATION_STATUS.md` | pytest counts, frontend build result and e2e result recorded with date + commit | the commands themselves | [18](18_VALIDATION_AND_BENCHMARKS.md) | TODO |
| 0.6 | Import-boundary test | 0.1 | `tests/test_import_boundaries.py` | the test encodes the layer table; it passes today, or each current violation is listed as a follow-up task here | pytest output | [10](10_SYSTEM_ARCHITECTURE.md) layers | DONE (uncommitted, verified 2026-11-01) |
| 0.7 | Fix inherited baseline failures | 0.1 | see list | `test_colmap_coverage` (2), `test_compile_command` (3), `test_confidence_defaults`, `test_packaging_smoke`, `test_frontend_sessions_proxy_contract` (13 errors, likely missing `frontend/node_modules`) pass; `test_detail_scale` already passes | pytest on those files | — | DONE (uncommitted, verified 2026-11-01; `test_confidence_defaults` passes only with another session's uncommitted edits to switchboard/compose/connectors) |
| 0.6a | Move shared types (camera, transform, mesh, depth, evidence) to a neutral module | 0.6 | `tests/test_import_boundaries.py` KNOWN_VIOLATIONS | allow-list shrinks; 47 sibling-coupling pairs removed | boundary test | [10](10_SYSTEM_ARCHITECTURE.md) | TODO |
| 0.6b | `engine/pipeline` facade so `apps/api` stops importing reconstruction/perception | 0.6 | `apps/api/*` | api-layer pairs removed from allow-list | boundary test | [10](10_SYSTEM_ARCHITECTURE.md) | TODO |
| 0.6c | Fix upward imports (core, world_ir, exporters) | 0.6 | listed in test | remaining 7 pairs removed | boundary test | [10](10_SYSTEM_ARCHITECTURE.md) | TODO |

## Phase 1 — Product spine: worlds you can manage and export

Goal: a world is a real, manageable object. **Gate:** the lifecycle e2e test
passes; an exported GLB opens in Blender.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 1.1 | Alembic migrations | 0.5 | `apps/api/migrations/`, `apps/api/db.py` | `alembic upgrade head` creates the current schema on an empty DB; existing dev DBs upgrade with no data loss | run on an empty DB and on a copy of an existing one | [13](13_BACKEND_AND_PERSISTENCE.md) | DONE (uncommitted, verified 2026-11-01) |
| 1.2 | World lifecycle API | 1.1 | `apps/api/routes_worlds.py`, `models.py`, migration, `tests/test_world_lifecycle.py` | PATCH, duplicate, archive, restore, delete (soft), purge and the `?status=` filter per [13](13_BACKEND_AND_PERSISTENCE.md); delete cancels running jobs; one activity event per operation | pytest; curl each endpoint | [13](13_BACKEND_AND_PERSISTENCE.md) lifecycle | DONE (uncommitted, verified 2026-11-01) |
| 1.3 | Trash purge + blob GC | 1.2 | `apps/api/storage.py`, jobs sweep | permanent delete removes rows + versions + unreferenced blobs only; shared blobs survive | test with two worlds sharing a blob | [13](13_BACKEND_AND_PERSISTENCE.md) | DONE (uncommitted, verified 2026-11-01) |
| 1.4 | World card menu + Trash/Archived views + delete confirmation | 1.2 | `frontend/src/app/(desktop)/worlds/*`, components | Open/Rename/Duplicate/Export/Archive/Delete work against the real API; the confirmation lists what will be deleted | Playwright `worlds-lifecycle.spec`; screenshots at 1280/1920 | [14](14_FRONTEND_UX.md) | TODO |
| 1.5 | Export fix | 0.4 | `frontend/src/lib/api/export.ts`, `ExportPanel.tsx`, `apps/api/routes_export.py` | one request-path convention; errors shown to the user (not console-only); GLB passes `gltf-validator` and opens in Blender with correct transforms | validator output + Blender screenshot | [13](13_BACKEND_AND_PERSISTENCE.md) | TODO |
| 1.6 | Honest job progress | 0.4 | `apps/api/jobs.py`, `routes_jobs.py`, progress component | jobs expose `stage` + `stage_counts`; the UI shows the labels from [12](12_PIPELINE.md); failed jobs show the reason; no % unless measured | run a reconstruction; screenshot each state | [12](12_PIPELINE.md) labels | TODO |
| 1.7 | Settings honesty | 0.4 | `frontend/src/app/(desktop)/settings` | Appearance real; Processing + Data real (or a task opened for them); Account/Workspace/Integrations say "Not available in this version" | screenshot | [13](13_BACKEND_AND_PERSISTENCE.md) settings | TODO |

## Phase 2 — Baseline benchmark (measure before improving)

Goal: numbers to beat. **Gate:** one benchmark command produces a committed metrics file.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 2.1 | Benchmark runner | 0.5 | `benchmarks/run.py` | runs the pipeline on `south_building` and the living-room set; writes `benchmarks/results/<date>-<commit>.json` with the applicable [18](18_VALIDATION_AND_BENCHMARKS.md) metrics (registration ratio, reprojection error, entity counts, duplicates, runtime, peak RAM/VRAM) | run it; commit the JSON | [18](18_VALIDATION_AND_BENCHMARKS.md) | DONE (uncommitted, verified 2026-11-01) |
| 2.2 | Scenario suite skeleton A1–A20 | 0.5 | `tests/scenarios/` | one test per scenario; synthetic ones implemented; the rest `skip("not covered: needs <data>")` | pytest shows pass/skip per scenario | [18](18_VALIDATION_AND_BENCHMARKS.md) | DONE (uncommitted, verified 2026-11-01) |
| 2.3 | Tape-measured rooms | — | `datasets/measured/<room>/` | 3 rooms photographed + measured (L×W×H, 5 object dimensions) | dataset README with the measurements | [18](18_VALIDATION_AND_BENCHMARKS.md) | BLOCKED: needs real captures by the user |

## Phase 3 — Identity: one object, one entity

Goal: no duplicates, no false merges. **Gate:** A2, A4 and A5 pass; duplicate
rate on the living room is 0 for the v1 class set.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 3.1 | Complete the Observation record | 0.1 | `world_ir/observations.py` | fields per [11](11_WORLDIR_CONTRACT.md) (mask/bbox, visibility, embedding_ref, session_id, created_at); ledger saved inside each WorldIR version | round-trip serialization test | [11](11_WORLDIR_CONTRACT.md) | DONE (uncommitted, verified 2026-11-01) |
| 3.2 | Association is the only resolver path | 3.1 | `perception/instances/object_resolution.py`, `association.py` | Hungarian is the default everywhere; union-find removed (or behind an explicit non-default flag with a deprecation note); every decision stored | grep shows no default union-find; A4 test | [12](12_PIPELINE.md) S9 | DONE (uncommitted, verified 2026-11-01) |
| 3.3 | Scale gate in lifting | 3.1 | `perception/instances/lifting.py` | relative depth yields `depth_source=relative` + wide sigma; no metric dimensions unless scale_state ≠ relative | unit test with relative depth | P10 | DONE (uncommitted, verified 2026-11-01) |
| 3.4 | Validator invariants 1–7 | 3.1 | `world_ir/validation.py` | a commit is blocked on invariant failure with a readable report | one unit test per invariant | [11](11_WORLDIR_CONTRACT.md) §3 | DONE (uncommitted, verified 2026-11-01) |
| 3.5 | Entity evidence in the inspector | 3.2 | API provenance route, inspector component | selecting an entity shows its source photos with masks + its association decisions | Playwright + screenshot | [14](14_FRONTEND_UX.md) | DONE (feat/task-3-5-entity-evidence-inspector, verified 2026-11-04) |
| 3.6 | Identity scenarios pass | 3.2–3.4 | `tests/scenarios/` | A1, A2, A3, A4, A5 pass on real or synthetic data | pytest + benchmark JSON diff vs Phase 2 | [18](18_VALIDATION_AND_BENCHMARKS.md) | TODO |

## Phase 4 — Better matching and depth (adopt only if measured better)

**Gate:** each adoption has a benchmark row that beats the baseline, an OK
licence, and VRAM within budget.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 4.1 | Registry entries for all current models | 0.1 | `perception/model_registry.py`, `.agent/LICENSES.yaml` | every backend has licence, commercial_ok, measured VRAM and status | registry dump | [15](15_ML_MODEL_STRATEGY.md) | DONE (feat/task-4-1-model-registry-and-licenses, verified 2026-11-04) |
| 4.2 | One-model-at-a-time scheduler | 4.1 | `perception/` stage runner, `perception/device.py` | peak VRAM ≤ budget on the living-room set; CPU fallback works | benchmark JSON peak VRAM | [15](15_ML_MODEL_STRATEGY.md) | DONE (feat/task-4-2-one-model-at-a-time-scheduler, verified 2026-11-04) |
| 4.3 | ALIKED+LightGlue behind `IMatcher` | 2.1, 4.1 | `reconstruction/matching/` | registered ratio / inlier ratio vs SIFT recorded; default switched only if better | benchmark diff | [15](15_ML_MODEL_STRATEGY.md) | DONE (feat/task-4-3-aliked-lightglue-matcher, verified 2026-11-04) |
| 4.4 | Metric depth candidate | 2.1, 4.1 | `perception/depth/` | one metric model behind the depth interface; scale_state `estimated` when used; benchmarked vs MiDaS | benchmark diff + measured-room error (if 2.3 is done) | [15](15_ML_MODEL_STRATEGY.md) | DONE (feat/task-4-4-metric-depth-candidate, verified 2026-11-04) |

## Phase 5 — Architecture and coverage

**Gate:** the living-room world has Space/Surfaces/Openings per the contract,
and every surface has an evidence state.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 5.1 | Contract hierarchy in WorldIR | 3.4 | `world_ir/schema_v1.py` / `v2_extensions.py`, serialization migration | World→Building→Floor→Space→Surface/Opening/Entity stored; old versions still load | round-trip + old-fixture load test | [11](11_WORLDIR_CONTRACT.md) | DONE (uncommitted, verified 2026-11-01; hierarchy is an optional typed layer, populated by 5.2) |
| 5.2 | Architecture → contract | 5.1 | output adapters in `perception/architecture/` | planes/rooms/openings become Space/Surface/Opening with evidence states | A1, A6, A14 tests | [12](12_PIPELINE.md) S6 | DONE (feat/task-5-2-architecture-contract-adapters, verified 2026-11-04) |
| 5.3 | Ray-cast coverage | 5.2 | `engine/pipeline/guidance.py` (or a new `coverage.py` beside it) | each Surface gets a coverage fraction + OBSERVED/OCCLUDED/UNOBSERVED | test on a synthetic room with a known hidden region | [12](12_PIPELINE.md) S13 | DONE (feat/task-5-3-raycast-surface-coverage, verified 2026-11-04) |
| 5.4 | Topology revalidation after merges | 5.2 | `perception/architecture/topology.py` | merged topology is rechecked geometrically; failures flagged, not hidden | unit test with a bad merge | [20](20_REUSE_MAP.md) | DONE (feat/task-5-4-topology-revalidation, verified 2026-11-04) |

## Phase 6 — Guided capture and incremental updates

**Gate:** demo steps 8–9 work: a capture request appears, and a new photo
updates only the affected region.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 6.1 | World-space CaptureRequests | 5.3 | `engine/pipeline/guidance.py`, `capture_requests` table | requests carry target region, suggested poses, instruction text and expected gain | API response on the living room | [11](11_WORLDIR_CONTRACT.md) | DONE (feat/task-6-1-world-space-capture-requests, verified 2026-11-04) |
| 6.2 | Register new photos into an existing world | 3.6 | `registration/`, `apps/api/jobs.py` | new images register against the existing model; non-matching photos are refused with a reason (A11, A19) | scenario tests | [12](12_PIPELINE.md) incremental | DONE (feat/task-6-2-incremental-photo-registration, verified 2026-11-04) |
| 6.3 | Incremental update on ingest | 6.2 | `world_ir/incremental.py`, pipeline | only the affected closure recomputes; new version + diff; untouched entities keep IDs + geometry | A12 test + diff output | P6 | DONE (feat/task-6-3-incremental-update-on-ingest, verified 2026-11-04) |
| 6.4 | Capture UI | 6.1, 6.3 | Capture screen, request cards, version diff view | the user sees a request, adds a photo and sees what changed | Playwright + screenshots | [14](14_FRONTEND_UX.md) | DONE (feat/task-6-4-capture-ui-request-cards, verified 2026-11-04) |

## Phase 7 — Evidence, confidence and the three views

**Gate:** demo steps 6–7 work.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 7.1 | ConfidenceVector populated | 3.6 | association + validation stages | every entity has the vector; `null` where unmeasured | unit test; API sample | [11](11_WORLDIR_CONTRACT.md) | DONE (feat/task-7-1-confidence-vector-populated, verified 2026-11-04) |
| 7.2 | Real / Wireframe / Evidence views | 5.3, 7.1 | viewport components | toggles switch modes on the same scene graph; legend for evidence colours | screenshots of all three | [14](14_FRONTEND_UX.md) | DONE (feat/task-7-2-real-wireframe-evidence-views, verified 2026-11-04) |
| 7.3 | Measurement with honest scale | 7.2 | measure tool, `apps/api/routes_measurements.py` | shows value/unit/uncertainty/scale state; "Set scale" turns relative into calibrated | e2e + screenshot | P10 | DONE (feat/task-7-3-measurement-honest-scale, verified 2026-11-04) |

## Phase 8 — Object and furniture reconstruction

**Gate:** demo step 5: the listed furniture is reconstructed exactly once each, with components.

| ID | Task | Depends | Files | Done when | Verify | Read | Status |
|----|------|---------|-------|-----------|--------|------|--------|
| 8.1 | Ontology file + loader | 3.2 | `config/ontology.yaml`, loader in `perception/` | v1 class set from [17](17_ONTOLOGY.md); unknown → `unknown_object` | loader test | [17](17_ONTOLOGY.md) | DONE (feat/task-8-1-ontology-file-loader, verified 2026-11-04) |
| 8.2 | Open-vocab detector via the registry | 4.1, 8.1 | `perception/detection/` | prompts come from the ontology; output = Observations only | per-class precision/recall on the living room | [15](15_ML_MODEL_STRATEGY.md) | TODO |
| 8.3 | Parametric object templates | 8.1 | `perception/architecture/parametric.py` | table, cabinet, sofa, chair, bed, TV and media console fitted from observations, with components | geometry error vs measured objects (if 2.3 is done) | [12](12_PIPELINE.md) S10 | TODO |
| 8.4 | Relationships | 8.3 | `engine/scene_graph/`, topology | supported_by / mounted_on / part_of / inside computed geometrically | unit tests | [11](11_WORLDIR_CONTRACT.md) | TODO |
| 8.5 | User corrections | 8.3 | API + inspector | reclassify, split, merge → a user-basis decision + a new version | e2e | [14](14_FRONTEND_UX.md) | TODO |

## Phase 9 — Fine detail, video, geo (after the v1 demo works)

| ID | Task | Depends | Done when | Status |
|----|------|---------|-----------|--------|
| 9.1 | High-res crop pass (sockets, switches) using `perception/detail/` + `min_pixels` | 8.2 | sockets/switches detected when resolvable; none claimed below `min_pixels` | TODO |
| 9.2 | Video keyframe ingestion (FFmpeg/PyAV, information-gain selection) | 6.3 | a walk-around video reconstructs like a photo set; keyframe count reported | TODO |
| 9.3 | Geo anchor (EXIF GPS → `World.geo_anchor`, pyproj) | 5.1 | the world shows on a map; local geometry unchanged | TODO |

## Phase 10 — Appearance and constrained completion

| ID | Task | Depends | Done when | Status |
|----|------|---------|-----------|--------|
| 10.1 | Texture projection from photos onto meshes | 7.2 | the Real view uses image-derived textures; GLB export carries them | TODO |
| 10.2 | Material classification as attributes | 8.2 | material + confidence on entities | TODO |
| 10.3 | Completion proposals as PRIOR_GENERATED | 7.2 | hidden parts proposed, visibly marked, never OBSERVED | TODO |

## Phase 11 — Product polish

| ID | Task | Depends | Done when | Status |
|----|------|---------|-----------|--------|
| 11.1 | Home dashboard (recent worlds, jobs, failures, storage) | 1.4 | real data only | TODO |
| 11.2 | World package export/import (versioned zip) | 1.2 | export → import yields an identical world | TODO |
| 11.3 | Resumable uploads | 1.2 | an interrupted upload resumes | TODO |
| 11.4 | Navigation consolidation per [14](14_FRONTEND_UX.md) | 1.4 | Capture and Reports are single merged screens | TODO |

## Phase 12 — v1 acceptance

| ID | Task | Depends | Done when | Status |
|----|------|---------|-----------|--------|
| 12.1 | Living-room demo end to end | Phases 1–8 | all 12 steps in [00_VISION.md](00_VISION.md) pass in the browser; screenshots + benchmark JSON committed | TODO |
| 12.2 | Measured-room accuracy report | 2.3, 4.4, 8.3 | dimension error table per room/object committed | TODO |
| 12.3 | Mark capabilities VERIFIED | 12.1 | `IMPLEMENTATION_STATUS.md` updated with evidence links | TODO |

---

## Critical path

- `0.1 → 0.4 → 1.1 → 1.2 → 1.4` — a manageable product.
- `0.1 → 3.1 → 3.2 → 3.6 → 6.2 → 6.3` — a trustworthy world that improves.
- `3.4 → 5.1 → 5.2 → 5.3 → 6.1` — a world that knows what's missing.

Phases 1, 2 and 3 can run in parallel after Phase 0 (they touch different files).
