# 37 — Reuse map (current state, consolidated)

> Read this **before writing code**. Most capabilities already have a starting
> point. Assessed by reading files on the current branch
> `feat/thermal-envelope-ashrae901` @ `f309041a` (10 November 2026). This is a
> code inventory, **not** runtime verification — every row gets re-verified by
> the task that touches it.
>
> Actions: **KEEP** (use as is) · **EXTEND** (add to it) · **FIX** (known
> defect) · **MERGE** (exists on a branch, bring to main) · **BUILD** (nothing
> usable) · **ADOPT** (integrate external OSS).
>
> The legacy `docs/blueprint/20_REUSE_MAP.md` (assessed @ `1d872fe` on
> `integration/all-worktrees`, 1 November 2026) remains the detailed capability
> inventory. This document consolidates it with current-state changes from
> documents 34–36 and the new modules added in the 40-item backlog foundation
> work.

---

## 0. Branch state

| Item | Current state |
|------|---------------|
| Branch | `feat/thermal-envelope-ashrae901` |
| Commit | `f309041a5ad301c59774fe21bb78103b54b046aa` |
| Working tree | Clean (after committing docs 34–37) |
| Unmerged branch `integration/all-worktrees` | 80 commits ahead of `main` (clean fast-forward); contains `world_ir/observations.py`, `perception/instances/association.py`, `reconstruction/matching/`, `perception/device.py` |

Task 0.1 in `30_BUILD_PLAN.md` merges `integration/all-worktrees` first (after its test suite passes).

---

## 1. New modules added (40-item backlog foundation, docs 34–36)

These are pure-Python, bounded, testable modules. Each has verified imports and a small functional test run.

| Module | What it provides | Action |
|--------|------------------|--------|
| `reconstruction/capture_director.py` | Intelligent capture director: missing-view identification, route planning, object-specific prompts (backlog items A.1, A.4, A.5) | KEEP; wire into `engine/pipeline/guidance.py` for world-space CaptureRequests (Phase 6) |
| `reconstruction/loop_closure.py` | Loop-closure detection between nearby cameras using epipolar verification + duplicate-location detection (B.9) | KEEP; integrate with `reconstruction/scene_identity.py` SceneRegistry |
| `reconstruction/uncertainty_map.py` | Voxelised spatial confidence field: strong/moderate/weak/unobserved/conflicting (B.10) | KEEP; feed ConfidenceVector + Studio uncertainty visualization |
| `reconstruction/architectural_constraints.py` | Constraint solver: walls-meet-floors/ceilings, openings-on-host-walls, windows-in-walls, stairs-connect-floors (D.16) | KEEP; run pre/post edit checks via `reconstruction/constraint_editing.py` |
| `reconstruction/measurement_confidence.py` | Confidence + provenance (tape/image/heuristic/unknown) per dimension (D.19) | KEEP; feed measurement tool `scale_state` rule |
| `representation_consistency.py` | Multi-representation consistency: point cloud/mesh/entity/slab disagreement detection (D.20) | KEEP; report in quality evaluation |
| `reconstruction/world_merge.py` | Three-way world merge: physical_change/user_edit/reconstruction_difference classification, manual-edit protection (E.22) | KEEP; wire into WorldStore commit path |
| `reconstruction/world_geography.py` | Geo-anchors per world, cross-world transforms, merge candidates (E.25) | KEEP; integrate with `world_ir/compose.py` |
| `reconstruction/resumable_jobs.py` | Per-stage checkpointing, idempotent retries, crash-safe recovery (H.39) | KEEP; wire into `apps/api/jobs.py` job lifecycle |
| `reconstruction/evidence_governance.py` | Full provenance chain media→…→export with conflict detection (H.40) | KEEP; feed WorldIR provenance links |
| `reconstruction/object_history.py` | Object-level change history across world versions (E.23) | KEEP; wire into WorldStore version diff |
| `reconstruction/damage_tracking.py` | Physical damage/deterioration tracking across versions (E.24) | KEEP; surface in inspector |
| `reconstruction/constraint_editing.py` | Constraint-aware edit proposals: pre/post-check solver run, manual-edit protection (F.27) | KEEP; route UI edit commands through it |
| `reconstruction/undo_redo.py` | Transactional undo/redo with before/after snapshots, forked history (F.28) | KEEP; wire into editing command bus |
| `reconstruction/floor_plan_validation.py` | Floor-plan vs 3D world validation: wall lengths, door widths, loop closure (F.30) | KEEP; run on plan generation |
| `reconstruction/pipeline_observability.py` | Per-stage observability: input/output hashes, runtime, memory, warnings, failure reasons (H.38) | KEEP; wire into pipeline stage runner |
| `reconstruction/spatial_selection.py` | Unified spatial selection: single/multi mode, select/deselect/hover, source tracking (F.26) | KEEP; wire into viewport/tree/search/inspector |
| `reconstruction/scene_inspector.py` | Rule-based (no LLM) natural-language scene inspection + edit proposals (F.29) | KEEP; expose via assistant API |

---

## 2. Capability → code → action (consolidated from `20_REUSE_MAP.md`)

### Ingestion & evidence

| Capability | Existing code | Action |
|------------|---------------|--------|
| Upload, size limits, evidence rows | `apps/api/routes_misc.py` | KEEP |
| Content-addressed storage | `apps/api/storage.py` | KEEP; add S3 adapter behind same interface |
| Image facts (EXIF, sharpness, class) | `evidence/image_check.py`, `apps/api/routes_evidence_quality.py` | KEEP; EXTEND with ExifTool for HEIC/RAW |
| Sessions | `evidence/session.py`, `session_store.py`, `apps/api/routes_sessions.py` | KEEP |
| Sensors, clocks, multi-source | `evidence/sensors.py`, `clocks.py`, `multi_source.py` | KEEP for Phase 9 |
| Video frames | `evidence/frames.py`, `depth_frames.py` | EXTEND with information-gain keyframe selection |

### Cameras & geometry

| Capability | Existing code | Action |
|------------|---------------|--------|
| Scene grouping | `reconstruction/scene_identity.py` | KEEP; route matching through `IMatcher` |
| Matcher interface | `reconstruction/matching/` (integration branch) | MERGE; ADOPT ALIKED+LightGlue |
| SfM / MVS | `reconstruction/backend/colmap_backend.py`, etc. | KEEP (COLMAP binary on PATH) |
| Scale | `reconstruction/scale.py` | EXTEND: emit `scale_state` + source + uncertainty |
| Dense + fusion | `reconstruction/dense_pipeline.py`, `dense_ingest.py`, `depth_to_points.py`, `fusion/` | KEEP |
| Meshing | `reconstruction/meshing/` | KEEP; depth capped at 8, threads capped at 4 (fixes 40+ min hangs) |
| Cross-view consistency | `reconstruction/consistency.py`, `orchestrator_consistency.py` | KEEP |
| Robustness / admission | `reconstruction/robustness.py`, `robustness_admission.py` | KEEP |
| Cross-session registration | `registration/registration.py`, `cross_session.py`, `landmarks.py` | EXTEND: wire into ingestion |

### Perception

| Capability | Existing code | Action |
|------------|---------------|--------|
| Detection | `perception/detection/maskrcnn_backend.py` | KEEP baseline; ADOPT open-vocab candidate |
| Segmentation | `perception/segmentation/sam_backend.py` | KEEP; benchmark SAM 2 |
| Depth | `perception/depth/midAS_backend.py` | KEEP fallback; ADOPT metric depth |
| Model registry | `perception/model_registry.py` | EXTEND with licence + benchmark fields |
| Lifting 2D→3D | `perception/instances/lifting.py` | FIX: no metric claim from relative depth |
| Object resolution | `perception/instances/object_resolution.py` | FIX: Hungarian is default; union-find only explicit option |
| Architecture | `perception/architecture/*` (~7k lines) | KEEP; EXTEND outputs to Space/Surface/Opening |

### World model & persistence

| Capability | Existing code | Action |
|------------|---------------|--------|
| WorldIR schema | `world_ir/schema_v1.py`, `v2_extensions.py`, `world_v1.py` | EXTEND to contract (hierarchy, EvidenceState, ConfidenceVector) |
| Validation | `world_ir/validation.py` | EXTEND with invariants 1–7 |
| Diff / versions | `world_ir/diff.py`, `worldstore/store.py` | KEEP |
| Incremental closure | `world_ir/incremental.py`, `engine/incremental.py` | EXTEND: wire to new-photo ingestion |
| Measurements | `world_ir/measure.py`, `apps/api/routes_measurements.py` | KEEP |
| Uncertainty, provenance | `uncertainty/`, `provenance/` | KEEP; feed ConfidenceVector + provenance links |

### Pipeline orchestration

| Capability | Existing code | Action |
|------------|---------------|--------|
| Progressive pipeline | `engine/pipeline/vertical_slice.py` | KEEP; move stages to Stage contract gradually |
| Capture guidance | `engine/pipeline/guidance.py` | EXTEND to world-space CaptureRequests |
| Jobs | `apps/api/jobs.py`, `routes_jobs.py` | KEEP; add stage + counts reporting; wire resumable_jobs |

### API & product

| Capability | Existing code | Action |
|------------|---------------|--------|
| Worlds CRUD + versions/commit/diff | `apps/api/routes_worlds.py` | KEEP |
| Rename/duplicate/archive/restore/delete/purge | `World.status` column exists, unused | BUILD (Phase 1) |
| DB migrations | `metadata.create_all` only | BUILD: Alembic |
| Export (GLB, USD, OBJ, IFC bridge, CityJSON) | `exporters/*`, `routes_export.py` | KEEP; FIX UI request paths + surfaced errors |
| Completion proposals | `routes_completion.py`, `routes_procedural.py` | KEEP; output is PRIOR_GENERATED/GENERATED |
| CLI / SDK | `apps/cli/main.py`, `sdk/reality.py` | KEEP |

### Frontend

| Capability | Existing code | Action |
|------------|---------------|--------|
| App shell, desktop routes, mobile routes | `frontend/src/app/(desktop)/*`, `app/m/*` | KEEP; consolidate per `36_UI_SPEC.md` |
| API proxy routes | `frontend/src/app/api/*` | KEEP; one typed client, one path convention |
| 3D viewport | three.js in `frontend/src/components/` | EXTEND: three view modes, entity selection |
| E2E | `frontend/e2e` (Playwright) | KEEP; add lifecycle + acceptance journeys |

---

## 2. External OSS to adopt (candidates — verify licence + benchmark first)

| Need | Project | Action |
|------|---------|--------|
| SfM/MVS | COLMAP (already used), GLOMAP | KEEP / benchmark |
| Learned matching | LightGlue + ALIKED/DISK, LoFTR, RoMa | ADOPT behind `IMatcher` |
| Segmentation | SAM 2 | benchmark vs SAM |
| Open-vocab detection | Grounding DINO, OWLv2 | ADOPT one |
| Metric depth | Depth Anything V2, Metric3D v2, UniDepth | benchmark, ADOPT one |
| Embeddings | DINOv2 | ADOPT as re-ID cue |
| Point clouds / meshes | Open3D | ADOPT for ICP, Poisson |
| Migrations | Alembic | ADOPT (Phase 1) |
| Media | FFmpeg / PyAV, ExifTool | ADOPT (Phase 9) |
| Geo | pyproj (PROJ) | ADOPT (Phase 9) |
| 3D export validation | Khronos `gltf-validator` | ADOPT in export tests |

---

## 3. Name mapping (glossary → code)

| Glossary | Code today |
|----------|------------|
| Observation (ledger) | `world_ir.observations.Observation` (integration) — not `schema_v1.Observation` |
| AssociationDecision | `world_ir.observations.AssociationDecision` |
| EvidenceState | split across `world_ir.quality_state.QualityState` and `Provenance` enum — unify into one `EvidenceState` |
| Space | rooms in `perception/architecture/room_graph.py`, `space_graph.py` |
| Session | `apps.api.models.Session`, `evidence.session` |
| Version | `apps.api.models.WorldVersion` (mirror) + WorldStore version (authoritative) |
| CaptureRequest | `engine/pipeline/guidance.py` (image level) → world-space via `capture_director.py` |
| ChangeSet | `world_ir/diff.py` + WorldStore version |
| Provenance chain | `reconstruction/evidence_governance.py` |

---

## 4. Out of scope for core (do not wire in)

`procedural/` and `engine/compiler/city_compiler.py` (World Factory, generated content), physics/simulation remnants, and `exporters/{sumo,godot,unreal,habitat,ros}`. These stay, labelled GENERATED / domain-specific, and must never write OBSERVED data.

---

## 5. Known limitations (from docs 34–36)

| Limitation | Evidence | Blocker |
|------------|----------|---------|
| No tape-measured rooms | `docs/blueprint/30_BUILD_PLAN.md` 2.3 | Blocks P0/P1 accuracy validation |
| No physical VIO backend | `tests/test_synthetic_vio.py` exists; binaries unavailable | Blocks indoor localization |
| No RGB-D device E2E | Synthetic tests only; no device | Blocks physical depth capture |
| No real measured-room accuracy | Depends on tape-measured rooms | Blocks P1.12 |
| Browser/Playwright E2E on current UI | Not yet run | Blocks UI verification |
| Blank-wall dense coverage | CPU plane-sweep drops textureless walls | Blocks dense geometry completeness |
