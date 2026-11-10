> **SUPERSEDED (2026-11-10).** Current status lives in [docs/STATUS.md](../../docs/STATUS.md); current plan in docs/blueprint/34_MASTER_PLAN.md. Kept for history.

# 20 — Reuse map: what exists, what to do with it

Read this **before writing code**. Most capabilities already have a starting point.

> Assessed by reading files on `main` @ `1d872fe` and branch
> `integration/all-worktrees` on 2026-11-01. This is a code inventory, **not**
> runtime verification — every row gets re-verified by the task that touches it.

Actions: **KEEP** (use as is) · **EXTEND** (add to it) · **FIX** (known defect) ·
**MERGE** (exists on a branch, bring to main) · **BUILD** (nothing usable) ·
**ADOPT** (integrate external OSS).

## 0. The biggest win first: unmerged work

`integration/all-worktrees` is **80 commits ahead of `main` and 0 behind**
(a clean fast-forward). It contains work the plan depends on:

| File (on the integration branch) | What it is |
|----------------------------------|------------|
| `world_ir/observations.py` | Observation ledger + `AssociationState` (all 8 states) + `AssociationDecision` |
| `perception/instances/association.py` | Gated Hungarian association engine: no transitive merges, AMBIGUOUS state; wired as the default resolver |
| `perception/entities/association.py` | Association record on promoted entities ("why is this one object") |
| `reconstruction/matching/{interface,sift}.py` | `IMatcher` interface + SIFT behind it + licence-gated registry |
| `perception/device.py` | Shared CUDA/CPU device selection, 6 GB budget, `REALITY_DEVICE` override |

→ Task 0.1 in [30_BUILD_PLAN.md](30_BUILD_PLAN.md) merges this first (after its test suite passes).

## 1. Capability → code → action

### Ingestion & evidence
| Capability | Existing code | Action |
|------------|---------------|--------|
| Upload, size limits, evidence rows | `apps/api/routes_misc.py` (uploads, evidence, delete evidence) | KEEP |
| Content-addressed storage | `apps/api/storage.py` | KEEP; add an S3 adapter later behind the same interface |
| Image facts (EXIF, sharpness, class) | `evidence/image_check.py`, `apps/api/routes_evidence_quality.py` | KEEP; EXTEND with ExifTool for HEIC/RAW (Phase 9) |
| Sessions | `evidence/session.py`, `session_store.py`, `apps/api/routes_sessions.py` | KEEP |
| Sensors, clocks, multi-source | `evidence/sensors.py`, `clocks.py`, `multi_source.py` | KEEP for Phase 9 |
| Video frames | `evidence/frames.py`, `depth_frames.py` | EXTEND with information-gain keyframe selection (Phase 9) |

### Cameras & geometry
| Capability | Existing code | Action |
|------------|---------------|--------|
| Scene grouping (same place?) | `reconstruction/scene_identity.py` (SIFT + epipolar RANSAC, diagnostics, SceneRegistry) | KEEP; route matching through `IMatcher` |
| Matcher interface | integration: `reconstruction/matching/` | MERGE; ADOPT ALIKED+LightGlue behind it (Phase 4) |
| SfM / MVS | `reconstruction/backend/colmap_backend.py`, `colmap_session.py`, `orchestrator.py`, `merge.py` | KEEP (COLMAP binary on PATH) |
| `features/`, `sfm/`, `registration/`, `mvs/`, `calibration/` READMEs | README-only stubs | Point the READMEs at the real modules; never fill them with duplicate code |
| Scale | `reconstruction/scale.py` | EXTEND: emit `scale_state` + source + uncertainty per the contract |
| Dense + fusion | `reconstruction/dense_pipeline.py`, `dense_ingest.py`, `depth_to_points.py`, `fusion/` | KEEP |
| Meshing | `reconstruction/meshing/` | KEEP |
| Cross-view consistency gate | `reconstruction/consistency.py`, `orchestrator_consistency.py` | KEEP |
| Robustness / admission | `reconstruction/robustness.py`, `robustness_admission.py` | KEEP |
| New-photo / cross-session registration | `registration/registration.py`, `cross_session.py`, `landmarks.py` | EXTEND: wire into ingestion (Phase 6) |

### Perception (ML → Observations)
| Capability | Existing code | Action |
|------------|---------------|--------|
| Detection | `perception/detection/maskrcnn_backend.py` | KEEP as baseline; ADOPT an open-vocab candidate (Phase 4/8) |
| Segmentation | `perception/segmentation/sam_backend.py` | KEEP; benchmark SAM 2 |
| Depth | `perception/depth/midAS_backend.py` (relative) | KEEP as fallback; ADOPT metric depth (Phase 4) |
| Model registry | `perception/model_registry.py` | EXTEND with licence + benchmark fields ([15](15_ML_MODEL_STRATEGY.md)) |
| Device selection | integration: `perception/device.py` | MERGE |
| Lifting 2D → 3D | `perception/instances/lifting.py` | FIX: no metric claim from relative depth (scale gate) |
| Object resolution | `perception/instances/object_resolution.py` (**union-find**, transitive) | FIX: the integration branch makes Hungarian the default — keep union-find only as an explicit option, then remove it |
| Appearance re-ID | `perception/instances/appearance.py` | KEEP as a weak cue; ADOPT DINOv2 embeddings later |
| Tracking (video) | `perception/tracking/temporal.py`, `instances/track_backend.py` | KEEP for Phase 9 |
| Quality | `perception/quality/assessment.py`, `detail_budget.py` | KEEP |
| Fine detail | `perception/detail/` (ROI, discovery, refinement) | KEEP; base for the S17 high-res pass |
| Architecture (planes, rooms, openings, windows, stairs, topology, parametric) | `perception/geometry/planes.py`, `perception/architecture/*` (~7k lines) | KEEP; EXTEND outputs to Space/Surface/Opening per the contract; FIX topology revalidation after merges |

### World model & persistence
| Capability | Existing code | Action |
|------------|---------------|--------|
| WorldIR schema | `world_ir/schema_v1.py`, `v2_extensions.py`, `world_v1.py` | EXTEND to [11](11_WORLDIR_CONTRACT.md) (hierarchy, EvidenceState, ConfidenceVector, CaptureRequest) |
| Validation | `world_ir/validation.py` | EXTEND with invariants 1–7 |
| Diff / versions | `world_ir/diff.py`, `worldstore/store.py`, `apps/api/worldstore_service.py` | KEEP |
| Entity re-ID across versions | `world_ir/entity_reid.py` | KEEP; extend to individual observations |
| Incremental closure | `world_ir/incremental.py`, `engine/incremental.py`, `engine/compiler/incremental_adapter.py` | EXTEND: wire to new-photo ingestion |
| Real vs synthetic firewall | `world_ir/quality_state.py` | KEEP; map onto EvidenceState |
| Measurements | `world_ir/measure.py`, `apps/api/routes_measurements.py` (scale-state rule) | KEEP |
| Uncertainty, provenance | `uncertainty/`, `provenance/` | KEEP; feed ConfidenceVector + provenance links |
| Spatial query | `engine/scene_graph/`, `apps/api/routes_query.py` | KEEP |

### Pipeline orchestration & guidance
| Capability | Existing code | Action |
|------------|---------------|--------|
| Progressive pipeline | `engine/pipeline/progressive.py`, `vertical_slice.py` (1.5k lines), `world_delta.py` | KEEP; move stages to the Stage contract gradually, never big-bang |
| Capture guidance | `engine/pipeline/guidance.py` (camera azimuth gaps — image level only) | EXTEND to world-space CaptureRequests (Phase 6) |
| Jobs (claim, heartbeat, retry, cancel) | `apps/api/jobs.py`, `apps/api/routes_jobs.py` | KEEP; add stage + counts reporting |

### API & product
| Capability | Existing code | Action |
|------------|---------------|--------|
| Worlds create/list/get/versions/commit/rollback/diff | `apps/api/routes_worlds.py` | KEEP |
| Rename / duplicate / archive / restore / delete / purge | — (the `World.status` column exists but is unused for this) | BUILD (Phase 1) |
| DB migrations | `metadata.create_all` only | BUILD: Alembic (Phase 1) |
| Export (GLB, USD, OBJ, IFC bridge, CityJSON…) | `exporters/*`, `apps/api/routes_export.py` | KEEP; FIX UI request paths + surfaced errors (Phase 1) |
| Completion proposals | `apps/api/routes_completion.py`, `routes_procedural.py` | KEEP; output is PRIOR_GENERATED / GENERATED |
| CLI / SDK | `apps/cli/main.py`, `sdk/reality.py` | KEEP |

### Frontend
| Capability | Existing code | Action |
|------------|---------------|--------|
| App shell, desktop routes, mobile routes | `frontend/src/app/(desktop)/*`, `frontend/src/app/m/*` | KEEP; consolidate per [14](14_FRONTEND_UX.md) |
| API proxy routes | `frontend/src/app/api/*` | KEEP; one typed client, one path convention |
| 3D viewport | three.js in `frontend/src/components/` (workspace) | EXTEND: three view modes, entity selection |
| World card actions, trash view | — | BUILD (Phase 1) |
| E2E | `frontend/e2e` (Playwright) | KEEP; add lifecycle + acceptance journeys |

### Out of scope for core (do not wire in)
`procedural/` and `engine/compiler/city_compiler.py` (World Factory, generated
content), physics/simulation remnants, and `exporters/{sumo,godot,unreal,habitat,ros}`.
They stay, labelled GENERATED / domain-specific, and must never write OBSERVED data.

## 2. External OSS to adopt (candidates — verify licence + benchmark first)

| Need | Project | Action |
|------|---------|--------|
| SfM/MVS | COLMAP (already used), GLOMAP | KEEP / benchmark |
| Learned matching | LightGlue + ALIKED/DISK (cvg/LightGlue), LoFTR (kornia), RoMa | ADOPT behind `IMatcher` |
| Segmentation | SAM 2 | benchmark vs SAM |
| Open-vocab detection | Grounding DINO, OWLv2 (HF transformers) | ADOPT one |
| Metric depth | Depth Anything V2 (check per-size licence), Metric3D v2, UniDepth | benchmark, ADOPT one |
| Embeddings | DINOv2 | ADOPT as a re-ID cue |
| Point clouds / meshes | Open3D | ADOPT where meshing/registration needs it (ICP, Poisson) |
| Migrations | Alembic | ADOPT (Phase 1) |
| Media | FFmpeg / PyAV, ExifTool | ADOPT (Phase 9) |
| Geo | pyproj (PROJ) | ADOPT (Phase 9) |
| 3D export validation | Khronos `gltf-validator` | ADOPT in export tests |

## 3. Name mapping (glossary → code)

| Glossary | Code today |
|----------|------------|
| Observation (ledger) | `world_ir.observations.Observation` (integration) — **not** `schema_v1.Observation`, which is a sensor-level record |
| AssociationDecision | `world_ir.observations.AssociationDecision` |
| EvidenceState | split across `world_ir.quality_state.QualityState` and the `Provenance` enum — unify into one `EvidenceState` enum |
| Space | rooms in `perception/architecture/room_graph.py`, `space_graph.py` |
| Session | `apps.api.models.Session`, `evidence.session` |
| Version | `apps.api.models.WorldVersion` (mirror) + WorldStore version (authoritative) |
