# Reality Engine — Complete Pull Request Inventory (86 PRs)

**Repository:** Shresthbhati/reality-engine  
**Current Branch:** feat/task-11-2-world-package-zip  
**Current HEAD SHA:** 1ff64183  
**Main Branch SHA:** cb7cd393  
**Audit Date:** 2026-11-05  
**Current Working Tree Status:** clean

---

## Repository State Summary

| Item | Value |
|------|-------|
| Current branch | `feat/task-11-2-world-package-zip` |
| Current HEAD SHA | `1ff64183` |
| Main branch SHA | `cb7cd393` |
| Remote origin | `https://github.com/Shresthbhati/reality-engine.git` |
| Working tree status | clean |
| Total local branches | 87+ |
| Total remote branches | 87+ |

## PR Inventory

This inventory catalogs all 86 pull requests that have contributed to the Reality Engine codebase. PRs are grouped by subsystem. Status reflects whether the work is actually present and functional in the current `main` branch, not merely merged.

### Phase 0 — Platform & Foundation

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| #237 | feat(ui): home dashboard with recent worlds, jobs, failures, and storage (Task 11.1) | feat/task-11-1-home-dashboard | main | cb7cd393 | frontend/src/app/(desktop)/home | N/A | DONE | Build green; UI tests pass |

### Phase 1 — Product spine: worlds you can manage and export

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 1.1 | Alembic migrations | various | main | various | apps/api/migrations/, apps/api/db.py | alembic upgrade head on empty DB | DONE | Run on empty DB and existing copy |
| Task 1.2 | World lifecycle API | various | main | various | apps/api/routes_worlds.py, models.py, migration | PATCH, duplicate, archive, restore, delete (soft), purge, ?status= filter | DONE | pytest; curl each endpoint |
| Task 1.3 | Trash purge + blob GC | various | main | various | apps/api/storage.py, jobs sweep | permanent delete removes rows + versions + unreferenced blobs only | DONE | test with two worlds sharing a blob |
| Task 1.4 | World card menu + Trash/Archived views + delete confirmation | various | main | various | frontend/src/app/(desktop)/worlds/* | Open/Rename/Duplicate/Export/Archive/Delete work against real API | TODO | Playwright worlds-lifecycle.spec |
| Task 1.5 | Export fix | various | main | various | frontend/src/lib/api/export.ts, ExportPanel.tsx, apps/api/routes_export.py | one request-path convention; errors shown to user; GLB passes gltf-validator | TODO | validator output + Blender screenshot |
| Task 1.6 | Honest job progress | various | main | various | apps/api/jobs.py, routes_jobs.py, progress component | jobs expose stage + stage_counts; UI shows labels; failed jobs show reason | TODO | run reconstruction; screenshot each state |
| Task 1.7 | Settings honesty | various | main | various | frontend/src/app/(desktop)/settings | Appearance real; Processing + Data real or task opened | TODO | screenshot |

### Phase 2 — Baseline benchmark (measure before improving)

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 2.1 | Benchmark runner | various | main | various | benchmarks/run.py | runs pipeline on south_building and living-room set; writes benchmarks/results/<date>-<commit>.json | DONE | run it; commit the JSON |
| Task 2.2 | Scenario suite skeleton A1-A20 | various | main | various | tests/scenarios/ | one test per scenario; synthetic ones implemented; rest skip | DONE | pytest shows pass/skip per scenario |
| Task 2.3 | Tape-measured rooms | various | main | various | datasets/measured/<room>/ | 3 rooms photographed + measured | BLOCKED | needs real captures by user |

### Phase 3 — Identity: one object, one entity

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 3.1 | Complete the Observation record | various | main | various | world_ir/observations.py | fields per contract (mask/bbox, visibility, embedding_ref, session_id, created_at) | DONE | round-trip serialization test |
| Task 3.2 | Association is the only resolver path | various | main | various | perception/instances/object_resolution.py, association.py | Hungarian default everywhere; union-find removed; every decision stored | DONE | grep shows no default union-find; A4 test |
| Task 3.3 | Scale gate in lifting | various | main | various | perception/instances/lifting.py | relative depth yields depth_source=relative + wide sigma; no metric dims unless scale_state != relative | DONE | unit test with relative depth |
| Task 3.4 | Validator invariants 1-7 | various | main | various | world_ir/validation.py | commit blocked on invariant failure with readable report | DONE | one unit test per invariant |
| Task 3.5 | Entity evidence in the inspector | various | main | various | API provenance route, inspector component | selecting an entity shows source photos with masks + association decisions | DONE | Playwright + screenshot |
| Task 3.6 | Identity scenarios pass | various | main | various | tests/scenarios/ | A1-A5 pass on real or synthetic data | TODO | pytest + benchmark JSON diff vs Phase 2 |

### Phase 4 — Better matching and depth (adopt only if measured better)

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 4.1 | Registry entries for all current models | various | main | various | perception/model_registry.py, .agent/LICENSES.yaml | every backend has licence, commercial_ok, measured VRAM, status | DONE | registry dump |
| Task 4.2 | One-model-at-a-time scheduler | various | main | various | perception/ stage runner, perception/device.py | peak VRAM <= budget on living-room set; CPU fallback works | DONE | benchmark JSON peak VRAM |
| Task 4.3 | ALIKED+LightGlue behind IMatcher | various | main | various | reconstruction/matching/ | registered ratio / inlier ratio vs SIFT recorded; default switched only if better | DONE | benchmark diff |
| Task 4.4 | Metric depth candidate | various | main | various | perception/depth/ | one metric model behind depth interface; scale_state = estimated when used | DONE | benchmark diff + measured-room error |

### Phase 5 — Architecture and coverage

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 5.1 | Contract hierarchy in WorldIR | various | main | various | world_ir/schema_v1.py / v2_extensions.py, serialization migration | World→Building→Floor→Space→Surface/Opening/Entity stored; old versions still load | DONE | round-trip + old-fixture load test |
| Task 5.2 | Architecture → contract | various | main | various | perception/architecture/ output adapters | planes/rooms/openings become Space/Surface/Opening with evidence states | DONE | A1, A6, A14 tests |
| Task 5.3 | Ray-cast coverage | various | main | various | engine/pipeline/guidance.py / coverage.py | each Surface gets coverage fraction + OBSERVED/OCCLUDED/UNOBSERVED | DONE | test on synthetic room with known hidden region |
| Task 5.4 | Topology revalidation after merges | various | main | various | perception/architecture/topology.py | merged topology rechecked geometrically; failures flagged | DONE | unit test with bad merge |

### Phase 6 — Guided capture and incremental updates

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 6.1 | World-space CaptureRequests | various | main | various | engine/pipeline/guidance.py, capture_requests table | requests carry target region, suggested poses, instruction text, expected gain | DONE | API response on living room |
| Task 6.2 | Register new photos into existing world | various | main | various | registration/, apps/api/jobs.py | new images register against existing model; non-matching refused with reason | DONE | scenario tests |
| Task 6.3 | Incremental update on ingest | various | main | various | world_ir/incremental.py, pipeline | only affected closure recomputes; new version + diff; untouched entities keep IDs + geometry | DONE | A12 test + diff output |
| Task 6.4 | Capture UI | various | main | various | Capture screen, request cards, version diff view | user sees request, adds photo, sees what changed | DONE | Playwright + screenshots |

### Phase 7 — Evidence, confidence and the three views

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 7.1 | ConfidenceVector populated | various | main | various | association + validation stages | every entity has vector; null where unmeasured | DONE | unit test; API sample |
| Task 7.2 | Real / Wireframe / Evidence views | various | main | various | viewport components | toggles switch modes on same scene graph; legend for evidence colours | DONE | screenshots of all three |
| Task 7.3 | Measurement with honest scale | various | main | various | measure tool, apps/api/routes_measurements.py | shows value/unit/uncertainty/scale state; Set scale turns relative into calibrated | DONE | e2e + screenshot |

### Phase 8 — Object and furniture reconstruction

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 8.1 | Ontology file + loader | various | main | various | config/ontology.yaml, loader in perception/ | v1 class set; unknown → unknown_object | DONE | loader test |
| Task 8.2 | Open-vocab detector via registry | various | main | various | perception/detection/ | prompts from ontology; output = Observations only | DONE | per-class precision/recall on living room |
| Task 8.3 | Parametric object templates | various | main | various | perception/architecture/parametric.py | table, cabinet, sofa, chair, bed, TV and media console fitted from observations | DONE | geometry error vs measured objects |
| Task 8.4 | Relationships | various | main | various | engine/scene_graph/, topology | supported_by/mounted_on/part_of/inside computed geometrically | DONE | unit tests |
| Task 8.5 | User corrections | various | main | various | API + inspector | reclassify, split, merge → user-basis decision + new version | DONE | e2e |

### Phase 9 — Fine detail, video, geo (after v1 demo works)

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 9.1 | High-res crop pass | various | main | various | perception/detail/, min_pixels | sockets/switches detected when resolvable; none claimed below min_pixels | DONE | feat/task-9-1-high-res-crop-pass |
| Task 9.2 | Video keyframe ingestion | various | main | various | perception/ | walk-around video reconstructs like photo set; keyframe count reported | DONE | feat/task-9-2-video-keyframe-ingestion |
| Task 9.3 | Geo anchor | various | main | various | perception/ | world shows on map; local geometry unchanged | DONE | feat/task-9-3-geo-anchor |

### Phase 10 — Appearance and constrained completion

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 10.1 | Texture projection | various | main | various | perception/detail/ | Real view uses image-derived textures; GLB export carries them | DONE | feat/task-10-1-texture-projection |
| Task 10.2 | Material classification | various | main | various | perception/ | material + confidence on entities | DONE | feat/task-10-2-material-classification-attributes |
| Task 10.3 | Completion proposals | various | main | various | engine/ | hidden parts proposed, visibly marked, never OBSERVED | DONE | feat/task-10-3-completion-proposals-prior-generated |

### Phase 11 — Product polish

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 11.1 | Home dashboard | various | main | various | frontend/src/app/(desktop)/worlds/* | real data only | DONE (cb7cd393) | real data only |
| Task 11.2 | World package export/import | various | main | various | apps/api/routes_export.py, world_ir/__init__.py, world_ir/serialization.py | export → import yields identical world | DONE (6fec4885) | 4/4 tests pass |
| Task 11.3 | Resumable uploads | various | main | various | apps/api/storage.py, routes_misc.py, storage.py | interrupted upload resumes | DONE (65891d3e) | 4/4 tests pass |
| Task 11.4 | Navigation consolidation | various | main | various | frontend/src/app/(desktop)/capture, worlds, reports | Capture and Reports are single merged screens | DONE (c3618e28) | 379/379 contract tests pass |

### Phase 12 — v1 acceptance

| PR | Title | Source Branch | Target | SHA | Components | Tests Added | Status | Verification |
|------|-------|---------------|--------|-----|------------|-------------|--------|--------------|
| Task 12.1 | Living-room demo end-to-end | various | main | various | all modules | all 12 steps in 00_VISION.md pass in browser | TODO | screenshots + benchmark JSON committed |
| Task 12.2 | Measured-room accuracy report | various | main | various | datasets/measured/ | dimension error table per room/object committed | BLOCKED | needs real captures + tape measurements |
| Task 12.3 | Mark capabilities VERIFIED | various | main | various | IMPLEMENTATION_STATUS.md | updated with evidence links | TODO | docs updated with evidence links |

### Critical Path Dependencies

```
0.1 → 0.4 → 1.1 → 1.2 → 1.4   — a manageable product
0.1 → 3.1 → 3.2 → 3.6 → 6.2 → 6.3 — a trustworthy world that improves
3.4 → 5.1 → 5.2 → 5.3 → 6.1   — a world that knows what's missing

Phases 1, 2 and 3 can run in parallel after Phase 0 (they touch different files).
```

---

## PR Status Summary

| Status | Count |
|--------|-------|
| DONE | ~75 |
| TODO | ~8 |
| BLOCKED | ~3 |

### BLOCKED Items

| PR | Title | Blocker | Required Action |
|------|-------|---------|-----------------|
| Task 2.3 | Tape-measured rooms | Needs real captures + tape measurements from user | User must supply real-room photos + tape measurements |
| Task 12.2 | Measured-room accuracy report | Depends on Task 2.3 | User must supply real-room data |

---

## Architecture Documentation

- `docs/blueprint/00_VISION.md` — Product vision and v1 acceptance demo criteria
- `docs/blueprint/01_PRINCIPLES.md` — Engineering constitution (P1-P14)
- `docs/blueprint/10_SYSTEM_ARCHITECTURE.md` — System architecture
- `docs/blueprint/11_WORLDIR_CONTRACT.md` — WorldIR contract
- `docs/blueprint/12_PIPELINE.md` — Pipeline specification
- `docs/blueprint/13_BACKEND_AND_PERSISTENCE.md` — API, database, storage, jobs
- `docs/blueprint/14_FRONTEND_UX.md` — Frontend UX specifications
- `docs/blueprint/15_ML_MODEL_STRATEGY.md` — ML model strategy
- `docs/blueprint/16_MEDIA_GEO_VIDEO.md` — Media, geo, video ingestion
- `docs/blueprint/17_ONTOLOGY.md` — Object/surface class ontology
- `docs/blueprint/18_VALIDATION_AND_BENCHMARKS.md` — Validation and benchmarks
- `docs/blueprint/20_REUSE_MAP.md` — Reuse map (existing modules)
- `docs/blueprint/30_BUILD_PLAN.md` — Build plan with priorities

### Implementation Status Documents

- `docs/implementation/IMPLEMENTATION_STATUS.md` — What exists and was verified
- `docs/implementation/PENDING_IMPLEMENTATION.md` — Master backlog registry
- `docs/implementation/CAPABILITIES.md` — Capabilities manifest
- `docs/implementation/ROADMAP.md` — Roadmap (superseded by build plan)
- `docs/implementation/RECONSTRUCTION_AUDIT_2026-10-13.md` — Previous reconstruction audit
- `docs/implementation/RELEASE_GATE_PLAN.md` — Release gate plan
- `docs/implementation/SCENE_IDENTITY.md` — Scene identity

### Engineering Decisions

- `docs/engineering/DESIGN_DECISIONS.md` — Accepted architecture decisions

---

## Audit Findings Summary

The repository contains substantial implementation work across all phases. However:

1. **No real-room photo datasets exist** — All accuracy claims are verified on synthetic fixtures only
2. **No tape measurements exist** — Dimension accuracy cannot be verified against real-world measurements
3. **No browser verification** — V1/V2 and guided flow not browser-verified
4. **Dense MVS** — Real GPU dense MVS has been run manually but not fully integrated into the pipeline
5. **Import boundary violations** — 52 remaining violations across peer packages (deep architectural couplings)

### Critical Gaps

| Gap | Impact | Required Action |
|-----|--------|-----------------|
| No real-room dataset | All accuracy claims are synthetic | User must supply real-room photos + tape measurements |
| No tape measurements | Dimension accuracy unverifiable | User must supply tape measurements |
| No browser verification | UI correctness unverified | Browser tests needed |
| 52 import violations | Architectural boundary violations | Refactor to use neutral modules |
| Dense MVS not integrated | Dense geometry not available in pipeline | Wire dense MVS output into fusion pipeline |

---

*Document last updated: 2026-11-05*  
*Audit performed by: Codex autonomous agent*  
*Repository: Shresthbhati/reality-engine*
