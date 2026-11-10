# 34 — Master Plan

> **Repo truth.** This document is the single source of truth for the current
> state of Reality Engine: what exists and is verified, the eleven governing
> decisions, the target architecture, the phased work sequence with exit
> gates, the risk register, and the runbook every model/agent follows. It
> supersedes any scattered status claims. Where this document and the code
> disagree, the code plus its tests win — and this document must be updated
> in the same commit.

---

## 1. Repo truth

**Repository:** `reality-engine` — a photo → 3D world → editable WorldIR → CAD/BIM export spatial engine.
**Branch:** `feat/thermal-envelope-ashrae901` (clean, pushed to origin).
**Entry points:** `reality compile` (CLI), `apps/api` (FastAPI on :8100), `frontend/` (Next.js Studio on :3100).
**Core schema:** `world_ir/world_v1.py` (WorldIR v1) — the single schema the Inspector, compiler, exporters, and Studio read.
**Persistence:** `worldstore/store.py` (SQLite) + Alembic migrations (`apps/api/migrations/`).
**Pipeline:** `engine/pipeline/vertical_slice.py` — the production compile path (SfM → depth → perception → mesh → WorldIR → artifacts).

### What is real and verified

| Area | Evidence |
|------|----------|
| Mapping spine (P0) | Real COLMAP SfM + dense MVS on `datasets/south_building` and `datasets/room_capture/`; `tests/integration`, `tests/test_cli_compile.py`, `tests/test_dense_*`, `tests/test_evidence_fusion.py` |
| Metric depth | MiDaS + SfM sparse-fit alignment, labeled `metric-by-alignment` everywhere; never presented as sensor-grade |
| Depth → points | Trusted-intrinsics unprojection with coordinate frames; honest skip when no trusted intrinsics |
| Detection/segmentation | Mask R-CNN (torchvision COCO) verified on real photographs; SAM backend present but broken in this env (torch-hub cache) |
| Object 3D + multi-view resolution | Lifting, bounds/dimensions, OBSERVED-vs-INFERRED; deterministic geometric+label matching |
| Persistent world | WorldStore SQLite + incremental compilation; crash/restart/arbitration verified under PROD-01 |
| Studio | Next.js client over WorldStore/WorldIR API: evidence/provenance browsing, timeline, diff, status; tsc 0, next build 0, contract tests pass |
| Import boundaries | `tests/test_import_boundaries.py` encodes the layer table; 47 sibling-coupling pairs removed; `engine/pipeline` facade stops `apps/api` importing `reconstruction`/`perception` directly |

### What is in-flight / verified locally, pending commit push
All work in this session is committed and pushed to `feat/thermal-envelope-ashrae901` (0 commits ahead of origin). The working tree is clean.

### What is NOT real (honest boundaries)
- **Tape-measured rooms** (`docs/blueprint/30_BUILD_PLAN.md` 2.3): no 3 real photographed+measured rooms exist in the repo. Blocked on user-provided captures.
- **Real VIO backend** (P3-02): `tests/test_synthetic_vio.py` exists; no built backend, no phone IMU/camera sync.
- **RGB-D device E2E** (P1-03): contract verified on synthetic RGB-D only; no physical device.
- **Real measured-room accuracy** (12.2): depends on 2.3.
- **Browser/Playwright E2E on the current UI**: not yet run against the rebuilt Studio.
- **Blank-wall dense coverage**: CPU plane-sweep drops textureless walls instead of filling; real-coverage improvement not measured on a real room.

---

## 2. The eleven decisions

These are the governing decisions for the target architecture. Each is typed as **SPEC REQUIREMENT** (mandated), **ENGINEERING DECISION** (choice where the spec is silent), or **TEMPORARY APPROXIMATION** (stand-in that must be replaced, tracked in `KNOWN_LIMITATIONS.md`). Only decisions actually reflected in code/tests are listed — this is not a wishlist.

| # | Decision | Type | Rationale | Files |
|---|----------|------|-----------|-------|
| 1 | Frozen (`@dataclass(frozen=True)`) for all WorldIR, config, and math value types | ENGINEERING DECISION | Immutability makes append-only observation history and deterministic replay structurally enforced rather than convention-based | `world_ir/*`, `engine/core/config.py`, `engine/physics/math3.py` |
| 2 | Custom minimal `Vec3`/`Mat3`/`Quat` instead of numpy | ENGINEERING DECISION | No solver needs real linear algebra beyond 3/4-element algebra; numpy is a heavy dependency. Revisit when fluids/structural FEM lands | `engine/physics/math3.py` |
| 3 | Config format is TOML-only | ENGINEERING DECISION | Matches `pyproject.toml`-based tooling; YAML adds a dependency for no current requirement | `engine/core/config.py` |
| 4 | Provenance enum is `OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT` (no separate `MEASURED`) | SPEC_AMBIGUITY | Directive lists `MEASURED` as distinct from `ESTIMATED`; existing `provenance/provenance.py` folds measurement into `OBSERVED`/`ESTIMATED`. Flagged, not silently picked | `provenance/provenance.py` |
| 5 | Inspector reads `world_ir.world_v1.WorldIR` directly, not `WorldRuntime` | ENGINEERING DECISION | `WorldRuntime` does not expose materials/geometries/causal_relations; those exist only on v1 schema. Decision stands even after the WorldRuntime fix | `engine/inspector/inspector.py` |
| 6 | `WorldIR.created_at`/`modified_at` default to `0.0` ("unset"), never wall-clock | ENGINEERING DECISION (bugfix) | Wall-clock defaults broke determinism: two identical `WorldIR()` objects could serialize differently purely by construction time | `world_ir/world_v1.py` |
| 7 | WorldStore (SQLite) + Alembic migrations for persistence; incremental compilation recompiles only the affected closure | ENGINEERING DECISION | A world is a real, manageable, versioned object; crash/restart/arbitration verified under PROD-01 | `worldstore/store.py`, `apps/api/migrations/`, `world_ir/incremental.py` |
| 8 | Multi-view identity is Hungarian association (the only default resolver path); union-find removed | ENGINEERING DECISION | Deterministic geometric+label matching; every association decision stored; no silent false merges | `perception/instances/association.py`, `object_resolution.py` |
| 9 | COLMAP is the SfM/dense MVS backend; availability-gated and capability-probed; never faked with a fallback mesh | ENGINEERING DECISION | An honest absence is better than fabricated geometry; `poisson_mesher` presence probed via `colmap help`, not binary existence | `reconstruction/meshing/surface.py`, `reconstruction/backend/colmap_backend.py` |
| 10 | Studio is a thin client over the WorldStore/WorldIR API; shows backend state only, no fake UI state | ENGINEERING DECISION | Session and World are the only primary concepts; the UI reflects real API state (stage labels, failure reasons, no measured %) | `frontend/src/app`, `apps/api` |
| 11 | Honest defaults: wall thickness is a labelled default (`thickness_basis="default_unmeasured"`), Poisson depth capped at 8, relative depth labeled `relative` never `metric` | ENGINEERING DECISION | No pipeline stage observes both faces of a wall; depth 10 (1024³ octree) caused 40+ min hangs on building-scale inputs | `perception/architecture/wall_slabs.py`, `reconstruction/meshing/surface.py`, `engine/pipeline/vertical_slice.py` |

---

## 3. Target architecture

```
Capture (photos/video/RGB-D)
        │
        ▼
┌─────────────────────────────────────────────┐
│  engine/pipeline/vertical_slice.py          │
│  (SfM/COLMAP → depth → perception → mesh)   │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  world_ir/world_v1.py  (WorldIR v1 schema)  │
│  entities · geometry · surfaces · materials │
│  · observations · provenance · relationships│
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  worldstore/store.py  (SQLite + Alembic)    │
│  versioned worlds · incremental compilation │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  apps/api  (FastAPI :8100)                  │
│  routes: worlds, jobs, export, measurements │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  frontend/  (Next.js Studio :3100)          │
│  four fixed zones + one drawer              │
└─────────────────────────────────────────────┘
```

### Layering rules (import boundaries, enforced by `tests/test_import_boundaries.py`)
- `apps/api` → may import `engine/pipeline` (facade) and `world_ir`; must NOT import `reconstruction/` or `perception/` directly (6 known violations removed via the facade).
- `engine/pipeline` → may import `reconstruction/` and `perception/`; the pure orchestration layer.
- `reconstruction/` → SfM, depth, meshing, capture quality, capture director, loop closure, uncertainty map, damage tracking, object history, pipeline observability, constraint editing, spatial selection, scene inspector.
- `perception/` → detection, segmentation, materials, architecture, instances, layout, fine-grained.
- `world_ir/` → schema, serialization, validation, geometry_data, artifact store — no imports from `reconstruction/` or `perception` (neutral).

### Artifact contract
Every produced geometry carries a content-addressed artifact in `world_ir/artifact_store` referenced by `Geometry.data_uri`/`data_hash`; no untraceable geometry is recorded.

---

## 4. Phases with exit gates

Execution order follows `docs/blueprint/30_BUILD_PLAN.md` (authoritative WHAT). Agents determine the HOW. Take the **first task not DONE whose dependencies are DONE**. Status values: `TODO · IN PROGRESS · DONE (<commit>) · BLOCKED: <reason>`. Never delete a future item because a partial interface exists; Deferred ≠ deleted.

### Phase 0 — Consolidate and baseline (nothing new — make it run)
**Gate:** the app starts with one command; the test baseline is recorded.

| ID | Task | Status |
|----|------|--------|
| 0.1 | Merge `integration/all-worktrees` into `main` | DONE |
| 0.2 | Environment doctor script | DONE |
| 0.3 | CUDA torch in the project venv | DONE (not needed) |
| 0.4 | One-command launcher (API + worker + frontend) | DONE |
| 0.5 | Record the baseline (pytest/frontend build/e2e) | DONE |
| 0.6 | Import-boundary test + violations | DONE |
| 0.6a–0.6c | Boundary cleanup (shared types, pipeline facade, upward imports) | DONE |
| 0.7 | Fix inherited baseline failures | DONE |

### Phase 1 — Product spine: worlds you can manage and export
**Gate:** the lifecycle e2e test passes; an exported GLB opens in Blender.

| ID | Task | Status |
|----|------|--------|
| 1.1 | Alembic migrations | DONE |
| 1.2 | World lifecycle API (PATCH/duplicate/archive/restore/delete/purge) | DONE |
| 1.3 | Trash purge + blob GC | DONE |
| 1.4 | World card menu + Trash/Archived views + delete confirmation | DONE |
| 1.5 | Export fix (GLB validator + Blender) | DONE |
| 1.6 | Honest job progress (stage + stage_counts, no fake %) | DONE |
| 1.7 | Settings honesty | DONE |

### Phase 2 — Baseline benchmark (measure before improving)
**Gate:** one benchmark command produces a committed metrics file.

| ID | Task | Status |
|----|------|--------|
| 2.1 | Benchmark runner | DONE |
| 2.2 | Scenario suite skeleton A1–A20 | DONE |
| 2.3 | Tape-measured rooms | **BLOCKED** — needs real captures by the user |

### Phase 3 — Identity: one object, one entity
**Gate:** A2, A4, A5 pass; duplicate rate on the living room is 0 for the v1 class set.

| ID | Task | Status |
|----|------|--------|
| 3.1 | Complete the Observation record | DONE |
| 3.2 | Association is the only resolver path | DONE |
| 3.3 | Scale gate in lifting | DONE |
| 3.4 | Validator invariants 1–7 | DONE |
| 3.5 | Entity evidence in the inspector | DONE |
| 3.6 | Identity scenarios pass | DONE |

### Phase 4 — Better matching and depth (adopt only if measured better)
**Gate:** each adoption has a benchmark row that beats the baseline, an OK licence, and VRAM within budget.

| ID | Task | Status |
|----|------|--------|
| 4.1 | Registry entries for all current models | DONE |
| 4.2 | One-model-at-a-time scheduler | DONE |
| 4.3 | ALIKED+LightGlue behind `IMatcher` | DONE |
| 4.4 | Metric depth candidate | DONE |

### Phase 5 — Architecture and coverage
**Gate:** the living-room world has Space/Surfaces/Openings; every surface has an evidence state.

| ID | Task | Status |
|----|------|--------|
| 5.1 | Contract hierarchy in WorldIR | DONE |
| 5.2 | Architecture → contract adapters | DONE |
| 5.3 | Ray-cast coverage | DONE |
| 5.4 | Topology revalidation after merges | DONE |

### Phase 6 — Guided capture and incremental updates
**Gate:** demo steps 8–9 work (capture request appears; new photo updates only the affected region).

| ID | Task | Status |
|----|------|--------|
| 6.1 | World-space CaptureRequests | DONE |
| 6.2 | Register new photos into an existing world | DONE |
| 6.3 | Incremental update on ingest | DONE |
| 6.4 | Capture UI | DONE |

### Phase 7 — Evidence, confidence and the three views
**Gate:** demo steps 6–7 work.

| ID | Task | Status |
|----|------|--------|
| 7.1 | ConfidenceVector populated | DONE |
| 7.2 | Real / Wireframe / Evidence views | DONE |
| 7.3 | Measurement with honest scale | DONE |

### Phase 8 — Object and furniture reconstruction
**Gate:** demo step 5 (listed furniture reconstructed exactly once each, with components).

| ID | Task | Status |
|----|------|--------|
| 8.1 | Ontology file + loader | DONE |
| 8.2 | Open-vocab detector via the registry | DONE |
| 8.3 | Parametric object templates | DONE |
| 8.4 | Relationships | DONE |
| 8.5 | User corrections | DONE |

### Phase 8b — Wire perception into the product path
**Gate:** an uploaded room photo produces furniture entities in the World.

| ID | Task | Status |
|----|------|--------|
| 8.6 | Scene-complete eval set (≥20 AI + ≥20 real labeled) | PARTIAL — scorer/runner/label-checker done; 40 labeled photos missing (needs a person) |
| 8.7 | Single-image path (DA-V2 depth, planes, detector → furniture) | PARTIAL — layout+furniture wired, acceptance unmeasured |
| 8.8 | Open-vocab detector + crop pass from API path | DONE |

### Phase U — UI/UX redesign track
**Gate:** UI matches `36_UI_SPEC.md` (40 routes → 12, four fixed zones + one drawer).

| ID | Task | Status |
|----|------|--------|
| U.1 | Frontend audit (routes→APIs, libs, duplicates, tokens) | DONE |
| U.2 | Library compatibility + licence matrix | DONE |
| U.3 | Shared design tokens (light + dark, WCAG AA) | DONE |
| U.4 | Home: New Session · Open World + recent + needs-attention | DONE |
| U.5 | Session: one guided flow with explicit processing states | DONE |
| U.6 | World workspace: viewport-first, hierarchy left, inspector right | DONE |
| U.7 | Session adoption review | DONE |
| U.8 | Mobile priorities + WCAG 2.2 AA pass | DONE |

### Phase 9 — Fine detail, video, geo
| ID | Task | Status |
|----|------|--------|
| 9.1 | High-res crop pass (sockets/switches) | DONE |
| 9.2 | Video keyframe ingestion | DONE |
| 9.3 | Geo anchor (EXIF GPS → World.geo_anchor) | DONE |

### Phase 10 — Appearance and constrained completion
| ID | Task | Status |
|----|------|--------|
| 10.1 | Texture projection from photos onto meshes | DONE |
| 10.2 | Material classification as attributes | DONE |
| 10.3 | Completion proposals as PRIOR_GENERATED | DONE |

### Phase 11 — Product polish
| ID | Task | Status |
|----|------|--------|
| 11.1 | Home dashboard | DONE |
| 11.2 | World package export/import (versioned zip) | DONE |
| 11.3 | Resumable uploads | DONE |
| 11.4 | Navigation consolidation | DONE |

### Phase 12 — v1 acceptance
| ID | Task | Status |
|----|------|--------|
| 12.1 | Living-room demo end to end (all 12 steps in `00_VISION.md`) | DONE |
| 12.2 | Measured-room accuracy report | **BLOCKED** — depends on 2.3 (tape-measured rooms) |
| 12.3 | Mark capabilities VERIFIED in `IMPLEMENTATION_STATUS.md` | DONE |

### Phase 13 — After v1 (capabilities not covered above)
| ID | Task | Status |
|----|------|--------|
| 13.1 | Branch audit + CODEOWNERS | DONE |
| 13.2 | pHash dedup + time/GPS session grouping | DONE |
| 13.3 | HEIC, 360°, PLY/LAS, OBJ/glTF ingestion | DONE |
| 13.4 | Learned room layout, non-Manhattan rooms | DONE |
| 13.5 | Floor plan SVG/PDF export | DONE |
| 13.6 | SAM 2 behind segmentation interface | DONE |
| 13.7 | Outdoor perception | DONE |
| 13.8 | Colour accuracy (ΔE) | DONE |
| 13.9 | MASt3R (NC flag) + DINOv2 re-ID | DONE |

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation / Owner |
|------|------------|--------|---------------------|
| No tape-measured rooms → accuracy claims unverified | High | High | Block 12.2; capture 3 measured rooms (user action); do not ship accuracy numbers without them |
| No physical VIO/RGB-D device → indoor localization unverified | High | Medium | Keep `tests/test_synthetic_vio.py`/`tests/test_synthetic_rgbd.py` as synthetic gates; label as synthetic |
| Blank-wall dense coverage poor | Medium | Medium | CPU plane-sweep refuses textureless walls (honest); do not claim coverage on blank walls |
| Poisson meshing hangs on building-scale clouds | Medium | High | Depth capped at 8, threads capped at 4 (implemented); monitor benchmark runtime |
| Studio UI drift from backend state | Medium | Medium | Thin-client contract (decision 10); contract tests + tsc/build in CI |
| Import-boundary regressions | Low | High | `tests/test_import_boundaries.py` in CI; facade ownership |
| Synthetic fixtures mistaken for real data | Low | High | `datasets/room_capture/` manifest + 1.0 m baseline; synthetic fixture asserted out-of-distribution, never hidden |

---

## 6. The runbook (how any model/agent works here)

This is the loop every model follows on every task. Do not skip steps.

### 1. Understand
- Read `AGENTS.md` (universal agent instructions) and this file.
- Identify which phase/gate the task belongs to; read its "Done when" and "Verify" in `30_BUILD_PLAN.md`.
- Read the relevant files before modifying them. Repository evidence > assumptions.

### 2. Investigate
- Confirm the task is not already DONE (check `IMPLEMENTATION_STATUS.md`, `PENDING_IMPLEMENTATION.md`, git log).
- Check import boundaries: does the file you're touching respect the layer table?
- Check for an existing component/utility/abstraction before creating a new one (reuse before creating).

### 3. Plan
- For non-trivial work, write a concise plan: relevant files, dependencies, architectural impact, sequence, validation strategy, risks.
- Keep plans proportional to complexity. Do not begin large changes without understanding how pieces fit.

### 4. Execute
- Smallest complete solution. Focused changes, clear abstractions, strong typing, explicit error handling.
- No fake implementations, placeholder data as complete, hard-coded API responses, or buttons that do nothing.
- Preserve existing work; do not discard uncommitted changes unless explicitly instructed.

### 5. Verify (mandatory)
- Run the strongest applicable validation: unit tests + type checking/build where applicable; UI: run the app + inspect; API: make an actual request; DB: validate schema/migrations + queries; build: run the build.
- Frontend: `npm --prefix frontend run build` and `tsc`. Backend: `python -m pytest -q`.
- Paste real output into the PR/handoff — never claim "tests pass" unless actually run.

### 6. Review
- Final self-review against correctness/completeness/integration/regression/edge cases/security/performance/maintainability/UX/verification/cleanliness.
- Review the diff: no accidental changes, debug code, secrets, generated junk, unrelated modifications.

### 7. Finish
- Update `docs/implementation/IMPLEMENTATION_STATUS.md` and this plan's status line (`DONE (<commit>)` or `BLOCKED: <reason>`).
- Update `.agent/EXECUTION_STATE.md` and `.agent/TASKS.yaml` so the next session resumes without rediscovery.
- Commit and push to GitHub; do not leave work uncommitted at session end.

### Standing rules
- Dependency before breadth: if P1.10 blocks P1.11, registration goes first.
- Verify before status: no IMPLEMENTED/VERIFIED without execution evidence.
- Deferred ≠ deleted: stay unwired until dependencies are real.
- Honesty first: an untraceable, empty, or unanchored result is refused, not recorded as success. Estimated values are labeled, never presented as measured.
