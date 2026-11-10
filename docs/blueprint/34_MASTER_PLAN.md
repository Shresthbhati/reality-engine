# 34 — Reality Engine Master Implementation Plan

> **For agentic workers:** execute task-by-task from `35_TASK_CARDS.md`. Read section 9 (Runbook) of this file first. One card = one branch = one commit. Steps use `- [ ]` checkboxes.

**Baseline:** `main` @ `199ca3bbf797ca515aa211f80a95c4850e89dbda`, re-inspected locally 2026-11-10 (clean worktree).
**Goal:** one dependable loop — real phone capture → evidence → staged reconstruction → validated WorldIR → honest quality report → reviewable ChangeSet → immutable World version → CAD/BIM deliverables → second visit updates the same World.
**Architecture:** modular FastAPI monolith + DB-backed durable job queue + isolated workers; one staged pipeline; WorldIR as the only canonical model; one Next.js app.
**Tech stack:** Python 3.10–3.12, FastAPI, SQLAlchemy async, Alembic, SQLite (dev) / Postgres+PostGIS (prod), COLMAP, PyTorch, Next.js 16, React 19, three.js 0.186, Tailwind 4, Zustand 5.
**Companion files:** `35_TASK_CARDS.md` (work queue), `36_UI_SPEC.md` (clean UI), `37_REUSE_MAP.md` (open-source repos).
**Supersedes:** `33_TURNAROUND_PLAN.md` (baseline `45ae43e3`, stale). `30_BUILD_PLAN.md` stays as history.

Legend used everywhere: **[V]** verified in source this session · **[A]** taken from the 9 Nov audit, not re-checked · **[D]** decision made by this plan · **[H]** needs a human.

---

## 1. Repository truth at 199ca3bb

### 1.1 Production call graph [V]

```
Browser (frontend/, Next.js)
  └─ POST /api/sessions/{id}/reconstruct        apps/api/routes_sessions.py:378
     POST /api/reconstructions                  apps/api/routes_reconstructions.py:106  (202)
        └─ enqueue_job()                        apps/api/jobs.py:403    → row in `jobs` (status "queued")
Worker process: python -m apps.api.worker       apps/api/worker.py:26 run_worker → worker_loop() jobs.py:1694
  ├─ reap_stale_jobs()                          jobs.py:1364  (heartbeat-based)
  ├─ claim_next_job()                           jobs.py:1440
  └─ process_next_job()                         jobs.py:1492
       └─ _run_reconstruct_session()            jobs.py:565
            └─ _run_reconstruct_session_inner() jobs.py:579–1244   ← ONE 665-LINE FUNCTION
                 job.stage = checking_reconstruction_backend → resolving_evidence → analyzing_evidence
                             → building_rough_model | (multi-view) → committing_version
                 ├─ engine.pipeline.vertical_slice.vertical_slice()   vertical_slice.py:289 (file = 2,084 lines)
                 │     COLMAP backend → WorldArbiter → scale → _depth_stage → _perception_stage
                 │     → _plane_sweep_stage → _fusion_stage → _dense_mvs_stage → _mesh_stage
                 │     → _detail_stage → _layout_evidence_stage → _sidecar_depth_stage
                 ├─ world_ir.validation.validate_world_ir()
                 └─ apps.api.worldstore_service  → WorldStore commit → WorldVersion row
Adoption: POST /api/worlds/{id}/adoptions[/{aid}/commit|reject]   apps/api/routes_adoptions.py:232–488
```

### 1.2 Facts that change the plan [V]

| # | Fact | Evidence |
|---|---|---|
| F1 | `UniversalSpatialPipeline` is **UNWIRED**. Only its own module, `engine/pipeline/__init__.py`, and its test reference it. | `grep -rn UniversalSpatialPipeline --include=*.py` → 3 files |
| F2 | The real pipeline is `vertical_slice()` plus a 665-line job function. No stage contract, no per-stage checkpoint. | `apps/api/jobs.py:579`, `engine/pipeline/vertical_slice.py` |
| F3 | Job states are `queued / running / partial / succeeded / failed / cancelled`; leases, heartbeat, stale reaping, cancel flag exist. No stage table. | `apps/api/jobs.py:79–84`, `models.py:249` |
| F4 | `REALITY_TEST_BACKEND` env var swaps the reconstruction backend inside the production job function. | `apps/api/jobs.py:624` (line 46 of the function) |
| F5 | Accuracy evaluator matches greedily within 1.5 m (walls) / 1.2 m (openings), loops only over matches, and uses `room_entities[0]`. Missing and extra elements cost nothing. | `perception/architecture/accuracy_report.py:120–145, 174, 305, 355` |
| F6 | Sessions proxy fake-empty bug is **already fixed**; it forwards the backend body and returns 502/503 on failure. | `frontend/src/app/api/sessions/route.ts` |
| F7 | CI core job runs 46 named test files; `pytest tests/ --collect-only` runs nothing; ruff `--exit-zero`; mypy `|| true`. 521 `test_*.py` files exist. | `.github/workflows/ci.yml:50–101,174,176` |
| F8 | 11 specialist/guard test files exist and are **not** in the core list: gaussian_splats, temporal_delta, mep_topology, material_takeoff_boq, life_safety_compliance, thermal_envelope_ashrae, universal_spatial_pipeline_rules11_12, no_fake_backend_in_production, clash_detection_bcf, collab_telepresence, section_elevation_drawings. | `ls tests/` vs `ci.yml` |
| F9 | Life-safety PDF defaults its author to `"Reality Engine Certified Auditor"`. | `apps/api/routes_life_safety.py:118`, `engine/pipeline/life_safety_compliance.py:629` |
| F10 | `Provenance` already has OBSERVED / RECONSTRUCTED / ESTIMATED / INFERRED / GENERATED / UNKNOWN / CONFLICT. `Visibility` has only FULL / PARTIAL / OCCLUDED. A second, smaller `StatementState` (OBSERVED / INFERRED) overlaps it. | `provenance/provenance.py:8`, `world_ir/observations.py:33`, `world_ir/statement_state.py:28` |
| F11 | Adoption already marks removals `needs_review`. There is no not-observed outcome. | `worldstore/proposals.py:47,162` |
| F12 | Model registry records licences and already lists `mast3r_vitlarge` as CC-BY-NC-4.0. Nothing blocks a non-commercial model from a production run. | `perception/model_registry.py:348` |
| F13 | Frontend: 40 `page.tsx` routes, 26 API proxy handlers, 225 backend routes, 32 Studio/workspace panel components, `SpatialWorkstation.tsx` 976 lines. Zustand is installed. | `find frontend/src/app -name page.tsx`, `ls frontend/src/components/{studio,workspace}` |
| F14 | Presence registry is in-memory per process. | `engine/pipeline/collab_presence.py:134` |
| F15 | 19 Markdown plans at repo root and 30+ under `docs/` compete as status sources. | `ls *.md`, `ls docs` |

### 1.3 Not inspected (state, don't guess)

Open PRs and branch protection on GitHub; live CI run for this SHA; `claim_next_job` locking semantics on Postgres; S3 adapter key layout; tenant checks per route; Stripe live path; anything on a physical phone. Each has a card that inspects before changing.

---

## 2. The eleven decisions

Each is the default the cards assume. Items marked **[H]** need the owner's yes before the dependent cards run.

| ID | Decision | Why | Replaces |
|---|---|---|---|
| D1 | **One staged pipeline.** Add `engine/pipeline/stages/` (stage contract + runner + run manifest). Wrap the existing `_xxx_stage` functions in place — strangler pattern, no rewrite. | F2: nothing can be observed, resumed or tested per stage today. Gives backlog items 38, 39 for free. | 665-line job function |
| D2 | **Fold or delete `UniversalSpatialPipeline`.** Its seven stages are inventoried; any that add value become stages under D1; the class and its test are then removed. | F1: two pipelines, one unused. | second "universal" pipeline |
| D3 | **Test backend leaves production code.** `REALITY_TEST_BACKEND` is honoured only when `REALITY_ENV=test`; otherwise the job fails loudly. | F4: a fake backend reachable in production is the definition of false success. | env seam in job function |
| D4 | **Evaluator counts what is missing.** One-to-one optimal matching, every ground-truth element accounted for, per-class precision / recall / F1, all rooms. Gate fails on any missing wall or opening. No threshold tuning. | F5. Release claims depend on it. | greedy matched-only scoring |
| D5 | **Phone-native tracking, not a self-built VIO.** Ingest ARKit / ARCore poses, intrinsics, depth and IMU as *pose priors* for COLMAP. Step 1 is an importer for an existing open capture format; step 2 is a thin native wrapper. ORB-SLAM3 / OpenVINS are dropped (GPL-3, unbuilt binaries). **[H]** | Solves backlog 6–9 with weeks of work instead of quarters; the phone already runs a calibrated VIO. | `trajectories/backend/selection.py` external SLAM binaries |
| D6 | **Feed-forward geometry as a second backend.** Add a metric multi-view backend (MapAnything Apache checkpoint; Depth Anything 3 as alternative) behind the existing `ReconstructionBackend` interface. `WorldArbiter` picks per scene. COLMAP stays. **[H]** for the GPU budget. | Blank walls, low overlap and few photos are exactly where SfM fails and where the first customer lives. | COLMAP-only pose path |
| D7 | **Licence gate.** A production run refuses any model whose registry licence is not on an allow-list. | F12. One CC-BY-NC weight in a paid product is a legal problem. | trust |
| D8 | **Keep the DB job queue.** Add a `job_stage_runs` table, resource classes and per-tenant concurrency. Adopt nothing new unless the crash tests in `S7xx` fail. | F3: leases, heartbeat and reaping already exist. Temporal would be fashion. | — |
| D9 | **Coverage is a first-class enum.** Add `ObservationCoverage` (observed-present / observed-absent / occluded / not-in-view / insufficient-quality / not-observed). Adoption gains a `not_observed` outcome computed from camera frusta. Retire `StatementState` into `Provenance`. | F10, F11. "Not seen" must never read as "removed". | boolean visibility |
| D10 | **UI: 40 routes → 12, 32 panels → 4 zones + 1 drawer.** See `36_UI_SPEC.md`. No second app, no renderer migration, no new state library. | F13. | panel sprawl |
| D11 | **One status ledger.** `docs/STATUS.md` generated from a YAML registry; every other status document gets a one-line "superseded" banner and moves to `docs/archive/`. CI lists what it runs in `docs/CI_TEST_REGISTRY.md`. | F7, F15. | 50 competing documents |

---

## 3. Target architecture (delta only)

### 3.1 What stays exactly as it is

FastAPI app and router layout · SQLAlchemy models · Alembic · `WorldStore` immutable versions · `worldstore/proposals.py` change assessment · `world_ir/` schema v1 + v2 extensions · COLMAP backend · model registry · S3 adapter · Next.js app, `SpatialWorkstation`, three.js viewport · durable queue in `apps/api/jobs.py`.

### 3.2 What is added

```
engine/pipeline/stages/
  base.py        StageSpec, StageContext, StageResult, Stage (Protocol)
  runner.py      run_stages(), RunManifest
  registry.py    PIPELINE_PROFILES = {"interior_survey_standard": [...]}
  s01_evidence.py … s13_changeset.py     thin wrappers over existing functions
apps/api/models.py           + JobStageRun
apps/api/migrations/versions + 0005_job_stage_runs.py
reconstruction/backend/      + feedforward_backend.py  (D6)
evidence/                    + arkit_capture.py        (D5)
world_ir/observations.py     + ObservationCoverage     (D9)
perception/architecture/accuracy_report.py   rewritten matching + ClassScore (D4)
perception/model_registry.py + assert_production_licences() (D7)
docs/STATUS.md, docs/status.yaml, docs/CI_TEST_REGISTRY.md, docs/DECISIONS_INDEX.md (D11)
```

### 3.3 Stage contract (the one new abstraction)

```python
# engine/pipeline/stages/base.py
ResourceClass = Literal["cpu", "gpu", "highmem", "export"]
StageStatus   = Literal["succeeded", "degraded", "skipped", "failed"]

@dataclass(frozen=True)
class StageSpec:
    name: str                 # "pose"
    version: str              # bump on behaviour change; part of the cache key
    resource_class: ResourceClass
    required: bool            # required + failed  → job FAILED
                              # optional + failed  → job PARTIAL

@dataclass
class StageResult:
    name: str
    status: StageStatus
    inputs_hash: str          # sha256 of (spec.version, sorted input artifact hashes, params json)
    outputs: dict[str, str]   # logical name -> artifact sha256
    metrics: dict[str, float]
    warnings: list[str]
    error: str | None
    started_at: datetime
    ended_at: datetime
    peak_rss_mb: float | None

class Stage(Protocol):
    spec: StageSpec
    def run(self, ctx: "StageContext") -> StageResult: ...

@dataclass
class RunManifest:
    pipeline_profile: str
    pipeline_version: str
    stages: list[StageResult]
    def grade(self, specs: list[StageSpec]) -> Literal["succeeded", "partial", "failed"]: ...
```

Grading rule, in one place: any required stage `failed` → `failed`; else any stage `failed` or `degraded` → `partial`; else `succeeded`. `skipped` never upgrades or downgrades. This replaces `_stage_facts_degraded` (`jobs.py:144`).

Resume rule: before running a stage the runner looks up `(job_id, name, inputs_hash)` in `job_stage_runs`; a `succeeded` row whose output artifacts still exist is reused. A retry never overwrites an artifact — artifacts are content-addressed already.

### 3.4 Pipeline profile `interior_survey_standard`

| # | Stage | Class | Required | Wraps today |
|---|---|---|---|---|
| 1 | `evidence_resolve` | cpu | yes | `jobs.py` "resolving_evidence" block |
| 2 | `evidence_quality` | cpu | yes | "analyzing_evidence" block, `reconstruction/capture_quality.py` |
| 3 | `capture_priors` | cpu | no | **new** (D5) ARKit poses / intrinsics / depth |
| 4 | `pose` | gpu | yes | COLMAP backend + `WorldArbiter`; feed-forward candidate (D6) |
| 5 | `scale` | cpu | yes | `reconstruction/scale`, `_device_depth_scale` |
| 6 | `depth` | gpu | no | `_depth_stage`, `_sidecar_depth_stage` |
| 7 | `dense` | highmem | no | `_plane_sweep_stage`, `_dense_mvs_stage`, `_fusion_stage` |
| 8 | `mesh` | highmem | no | `_mesh_stage` |
| 9 | `perception` | gpu | no | `_perception_stage`, `_detail_stage` |
| 10 | `layout` | cpu | yes | `_layout_evidence_stage`, room solvers in `perception/architecture/` |
| 11 | `compile_validate` | cpu | yes | WorldIR compile + `validate_world_ir` |
| 12 | `quality_report` | cpu | yes | **new**: geometry, coverage and uncertainty summary; `NOT_EVALUABLE` is never `PASS` |
| 13 | `changeset_commit` | cpu | yes | "committing_version" block |

---

## 4. Phases, gates and order

Order is by dependency, not by calendar. Phase U (UI) runs in parallel from the start because it touches only `frontend/`.

```
P0 Truth ──► P1 One pipeline ──► P3 Geometry upgrade ──► P4 Persistent World ──► P7 Expansion
   │              │                     ▲
   │              └──► P2 Real data ────┘        (P2 needs humans with a tape measure)
   ├──► PU Clean UI (parallel, frontend only)
   └──► P6 SaaS operations (parallel after P1)
P5 Capture: importer after P1; native wrapper after P2 proves the loop
```

| Phase | Goal | Exit gate (evidence, not opinion) | Cards |
|---|---|---|---|
| **P0 Truth** | Stop false success; make CI say what it runs. | `docs/CI_TEST_REGISTRY.md` matches `ci.yml`; 11 unlisted suites triaged into fast / nightly; licence gate and test-backend gate green; "Certified" wording gone; `docs/STATUS.md` generated. | T001–T012 |
| **P1 One pipeline** | Every job runs through the stage runner; stages are persisted. | Golden-path test passes through `run_stages`; killing the worker mid-job and restarting reuses finished stages; `UniversalSpatialPipeline` deleted or folded; `GET /api/jobs/{id}/stages` returns real rows. | P101–P114 |
| **P2 Real data** | Six measured rooms through the real API, scored honestly. | `benchmarks/measured_rooms/` holds raw photos, consent, ground truth; evaluator report shows per-class P/R/F1 and missing lists; exports opened in two external tools with screenshots. | E201–E212 |
| **P3 Geometry upgrade** | Beat the P2 baseline on the same six rooms. | Recall and dimensional error improve on the blind hold-out with no threshold change; licences pass D7. | G301–G310 |
| **P4 Persistent World** | A second visit updates the same World safely. | Three-room apartment + revisit: zero duplicate Worlds, zero overwritten edits, not-observed never shown as removed. | W401–W410 |
| **PU Clean UI** | Twelve routes, four zones. | Route inventory shows 12 primary routes, the rest redirects; first-time user reaches a deliverable without opening a specialist tool; axe checks pass on Home, Capture, Studio. | U501–U520 |
| **P5 Capture** | Sensor-backed capture. | An ARKit capture imports with poses, intrinsics, depth; pose-prior run registers ≥ as many images as the no-prior run on the same set. | C601–C606 |
| **P6 SaaS ops** | Safe to host. | Cross-tenant test suite covers every router; worker-kill tests pass on Postgres; Stripe test-mode walkthrough recorded. | S701–S712 |
| **P7 Expansion** | Customer-pulled only. | A named customer workflow per item. | X8xx (listed, not carded) |

### 4.1 The 40 improvements, placed

| # | Improvement | Where it lands |
|---|---|---|
| 1 Capture director · 2 coverage map · 5 object prompts | PU `U515` (surface existing `capture-coverage` + `CaptureRequest`), deeper work P7 |
| 3 Image quality | P1 stage 2 (exists: `reconstruction/capture_quality.py`) + `U514` |
| 4 Route planning | P7 |
| 6 Timestamp sync · 7 VIO · 8 floors · 9 loop closure | P5 via D5 (the phone does 7 and 9; we consume the result) |
| 10 Uncertainty map | P1 stage 12 + `U510` evidence legend |
| 11 Fine components · 12 glass · 13 occlusion · 15 materials | P3 `G306–G309` (12, 13), P7 (11, 15) |
| 14 Persistent identity · 21 registration · 22 three-way merge · 23 change history | P4 |
| 16 Constraint solver · 18 watertightness · 20 representation consistency | P1 stage 11–12 (`P110`, `P111`) |
| 17 Parametric geometry · 27 constraint-aware editing | P7 after P4 |
| 19 Measurement tolerances | P2 `E206` (calibrated from real residuals) |
| 24 Damage tracking · 25 geographic identity | P7 |
| 26 Unified selection · 28 undo · 30 plan validation | PU `U508`, exists (verify `U509`), P1 stage 12 |
| 29 NL inspection | P7; deterministic copilot stays as is |
| 31–35 Rendering | P7; `U511` gives the representation switch only |
| 36 Benchmark · 37 false negatives | P2, D4 |
| 38 Stage observability · 39 resumable jobs | P1 (D1) |
| 40 Provenance governance | P1 stage manifest + P4 D9 |

---

## 5. Requirement traceability (condensed)

Status vocabulary: MISSING · FOUNDATION · PARTIAL · UNWIRED · IMPL-UNVERIFIED · CONTRACT-VERIFIED · FIELD-VALIDATED · DEFERRED.

| Requirement | Status | Evidence | Next |
|---|---|---|---|
| Photo → World through the API | PARTIAL [V] | call graph 1.1; `tests/test_golden_path_api_to_glb.py` in CI | P101–P108 |
| Stage contract / observability | MISSING [V] | F2 | P101–P105 |
| Resumable jobs | FOUNDATION [V] | leases + heartbeat, no checkpoints (F3) | P104, P109 |
| Honest job states | CONTRACT-VERIFIED [V] | `jobs.py:79–91`, `reconstruction_state.py` | T007 (unknown state in UI) |
| Universal pipeline | UNWIRED [V] | F1 | P112–P113 |
| Accuracy gate | IMPL-UNVERIFIED, defective [V] | F5 | E201–E205 |
| Measured-room dataset | MISSING [A] | no `benchmarks/measured_rooms/` | E207–E210 [H] |
| Closed windows / glass | IMPL-UNVERIFIED [A] | `tests/test_closed_windows.py` (synthetic) | G306–G307 |
| Feed-forward geometry | MISSING [V] | registry has MASt3R only, NC | G301–G305 |
| VIO / IMU | FOUNDATION [A] | `evidence/sensors.py` parsers, no consumer | C601–C606 |
| Provenance classes | CONTRACT-VERIFIED [V] | F10 | W401 |
| Not-observed vs removed | MISSING [V] | F10, F11 | W402–W405 |
| Session → World adoption | IMPL-UNVERIFIED [V] | `routes_adoptions.py`, `tests/test_world_adoption_api.py` in CI | W406–W410 |
| Typed edit commands + undo | IMPL-UNVERIFIED [A] | `engine/commands/`, `WorldEditCommand` model | U509 |
| CAD/BIM export | IMPL-UNVERIFIED [A] | `exporters/`, IfcOpenShell + ezdxf extras | E211–E212 [H] |
| Specialist tools (MEP, clash, takeoff, life-safety, thermal) | IMPL-UNVERIFIED, not in CI [V] | F8, F9 | T003–T006, U512 |
| Gaussian splats | PARTIAL (I/O only) [A] | `engine/pipeline/gaussian_splats.py` | DEFERRED to P7 |
| Tenant isolation | IMPL-UNVERIFIED [A] | `apps/api/tenancy.py`, `tests/test_security_dast.py` in-process | S701–S704 |
| Billing | IMPL-UNVERIFIED [A] | `billing_service.py` simulated without keys | S710 [H] |
| Distributed presence | FOUNDATION [V] | F14 | S711 (document limit; build when multi-worker) |
| One status source | DOCUMENTATION-DEBT [V] | F15 | T010–T011 |

---

## 6. Model allocation

| Tier | Who | Gets | Never gets |
|---|---|---|---|
| **N** | Nemotron / Qwen-class | Cards tagged `[N]`: tests against a given signature, wrappers with the body described, wording changes, CI edits, redirects, inventories, doc generation. | Anything that says "decide", schema changes, merge semantics, auth. |
| **S** | Fable / Opus / Sonnet-high | Cards tagged `[S]`: the stage runner core, evaluator matching, coverage semantics, backend integration, pipeline migration of the job function, review of every `[N]` diff. | Final say on `[H]` items. |
| **H** | You | Tape measure, phone, consent forms, external CAD checks, GPU budget, Stripe keys, D5 and D6 approval. | — |

Why Nemotron can match a strong model here: every `[N]` card fixes the file list, the signatures, the test names and assertions, and the command. The card leaves the function body and nothing else. Where a card needs a fact not yet verified, its step 0 is a `grep` that either confirms it or stops the task.

Cost shape: about 70 % of cards are `[N]`, 22 % `[S]`, 8 % `[H]`. The `[S]` cards come first in each phase and produce the interfaces the `[N]` cards consume.

---

## 7. Risk register

| Risk | Likelihood | Impact | Control |
|---|---|---|---|
| Real rooms score far below synthetic | High | High | P2 before P3; publish the baseline, do not tune thresholds (D4) |
| Feed-forward model does not fit 6 GB VRAM | High | Medium | `G301` measures first; profile falls back to COLMAP; cloud GPU class is an [H] decision |
| Non-commercial weights slip in | Medium | High | D7 gate in CI and at job start |
| Stage migration breaks the golden path | Medium | High | Strangler: one stage per card, golden-path test after each, old path kept until `P108` |
| Cheap model edits outside scope | Medium | Medium | Allowed-files list per card; reviewer runs `git diff --stat` against it; two-strike escalation |
| Cross-tenant leak | Low | Critical | `S701` enumerates every router; test per resource type |
| Duplicate Worlds on revisit | Medium | High | P4 gate is a physical three-room test |
| UI consolidation removes a route someone uses | Medium | Low | Redirects, never deletions, until `U520` |
| Specialist output read as certification | Medium | High | `T004` wording, `U512` "Preliminary" badge, tests assert the disclaimer |
| Customer media privacy | Medium | High | Dataset protocol `E207` with consent and retention; no customer data in training |

---

## 8. Decisions that need you [H]

1. **D5** — approve phone-native tracking and drop the external SLAM binaries. Recommended: yes.
2. **D6** — approve a feed-forward backend, and say whether a cloud GPU is allowed for jobs the local 6 GB card cannot run. Recommended: yes, cloud allowed per project.
3. **Tolerances** — wall ±2 cm, area ±1 %, opening ±2 cm are still proposals. Keep as targets; report honestly until two firms confirm.
4. **Six rooms** — who measures, where, and which phone. Nothing in P2 or P3 can be proven without this.
5. **Primary CAD/BIM target** — pick two external tools for `E211–E212` (suggest Bonsai/BlenderBIM for IFC because it is free, plus whatever the first firm uses for DXF).
6. **Branch protection on `main`** — a GitHub setting; require CI + E2E.
7. **Delete vs archive** the 19 root Markdown plans. Recommended: archive with a banner (`T011`).

---

## 9. Runbook for any executing model

Read this once per session. It is the whole process.

1. **Pick** the lowest-numbered card in `35_TASK_CARDS.md` whose `Needs` are all merged and whose tier matches you. Never start a `[S]` or `[H]` card as an `[N]` worker.
2. **Branch:** `git checkout -b task/<CARD-ID>` from the current integration branch. Record `git rev-parse HEAD` in your report.
3. **Step 0 of every card** is its `Check` command. If the output does not match what the card says, **stop and report**; do not adapt the task.
4. **Touch only** the paths under `Files`. If you believe another file must change, stop and report.
5. **Order of work:** write the test → run it and see it fail → implement → run it and see it pass → run the card's `Regression` command.
6. **Commands** (repo root unless stated):
   - Python tests: `pytest <path> -q --tb=short`
   - Golden path: `pytest tests/test_golden_path_api_to_glb.py tests/test_durable_job_queue.py tests/test_world_adoption_api.py -q --tb=short`
   - Frontend: `cd frontend && npx tsc --noEmit && npm run lint && npm run test:contract`
7. **Commit** once, message `<CARD-ID>: <title>`. Do not push. Do not open a PR. Do not amend other commits.
8. **Report** in exactly this shape:

```
CARD: <id>          BASE: <sha>          COMMIT: <sha>
FILES CHANGED: <git diff --stat output>
TESTS RUN: <command>  →  <last line of output, verbatim>
REGRESSION: <command> →  <last line of output, verbatim>
NOT RUN: <what and why, or "none">
DEVIATIONS: <anything that differs from the card, or "none">
```

9. **Hard rules.** No fake data, no success-shaped fallback, no `except: pass`, no new dependency, no schema or API shape change unless the card says so, no deleting a fixture, no editing a test to make it pass unless the card names that test. Never write a test result you did not see printed.
10. **Two strikes.** If the card's test still fails after two honest attempts, stop and report the failing output. A stronger model takes it.
11. **Reviewer checklist** (a different model or a human): diff inside `Files`? test fails on the base commit? assertions match the card? no rule in 9 broken? commands in the report reproducible?

---

## 10. Self-review of this plan

- **Coverage:** every handoff priority (baseline, truth, CI, evaluator, dataset, WorldIR, merge, capture, SaaS, expansion) maps to a phase in section 4; all 40 improvements are placed in 4.1.
- **Unverified inputs:** rows marked [A] in section 5 rest on the 9 Nov audit. Their cards start with a check step.
- **Known gaps:** no card writes native iOS/Android code (blocked on D5 and a device); P7 is a list, not cards, on purpose.
