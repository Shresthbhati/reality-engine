# PIPELINE EXECUTION BASELINE

**Base commit:** `54103f4b` (feat(kilo): real-photo reconstruction accuracy + acceptance suite)
**Branch:** `cline/216df`
**Investigation date:** 2026-11-28

This document establishes the verified baseline of the Reality Engine reconstruction pipeline execution architecture before any changes from this mandate.

---

## 1. Current Commit and Branch State

| Field | Value |
|-------|-------|
| Base commit SHA | `54103f4b` |
| Base commit message | `feat(kilo): real-photo reconstruction accuracy + acceptance suite (RCPAS Phase 2)` |
| Working branch | `cline/216df` |
| Remote | `origin: https://github.com/Shresthbhati/reality-engine.git` |

## 2. Production Reconstruction Entry Points

Three production entry points can launch reconstruction:

### 2.1 API Worker Path (production-active)
**Route:** `POST /api/reconstructions` → `apps/api/routes_reconstructions.py` → enqueues `RECONSTRUCT_SESSION` job → `apps/api/jobs.py:_run_reconstruct_session`

### 2.2 CLI Path (production-active)
**Command:** `reality reconstruct <package> --output <world.json>`
**File:** `apps/cli/main.py:cmd_reconstruct`

### 2.3 CLI `compile` Path (production-active)
**Command:** `reality compile <dataset> --output <dir>`
**File:** `apps/cli/main.py:cmd_compile`

### 2.4 CLI `run` Path (production-active)
**Command:** `reality run <input> --output <dir>`
**File:** `apps/cli/main.py:cmd_run`

### 2.5 Universal Pipeline Path (production-active, separate framework)
**File:** `engine/orchestrator/universal_pipeline.py:UniversalPipeline.reconstruct`

---

## 3. Verified Call Graph

### API WORKER PATH (production):

```
POST /api/reconstructions (routes_reconstructions.py)
  └─> enqueue_job(RECONSTRUCT_SESSION)
        └─> worker_loop → process_next_job → _run_reconstruct_session
              ├─ seg_evidence_resolve (evidence_resolve stage segment)
              ├─ seg_evidence_quality (evidence_quality stage segment)
              ├─ seg_reconstruct:
              │   ├─ run_progressive (progressive.py)
              │   │   ├─ vertical_slice (for >=2 images, multi-view)
              │   │   │   ├─ run_fine_stages (stages/ladder.py)
              │   │   │   │   ├─ Pose (s04_pose) → run_pose
              │   │   │   │   │   └─ ReconstructionOrchestrator
              │   │   │   │   │       └─ ColmapReconstructionBackend
              │   │   │   │   ├─ Scale (s05_scale) → run_scale + run_frame
              │   │   │   │   ├─ Depth (s06_depth) → run_depth
              │   │   │   │   ├─ Layout (s10_layout) → run_layout
              │   │   │   │   │   └─ compile_reconstruction_to_world
              │   │   │   │   ├─ Perception (s09_perception) → run_perception
              │   │   │   │   ├─ Mesh (s08_mesh) → run_mesh
              │   │   │   │   ├─ Dense (s07_dense) → run_dense
              │   │   │   │   ├─ Detail (s10_detail) → run_detail
              │   │   │   │   └─ Finalize (s10_finalize) → run_finalize
              │   │   │   └─ run_progressive fallback: single-image bootstrap
              │   │   └─ ProgressiveResult
              │   └─ seg_reconstruct returns SegmentReport
              ├─ seg_compile_validate (compile_validate stage segment)
              ├─ seg_changeset_commit (changeset_commit stage segment)
              │   └─ worldstore_service.commit_version (CAS adoption)
              └─ _grade_result → SUCCEEDED/PARTIAL/FAILED/CANCELLED
                    └─ _verify_outcome (WorldStore verification)
```

### CLI PATH (production):

```
reality reconstruct <package> (cmd_reconstruct)
  └─> ReconstructionOrchestrator([ColmapReconstructionBackend])
        └─> backend.reconstruct(evidence) → ReconstructionResult
  └─> reality.compile_world_from_reconstruction(result, options)
        └─> compile_reconstruction_to_world → WorldIR
  └─> write world.json
```

### CLI COMPILE PATH (production):

```
reality compile <dataset> (cmd_compile)
  └─> load_capture_dataset(dataset)
  └─> vertical_slice(items, options)
        └─> run_fine_stages (same fine chain as API worker)
  └─> write worldir.json, report.json, points.ply, cameras.json, exports
```

### UNIVERSAL PIPELINE (production, separate framework):

```
UniversalPipeline.reconstruct (universal_pipeline.py)
  ├─ select_device + compute_governor_policy
  ├─ import_folder → EvidencePackage
  ├─ ReconstructionOrchestrator → ReconstructionResult
  ├─ MetricScaleFusionEngine → ScaleState
  ├─ compile_reconstruction_to_world → WorldIR
  ├─ building_envelope, fine_grained_perception, materials,
  │   spatial_reasoning, design_style, physical_plausibility,
  │   completeness, shared_walls, georeferencing
  ├─ validate_world_ir → WorldValidationReport
  └─ Optional WorldStore.save_version + sdk.reality.export
```
---

## 4. Stage Runner Architecture

### 4.1 Stage Contract (`engine/pipeline/stages/base.py`)

```python
StageSpec:
  name: str                    # stable stage identifier
  version: str                 # implementation version (part of cache key)
  resource_class: Literal["cpu", "gpu", "highmem", "export"]
  required: bool               # required+failed→FAILED; optional+failed→PARTIAL
  inputs: tuple[str, ...] | None # artifact names this stage reads; None = all earlier
  requires_outputs: tuple[str, ...] = ()  # must-produce artifacts; missing→failed
  idempotent: bool = False     # safe to re-run after exception
  retries: int = 0             # extra attempts (only when idempotent)

StageResult:
  status: Literal["succeeded", "degraded", "skipped", "failed"]
  inputs_hash: str             # sha256 of (version, sorted input hashes, params)
  outputs: dict[str, str]      # logical name -> artifact sha256
  started_at / ended_at: datetime
  peak_rss_mb: float | None
```

**Statuses supported:** `succeeded`, `degraded`, `skipped`, `failed`
**Missing statuses from mandate:** `queued`, `running`, `cancellation_requested`, `cancelled`, `timed_out`

### 4.2 Stage Runner (`engine/pipeline/stages/runner.py`)

Key behaviors verified:
- Sequential execution via `run_stages()`
- **Cancellation:** `StageCancelled` raised between stages via `should_cancel()` callback
- **Timeout:** `_run_with_deadline` uses `ThreadPoolExecutor(max_workers=1)` with `fut.result(timeout=...)` — on `TimeoutError`, the thread is **ABANDONED** (not killed), result discarded
- **Checkpoints:** `DurableJobCheckpointStore` persists stage rows; coarse segments are explicitly NOT cacheable
- **Admission:** `StageAdmission` from `engine.hardware.admission` for memory-aware admission control
- **Retries:** `1 + max(0, spec.retries)` when `spec.idempotent`, never retried when `StageTimeout`
- **Missing outputs:** Required missing outputs convert `succeeded`/`degraded` → `failed`

**Known issues identified:**
1. Timeout handling abandons the thread (not safe cancellation) — computation continues mutating shared state
2. `inputs_hash` in `compute_inputs_hash` hashes ALL artifacts in `ctx.artifacts`, not just declared inputs — this is the **cache-key defect**
3. No `queued`/`running`/`cancellation_requested`/`cancelled`/`timed_out` statuses in `StageStatus`
4. No atomic commit of stage output
5. No validation layer for stage outputs before downstream consumption

### 4.3 Fine Stage Chain (`engine/pipeline/stages/ladder.py`)

```python
def fine_chain() -> list:
    return [Pose, Scale, Depth, Layout, Perception, Mesh, Dense, Detail, Finalize]
```

- `Pose` (s04): `run_pose` → `ReconstructionOrchestrator` → backend
- `Scale` (s05): `run_scale` + `run_frame` (metric anchoring + canonicalization)
- `Depth` (s06): `run_depth` (optional monocular depth)
- `Layout` (s10_layout): `run_layout` (scene gate + compile_reconstruction_to_world)
- `Perception` (s09): `run_perception` (Mask R-CNN lifting)
- `Mesh` (s08): `run_mesh` (Poisson surface reconstruction)
- `Dense` (s07): `run_dense` (dense MVS + plane sweep + fusion)
- `Detail` (s10_detail): `run_detail` (detail discovery + ROI refinement)
- `Finalize` (s10_finalize): `run_finalize` (metadata + diagnostics + VerticalSliceResult)

### 4.4 Ladder Report


---

## 5. Orchestration Paths Classification

| Path | Classification | Uses Stage Runner | Uses WorldStore | CAS Adoption | Job Lifecycle |
|------|---------------|-------------------|-----------------|--------------|---------------|
| API worker (`_run_reconstruct_session`) | production-active | Yes (segments) | Yes (commit_version) | Yes | Yes |
| CLI `reconstruct` | production-active | No (orchestrator direct) | No | No | No |
| CLI `compile` | production-active | Yes (vertical_slice→fine_chain) | No | No | No |
| CLI `run` | production-active | No (UniversalPipeline) | Optional | No | No |
| UniversalPipeline | production-active | No (own orchestration) | Optional | No | No |
| `vertical_slice()` (direct) | library/public | Yes (fine_chain) | No | No | No |
| `run_progressive()` | library/public | No (own progression logic) | No | No | No |

---

## 6. Duplicate Orchestration Implementations

### 6.1 `run_progressive` vs `vertical_slice` vs `fine_chain`

Three overlapping orchestration mechanisms exist:

1. **`vertical_slice()`** (vertical_slice.py:708): Delegates to `run_fine_stages(evidence_items, options, stage_sink)` — runs fine stages in data-flow order
2. **`run_progressive()`** (progressive.py:206): Implements its own 6-level progression (L0 single-image → L5 incremental), calls `vertical_slice_fn` for multi-view (L2), falls back to single-image bootstrap
3. **`fine_chain()`** (ladder.py:43): Defines `[Pose, Scale, Depth, Layout, Perception, Mesh, Dense, Detail, Finalize]` — the data-flow order used by `vertical_slice()`

**Assessment:** These are NOT redundant — they serve different purposes:
- `vertical_slice` = stage-segmented execution for the API worker path
- `run_progressive` = progressive level selection (multi-view → single-image fallback)
- `fine_chain` = ordered list of fine stages within a rung

They are composed: `run_progressive` calls `vertical_slice_fn` (which calls `run_fine_stages` which uses `fine_chain`).

### 6.2 `ReconstructionOrchestrator` vs `UniversalPipeline`

Two backend orchestration frameworks:

1. **`ReconstructionOrchestrator`** (reconstruction/orchestrator.py): Selects from candidate backends (COLMAP, PythonSfM, fake), validates evidence, stamps provenance, records attempts
2. **`UniversalPipeline`** (engine/orchestrator/universal_pipeline.py): Standalone framework with hardware profiling, its own scale anchoring (`MetricScaleFusionEngine` vs `anchor_metric_scale`), own compilation, own validation

**Assessment:** These are genuinely separate frameworks with different concerns. `UniversalPipeline` is a higher-level autonomous product path; `ReconstructionOrchestrator` is a backend selection/execution layer. Both can coexist but should not duplicate stage execution logic.

### 6.3 `anchor_metric_scale` vs `MetricScaleFusionEngine`

Two scale anchoring implementations:
- `reconstruction.scale.anchor_metric_scale` (used by `vertical_slice.run_scale`): measured baseline → `ScaledReconstruction`
- `MetricScaleFusionEngine.fuse` (used by `UniversalPipeline._anchor_scale`): multi-way fusion → `ScaleState`

**Assessment:** These serve different paths. `anchor_metric_scale` is the stage-contract path; `MetricScaleFusionEngine` is the universal pipeline path.

---

## 7. Current Defects Identified

| # | Defect | Location | Severity |
|---|--------|----------|----------|
| D1 | Stage timeout abandons thread — computation continues mutating shared state | `runner.py:_run_with_deadline` | Critical |
| D2 | `compute_inputs_hash` hashes ALL artifacts, not just declared inputs | `base.py:compute_inputs_hash` | Critical (cache-key defect) |
| D3 | No `queued`/`running`/`cancellation_requested`/`cancelled`/`timed_out` stage statuses | `base.py:StageStatus` | High |
| D4 | No atomic commit of stage output | `runner.py` | High |
| D5 | No output validation layer before downstream consumption | Multiple | High |
| D6 | `UniversalPipeline` duplicates orchestration logic (scale, compile, validation) | `universal_pipeline.py` | Medium |
| D7 | No dependency-aware cache invalidation | `runner.py` + `lineage.py` | Medium |
| D8 | No per-stage hardware execution provenance (requested vs actual device) | Multiple | Medium |
| D9 | No structured execution record/diagnostics output | Multiple | Medium |
| D10 | CLI paths bypass job lifecycle and CAS adoption | `cli/main.py` | Medium |

---

## 8. Production Path Marked

The **authoritative production path** is the API worker path:
```
POST /api/reconstructions → RECONSTRUCT_SESSION job → _run_reconstruct_session → 
seg_reconstruct → run_progressive → vertical_slice → run_fine_stages → 
fine_chain stages → compile_validate → changeset_commit → commit_version → 
_verify_outcome → _grade_result
```

This path uses the stage runner, WorldStore, CAS adoption, and job lifecycle. The CLI paths are production-active but operate outside the job system and WorldStore commit protocol.

---

## 9. Related Documentation

- `docs/architecture/` — System architecture decisions
- `docs/BUILD_SPEC.md` — Build specification
- `docs/engineering/DESIGN_DECISIONS.md` — Accepted architecture decisions
- `docs/engineering/DISASTER_RECOVERY.md` — Disaster recovery procedures
- `apps/api/jobs.py` — Durable job execution (the job handler)
- `engine/pipeline/stages/runner.py` — Stage runner
- `engine/pipeline/stages/base.py` — Stage contract
- `engine/pipeline/vertical_slice.py` — Vertical slice (stage chain)
- `engine/pipeline/progressive.py` — Progressive reconstruction
- `reconstruction/orchestrator.py` — Reconstruction orchestrator
- `engine/orchestrator/universal_pipeline.py` — Universal pipeline
- `apps/api/worldstore_service.py` — WorldStore bridge
- `worldstore/store.py` — WorldStore (content-addressed version store)
`LadderReport` wraps each fine stage and reports its result from the ladder's fine stage sink, or `skipped` with a reason when the fine stage was not part of the used rung.