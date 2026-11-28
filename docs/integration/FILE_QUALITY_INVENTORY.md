# File-Quality Inventory — Reality Engine V1

> Per-file quality metrics for the oversized and central modules flagged by the Kilo mandate (mandate §E).
> Metrics measured with a stdlib-only AST-based scanner (LOC, cyclomatic complexity, public classes/functions,
> dependency count, test presence, production callers). Complexity = McCabe-style decision-point count (1 baseline).
>
> Prioritised: oversized and central modules rather than indiscriminately splitting all files.
>
> Status: **measured**.

---

## Measurement method

- **LOC:** non-blank lines.
- **Cyclomatic complexity:** AST decision points (if/for/while/try/except/with/boolop/comprehension) + 1 baseline.
- **Dependency count:** distinct top-level import modules.
- **Has test:** a `tests/` or `tests/integration/` mirror (`test_<name>.py` or `<name>_test.py`) exists.
- **Production callers:** grep for `import <module>` / `from <module>` / `import <symbol>` across `engine/`, `reconstruction/`, `perception/`, `world_ir/`, `worldstore/`, `apps/`.

## Prioritised file inventory

| File | LOC | Cyclomatic complexity | Public classes | Public funcs | Total classes | Total funcs | Deps | Has test | Production callers (up to 8) |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| `engine/pipeline/vertical_slice.py` | 2369 | 465 | 12 | 12 | 12 | 43 | 22 | No | apps/api/jobs.py, apps/api/models.py, apps/cli/main.py, engine/pipeline/progressive.py, stages s04_pose/s05_scale/s06_depth/s07_dense |
| `apps/api/jobs.py` | 2111 | 483 | 3 | 16 | 4 | 67 | 28 | Yes | apps/api/main.py, routes_misc/reconstructions/sessions/worlds, worker.py |
| `apps/api/routes_worlds.py` | 2217 | 539 | 12 | 39 | 12 | 54 | 19 | No | main.py, routes_artifacts/bcf/collab/drawings/export/geo/life_safety |
| `engine/compiler/world_compiler.py` | 1070 | 285 | 4 | 3 | 4 | 5 | 10 | Yes | compiler/__init__.py, orchestrator/universal_pipeline.py, vertical_slice.py |
| `perception/engine_orchestrator.py` | 764 | 152 | 4 | 10 | 4 | 32 | 10 | No | depth/detection/engine/maskrcnn/midas/multi_engine/segmentation adapters |
| `apps/api/evidence_world_stage.py` | 749 | 272 | 1 | 17 | 2 | 27 | 19 | No | apps/api/jobs.py, routes_evidence_world.py, engine/multiengine/run_manifest.py |
| `engine/pipeline/world_delta.py` | 593 | 251 | 0 | 15 | 0 | 21 | 8 | Yes | apps/api/jobs.py, candidate_selection.py, vertical_slice.py |
| `world_ir/validation.py` | 811 | 294 | 3 | 7 | 3 | 30 | 7 | No | evidence_world_stage, jobs.py, routes_evidence_world/routes_worlds, commands/processor, compiler, multiengine/world_scale, universal_pipeline |
| `perception/instances/association.py` | 550 | 131 | 2 | 9 | 2 | 17 | 9 | Yes | vertical_slice.py, object_resolution.py, track_backend.py |
| `engine/pipeline/progressive.py` | 397 | 131 | 4 | 1 | 4 | 7 | 10 | No | apps/api/jobs.py |
| `world_ir/world_v1.py` | 436 | 73 | 5 | 17 | 5 | 19 | 10 | No | evidence_world_stage, render_service, routes_adoptions/completion/evidence_world/procedural/proposals/worlds |
| `engine/orchestrator/universal_pipeline.py` | 460 | 65 | 4 | 4 | 4 | 8 | 19 | Yes | orchestrator/__init__.py |
| `engine/pipeline/stages/s09_perception.py` | 84 | 18 | 1 | 2 | 1 | 3 | 8 | No | stages/ladder.py |
| `engine/pipeline/stages/runner.py` | 123 | 36 | 4 | 5 | 4 | 6 | 6 | No | apps/api/jobs.py |
| `engine/pipeline/stages/registry.py` | 68 | 7 | 2 | 2 | 2 | 2 | 4 | No | jobs.py, stages s01/s02/s03/s11/s13 |
| `perception/entities/taxonomy.py` | 136 | 28 | 1 | 8 | 2 | 11 | 4 | No | entities/__init__, small_objects, switchboard, environment/outdoor, fine_grained/pipeline, fixtures/detector, instances/bounding, spatial_reasoning |

## Responsibility cohesion assessment

| File | Cohesion | Verdict |
|---|---|---|
| `vertical_slice.py` | **Low** — monolithic capture-to-world: pose, scale, depth, layout, perception, mesh, dense, detail, finalize all in one file | Split into per-stage modules (sNN) already exist; vertical_slice should become a thin stage orchestrator composition |
| `jobs.py` | **Low** — job queue + lease + segment orchestrator + reconstruct inner + durables + fencing all in one | Split into `job_queue.py`, `lease.py`, `reconstruct_session.py`, `segments/` |
| `routes_worlds.py` | **Low** — world CRUD, import package, corrections, versions, search all in one 2217-line router | Split into `worlds_crud.py`, `worlds_versions.py`, `worlds_import.py`, `worlds_corrections.py`, `worlds_search.py` |
| `world_compiler.py` | Medium — compile reconstruction to WorldIR + plane/room promotion | Keep, but extract plane/room promotion to `compiler/promotion.py` |
| `world_delta.py` | Medium — snapshot/delta/reconcile/decide/physical_changes | Keep; extract `reconcile_conflicts` to `world_delta/reconcile.py` |
| `evidence_world_stage.py` | Medium — opt-in evidence bridge (detection/depth/lift/association/canonical/persist) | Keep; flag opt-in (RE_EVIDENCE_WORLD) |
| `perception/engine_orchestrator.py` | Medium — multi-engine registration/execution/timeouts | Keep; production caller TBD |
| `world_ir/validation.py` | Medium — WorldIR structural/geometric/provenance gate | Keep; extract architectural constraints to `validation/constraints.py` |
| `progressive.py` | Medium — progressive ladder | Keep; composed by vertical_slice |
| `engine_orchestrator` / `perception/engine_orchestrator.py` | possible name collision / competing orchestrators | Clarify ownership (see IG-1) |

## Error handling

- `vertical_slice.py`: wraps stages in `VerticalSliceError`; partial registration reported degraded; missing intrinsics -> skipped (never guessed). No structured per-stage error schema.
- `jobs.py`: `JobOutcomeError` (deterministic, never retried), `LeaseLost`, `ConcurrentModificationError` retryable; fenced compare-and-swap finalize; cancelled job never erases an adopted version.
- `runner.py`: timed-out stage abandoned (thread not killed; result discarded; never retried) — shared-state mutation risk (IG-4).
- `world_compiler.py`: `CompileInputError`, `WorldValidationGateError` (refused, never silently corrupted).
- `world_delta.py`: conflicts never silently resolved; both hypotheses kept; REJECT leaves HEAD authoritative.
- `validation.py`: errors hard-fail persist; warnings soft.

## Test coverage

- Priority oversized files **without** a dedicated test mirror: `vertical_slice.py`, `routes_worlds.py`, `engine_orchestrator.py`, `evidence_world_stage.py`, `validation.py`, `world_v1.py`, `progressive.py`, `runner.py`, `registry.py`, `taxonomy.py`, `s09_perception.py`.
- Priority files **with** a test mirror: `jobs.py`, `world_compiler.py`, `world_delta.py`, `association.py`, `universal_pipeline.py`.
- Coverage is not a gate; untested oversized central files are a regression risk for refactoring.

## Duplicate logic

- Association: `perception/instances/association.py` (production, gated Hungarian), `engine/multiengine/association.py` (ray-CPA), `engine/multiengine/spatial_intelligence/association.py` (temporal identity), `perception/entities/association.py` (union-find legacy).
- Taxonomy: `perception/entities/taxonomy.py` (ontology.yaml shim), `engine/multiengine/spatial_intelligence/taxonomy.py` (Domain/TaxonomyClass), `perception/style/taxonomy.py`, `perception/attributes/taxonomy.py`.
- Camera/scale contracts: `world_ir/camera.py`, `perception/spatial_inputs.py` CameraInput, `world_ir/coordinates.py` Frame/Transform; `reconstruction/scale.py` ScaleState, `world_ir/schema_v1.py` Measurement, `perception/spatial_inputs.py` ScaleStateInput, `perception/engine_evidence.py` CoordinateFrame.scale_state.

## Production callers

Central modules with many production callers (`jobs.py`, `world_compiler.py`, `validation.py`, `world_v1.py`, `association.py`, `routes_worlds.py`, `vertical_slice.py`, `taxonomy.py`) — contract changes here require coordinated migration.

## Recommended refactor targets (by impact)

1. `vertical_slice.py` (2369 LOC, CC 465) — highest impact; thin orchestrator composition.
2. `jobs.py` (2111 LOC, CC 483) — job lifecycle core.
3. `routes_worlds.py` (2217 LOC, CC 539) — API surface.
4. `world_delta.py` (593 LOC, CC 251) — adoption correctness core.
5. `world_compiler.py` (1070 LOC, CC 285) — WorldIR production.
6. `evidence_world_stage.py` (749 LOC, CC 272) — opt-in bridge.
