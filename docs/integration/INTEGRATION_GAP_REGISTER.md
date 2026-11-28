# Integration Gap Register — Reality Engine V1

> Detects dead modules, unreachable engines, duplicate orchestrators, incompatible contracts, feature flags that disable
> intended release functionality, stages with no output consumer, outputs with no producer, and data that disappears
> between stages. Also builds the basis for the automated production-path reachability report.
>
> Part of the Kilo mandate (mandate §E-F). Owner: Principal Systems Architect / Integration Engineer.
> Status: **observed from code**.

---

## IG-1 — Competing / duplicate orchestrators (architectural gap)

| # | Module | Role | Production caller? | Gap |
|---|---|---|---|---|
| IG-1a | `engine/pipeline/vertical_slice.py` (`vertical_slice()`) | Monolithic capture-to-world (pose->scale->depth->layout->perception->mesh->dense->detail->finalize) | YES — via `progressive.run_progressive` vertical_slice_fn and by sNN stage classes directly | Core production path but also invoked outside stage runner (defect #2) |
| IG-1b | `engine/pipeline/progressive.py` (`run_progressive()`) | Progressive ladder: multi-view rung (calls vertical_slice_fn) falling back to single-image bootstrap | YES — `apps/api/jobs.py` seg_reconstruct | Overlaps vertical_slice; fine stages run both inside ladder and as standalone sNN classes |
| IG-1c | `engine/orchestrator/universal_pipeline.py` (`UniversalPipeline.reconstruct()`) | SECOND full implementation: ingest -> reconstruct -> scale -> compile -> materials -> spatial -> style -> plausibility -> completeness -> georeference -> validate -> persist/export | NO — STATUS.md S5-05: UNWIRED; not called by any production job | Dead parallel implementation of the same chain; "UNWIRED" |
| IG-1d | `engine/pipeline/stages/registry.py` (`build_stages()`) | Coarse stage runner for jobs.py (P106/P108/P110 profile) | YES — `apps/api/jobs.py` drive() | Runs coarse segments; fine stages reported via LadderReport — two hosts drive overlapping logic |

**Required action:** Declare `universal_pipeline.py` the single canonical orchestrator OR remove it; collapse `vertical_slice` + `progressive` + `registry.build_stages` into one stage DAG executed by `run_stages`.

---

## IG-2 — Stages invoked outside the stage runner (defect #2)

| # | Stage/function | Called by | Gap |
|---|---|---|---|
| IG-2a | `vertical_slice.run_pose/run_scale/run_frame/run_depth/run_layout/run_perception/run_mesh/run_dense/run_detail/run_finalize` | `vertical_slice()` (monolithic) AND each `engine/pipeline/stages/sNN_*.py` class | Same `run_*` functions execute inside and outside `run_stages`; cache/checkpoint semantics differ between the two paths |
| IG-2b | `run_progressive()` | `vertical_slice()` (as vertical_slice_fn) AND `apps/api/jobs.py` seg_reconstruct directly | Progressive ladder bypasses the coarse stage runner's checkpoint/resume for the reconstruct rung |

**Required action:** Route all stage execution through `run_stages`; make `vertical_slice()` a thin composition of stages, not a parallel runner.

---

## IG-3 — Cache keys hash all accumulated artifacts (defect #3)

**Module:** `engine/pipeline/stages/runner.py` `run_stages()` — `compute_inputs_hash(spec.version, ctx.artifacts, ctx.params)` hashes `ctx.artifacts` (everything accumulated so far) instead of declared `spec.inputs`.

**Gap:** `StageSpec.inputs` (declared logical artifact names) is ignored by the hash; lineage tracked separately in `LineageGraph` but not used for cache keys. A change to an unrelated earlier artifact invalidates all downstream cache entries.

**Required action:** `compute_inputs_hash` should hash `spec.inputs` resolved through `ctx.artifacts` (declared inputs only), not the whole `ctx.artifacts`.

---

## IG-4 — Timed-out stage threads continue mutating shared state (defect #4)

**Module:** `engine/pipeline/stages/runner.py` `_run_with_deadline()` — `ex.shutdown(wait=False, cancel_futures=True)`; the straggler thread keeps running and may mutate shared `ctx.state` / artifacts.

**Gap:** Python cannot kill a thread; a timed-out stage is "abandoned" but its late result is only discarded — side effects on shared mutable objects persist. The `StageTimeout` docstring acknowledges this.

**Required action:** Document the abandon contract; make stage `run()` read-only on shared state where possible, or add a post-timeout consistency check on `ctx.state`.

---

## IG-5 — Evidence-world integration is opt-in (defect #5)

**Module:** `apps/api/evidence_world_stage.py` — gated by `RE_EVIDENCE_WORLD=1` (`enabled()`).

**Gap:** Gemini's detection/segmentation/spatial lifting/association -> canonical -> WorldIR bridge runs ONLY when the env flag is on. It is NOT on the canonical production path (default API job skips it). `readiness().quality` can never reach READY through the default path for evidence-rich worlds.

**Required action:** Wire `evidence_world_stage.run_for_job` into the default pipeline (or explicitly document it as experimental/off the release path per RELEASE.md "What does not ship in 1.0").

---

## IG-6 — Spatial intelligence not connected to canonical production path (defect #6)

| # | Module | Gap |
|---|---|---|
| IG-6a | `engine/pipeline/stages/s14_spatial_global.py` | Registered ONLY in non-default profile `interior_survey_spatial`; selected only when `RE_SPATIAL_GLOBAL` is on or `_sg.enabled()`. Default profile `interior_survey_standard` omits it. |
| IG-6b | `perception/engine_orchestrator.py` (`MultiEngineOrchestrator`) | Real multi-engine orchestration exists but no production caller wires it into the canonical job; `evidence_world_stage` runs it separately (opt-in). |
| IG-6c | `perception/spatial_inputs.py` (`spatial_inputs.v1`) | Producer/consumer contract exists but the default job's `_spatial_inputs_provider` is only active when `spatial_enabled`. |

**Required action:** Move `spatial_global` into the default profile (or remove the flag) and wire the multi-engine orchestrator into the canonical path.

---

## IG-7 — Engine availability != successful execution (defect #7)

| # | Module | Gap |
|---|---|---|
| IG-7a | `reconstruction/orchestrator.py` `ReconstructionOrchestrator.run()` | Raises `ReconstructionOrchestrationError` when all backends fail — honest. |
| IG-7b | `engine/orchestrator/universal_pipeline.py` `_run_reconstruction()` | Lets COLMAP exceptions propagate (no orchestrator wrapper) — inconsistent error surface. |
| IG-7c | `vertical_slice.run_pose()` | Wraps in `VerticalSliceError` — consistent. |

**Required action:** Make error surface uniform: all reconstruction entry points raise the same wrapped error type; availability != success.

---

## IG-8 — WorldIR vs evidence-world divergence (defect #8)

**Modules:** `worldstore/store.py` (canonical WorldStore versions) vs `apps/api/evidence_world_stage.py` (`<WORLDSTORE_ROOT>/evidence-world/<team>/<world>/<job>/worldstore`, one immutable version per run).

**Gap:** Two WorldIRs per job. Connection only via opt-in `POST /api/worlds/{id}/evidence-world/{run}/propose` then `/api/worlds/{id}/proposals/{pid}/accept` (frame/unit gate: HEAD must be exactly the version the same job committed beside this run). Not physically validated on real data beyond run 14 (DESIGN_DECISIONS.md section 8).

**Required action:** Decide whether the evidence world merges into HEAD (currently BLOCKED per DESIGN_DECISIONS.md "MERGE BLOCKED") or remains permanently separate; document the contract.

---

## IG-9 — Inconsistent entity/geometry/evidence contracts (defect #9)

### IG-9a — Association (≥4 implementations)

| Module | Key symbol(s) | Production caller(s) |
|---|---|---|
| `perception/instances/association.py` | `resolve_hypotheses` | `vertical_slice._perception_stage` (PRODUCTION) |
| `perception/entities/association.py` | `associate`, `EntityObservation` | tests only (`test_entity_perception`, `test_universal_expansion_e2e`, `test_task_13_9_mast3r_dinov2`) |
| `engine/multiengine/association.py` | `Candidate`, `PairScore`, `associate`, `AssociatedEntity` | `spatial_stage.py`, `evidence_flow.py`, `canonical.py`, `identity.py`, `pair_scoring*.py`, `conflicts.py`, `multi_view_association.py` |
| `engine/multiengine/spatial_intelligence/association.py` | `CrossViewRayCandidate`, `TemporalIdentityTracker`, `PhysicalMergedEntity` | `real_data_benchmark`, `integration.py`, `active_capture.py`, benchmarks |

**Gap:** Four association implementations with different semantics (gated Hungarian vs union-find vs ray-CPA triangulation). Production vertical_slice uses `perception/instances/association.py`; evidence_world_stage uses `engine/multiengine/association.py`. Results can diverge.

### IG-9b — Taxonomy (≥4 implementations)

| Module | Key symbol(s) | Production caller(s) |
|---|---|---|
| `perception/entities/taxonomy.py` | `EntityClass`, `_R`, `get()` (reads config/ontology.yaml) | perception modules, benchmarks |
| `engine/multiengine/spatial_intelligence/taxonomy.py` | `Domain`, `TaxonomyClass`, `resolve_taxonomy_class` | detection_pipeline, completeness, association, scene_graph, relationships, integration, geometry_validator, benchmarks, datasets, intelligence |
| `perception/style/taxonomy.py` | `DesignStyle`, `StyleProfile` | `perception/style/analyzer.py` |
| `perception/attributes/taxonomy.py` | attribute taxonomy | perception attributes |

**Gap:** Multiple ontologies. BUILD_SPEC B03 mandates ONE ontology at `config/ontology.yaml` read by `perception/ontology.py`; `perception/entities/taxonomy.py` is the shim. `engine/multiengine/spatial_intelligence/taxonomy.py` is a parallel ontology not yet converged.

### IG-9c — Camera/scale contracts (≥3 representations)

- Camera frame: `world_ir/camera.py` PinholeCamera vs `perception/spatial_inputs.py` CameraInput vs `world_ir/coordinates.py` Frame/Transform.
- Scale state: `reconstruction/scale.py` ScaleState vs `world_ir/schema_v1.py` Measurement vs `perception/spatial_inputs.py` ScaleStateInput vs `perception/engine_evidence.py` CoordinateFrame.scale_state.

---

## IG-10 — Production path divergence (defect #10)

| Path | Entry point | Pipeline subset executed | Status |
|---|---|---|---|
| API production job | `apps/api/jobs.py` `_run_reconstruct_session_inner` | evidence_resolve, evidence_quality, reconstruct (progressive ladder: pose/scale/depth/layout/perception/mesh/dense/detail/finalize via LadderReport), quality_report, compile_validate, changeset_commit | ACTIVE |
| API evidence world | `evidence_world_stage.run_for_job` | detection pipeline -> depth/lift -> spatial lift/association -> canonical entities -> WorldIR bridge -> persist_world | OPT-IN (RE_EVIDENCE_WORLD=1) |
| CLI | `apps/cli/main.py` `reality compile` | vertical_slice() directly, with REALITY_TEST_BACKEND seam | SECONDARY |
| Benchmark | `tests/integration/test_release_gate.py` | upload -> reconstruct -> World v1 -> restart -> reload -> correct -> v2 -> diff -> export | GOLDEN |
| Universal pipeline | `engine/orchestrator/universal_pipeline.py` | Full second implementation | UNWIRED |

**Gap:** API, CLI, and benchmark execute different subsets. The benchmark golden path is the most complete; the default API job omits the evidence world and spatial intelligence.

---

## IG-11 — Stages with no output consumer

| # | Stage | Output | Consumer? | Gap |
|---|---|---|---|---|
| IG-11a | `s12_quality_report` | mesh validity metrics (mesh_*) | Job report only; no consumer acts on FAIL/WARN | `required=True` but never fails the job — "never a success-shaped PASS"; grading unaffected |
| IG-11b | `s14_spatial_global` | spatial_world hash + metrics | Default profile omits it; opt-in profile has consumer | Unreachable in default path |
| IG-11c | `s06_depth` (when skipped) | metric_maps | Perception only runs if depth maps survived | Skipped depth silently disables perception — recorded, not hidden |

---

## IG-12 — Outputs with no producer

| # | Output | Declared producer? | Actual producer | Gap |
|---|---|---|---|---|
| IG-12a | `s11_compile_validate.requires_outputs=("world_ir",)` | seg_compile_validate (jobs.py) | seg_compile_validate stores content-addressed world | OK — has producer |
| IG-12b | `universal_pipeline` stages | UniversalPipeline | none (UNWIRED) | All outputs have no production producer |
| IG-12c | `evidence_world_stage` entities | evidence_world_stage | itself (opt-in) | OK but off the default path |

---

## IG-13 — Data that disappears between stages

| # | Data | Producer | Consumer | Gap |
|---|---|---|---|---|
| IG-13a | Per-camera support / reconstruction diagnostics | `vertical_slice.run_finalize` (reconstruction_diagnostics) | Job report only (`prog.stage_facts.get("reconstruction_diagnostics")`) | Not fed back into correction/adoption decision |
| IG-13b | Correction plan | `vertical_slice.run_finalize` (correction_plan) | Job report only | "recommendation only; nothing was applied" — never consumed |
| IG-13c | Depth maps | `vertical_slice._depth_stage` (metric_maps) | Perception stage (same maps) | OK — one owner |
| IG-13d | Dense points | `_dense_mvs_stage` / `_plane_sweep_stage` (appended to result.points) | `_fusion_stage` (plural-source fusion) | OK |

---

## IG-14 — Feature flags that disable intended release functionality

| # | Flag | Default | Effect | Release relevance |
|---|---|---|---|---|
| IG-14a | `RE_EVIDENCE_WORLD` | OFF | Disables evidence_world_stage (Gemini bridge) | Mandate defect #5; RELEASE.md says video/SAM/object tracking "not on the live API path" |
| IG-14b | `RE_SPATIAL_GLOBAL` | OFF | Omits `spatial_global` from default profile | Mandate defect #6; spatial intelligence off the canonical path |
| IG-14c | `REALITY_DENSE_AUTO` | ON (but `dense_mvs_enabled=False`) | Evidence-gated dense escalation off by default | Level 3 dense not part of default product path |
| IG-14d | `REALITY_OPEN_VOCAB` | unset -> "grounded" | Grounding DINO vs Mask R-CNN | Open-vocab furniture — RELEASE.md: not on live API path |
| IG-14e | `REALITY_TEST_BACKEND` | unset | Injects synthetic backend (tests only) | Test seam; refused outside REALITY_ENV=test |
| IG-14f | `REALITY_LAYOUT_MODELS` / `REALITY_OPEN_VOCAB` = "0" | unset | Opt-out for learned layout / open-vocab | Learned single-image layout — implemented locally, opt-out |

---

## IG-15 — Dead / unreachable modules (candidates)

Candidates requiring verification before removal:
- `engine/orchestrator/universal_pipeline.py` — UNWIRED, no production caller (IG-1c)
- `perception/entities/association.py` — no production caller, tests only (IG-9a)
- `perception/style/taxonomy.py` — used only by `perception/style/analyzer.py`; verify production usage
- Exporters under `exporters/` (usd/citygml/cityjson/unreal/blender/godot/habitat/ros/sumo/citydb/tiles3d/mesh/gis/report/profiles) — DESIGN_DECISIONS.md: "structure-tested, but OPTIONAL_DEFERRED: no further work until a user of one of those ecosystems needs it, and nothing claims they load"

**Required action:** Run a reachability analysis (see automated report below) before deleting any candidate.

---

## IG-16 — Incompatible contracts between producer and consumer

| # | Producer | Consumer | Incompatibility |
|---|---|---|---|
| IG-16a | `perception/instances/association.py` `resolve_hypotheses` (Hungarian, per-image one-to-one, AMBIGUOUS kept apart) | `vertical_slice._perception_stage` | OK (same module family) |
| IG-16b | `engine/multiengine/association.py` `associate` | `evidence_world_stage.run_for_job` | Different association semantics than production vertical_slice — results diverge |
| IG-16c | `perception/engine_evidence.py` `ObservationRecord.confidence` default 0.5 | Contract says None = not measured | Default 0.5 fabricates confidence — violates contract rule |
| IG-16d | `world_ir/observations.py` `Observation.confidence` default 0.5 | Same issue | Fabricatable default |

---

## IG-17 — Reachability summary (to be automated)

The automated production-path reachability report (Section IG-18) must verify:
1. Every production API entry point reaches a persisted WorldStore version.
2. `universal_pipeline.py` is unreachable from production code.
3. `s14_spatial_global` is unreachable from the default profile.
4. `evidence_world_stage` is reachable only behind `RE_EVIDENCE_WORLD=1`.
5. All `requires_outputs` are produced by their stage.
6. Every `outputs` entry has a downstream consumer or is persisted.

---

## IG-18 — Automated production-path reachability report (to build)

Build a Python script (`scripts/reachability_report.py`) that:
- Parses `apps/api/jobs.py` to extract the stage sequence and `requires_outputs`.
- Parses `engine/pipeline/stages/registry.py` `PIPELINE_PROFILES`.
- Greps for imports/calls to `universal_pipeline`, `engine_orchestrator.MultiEngineOrchestrator`, `evidence_world_stage.run_for_job`.
- Verifies each stage's `outputs` are consumed by a later stage or persisted (WorldStore / `outputs["world_ir"]` / `outputs["observations"]`).
- Emits a JSON/Markdown report: reachable stages, unreachable engines, outputs without consumers, stages without producers, flag-gated functionality.

**Output location:** `docs/integration/reachability_report.md` (generated) + `scripts/reachability_report.py`.

---

## Required fixes (prioritised)

| Priority | Fix | Gap refs |
|---|---|---|
| P0 | Declare one canonical orchestrator; remove/repoint `universal_pipeline.py` | IG-1 |
| P0 | Route all stage execution through `run_stages` (remove `run_*` dual invocation) | IG-2 |
| P0 | Fix `compute_inputs_hash` to hash declared `spec.inputs` only | IG-3 |
| P1 | Add post-timeout shared-state consistency check | IG-4 |
| P1 | Wire `evidence_world_stage` into default path OR document as experimental | IG-5 |
| P1 | Move `spatial_global` into default profile OR remove flag | IG-6 |
| P1 | Unify reconstruction error surface | IG-7 |
| P2 | Decide evidence-world merge (BLOCKED per DESIGN_DECISIONS.md) | IG-8 |
| P2 | Converge association implementations to one | IG-9a |
| P2 | Converge taxonomy implementations to `config/ontology.yaml` | IG-9b |
| P2 | Converge camera/scale contracts | IG-9c |
| P2 | Align API/CLI/benchmark pipeline subsets | IG-10 |
| P3 | Remove dead modules after reachability verification | IG-11/IG-12/IG-15 |
| P3 | Fix fabricatable confidence defaults (0.5) | IG-16c/d |
