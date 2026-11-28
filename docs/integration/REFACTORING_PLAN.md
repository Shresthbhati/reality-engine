# Prioritised Refactoring Plan — Reality Engine V1

> Specific refactoring proposals for `vertical_slice.py`, `jobs.py`, `detection_pipeline.py`,
> `routes_worlds.py`, `association.py` and `taxonomy.py`, each with migration and regression plans.
>
> Mandate rule honored: no architecture rewrite without a demonstrated defect + migration plan.
> Every defect below is demonstrated in `docs/integration/ARCHITECTURE_MAP.md` and
> `docs/integration/INTEGRATION_GAP_REGISTER.md`.
>
> Owner: Principal Systems Architect / Integration Engineer. Coordinate with subsystem owners
> (Sonnet reconstruction, FreeBuff perception, Cline frontend, OpenCode storage/security) before
> touching their owned code.

---

## Ordering principle

Refactor in this order so each step has a working regression net before the next:

1. `taxonomy.py` (leaf contract, no dependents outside perception) — establishes canonical ontology.
2. `association.py` (depends on taxonomy) — unifies entity identity.
3. `vertical_slice.py` (depends on association) — thin stage orchestrator.
4. `jobs.py` (calls vertical_slice/progressive) — job lifecycle core.
5. `routes_worlds.py` (API surface, calls jobs) — API cleanup.
6. `detection_pipeline.py` (perception detection, used by evidence_world path) — perception cleanup.

Each step: **implement -> add/extend regression test -> run existing suite -> coordinate with owner -> merge.**

---

## 1. taxonomy.py — converge ontology contracts

**Demonstrated defect:** Four ontology implementations
(`perception/entities/taxonomy.py`, `engine/multiengine/spatial_intelligence/taxonomy.py`,
`perception/style/taxonomy.py`, `perception/attributes/taxonomy.py`) define inconsistent entity
identity and geometry contracts (mandate defect #9). BUILD_SPEC B03 mandates ONE ontology at
`config/ontology.yaml` read by `perception/ontology.py`.

**Proposal:**
- Make `perception/entities/taxonomy.py` the single shim over `config/ontology.yaml` (it already
  delegates; keep `get/all_classes/children/ancestors/class_for_label/size_plausible`).
- Deprecate `engine/multiengine/spatial_intelligence/taxonomy.py` (`Domain`, `TaxonomyClass`,
  `resolve_taxonomy_class`, `are_semantically_compatible`) — route all callers (detection_pipeline,
  completeness, association, scene_graph, relationships, integration, geometry_validator, benchmarks,
  datasets, intelligence) to `perception/entities/taxonomy.py` via compatibility shims.
- Add `Ontology.validate()` (cycles, components/attach paths exist, "unknown" in every attribute list).

**Migration:** Shims preserve old symbols (`Domain`, `TaxonomyClass`, `resolve_taxonomy_class`) so
existing callers compile; gradually update call sites to `perception.entities.taxonomy`.

**Regression:** Existing `tests/test_december_perception_suite.py`, `tests/test_spatial_intelligence.py`,
`tests/test_perception_sprint7_scale_safety.py` must pass unchanged through the shim layer.

---

## 2. association.py — unify entity identity across views

**Demonstrated defect:** Four association implementations
(`perception/instances/association.py` production gated-Hungarian,
`engine/multiengine/association.py` ray-CPA,
`engine/multiengine/spatial_intelligence/association.py` temporal identity,
`perception/entities/association.py` union-find legacy) with different semantics. Production
vertical_slice uses `perception/instances/association.py`; evidence_world_stage uses
`engine/multiengine/association.py` — results diverge (mandate defect #9).

**Proposal:**
- Keep `perception/instances/association.py` (`resolve_hypotheses`) as the production association
  (gated Hungarian, per-image one-to-one, AMBIGUOUS kept apart, two-view epipolar verification).
- Deprecate `engine/multiengine/association.py` (`associate`, `Candidate`, `PairScore`,
  `AssociatedEntity`) — route evidence_world_stage callers to `resolve_hypotheses` via a compatibility
  shim emitting the same `AssociationDecision` shape.
- Deprecate `engine/multiengine/spatial_intelligence/association.py` ray/temporal machinery to
  benchmarks/experimental only (not on the production path).
- Fix `Observation.confidence` default 0.5 fabricatable default (Contract C7/C8) -> `None` default
  with explicit "not measured".

**Migration:** Compatibility shim maps `associate(...)` -> `resolve_hypotheses(...)` returning
`AssociationDecision` list; evidence_world_stage updated to consume the canonical resolver.

**Regression:** `tests/test_association.py`, `tests/test_instance_association_e05_3.py`,
`tests/test_multiengine_association.py`, `tests/test_multi_view_association.py`,
`tests/test_entity_perception.py`, `tests/test_scale_gate.py` pass against the shim layer.

---

## 3. vertical_slice.py — thin stage-orchestrator composition

**Demonstrated defect:** 2369 LOC, CC 465 (measured). Monolithic capture-to-world
(pose->scale->depth->layout->perception->mesh->dense->detail->finalize). Same `run_*` functions
executed both inside `vertical_slice()` AND by each `sNN_*.py` stage class — stages invoked outside
the stage runner (mandate defect #2).

**Proposal:**
- `vertical_slice.py` becomes a thin composition of the existing `sNN_*.py` stages through
  `engine/pipeline/stages/runner.run_stages()`; removes the inline `run_pose/run_scale/...`
  implementations (they move into their stage classes as `run()`).
- One code path for every stage: `vertical_slice()` == `run_stages(build_stages(...))`.
- Keep public `vertical_slice()` signature as a compatibility wrapper calling `run_stages`.

**Migration:** `progressive.run_progressive` vertical_slice_fn and each `sNN_*.py` stage class updated
to call the canonical stage `run()`; callers (`apps/api/jobs.py`, `apps/cli/main.py`) unchanged.

**Regression:** `tests/test_release_gate.py` golden path (upload->reconstruct->World v1->restart->reload->correct->v2->diff->export),
`tests/test_spatial_stage.py`, `tests/test_universal_expansion_e2e.py` pass.

---

## 4. jobs.py — job lifecycle core

**Demonstrated defect:** 2111 LOC, CC 483 (measured). Job queue + lease + segment orchestrator +
reconstruct inner + durables + fencing all in one file. `_run_reconstruct_session_inner` runs
`run_progressive` directly while `registry.build_stages` also exposes fine stages through `run_stages`
— two hosts drive overlapping logic (mandate defect #2 / IG-1d).

**Proposal:**
- Extract `job_queue.py` (enqueue/process_next/lease/fencing), `lease.py` (worker lease + attempt
  fencing), `reconstruct_session.py` (the `_run_reconstruct_session_inner` logic), `segments/`
  (evidence_resolve, evidence_quality, reconstruct, compile_validate, changeset_commit).
- `reconstruct` segment routes through `run_stages` (single host), reporting fine stages via
  `LadderReport` inside the reconstruct rung.
- `compute_inputs_hash` fixed to hash declared `spec.inputs` only (mandate defect #3 / IG-3).

**Migration:** `apps/api/jobs.py` becomes a thin facade re-exporting from the extracted modules;
`worker.py`, `main.py`, routes unchanged.

**Regression:** `tests/test_durable_job_queue.py`, `tests/test_job_lease_and_fencing.py`,
`tests/test_reliability_journey.py` pass.

---

## 5. routes_worlds.py — API surface cleanup

**Demonstrated defect:** 2217 LOC, CC 539 (measured). World CRUD, import package, corrections,
versions, search all in one router.

**Proposal:**
- Split into `worlds_crud.py` (create/list/get), `worlds_versions.py` (version list/get/diff),
- `worlds_import.py` (package import), `worlds_corrections.py` (correction commands + version fencing),
- `worlds_search.py` (find/nearest/within_radius/contents/container).
- Shared `WorldIn`/`WorldOut` schemas moved to `apps/api/schemas/worlds.py`.

**Migration:** `routes_worlds.py` becomes a router aggregating the sub-routers; URL prefixes unchanged
so no client breakage.

**Regression:** Existing FastAPI route tests (if any) + `tests/integration` golden path pass; no URL
changes.

---

## 6. detection_pipeline.py — perception detection cleanup

**Demonstrated defect:** 2624 LOC (measured), one of the largest files. High-fidelity detection,
tiled inference, mask refinement, cross-engine reconciliation all in one module.

**Proposal:**
- Split into `detection_pipeline/domain_prompts.py`, `tiled_inference.py`, `mask_refinement.py`,
- `duplicate_suppression.py`, `cross_engine_reconciliation.py`, keeping `DetectionPipeline`
  orchestrator class as a thin composition.
- `DetectionBox` dataclass moves to `detection_pipeline/box.py` (shared contract).

**Migration:** `DetectionPipeline` public API preserved; sub-modules importable independently.

**Regression:** Perception benchmark callers (`tests/test_real_perception_benchmark.py`,
`benchmarks/perception/*`) pass against preserved public API.

---

## Cross-cutting fixes applied in parallel (defect fixes, not rewrites)

| Defect | Fix | File |
|---|---|---|
| IG-3 cache keys hash all artifacts | `compute_inputs_hash` hashes declared `spec.inputs` only | `engine/pipeline/stages/runner.py` |
| IG-4 timed-out threads mutate shared state | document abandon contract; post-timeout consistency check on `ctx.state` | `engine/pipeline/stages/runner.py` |
| IG-5 evidence world opt-in | wire `evidence_world_stage.run_for_job` into default path OR document experimental | `apps/api/evidence_world_stage.py`, `apps/api/jobs.py` |
| IG-6 spatial intelligence off path | move `s14_spatial_global` into default profile OR remove flag | `engine/pipeline/stages/registry.py` |
| IG-7 engine availability != success | uniform wrapped reconstruction error type | `reconstruction/orchestrator.py`, `engine/orchestrator/universal_pipeline.py` |
| IG-8 WorldIR/evidence divergence | decide merge (BLOCKED per DESIGN_DECISIONS.md) — document contract | `worldstore/proposals.py` |
| IG-9 inconsistent contracts | taxonomy + association convergence (steps 1-2) | multiple |
| IG-10 path divergence | align API/CLI/benchmark subsets | `apps/api/jobs.py`, `apps/cli/main.py` |
| IG-16c/d fabricatable confidence defaults | `Observation.confidence` default None | `world_ir/observations.py`, `perception/engine_evidence.py` |

---

## Migration & regression principles (apply to every step)

1. **No silent behavior change:** public symbols preserved via shims; old call sites compile unchanged.
2. **Tests before refactor:** each target's existing test suite must pass before and after.
3. **Coordinate with owners:** reconstruction (Sonnet), perception (FreeBuff), frontend (Cline),
   storage/security (OpenCode) sign off on contract changes in their domain.
4. **Registry/docs updated in same change:** ARCHITECTURE_MAP, CONTRACT_REGISTRY, GAP REGISTER updated
   when a module is deprecated or a contract converges.
5. **Rollback:** each PR is revertible; shims keep the old path until the new one is verified.
