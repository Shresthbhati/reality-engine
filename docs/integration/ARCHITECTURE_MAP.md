# Architecture Map — Reality Engine V1

> Generated as part of the Kilo mandate: "make all Reality Engine subsystems operate as one coherent product."
>
> Source of truth: inspected files under `engine/`, `reconstruction/`, `perception/`, `world_ir/`, `worldstore/`,
> `apps/api/`, `apps/cli/`, `frontend/`, `exporters/`, `tests/integration/` (see Section "Inspected surface").
>
> Status: **observed**, not aspirational. No rewrite without a demonstrated defect + migration plan.

## 1. Module dependency graph (key subsystems only)

```
Tier 0 — canonical data model / provenance / camera math
├── provenance/          (Provenance, Uncertainty)                no production deps
├── uncertainty/         (UncertainScalar)
├── engine/core/         (units, rng, math, device)              Tier 0-1 foundations
├── world_ir/schema_v1.py (Entity, Geometry, Observation, Relationship, Measurements)
├── world_ir/observations.py  (Observation, AssociationDecision, LEDGER_KEY)
├── world_ir/camera.py  (PinholeCamera, CameraIntrinsics, CameraExtrinsics, Vec3)
└── engine/pipeline/stages/   (base.py StageSpec/StageResult/StageContext/Stage contract)

Tier 1 — reconstruction (production path)
├── reconstruction/orchestrator.py      ReconstructionOrchestrator — validates, selects, executes backends
├── reconstruction/backend/interface.py  IReconstructionBackend contract
├── reconstruction/backend/colmap_backend.py  ColmapReconstructionBackend (the real backend)
├── reconstruction/backend/fake.py        FakeReconstructionBackend (test seam)
├── reconstruction/colmap_session.py      ColmapSession — persistent per-world COLMAP workspace
├── reconstruction/scale/                 ScaleReference, ScaleState, anchor_metric_scale, unscaled
├── reconstruction/frame.py               frame canonicalization (canonicalize_like_head, refine_up_from_planes)
├── reconstruction/scene_scale.py         working_unit, scale refinement
├── reconstruction/sfm/                   COLMAP sparse SfM
├── reconstruction/mvs/                   dense MVS (patch_match_stereo, stereo_fusion)
├── reconstruction/fusion/                voxel cloud, measurements, plural-source fusion
├── reconstruction/meshing/               MeshData, poisson mesher
├── reconstruction/depth_to_points.py     depth map -> points, metricize_all, trim_to_scene
└── reconstruction/diagnostics.py         analyze reconstruction results

Tier 2 — pipeline orchestration (P106/P108/P110)
├── engine/pipeline/stages/base.py          StageSpec / StageResult / StageContext / Stage protocol
├── engine/pipeline/stages/registry.py      PIPELINE_PROFILES (interior_survey_standard, interior_survey_spatial)
├── engine/pipeline/stages/runner.py        run_stages — sequential runner, checkpoint reuse, timeouts, admission
├── engine/pipeline/stages/ladder.py        Ladder — fine stages run inside progressive.run_progressive's multi-view rung
├── engine/pipeline/stages/content_store.py put_bytes/get_bytes/put_array/... content-addressed storage
├── engine/pipeline/stages/lineage.py       LineageGraph — producer/consumer/reads tracking
├── engine/pipeline/stages/facts.py         facts_status / combine — truthful StageStatus mapping
├── engine/pipeline/stages/run_record.py    build_run_record — RE-MLOPS-000001 run record
├── engine/pipeline/stages/s01_evidence.py   EvidenceResolve (host segment wrapper)
├── engine/pipeline/stages/s02_quality.py    EvidenceQuality (host segment wrapper)
├── engine/pipeline/stages/s03_reconstruct.py Reconstruct (host segment wrapper — runs progressive ladder)
├── engine/pipeline/stages/s04_pose.py   Pose — vertical_slice.run_pose, committed_intrinsics
├── engine/pipeline/stages/s05_scale.py   Scale — vertical_slice.run_scale + run_frame
├── engine/pipeline/stages/s06_depth.py   Depth — vertical_slice.run_depth, refine_frame_from_dense
├── engine/pipeline/stages/s07_dense.py   Dense — vertical_slice.run_dense
├── engine/pipeline/stages/s08_mesh.py    Mesh — vertical_slice.run_mesh
├── engine/pipeline/stages/s09_perception.py Perception — vertical_slice.run_perception, observation-table persistence
├── engine/pipeline/stages/s10_layout.py   Layout — vertical_slice.run_layout
├── engine/pipeline/stages/s10_detail.py   Detail — vertical_slice.run_detail
├── engine/pipeline/stages/s10_finalize.py Finalize — vertical_slice.run_finalize
├── engine/pipeline/stages/s11_compile_validate.py CompileValidate (host segment wrapper)
├── engine/pipeline/stages/s12_quality_report.py QualityReport — mesh validity (trimesh)
├── engine/pipeline/stages/s13_changeset.py ChangesetCommit (host segment wrapper)
├── engine/pipeline/stages/s14_spatial_global.py SpatialGlobal (opt-in global spatial optimization)
├── engine/pipeline/vertical_slice.py      vertical_slice() — monolithic capture-to-world (pose->scale->depth->layout->perception->mesh->dense->detail->finalize)
├── engine/pipeline/progressive.py         run_progressive() — progressive ladder (multi-view, falling back to single-image bootstrap)
├── engine/pipeline/world_delta.py         world_delta — candidate adoption protocol (snapshot/compute_delta/reconcile_conflicts/decide/physical_changes)
├── engine/pipeline/dense_gate.py          dense_readiness — Level 3 dense justification criteria
└── engine/pipeline/spatial_continuity.py  reconcile — geometry-aware structural continuity between versions

Tier 3 — WorldIR / WorldStore persistence
├── world_ir/world_v1.py       WorldIR — canonical semantic world (entities, geometries, branches, metadata)
├── world_ir/validation.py     validate_world_ir — structural + geometric + provenance gate
├── world_ir/artifact_store.py ArtifactStore — content-addressed blob store (STORAGE_ROOT)
├── world_ir/geometry_data.py  MeshData / PointCloudData — runtime geometry payloads
├── world_ir/diff.py           diff_worlds — world diffing
├── world_ir/entity_reid.py    carry_identity — entity identity carry-over across versions
├── worldstore/store.py        WorldStore — immutable versioned world persistence (save_version/verify_version/load_version)
├── worldstore/lifecycle.py    adoption intent / quarantine / reconcile_adoptions
├── worldstore/proposals.py    evidence-world -> committed world proposal protocol
└── worldstore/tiles.py        tile queries

Tier 4 — perception / multi-engine intelligence (opt-in / experimental)
├── perception/engine_orchestrator.py    MultiEngineOrchestrator — engine registration, dependency/VRAM-aware execution, timeouts
├── perception/engine_evidence.py          UnifiedEngineOutput / ObservationRecord / ModelIdentity / CoordinateFrame
├── perception/engine_matrix_bridge.py     imports engine observations into the run's Evidence Matrix
├── perception/spatial_inputs.py           spatial_inputs.v1 contract producer/consumer registry
├── perception/multiview/room_associator.py
├── perception/fine_grained/               FineGrainedPerceptionEngine
├── perception/detection/                  detection backends (maskrcnn, grounding_dino, florence2)
├── perception/depth/                      depth backends (depth_anything, midas)
├── perception/geometry/                   plane detection, orientation, room_box
├── perception/architecture/               room_graph, corridor, windows, stairs, wall_slabs, room_join
├── perception/instances/association.py    resolve_hypotheses — gated Hungarian association (production vertical_slice path)
├── perception/instances/object_resolution.py merge_hypotheses / fuse_overlapping_candidates
├── perception/entities/taxonomy.py        ontology-backed entity taxonomy shim (reads config/ontology.yaml)
└── perception/ontology.py                 Ontology / OntologyNode — canonical ontology (BUILD_SPEC B03)

Tier 5 — applications (production entry points)
├── apps/api/jobs.py            DurableJob worker: RECONSTRUCT_SESSION handler (seg_evidence_resolve -> seg_changeset_commit)
├── apps/api/routes_worlds.py   /api/worlds REST (CRUD, import package, corrections, versions)
├── apps/api/routes_evidence_world.py /api/worlds/{id}/evidence-world/* (opt-in RE_EVIDENCE_WORLD)
├── apps/api/routes_query.py    /api/worlds/{id}/find, /nearest, /within_radius, /contents, /container
├── apps/api/worldstore_service.py commit_version / get_store / nearest_report / list_scale_references
├── apps/cli/main.py            CLI entry point (reality compile / verify / export)
└── apps/cli/api_bridge.py      CLI <-> API bridge

Tier 6 — exports
├── exporters/ifc_bridge.py     IFC export
├── exporters/gltf/             glTF export
├── exporters/floorplan.py      PDF/DXF floor plan export
├── exporters/usd/, /citygml/, /cityjson/, /unreal/, /blender/, /godot/, /habitat/, /ros/, /sumo/, /citydb/, /tiles3d/, /mesh/, /gis/, /report/, /profiles/
```

## 2. Ownership (who owns what)

| Subsystem | Owner (per mandate) | Integration owner (this role) |
|---|---|---|
| Sonnet's reconstruction algorithms | Sonnet | coordinate contracts only |
| FreeBuff's perception algorithms | FreeBuff | coordinate contracts only |
| Cline's frontend | Cline | coordinate contracts only |
| OpenCode's storage/security | OpenCode | coordinate contracts only |
| Cross-system contracts / stage orchestration | — | **this role** |
| WorldIR canonical model | shared | coordinate |
| WorldStore persistence | shared | coordinate |
| pipeline stages (engine/pipeline/stages) | shared | coordinate |
| vertical_slice.py | shared | refactor + integration |
| jobs.py (API job worker) | shared | refactor + integration |

## 3. Duplicated abstractions (exact overlaps the mandate flagged)

### 3.1 Competing orchestrators / pipelines (≥4)

1. **`engine/pipeline/vertical_slice.py`** (`vertical_slice()`) — the monolithic capture-to-world call used by `progressive.run_progressive` and the fine-stage ladder.
2. **`engine/pipeline/progressive.py`** (`run_progressive()`) — progressive ladder: multi-view rung (calls `vertical_slice_fn`) falling back to single-image bootstrap.
3. **`engine/orchestrator/universal_pipeline.py`** (`UniversalPipeline.reconstruct()`) — "Universal Intelligence Expansion Epic E01-1": ingestion -> reconstruction -> scale -> compile -> materials -> spatial reasoning -> style -> plausibility -> completeness -> georeferencing -> validation -> persist/export. A fully parallel, second implementation of the same chain.
4. **`engine/pipeline/stages/registry.py`** (`build_stages()`) — coarse stage runner for `apps/api/jobs.py` (P106/P108/P110 profile `interior_survey_standard`), wiring segments through `run_stages`.

### 3.2 Stage invocation outside the stage runner

- `vertical_slice.run_pose/run_scale/run_frame/run_depth/run_layout/run_perception/run_mesh/run_dense/run_detail/run_finalize` are called directly from `vertical_slice()` AND from each `sNN_*.py` stage class. The same `run_*` functions execute both inside and outside `run_stages` — the exact defect "stages may be invoked outside the stage runner."
- `apps/api/jobs.py::_run_reconstruct_session_inner` runs `run_progressive` directly (segments), while `registry.build_stages` also exposes `Reconstruct`/`LadderReport`/`CompileValidate` etc. through `run_stages`. Two different hosts drive overlapping logic.

### 3.3 Competing association / taxonomy modules (≥4 each)

**Association:**
- `perception/instances/association.py` — `resolve_hypotheses` (used by `vertical_slice.run_perception`, production).
- `perception/entities/association.py` — `associate`, `EntityObservation` (used by tests/test_entity_perception).
- `engine/multiengine/association.py` — `Candidate`, `PairScore`, `associate`, `AssociatedEntity` (used by `spatial_stage.py`, `evidence_flow.py`, `canonical.py`, `identity.py`, `pair_scoring*.py`, `conflicts.py`, `multi_view_association.py`).
- `engine/multiengine/spatial_intelligence/association.py` — `CrossViewRayCandidate`, triangulation, `TemporalIdentityTracker`, `PhysicalMergedEntity` (heavy, 1697 lines; used by real_data_benchmark, integration.py, active_capture.py).

**Taxonomy:**
- `perception/entities/taxonomy.py` — `EntityClass`, `_R`, `get()`, `all_classes()` (reads `config/ontology.yaml`).
- `perception/style/taxonomy.py` — `DesignStyle`, `StyleProfile`, `STYLE_PROFILES`.
- `perception/attributes/taxonomy.py` — attribute taxonomy.
- `engine/multiengine/spatial_intelligence/taxonomy.py` — `Domain`, `TaxonomyClass`, `resolve_taxonomy_class`, `are_semantically_compatible` (used by detection_pipeline, completeness, association, scene_graph, relationships, integration, geometry_validator, benchmarks, datasets, intelligence).

## 4. Inconsistent entity/geometry/evidence contracts (≥3)

| Contract | Definitions | Produced by | Consumed by |
|---|---|---|---|
| Camera frame | `world_ir/camera.py` PinholeCamera + `world_ir/coordinates.py` Frame/Transform | `reconstruction/frame.py`, `vertical_slice.run_frame` | compiler, depth, perception, spatial_inputs |
| Entity identity | `world_ir/schema_v1.py` Entity.id + `world_ir/entity_reid.py` carry_identity | compiler, changeset_commit | diff, WorldStore versions |
| Ontology | `perception/entities/taxonomy.py` (reads config/ontology.yaml) vs `engine/multiengine/spatial_intelligence/taxonomy.py` Domain/TaxonomyClass | perception | detection, association, benchmarks |
| Geometry refs | `world_ir/schema_v1.py` Geometry (id, type, bounds, data_uri/data_hash, provenance) | `engine/compiler/world_compiler.py`, `_fusion_stage`, `_mesh_stage` | WorldStore, exporters, viewer |
| Evidence observations | `world_ir/observations.py` Observation vs `perception/engine_evidence.py` ObservationRecord vs `world_ir/observation_table.py` ObservationTable | perception stages, evidence flow | compiler, evidence_world_stage, readiness |
| Scale states | `reconstruction/scale.py` ScaleState vs `world_ir/schema_v1.py` Measurement vs `perception/spatial_inputs.py` ScaleStateInput | scale stage, jobs.py | spatial_global, readiness, export |

## 5. Production path divergence (API vs CLI vs benchmark)

| Path | What it runs | Status |
|---|---|---|
| **API production job** (`apps/api/jobs.py`) | segments evidence_resolve -> evidence_quality -> reconstruct (run_progressive) -> compile_validate -> changeset_commit; fine stages (pose/scale/depth/layout/perception/mesh/dense/detail/finalize) reported via `ladder.LadderReport` inside the reconstruct rung | **active production path** |
| **Vertical slice direct** (`engine/pipeline/vertical_slice.py`) | pose -> scale -> depth -> layout -> perception -> mesh -> dense -> detail -> finalize in one call | used by progressive rung + tests |
| **Universal pipeline** (`engine/orchestrator/universal_pipeline.py`) | full second implementation (ingest -> reconstruct -> scale -> compile -> materials -> spatial -> style -> plausibility -> completeness -> georeference -> validate -> persist/export) | **UNWIRED** (per STATUS.md S5-05); not called by any production job |
| **CLI** (`apps/cli/main.py`) | `reality compile` with `REALITY_TEST_BACKEND` seam | secondary |
| **Benchmark** (`tests/integration/test_release_gate.py`) | upload -> reconstruct -> World v1 -> restart -> reload -> correct -> v2 -> diff -> export | golden path |

## 6. Known architectural defects (mapped from mandate §B)

| # | Defect | Evidence |
|---|---|---|
| 1 | Multiple orchestration/pipeline abstractions compete | vertical_slice.py vs progressive.py vs universal_pipeline.py vs registry.build_stages (Section 3.1) |
| 2 | Stages invoked outside the stage runner | vertical_slice.run_* called both by vertical_slice() and by sNN_*.py stage classes (Section 3.2) |
| 3 | Cache keys hash all accumulated artifacts instead of declared stage inputs | `runner.py` `compute_inputs_hash(spec.version, ctx.artifacts, ctx.params)` hashes `ctx.artifacts` (everything accumulated so far), not declared `spec.inputs` |
| 4 | Timed-out stage threads may continue mutating shared state | `runner.py` `_run_with_deadline` abandons the thread (`ex.shutdown(wait=False, cancel_futures=True)`) — the straggler keeps running and may mutate shared state; `StageTimeout` docstring acknowledges this |
| 5 | Evidence-world integration is opt-in | `evidence_world_stage.enabled()` gated by `RE_EVIDENCE_WORLD=1`; not on the canonical production path (Section 3.2 / STATUS.md) |
| 6 | Spatial intelligence may not be connected to canonical production path | `spatial_global` stage only in non-default profile `interior_survey_spatial` (gated by `RE_SPATIAL_GLOBAL`); evidence stage opt-in |
| 7 | Engine availability != successful execution | `ReconstructionOrchestrator.run()` raises `ReconstructionOrchestrationError` when all backends fail — but `universal_pipeline._run_reconstruction` lets COLMAP exceptions propagate; `vertical_slice.run_pose` wraps in `VerticalSliceError` |
| 8 | WorldIR and evidence-world representations may diverge | Two WorldIRs per job: vertical-slice world (HEAD) and opt-in evidence world (`evidence-world/<team>/<world>/<job>/`); connection only via opt-in `propose` (DESIGN_DECISIONS.md section 7) |
| 9 | Multiple modules define inconsistent entity/geometry/evidence contracts | Section 4 (≥4 association, ≥4 taxonomy, ≥3 camera/scale contracts) |
| 10 | Production API / CLI / benchmark execute different subsets of the pipeline | Section 5 (API runs coarse segments; benchmark runs golden loop; CLI runs vertical slice directly) |

## 7. Inspected surface

Files inspected (the mandate's priority list + supporting modules):

- `engine/pipeline/vertical_slice.py` (968+ lines, monolithic capture-to-world)
- `engine/pipeline/progressive.py` (441 lines)
- `engine/pipeline/stages/{base,registry,runner,lineage,run_record,facts,content_store,artifacts,s01_evidence,s02_quality,s03_reconstruct,s04_pose,s05_scale,s06_depth,s07_dense,s08_mesh,s09_perception,s10_layout,s10_detail,s10_finalize,s11_compile_validate,s12_quality_report,s13_changeset,s14_spatial_global}`
- `engine/orchestrator/universal_pipeline.py` (513 lines)
- `reconstruction/orchestrator.py` (464 lines)
- `engine/compiler/world_compiler.py` (993+ lines)
- `apps/api/jobs.py` (1924+ lines, incl. `_run_reconstruct_session_inner`)
- `apps/api/evidence_world_stage.py` (815+ lines)
- `perception/engine_orchestrator.py` (871 lines)
- `perception/engine_evidence.py`, `perception/spatial_inputs.py`
- `world_ir/{schema_v1,observations,observation_table,validation,world_v1,artifact_store,geometry_data,diff,entity_reid}`
- `engine/pipeline/world_delta.py`, `dense_gate.py`, `spatial_continuity.py`
- `exporters/{ifc_bridge,gltf,floorplan,...}`, `worldstore/store.py`
- `tests/integration/{test_release_gate,test_golden_path_real_photos,test_durable_job_queue,test_reliability_journey,...}`

## 8. Module ownership list (for coordination)

Each non-trivial module's owner to coordinate before any contract change:

- `engine/pipeline/vertical_slice.py` — shared (refactor target)
- `engine/pipeline/progressive.py` — shared
- `engine/pipeline/stages/runner.py` — shared (defect #3, #4)
- `engine/pipeline/stages/registry.py` — shared (defect #1)
- `engine/orchestrator/universal_pipeline.py` — shared (UNWIRED)
- `reconstruction/orchestrator.py` — reconstruction owner
- `engine/compiler/world_compiler.py` — shared
- `apps/api/jobs.py` — shared (refactor target)
- `apps/api/evidence_world_stage.py` — shared (opt-in evidence path)
- `perception/engine_orchestrator.py` — FreeBuff (perception owner)
- `world_ir/validation.py`, `world_ir/world_v1.py` — shared
- `worldstore/store.py` — OpenCode (storage owner)
