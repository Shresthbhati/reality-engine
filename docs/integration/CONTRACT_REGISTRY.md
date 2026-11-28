# Contract Registry — Reality Engine V1

> Authoritative, consistent contracts for subsystem boundaries. Every contract names: schema location, producer,
> consumer, required dependencies, error states, and the rule "no implicit conversions that conceal incompatible values."
>
> Part of the Kilo mandate (mandate §D). Owner: Principal Systems Architect / Integration Engineer.
> Status: **observed from code** + **prescriptive** where contracts conflict.

## Contract C1 — Image identity

**Schema:** `evidence.session.EvidenceItem` (`id, kind, source_uri, sha256`); `apps/api/models.py` `Evidence.checksum` (sha256 hex); `reconstruction/scale.py` `ScaleReference(evidence_id_a, evidence_id_b, distance_m, method, model_distance, unit_check_sep)`.

**Producer:** `apps/api/jobs.py` `_drop_exact_duplicates` + `_evidence_intrinsics` + `reconstruction/scale.py` `consolidate_references`.

**Consumer:** registration (camera <-> path matching via `idmap`), scale anchoring (measured baselines between two registered cameras), `evidence_world_stage.cameras_from_poses`.

**Rule:** No implicit conversion. sha256 of stored artifact is the identity; byte-identical duplicates dropped with reason; a malformed stored reference is skipped, never guessed at.

**Conflicts:** None direct, but identity is duplicated across `Evidence.checksum`, `EvidenceItem.sha256`, and `ScaleReference` — all must use sha256 hex consistently.

---

## Contract C2 — Camera coordinate frames

**Schema:** `world_ir/camera.py` `PinholeCamera(k, extrinsics)`, `CameraIntrinsics(fx, fy, cx, cy, width, height)`, `CameraExtrinsics(position, rotation)`; `world_ir/coordinates.py` `Frame` (WORLD/CAMERA/etc.), `Transform`; `reconstruction/frame.py` `canonicalize_like_head` (dominant plane -> +Y); `perception/spatial_inputs.py` `CameraInput(camera_id, image_id, source_uri, position_m, orientation_wxyz, intrinsics, pose_source)`.

**Convention:** Right-handed, Y-up canonical world frame (DESIGN_DECISIONS.md axis convention). Camera poses = camera CENTERS with camera-to-world rotations (wxyz quaternion). `reconstruction/frame.py` `rotate_in_place` + `refine_up_from_planes` re-fit "up" to dense cloud.

**Producer:** `reconstruction/orchestrator.py` `ReconstructionOrchestrator` (COLMAP); `vertical_slice.run_frame` (canonicalize + refine); `apps/api/jobs.py` `_spatial_inputs_provider` (builds `CameraInput` from `prog.camera_poses`).

**Consumer:** depth unprojection (`camera_from_pose`), perception lifting, `evidence_world_stage.cameras_from_poses`, `spatial_global` optimizer, `world_delta.camera_consistency` (cross-version alignment via Umeyama similarity).

**Rule:** No implicit frame conversion. `R_UP` maps job frame (+Y up) <-> package working frame (Z-up); every record states its frame; SfM frame mapped into job frame by residual-checked similarity. `world_delta` compares camera layout after similarity alignment; < `MIN_COMMON_CAMERAS` (4) shared = not measurable.

**Conflicts:** `exporters/boxes.py` states "WorldIR is Z-up" (latent contradiction with Y-up decision); `perception/engine_evidence.py` `CoordinateFrame.scale_state` in {metric, sfm_gauge, relative, image}; `spatial_inputs.py` `CameraInput.pose_source` in {colmap, vio, registered, unknown}.

---

## Contract C3 — Scale states

**Schema:** `reconstruction/scale.py` `ScaleState` {METRIC, RELATIVE}; `world_ir/schema_v1.py` `Measurement` (value, unit [SI only], precision, provenance, confidence, uncertain); `perception/spatial_inputs.py` `ScaleStateInput(basis, unit, is_metric)`; `evidence_world_stage.readiness()` scale states per entity (`metric_anchored` / `sfm_units` / `unanchored` / `no_depth_evidence`).

**Producer:** `vertical_slice.run_scale` (anchor_metric_scale / unscaled / working_unit); `apps/api/jobs.py` `_intrinsics_options` + baselines; `world_ir.metadata["scale"]` (state, meters_per_unit, note, calibration, scale_basis, dimension_state).

**Consumer:** depth (device depth only coherent when metric), perception (lifting needs metric depth), mesh (requires metric world), fusion (refuses RELATIVE), `evidence_world_stage` (scale_state per entity transform), `readiness().metric_export_ready` (true only when every placed entity metric_anchored).

**Rule:** Metric scale comes ONLY from explicit operator-measured baseline (or known_length / prior_accepted_by_user with user consent — DESIGN_DECISIONS.md Decision 023 amendment). Unanchored worlds stay RELATIVE with unit-less distances. No code path invents a scale factor. `meters_per_unit = None` for RELATIVE; never `1.0` by assumption.

**Conflicts:** `perception/engine_evidence.py` `CoordinateFrame.scale_state` vs `reconstruction/scale.py` `ScaleState` vs `spatial_inputs.py` `ScaleStateInput.basis` — three enums for the same concept; `integration._metric3d_scale` maps evidence-world `scale_state` to `ScaleState` (gauge never becomes relative depth or metres).

---

## Contract C4 — Entity identity

**Schema:** `world_ir/schema_v1.py` `Entity.id` (uuid4 `ent-...`), `Entity.type` (`EntityType` enum incl. ROOM/WALL/FLOOR/CEILING/DOOR/WINDOW/STAIRS/STOREY/CORRIDOR/SUPPORT_SURFACE/FURNITURE/...), `Entity.geometry_ids`, `Entity.relationships` (`RelationshipKind`), `Entity.custom_properties` (extension slot incl. `reconstruction` trace, `attributes`, `observation_ids`, `boundary_element_ids`, `space_id`/`room_id`, `host_wall_id`).

**Producer:** `engine/compiler/world_compiler.py` (promote_plane_to_entity / promote_room_to_entity); `vertical_slice._perception_stage` (entity ids `entity-object-<slug>-<NN>` deterministic by class + position); `world_ir.entity_reid.carry_identity` (identity carry-over across versions).

**Consumer:** `world_delta.entity_records` / `entity_continuity` (continuity by type + position in previous frame, ids re-derived); `validate_world_ir` (duplicate identity, boundary references, single storey membership); WorldStore versions; exporters.

**Rule:** One physical object = one entity however many photos show it. Identity comes only from measured spatial continuity (one-to-one preserved/refined, same type); splits/merges/ambiguities keep their own ids and record lineage. `Entity.custom_properties["reconstruction"]` trace stamped by reconstruction job before persistence (session/backend/config) — WARNING if missing on RECONSTRUCTED entities.

**Conflicts:** `world_ir.schema_v1.py` `Entity.type` is the canonical enum; `perception/entities/taxonomy.py` `EntityClass.world_type` maps to it; `engine/multiengine/spatial_intelligence/taxonomy.py` `TaxonomyClass` is a third ontology (Domain-based). All three must converge on `config/ontology.yaml` (BUILD_SPEC B03).

---

## Contract C5 — Ontology

**Schema:** `config/ontology.yaml` + `perception/ontology.py` `Ontology` / `OntologyNode` (path, label, parent, children, alias_to_path, attributes incl. "unknown", aliases, components, attach, typical_dims_m, min_pixels, prompts, geometry, world_type, small); `perception/entities/taxonomy.py` `EntityClass` shim reading the ontology.

**Producer:** B03 build (ports every class in `taxonomy.py` into YAML; `taxonomy.get/children/ancestors/class_for_label/size_plausible` delegate).

**Consumer:** detection (prompts per node), association (label compatibility `labels_compatible`), benchmarks (inventory scorer), `perception/entities/taxonomy.py` legacy consumers.

**Rule:** `Ontology.validate()`: every components/attach path exists; no cycle; every attribute list contains "unknown". Backward-compatible shim: old `taxonomy.all_classes()` ids still resolve.

**Conflicts:** `engine/multiengine/spatial_intelligence/taxonomy.py` (Domain/TaxonomyClass) is used by detection_pipeline, completeness, association, scene_graph, relationships, integration, geometry_validator, benchmarks, datasets, intelligence — a second ontology. `perception/style/taxonomy.py` (DesignStyle) and `perception/attributes/taxonomy.py` are third/fourth. Contract target: single canonical ontology at `config/ontology.yaml`.

---

## Contract C6 — Geometry references

**Schema:** `world_ir/schema_v1.py` `Geometry` (id, type `GeometryType` {MESH, POINTCLOUD, PLANE, BOX, SPHERE, CYLINDER, CAPSULE, TERRAIN, VOXEL, IMPLICIT, UNKNOWN}, lod_level, vertex_count, triangle_count, data_uri, data_hash, bounds_min/max `Vector3`, provenance `Provenance`, confidence, observations, quality_metrics); `world_ir/geometry_data.py` `MeshData` / `PointCloudData`; `world_ir/artifact_store.py` `ArtifactStore` (sha256:// URI -> `<root>/<h[:2]>/<h[2:4]>/<h>`).

**Producer:** `engine/compiler/world_compiler.py` (window/stair/corridor geometries); `vertical_slice._fusion_stage` (geom-fused-pointcloud POINTCLOUD); `vertical_slice._mesh_stage` (Poisson mesh); `reconstruction/meshing/` (MeshData).

**Consumer:** WorldStore persistence (content-addressed); exporters (read via `data_uri`/`data_hash`); `world_delta.entity_records` (bounds + sampled points); `s12_quality_report` (mesh validity via trimesh on `MeshData.from_bytes(store.get(geom.data_uri))`).

**Rule:** Compiled worlds reference geometry through `(data_uri, data_hash)` artifact pointers, never embedding payloads (DESIGN_DECISIONS.md Decision 018). Bounds must not be inverted; non-finite values are ERROR in `validate_world_ir`; a bounding box is an approximation labeled INFERRED — never presented as mesh geometry (Decision 017).

**Conflicts:** `world_ir.geometry_data.MeshData` vs `reconstruction/meshing/` MeshData — need to confirm single canonical runtime mesh representation (Decision 017 names `reconstruction/meshing/mesh.py` MeshData as canonical).

---

## Contract C7 — Evidence observations

**Schema:** `world_ir/observations.py` `Observation` (observation_id, image_id, label, world_pos, sigma_m, depth_source `DepthSource`{METRIC/MVS/RELATIVE}, model, license, normal, bounds_min/max, confidence, mask_uri, bbox_px, visibility, embedding_ref, session_id, created_at); `world_ir/observation_table.py` `ObservationTable` (columnar struct-of-arrays: obs_id, image_id, class_path, bbox_px, score, mask_ref, emb_row, world_pos, sigma_m, visibility, entity_row, model, attrs_json); `perception/engine_evidence.py` `ObservationRecord` (evidence_id, source_image_id, coordinate_frame, model_id, label, category, bbox_px, world_pos, mask, material_hypothesis, confidence, sigma_m, ...).

**Producer:** perception stages (`vertical_slice._perception_stage` lifts masks via metric depth -> 3D hypotheses); `perception/spatial_inputs.py` producers (`produce_spatial_inputs_from_scene_graph` / `from_snapshot`).

**Consumer:** `world_ir.entity.observations`; `world_delta.entity_records` (supporting_evidence_ids); `validate_world_ir` invariant 2 (OBSERVED/PARTIALLY_OBSERVED entities cite at least one observation_id); `ObservationTable.to_inventory()` (benchmark contract: image_id, entity_id, category, subtype, bbox; unmeasured fields None + unavailable reason); `evidence_world_stage` (evidence matrix).

**Rule:** An `Observation` is one 2D detection lifted into the world frame; `depth_source` states the truth (METRIC/MVS/RELATIVE); sigma via `sigma_for(depth_source, range_m)` = max(SIGMA_FLOOR_M 0.05m, fraction*range) {METRIC 0.02, MVS 0.03, RELATIVE 0.25}. `world_pos=None` means not lifted — never a made-up position. `embedding_ref` references a stored appearance embedding, never the vector itself. `LEDGER_KEY = "observation_ledger"` stored in `world.metadata`.

**Conflicts:** `Observation` (world_ir/observations.py) vs `ObservationRecord` (perception/engine_evidence.py) vs columnar `ObservationTable` — three shapes for the same concept; `engine_evidence.compare_observations` only compares positions when same frame id + same scale state in {metric, sfm_gauge} (otherwise scale_state_mismatch note, never a metres comparison).

---

## Contract C8 — Uncertainty

**Schema:** `provenance.py` `Provenance` enum {OBSERVED, RECONSTRUCTED, ESTIMATED, INFERRED, GENERATED, UNKNOWN, CONFLICT} + `Uncertainty(confidence, note)`; `uncertainty.py` `UncertainScalar` (value + sigma + basis); `world_ir/schema_v1.py` `Measurement.uncertain` + `Entity.confidence_vector` `ConfidenceVector` (11 dims, None = not measured).

**Producer:** `reconstruction/orchestrator._stamp_provenance` (stamps backend name on point/pose uncertainty notes); `vertical_slice` stages (provenance INFERRED for perception entities, ESTIMATED for measurements); `world_ir.validation._check_provenance_confidence` (UNKNOWN provenance with confidence >= 0.9 is ERROR).

**Consumer:** `validate_world_ir` (confidence_out_of_range, unknown_provenance_high_confidence); `readiness().quality` (uncertainty_position / uncertainty_dimensions coverage); `world_delta` (cand_confidence / prev_confidence on relations).

**Rule:** `None` means "not measured" — never fabricated or defaulted to 0.5. `Measurement.uncertain` supersedes legacy precision/confidence for formal propagation (P10-02). `Entity.confidence_vector` dimensions each None or float in [0,1]. A confidence is never upgraded (compiler never upgrades a value's status).

**Conflicts:** `ConfidenceVector` vs `Measurement.precision` vs `Uncertainty.confidence` vs `Observation.confidence` — four uncertainty representations; `Observation.confidence` default 0.5 (a fabricatable default — contract risk).

---

## Contract C9 — Stage execution

**Schema:** `engine/pipeline/stages/base.py` `StageSpec(name, version, resource_class{cpu|gpu|highmem|export}, required, inputs?, requires_outputs, idempotent, retries)`; `StageResult(name, status{succeeded|degraded|skipped|failed}, inputs_hash, outputs{logical_name->sha}, metrics, warnings, error, started_at, ended_at, peak_rss_mb)`; `StageContext(job_id, workdir, artifacts{logical->sha}, params, state{in-memory}, log)`; `RunManifest.grade(specs)` (required failed -> failed; any failed/degraded -> partial; else succeeded).

**Producer:** `engine/pipeline/stages/runner.py` `run_stages()` (sequential, checkpoint reuse, timeouts, admission, RSS); `engine/pipeline/stages/registry.py` `build_stages(profile)`; each `sNN_*.py` stage class.

**Consumer:** `apps/api/jobs.py` `drive()` -> `run_stages(...)` with `DbCheckpointStore`, `should_cancel`, `on_end`, `stage_timeouts`, `admission=StageAdmission.from_env()`.

**Rule:** `inputs_hash = compute_inputs_hash(spec.version, ctx.artifacts, ctx.params)` hashes ALL accumulated artifacts — NOT declared `spec.inputs` (mandate defect #3). `requires_outputs` enforced: success/degraded without declared output -> failed. Timed-out stage abandoned (thread not killed; result discarded; never retried — mandate defect #4). `idempotent` gates retries; non-idempotent never re-run. `StageCancelled` between stages only; cancel before claim -> CANCELLED immediately; after HEAD adoption -> version STANDS, job grades CANCELLED.

**Conflicts:** `StageSpec.inputs=None` means "assume all earlier ones (lineage)" — ambiguous; `LineageGraph` tracks producer/consumer/reads but `compute_inputs_hash` doesn't use `spec.inputs`. Two checkpoint stores: `DurableJobCheckpointStore` (in-memory host segments, coarse) + `DbCheckpointStore` (durable `job_stage_runs`).

---

## Contract C10 — Job lifecycle

**Schema:** `apps/api/jobs.py` states {queued, running, partial, succeeded, failed, cancelled}; `TERMINAL_STAGE` {succeeded:done, partial:done_with_warnings, failed:failed, cancelled:cancelled}; `JobOutcomeError` (deterministic, never retried); `LeaseLost` (worker reaped/requeued); `_verify_outcome` (version_id must exist in DB + WorldStore verify_version clean).

**Producer:** `apps/api/jobs.py` `enqueue_job` / `process_next_job` / `_run_with_lease` / `_finalize_fenced` (compare-and-swap UPDATE while lease holds).

**Consumer:** API `/api/jobs/{id}`; workers; `readiness()`; `_notify_world_version` (notification + activity event on world.version_created).

**Rule:** `SUCCEEDED` gated (output exists, validates, persists, provenance present) — a timeout/crash/degraded run can never become SUCCEEDED. `PARTIAL` = version adopted but degraded (partial registration, skipped evidence, failed optional stages); degradation reasons on payload. Once a version is adopted, cancellation cannot erase it (durable work not un-written) — job grades CANCELLED with `adopted_before_cancel` recorded.

**Conflicts:** `TERMINAL_STATES` includes PARTIAL as terminal, but `TERMINAL_STAGE` maps partial->done_with_warnings; UI contract (STUDIO_CONSUMER_UX_PLAN B-01): done | done_with_warnings | failed | cancelled.

---

## Contract C11 — World versioning

**Schema:** `worldstore/store.py` `WorldStore` (`save_version(world, parent, version_id) -> stored`; `verify_version`; `load_version`; `list_versions`); `world_ir/world_v1.py` `WorldIR` (id, name, version, main_branch_id, branches, scenarios, entities/geometries/materials/surfaces/components, transforms, observations, metadata, hierarchy, geo_anchor); `world_delta.snapshot(...)` / `compute_delta(...)` / `decide(...)` / `reconcile_conflicts(...)`.

**Producer:** `apps/api/jobs.py` seg_changeset_commit (`worldstore_service.commit_version`); `worldstore/lifecycle.py` (adoption intent -> WorldStore version -> read-back verify -> DB HEAD + mirror in one transaction -> intent cleared; quarantine on fault).

**Consumer:** inspection routes (`/api/worlds/{id}/versions/{vid}`), correction (expected_version_id fencing), exporters (named version), `world_delta` (diff between HEAD and candidate), `evidence_world_stage` (head_version / committed_version).

**Rule:** Versions immutable; version id = content hash (sha256 of world dict) -> idempotent re-persist. Adoption order: intent file -> WorldStore version -> read-back verify (BEFORE pointer moves) -> DB HEAD + mirror row in ONE transaction -> intent cleared. Failed adoption undoes its own pending HEAD move. Rollback moves only the HEAD pointer; versions stay immutable; COLMAP base set aside (renamed), next version chains onto rolled-back HEAD.

**Conflicts:** `WorldIR.main_branch_id` defaults to `branch-main-<id>`; `branches`/`scenarios` additive. `world_metadata["scale"]` carries scale state per version; `world.metadata["progressive"]` carries level/level_name/model_state/attempts.

---

## Contract C12 — Artifact persistence

**Schema:** `engine/pipeline/stages/content_store.py` `store_bytes(data) -> (digest, path)`; `resolve_artifact(uri) -> Path | None`; `verify_artifact(uri) -> bool`; `engine/pipeline/stages/artifacts.py` (put_bytes/get_bytes/put_array/get_array/put_json/get_json/put_npz/get_npz); `apps.api.storage.store_bytes` (reused — no second store); `STORAGE_ROOT` env (default `./data/artifacts`).

**Producer:** `artifacts.py` / `content_store.py` (content-addressed blobs by sha256, atomic temp+replace, truncated blobs rewritten); `worldstore/store.py` (WorldStore versions under WORLDSTORE_ROOT).

**Consumer:** stage `outputs` (logical name -> artifact sha); `world_delta.entity_records` (reads geometry payloads via artifact_store.get(g.data_uri)); `s12_quality_report` (mesh via store.get); exporters (data_uri/data_hash); `evidence_world_stage.run_artifact_store` (chained: run artifacts + worldstore artifacts).

**Rule:** Blobs stored at `<root>/<h[:2]>/<h[2:4]>/<h>` referenced as `sha256://<h>`; raw paths never given to clients; path traversal protection (`path.relative_to(root)`); existing blob trusted only if size matches; crash mid-write leaves no partial blob (atomic replace). A reader handed only one of run-artifacts vs worldstore-artifacts silently loses the other's geometry — export/viewer resolve through both (ChainedArtifactStore).

**Conflicts:** `engine.pipeline.stages.content_store` vs `apps.api.storage` — must re-export the same store (no second store); `STORAGE_ROOT` default `./data/artifacts` relative resolved to absolute (relative root broke round-trip equality — fixed).

---

## Cross-cutting contract rules (mandate §D)

1. **Schema validation at subsystem boundaries** — `validate_world_ir` gate before returning from compiler; `SpatialInputsV1.validate()`; `ObservationTable` schema fixed.
2. **No implicit conversions** — scale state/units stated on every entity transform; camera frame stated on every record; depth_source states the truth; `None` = not measured, never defaulted.
3. **Provenance on every field** — `Provenance` enum on Entity/Geometry/Measurement/Relationship; `confidence_basis` recorded; `calibration` record shared between scale stage and evidence-world calibration.
4. **Optional stages explicit** — depth/perception/mesh/dense/detail/spatial_global/evidence_world all record skip reasons; never silent degradation.
