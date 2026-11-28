# Canonical Execution Path — Reality Engine V1

> The authoritative lifecycle for one capture-to-world product run. This is the single path the production API, CLI,
> and benchmark must all execute. Optional stages are explicit; their omission is visible in the job report.
>
> Produced as part of the Kilo mandate (mandate §C). Owner: Principal Systems Architect / Integration Engineer.
> Status: **observed from code** + **prescriptive** for the canonical path.

## 0. Lifecycle overview

```
Upload
  -> validation            (s01 evidence_resolve + s02 evidence_quality)
  -> image identity        (checksum / duplicate detection)
  -> calibration           (intrinsics: EXIF / operator-declared / COLMAP-estimated)
  -> registration          (s04 pose: SfM camera registration)
  -> scale                 (s05 scale: metric anchoring or honest RELATIVE)
  -> depth                 (s06 depth: optional monocular / sidecar depth, skipped visibly)
  -> perception            (s09 perception: optional object detection/lifting)
  -> association           (perception.instances.association.resolve_hypotheses, gated Hungarian)
  -> geometry              (s08 mesh + s07 dense + plural-source fusion -> WorldIR geometry)
  -> evidence fusion       (evidence observations fused into entities; WorldIR)
  -> validation            (s11 compile_validate: WorldIR validation gate)
  -> persistence           (WorldStore save_version + read-back verify)
  -> inspection            (readiness() / routes_worlds / routes_query)
  -> correction            (typed edit commands + undo -> new version)
  -> new version           (HEAD advances only on adoption; read-back verified)
  -> diff                  (world_delta between HEAD and candidate)
  -> export                (IFC, glTF, USD, DXF, PDF from a named version)
```

Every stage defines: Owner / Inputs / Outputs / Schema / Required dependencies / Error states / Timeout behaviour / Resource requirements / Provenance / Validation / Downstream consumers.

Not every stage runs for every input. Optional stages (depth, perception, mesh, dense, detail, spatial_global) are explicit; skips are recorded with reasons, never silent.

---

## Stage 1 — Validation (evidence_resolve + evidence_quality)

**Owner:** API job worker (`apps/api/jobs.py` seg_evidence_resolve / seg_evidence_quality), exposed as coarse stages `evidence_resolve` / `evidence_quality`.

**Inputs:** Session's photo evidence rows (Evidence with artifact_uri), world.id.

**Outputs:** Resolved `(evidence_row, path)` list; skipped/duplicate evidence with reasons; `ImageFacts` per image (ok, evidence_class, class_basis, geometry_eligible, width, height).

**Schema:** `evidence.image_facts` stored on Evidence.metadata_json; `resolved`, `skipped_evidence`, `dupes` in job ctx.

**Required dependencies:** artifact store resolvable (`apps.api.storage.resolve_artifact`).

**Error states:** `InsufficientEvidence` when 0 usable photos; garbage "photos" refused before COLMAP (measured >90s wedge).

**Timeout behaviour:** No per-stage deadline by default; `RE_STAGE_TIMEOUTS_S` can set one; timed-out stage is abandoned (recorded failed, never retried).

**Resource requirements:** CPU, disk I/O for artifact resolution + sha256.

**Provenance:** checksum (sha256) recorded per evidence item.

**Validation:** `evidence.ok && evidence.geometry_eligible` required for geometry input; non-photographic material kept as context only, never mixed into observed geometry.

**Downstream consumers:** `seg_reconstruct` (run_inputs), scene gate.

**Defect note:** `ctx.artifacts` in `runner.py` accumulates ALL artifacts; `compute_inputs_hash` hashes them all instead of declared `spec.inputs` (mandate defect #3).

---

## Stage 2 — Image identity

**Owner:** `apps/api/jobs.py` `_drop_exact_duplicates` + `reconstruction/scale.py` ScaleReference / `evidence.session.EvidenceItem`.

**Inputs:** Resolved evidence rows with paths and checksums.

**Outputs:** Kept (first of each checksum) + dropped byte-identical duplicates (with reason); `evidence_item.sha256` set.

**Schema:** `Evidence.checksum` (sha256 hex); `idmap = {evidence_id: path}`.

**Required dependencies:** artifact resolution complete.

**Error states:** None (dedup is non-fatal; duplicates reported).

**Timeout behaviour:** None.

**Resource requirements:** CPU + disk I/O (sha256 streaming, 1 MiB chunks).

**Provenance:** sha256 of stored artifact.

**Validation:** `path.stat().st_size == len(data)` when rewriting blobs; truncated blobs rewritten (atomic replace).

**Downstream consumers:** registration (camera -> path matching), intrinsics, scale anchoring references.

---

## Stage 3 — Calibration (intrinsics)

**Owner:** `apps/api/jobs.py` `_intrinsics_options` + `_evidence_intrinsics` + `engine/pipeline/stages/s04_pose.py` `committed_intrinsics` / `adopt_solved_intrinsics`.

**Inputs:** EXIF `focal_length_35mm` per photo, operator camera declaration (`POST /api/worlds/{id}/camera`), COLMAP-estimated intrinsics from a previous sparse model, image size (most common among eligible photos).

**Outputs:** `intrinsics = (fx, fy, cx, cy)` + `intrinsics_trust = "estimated"` (never "trusted" unless operator-declared).

**Schema:** `VerticalSliceOptions.intrinsics` / `.intrinsics_trust`; record stored under `world.metadata["scale"]["calibration"]`.

**Required dependencies:** All eligible photos share one image size; EXIF focal length agreed (1%) or COLMAP estimated model available.

**Error states:** intrinsics unset -> depth/perception/meshing stages skip with explicit reason ("refusing to guess one"); never invented.

**Timeout behaviour:** None.

**Resource requirements:** CPU only.

**Provenance:** trust = "estimated" (COLMAP not pinned, distortion ignored); source recorded (`operator_declared_camera` / `exif_focal_length_35mm` / `colmap_estimated_previous_sparse_model` / `colmap_estimated_this_run`).

**Validation:** every camera agrees within 1%, pinhole-like (non-fisheye), real image size.

**Downstream consumers:** depth stage (unprojection needs camera model), perception stage (lifting needs intrinsics), mesh stage, `evidence_world_stage.cameras_from_poses`.

**Defect note:** Intrinsics adopted post-solve (`adopt_solved_intrinsics`) mutate `options` mid-run; two candidates judged as worlds may disagree if timing differs.

---

## Stage 4 — Registration (pose)

**Owner:** `engine/pipeline/vertical_slice.py` `run_pose()` -> `reconstruction/orchestrator.py` `ReconstructionOrchestrator` -> `reconstruction/backend/colmap_backend.py` `ColmapReconstructionBackend`.

**Inputs:** `Sequence[EvidenceItem]` (>= 2), `VerticalSliceOptions` (colmap_binary, colmap_session, intrinsics if trusted).

**Outputs:** `PoseOutcome(evidence_items, backend, run, result)` where `result` = `ReconstructionResult` (camera_poses, points, registration_status, merge_report, dense_points, dense_report).

**Schema:** `ReconstructionResult` (frozen dataclass in `reconstruction/backend/interface.py`); camera poses = camera CENTERS with camera-to-world rotations (wxyz quaternion); `registration_status` in {success, partial, failed}.

**Required dependencies:** COLMAP binary available (or `reconstruction_backend` injected for tests); >= 2 image evidence items; `ReconstructionOrchestrator.validate_evidence` input gate.

**Error states:** `VerticalSliceError` when < 2 images, reconstruction exception, `registration_status == "failed"`, no points, or `UnvalidatableReconstructionError`. Partial registration is reported (degraded), never silently upgraded.

**Timeout behaviour:** `run_stages` per-stage deadline via `RE_STAGE_TIMEOUTS_S`; timed-out stage abandoned (thread not killed; result discarded; never retried — mandate defect #4).

**Resource requirements:** GPU (COLMAP), CPU, RAM; COLMAP on 6 GB GPU measured: 115 points / 79.5 s on 10-photo set.

**Provenance:** `ReconstructionOrchestrator._stamp_provenance` stamps `RECONSTRUCTED` + backend name on every point/pose uncertainty note; `run.diagnostics.backend_name` / `duration_s`.

**Validation:** `validate_reconstruction_output(result)`; `registration_status != "failed"` and `result.points` non-empty.

**Downstream consumers:** scale stage, depth stage, layout stage, perception stage, `evidence_world_stage` (camera poses -> `ew_cams`), `world_delta` snapshot.

---

## Stage 5 — Scale

**Owner:** `engine/pipeline/vertical_slice.py` `run_scale()` + `run_frame()` -> `reconstruction/scale.py` (`anchor_metric_scale`, `unscaled`, `working_unit`) + `reconstruction/frame.py` `canonicalize_like_head`.

**Inputs:** Scaled reconstruction result, evidence items, `VerticalSliceOptions.measured_baselines` (operator-measured distances between two registered cameras), `head_unit_k` (RELATIVE head working unit), `head_frame` (HEAD canonical rotation).

**Outputs:** `ScaleOutcome(result, meters_per_unit, scale_state, scale_note, unit_k, unit_facts, scale_error, calibration)` + `FrameOutcome(result, frame_record, unit_k, unit_facts, scale_note)`.

**Schema:** `world.metadata["scale"] = {state, meters_per_unit, note, provisional_unit_k, provisional_unit_base_k, provisional_unit, calibration}`; frame canonicalization -> dominant plane -> +Y.

**Required dependencies:** pose stage result; measured baselines (optional — without them world stays RELATIVE, never metric-by-assumption).

**Error states:** `ScaleAnchoringError` per baseline (tried in order; first whose pair both registered wins). No baseline registered -> RELATIVE with working unit (never a fake metric). `FrameCanonicalizationError` -> `VerticalSliceError` (required stage; failure aborts the run).

**Timeout behaviour:** None (CPU).

**Resource requirements:** CPU.

**Provenance:** `scale_basis` / `dimension_state` (MEASURED vs ESTIMATED); `calibration` record (same shape evidence-world calibration stores); trust order: camera baseline > known length > accepted prior (DESIGN_DECISIONS.md Decision 023 amendment).

**Validation:** RELATIVE worlds never claim meters; metric worlds carry `meters_per_unit`; world scale state recorded in metadata.

**Downstream consumers:** depth (unprojection metricized against sparse cloud), perception (lifting needs metric depth), mesh (requires metric world), fusion (requires METRIC scale), `evidence_world_stage` (scale_state per entity), `readiness()` (metric_export_ready).

**Defect note:** Bug fixed 2026-09-14: `result` was never reassigned after `scaled = anchor_metric_scale(...)`, so the world claimed meters while downstream geometry stayed in model units. Fixed in `run_scale`.

---

## Stage 6 — Depth (optional)

**Owner:** `engine/pipeline/vertical_slice.py` `_depth_stage()` / `run_depth()` -> `perception.depth` backends (`DepthAnythingV2Backend`, `MiDaSDepthBackend`) + `reconstruction/depth_to_points.py` `metricize_all`.

**Inputs:** Scaled + frame-canonical reconstruction, evidence items, `VerticalSliceOptions.depth_model` (None = disabled), trusted intrinsics.

**Outputs:** `DepthOutcome(facts, metric_maps)`; depth points appended to `result.points` IN PLACE (track_id `depth-`); metricized depth maps returned for perception lifting.

**Schema:** facts = {status, model, maps, metricized, metricize_failed, dense_points, stride, residual_median_m, alignments, note}.

**Required dependencies:** `depth_model` set, trusted intrinsics, METRIC scale (device depth sidecars only coherent when metric).

**Error states:** `depth_model=None` -> skipped ("disabled"); no trusted intrinsics -> skipped ("refusing to guess one"); backend import failure -> skipped; backend exception -> skipped (never a crash). RGB-D sidecar path first (real 16-bit PNG device depth); unaligned relative depth stays RELATIVE_DEPTH.

**Timeout behaviour:** GPU depth backend run under `engine.core.device.unloading`; no hard deadline by default.

**Resource requirements:** GPU (Depth Anything V2 Small FP16 ~0.42 GB VRAM; MiDaS DPT_Hybrid ~0.68 GB).

**Provenance:** `scale_state` = relative (monocular per-view alignment vs SfM cloud = documented approximation); `metricize_failed` count recorded.

**Validation:** points trimmed to camera envelope (`trim_to_scene`); invalid-pixel masks / NaN holes carried (`DepthMap.valid_mask`).

**Downstream consumers:** perception stage (lifting masks against the SAME metricized maps — one owner of depth state), fusion stage (depth points source), `evidence_world_stage` (depth records).

---

## Stage 7 — Perception (optional)

**Owner:** `engine/pipeline/vertical_slice.py` `_perception_stage()` / `run_perception()` -> `perception.detection` (Mask R-CNN / Grounding DINO) + `perception.instances.lifting` + `perception.instances.association.resolve_hypotheses`.

**Inputs:** Reconstruction result, layout world, evidence items, metricized depth maps, `VerticalSliceOptions.perception_model` (None = disabled), intrinsics, scene_gate (indoor only), object_resolver ("association" | "union_find"), perception_mask_source ("sam2" | "depth_band"), perception_appearance ("dinov2" | None).

**Outputs:** `PerceptionOutcome(facts)`; object entities promoted INTO `world` IN PLACE; `ObservationTable` artifact persisted (sha256 -> `outputs["observations"]`).

**Schema:** facts = {status, model, views_segmented, masks_considered, hypotheses_lifted, candidates_merged, entities_promoted, resolution, detections_2d, small_fittings, second_opinion, note}.

**Required dependencies:** `perception_model` set, trusted intrinsics, metric depth maps survived, scene_gate allows indoor (detector vocabulary is indoor furniture).

**Error states:** disabled -> skipped; no intrinsics -> skipped; no metric depth -> skipped ("refusing to lift masks against relative depth"); dependencies unavailable -> skipped; world invalid after promotion -> `rejected` (objects removed, world left as it was — optional enhancement never sinks the structural world).

**Timeout behaviour:** GPU detector under `unloading(detector)`; no hard deadline by default.

**Resource requirements:** GPU (Mask R-CNN ~1.55 GB VRAM; Grounding DINO tiny FP16 ~1.7 GB; SAM 2 ~1.56 GB).

**Provenance:** entities `provenance=INFERRED` with per-entity evidence ids; appearance embeddings (DINOv2) optional; `association` resolver record (`resolver`, `observations`, `images`, `decision`, `ambiguous_with`).

**Validation:** `validate_world_ir(world)` after promotion; on error, roll back all added entities/geometries/ledger.

**Downstream consumers:** dense stage (gates structure pass), mesh stage, `world_delta` entity continuity, `evidence_world_stage` (entities -> canonical), `readiness()` coverage.

**Defect note:** `absorb_scale_drift_duplicates` deliberately NOT applied (unverified on real data); small-object taxonomy off by default (low precision).

---

## Stage 8 — Association (object identity across views)

**Owner:** `perception/instances/association.py` `resolve_hypotheses()` (production path used by vertical_slice) — NOTE: competing implementations exist in `engine/multiengine/association.py`, `engine/multiengine/spatial_intelligence/association.py`, `perception/entities/association.py` (mandate defect #9).

**Inputs:** 3D hypotheses (lifted from perception), camera positions, cameras (PinholeCamera), appearance embeddings (optional), world (existing entities).

**Outputs:** `candidates` (merged 3D object hypotheses) + `AssociationDecision` per observation (state: SAME_ENTITY_CONFIRMED / DIFFERENT_ENTITY / AMBIGUOUS / ...), entity ids (`entity-object-<slug>-<NN>`).

**Schema:** `AssociationDecision` (observation_id, state, entity_id, probability, reason, dist_m, mahalanobis_sq, normal_diff_deg, iou3d, geom_verified, candidates).

**Required dependencies:** perception hypotheses; cameras for epipolar verification (optional but preferred).

**Error states:** unknown `object_resolver` -> `VerticalSliceError`; same-image rows never merge (one object cannot appear twice in one photo); ambiguous kept apart.

**Timeout behaviour:** None (CPU; Hungarian assignment O(n³) fine for n < 500).

**Resource requirements:** CPU.

**Provenance:** per-observation `embedding_ref` (appearance), `geom_verified` flag (two-view epipolar), `probability` (log-likelihood weight).

**Validation:** gates = label compatibility, 3D Mahalanobis distance (uncertainty-aware), normal difference, appearance consistency, two-view epipolar geometric verification; AMBIGUOUS when best/second-best within log(2) margin without appearance disambiguation.

**Downstream consumers:** perception stage (promotion into world), `evidence_world_stage` (entities), `readiness()` multiview association coverage.

---

## Stage 9 — Geometry (mesh + dense + fusion)

**Owner:** `engine/pipeline/vertical_slice.py` `_mesh_stage()`, `_dense_mvs_stage()`, `_plane_sweep_stage()`, `_fusion_stage()` + `reconstruction/meshing/` + `reconstruction/mvs/` + `reconstruction/fusion/consumer.py`.

**Inputs:** Scaled reconstruction result (points with track_ids), layout world, `VerticalSliceOptions` (mesh_enabled, dense_mvs_enabled, plane_sweep_enabled, artifact_store), scale_state, meters_per_unit.

**Outputs:** `MeshOutcome(facts)`, `DenseOutcome(world, diagnostics, dense_mvs_facts, plane_sweep_facts, fusion_facts)`; fused pointcloud geometry artifact (`geom-fused-pointcloud`) attached to `world.geometries`; Poisson mesh artifact -> WorldIR geometry + entity.

**Schema:** mesh facts = {status, vertices, triangles, input_points}; dense_mvs facts = {status, dense_points_added, ...}; fusion facts = {status, n_sparse, n_depth, n_dense_mvs, n_associated, n_conflicts, artifact_uri, artifact_sha256, geometry_id}.

**Required dependencies:** `artifact_store` for traceable geometry payloads; METRIC scale (fusion refuses RELATIVE — "fusing unit-less ranges would silently claim meters"); >= 2 sources with >= 10 points each for plural fusion.

**Error states:** mesh disabled / not metric / no artifact store / too few points / deps unavailable -> skipped (never fabricated mesh); dense MVS OFF by default (explicit operator decision); plane sweep metric-only.

**Timeout behaviour:** GPU / highmem stages; COLMAP dense (`patch_match_stereo` + `stereo_fusion`) heavyweight; `dense_auto` evidence-gated via `dense_gate.dense_readiness()`.

**Resource requirements:** GPU (COLMAP dense), highmem (Poisson meshing), CPU (plane sweep).

**Provenance:** geometry `provenance=ESTIMATED`, `confidence=0.6`; `data_uri`/`data_hash` content-addressed; `track_id` prefixes distinguish sources (`sfm`, `depth-`, `dense_mvs:`, `plane_sweep:`).

**Validation:** `mesh_validity` (trimesh: watertight, degenerate faces, components) reported via `s12_quality_report` (NOT_EVALUABLE / PASS / WARN / FAIL); fusion conflicts recorded, never winner-picked.

**Downstream consumers:** detail stage (reads fused points), `world_delta` entity_records, WorldStore persistence, exporters.

**Defect note:** mesh stage requires metric world; `requires_outputs` on `compile_validate` = `("world_ir",)` ensures a published reference always points at a stored artifact.

---

## Stage 10 — Evidence fusion -> WorldIR (compile)

**Owner:** `engine/compiler/world_compiler.py` `compile_reconstruction_to_world()` -> `evidence/promote_planes.py`, `evidence/promote_rooms.py`, `perception/geometry/planes.py`, `perception/architecture/*`.

**Inputs:** `ReconstructionResult` (poses, points), `CompileOptions` (seed, up, distance_tolerance_m, min_inliers, room_prior, structure_enabled, layout_edges_ab, artifact_store, appearance_views).

**Outputs:** `WorldIR` (entities, geometries, relationships, measurements, metadata) + `CompileDiagnostics`.

**Schema:** `WorldIR.to_dict()` (deterministic, sorted keys); `CompileDiagnostics` (points_total, planes_total, planes_by_role, entities_created, rooms_detected, measurements_count, relationships_count, unknown_point_fraction, interior_warnings, validation_issues).

**Required dependencies:** reconstruction result not failed, points non-empty, camera poses present (classification is camera-side, refuses to guess).

**Error states:** `CompileInputError` (failed/empty reconstruction, no cameras); `WorldValidationGateError` (compiled world failed validation — refused, never silently corrupted).

**Timeout behaviour:** CPU; `refine_budget_s` caps room refinement (60 s default).

**Resource requirements:** CPU.

**Provenance:** points=RECONSTRUCTED, typed structure=INFERRED, measurements=ESTIMATED; compiler never upgrades a value's status.

**Validation:** `validate_world_ir(world)` gate runs BEFORE returning; unclassifiable planes and unassigned points accounted for in diagnostics (never silently dropped).

**Downstream consumers:** `vertical_slice.run_finalize` (metadata stamping), `evidence_world_stage.persist_world`, `world_delta` adoption, exporters.

---

## Stage 11 — Validation (compile_validate)

**Owner:** `apps/api/jobs.py` seg_compile_validate -> `world_ir.validation.validate_world_ir` + `world_ir.constraints.check_architectural_constraints`.

**Inputs:** Compiled `WorldIR` (`world_ir`).

**Outputs:** `validation` report (issues: errors + warnings); content-addressed world sha (`outputs["world_ir"]`).

**Schema:** `WorldValidationReport` (issues with severity ERROR/WARNING, code, message); `validation.is_valid()`.

**Required dependencies:** compiled world; provenance stamp (`_stamp_reconstruction_provenance` ran first).

**Error states:** `validate_world_ir` errors -> `RuntimeError` (world not persisted, HEAD intact); `_assert_reconstruction_traced` -> RuntimeError on untraced RECONSTRUCTED entities; `ConcurrentModificationError` -> retryable.

**Timeout behaviour:** None (CPU validation).

**Resource requirements:** CPU.

**Provenance:** every RECONSTRUCTED entity must carry `custom_properties.reconstruction` (session/backend/config) — WARNING if missing (pre-stamp worlds loadable; new drops flagged).

**Validation:** Blueprint invariants 1-7 (evidence_state vocabulary, OBSERVED cites observation, INFERRED not also OBSERVED, metric units only in non-relative worlds, id uniqueness, duplicate guard, opening->surface, surface->space). Structural self-check (dangling refs, temporal/branch consistency).

**Downstream consumers:** changeset_commit (adoption decision), `readiness()`, WorldStore.

---

## Stage 12 — Persistence

**Owner:** `apps/api/jobs.py` seg_changeset_commit -> `worldstore.store.WorldStore` + `worldstore/lifecycle.py` (adoption intent / quarantine).

**Inputs:** Validated `WorldIR`, head version id, report, points/cameras bytes.

**Outputs:** New immutable `WorldStore` version (`version_row.id`); DB HEAD pointer moved in ONE transaction with mirror row; intent cleared.

**Schema:** `WorldVersion` (id, world_id, version_id, parent, source_session_ids, report); `content_sha256` derived from world content (idempotent re-persist).

**Required dependencies:** validation passed; read-back verification BEFORE pointer moves (`store.verify_version(version_id)` then `store.load_version(vid)` must equal saved dict).

**Error states:** `ConcurrentModificationError` -> retryable (next attempt reloads HEAD); adoption intent file + WorldStore version + read-back verify + DB HEAD+mirror in one transaction (DESIGN_DECISIONS.md adoption protocol); failed adoption undoes its own pending HEAD move; quarantined versions kept indefinitely.

**Timeout behaviour:** Job-level timeout (`JOB_TIMEOUT_SECONDS` default 1800 s); lease fencing (`Job.worker_id`/`Job.attempts`) prevents stale workers writing.

**Resource requirements:** CPU + disk (WorldStore versions under `WORLDSTORE_ROOT`).

**Provenance:** version id = content hash (sha256 of world dict); run record (`build_run_record`: code commit, python, platform, seeds, colmap, pipeline profile/version, stages, lineage, models_available).

**Validation:** read-back verification (hashes clean); `_notify_world_version` (notification + activity event).

**Downstream consumers:** inspection routes, correction, exporters, next incremental registration (COLMAP session committed only after adoption).

**Defect note:** mandate defect #7 — engine availability != successful execution; `_assert_reconstruction_traced` prevents permanently untraceable state.

---

## Stage 13 — Inspection

**Owner:** `apps/api/routes_worlds.py` + `apps/api/routes_evidence_world.py` + `apps/api/routes_query.py` + `evidence_world_stage.readiness()`.

**Inputs:** World id (+ optional version id), tenant/team authorization.

**Outputs:** World summary (entities, geometries, scale_state, current_version_id), version report (stages: reconstruction, scale, depth, perception, mesh, dense_gate, dense_mvs, timings), evidence records, excluded evidence, readiness (stage_status, quality verdict, export_ready, metric_export_ready).

**Schema:** JSON; `readiness()` derives ONLY from persisted disk (snapshot, evidence matrix, persisted world); `physical_accuracy = "NOT_MEASURED"` unless measured.

**Required dependencies:** persisted world version.

**Error states:** unreadable snapshot -> `stage_status: "unreadable"`; no entities -> `state: "empty"`; partial placement/size -> `state: "partial"`; all placed+sized+complete -> `state: "complete"`.

**Timeout behaviour:** None.

**Resource requirements:** CPU + disk I/O (read version from WorldStore).

**Provenance:** nothing estimated/seeded/defaulted; unreadable or missing piece reported as such.

**Validation:** `world_calibration(w)` (one reader for evidence-world + legacy job's calibration record); `manifest.verify(root)` integrity check.

**Downstream consumers:** Studio viewer (SpatialWorkstation), correction UI, export.

---

## Stage 14 — Correction

**Owner:** `engine/commands/` (WorldEditCommand model) + `apps/api` correction routes + `world_ir.corrections` (`carry_corrections`, `carry_decisions`).

**Inputs:** World id + version id (`expected_version_id` for version-aware correction), edit command (merge/split/reclassify).

**Outputs:** New WorldIR version with correction applied; correction carry-over onto rebuilt world (F6 / T-4.2).

**Schema:** typed edit commands + undo; `expected_version_id` -> 409 if HEAD moved (correction never applied on a version the user never saw).

**Required dependencies:** HEAD version; user authorization.

**Error states:** 409 on version mismatch; correction replay failure visible on world metadata (`correction_carry`), never loses the rebuilt world.

**Timeout behaviour:** None.

**Resource requirements:** CPU.

**Provenance:** corrections are versioned world edits (basis tracked: visual / user_stated / ocr).

**Validation:** re-validate after correction; `expected_version_id` fencing.

**Downstream consumers:** new version (diff + export), `world_delta` entity continuity (identity carry-over keeps ids for same physical surface).

**Defect note:** mandate defect #8 — WorldIR and evidence-world representations may diverge; correction reachable via `/api/worlds/{id}/evidence-world/{run}/propose` then `/api/worlds/{id}/proposals/{pid}/accept` (frame/unit gate: HEAD must be exactly the version the same job committed beside this run).

---

## Stage 15 — New version (adoption)

**Owner:** `world_delta.decide()` + `world_delta.reconcile_conflicts()` + `apps/api/jobs.py` seg_changeset_commit adoption decision.

**Inputs:** Candidate world + HEAD snapshot + delta + conflicts.

**Outputs:** Verdict {ACCEPT, ACCEPT_WITH_UNCERTAINTY, REJECT} + reasons + uncertainties (each citing a measured fact); new version id if adopted.

**Schema:** `world_delta.snapshot(...)`; `decide(delta, conflicts)`; `physical_changes(delta, conflicts)`; `describe_changes(...)`.

**Required dependencies:** validation passed; `world_delta.is_unchanged()` check (no-op re-run -> no duplicate version).

**Error states:** REJECT -> HEAD stays authoritative, new evidence kept and retried; conflicts never silently resolved (both hypotheses kept with provenance until a later version agrees).

**Timeout behaviour:** None.

**Resource requirements:** CPU.

**Provenance:** camera alignment (Umeyama similarity) across versions; entity continuity by TYPE + POSITION in previous frame (ids re-derived each rebuild, ids useless for matching).

**Validation:** `world_delta.is_unchanged()` prevents meaningless duplicate versions; `reconcile_conflicts` carries unresolved conflicts forward (never drops).

**Downstream consumers:** WorldStore persistence (HEAD pointer), inspection, diff, export.

---

## Stage 16 — Diff

**Owner:** `world_ir.diff.diff_worlds` + `world_delta.physical_changes()` / `describe_changes()`.

**Inputs:** Two WorldIR versions (HEAD vs candidate).

**Outputs:** Per-region changes (added/extended/refined/preserved/represented_differently/not_reproduced/uncertain), conflicts (unresolved/resolved), camera movement, structure counts, entity continuity (preserved/refined/extended/reduced/split/merge/regrouped/ambiguous/removed/new).

**Schema:** `physical_changes` dict; `describe_changes` list of plain-language statements each derived from a measured field.

**Required dependencies:** both versions persisted; camera alignment for cross-frame comparison.

**Error states:** not measurable -> reported as such (never skipped silently); `topology_disconnected_rooms` WARNING (partial capture may honestly lack connecting evidence).

**Timeout behaviour:** None.

**Resource requirements:** CPU.

**Provenance:** every reason cites a measured fact; no scalar "quality score".

**Validation:** containment cycles / self-relationships ERROR; disconnected room groups WARNING (never error).

**Downstream consumers:** Studio change-set review, adoption decision, export readiness.

---

## Stage 17 — Export

**Owner:** `exporters/ifc_bridge.py` + `exporters/gltf/` + `exporters/floorplan.py` (+ usd/citygml/cityjson/unreal/blender/godot/habitat/ros/sumo/citydb/mesh/gis/report/profiles).

**Inputs:** Named WorldIR version id; export format.

**Outputs:** IFC / glTF / USD / DXF / PDF / CityGML / CityJSON / Blender / Godot / Habitat / ROS / SUMO / 3DCityDB / 3D Tiles file.

**Schema:** parts export as child nodes (glTF) / `IfcRelAggregates` (IFC); attributes as glTF `extras` + IFC property set `Pset_RealityEngine` with `Basis` and `Confidence`; unknown attributes export as absent with `Basis="unknown"` (never a default).

**Required dependencies:** persisted version; `readiness().export_ready` (placed entities + no verification problems); `metric_export_ready` only when every placed entity is `metric_anchored` + calibration valid/unvalidated.

**Error states:** FAILED/BLOCKED -> export refused; PARTIAL -> exports carry `X-Quality-Verdict`; no entity with measured position -> `export_not_ready_reason`.

**Timeout behaviour:** None.

**Resource requirements:** CPU + disk.

**Provenance:** export carries provenance/basis on every element; never invents geometry.

**Validation:** `validate_world_ir` clean on source version; round-trip through ifcopenshell.open / glTF loader (part count + attribute values survive).

**Defect note:** `exporters/boxes.py` states "WorldIR is Z-up" which contradicts the Y-up axis convention (DESIGN_DECISIONS.md) — latent inconsistency, not fixed here.

---

## Optional stages & omissions (explicit)

| Stage | Default | When it runs | Omission visible as |
|---|---|---|---|
| depth | disabled (`depth_model=None`) | `depth_model` set + trusted intrinsics + metric scale | `facts.status="skipped"` + note |
| perception | `perception_model` set (maskrcnn/grounded) | model available + intrinsics + metric depth + indoor gate | `facts.status="skipped"` + note |
| mesh | `mesh_enabled=True` but requires metric + artifact_store + enough points | metric world + store + >= points | `facts.status="skipped"` + reason |
| dense MVS | `dense_mvs_enabled=False` (OFF by default) | explicit opt-in OR `dense_auto` evidence-gated | `facts.status` in {disabled, skipped, rejected, equivalent} |
| detail | `detail_enabled=True` | trusted intrinsics + cameras | `facts.status="skipped"` + note |
| spatial_global | non-default profile only (`RE_SPATIAL_GLOBAL`) | env flag + spatial inputs provider | stage not in default profile |

## The single production path (API job)

`evidence_resolve` -> `evidence_quality` -> `reconstruct` (progressive ladder: multi-view rung runs fine stages pose/scale/depth/layout/perception/mesh/dense/detail/finalize via `ladder.run_fine_stages`) -> `quality_report` -> `compile_validate` -> `changeset_commit` (adoption + WorldStore persist + read-back verify).

The `evidence_world_stage` (Gemini's detection/segmentation/spatial lifting/association -> canonical -> WorldIR bridge) runs ONLY when `RE_EVIDENCE_WORLD=1` (opt-in, mandate defect #5) and never changes the job's grade.

## Cross-cutting invariants

- No model with a non-commercial or undeclared licence in the default path (`assert_production_licences`).
- Engine availability != successful execution (defect #7); a timed-out/cancelled/failed job never shows as success.
- No implicit conversions that conceal incompatible values (mandate §D); scale state / units stated on every entity transform.
- WorldIR and evidence-world representations diverge unless the opt-in propose/accept gate is used (defect #8).
- One object = one entity however many photos show it; a detection is an observation, never an entity (BUILD_SPEC rule 2).
