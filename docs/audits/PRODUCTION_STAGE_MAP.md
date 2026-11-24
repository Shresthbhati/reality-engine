# Production reconstruction path: stage map (Sprint 9 P0 audit)

An audit, not a plan. Every claim is tagged:

- **[CODE]** read from source, with `file:line` (not executed by this audit).
- **[OBSERVED]** measured by actually running the production job with `scripts/stage_map_probe.py`
  (raw data: `docs/audits/stage_map_run.json`).

"Production job" = the API job path: `POST /api/sessions/{id}/reconstruct` -> `enqueue_job` (`apps/api/jobs.py:421`)
-> worker (`apps/api/worker.py:26`, `jobs.worker_loop` `jobs.py:2192`) -> `process_next_job` (`jobs.py:1956`) ->
`_run_reconstruct_session_inner` (`jobs.py:649`). Benchmarks and tests are not the production path and are not used here.

PIPELINE_STAGES: evidence_resolve, evidence_quality, reconstruct, pose, scale, depth, layout, perception, mesh, dense, detail, finalize, quality_report, compile_validate, changeset_commit

(The line above is checked against `engine/pipeline/stages/registry.py:PIPELINE_PROFILES` by `tests/test_stage_map_invariants.py`.)

## 0. How the job is actually organised

1. **15 recorded stage rows** (`job_stage_runs`, `apps/api/stage_checkpoints.py:68`), run by `run_stages`
   (`jobs.py:1423`, `engine/pipeline/stages/runner.py`). Order is fixed by `PIPELINE_PROFILES`
   (`engine/pipeline/stages/registry.py:19`).
2. Only **five** of them do host work through "segments" closed over the job's mutable state
   (`jobs.py:1403`): `evidence_resolve` (716), `evidence_quality` (841), `reconstruct` (895), `compile_validate` (987),
   `changeset_commit` (1078). **[CODE]**
3. `reconstruct` runs the whole progressive ladder `run_progressive` (`jobs.py:968`,
   `engine/pipeline/progressive.py:206`), whose multi-view rung is `vertical_slice` -> `run_fine_stages`
   (`engine/pipeline/stages/ladder.py:58`): pose, scale, depth, layout, perception, mesh, dense, detail, finalize run
   **inside** `reconstruct`. The rows `pose ... finalize` are **replays** of those inner `StageResult`s
   (`LadderReport.run`, `ladder.py:98`). Consequence: `reconstruct`'s wall time CONTAINS the nine rows (do not add them), and
   if the multi-view rung is not used they are recorded as `skipped` with the ladder's reason (`ladder.py:103`). **[CODE]**
4. `quality_report` (`engine/pipeline/stages/s12_quality_report.py:64`) has no segment: it reads the mesh left in
   `ctx.state["ladder"]`. **[CODE]**
5. The **evidence-world flow** (real detection engines -> Evidence Matrix -> depth engines -> spatial lift -> TSDF fusion ->
   canonical entities -> a *separate* WorldIR) is **not part of the default job**: it runs inside `changeset_commit` only
   when `RE_EVIDENCE_WORLD=1` (`jobs.py:1208`, `apps/api/evidence_world_stage.py:34`). Its own enablement table says it is
   not ready to be default (`evidence_world_stage.py:42-59`: 4 criteria unmet). **[CODE]**
6. Coarse stages are never resumed from checkpoint: `DurableJobCheckpointStore.lookup` returns `None` for
   `evidence_resolve, evidence_quality, reconstruct, compile_validate, changeset_commit` (`jobs.py:609`). **[CODE]**

Default flags that change what runs (all **[CODE]**): `RE_EVIDENCE_WORLD` (off), `RE_EVIDENCE_WORLD_DEPTH_ENGINES` (empty),
`RE_EVIDENCE_WORLD_DETECTION` (`pipeline`), `RE_EVIDENCE_WORLD_LIFT` (`spatial`), `REALITY_OPEN_VOCAB` (on; `0` = Mask R-CNN,
`jobs.py:942`), `REALITY_DENSE_AUTO` (on; `engine/pipeline/dense_gate.py:44`), `REALITY_DENSE_QUALITY` (fast tier,
`jobs.py:284`), `RE_COLMAP_GPU` (auto-detect, `reconstruction/backend/colmap_config.py:133`), `RE_MEMORY_ADMISSION`
(`enforce`, `engine/hardware/admission.py`), `RE_STAGE_TIMEOUTS_S` (none, `jobs.py:1629`), `RE_SCENE_GATE`
(`vertical_slice.py:654`), `COLMAP_SUBPROCESS_TIMEOUT_SECONDS` (3600, `colmap_backend.py:631`),
`REALITY_TEST_BACKEND` (refused unless `REALITY_ENV=test`, `jobs.py:766-772`).

## 1. Stage-by-stage (the 15 requested stages)

Scale/frame vocabulary used below: **SfM gauge** (COLMAP's arbitrary unit) -> **working unit** (relative world, gauge rescaled
so median camera-to-point distance is a nominal value, `reconstruction/scene_scale.py:60`, then refined to a nominal eye height,
`vertical_slice.py:225`; "NOT metres") -> **estimated** (accepted typical-size prior) -> **anchored-unvalidated** ->
**validated** (`engine/multiengine/world_scale.py`).

### S1 Image ingestion
| | |
|---|---|
| Invoked by job? | Upload is a separate API call (`apps/api/routes_misc.py:210` `store_bytes`, `:255` enqueue `PROCESS_EVIDENCE`); the reconstruct job re-resolves via stage `evidence_resolve` (`jobs.py:716-839`). **[CODE]** |
| Contract | in: multipart image (<= `UPLOAD_MAX_BYTES`, default 1 GiB; magic-byte sniff `routes_misc.py:121`). out: `Evidence` row (`checksum`, `artifact_uri=sha256://...`), content store blob; at job time `resolved: [(Evidence, Path)]` minus operator-rejected, unresolvable and exact-duplicate files (`_drop_exact_duplicates` `jobs.py:481`). |
| Frame / scale | none yet |
| Backend | filesystem content store (`apps/api/storage.py`) |
| Failure | zero usable photos -> `InsufficientEvidence` (`jobs.py:833`) -> job FAILED; skipped items listed in `skipped_evidence`, graded PARTIAL (`jobs.py:1462`) |
| Hashes | SHA-256 of every upload (`Evidence.checksum`); `PROCESS_EVIDENCE` job re-hashes and compares (`jobs.py:524-526`) |

### S2 Image quality
| | |
|---|---|
| Invoked | stage `evidence_quality` (`jobs.py:841`): `inspect_image` per photo (`evidence/image_check.py:147`) then the **scene-identity gate** (`jobs.py:539`). **[CODE]** |
| Contract | out: `ImageFacts` (decode ok, size, EXIF orientation/make/model/35mm focal, `sharpness` = ranking not a calibrated blur score, `evidence_class`, `geometry_eligible`) stored in `Evidence.metadata_json["image_facts"]`; gate out: groups + `excluded{evidence_id: reason}`. |
| Frame / scale | none |
| Models | classical (PIL/OpenCV). Gate matcher `make_pair_verifier(matcher_name="auto")` = **LightGlue (+ALIKED) if available else SIFT** (`reconstruction/scene_identity.py:98-101`), thresholds `min_inliers=12`, `min_inlier_ratio=0.2` (provisional, `:114-115`). |
| Failure | non-image bytes never reach COLMAP; non-geometry classes become context-only; gate failure is recorded `state: unavailable` and nothing is excluded (`jobs.py:568-570`). No per-photo blur/exposure rejection: `sharpness` only ranks. |
| Hashes | none new (facts live in DB metadata) |

### S3 Camera calibration
| | |
|---|---|
| Invoked | `_intrinsics_options` before the solve (`jobs.py:361`); COLMAP single-camera / per-folder camera (`colmap_backend.py:786-818`); `adopt_solved_intrinsics` after the solve (`engine/pipeline/stages/s04_pose.py:32`). **[CODE]** |
| Contract | intrinsics priority: operator-declared (`POST /api/worlds/{id}/camera`) > EXIF 35 mm-equivalent focal (`fx=f35*diag/43.267`, principal point = image centre) > COLMAP's previous model > none (`jobs.py:366-373`). All recorded as trust `"estimated"` so COLMAP is **not** pinned (`jobs.py:404`); `trusted` pins focal+extra params (`colmap_backend.py:561-570`). |
| Frame / scale | pixels only; distortion ignored downstream (`jobs.py:324` caveat). |
| Backend | none (heuristic) + COLMAP's own self-calibration |
| Failure | no intrinsics -> depth, perception, mesh-from-depth **skip with a reason** (`vertical_slice.py:1160-1165`, `:1480`). |
| Hashes | camera record in report `camera_model` (not hashed) |

### S4 Feature extraction
| | |
|---|---|
| Invoked | inside `pose`/`reconstruct`: `colmap feature_extractor` (`reconstruction/backend/colmap_backend.py:836`, session path). **[CODE]** |
| Contract | in: copied images in the per-world COLMAP workspace (`ColmapSession`, `reconstruction/colmap_session.py`); out: SQLite `database.db` (keypoints + descriptors). |
| Backend | COLMAP SIFT with **affine shape + domain-size pooling** (`DEFAULT_ROBUST_SIFT=True` `:254`, args `:271-281`), which is **CPU-only**, so extraction is on CPU even when a GPU is present (`:821`). |
| Failure | COLMAP step error -> `ReconstructionStepError` -> `VerticalSliceError` -> ladder falls back to single-image level (`vertical_slice.py:392-393`, `progressive.py:253`). Timeout `COLMAP_SUBPROCESS_TIMEOUT_SECONDS`=3600. |
| Hashes | none (workspace is mutable; `ColmapSession` manifest records image names, not hashes) |

### S5 Matching
| | |
|---|---|
| Invoked | `colmap exhaustive_matcher` (`colmap_backend.py:838`). **[CODE]** |
| Backend | exhaustive (O(n^2) pairs), GPU if `detect_colmap_gpu` says CUDA build + >= `MIN_FREE_VRAM_MB`=1536 free (`colmap_config.py:109,133`), else CPU. `--FeatureMatching.num_threads 1` when "deterministic" (`:839`). Guided matching off (`:255`). |
| Duplicate | the scene gate (S2) does its **own** LightGlue/SIFT matching of the same photos (`scene_identity.py:98`); its matches are not reused by COLMAP. |
| Failure | as S4 |
| Hashes | none |

### S6 Structure from motion
| | |
|---|---|
| Invoked | stage `pose` -> `run_pose` (`vertical_slice.py:340`): `ReconstructionOrchestrator(backends=[backend])` (`:389`) -> `ColmapReconstructionBackend._reconstruct_in_session` (`colmap_backend.py:747`). **[CODE]** |
| Contract | out: `ReconstructionResult(points, camera_poses, registration_status)` in **raw COLMAP gauge**, all sub-models merged (`_parse_sparse_dir` `:703`). First job of a world: `mapper` with up to 3 attempts (seed 0, seed 1, relaxed thresholds; retried while < 80% registered; `MAPPER_ATTEMPTS` `:261-268`, `full_mapper` `:843`). Later jobs: `image_registrator` + `point_triangulator` + `bundle_adjuster` vs a full rebuild, judged as worlds by `WorldArbiter` (`:907-969`, `vertical_slice.py:382-388`). |
| Frame / scale | **SfM gauge**, COLMAP's frame |
| Backend | COLMAP binary on PATH (`shutil.which`, `colmap_backend.py:486`). `reconstruction_backend_type="python"` (`reconstruction/backend/python_backend.py`) exists but is never selected by the job (`vertical_slice.py:373`). The orchestrator is constructed with ONE backend, so its fallback chain is unused (`reconstruction/orchestrator.py:301-391`). |
| Failure | `registration_status=="failed"` or 0 points -> `VerticalSliceError` (`vertical_slice.py:413-418`) -> ladder falls back to single-image bootstrap (level 0/1, PARTIAL) or, if even that fails, job FAILED (`progressive.py:343-348`). Output validation `validate_reconstruction_output` (`reconstruction/validation.py:120`). |
| Hashes | none for the COLMAP model; `run_record` records seed + versions (`jobs.py:1497`) |

### S7 Depth
| | |
|---|---|
| Invoked | stage `depth` -> `_depth_stage` (`vertical_slice.py:1125`). Evidence-world depth (`evidence_world_stage.py:639 _run_depth`) only when `RE_EVIDENCE_WORLD_DEPTH_ENGINES` is set. **[CODE]** |
| Contract | job path: in: photos + intrinsics + SfM result; out: per-view depth maps, each affinely "metricized" against the sparse points (`reconstruction/depth_to_points.py:330 metricize_all`), unprojected at stride 16 and **appended in place to `result.points`** (`vertical_slice.py:1191-1208`). Evidence-world path: `metric3d.v1` records with raw `.npy`, state `RELATIVE_UNSCALED`, cached by checkpoint hash (`evidence_world_stage.py:592-636`). |
| Frame / scale | job path: depth aligned to the **working-unit** SfM points (scale-by-alignment, residual recorded). Evidence-world: SfM gauge. |
| Models | job: MiDaS `DPT_Hybrid` (`VerticalSliceOptions.depth_model`, `vertical_slice.py:104`; `depth_anything*` names switch to `DepthAnythingV2Backend`, `:1175`). Evidence-world engines `midas`, `depth_anything` (Depth-Anything-V2-Small-hf, `adapters/depth_adapters.py:129`); metric-indoor / V3 adapters exist (`:272`, `:281`) but are not in the default list. RGB-D sidecar depth beats both (`vertical_slice.py:1145`). |
| Failure | never raises: skip with a note (no intrinsics, backend failure, no alignable views) (`:1149-1189`) -> `depth` stage `skipped`/degraded -> PARTIAL. |
| Hashes | job path: none. Evidence-world: raw-array artifact refs + `checkpoint_sha256` when the HF file is found (`:601-613,652-655`) |

### S8 Lifting (2D detections -> 3D)
| | |
|---|---|
| Invoked | stage `perception` -> `_perception_stage` (`vertical_slice.py:1437`), default `perception_model="grounded"` (`jobs.py:942`). Evidence-world: `run_spatial_stage` (`engine/multiengine/spatial_stage.py:169`) when depth engines are requested. **[CODE]** |
| Contract | job: Grounding DINO tiny boxes + SAM2 masks + DINOv2 appearance -> `lift_region_to_3d` against the metricized depth maps -> `perception.instances.association.resolve_hypotheses` -> entities (`:1475-1560`). Skips with a reason when there are no metricized depth maps (`:1471`). Evidence-world: multi-view ray triangulation in a Z-up working frame, mapped back (`spatial_stage.py:8`). |
| Frame / scale | job: working unit (relative world) or metres (anchored); depth labelled `DepthSource.MVS` "metric-by-alignment, approximate" (`:1541`) even when the world is RELATIVE. Evidence-world: SFM_GAUGE; the stage raises if gauge would become metric (`spatial_stage.py:285-286`). |
| Models | Grounding DINO tiny, SAM 2 (box prompt), DINOv2-small; `REALITY_OPEN_VOCAB=0` -> Mask R-CNN (COCO). |
| Failure | never raises; skip note. Single-view-only objects dropped by `filter_implausible` (`:1562-1575`). |
| Hashes | none |

### S9 Fusion
| | |
|---|---|
| Invoked | stage `dense` -> `run_dense` (`vertical_slice.py:561`): COLMAP dense MVS (auto-gated, judged) -> dense-structure pass -> CPU plane sweep (off) -> `_fusion_stage` (`:1794`, **metric worlds only**). Stage `mesh` -> `_mesh_stage` (`:2164`, Poisson, **metric worlds only**). Evidence-world: Open3D TSDF over aligned depth (`spatial_stage.py:414 _fuse_depth`) -> mesh artifact. **[CODE]** |
| Contract | job: sparse + depth-unprojected + dense-MVS points fused by track-id source (`:1815-1830`) into a POINTCLOUD geometry; mesh artifact into the artifact store. |
| Frame / scale | both gates skip when `scale_state != "metric"` (`:1818`, `:2191`), so **a relative world gets no fusion and no mesh from the job path**. Dense MVS is not gated on scale. |
| Models | COLMAP `patch_match_stereo` + `stereo_fusion` (fast tier: 800 px, 4 source views, photometric only, `jobs.py:284-291`), Poisson meshing via COLMAP `poisson_mesher`, Open3D TSDF (evidence-world). |
| Failure | each sub-step returns a status dict (`ran/skipped/failed/rejected/equivalent`); a refused dense run leaves the sparse level (`progressive.py:296-310`). |
| Hashes | meshes/pointclouds are content-addressed in `FileArtifactStore` (`geometry.data_hash`); evidence-world mesh `sha256` recorded and re-verified in readiness (`evidence_world_stage.py:518-523`) |

### S10 Geometry validation
Multiple layered gates **[CODE]**: `validate_reconstruction_output` (`vertical_slice.py:419`); `world_ir.validation.validate_world_ir`
(hard fail, `jobs.py:1053-1058`); provenance hard gate `_assert_reconstruction_traced` (`jobs.py:1065`); architectural constraints
as degradations not failures (`jobs.py:1067-1068`); the **adoption gate** `world_delta.decide` -> REJECT keeps the previous HEAD
(`jobs.py:1084-1198`); read-back hash verification before the HEAD moves (`worldstore_service.py:271`, `jobs.py:1342`).
Evidence-world: `_withhold_unresolved_duplicates` (`evidence_world_stage.py:300`), `persist_world` validation + reload compare
(`:344-390`), `engine/multiengine/quality_gate.assess` -> READY/PARTIAL/BLOCKED/FAILED (`:586`). No gate checks metric accuracy
against any ground truth at job time.

### S11 Spatial relationships
Three independent producers **[CODE]**: (a) the compiler writes CONTAINS/PART_OF/CONNECTS between rooms, walls, openings,
corridors (`engine/compiler/world_compiler.py:490-900`); (b) `perception.spatial_reasoning.infer_spatial_relationships`
(ATTACHED_TO / RESTS_ON / CONTAINS) in `finalize` (`vertical_slice.py:778-785`); (c) evidence-world
`make_scene_relations_fn` (`spatial_stage.py:440`, spatial_intelligence package, thresholds calibrated for metres but fed gauge,
`spatial_stage.py:15-16,43-47`).

### S12 WorldIR
Job path: stage `layout` -> `compile_reconstruction_to_world` (`vertical_slice.py:686-693`; planes -> classification -> rooms ->
validation gate); `finalize` stamps `metadata.scale/frame/depth/perception/mesh/fusion` (`:758-800`); `compile_validate` stamps
provenance, validates, stores the world JSON content-addressed and publishes `outputs={"world_ir": sha}` (`jobs.py:1074-1076`).
Evidence-world builds a **second, different WorldIR** (`evw-<job>`) from the Evidence Matrix via `worldir_bridge`
(`evidence_world_stage.py:361`) that is not the committed version.

### S13 Persistence
`worldstore_service.commit_version` (`apps/api/worldstore_service.py:199`): WorldStore version JSON + content-addressed world
artifact (`worldstore/store.py:190-254`), `verify_version`, then HEAD pointer + DB mirror row with the full report, `points.ply`
and `cameras.json` blobs as `sha256://` URIs (`:277-312`). Evidence runs persist under
`<WORLDSTORE_ROOT>/evidence-world/<team>/<world>/<job>/` with their own WorldStore, snapshot, run manifest
(`content_sha256`) and `committed_version.json` linking to the committed version (`evidence_world_stage.py:393-406`).

### S14 Viewer
API: `/api/worlds/{id}/points` (decodes the WorldIR geometries' own artifacts, **not** the stored `points.ply`,
`routes_worlds.py:920-1000`), `/cameras` (`:1066`), `/worldir` (`:676`). Clients: Next.js `frontend/src/lib/viewport/loaders.ts`,
`SpatialWorkstation.tsx`; standalone `apps/viewer/src/main.js`.

### S15 Exports
`POST /api/worlds/{id}/export` (`apps/api/routes_export.py:149`) -> `sdk.reality.export` (`sdk/reality.py:253`) /
`export_profile` (`:298`) -> `exporters/*` (gltf, usd, blender, cityjson, citygml, ifc, geojson, sdf, tscn, unreal, sumo, habitat,
godot profile, floorplan). Each export result carries `scale_block(world)`. Evidence-world has its own export route
(`routes_evidence_world.py:158`) and `metric_export_ready` (`evidence_world_stage.py:569`).

OBSERVED_PLACEHOLDER
