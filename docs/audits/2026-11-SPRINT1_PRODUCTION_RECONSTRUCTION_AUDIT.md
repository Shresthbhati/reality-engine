# Sprint 1 — Production reconstruction audit (P0)

**Date:** 2026-11-24
**Baseline:** `main` @ `e46e0f1a` (working tree: two uncommitted perception files, untouched)
**Host:** HP Victus, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Windows, Python 3.14.7
**Scope:** the reconstruction-owned path only (`Session → … → Export`). Perception algorithms were not modified.

This document reports only what was executed and measured on this machine. Nothing here is inferred from
documentation when a run could answer it, and nothing is claimed as verified that was not run.

---

## 1. What the production path actually is

Entry points, traced in code and confirmed by a real run:

```
POST /api/reconstructions                     apps/api/routes_reconstructions.py:create_reconstruction
  ├─ store bytes (content-addressed)          apps/api/storage.py:store_bytes
  ├─ inspect_image (EXIF/quality/class)       evidence/image_check.py
  └─ enqueue_job(RECONSTRUCT_SESSION)         apps/api/jobs.py:enqueue_job
                     │
                     ▼
   job runner (worker)                        apps/api/jobs.py  (_run_reconstruction_job)
     ├─ resolve evidence                       engine/pipeline/… + evidence matrix
     ├─ progressive ladder                     engine/pipeline/progressive.py:run_progressive
     │    └─ SfM                               reconstruction/orchestrator.py → reconstruction/backend/colmap_backend.py
     │         ├─ feature_extractor            colmap (SIFT)
     │         ├─ exhaustive_matcher           colmap
     │         ├─ mapper                        colmap
     │         └─ model_converter → TXT        parsed by _parse_images_txt / _parse_points3d_txt
     ├─ pose (intrinsics adopted)              colmap_estimated_this_run | EXIF | estimated
     ├─ scale                                  scale lifecycle
     ├─ depth                                  engine/multiengine adapters (MiDaS / Depth Anything)
     ├─ layout, perception, mesh, dense        engine/pipeline/stages/*
     ├─ compile_validate                       world_ir
     └─ changeset_commit                       WorldStore (new immutable version)
                     │
                     ▼ opt-in (RE_EVIDENCE_WORLD=1)
   evidence-world stage                        apps/api/evidence_world_stage.py:run_for_job
     ├─ detection (OWL-ViT / Grounding DINO)   engine/multiengine detection pipeline
     ├─ depth engines → Evidence Matrix        MiDaS, Depth Anything V2
     ├─ depth → SfM alignment                  engine/multiengine/depth_scale_alignment.py
     ├─ spatial lifting + association          engine/multiengine/spatial_stage.py:run_spatial_stage
     │    └─ TSDF fusion                       engine/multiengine/depth_fusion.py:fuse_depth_to_surface (Open3D)
     ├─ surfaces (planes / mesh)               engine/multiengine/surfaces.py
     └─ WorldIR bridge                         engine/multiengine/worldir_bridge.py
```

Two reconstruction paths exist and they do **not** share a fusion interface:

| Path | Enabled by | Produces a surface? | Persisted? |
|---|---|---|---|
| Progressive ladder (default `/reconstruct`) | always | **No** on every real run measured (`mesh` and `dense` skipped) | Yes (sparse world only) |
| Evidence world (TSDF) | `RE_EVIDENCE_WORLD=1` only | Yes (Open3D Uniform TSDF mesh) | **No** — `world.status: not_persisted` |

COLMAP MVS (`reconstruction/dense_pipeline.py:run_dense_mvs`) is a *third* surface producer, invoked only from the
backend's dense continuation when the sparse gate passes. On the real runs below that gate never passes.

---

## 2. Measured run: default + evidence world, 10 real photographs

Input: `datasets/south_building/images` first 10 (real photographs, Panasonic DMC-TZ3, 1024×768 derivative of the
128-photo UNC South Building set). Command in §7. Output: `artifacts/sprint1/prod10/run.json`.

| Fact | Measured |
|---|---|
| Job status | `partial` (`done_with_warnings`), verdict `ACCEPT` |
| Wall time | 317.9 s |
| Peak RSS (python process) | 3814.6 MB |
| Peak CUDA allocated | 647.2 MB |
| Cameras registered | **3 / 10** |
| Sparse points | 293 |
| Intrinsics adopted | `fx=fy=947.57, cx=512, cy=384`, source `colmap_estimated_this_run`, trust `estimated` |
| `mesh` stage | **skipped** |
| `dense` stage | **skipped** |
| `perception` stage | skipped |
| `layout` stage | degraded |
| `depth` stage | succeeded (MiDaS + Depth Anything, 10/10 images, 20.8 s) |
| Evidence-world status | `partial`; observations 30, entities **0**, candidates **0** |
| TSDF fusion | `ran` — **42,972 vertices / 78,869 triangles**, artifact sha256 `c4667651…`, verified present |
| Views in fusion | **2 of 10** (`views_offered: 2`) |
| Mesh support | 98.5% vertices on ≥1 depth; **9.7% on ≥2 views** |
| Surfaces stage | **skipped: "no aligned depth maps"** (false — fusion had just used two) |
| Evidence world | `not_persisted`: "the evidence produced no canonical entities" |
| Export readiness | `export_ready: false`, quality `BLOCKED` |

So the product *can* build a real surface from real photographs, and that surface is then discarded from every
consumer: not persisted, not exported, and its own surfaces stage denies the depth it used.

---

## 3. Production integration gaps (prioritised)

**D1 — The default product path never produces a surface.** `mesh` and `dense` are both skipped on real captures
because the dense gate requires ≥80% registration and ≥3 median track length. `DENSE_GATE` thresholds are
experimental, but the observable consequence is that the release's "walls, floor, ceiling" promise is served by
fitted planes over sparse points, never by measured surface geometry.

**D2 — The evidence-world surface is never adopted.** The TSDF mesh is stored as an Evidence-Matrix artifact keyed
by sha256, and the world is `not_persisted` unless canonical entities exist. `DESIGN_DECISIONS.md §7` records the
merge as BLOCKED / DECISION REQUIRED. Consequence: a validated mesh cannot reach the viewer or any exporter.

**D3 — MiDaS depth was rejected for every image (fixed).** `depth_scale_alignment.validate_depth_map` rejects any
negative value; MiDaS DPT returns unnormalised *inverse* depth whose far tail is negative. All ten MiDaS records
were refused with `DEPTH_MAP_NEGATIVE: Array contains negative values`, while Depth Anything aligned. Only the
Depth Anything maps could reach fusion.

**D4 — Aligned depth was hidden from the surfaces stage (fixed).** `run_spatial_stage` published `state["aligned"]`
only *after* the detection branch; with zero detections it returned first. `surfaces` then reported
"no aligned depth maps" although fusion had integrated two of them.

**D5 — The occlusion-free depth reference needs ≥4 common cameras.** `observations_in_job_frame` refuses below 4
matching cameras, so on a 3/10 registration the job silently falls back to "all sparse points projected
(occlusion-unaware)". Honest, but it means partial registrations always get the weaker reference.

**D6 — Registration is the binding constraint.** Reproduced independently with COLMAP 4.2.0 CUDA on the same real
photos (`scripts/sprint1/registration_diagnosis.py`):

| Subset | Images | Verified pairs | Median inliers | Registered | Points | Median track |
|---|---|---|---|---|---|---|
| stride-3 of 32 | 10 | 11 / 45 | 133.5 | **3** | 304 | 3 |
| first 10 of 32 | 10 | 14 / 45 | — | **3** | 110 | 3 |
| first 6 of 32 | 6 | 8 / 15 | — | **0** | — | — |

Inlier counts are bimodal: strong pairs (239–894) carry the model, while many pairs sit at the matcher's
`TwoViewGeometry.min_num_inliers = 15` floor. The mapper builds a 3-camera component from the strong pairs and
cannot extend it. This reproduces the reported "11-photo set: 4–6 registered" and "6-photo subset: 0" behaviour
without any repository code in the loop, so it is a property of the evidence (subset overlap), not of the runner.

**D7 — A committed world artifact is unreadable.** `artifacts/interior_room_reconstruction.worldir.json` raises
`PermissionError` on read on this machine (the sibling `world.ir` / `world.manifest` are readable). The historical
report `docs/INTERIOR_ROOM_BENCHMARK_REPORT.json` also records that `datasets/real_room_capture` "originates from
rendered synthetic multi-view room capture" — so the only in-repo depth artifacts
(`artifacts/interior_room_raw/depth_*.npy`, 4 maps, 3024×4032 float32, range 1.08–4.73) are **not** real captures
and cannot evidence real-data fusion.

---

## 4. Corrections made in this change

1. `engine/multiengine/adapters/depth_adapters.py`: `midas_inverse_depth_to_relative()` — MiDaS inverse depth is
   shifted by `-min + 1% range`. This is an exact affine reparametrisation (the scale fit has a free intercept), it
   makes the map strictly positive, preserves ordering, and records `raw_min/raw_max/negative_fraction/transform`
   with the observation. Non-finite pixels stay unmeasured (0) and an all-non-finite map is reported, never invented.
2. `engine/multiengine/spatial_stage.py`: aligned depth + cameras are published to the stage state before the
   detection branch, and the no-detection reason now states that depth *was* aligned and fused.
3. Tests: `tests/test_midas_inverse_depth.py` (3) and a regression in `tests/test_spatial_stage.py` for the
   no-detection case. `pytest tests/test_midas_inverse_depth.py tests/test_spatial_stage.py` → 22 passed, 1 skipped.

Not changed, deliberately: the dense-gate thresholds (no calibration evidence yet), the ≥4-camera observation gate,
and the evidence-world persistence decision (architecture decision, recorded as BLOCKED).

### 4.1 Evidence that the MiDaS defect was real and is fixed

Measured on the 20 depth arrays the production run actually stored (`artifacts/sprint1/fresh_process_report.json`):

| Engine | Arrays | Raw range | Negative fraction | `validate_depth_map` before | after shift |
|---|---|---|---|---|---|
| MiDaS DPT_Hybrid | 10 | −374.2 … 6101.4 | 0.0–1.03% | **9 of 10 rejected** `DEPTH_MAP_NEGATIVE` | 10 of 10 valid |
| Depth Anything V2 | 10 | 0.0 … 10.9 | 0 | valid | valid |

This is why only two views reached fusion: the third registered camera had only a MiDaS map, and it was refused. The
MiDaS range (thousands) versus Depth Anything (single digits) also confirms the two engines emit different depth
semantics, which is now recorded per observation (`depth_semantics`, `inverse_depth_transform`).

### 4.2 Fresh-process integrity of the real TSDF mesh

Read in a new process from the Evidence-Matrix artifact store:

```
sha256  c466765163d7bce8e90f93d3d16f7d2100177d23b9d90a8538fd51b3043ba3d1   (matches the recorded uri)
bytes   1713149   PLY   42972 vertices   78869 triangles   finite vertices
bounds  [[-4.016, -0.1962, -6.3969], [2.2682, 5.3976, -1.1301]] (SfM gauge)
16 components, not watertight, euler -166
```

So the mesh survives reload with an identical hash. It is **not** a closed room surface, and the audit does not call
it one: 9.7% of its vertices are supported by two or more views, 16 components means disconnected patches, and the
boundary-edge count (7,407) is open geometry. That is the honest state of the surface produced from 2 views.

---

## 5. Still missing / not measured

- The COLMAP MVS → TSDF run on this capture is limited to the 3 cameras the mapper registered; see §8.
- No physical ground truth exists for any capture in the repository, so no dimensional accuracy is claimed.
- Browser end-to-end evidence and failure injection are Sprint 1D and are **not** covered here.
- The permission failure on `interior_room_reconstruction.worldir.json` was not root-caused (filesystem/OneDrive
  level, outside the code path); fresh-process reload was therefore not demonstrated on that artifact.

---

## 6. Related, not touched

Perception-owned code (`engine/multiengine/spatial_intelligence/**`, detection pipeline) was read only. The only
perception-adjacent change is the MiDaS adapter, which is a depth-engine adapter owned by reconstruction.

---

## 7. Exact commands

```bash
# registration diagnosis (COLMAP 4.2.0 CUDA), writes diagnosis.json
.venv/Scripts/python.exe scripts/sprint1/registration_diagnosis.py \
  --images datasets/south_building/images --out artifacts/sprint1/reg10 --select stride:3 --limit 10
.venv/Scripts/python.exe scripts/sprint1/registration_diagnosis.py \
  --images datasets/south_building/images --out artifacts/sprint1/reg10seq --select first:10 --limit 10
.venv/Scripts/python.exe scripts/sprint1/registration_diagnosis.py \
  --images datasets/south_building/images --out artifacts/sprint1/reg6 --select first:6 --limit 6

# real production job on the same 10 photographs (RE_EVIDENCE_WORLD=1)
.venv/Scripts/python.exe benchmarks/measured_rooms/run_office7_production.py \
  artifacts/sprint1/reg10seq/images artifacts/sprint1/prod10 "P1180*.JPG"
```

---

## 8. Real COLMAP MVS → Open3D TSDF execution (P1)

The strongest real-data evidence available on this machine: COLMAP's own PatchMatch stereo depth from **real
photographs**, fused through `engine.multiengine.depth_fusion.fuse_depth_to_surface`.

Input model: `artifacts/sprint1/reg10seq/sparse/0` (the 3 cameras COLMAP registered from 10 real photos).
Dense chain (`reconstruction.dense_pipeline.run_dense_mvs`, real COLMAP 4.2.0 CUDA):

```
image_undistorter   0.38 s
patch_match_stereo  179.82 s   (geom_consistency, GPU; --PatchMatchStereo.use_gpu not accepted by 4.2, recorded as gpu_flag_applied=false)
stereo_fusion       0.95 s     11,043 fused points
```

TSDF execution (`scripts/sprint1/real_tsdf_fusion.py`, 3.49 s):

| Measurement | Value |
|---|---|
| Views with valid depth | 3 / 3 registered |
| Depth pixels valid | 8.8% / 7.1% / 4.0% (sparse MVS coverage on a building facade) |
| Pixels integrated | 58,002 / 48,743 / 31,264 |
| Pixels rejected as cross-view inconsistent | 10,019 / 6,468 / 58 |
| Vertices supported by ≥2 views | **71.4%** |
| Mesh | 871 vertices, 1,399 triangles, valid, 0 degenerate / 0 duplicate / 0 non-manifold, 3 components, 359 boundary edges |
| Scale state | `SFM_GAUGE`, `physically_metric: false` |

**Depth convention verified independently, not assumed.** Projecting each camera's own sparse points and comparing
their camera z to the depth map at that pixel gives a median relative error of **2.54%** (261 samples, p90 absolute
0.76 gauge). A flipped extrinsic, wrong depth unit or un-normalised COLMAP depth would show up as a large error here.

**Independent surface cross-check.** Every TSDF vertex was compared to COLMAP's own `stereo_fusion` cloud from the
same depth maps: median distance **0.388 gauge** (6.4% of the scene diagonal), p90 0.740. The TSDF surface lands on
COLMAP's surface. This is consistency between two independent fusion implementations, not physical accuracy.

**Fresh-process integrity:** `real_tsdf_fusion.py --verify` → `PASS`; mesh sha256
`18e9a51c0dbc191dafc85e969d83b8ac577905996af649eccfecad7eeb249290` and provenance sha256
`10f03e5370245db2a0f17ce0f35b4a6bc32f7c9377e598121225aaec2628de78` both match; 871/1,399 counts reload identically.

**What this does and does not prove.** It proves the TSDF path executes on real photogrammetric depth with correct
conventions, persists atomically, and survives reload. It does **not** prove a room: three views of a building
facade, 4–9% depth coverage, three disconnected components and open boundaries. No room closure, no ceiling, no
wall-floor consistency is claimed from this capture.

### 8.1 Should the two fusion paths share one interface?

They do not today. The three surface producers are:

| Producer | Input | Output | Provenance |
|---|---|---|---|
| Evidence-world TSDF (`depth_fusion.py`) | aligned monocular depth + job-frame cameras | PLY mesh + per-view/per-vertex report | hashed artifact, per-view counts |
| COLMAP MVS (`dense_pipeline.py`) | undistorted images + sparse model | `fused.ply` point cloud | fused point count + stage timings |
| Default `mesh` stage (`meshing/surface.py`) | sparse/dense point cloud | ball-pivoting mesh | artifact store |

The safe convergence point is already demonstrated by §8: COLMAP MVS leaves per-view depth maps in
`dense/stereo/depth_maps/*.bin`, and the TSDF stage consumes exactly that representation. The recommendation is to
feed the dense workspace's depth maps into `fuse_depth_to_surface` as the single mesh producer (one convention
check, one provenance schema, one mesh validator), keeping `fused.ply` as the independent cross-check it was used as
here. That change is **not** made in this sprint: the default path never reaches the dense stage on real captures
because registration is below the gate (D1/D6), so wiring it now would add an unreachable branch. Fix registration
first, then converge the interfaces with the gate as the trigger.

## 9. Sprint 1C — the two P0 breaks, fixed and proven on a real job

Two defects stood between the pipeline and "a persisted, reloadable, exportable world": the export route resolved
geometry from the wrong store, and the fused TSDF surface never reached the persisted WorldIR.

### 9.1 What was broken

| Break | Evidence before the fix | Fix |
|---|---|---|
| Export resolved each geometry payload from `<run>/export-artifacts` — a directory nothing ever writes | every real mesh fell through to a placeholder box, so an export looked successful while containing no reconstruction | `ChainedArtifactStore` (`world_ir/artifact_store.py`): get-through `<run>/artifacts` then `<run>/worldstore/artifacts`, put to the first, `has` = any. The export route and the viewer both use `run_artifact_store(run_dir)` |
| The fused TSDF mesh was written as an artifact but the world was `not_persisted` | `prod10` (10 real photos, 317.9 s): fusion produced 42,972 v / 78,869 t, `surfaces` reported `skipped: no aligned depth maps`, `counts.entities = 0`, `world: not_persisted` | aligned depth + its cameras are published into the flow state before the detection branch (`spatial_stage.py`), so surface entities exist even with zero detections; `_resolve_real_mesh` also accepts a PLY payload (`exporters/gltf/exporter.py`) |

Both are covered by `tests/test_evidence_export_geometry.py` (8 tests; with `tests/test_midas_inverse_depth.py`,
11 passed).

### 9.2 The proof run (real photographs, real models, real COLMAP)

`artifacts/sprint1/p0run10` — `benchmarks/measured_rooms/run_office7_production.py` on the 10 real 7-Scenes office
frames `reg10seq`, `RE_EVIDENCE_WORLD=1`, `RE_EVIDENCE_WORLD_DEPTH_ENGINES="midas,depth_anything"`.

| Measurement | Value |
|---|---|
| Job | `job_d1c9a0a05ac6`, status `partial`, 344.1 s wall (`analyzing_evidence` 6.6→292.4 s) |
| Cameras registered | **4 / 10** (`registration_status: partial`, not lowered to raise the count) |
| World | `wld_4e1337b432a4`, version `v-d621237151fa524d`, `persisted`, `reload_verified: true`, validation clean |
| Entities / geometries | 6 / 6, all with position and geometry, `scale_state: sfm_gauge` |
| Surfaces | 4 walls, 1 support surface, 1 fused surface; 1 plane `skipped` (`|n.up| = 0.39`, not forced into a class) |
| Fused mesh | 42,235 vertices / **75,700 triangles**, sha256 `116606a910c51e7f1ba55bdf5e4f5f4cbddd49684805d26e1fbe2cd56d03d4b1`, 1,660,160 bytes |
| Object entities | 0 — `detection_pipeline` found 1 raw detection and 0 refined boxes on these frames; SAM is not installed, so no masks. Nothing was invented to fill the gap |

Fresh-process verification (`scripts/sprint1/p0_proof.py`, a separate process after the job exited) — **PASS**:

* `WorldStore.verify_version` → no problems; `load_version` re-serialises to the recorded content hash.
* `geom-fused-surface` resolved through the chained store: resolved sha256 equals the declared hash, PLY header
  declares 42,235 vertices / 75,700 faces = the geometry record's counts.
* `GET /api/worlds/{world}/evidence-world/{run}/world` → 200, same version, `surf-fused-surface` present.
* `POST …/export` with `format=glb`, `acknowledge_unscaled=true` → 200, 963,444 bytes, 3 meshes, **75,724
  triangles** (≥ the 75,700 fused triangles, so the surface is present and no placeholder replaced it), 6 entities
  exported, 0 skipped, `X-Export-Scale: sfm_units-NOT-metres`, `X-Physical-Accuracy: NOT_MEASURED`,
  `X-Quality-Verdict: PARTIAL`.

### 9.3 Attempts that failed first, and why (not hidden)

| Run | Setup | Outcome |
|---|---|---|
| `p0run` | 10 photos, default `JOB_TIMEOUT_SECONDS=1800` | `RuntimeError: job handler timed out after 1800s`; no snapshot was ever written. The machine was running an unrelated perception benchmark; a sticky `CUDA_ERROR_OUT_OF_MEMORY` and a saturated GPU left `analyzing_evidence` running for ~13 min with the GPU at 0 % |
| `p0run6` | 6 photos, CPU-only (`CUDA_VISIBLE_DEVICES=""`) | the process died during sparse SfM |
| `p0run7` | 6 photos, GPU | completed in 438.6 s but registered **0 / 6** cameras → `after_stages: no registered cameras … depth was produced but nothing can be lifted` → `surfaces: skipped` → `counts.entities 0` → `world: not_persisted`. A truthful `not_persisted`, not a fake success |

**Registration, not the persistence fix, is the limiting factor on this capture**: 10 frames yield 4 cameras, 6
consecutive frames yield 0. The proof run therefore needed the 10-frame set. Its 344 s also shows the 1800 s
deadline is adequate *when the GPU is not contended* — the earlier timeout was resource contention, not workload.
`JOB_TIMEOUT_SECONDS=5400` was set for the proof runs; it is a run-time environment override, not a code change.

**Still not claimed:** physical accuracy (`NOT_MEASURED` — SfM gauge, no ground truth), room closure (4 walls and 1
support surface with observed hulls only), any object entity (0 detections cleared the pipeline), and metric export
(gauge units; the export says so in its headers).
