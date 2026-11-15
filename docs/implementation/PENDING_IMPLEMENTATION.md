> **SUPERSEDED (2026-11-10).** Current status lives in [docs/STATUS.md](../STATUS.md); current plan in docs/blueprint/34_MASTER_PLAN.md. Kept for history.

# Reality Engine — Pending Implementation (Master Backlog)

> Capabilities that are architecturally intended but not yet fully
> implemented and verified. This is an engineering backlog, not a claim
> that these features exist. Every entry names the evidence required to
> advance its status; a status may only move when that evidence exists.

Last reconciled: 2026-10-31 (room fidelity items S/D/O/M/V/U, site composer and landmark rows added; earlier rows 2026-09-14: mapping campaign; PRs #11, #12, #14, #16 merged; #17/#18 by concurrent agents; #19 open). P0-P8 status lines were also reconciled 2026-10-17 against `.agent/TASKS.yaml` (the P0.x entries and the PROD/CE/PT tables were not re-audited there).

Detailed per-subsystem specifications live in [`docs/future/`](../future/).
Architecture rationale lives in [`../engineering/DESIGN_DECISIONS.md`](../engineering/DESIGN_DECISIONS.md).
The priority order is defined in [`ROADMAP.md`](ROADMAP.md) — it is the
authoritative WHAT; agents determine the HOW.

---

# Status Definitions

| Status | Meaning |
|---|---|
| VERIFIED | Executable implementation + integration + meaningful verification (unit and/or real-data as specified) |
| IMPLEMENTED | Working implementation and tests; broader verification pending |
| PARTIAL | Significant implementation exists but important functionality remains |
| FOUNDATION | Interfaces/schema/infrastructure exist; no or minimal real behavior |
| EXPERIMENTAL | Works but lacks production validation |
| UNVERIFIED | Implementation exists but verification is insufficient |
| BLOCKED | Requires an explicit technical/product decision or external dependency |
| MISSING | No meaningful implementation |
| BROKEN | Implementation exists but current verification fails |

# Global Rule

Never remove an item because a partial interface was created. Move a
status only when the listed evidence exists. Every transition requires:
implementation + integration + tests + diagnostics + documentation.

---

# P0 — MAPPING SPINE (current campaign)

## P0.1 — Real capture dataset

**Status: VERIFIED** — `datasets/room_capture/` (28 real photos, manifest,
1.0 m measured baseline). Honest boundary documented: the synthetic
render fixture (`tests/fixtures/`) is out-of-distribution for COCO and
yields zero detections — asserted, not hidden.

## P0.2 — Reproducible reconstruction (COLMAP)

**Status: IMPLEMENTED** — Real COLMAP backend, version recorded in
provenance, explicit configuration, availability separate from
acceptance. Remaining: parameter pinning to a documented config file
(currently ad-hoc in `colmap_backend.py`).

## P0.3 — Model registry / offline checkpoints

**Status: IMPLEMENTED** — `perception/model_registry.py`: name, task,
source, path, SHA256, verification gate; acquisition recorded after real
loads. Remaining: registry applied to MiDaS/SAM paths (currently only the
Mask R-CNN path).

## P0.4 — Real metric depth

**Status: IMPLEMENTED** — MiDaS + SfM sparse-fit scale anchoring, labeled
`metric-by-alignment` everywhere (never presented as sensor-grade
metric). Remaining: registry-pinned checkpoints.

## P0.5 — Depth → 3D points

**Status: VERIFIED** — unprojection with trusted intrinsics, explicit
coordinate frames, invalid-depth handling, provenance. Honest skip when
no intrinsics are trusted (asserted in `tests/test_cli_compile.py`).

## P0.6 — Real segmentation / P0.7 — Real detection

**Status: VERIFIED** — `perception/detection/maskrcnn_backend.py`: real
COCO Mask R-CNN (torchvision), one forward pass → detections + instance
masks at source resolution. Verified on a real photograph (`book` 0.78,
`person` 0.51). SAM backend exists but is BROKEN in this environment
(torch-hub cache layout); tracked below.

## P0.8 — Detection + segmentation + depth convergence

**Status: VERIFIED** — pipeline stage 3.5: detect → segment → lift →
multi-view merge → promote → re-validate. Entity ↔ observation
traceability preserved through evidence ids.

## P0.9 — Object 3D reconstruction

**Status: IMPLEMENTED** — lifting, centroid/bounds/dimensions,
OBSERVED-vs-INFERRED distinction. KNOWN APPROXIMATION: object geometry is
AABB-based, provenance `INFERRED`. Remaining: oriented bounds / mask-
constrained geometry.

## P0.10 — Multi-view object resolution

**Status: IMPLEMENTED** — deterministic geometric + label matching.
Remaining (P2): appearance embeddings, epipolar verification.

## P0.11 — Dense multi-view geometry

**Status: DONE for COLMAP dense MVS (ledger P6-01, P6-02; `tests/test_dense_*`, `tests/test_evidence_fusion.py`; real CUDA COLMAP dense on South Building per PROD-01; reconciled 2026-10-17), PARTIAL for blank-wall coverage.** Dense geometry comes from metricized mono-depth
(per-view MiDaS + SfM alignment), true COLMAP MVS where the binary
supports it, and (2026-10-31) a CPU plane-sweep backend:
`reconstruction/backend/plane_sweep.py`, registered in
`reconstruction/backend/dense_registry.py`, opt-in through
`VerticalSliceOptions.plane_sweep_enabled`. The sweep derives its depth
range from the sparse cloud, keeps a pixel only when one candidate depth
beats the runner-up by a margin in every neighbour, and drops a blank
wall instead of filling it; points enter `result.points` as
`plane_sweep:` tracks and go through normal fusion. Tests:
`tests/test_plane_sweep.py` (known-depth recovery, textureless refusal,
pipeline stage). Remaining: measured wall-coverage improvement on a real
room fixture (no real capture in the repo), CUDA/learned MVS options.

## Room reconstruction fidelity (Phase 7, 2026-10-31)

Plan: `docs/superpowers/plans/2026-10-30-room-reconstruction-fidelity.md`.
Baseline and scorecard: `docs/reconstruction/ROOM_FIDELITY_BASELINE.md`,
`scripts/room_fidelity_report.py`, CI gate `--fail-on-regression`.

| # | Item | Status | Evidence / remaining |
|---|---|---|---|
| F1 | Capture guidance | IMPLEMENTED | `reconstruction/capture_quality.py`, `evidence/image_check.py`, coverage/hints in `engine/pipeline/guidance.py`; UI `CaptureCoverage.tsx`; `tests/test_capture_quality.py`. Not yet validated on a re-capture by a user. |
| F2 | Phone LiDAR / depth ingestion | IMPLEMENTED (synthetic) | 16-bit sidecars `evidence/depth_frames.py`; metric scale from device depth `reconstruction/depth_scale.py`, wired in `vertical_slice` stage 2; unprojection in the sidecar depth stage. `tests/test_depth_scale.py`, `tests/test_synthetic_rgbd.py`. PHYSICAL DEVICE VERIFICATION PENDING. |
| F3 | Dense geometry for blank walls | PARTIAL | CPU plane sweep above; textureless surfaces refuse by design. Real-fixture wall coverage not measured. |
| F4 | Learned room-layout evidence | PARTIAL — BLOCKED on model choice | Sidecar contract, validation, floor-plane projection and lower-confidence fusion are implemented (`perception/architecture/layout_evidence.py`, `room_layout.solve_room_layout(layout_edges_ab=...)`); `tests/test_layout_evidence.py`. No HorizonNet/LayoutNet weights are shipped; `estimator_status()` reports UNAVAILABLE. Decision required: which offline layout model to install (plan section 6, item 3). |
| F5 | Lighting-independent colour | IMPLEMENTED | Per-view white-balance/exposure normalisation in `perception/materials/surface_appearance.py`; `tests/test_surface_appearance.py`. Limited to overlapping views; no colour-card verification. |
| F6 | User corrections as evidence | IMPLEMENTED | API `routes_worlds.py` corrections + `apps/api/models.py`; UI in the inspector; `tests/test_corrections.py`. |
| F7 | Multi-room / whole-home | IMPLEMENTED | `perception/architecture/room_join.py` joins shared doorways; building promotion defect fixed; `tests/test_room_join.py`, `tests/test_room_fidelity_fixture.py`. |
| F8 | Floor plan / room report | IMPLEMENTED | `exporters/floorplan.py` (SVG + DXF) and room report with wall areas net of openings; API route; `tests/test_floorplan_export.py`. Room report also renders as a hand-built PDF (`report_to_pdf`, wired through `sdk/reality.py` and `apps/api/routes_export.py`; `tests/test_floorplan_export.py`). No real-room plan has been checked against a tape measure. |
| F9 | Accuracy scorecard in CI | IMPLEMENTED | `scripts/room_fidelity_report.py --fail-on-regression`, baseline JSON, `.github/workflows/ci.yml`. |
| F10 | Progressive preview + timings | PARTIAL | Per-stage `elapsed_s` in progressive attempts and scorecard timings; progressive levels already return the strongest valid world. No performance target is set until real-capture baselines exist. |

### Room fidelity plan items S/D/O/M/V/U (registered 2026-10-31)

Plan: `docs/superpowers/plans/2026-10-30-room-reconstruction-fidelity.md`. HONEST BOUNDARY for every row: verified on
SYNTHETIC fixtures only. No real-room photos and no tape measurements exist in the repo, so NOTHING below is VERIFIED
and nothing is browser-verified. Statuses are held at PARTIAL/EXPERIMENTAL accordingly.

| # | Item | Status | Evidence / remaining |
|---|---|---|---|
| P0 | Phase 0 baseline + CI gate | PARTIAL (synthetic only) | `scripts/room_fidelity_report.py`, `docs/reconstruction/ROOM_FIDELITY_BASELINE.md` + `room_fidelity_baseline.json`, `synthetic/fidelity_room` fixtures, CI job `fidelity-gate` in `.github/workflows/ci.yml`. The gate guards synthetic regressions only. |
| S1-S4 | Structure: solid wall slabs with carved openings, Manhattan snapping, inferred closing walls, layout-consistency over the union of facing-wall spans | PARTIAL | `perception/architecture/wall_slabs.py`; `tests/test_wall_entities.py`, `tests/test_room_fidelity_fixture.py`. Wall thickness is always the labelled 0.10 m default (not measured). Real-room accuracy unknown. |
| D1 | Dimension shape `{value, unit, state, basis, uncertainty_m}` (MEASURED / RECONSTRUCTED / ESTIMATED / VERIFIED) | PARTIAL | `world_ir/measure.py`; `tests/test_dimension_state.py`. |
| D2/D2b | Known-length scale + scale SUGGESTION from a fitted door (user must accept; Decision 023 amendment) | PARTIAL | `reconstruction/scale.py`, `GET/POST /api/worlds/{id}/scale*`; `tests/test_scale_suggestion.py`. Accepting a suggestion has NEVER been exercised end to end (UI to API to rebuild to ESTIMATED dimensions). `scale.py` drops `dense_points` on rebuild. |
| D3/D4 | Dimension consistency + dimension gates | EXPERIMENTAL | `perception/architecture/dimension_consistency.py`; `tests/test_dimension_gates.py` checks 2 % on L/W/H and 6 cm on openings against SYNTHETIC ground truth only. The real-room variant is opt-in and SKIPPED (needs photos + tape measurements from the user). |
| O1/O2 | Openings / furniture from a learned 2D door/window detector | BLOCKED - DECISION REQUIRED | DECISION REQUIRED: which 2D door/window detector may be installed. OPTIONS: (a) install an open-vocabulary detector such as a YOLO/GroundingDINO-class model; (b) keep the geometry-only opening host already present; (c) defer. TRADEOFFS: accuracy vs install weight and licence. RECOMMENDATION: ask the user; keep geometry-only (b) meanwhile. DEPENDENCIES: model registry (`perception/model_registry.py`), offline checkpoint policy. |
| M1/M2 | Surface colour (CIELAB robust median), hash-addressed texture atlas, unobserved texels transparent | PARTIAL | `perception/materials/room_surfaces.py`; `tests/test_room_surfaces.py`. `dense_ingest` persists the cloud without colour. |
| M3 | Material hypotheses | EXPERIMENTAL | `perception/materials/material_hypotheses.py`: state ESTIMATED, `asserted=False`, confidence <= 0.4; compiler runs it via `CompileOptions.appearance_views`; `tests/test_material_hypotheses.py`. NOT wired to any physics-material mapping (by design, an unverified label must not drive physics). |
| V1/V2 | Viewer: room shell + atlas texture, evidence summary in the inspector | EXPERIMENTAL (not browser-verified) | `frontend/src/lib/viewport/room-shell-model.ts`, `three-scene.ts` (atlas on `MeshLambertMaterial`), `inspector-evidence-model.ts`, `AdaptiveInspector` "Evidence summary", `apps/api/routes_artifacts.py` (texture by hash; `tests/test_api_world_artifacts.py`). Frontend 171/171 contract tests and `tsc --noEmit` exit 0; NO browser run. UVs on thin wall sides are stretched; the artifact route parses the full head WorldIR per request (uncached); the photo list shows view ids, not thumbnails. |
| U1-U3 | UX: capture guidance and coverage, guided flow | PARTIAL | `frontend/src/lib/reconstruct/flow-model.ts`; guided-flow steps 4 and 5 always show "needs action"; capture-coverage routes have never run against a stored world. |

Known limits recorded with this change (all OPEN): wall thickness is the labelled 0.10 m default; the camera view cone
(30 deg half-angle, 8 m) used for coverage is an assumed model, not a calibrated one; `scale.py` drops `dense_points` on
rebuild; `dense_ingest` persists the cloud without colour; capture-coverage routes never run against a stored world;
accepting a scale suggestion never exercised end to end; guided-flow steps 4/5 always show "needs action"; the artifact
route is uncached; UVs on thin wall sides are stretched; the photo list shows view ids rather than thumbnails.

Dense CPU plane sweep exposure (F3): `apps/api/routes_reconstructions.py` accepts form fields `dense_cpu`,
`plane_sweep_max_views`, `plane_sweep_stride`, `plane_sweep_planes` and `apps/api/jobs.py` forwards them
(`tests/test_api_dense_cpu.py`, 8 tests). OFF by default. No real end-to-end job has been run with `dense_cpu=true`, so F3
stays PARTIAL.

Test status for this campaign (observed, not a full-suite claim): 44 tests in the freebuff-owned files; frontend 171/171
contract tests with `tsc --noEmit` exit 0; `tests/test_api_dense_cpu.py` + `tests/test_api_world_artifacts.py` 19 tests;
a backend agent reported 203 passed, 1 skipped, 0 failed over its own suites. A CLEAN FULL python suite result is NOT
yet available (a run was still in progress when this was written); do not read this section as a full-suite pass.

### Site composer and landmark reconstruction (registered 2026-10-31)

| ID | Item | Status | Detail |
|---|---|---|---|
| SITE-1 | Site composer: user-directed placement of several reconstructed worlds on a metric grid | FOUNDATION (being built; not verified) | Tables `Site` / `SitePlacement`, routes `/api/sites`, frontend `/sites`. This is MANUAL placement only: the user decides where each world goes. It is NOT automatic cross-source or geo-registration, and it does not move [`P1.10 Cross-source registration`](../future/registration/CROSS_SOURCE_REGISTRATION.md), which stays MISSING. Do not advance past FOUNDATION until routes, UI and tests exist and have been run. |
| LAND-1 | Landmark / curved-structure reconstruction from a single image (domes, arches, towers, facade ornament, e.g. Taj Mahal, Burj Khalifa) | PARTIAL | FOUNDATION: `perception/architecture/landmark.py` fits single-image 3D point clusters to curved landmark kinds (dome via sphere fit, arch via circle cross-section, tower via vertical cylinder fit) with measured residuals (rms/max), honest FitRefused when evidence is insufficient, deterministic output, relative-vs-measured scale handling, and a `LandmarkErrorReport` for the acceptance-criterion measurement. WorldIR DOME/ARCH entity types and SPHERE/CYLINDER/CAPSULE/IMPLICIT/MESH geometry types already exist; the ArchitectureRegistry registers arch/dome (SPHERE), arch/arch (CIRCLE), tower (CYLINDER). Tests: `tests/test_landmark.py` (27 passed). ACCEPTANCE CRITERION (stated): measured against a ground-truth model with a stated error metric and threshold; not by eye. DEPENDENCIES: learned mono/multi-view model or multi-view capture; real landmark photo set + surveyed dimensions. REAL-DATA VERIFICATION BLOCKED — no real landmark fixture exists in this environment; all verification is synthetic. |

## P0.12 — Real mesh representation / P0.13 — Mesh → WorldIR

**Status: VERIFIED** — `reconstruction/meshing/`: `MeshData` (canonical),
deterministic PLY I/O, voxel downsample + statistical outlier filter +
camera-oriented PCA normals, camera-envelope filter (sparse cloud bounds
the plausible scene), real COLMAP `poisson_mesher` backend (CPU;
depth-density adaptation + trim retry). Mesh persists through
ArtifactStore with SHA256; WorldIR `Geometry(type=MESH, data_uri,
data_hash)`; glTF exports real triangles (uint32 indices).
Observed E2E: 240k-vert mesh at room scale (15.8 m extent).

## P0.14 — World compiler convergence

**Status: IMPLEMENTED** — cameras, scale, sparse+dense geometry, depth,
perception, planes, rooms, mesh all converge into WorldIR v1 via the
compiler. Remaining: material hypotheses.

## P0.15 — Spatial relationships

**Status: PARTIAL** — inside/contains/adjacent/near/above-below from
geometry. Remaining: supports/connected-to/part-of with provenance;
relationships currently 0 on the room dataset (capture coverage).

## P0.16 — Geometric measurement

**Status: IMPLEMENTED** — object dimensions, distances, room extents;
measured/reconstructed/inferred distinction. Remaining: E2E assertion of
a measurement against the known 0.44 m baseline (ground-truth gate).

## P0.17 — WorldIR validation

**Status: VERIFIED** — geometry/artifact existence, coordinate frames,
transforms, units, entity/relationship references, provenance,
confidence, bounds. Compilation fails loudly.

## P0.18 — Real end-to-end acceptance test

**Status: PARTIAL (automated spine; mesh leg unverified on committed data)** —
`tests/integration/test_acceptance_p018.py` (slow, integration; skips with a
reason when COLMAP/MiDaS or `datasets/south_building` are absent) drives the
real `reality compile` CLI on a 10-view window of the committed real-photo
South Building set. Evidence 2026-10-17: `pytest tests/integration/test_acceptance_p018.py`
-> 2 passed in 137 s (real COLMAP 9/10 cameras, 29,963 points; real MiDaS
depth 10 maps / 9 metricized). Asserted: CLI exit 0, SfM backend + >=6/10
cameras + >=1000 points, entities compiled, `reality validate` passes,
points.ply / cameras.json / glTF written, scale declared RELATIVE with
mesh+depth honestly refused when unanchored, and (anchored) METRIC scale +
depth ran + dense points. Caveats: South Building has no tape measurement, so
the metric run uses a test-declared nominal 1.0 m baseline (not real-world
accuracy); the **mesh stage is not covered** — CPU Poisson meshing had not
finished after 40+ minutes on this data (hang/perf finding, needs
investigation); original 25/28-camera room-capture run remains manual.
Next: diagnose the mesh-stage slowness, add a real measured baseline dataset,
then promote to VERIFIED.

## P0.19 — Real viewer

**Status: IMPLEMENTED** — browser Studio viewer: real point cloud, real
mesh, cameras, entities, selection, inspector, measurements, provenance
panel, framing, visibility controls. Consumes WorldIR + artifacts only.
Remaining (P6): evidence/observation browsing, diff, timeline.

## P0.20 — One-command product path

**Status: IMPLEMENTED** (PR #19 open) — `reality compile <dataset>`:
full pipeline, machine-readable report with per-stage status, no green
success on failure, deterministic offline tests via
`REALITY_TEST_BACKEND` seam.

---

# P1 — SENSOR + TEMPORAL FOUNDATION

## P1.4 — Depth sidecar ingestion (RGB-D)

**Status: BLOCKED (hardware) — software path IMPLEMENTED (ledger P1-03; `tests/test_synthetic_rgbd.py`); no physical RGB-D device.** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/sensor-ingestion/DEPTH_INGESTION.md`](../future/sensor-ingestion/DEPTH_INGESTION.md)
Depth sidecar discovery exists; normalized ingestion does not.
**Decision 001 (ACCEPTED): 16-bit PNG first** — see
[`../engineering/DESIGN_DECISIONS.md`](../engineering/DESIGN_DECISIONS.md).
Critical rule: `depth_m = raw * scale` — never assume raw == meters.

Evidence to advance: canonical `DepthFrame`, PNG decoder with explicit
scale/invalid handling, calibration association, corrupt-file rejection,
real RGB-D capture validated.

## P1.5 — IMU / GNSS / telemetry ingestion

**Status: PARTIAL** — real CSV ingestion + canonical Source/Evidence
identity landed via PR #17/#18. Remaining: unit contracts (NED/ENU),
per-sample provenance, sync hooks.

## P1.6 — Timestamp synchronization

**Status: DONE (ledger P2-01; `tests/test_clock.py`, `tests/test_clocks.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/synchronization/TIME_SYNCHRONIZATION.md`](../future/synchronization/TIME_SYNCHRONIZATION.md)
Model: `t_global = a·t_sensor + b`, every sample retains original +
normalized timestamp, clock id, offset/drift, uncertainty, method.
Evidence to advance: `ClockModel`/`TimeAlignment` with diagnostics,
offset + drift tests, cross-stream test.

## P1.8 — VIO / trajectory

**Status: BLOCKED (no built backend) — adapters/selection/diagnostics IMPLEMENTED (ledger P3-01 DONE, P3-02 BLOCKED, P3-03 DONE; `tests/test_synthetic_vio.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/vio/VIO_TRAJECTORY.md`](../future/vio/VIO_TRAJECTORY.md)
Backend abstraction (`TrajectoryBackend`) wrapping a mature system
(ORB-SLAM3 / OpenVINS / Basalt). Never vendor from scratch. Evidence:
trajectory continuity + reprojection residual + drift diagnostics.

## P1.10 — Cross-source registration

**Status: DONE (ledger P4-01; `tests/test_registration.py`, `tests/test_landmark_registration.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/registration/CROSS_SOURCE_REGISTRATION.md`](../future/registration/CROSS_SOURCE_REGISTRATION.md)
Known-extrinsics → GNSS prior → coarse alignment → point-to-plane ICP →
robust optimization → covariance → accept/reject. Never mark successful
without measurable residual + valid correspondence set.

## P1.11 — Dense MVS + depth fusion

**Status: DONE (ledger P6-01/P6-02; CUDA blocker resolved 2026-09-15).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry.
→ spec in [`../future/dense-reconstruction/`](../future/dense-reconstruction/DENSE_MVS.md).
Fusion: inverse-variance weighting, robust estimators for outlier-heavy
data; diagnostics: confidence, consistency, density, holes.

# P1.12/13 — Canonical MeshData

**Status: IMPLEMENTED** — see P0.12/13. Remaining: LOD, quality metrics,
dense-MVS integration.

---

# P2 — PERCEPTION + IDENTITY

## P2.14 — Multi-view object identity

**Status: DONE (ledger P7-01; `tests/test_multiview_identity_appearance.py`, `tests/test_track_backend.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/perception/MULTI_VIEW_IDENTITY.md`](../future/perception/MULTI_VIEW_IDENTITY.md)
Current merging is label + geometric proximity — an honest heuristic,
never presented as a learned embedding. Evidence to advance: embedding
backend behind an interface, epipolar verification, identity benchmark.

### ML reconstruction, overlap intelligence and export plan (2026-11-01)

Source plan: `2026-11-01-ml-reconstruction-overlap-plan.md`. Status per step, with the evidence that exists today:

| Item | Status | Evidence / what is missing |
|---|---|---|
| Step 0: CUDA torch + `perception/device.py` | IMPLEMENTED | torch 2.14.0+cu130 on RTX 4050 (6 GB), fp16 matmul ran on GPU; `tests/test_device.py` passes on GPU and with `REALITY_DEVICE=cpu`; SAM and MiDaS backends use it. Other backends (Mask R-CNN) not yet wired. Dataset access requests (ScanNet++, ARKitScenes) NOT filed. |
| Step 1: export fix | IMPLEMENTED | `GET /export/formats` + `downloadArtifact()` + icon grid; failures show status and server reason. Browser-verified (glTF download, 404 PLY error banner) in an older UI worktree; NOT yet browser-verified in this UI. 14/14 export route tests. Whether the 15 s default timeout was the user's real failure is not confirmed. |
| Phase 1: observation ledger + association engine | FOUNDATION | `world_ir/observations.py`, `perception/instances/association.py`, `tests/test_association.py` (scenarios 2, 4, 5, 16, chain-vs-union-find, AMBIGUOUS, relative-depth widening, occlusion vs intersection). Synthetic only. NOT wired into the pipeline (`object_resolution.merge_hypotheses` is still what runs), no epipolar `geom_verified`, no negative-visibility evidence, no CONFLICTED surface fusion, ledger not persisted into WorldStore, sigma constants untuned. |
| Phase 2: IMatcher (ALIKED+LightGlue), metric depth, benchmarks | MISSING | `reconstruction/{features,matching}` are still README stubs. |
| Phase 3: cross-session identity via ledger | MISSING | Needs Phase 1 wired in. |
| Phase 4: coverage map, capture requests | MISSING | |
| Phase 5/6/7: completion, appearance, whole-journey validation | MISSING | |

Open diagnosis (2026-11-01): the user's 10-photo room registers 9/10 cameras with COLMAP yet compiles to 2 walls and no floor; dense MVS runs and is rejected ("no material gain"). Root cause not found. COLMAP also refuses workspace paths over 170 characters (reproduced with a 262-character store path), which silently drops the pipeline to unregistered single-view fusion.

## P2.15 — Tracking / temporal identity

**Status: DONE (ledger P7-02; 2D box association / camera-motion compensation items below are not ledger scope).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. The 3D observation-level identity-lifecycle layer
is implemented and tested: `perception/tracking/temporal.py` —
time-ordered `TimedObservation` chains per label into `TrackRecord`s
(track_id, observation history, measured path/duration/implied speed),
continuation gated by BOTH a speed budget and a gap budget, untimed
observations returned separately, confidence = min of measured member
confidences (a fabricated 1.0 was caught in review and removed).
Adapter: `instance_tracks_from` -> `InstanceTrack` (refuses tracks
without measured confidence or with missing regions). Tests:
`tests/test_temporal_tracking.py` (14, red-first). Still missing:
2D detection-box association (ByteTrack/OC-SORT-role), camera-motion
compensation, ID-switch/fragmentation diagnostics. Canonical ledger:
`.agent/TASKS.yaml` P7-02.

## P2.16 — Structural perception (doors/windows/stairs)

**Status: BLOCKED on real indoor data (ledger P7-03; synthetic-verified, `tests/test_synthetic_indoor.py`; open limits PT-06 below).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Originally: doors/windows existed only as
room-boundary-derived openings. Evidence: geometry + segmentation backed
structural hypotheses with provenance.

## P2.20 — Scene-complete reconstruction (detect and rebuild every visible object)

**Status: MISSING (pieces exist, not wired) → [`../future/perception/SCENE_COMPLETE_RECONSTRUCTION.md`](../future/perception/SCENE_COMPLETE_RECONSTRUCTION.md)**
Added 2026-11-06. Images that fall to the single-image path (incl. all AI-generated
rooms) get no object detection at all (`engine/pipeline/single_image.py`), and
`furniture_from_detections` is not wired. M1 is wiring the existing detector + placer
+ Depth Anything V2 into that path. Evidence to advance: labelled eval set +
per-tier recall/precision and floor-contact error as specified.

## P2.17 — Material perception

**Status: MISSING → [`../future/perception/MATERIAL_PERCEPTION.md`](../future/perception/MATERIAL_PERCEPTION.md)**
Evidence-backed hypotheses only; never hallucinate labels without
visual evidence.

---

# P3 — UNCERTAINTY + PROVENANCE

## P3.18 — Uncertainty propagation

**Status: DONE (ledger P10-02; `tests/test_uncertainty_propagation.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/uncertainty/UNCERTAINTY_PROPAGATION.md`](../future/uncertainty/UNCERTAINTY_PROPAGATION.md)
Schema fields exist (per-point/per-pose Uncertainty); propagation does
not. First-order: `Σ_y = J Σ_x Jᵀ`. Critical rule: confidence ≠
uncertainty — both represented independently. Evidence: covariance
propagation through lift → merge → measurement with tests.

## P3.19 — Provenance graph

**Status: DONE (ledger P10-01; `tests/test_provenance_graph.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/provenance/PROVENANCE_GRAPH.md`](../future/provenance/PROVENANCE_GRAPH.md)
Every entity carries evidence ids and provenance enum, but the chain
(entity → observation → artifact → frame → source) is not a queryable
graph. Evidence: lineage query API + round-trip test from entity back to
source record.

---

# P4 — PERSISTENT WORLD

## P4.20 — WorldStore

**Status: DONE (ledger P11-01; `worldstore/store.py`, `tests/test_world_store*.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/worldstore/WORLDSTORE.md`](../future/worldstore/WORLDSTORE.md)
SQLite first; PostgreSQL later. Binary data lives in ArtifactStore,
never in relational rows. Requires: world versions, entity history,
immutable evidence, transactions.

Caveat (2026-11-07): what exists is the file-backed store (`worldstore/store.py`, `worldstore/tiles.py`). The SQLite/PostgreSQL
relational store from the spec and entity-history queries are NOT implemented.

## P4.21 — Incremental compilation

**Status: DONE (ledger P12-01; `tests/test_incremental_world_e2e.py`, `tests/test_true_localized_incremental_acceptance.py`); remaining incremental gaps are tracked as CE-02 below.** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/worldstore/INCREMENTAL_COMPILATION.md`](../future/worldstore/INCREMENTAL_COMPILATION.md)
Dependency graph + invalidations + regional recompilation + versioning.

---

# P6 — REALITY STUDIO

**Status: IMPLEMENTED** (no standalone ledger task; PROD-01 reconciliation 2026-10-02: tsc 0, next build 0, contract tests; `tests/test_reality_studio_primary_journey.py`). Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Still missing: uncertainty visualization, simulation controls. (Original: browser viewer only, missing source/
evidence browsers, uncertainty + provenance visualization, timeline,
diff/branch views, simulation controls.) Architectural rule: Studio is a
client of WorldIR/WorldStore; it never becomes the reconstruction engine.

**Plan (2026-10-13; Phases 0, 1 and 2 done, see its section 11):** [`../future/studio/STUDIO_CONSUMER_UX_PLAN.md`](../future/studio/STUDIO_CONSUMER_UX_PLAN.md) -- consumer UX redesign (UX-01..12) and missing backend (B-00..12), phased with acceptance criteria.

---

# P7 — DOMAINS + INFRASTRUCTURE

## P7.25 — GIS

**Status: MISSING** → [`../future/gis/GIS.md`](../future/gis/GIS.md). CRS/WGS84/ECEF/ENU, GeoTIFF/DEM, georeferencing. pyproj etc. optional extras, never core deps.

## P7.26 — Robotics

**Status: MISSING** → [`../future/robotics/ROBOTICS.md`](../future/robotics/ROBOTICS.md). ROS2/TF/occupancy via adapters only.

## P7.27 — Large world

**Status: DONE (ledger P13-01, P13-02; `tests/test_spatial_tiling.py`, `tests/test_worldstore_tiles.py`).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/large-world/LARGE_WORLD.md`](../future/large-world/LARGE_WORLD.md). Spatial cells, octree/LOD, artifact paging, streaming indexes.

## P7.28 — Packaging

**Status: IMPLEMENTED (`tests/test_packaging_smoke.py`; clean-venv run found and fixed missing deps, commit dc35a79).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Originally: CLI entry point existed; wheel contents + clean-venv install not verified. Evidence: build + install + import + CLI in a clean environment.

## P7.29 — CI

**Status: PARTIAL (`.github/workflows/ci.yml` present; not executed in this review).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Originally: no GitHub workflows existed. Evidence: unit CI, deterministic CLI E2E CI, marked real-model tests separated (weights/network never required for green), packaging smoke test.

Caveat (2026-11-07): downgraded to PARTIAL by the 2026-11-07 audit: `main` Core Tests were red (4 stale test doubles, fixed on the M0 branch); a green run and a clean-venv wheel install in CI are still unverified.

---

# P8 — SIMULATION (downstream; kept unwired)

## P8.30 — Physics → WorldIR / reverse updates

**Status: DONE as generic world simulation (ledger P17-01; `tests/test_simulation_loop.py`); physics engines moved to `reality-engine-child` 2026-09-18.** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Original text: rigid bodies, collision, contact solver, destruction/fire/wind foundations existed. Missing: CCD, joints, physics→WorldIR state write-back with provenance (simulated state must stay distinguishable from observed).

## P8.31 — Advanced simulation

**Status: MISSING (deliberately deferred).** Fluids (SPH/FLIP), thermal
coupling, structural FEM, high-fidelity fire. Do not begin before the
mapping spine and P4 persistence are complete.

---

# PROD — Evidence-progressive product path (2026-09-29)

Design and evidence: [`../PROGRESSIVE_RECONSTRUCTION.md`](../PROGRESSIVE_RECONSTRUCTION.md).

## PROD.1 — One-action reconstruction (photos -> world -> refine)

**Status: PARTIAL (verified on one real dataset; updated 2026-10-02).** Implemented and executed
with real COLMAP/MiDaS on the South Building photographs and in a real
browser. Since 2026-09-29: judged automatic dense Level 3, user-measured metric scale, single-image openings,
stable entity identity, frame-preserving candidate arbitration and the adoption protocol (see DESIGN_DECISIONS) are
implemented and tested. Remaining: a real corridor and a real room photograph dataset (none exist: UNVERIFIED) and the PROD.3
confidence-defaults audit. The per-version evidence panel landed 2026-10-03.

## PROD.2 — Stable entity identity across versions

**Status: IMPLEMENTED (heuristic).** `world_ir/entity_reid.py` is wired into the rebuild
(`apps/api/jobs.py::_carry_entity_identity`, tests/test_entity_reid.py); identity decisions are recorded on the version.

## PROD.3 — Confidence defaults

**Status: IMPLEMENTED.** `Uncertainty` default is 0.0 (unknown); `Observation`, `Relationship`, `CausalRelation`
constructor AND `from_dict` defaults are 0.5 / `Provenance.UNKNOWN` ("not measured"), never 1.0 / OBSERVED; building/storey/
opening producers pass measured values; the commit route rejects invalid confidence instead of coercing it.
`tests/test_confidence_defaults.py` pins the defaults and statically checks that every producer in the repo states its
confidence explicitly. Raw-sensor decode fidelity (depth frames) legitimately states 1.0 with its basis.
(The previous "left as documented risk" note was stale: the audit on 2026-10-03 found the defaults already corrected.)


# P9 — WORLD FACTORY (additive to the roadmap, registered 2026-10-18)

Plan: [`../superpowers/plans/2026-10-18-world-factory-and-studio-master-plan.md`](../superpowers/plans/2026-10-18-world-factory-and-studio-master-plan.md).
Product vision: [`../product/VISION.md`](../product/VISION.md). Creation outputs are always `StatementState.PROCEDURAL` /
`Provenance.GENERATED`; they never masquerade as observed geometry.

| ID | Item | Status | Detail |
|---|---|---|---|
| P9.1 | Room builder | PARTIAL | `procedural/room_grammar.py`, `POST /api/worlds/{id}/rooms`, Studio **Build** dialog (`BuildModal.tsx`). Missing: openings UI, parameter edit + scoped regeneration, templates. |
| P9.2 | Corridor builder | PARTIAL (new 2026-10-18) | `procedural/urban_grammar.generate_corridor`, `POST .../corridors`, Build dialog; `tests/test_procedural_urban_grammar.py`, `tests/test_application_api.py`; browser-verified. Straight, axis-aligned only; no door cut into joined rooms. |
| P9.3 | City / district builder | PARTIAL (C5, 2026-10-18; API only, no UI) | `generate_city` unchanged (grid of lots, max 100). NEW `procedural/district_grammar.py` + kind `district`: grid or spine layout with named STREET entities, one PLOT per parcel, seeded building families (default 3, drawn from the seed and STORED in params), per-parcel family from `Random(seed:i:j)` (independent of other parcels), per-parcel overrides, detail policy `full` (real `generate_building`) vs `footprint_only` (one box, summed storey height) by distance/count cap (default max_full=25). `POST/PUT/DELETE /worlds/{id}/districts` (409 duplicate/dependents, `?cascade=true`, PUT reports changed/unchanged). Tests: `tests/test_district_grammar.py`, `test_application_api.py::test_district_*` (determinism, one-parcel edit leaves others byte-identical, in-place footprint->full upgrade, 409, cascade). MEASURED (docs/BENCHMARKS.md): 500 buildings = 1,048 entities in 32 ms footprint_only; 3,623 entities in 65 ms at default cap; 63,317 entities / 67 MB JSON in 753 ms all-full; API POST 119 ms / 467 ms. NOT done: parks, arterials/non-grid road graph, `shell`/`accessible_interior`/`hero` detail tiers, separate `road_network.py`, viewport LOD/instancing and browser frame time (so NO 'district scale' rendering claim), Build-dialog UI, parcels override edit replaces the whole map, `regenerate` re-runs the whole district (5.0 s for all-full 500). Viewport still renders streets rotated (axis convention, plan task B1). |
| P9.4 | Apartment layout (spatial program -> adjacency graph -> plan) | PARTIAL (2026-10-18; API only, no UI) | `procedural/layout_solver.py` (deterministic guillotine/BSP search + spine seeds, hard area/aspect/adjacency constraints, up to 3 alternatives, honest `ok=False` with named violations; hallway default aspect_max 4.0), `procedural/apartment_grammar.py` (rooms, shared walls once, doors at satisfied adjacencies + entrance, corridor link only when a room has no door path), `POST /apartments/solve` and `/apartments`. Verified live: a 6-room brief (hall touching 4 rooms) solves with 3 plans in ~0.3 s solver time and commits 6 rooms / 6 doors / 19 walls; infeasible brief returns "rooms need at least 60 m2 but footprint 5x5 offers 25 m2". Limits: rectangular footprint and guillotine layouts only; search is bounded (ok=False is not a proof of infeasibility); doors are boxes in the wall plane, walls not carved; single storey; no PUT route or UI yet. |
| P9.5 | Building factory (floors, stairs, facade rules) | PARTIAL | C4 grammar + /buildings routes live-verified: 3-storey tower = 3 STOREY, 2 STAIRS, 1 ROOF, 32 WINDOW, 16 DOOR (128 entities); raising top storey 3.0->4.5 m changed 19 entities, 0 of 81 lower-storey entities changed. Remaining: Build-dialog Building tab, Generated-panel edit for buildings/apartments. |
| P9.6 | Scoped regeneration with dependency tracking | PARTIAL (2026-10-18) | `procedural/spec_store.py` (specs stamped on entities, `regenerate`, attach dependents), `GET /generated`, `PUT`/`DELETE /{rooms,corridors,cities}/{name}`, duplicate POST -> 409. Verified live: widening room A 4->7 m moved its attached corridor (x 4-10 -> 7-13) and left room B byte-identical; delete with dependents -> 409, cascade removes both. 167 backend tests pass. Missing: UI to edit (plan C2), city PUT/DELETE dedicated tests, legacy pre-spec worlds cannot be edited (409 by design). |
| P9.7 | Completion mode (reconstructed shell + labelled generated proposals) | PARTIAL (Phase D backend slice, 2026-10-18; API only, no UI) | `apps/api/routes_completion.py` + table `completion_proposals` (create_all). `POST /worlds/{id}/completion/proposals` (1-3 seeded alternatives from a template or user room program, solved by `layout_solver` inside the axis-aligned bbox of non-generated RECONSTRUCTED/OBSERVED floor/wall/room entities), `GET` list, `GET .../{pid}/diff` (added/changed/removed + provenance counts), `POST .../accept` (via `_generate_and_commit`; entities stay GENERATED, stamped `completion_proposal`; 409 if shell changed or name taken), `POST .../reject`. 422 when no usable measured shell. Units `model_units` unless the world has a metric/authored measurement. Tests: `tests/test_completion_mode.py`. NOT done: UI, non-rectangular/rotated shells (bbox approximation), multi-storey, aligning to reconstructed openings, furniture, re-proposal on stale, any real-data validation (Phase E). Plan: `docs/superpowers/plans/2026-10-18-phase-d-completion.md`. |
| P9.8 | Game-ready export profile (collision, LOD, validation) | PARTIAL | C6 Godot profile `exporters/profiles.py` -> `exporters/godot/profile.py` (GLB with per-entity nodes, `-col` collision hints, manifest.json with kinds/provenance/skips/units, scene.tscn); round-trip tested in `tests/test_godot_profile.py`. NOT done: LOD proxy, City/District/Building/Room hierarchy, opening in Godot (external, UNVERIFIED). SDK `reality.export_profile` and API `POST /api/worlds/{id}/export {"format":"godot"}` are registered (one JSON document, each file `{encoding, content}`, GLB base64) and tested in `tests/test_application_api.py`. |
| P9.9 | Assistant tool layer (Phase F first slice) | PARTIAL (2026-10-18; backend only, NO LLM wiring, no UI) | `assistant/tools.py` (pure, deterministic read tools: list_entities, get_entity, find_by_name, measure via `world_ir/measure.py`, explain_provenance, search) and `apps/api/routes_assistant.py` + table `assistant_plans` (create_all). `GET /api/assistant/tools` (JSON-schema style), `POST /worlds/{id}/assistant/call`, `POST /worlds/{id}/assistant/confirm`. `propose_edit` accepts only `update_generated` (PUT semantics) and `delete_generated` (dependents check, optional cascade); rename/anything else -> 422; it stores a plan with preview counts and never commits. Confirm runs through `_generate_and_commit`; plan bound to the HEAD it previewed against, 409 + stale if HEAD moved, expired (30 min) or replayed. explain_provenance states `Not recorded`/`Unknown` for absent evidence/confidence (the 0.5 entity default is not presented as a confidence). Tests: `tests/test_assistant_tools.py` (17 pass). NOT done: any model/LLM wiring or tool-call loop, semantic (embedding) search, capture guidance, self-correction loop, UI (Studio AI tab still disabled), propose for non-generated (reconstructed) entities, evidence-record lookup beyond ids already on the entity; confirm re-runs the edit on whatever HEAD exists at commit time (the stale check precedes it, a HEAD move in that gap is handled by the existing retry, not re-flagged). |

# P6b — STUDIO PARITY WITH THE REFERENCE DESIGNS (registered 2026-10-18)

Reference images: `C:\Users\shres\Downloads\reference` (~125 screens). Decision (2026-10-18, owner): match them **as exactly
as possible, new design tokens allowed**. Status: **PARTIAL (2026-10-18)**. Gap audit recorded in
`docs/screenshots/parity/B0-gap-audit.md`. Done and browser-verified: navy tokens, unified shell (top mode bar, icon rail,
status footer), Home dashboard, World Intent wizard with an honest resource profile (`POST /api/intent/estimate`,
`world_intents` table), palette sweep, auto-frame of cloud-less worlds, Evidence Quality Firewall + asset inspector (measured
checks only; a Rejected decision is enforced in reconstruction input), measurement mode (distance/height/area/elevation,
server-recomputed values, unit honesty: model units when scale is unknown, "exact" only for authored geometry; volume / road
width / footprint disabled). Remaining slices: map/plot view, pipeline graph view, mobile capture, Playwright smoke tests over
all routes, remaining neutral-grey sweep (plan Phase B). Known gaps vs references: no analysis overlay (no stored feature
locations), no metric uncertainty sigma for reconstructed worlds, similar-frames strip, firewall steps header and statistics
tabs.

## Continuous evidence — gaps against the "same world keeps evolving" principle (2026-09-29)

| ID | Item | Status | Detail |
|---|---|---|---|
| CE-01 | Explicit evidence conflicts in WorldIR | PARTIAL: camera-pose and planar-geometry conflicts IMPLEMENTED | Both hypotheses are kept with provenance and history in `WorldIR.metadata["conflicts"]`, carried into each new frame, resolved only by new supporting evidence (unit-tested lifecycle). NOT implemented: topology conflicts (room/storey structure), semantic conflicts (door exists vs not), opening/box-geometry conflicts, and any conflict between *photographs* as such (the pipeline has no per-observation contradiction test). Conflicts live in report + WorldIR metadata, not as `Relationship` objects. |
| CE-02 | Incremental integration into WorldIR(Vn) | PARTIAL: COLMAP registration is incremental (`image_registrator`, IMPLEMENTED, see PROGRESSIVE_RECONSTRUCTION.md); everything after COLMAP (frame alignment, depth, perception, planes, compile) is still recomputed for the whole world, and there is no region-aware local refinement | Each version is rebuilt from the UNION of all evidence (`compile(A+B)` = V2), not by registering only the new evidence into the previous WorldIR. Consequence: nothing is lost and late bridges work (A -> C -> B), but entity ids are re-derived and the cost grows with total evidence. Alternative: incremental COLMAP registration + `world_ir/entity_reid.py`. Not chosen yet; needs measurement of rebuild cost at 100+ photos. |
| CE-03 | Non-photo evidence in the same world | PARTIAL | Floor plans / renders / historical photos are classified and kept as context, but they do not yet constrain geometry (no layout prior, no era separation in the rebuild). |
| CE-04 | Hidden geometry UNKNOWN -> OBSERVED transitions are not asserted | PARTIAL | The single-view path records an UNKNOWN entity; multi-view versions do not carry it forward, so a diff cannot show UNKNOWN turning into OBSERVED. |
| CE-05 | Candidate-vs-HEAD acceptance | IMPLEMENTED, VERIFIED on real photos | `world_delta.py`; V1 kept through a crashed rebuild and a worse (single-view fallback) rebuild, waiting photos placed on the next healthy rebuild (`test_failed_and_worse_candidates_never_replace_HEAD_...`). Entity-level regression is now judged from geometry (see CE-06). |
| CE-06 | Spatial continuity between versions | IMPLEMENTED (heuristic), VERIFIED on real photos | `spatial_continuity.py`: footprint overlap of fitted planes; preserved / refined / extended / reduced / split / merge / regrouped / ambiguous / removed / new; regions with the new photographs that touched them. Heuristic: thresholds (20 deg, 10% of extent, 60% coverage) are not calibrated against ground truth; only planar structure has shape data (openings/boxes fall back to centre distance); segmentation instability makes labels vary run to run; regions are spatial clusters, not semantic parts (no "front facade" naming). |
| CE-07 | Persistent COLMAP state (`image_registrator`) | IMPLEMENTED, measured | See PROGRESSIVE_RECONSTRUCTION.md. Preserves COLMAP's frame and saves ~18-20% runtime on 6-12 photo scenes; **no quality gain** over a fresh rebuild. Larger worlds not measured. |
| CE-08 | Incremental-vs-full COLMAP candidate arbitration by WORLD quality | IMPLEMENTED; VERIFIED (unit + real COLMAP + product path) | `engine/pipeline/candidate_selection.py`: both candidates become world snapshots and go through `world_delta.compute_delta/reconcile_conflicts/decide` against HEAD; an ordered hierarchy (validity, acceptance, established photos lost, established surfaces lost, unsupported moves/conflicts, camera stability (margin 0.02), uncertainty, photos placed, refined/extended/new surfaces, tie -> incremental) decides and records WHY. No scalar score. A full rebuild is also tried when the incremental candidate is not a clean ACCEPT, not only when photos are unplaced. Limits: candidates are compared on the sparse-compiled world (no depth/perception/mesh), the margins are tolerances not calibrated constants, every non-ACCEPT incremental costs one extra mapper run. Tests: `tests/test_candidate_selection.py` (cases A-F), `tests/integration/test_colmap_failure_injection.py`, live decision text in `tests/integration/test_reliability_journey.py`. |
| CE-09 | Failure injection / restart / corruption / concurrency | PARTIALLY VERIFIED | Real COLMAP: crash before/after each of feature_extractor, matcher, image_registrator, point_triangulator, bundle_adjuster, mapper; crash before commit, after commit; 5 corruption cases (missing model/db, bad manifest, truncated model file, stale staging, removed evidence): `current/` byte-identical after every failure. Product path: crash in WorldIR compile / world_delta / decide / validation / conflict reconcile / WorldStore commit; fault after commit and read-back failure (HEAD valid, versions coherent, world converges); real process-tree kill mid-job (state A or B, never mixed); racing uploads. Kill exactly between WorldStore commit and COLMAP commit is covered by the stage-synchronised kill tests (8 boundaries). Fast fault-injection tests (no real COLMAP, `tests/test_failure_modes_colmap.py`, `tests/test_failure_modes_worldstore.py`) now cover: subprocess timeout at every step (typed `TIMED OUT` error, incremental chain falls back to full), reconstruction-parse failure (typed `model_parse` error; incremental falls back), disk-full (ENOSPC on image copy, manifest write, artifact write; retry converges), fault/kill inside `ColmapSession.commit`, corrupted WorldStore artifact (HEAD / ancestor / shared blob), and two parallel jobs from one HEAD (one adopts, loser quarantined and rebases onto the winner, duplicate retry deduplicated). Defects fixed by these tests: torn artifact blob poisoned retries (`FileArtifactStore.put` now atomic + re-verifies existing blob); `ColmapSession.commit` could lose `current/` between its two renames (now self-heals); malformed COLMAP output raised raw ValueError (now typed `model_parse`). Still NOT covered: reconstruction-parse/timeout through the real API job path with real COLMAP. |
| CE-10 | Rollback of HEAD | IMPLEMENTED (pointer model); VERIFIED on real photos | `POST /api/worlds/{id}/rollback` -> `worldstore_service.rollback_head`: only the HEAD pointer moves, versions stay immutable, evidence is kept and shows as outside the current model, the COLMAP base is renamed `superseded-by-rollback-*` (set aside BEFORE the pointer moves, so a crash cannot leave HEAD seeded by the wrong state), the next version chains onto the rolled-back HEAD. Consequence: after a rollback the next run is a full rebuild (no per-version COLMAP snapshots). |
| CE-11 | Cross-store recovery journal | NOT IMPLEMENTED (decision) -- WorldStore<->DB boundary now has an adoption-intent protocol, see DESIGN_DECISIONS | The COLMAP session is DERIVED state, validated on every run against the evidence set and pipeline signature, and it commits strictly after WorldStore adoption + read-back. The only reachable cross-store disagreement is "COLMAP one version behind", which converges on the next run (`test_crash_after_worldstore_adoption_...`); "COLMAP ahead of WorldStore" is unreachable by ordering. A fault after adoption is now a recorded degradation, not a FAILED job (the retry used to adopt a duplicate version). Revisit if the COLMAP state ever becomes authoritative. |


## Prototype-completion sprint (2026-10-03) -- scope, verification levels, what is pinned and what is external

The prototype is judged by the core loop (evidence -> world -> more evidence -> same world -> version -> export). See
DESIGN_DECISIONS "The working prototype is judged by the core loop". Verification levels per capability:
`reality verification [--runtime]`, `GET /api/system/verification`.

| ID | Item | Status | Detail |
|---|---|---|---|
| PT-01 | Core exports glTF, USD(A), IFC, CityGML, CityJSON, Blender | DONE (REAL-WORLD VERIFIED: a world reconstructed from real photographs, all six parsed back with independent readers) | `tests/test_core_exports.py`, `tests/test_export_route.py`. IFC is now served by the SDK/API/CLI (was a file-only bridge). External: opening the files in Blender / a USD viewer / a BIM tool. |
| PT-02 | Ecosystem exports GIS, ROS, Habitat, SUMO, Unreal, Godot (P16-02) | OPTIONAL_DEFERRED | Writers exist and are structure-tested (`tests/test_box_exporters.py`); not required for the prototype; target-runtime acceptance is external. |
| PT-03 | Synthetic RGB-D (P1-03) | SYNTHETIC RGB-D VERIFIED -- PHYSICAL DEVICE VERIFICATION PENDING | `synthetic/rgbd.py`, `tests/test_synthetic_rgbd.py`. |
| PT-04 | Synthetic VIO (P3-02) | SYNTHETIC VIO PIPELINE VERIFIED -- REAL BACKEND / HARDWARE VERIFICATION PENDING | `synthetic/vio.py`, `tests/test_synthetic_vio.py`; fixed `FrameGraph.load_from_trajectory`; new `trajectories/worldir.py`. |
| PT-05 | Synthetic indoor architecture (P7-03) | SYNTHETIC INDOOR VERIFIED -- REAL INDOOR DATASET VERIFICATION PENDING | `synthetic/indoor.py`, `tests/test_synthetic_indoor.py`. |
| PT-06 | Known limits found by PT-05 (strict xfail, not hidden) | OPEN ENGINEERING | rooms joined by doorways over one floor merge into one room; stairs not wired into interior assembly; `detect_stairs` refuses an unsegmented flight or one with unobserved treads; phantom openings under realistic depth noise (sigma ~1.6 cm at 5 m); mesh validation (`reconstruction/meshing/validation.py`) is too slow on ~50k-point clouds (a 16-view synthetic capture spent minutes there). |
| PT-07 | 3DCityDB availability AVAILABLE / UNAVAILABLE / FAILED (P14-01) | DONE (probe); real database verification EXTERNAL | `exporters/citydb.probe_citydb`. |
| PT-08 | Studio shows the ten change categories and how a version was built | DONE | `ReconstructionStatusBar` (`change-counts`, `how-built`); `frontend/src/lib/api/reconstruction-decision.ts`; contract tests. |
| PT-09 | Canonical demo journey | DONE | `scripts/demo_journey.py`, `tests/integration/test_demo_journey.py`; measured metrics recorded, nothing hard-coded. |

## P0.12 — Multi-view scene identity and unified reconstruction — EXPERIMENTAL
Verified-correspondence scene grouping, registration diagnostics with explicit outcomes, bounded measured self-correction
and a persistent scene registry are implemented and integrated into the reconstruct job (`reconstruction/scene_identity.py`,
`apps/api/jobs.py`). Verified on synthetic shared-scene imagery with real COLMAP. Open: real-photo threshold calibration,
retrieval index beyond ~65 images, per-stage retry, separate-world creation for extra groups (product decision).
Spec/evidence: `docs/implementation/SCENE_IDENTITY.md`.
