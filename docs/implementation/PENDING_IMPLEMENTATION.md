# Reality Engine — Pending Implementation (Master Backlog)

> Capabilities that are architecturally intended but not yet fully
> implemented and verified. This is an engineering backlog, not a claim
> that these features exist. Every entry names the evidence required to
> advance its status; a status may only move when that evidence exists.

Last reconciled: 2026-10-17 for the P0-P8 status lines (against `.agent/TASKS.yaml`; the P0.x entries and the PROD/CE/PT tables were not re-audited). Earlier reconciliation: 2026-09-14.

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

**Status: DONE (ledger P6-01, P6-02; `tests/test_dense_*`, `tests/test_evidence_fusion.py`; real CUDA COLMAP dense on South Building per PROD-01).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry.

(Original, superseded:) dense geometry came from metricized
mono-depth (per-view MiDaS + SfM alignment), NOT true multi-view stereo.
COLMAP's `patch_match_stereo` is BLOCKED: the available binary is
CPU-only (4.2.0, no CUDA). Remaining: dense MVS backend (CUDA COLMAP,
OpenMVS, or hybrid), cross-view depth consistency checks.

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

## P4.21 — Incremental compilation

**Status: DONE (ledger P12-01; `tests/test_incremental_world_e2e.py`, `tests/test_true_localized_incremental_acceptance.py`); remaining incremental gaps are tracked as CE-02 below.** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Spec: [`../future/worldstore/INCREMENTAL_COMPILATION.md`](../future/worldstore/INCREMENTAL_COMPILATION.md)
Dependency graph + invalidations + regional recompilation + versioning.

---

# P6 — REALITY STUDIO

**Status: IMPLEMENTED** (no standalone ledger task; PROD-01 reconciliation 2026-10-02: tsc 0, next build 0, contract tests; `tests/test_reality_studio_primary_journey.py`). Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Still missing: uncertainty visualization, simulation controls. (Original: browser viewer only, missing source/
evidence browsers, uncertainty + provenance visualization, timeline,
diff/branch views, simulation controls.) Architectural rule: Studio is a
client of WorldIR/WorldStore; it never becomes the reconstruction engine.

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

**Status: IMPLEMENTED (`.github/workflows/ci.yml` present; not executed in this review).** Reconciled 2026-10-17 against `.agent/TASKS.yaml`; the older text below this line is the original backlog entry. Originally: no GitHub workflows existed. Evidence: unit CI, deterministic CLI E2E CI, marked real-model tests separated (weights/network never required for green), packaging smoke test.

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
