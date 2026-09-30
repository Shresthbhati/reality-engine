# Reality Engine — Pending Implementation (Master Backlog)

> Capabilities that are architecturally intended but not yet fully
> implemented and verified. This is an engineering backlog, not a claim
> that these features exist. Every entry names the evidence required to
> advance its status; a status may only move when that evidence exists.

Last reconciled: 2026-09-14 (mapping campaign; PRs #11, #12, #14, #16 merged; #17/#18 by concurrent agents; #19 open).

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

**Status: PARTIAL** — dense geometry currently comes from metricized
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

**Status: PARTIAL** — full chain runs and is verified on the room
dataset (25/28 cameras, metric, 121k points, 103 entities, validated
WorldIR, mesh, glTF). Remaining: the acceptance test is manual/detached,
not a committed automated test (COLMAP + models + ~30 min make it
opt-in). Needs a marked real-data acceptance test.

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

**Status: PARTIAL → spec in [`../future/sensor-ingestion/DEPTH_INGESTION.md`](../future/sensor-ingestion/DEPTH_INGESTION.md)**
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

**Status: MISSING → spec in [`../future/synchronization/TIME_SYNCHRONIZATION.md`](../future/synchronization/TIME_SYNCHRONIZATION.md)**
Model: `t_global = a·t_sensor + b`, every sample retains original +
normalized timestamp, clock id, offset/drift, uncertainty, method.
Evidence to advance: `ClockModel`/`TimeAlignment` with diagnostics,
offset + drift tests, cross-stream test.

## P1.8 — VIO / trajectory

**Status: MISSING → spec in [`../future/vio/VIO_TRAJECTORY.md`](../future/vio/VIO_TRAJECTORY.md)**
Backend abstraction (`TrajectoryBackend`) wrapping a mature system
(ORB-SLAM3 / OpenVINS / Basalt). Never vendor from scratch. Evidence:
trajectory continuity + reprojection residual + drift diagnostics.

## P1.10 — Cross-source registration

**Status: MISSING → spec in [`../future/registration/CROSS_SOURCE_REGISTRATION.md`](../future/registration/CROSS_SOURCE_REGISTRATION.md)**
Known-extrinsics → GNSS prior → coarse alignment → point-to-plane ICP →
robust optimization → covariance → accept/reject. Never mark successful
without measurable residual + valid correspondence set.

## P1.11 — Dense MVS + depth fusion

**Status: PARTIAL** (mono-depth path exists; true MVS blocked on CUDA)
→ spec in [`../future/dense-reconstruction/`](../future/dense-reconstruction/DENSE_MVS.md).
Fusion: inverse-variance weighting, robust estimators for outlier-heavy
data; diagnostics: confidence, consistency, density, holes.

# P1.12/13 — Canonical MeshData

**Status: IMPLEMENTED** — see P0.12/13. Remaining: LOD, quality metrics,
dense-MVS integration.

---

# P2 — PERCEPTION + IDENTITY

## P2.14 — Multi-view object identity

**Status: PARTIAL → spec in [`../future/perception/MULTI_VIEW_IDENTITY.md`](../future/perception/MULTI_VIEW_IDENTITY.md)**
Current merging is label + geometric proximity — an honest heuristic,
never presented as a learned embedding. Evidence to advance: embedding
backend behind an interface, epipolar verification, identity benchmark.

## P2.15 — Tracking / temporal identity

**Status: PARTIAL.** The 3D observation-level identity-lifecycle layer
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

**Status: MISSING.** Doors/windows currently exist only as
room-boundary-derived openings. Evidence: geometry + segmentation backed
structural hypotheses with provenance.

## P2.17 — Material perception

**Status: MISSING → [`../future/perception/MATERIAL_PERCEPTION.md`](../future/perception/MATERIAL_PERCEPTION.md)**
Evidence-backed hypotheses only; never hallucinate labels without
visual evidence.

---

# P3 — UNCERTAINTY + PROVENANCE

## P3.18 — Uncertainty propagation

**Status: PARTIAL → spec in [`../future/uncertainty/UNCERTAINTY_PROPAGATION.md`](../future/uncertainty/UNCERTAINTY_PROPAGATION.md)**
Schema fields exist (per-point/per-pose Uncertainty); propagation does
not. First-order: `Σ_y = J Σ_x Jᵀ`. Critical rule: confidence ≠
uncertainty — both represented independently. Evidence: covariance
propagation through lift → merge → measurement with tests.

## P3.19 — Provenance graph

**Status: PARTIAL → spec in [`../future/provenance/PROVENANCE_GRAPH.md`](../future/provenance/PROVENANCE_GRAPH.md)**
Every entity carries evidence ids and provenance enum, but the chain
(entity → observation → artifact → frame → source) is not a queryable
graph. Evidence: lineage query API + round-trip test from entity back to
source record.

---

# P4 — PERSISTENT WORLD

## P4.20 — WorldStore

**Status: MISSING → spec in [`../future/worldstore/WORLDSTORE.md`](../future/worldstore/WORLDSTORE.md)**
SQLite first; PostgreSQL later. Binary data lives in ArtifactStore,
never in relational rows. Requires: world versions, entity history,
immutable evidence, transactions.

## P4.21 — Incremental compilation

**Status: MISSING → [`../future/worldstore/INCREMENTAL_COMPILATION.md`](../future/worldstore/INCREMENTAL_COMPILATION.md)**
Dependency graph + invalidations + regional recompilation + versioning.

---

# P6 — REALITY STUDIO

**Status: FOUNDATION** — browser viewer (see P0.19). Missing: source/
evidence browsers, uncertainty + provenance visualization, timeline,
diff/branch views, simulation controls. Architectural rule: Studio is a
client of WorldIR/WorldStore; it never becomes the reconstruction engine.

---

# P7 — DOMAINS + INFRASTRUCTURE

## P7.25 — GIS

**Status: MISSING** → [`../future/gis/GIS.md`](../future/gis/GIS.md). CRS/WGS84/ECEF/ENU, GeoTIFF/DEM, georeferencing. pyproj etc. optional extras, never core deps.

## P7.26 — Robotics

**Status: MISSING** → [`../future/robotics/ROBOTICS.md`](../future/robotics/ROBOTICS.md). ROS2/TF/occupancy via adapters only.

## P7.27 — Large world

**Status: MISSING** → [`../future/large-world/LARGE_WORLD.md`](../future/large-world/LARGE_WORLD.md). Spatial cells, octree/LOD, artifact paging, streaming indexes.

## P7.28 — Packaging

**Status: PARTIAL.** CLI entry point exists; wheel contents + clean-venv install not verified. Evidence: build + install + import + CLI in a clean environment.

## P7.29 — CI

**Status: MISSING.** No GitHub workflows exist. Evidence: unit CI, deterministic CLI E2E CI, marked real-model tests separated (weights/network never required for green), packaging smoke test.

---

# P8 — SIMULATION (downstream; kept unwired)

## P8.30 — Physics → WorldIR / reverse updates

**Status: PARTIAL.** Rigid bodies, collision, contact solver, destruction/fire/wind foundations exist. Missing: CCD, joints, physics→WorldIR state write-back with provenance (simulated state must stay distinguishable from observed).

## P8.31 — Advanced simulation

**Status: MISSING (deliberately deferred).** Fluids (SPH/FLIP), thermal
coupling, structural FEM, high-fidelity fire. Do not begin before the
mapping spine and P4 persistence are complete.

---

# PROD — Evidence-progressive product path (2026-09-29)

Design and evidence: [`../PROGRESSIVE_RECONSTRUCTION.md`](../PROGRESSIVE_RECONSTRUCTION.md).

## PROD.1 — One-action reconstruction (photos -> world -> refine)

**Status: PARTIAL (verified on one real dataset).** Implemented and executed
with real COLMAP/MiDaS on the South Building photographs and in a real
browser. Missing: a real corridor dataset and a real room dataset
(UNVERIFIED), dense Level 3, metric scale, single-image opening detection,
per-version evidence panel while inspecting an old version.

## PROD.2 — Stable entity identity across versions

**Status: MISSING.** V(n) is rebuilt from the union of evidence, so plane ids
are re-derived; a diff between a single-view hypothesis and a multi-view model
correctly reports remove+add, but a refinement of the same structure can also
churn ids. Wire `world_ir/entity_reid.py` into the rebuild.

## PROD.3 — Confidence defaults

**Status: PARTIAL.** `Uncertainty` default is now 0.0 (unknown); building/storey/
opening producers no longer default to 1.0; the commit route rejects invalid
confidence instead of coercing it. Left as documented risk: `Observation` and
`Relationship` constructor defaults (1.0 / OBSERVED) and `CausalRelation`
deserialisation default; raw-sensor decode fidelity (depth frames) legitimately
states 1.0 with its basis.


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
| CE-09 | Failure injection / restart / corruption / concurrency | PARTIALLY VERIFIED | Real COLMAP: crash before/after each of feature_extractor, matcher, image_registrator, point_triangulator, bundle_adjuster, mapper; crash before commit, after commit; 5 corruption cases (missing model/db, bad manifest, truncated model file, stale staging, removed evidence): `current/` byte-identical after every failure. Product path: crash in WorldIR compile / world_delta / decide / validation / conflict reconcile / WorldStore commit; fault after commit and read-back failure (HEAD valid, versions coherent, world converges); real process-tree kill mid-job (state A or B, never mixed); racing uploads. NOT covered: reconstruction-parse failure, subprocess timeout, disk-full, kill exactly between WorldStore commit and COLMAP commit (only faulted in-process), stale-candidate rebase with two parallel jobs from one HEAD, corrupted WorldStore artifact. |
| CE-10 | Rollback of HEAD | IMPLEMENTED (pointer model); VERIFIED on real photos | `POST /api/worlds/{id}/rollback` -> `worldstore_service.rollback_head`: only the HEAD pointer moves, versions stay immutable, evidence is kept and shows as outside the current model, the COLMAP base is renamed `superseded-by-rollback-*` (set aside BEFORE the pointer moves, so a crash cannot leave HEAD seeded by the wrong state), the next version chains onto the rolled-back HEAD. Consequence: after a rollback the next run is a full rebuild (no per-version COLMAP snapshots). |
| CE-11 | Cross-store recovery journal | NOT IMPLEMENTED (decision) | The COLMAP session is DERIVED state, validated on every run against the evidence set and pipeline signature, and it commits strictly after WorldStore adoption + read-back. The only reachable cross-store disagreement is "COLMAP one version behind", which converges on the next run (`test_crash_after_worldstore_adoption_...`); "COLMAP ahead of WorldStore" is unreachable by ordering. A job whose post-commit step fails reports FAILED although the new HEAD is valid (honest but noisy) - a journal would let recovery mark such a job complete. Revisit if the COLMAP state ever becomes authoritative. |
