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

**Status: MISSING.** Evidence: track entities across ordered frames with
temporal association diagnostics.

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
