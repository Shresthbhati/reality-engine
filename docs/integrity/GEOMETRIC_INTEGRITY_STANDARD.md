# Geometric Integrity Standard — Reality Engine V1

> Authoritative geometric validity rules for every transformation in the reconstructed world: camera intrinsics/extrinsics, world↔camera transforms, coordinate handedness, axis conventions, metric/relative scale, rotation conventions, quaternion normalization, translation units, point-cloud consistency, surface normals, mesh winding/topology, degenerate geometry, duplicate vertices, plane estimation, room boundaries, openings, floor/ceiling, object placement/dimensions, occlusion/visibility.
>
> Part of the KILO integrity mandate (§3: "Geometric integrity programme"). Owner: World Reconstruction Integrity.
> Status: established from inspected code (`world_ir/coordinates.py`, `world_ir/geo.py`, `reconstruction/fusion/`, `reconstruction/meshing/`, `perception/geometry/`, `world_ir/schema_v1.py`).

## 0. Governing rule

> Introduce validation at every relevant boundary. Never silently repair invalid geometry without recording the operation and its provenance.

Every geometric value that crosses a subsystem boundary is validated at that boundary. Invalid geometry is reported as a validation issue (never silently healed, never defaulted to a plausible value).

## 1. Coordinate frames & axis conventions

**Canonical convention:** right-handed, **Y-up** world frame.

| Item | Contract |
|---|---|
| Canonical world frame | `Frame.WORLD`, right-handed, +Y up |
| Package working frame (legacy SfM/COLMAP) | +Z up |
| Frame mapping | `R_UP` maps job frame (+Y up) ↔ package working frame (Z-up); applied by residual-checked similarity, never silently |
| Camera poses | camera **CENTERS** with camera-to-world rotations (wxyz quaternion) |
| SfM→job alignment | residual-checked similarity transform (Umeyama); < `MIN_COMMON_CAMERAS` (4) shared cameras = not measurable |
| Every transform record | states its `source_frame` and `target_frame` (`world_ir/coordinates.Transform`); never a bare matrix with implicit meaning |
| `Frame` enum values | `camera`, `sensor`, `session-local`, `building-local`, `world`, `ENU`, `UTM`, `WGS84`, `ECEF`, `engine-local`, `storey-local` |

**Validation:** A transform is valid only if its matrix is finite, homogeneous (bottom row `[0,0,0,1]`), and its frame pair is declared. `world_ir/coordinates._mat_inverse_rigid` handles rigid transforms only (rotation + translation, no scale/shear) — this limitation is documented; non-rigid transforms must not be passed to it.

## 2. Camera intrinsics & extrinsics

**Schema location:** `world_ir/camera.py` `PinholeCamera`, `CameraIntrinsics`, `CameraExtrinsics`.

| Field | Type | Units | Validation |
|---|---|---|---|
| `fx`, `fy` | px | px | > 0, finite |
| `cx`, `cy` | px | px | within image bounds |
| `width`, `height` | px | px | > 0, integers |
| `position` | `(x,y,z)` | m | world, finite |
| `rotation` | `(w,x,y,z)` | — | world, unit quaternion |
| `pose_source` | `str` | — | `colmap\|vio\|registered\|unknown` |

**Contract rules (from `vertical_slice.run_pose` / `_intrinsics_options`):**
- Every camera must agree on image size within 1%; pinhole-like (non-fisheye).
- Intrinsics trust = `"estimated"` (never `"trusted"` unless operator-declared). COLMAP is not pinned; distortion is ignored.
- Intrinsics source recorded: `operator_declared_camera` \| `exif_focal_length_35mm` \| `colmap_estimated_previous_sparse_model` \| `colmap_estimated_this_run`.
- Missing/unset intrinsics → depth/perception/meshing stages **skip with explicit reason** ("refusing to guess one"); never invented.
- Operator-declared camera (`POST /api/worlds/{id}/camera`) is the only path to `"trusted"` trust.

## 3. Scale states & metric vs relative

**Schema location:** `reconstruction/scale.py` `ScaleState` {METRIC, RELATIVE}; `world_ir/schema_v1.py` `Measurement`.

| Scale state | Units | Source | Permitted |
|---|---|---|---|
| `METRIC` | m | operator-measured baseline between two registered cameras (or known_length / prior accepted by user with consent) | Physical dimension claims, mesh, fusion, metric export |
| `RELATIVE` | unit-less | SfM / monocular depth alignment | Relative structure only; **never** claimed as meters |

**Contract rules:**
- Metric scale comes **ONLY** from an explicit operator-measured baseline (`VerticalSliceOptions.measured_baselines`), tried in order; first whose camera pair is both registered wins.
- No baseline registered → world stays RELATIVE with unit-less distances. **No code path invents a scale factor.**
- `meters_per_unit = None` for RELATIVE; never `1.0` by assumption.
- `reconstruction/scale.py` bug (fixed 2026-09-14): `result` was never reassigned after `anchor_metric_scale(...)`, so the world claimed meters while geometry stayed in model units. Fixed — `result` reassigned to `scaled`.
- Device depth (16-bit PNG RGB-D sidecars) is coherent only when the world is METRIC; unaligned relative depth stays `RELATIVE_DEPTH`.

**Validation:** RELATIVE worlds never claim meters; metric worlds carry `meters_per_unit`; scale state recorded in `world.metadata["scale"]`.

## 4. Rotation conventions & quaternion normalization

| Item | Contract |
|---|---|
| Quaternion order | `(w, x, y, z)` |
| Normalization | quaternions must be unit-norm; validation checks \|q\| ≈ 1 |
| `make_rigid_transform` default uncertainty | `Uncertainty()` = confidence **0.0** (unknown), **NOT 1.0** — a transform whose producer did not state confidence must not read as certain |
| Transform composition (`then`) | confidence multiplied — a chained transform is never more certain than either hop |
| `_mat_inverse_rigid` | rigid only (R^-1 = R^T); documented limitation |

**Validation:** Non-unit quaternions = validation issue; never silently re-normalized without recording the operation and provenance.

## 5. Translation units

| Item | Contract |
|---|---|
| Translation | metres (m) in metric worlds; unit-less in RELATIVE worlds |
| `Measurement.unit` | SI units only (`meter`, `kg`, `second`, ...) |
| `Measurement.precision` | standard deviation / margin of error (legacy) |
| `Measurement.uncertain` | `UncertainScalar` (value + sigma + basis) — formal replacement, P10-02 |

**Contract rule:** SI units only; no implicit unit conversion. A `Measurement` with `unit != "meter"` for a length = validation issue.

## 6. Point-cloud consistency

| Item | Contract |
|---|---|
| Points | finite (x,y,z); no NaN/Inf |
| Track IDs | prefixes distinguish sources: `sfm`, `depth-`, `dense_mvs:`, `plane_sweep:` |
| Trimming | points trimmed to camera envelope (`trim_to_scene`) |
| Invalid pixels | NaN holes / invalid-pixel masks carried (`DepthMap.valid_mask`) — never interpolated silently |
| Depth source truth | `depth_source` states the truth (METRIC / MVS / RELATIVE); `sigma_for(depth_source, range_m)` = max(SIGMA_FLOOR_M 0.05m, fraction*range) {METRIC 0.02, MVS 0.03, RELATIVE 0.25} |

**Validation:** Non-finite point = validation issue. `world_pos=None` means "not lifted" — never a made-up position.

## 7. Surface normals

| Item | Contract |
|---|---|
| Normal | unit vector; `float(on @ e.normal)` sign-flipped so normals face consistently |
| Normal difference gate | `MAX_NORMAL_DIFF_DEG = 30.0`; `NORMAL_SIGMA_DEG = 15.0` |
| `world_ir/schema_v1.py` `Surface.normal_direction` | `Optional[Vector3]` |

**Validation:** Non-unit normal = validation issue; sign inconsistency across a surface = reported, not silently flipped without provenance.

## 8. Mesh winding, topology & degenerate geometry

| Item | Contract |
|---|---|
| Mesh validity | reported via `s12_quality_report` (trimesh: watertight, degenerate faces, components) → NOT_EVALUABLE / PASS / WARN / FAIL |
| Degenerate triangles | reported as WARN/FAIL; never silently removed |
| Duplicate vertices | reported; never silently deduplicated without provenance |
| Disconnected geometry | components reported; never silently merged |
| Bounding box | `bounds_min`/`bounds_max` must not be inverted; non-finite = ERROR |
| Bounding box semantics | an approximation labeled INFERRED — never presented as mesh geometry |

**Validation:** A mesh is FAIL only when it fails watertight/degenerate/component checks; WARN when marginal. Never upgraded silently to PASS.

## 9. Plane estimation

**Schema location:** `perception/geometry/planes.py`; `engine/pipeline/spatial_continuity.py` `fit_plane`.

| Item | Contract |
|---|---|
| Plane fit | PCA through points; requires ≥ `MIN_PLANE_POINTS` (8) |
| Planarity test | RMS out-of-plane ≤ `PLANE_RMS_TO_WIDTH` (0.10) × short in-plane extent |
| Planar types | `wall`, `floor`, `ceiling` |
| Points sampling | deterministic strided sample ≤ `MAX_POINTS` (160) |

**Validation:** Fewer than 8 points → no plane fitted (falls back to centre-only); never fabricates a plane from insufficient points.

## 10. Room boundaries, floor & ceiling, openings

| Item | Contract |
|---|---|
| Room | enclosed interior space (`EntityType.ROOM`); boundary elements referenced; single storey membership |
| Floor / ceiling | horizontal walking surface / upper interior boundary (`EntityType.FLOOR` / `EntityType.CEILING`) |
| Opening | wall void (`EntityType.OPENING`); referenced to a surface; opening→surface, surface→space invariant |
| Wall / floor / ceiling | planar types; opening→surface, surface→space invariant |

**Validation (Blueprint invariants 1-7):**
1. `evidence_state` vocabulary valid.
2. OBSERVED/PARTIALLY_OBSERVED entities cite at least one `observation_id`.
3. INFERRED is not also OBSERVED.
4. Metric units only in non-relative worlds.
5. Entity id uniqueness.
6. Duplicate guard.
7. `opening→surface`, `surface→space` references valid.

**Disconnected rooms:** `topology_disconnected_rooms` = WARNING (partial capture may honestly lack connecting evidence) — never ERROR.

## 11. Object placement & dimensions

| Item | Contract |
|---|---|
| Object position | finite (x,y,z); lifted against metricized depth maps (same maps as perception — one owner of depth state) |
| Object dimensions | finite; `bounds_min`/`bounds_max` not inverted |
| One object = one entity | however many photos show it |
| A detection is an observation | never an entity (BUILD_SPEC rule 2) |

**Validation:** Object outside camera envelope = validation issue. Dimensions with inverted bounds = ERROR.

## 12. Occlusion & visibility

| Item | Contract |
|---|---|
| `Visibility` enum | `FULL`, `PARTIAL`, `OCCLUDED`, `NOT_OBSERVED` |
| `visibility=None` | not assessed (never defaulted) |
| Occlusion relationship | `RelationshipKind.OCCLUDES` — line-of-sight occlusion between entities |

**Validation:** `visibility` assessed honestly; an occluded object is not presented as fully visible.

## 13. Depth & fusion

| Item | Contract |
|---|---|
| Depth stage | disabled unless `depth_model` set + trusted intrinsics + METRIC scale |
| Depth points | appended IN PLACE to `result.points` (track_id `depth-`) |
| Fusion | plural-source; ≥ 2 sources with ≥ 10 points each; conflicts recorded, never winner-picked; refuses RELATIVE ("fusing unit-less ranges would silently claim meters") |
| Dense MVS | OFF by default; explicit operator decision or evidence-gated (`dense_gate.dense_readiness`) |

**Validation:** Fusion conflicts recorded with provenance; never silently resolved to a single value.

## 14. Validation boundaries (where to check)

| Boundary | What is validated |
|---|---|
| Camera → depth | trusted intrinsics + METRIC scale required; else skip with reason |
| Depth → perception | metricized depth maps survived; scene gate allows indoor |
| Points → mesh | METRIC world + artifact store + ≥ points; else skip |
| Fusion → WorldIR | `validate_world_ir` gate before returning; refused, never silently corrupted |
| WorldIR → persist | read-back verify (`verify_version` clean) before HEAD pointer moves |

**Never:** silently repair invalid geometry without recording the operation and its provenance. Every repair is a validation issue with provenance.
