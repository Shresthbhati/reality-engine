# WorldIR Canonical Contract — Reality Engine V1

> Authoritative, consistent contract for the canonical WorldIR world representation.
> Resolves competing representations into ONE contract per concept instead of adding another parallel contract.
>
> Part of the KILO integrity mandate (§2: "Own the canonical WorldIR contract"). Owner: World Reconstruction Integrity.
> Coordinates with: Gemini (perception observation contracts), Cline (SAM2 mask references).
> Status: established from inspected code (`world_ir/schema_v1.py`, `world_ir/world_v1.py`, `world_ir/coordinates.py`,
> `provenance/provenance.py`, `uncertainty/__init__.py`, `worldstore/store.py`).

## 0. Governing rule

> Resolve competing representations instead of adding another parallel contract.

Before this contract, the codebase had parallel representations for the same idea:

| Concept | Competing representations (pre-contract) | Canonical (this contract) |
|---|---|---|
| Entity confidence | `schema_v1.Entity.confidence: Optional[float] = None` (None = not measured) vs `world_ir/observations.Observation.confidence: float = 0.5` (fabricated default) | `Optional[float]`, None = not measured (observations.py aligned) |
| Scale state | `RELATIVE`/`METRIC` vs `metric_anchored`/`sfm_gauge`/`relative`/`unknown` vs `is_metric` booleans | `ScaleState` (metric | relative) + explicit units |
| Camera frame | `world_ir/camera.py` PinholeCamera vs `perception/spatial_inputs.CameraInput` vs `world_ir/coordinates.Frame` | `coordinates.Frame` + `Transform(source_frame, target_frame)` |
| Observation | `schema_v1.Observation` (world ledger) vs `world_ir/observations.Observation` (perception association) vs `perception/engine_evidence.ObservationRecord` | two distinct, named concepts: `WorldIRObservation` (ledger) + `PerceptionObservation` (association input) |

This document is the single source of truth. Every field below documents: Type / Units / Coordinate frame / Nullability / Validation requirements / Provenance requirements / Persistence behavior / Versioning behavior.

## 1. World

**Schema location:** `world_ir/world_v1.py` `WorldIR`.

A World is the authoritative, versioned container for all reconstructed reality: entities, geometries, materials, surfaces, components, relationships, observations, transforms, branches, scenarios, and provenance.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `id` | `str` (`world-<uuid>`) | — | — | No | non-empty, matches `^world-` | GENERATED | stored as version content | part of content hash |
| `name` | `str` | — | — | Yes ("") | — | UNKNOWN | stored | carried across versions |
| `version` | `int` | — | — | No | >= 1 | GENERATED | stored | schema version, not world version |
| `created_at` / `modified_at` | `float` | epoch seconds | — | No | 0.0 = unset (never `datetime.now()` by default) | GENERATED | stamped explicitly at persist | not part of identity |
| `entities` | `dict[str, Entity]` | — | — | No | unique ids | per-entity | stored | versioned |
| `geometries` | `dict[str, Geometry]` | — | — | No | unique ids | per-geometry | stored | versioned |
| `materials`, `surfaces`, `components` | `dict[str, *]` | — | — | No | unique ids | per-item | stored | versioned |
| `relationships` | on `Entity` | — | — | No | target must exist (dangling = validation issue) | per-relationship | stored | versioned |
| `observations` | `dict[str, WorldIRObservation]` | — | — | No | unique ids | per-observation | stored | versioned |
| `coordinate_frame` | `Frame` | — | — | No | one of `Frame` enum | GENERATED | stored | stated per world |
| `transforms` | `dict[str, Transform]` | — | per-entry source→target | No | frame pairs valid | per-transform | stored | versioned |
| `global_provenance` | `Provenance` | — | — | No | one of enum | GENERATED | stored | — |
| `global_confidence` | `Optional[float]` | [0,1] | — | Yes | None = not measured; never 0.5 default | GENERATED | stored | — |
| `global_uncertainty` | `Uncertainty` | — | — | No | confidence in [0,1] | GENERATED | stored | — |
| `metadata` | `dict` | — | — | Yes | no secrets | UNKNOWN | stored | — |
| `environment`, `physics` | `dict` (opaque) | — | — | Yes | never interpreted/mutated by WorldIR | UNKNOWN | stored | pass-through |
| `temporal_history` | `TemporalHistory` | — | — | Yes | ordered events | per-event | stored | additive (P9-01) |
| `hierarchy` | `Hierarchy` | — | — | Yes | valid parent chain | — | stored | additive (P9-01) |
| `geo_anchor` | `Optional[GeoAnchor]` | — | WGS84 | Yes | valid lat/lon | — | stored | — |

**Invariants:**
- Serialization is bit-identical after deserialization (`to_dict`/`from_dict` round-trip).
- `created_at`/`modified_at` default to `0.0` ("unset") — callers stamp wall-clock explicitly at persist; never `datetime.now()` as a default (determinism).
- `global_confidence` None = not measured; never defaulted to 0.5 (manufactured confidence forbidden).
- No implicit unit or frame conversions anywhere in WorldIR code.

**Validation:** `WorldIR.validate()` returns a list of issues (dangling relationship targets, unknown geometry/material/surface/component references, temporal-event entity references, cause/effect event references, branch parent references, scenario branch references, hierarchy problems). Callers must treat non-empty as invalid before persist.

## 2. WorldVersion

**Schema location:** `worldstore/store.py` `StoredVersion` + version record JSON under `<root>/versions/<vid>.json`.

A WorldVersion is an immutable, content-addressed, persisted snapshot of a WorldIR plus its lineage and integrity metadata.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `version_id` | `str` (`v-<hex12>` or explicit) | — | — | No | `^[A-Za-z0-9_-]{1,128}$`, unique | GENERATED | filename `<vid>.json` | immutable id |
| `world_id` | `str` | — | — | No | matches source world id | GENERATED | stored | links versions of same world |
| `parent` | `Optional[str]` | — | — | Yes | must be an existing version id | GENERATED | stored | forms DAG (V1->V2->V3) |
| `artifact_uri` | `str` (`artifact://<sha256>`) | — | — | No | matches content-addressed store | GENERATED | stored | content hash |
| `artifact_hash` | `str` (sha256 hex) | — | — | No | 64 hex | GENERATED | stored | integrity seal |
| `changed_entity_ids` / `changed_geometry_ids` | `tuple[str]` | — | — | Yes () | computed via `diff_worlds` at save | GENERATED | stored | diff from parent |
| `source_session_ids` | `tuple[str]` | — | — | Yes () | session ids | GENERATED | stored | evidence trace link |

**Persistence behavior (worldstore/store.py `save_version`):**
- Every save serializes the WorldIR to canonical dict + persists referenced artifact bytes (FileArtifactStore) under a new version id.
- Versions are IMMUTABLE: re-saving an existing `version_id` is refused (observed reality never silently overwritten).
- Atomic writes (temp file + fsync + os.replace) — a crash mid-write never leaves a truncated record.
- Cross-process sequence lock guards the read-modify-write of `sequence.json` (lost-update / WinError 5 prevention).
- `verify_version()` re-digests stored bytes against `artifact_hash`; hash mismatch or missing artifact = explicit failure, never silent.
- `load_version()` re-verifies the hash before returning a WorldIR — corrupted/tampered bytes raise `WorldStoreError`.
- Quarantine: never-adopted versions move to `quarantine/` (nothing deleted; content-addressed artifact kept).
- Retention: `prune_quarantine` removes old unreferenced blobs only after age floor + keep-last + no-live-parent checks; live versions and lineage never touched.

**Versioning behavior:**
- Lineage: each version records `parent`; `ancestors()` returns root-first chain; cycle-safe walk.
- Content-addressed: re-persisting an identical world produces the same digest (idempotent re-persist safe).
- A successful job must not conceal a failed persistence operation — every persistence result is verified (`verify_version` clean) before the version is adopted as HEAD.

## 3. Entity

**Schema location:** `world_ir/schema_v1.py` `Entity`.

An entity is a world object with full provenance tracking.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `id` | `str` (`ent-<uuid>`) | — | — | No | non-empty | per-entity | stored | versioned; identity by continuity, not re-derived id |
| `type` | `EntityType` | — | — | No | one of enum | per-entity | stored | versioned |
| `name` | `str` | — | — | Yes | — | UNKNOWN | stored | — |
| `transform` | `dict\|Transform` | — | source→target | Yes | valid Transform frame pair | per-entity | stored | versioned |
| `geometry_ids` | `list[str]` | — | — | No | each must exist in `geometries` | per-entity | stored | versioned |
| `material_ids`, `surface_ids`, `component_ids` | `list[str]` | — | — | No | each must exist | per-entity | stored | versioned |
| `relationships` | `list[Relationship]` | — | — | No | target exists (dangling = issue) | per-relationship | stored | versioned |
| `semantic_labels` | `list[str]` | — | — | Yes | — | UNKNOWN | stored | — |
| `observations` | `list[WorldIRObservation]` | — | — | No | unique ids | per-observation | stored | versioned |
| `temporal_events` | `list[TemporalEvent]` | — | — | No | — | per-event | stored | versioned |
| `provenance` | `Provenance` | — | — | No | one of enum | per-entity | stored | versioned |
| `confidence` | `Optional[float]` | [0,1] | — | Yes | None = not measured; never 0.5 default | measured/stated | stored | versioned |
| `confidence_basis` | `Optional[str]` | — | — | Yes | required if confidence measured | measured | stored | — |
| `confidence_vector` | `ConfidenceVector` | per-dim [0,1] | — | Yes | each None or [0,1] | measured | stored | additive (P9-01) |
| `uncertainty` | `Uncertainty` | — | — | No | confidence [0,1] | per-entity | stored | versioned |
| `custom_properties` | `dict` | — | — | Yes | no secrets | UNKNOWN | stored | extension slot |
| `statement_state` | `Optional[StatementState]` | — | — | Yes | one of enum | derived | stored | additive (P9-01) |
| `external_references` | `tuple[ExternalReference]` | — | — | Yes | kind/uri/role required; checksum `<algo>:<hex>` | per-reference | stored | additive (P9-01) |

**Contract rules:**
- `None` means "not measured" — never fabricated, never defaulted to 0.5.
- `confidence_basis` documents how a measured confidence was obtained; required whenever `confidence` is not None.
- `Entity.material` / `material_confidence` read from `custom_properties.attributes` (extension slot) — not a competing top-level field.
- One physical object = one entity however many photos show it; identity continuity across versions is by TYPE + POSITION in the previous frame (ids are re-derived each rebuild and are NOT stable match keys).

## 4. Observation

There are TWO distinct observation concepts; do not conflate them.

### 4a. WorldIRObservation (world ledger)

**Schema location:** `world_ir/schema_v1.py` `Observation`.

The record of one sensory input stored inside a WorldIR version.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `id` | `str` (`obs-<uuid>`) | — | — | No | non-empty | — | stored | versioned |
| `sensor_type` | `str` | — | — | Yes | "camera\|lidar\|radar\|manual" | — | stored | — |
| `timestamp` | `float` | epoch seconds | — | No | — | — | stored | — |
| `frame_id` | `str` | — | sensor frame | Yes | — | — | stored | — |
| `data_uri` | `str` | — | — | Yes | file:// or remote URL | — | stored | — |
| `data_hash` | `str` | — | — | Yes | SHA256 | — | stored | integrity |
| `metadata` | `dict` | — | — | Yes | sensor-specific (focal length, etc.) | — | stored | — |
| `confidence` | `Optional[float]` | [0,1] | — | Yes | None = not measured | measured | stored | — |
| `uncertainty` | `Uncertainty` | — | — | No | confidence [0,1] | per-observation | stored | — |

### 4b. PerceptionObservation (association input)

**Schema location:** `world_ir/observations.py` `Observation` (perception association input; separate from the WorldIR ledger above).

One 2D detection lifted into the world frame, used by multi-view association.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `observation_id` | `str` | — | — | No | non-empty | — | ledger | — |
| `image_id` | `str` | — | — | No | — | — | ledger | — |
| `label` | `str` | — | — | No | — | — | ledger | — |
| `world_pos` | `(x,y,z)` | m | world | No | finite | measured | ledger | — |
| `sigma_m` | `float` | m | — | No | >= SIGMA_FLOOR_M (0.05) | measured | ledger | — |
| `depth_source` | `DepthSource` | — | — | No | metric\|mvs\|relative | measured | ledger | — |
| `model` | `str` | — | — | Yes | `name@version` | measured | ledger | — |
| `license` | `str` | — | — | Yes | — | measured | ledger | — |
| `normal` | `(x,y,z)` | — | — | Yes | unit | measured | ledger | — |
| `bbox_px` | `(x0,y0,x1,y1)` | px | image | Yes | — | — | ledger | — |
| `confidence` | `Optional[float]` | [0,1] | — | Yes | **None = not measured; never 0.5** | measured | ledger | — |
| `mask_uri` / `mask_ref` | `str` | — | — | Yes | artifact store URI | — | ledger | SAM2 mask ref (coordinate with Cline) |
| `visibility` | `Visibility` | — | — | Yes | full\|partial\|occluded\|not_observed | — | ledger | — |
| `embedding_ref` | `str` | — | — | Yes | reference, never the vector | — | ledger | — |
| `session_id`, `created_at` | `str` | — | — | Yes | ISO-8601 | — | ledger | — |

**Contract rule (aligned with WorldIR):** `confidence` None = not measured; the `0.5` fabricated default was removed (see integrity fix). Association code (`perception/instances/association.py`) treats None as "no measurement" and does not fabricate a positive confidence.

## 5. Evidence

Evidence is the raw capture an observation traces back to. It is NOT an observation; an observation is a detection lifted from evidence.

Evidence fields live on the acquisition `Evidence` record (apps/api) and are linked to observations via `observation.evidence_id` / `data_uri` + `data_hash`.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `evidence_id` | `str` | — | — | No | — | — | stored | — |
| `image_id` | `str` | — | — | No | — | — | stored | — |
| `source_uri` | `str` | — | — | No | resolvable artifact | — | stored | — |
| `sha256` | `str` | — | — | No | 64 hex | — | stored | identity |
| `width`/`height` | `int` | px | — | No | > 0 | — | stored | — |
| `ok` / `geometry_eligible` | `bool` | — | — | No | — | — | stored | — |
| `checksum` | `str` | — | — | No | sha256 of stored artifact | — | stored | byte-identity |

**Contract rule:** Byte-identical evidence is deduplicated (first kept, duplicates dropped with reason). Garbage "photos" are refused before COLMAP. Missing evidence never becomes positive evidence.

## 6. Geometry

**Schema location:** `world_ir/schema_v1.py` `Geometry`.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `id` | `str` (`geom-<uuid>`) | — | — | No | — | — | stored | versioned |
| `type` | `GeometryType` | — | — | No | mesh\|pointcloud\|plane\|box\|sphere\|cylinder\|capsule\|terrain\|voxel\|implicit\|unknown | — | stored | versioned |
| `lod_level` | `int` | — | — | No | 0=coarse | — | stored | — |
| `vertex_count` / `triangle_count` | `Optional[int]` | — | — | Yes | >= 0 | — | stored | — |
| `data_uri` / `data_hash` | `str` | — | — | Yes | `sha256://<h>` + 64 hex | — | content-addressed | stored via artifact store |
| `bounds_min` / `bounds_max` | `Optional[Vector3]` | m | world | Yes | not inverted, finite | — | stored | versioned |
| `provenance` | `Provenance` | — | — | No | one of enum | — | stored | versioned |
| `confidence` | `Optional[float]` | [0,1] | — | Yes | None = not measured | measured | stored | — |
| `quality_metrics` | `Optional[dict]` | — | — | Yes | recorded facts, never fabricated | measured | stored | — |
| `observations` | `list[WorldIRObservation]` | — | — | No | — | — | stored | — |

**Contract rules:**
- Geometry is referenced through `(data_uri, data_hash)` content-addressed artifact pointers — never embedded payloads (worldstore decision).
- A bounding box is an approximation labeled INFERRED — never presented as mesh geometry.
- Non-finite values or inverted bounds = validation ERROR.
- Mesh validity (watertight, degenerate faces, components) reported via `s12_quality_report` (NOT_EVALUABLE / PASS / WARN / FAIL); quality never silently upgraded to PASS.

## 7. CoordinateFrame & Camera

**Schema location:** `world_ir/coordinates.py` `Frame`, `Transform` (+ `world_ir/camera.py` `PinholeCamera` for camera intrinsics/extrinsics).

| Item | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `Frame` | enum | — | — | No | camera\|sensor\|session-local\|building-local\|world\|ENU\|UTM\|WGS84\|ECEF\|engine-local\|storey-local | — | stored | — |
| `Transform` | dataclass | — | source→target | No | 4x4 matrix, frame pair valid | per-transform | stored | versioned |
| `Transform.matrix` | `Mat4` (4x4) | — | — | No | finite, homogeneous | — | stored | — |
| `Transform.uncertainty` | `Uncertainty` | — | — | No | confidence [0,1]; default 0.0 (unknown), **never 1.0 unless measured** | — | stored | — |
| `PinholeCamera` | dataclass | — | camera | — | intrinsics + extrinsics | — | stored | — |
| `CameraIntrinsics` | `(fx,fy,cx,cy,width,height)` | px/px/px/px/px/px | — | No | positive, agrees within 1% across cameras | estimated | stored | adopted post-solve |
| `CameraExtrinsics` | `(position, rotation)` | m / wxyz | world | No | — | — | stored | — |

**Contract rules:**
- Every transform stores `source_frame`, `target_frame`, matrix, timestamp, uncertainty — never a bare matrix with implicit meaning.
- Right-handed, Y-up canonical world frame (axis convention). `R_UP` maps job frame (+Y up) ↔ package working frame (Z-up); every record states its frame.
- SfM frame mapped into job frame by residual-checked similarity.
- `make_rigid_transform` default uncertainty is `Uncertainty()` = confidence 0.0 (unknown), NOT 1.0. A transform whose producer did not state confidence must not read as certain.
- Transform composition (`then`) multiplies confidences — a chained transform is never more certain than either hop.
- No implicit frame conversion; camera poses = camera CENTERS with camera-to-world rotations (wxyz quaternion).

## 8. SpatialRelationship

**Schema location:** `world_ir/schema_v1.py` `Relationship`.

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `kind` | `RelationshipKind` | — | — | No | one of enum | — | stored | versioned |
| `target_id` | `str` | — | — | No | must reference an existing entity (dangling = validation issue) | — | stored | versioned |
| `confidence` | `Optional[float]` | [0,1] | — | Yes | None = not measured | measured | stored | — |
| `provenance` | `Provenance` | — | — | No | one of enum | — | stored | versioned |
| `metadata` | `dict` | — | — | Yes | — | — | stored | — |

**Contract rule:** An unstated confidence/provenance must never read as "observed with certainty" — defaults say "not measured, unknown origin."

## 9. Confidence

**Schema location:** `world_ir/schema_v1.py` `ConfidenceVector` + per-field `Optional[float]`.

Confidence is a producer-stated probability in [0.0, 1.0]. It is NOT geometric accuracy.

| Concept | Representation | Nullable | Meaning |
|---|---|---|---|
| Scalar confidence (entity/geometry/observation/relationship) | `Optional[float]` | Yes | None = not measured |
| `confidence_basis` | `Optional[str]` | Yes | how measured; required when confidence is not None |
| `ConfidenceVector` | 11 dims (classification, identity, position, orientation, dimensions, geometry, completeness, material, colour, topology, cross_view) | each None or [0,1] | None = dimension not measured |
| `Uncertainty.confidence` | `float` | No | 0.0 default = unknown; never 1.0 unless measured |

**Contract rules (mandate §4):**
- Confidence is not confused with geometric accuracy.
- Missing evidence does not become positive evidence (None never coerced to 0.0/0.5/1.0 as "measured").
- A confidence is never upgraded (compiler never upgrades a value's status).
- `Uncertainty` default is 0.0 (unknown), never 1.0.

## 10. Uncertainty

**Schema location:** `provenance/provenance.py` `Uncertainty` + `uncertainty/__init__.py` `Uncertain` (value + sigma + basis).

| Field | Type | Units | Frame | Nullable | Validation | Provenance | Persistence | Versioning |
|---|---|---|---|---|---|---|---|---|
| `confidence` | `float` | [0,1] | — | No | 0.0 = unknown, never 1.0 by default | — | stored | — |
| `note` | `Optional[str]` | — | — | Yes | — | — | stored | — |
| `calibrated` | `bool` | — | — | No | True only with measured physical ground truth | — | stored | — |
| `Uncertain.value` / `.sigma` / `.basis` | value / m-or-rad / str | per-quantity | — | sigma None = unknown | None propagates as UNKNOWN (never zero, never high confidence) | measured | stored | — |

**Contract rules:**
- Spread (sigma) and confidence (producer-stated probability) are different facts and are never converted into one another.
- UNKNOWN propagates as UNKNOWN: a quantity computed from an unknown input is unknown (None), never zero and never high confidence.
- Downstream geometry inherits upstream sigma (opening position carries host offset sigma; room area carries propagated corner-node sigma).

## 11. Provenance

**Schema location:** `provenance/provenance.py` `Provenance`.

| Value | Meaning |
|---|---|
| `OBSERVED` | directly promoted from sensor evidence |
| `RECONSTRUCTED` | computed by reconstruction (fitted quantity, not directly observed) |
| `ESTIMATED` | estimated from partial evidence |
| `INFERRED` | inferred |
| `GENERATED` | produced by simulation/procedural rule |
| `UNKNOWN` | origin not known |
| `CONFLICT` | conflicting observations |

**Contract rule:** GENERATED and UNKNOWN values are never canonical reality. `Provenanced.is_canonical()` returns False for GENERATED/UNKNOWN/CONFLICT.

## 12. Identity

| Concept | Contract |
|---|---|
| Entity id | `ent-<uuid>`; re-derived each rebuild — NOT a stable match key |
| Identity continuity | by TYPE + POSITION in previous frame (world_delta.entity_records); ids re-derived, so matching is by measured continuity |
| Observation id | `obs-<uuid>`; unique per detection |
| Geometry id | `geom-<uuid>`; content-addressed via data_hash |
| Version id | `v-<hex12>`; content hash of world dict |
| One physical object | = one entity however many photos show it; a detection is an observation, never an entity |
| Do not merge | entities solely because class labels match |

## 13. Mask references & Asset references

| Concept | Contract |
|---|---|
| Mask ref | `observation.mask_uri` / `mask_ref` — artifact store URI, never embedded pixels; coordinate with Cline on SAM2 mask references |
| Asset ref | `ExternalReference` (kind, uri, role, checksum `<algo>:<hex>`, provenance) — typed link to a resource the world does not own; checksum RECORDED, never computed by WorldIR |
| Geometry ref | `Geometry.data_uri`/`data_hash` — `sha256://<h>` content-addressed |
| ExternalReference validation | kind, uri, role required; checksum must be `<algo>:<hex>` with algo in {sha256,sha1,md5,blake3} |

## 14. Change history

| Concept | Contract |
|---|---|
| Version lineage | `WorldStore.parents()` / `ancestors()` — root-first chain, cycle-safe |
| Diff | `world_ir.diff.diff_worlds` + `world_delta` — per-region change (added/extended/refined/preserved/represented_differently/not_reproduced/uncertain), conflicts never silently resolved |
| Change record | `StoredVersion.changed_entity_ids` / `changed_geometry_ids` computed via `diff_worlds(parent, world)` at save |
| Adoption | intent file → WorldStore version → read-back verify → DB HEAD + mirror row in ONE transaction → intent cleared; failed adoption undoes its own pending HEAD move |
| Rollback | moves only the HEAD pointer; versions stay immutable; COLMAP base set aside, next version chains onto rolled-back HEAD |

## 15. What this contract forbids (mandate §9)

- Do not manufacture confidence (`None` is not `0.5`).
- Do not silently drop observations.
- Do not hide invalid geometry (validation issues reported, never healed).
- Do not convert unknown values into plausible defaults.
- Do not create a second competing WorldIR (this is the single contract).
- Do not weaken existing tests.
- Do not replace production functionality with mocks.
- Do not claim geometric accuracy without ground truth.
