# 11 — WorldIR contract (the shared data model)

Every stage, the API, the UI and every exporter speak this. If you need a field
that is not here, add it here first (with a reason), then in code.

**Implementation home:** `world_ir/` (`schema_v1.py`, `v2_extensions.py`, and
`observations.py` on the integration branch). Do **not** start a new schema
module — extend these. Version the schema (`schema_version`) and keep a
migration for every change (`world_ir/serialization.py`).

## 1. Hierarchy (P8)

```
World
 └─ Building*          (optional for a single room; implicit default building)
     └─ Floor*         (implicit default floor for v1)
         └─ Space*     (room, corridor, stairwell, outdoor zone)
             ├─ Surface*      (wall face, floor region, ceiling region)
             │    └─ Opening* (door, window, passage) → links to an Entity (the door/window object)
             └─ Entity*       (sofa, TV, socket, plant …)
                  └─ EntityComponent*  (cushion, leg, sash …)
Camera*, Image*, CaptureSession*  (world-level)
Observation*, AssociationDecision* (ledger, world-level)
Relationship*, Measurement*, CaptureRequest*, ValidationReport (world-level)
```

A single room uses the same tree with one Building, one Floor and one Space.

## 2. Records (target fields)

Types: `Vec3 = [x,y,z]`, `Quat = [w,x,y,z]`, IDs are strings, times are ISO-8601 UTC.

### World (per version)
`id, schema_version, version_id, parent_version_id, name, coordinate_frame,
scale_state (relative|estimated|calibrated), scale_source?,
geo_anchor? (lat, lon, alt, heading, crs, accuracy_m), created_at`

### Space
`id, kind (room|corridor|stair|outdoor|unknown), name?, parent_floor_id,
boundary_polygon (floor-plane polygon), floor_height, ceiling_height?,
evidence_state, confidence, connected_space_ids[]`

### Surface
`id, kind (wall|floor|ceiling|other), space_id, plane (normal Vec3, offset), polygon (3D),
material_id?, evidence_state, coverage (0–1 observed fraction), observation_ids[]`

### Opening
`id, kind (door|window|passage), surface_id, polygon, width, height, sill_height?,
entity_id? (the door/window object), evidence_state`

### Entity
```
id                    stable across versions
class_path            ontology path, e.g. "furniture/seating/sofa/sectional"
name                  human readable
parent_space_id
parent_entity_id?     when it is mounted on / embedded in another entity
transform             position Vec3, rotation Quat
bounds                oriented box: center, half_extents, rotation
geometry_refs[]       {kind: parametric|mesh|points|proxy, lod: 0..4, uri, hash}
components[]          EntityComponent
material_ids[]
mobility              fixed|movable|unknown
evidence_state        overall; per-part states live on components/surfaces
confidence            ConfidenceVector
observation_ids[]     all supporting observations (provenance, P7)
association_ids[]     decisions that created/updated it
first_seen_version, last_updated_version
```

### EntityComponent
`id, role (e.g. "seat_cushion", "leg"), class_path?, transform (local), geometry_refs[],
evidence_state, confidence`

### Image / Camera
`Image: id, evidence_id (original file), session_id, width, height, captured_at?, exif{},
quality{sharpness, exposure, blur_flag}, gps?`

`Camera: image_id, intrinsics (model, fx, fy, cx, cy, distortion[]), pose (position, rotation)
in world frame, registration_state (registered|rejected|pending), reprojection_error_px?`

### Observation (ledger) — matches `world_ir/observations.py`
```
observation_id, image_id, label (ontology path or raw label), model ("name@version"), license,
mask_uri? | bbox_px?, world_pos Vec3, sigma_m, depth_source (metric|mvs|relative),
normal?, bounds_min?, bounds_max?, confidence, visibility (full|partial|occluded),
embedding_ref?, session_id, created_at
```
Fields `mask_uri`, `bbox_px`, `visibility`, `embedding_ref`, `session_id`,
`created_at` are additions to the current dataclass.

### AssociationDecision — matches `world_ir/observations.py`
`observation_id, state (see glossary), entity_id?, probability, reason, dist_m?,
mahalanobis_sq?, normal_diff_deg?, geom_verified?, candidates[(entity_id, score)]`

### ConfidenceVector
`classification, identity, position, orientation, dimensions, geometry,
completeness, material, colour, topology, cross_view` — each `float 0..1 | null`.
`null` means "not measured", never a default 0.5.

### Relationship
`id, kind, source_id, target_id, evidence_state, confidence, basis ("geometric"|"semantic"|"user")`

Kinds (v1 set): `inside, contains, part_of, supported_by, rests_on, mounted_on,
embedded_in, attached_to, adjacent_to, connects (space↔space via opening), occludes (per image)`.
Extend the existing `RelationshipKind` enum in `schema_v1.py`; don't add a second enum.

### Measurement
`id, kind (distance|width|height|depth|area|volume), target_ids[], value, unit,
uncertainty?, scale_state, method ("multi-view"|"plane-fit"|"user"|...), provenance_ids[]`

### CaptureRequest
`id, reason, target_region (bounds), target_ids[], suggested_poses[{position, look_at}],
expected_gain (0–1), instruction (short text), status (open|satisfied|dismissed), created_version`

### ValidationReport
`version_id, checks[{name, passed, detail}], duplicate_entity_count, conflict_count,
registered_image_ratio, coverage_fraction, warnings[]`

## 3. Invariants (the validator must enforce them — `world_ir/validation.py`)

1. Every Entity/Surface/Opening has an `evidence_state`.
2. Every Entity with `evidence_state ∈ {OBSERVED, PARTIALLY_OBSERVED}` has ≥1 `observation_id`.
3. Anything `INFERRED` or `PRIOR_GENERATED` is never also labelled `OBSERVED`.
4. Metric units appear only when `scale_state ≠ relative`.
5. IDs are unique; every referenced ID exists.
6. No two entities of the same class whose bounds overlap > 50 % IoU, unless a
   decision marked them `DIFFERENT_ENTITY` or `PHYSICAL_INTERSECTION` (duplicate guard, P2).
7. Each Opening references a Surface; each Surface references a Space.

## 4. Versioning & change

- A version is immutable. A change = a new version with `parent_version_id`.
- `world_ir/diff.py` computes added/removed/changed entities between versions.
- Entity IDs carry across versions via `world_ir/entity_reid.py`.
- Incremental updates touch only the affected closure (`world_ir/incremental.py`).
