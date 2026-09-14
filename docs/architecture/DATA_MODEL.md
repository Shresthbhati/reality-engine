# Reality Engine — Data Model

Canonical data flow and the contract between each layer. Types below are
the **runtime-owned** representations; JSON schemas for persistence are
defined by their `to_dict()`/`from_dict()` pairs.

## 1. Source

What the user supplied. Managed by `evidence/multi_source.py`.

- `MultiSourceSession` — the acquisition container
- `SourceRecord` — one file/stream/dataset: identity, content hash,
  format, acquisition metadata, classification (photo/video/lidar/rgbd/…)
- Unified source identity (P1-01): every record carries
  `acquisition_id` (content-derived `acq-<hash16>` by default, so the
  same bytes ingested anywhere share one acquisition identity — the
  dedupe rule made explicit; a real device-side acquisition uuid
  overrides it once the sensor model carries one), `device_id`
  (only what the caller declared via an explicit `EvidenceSource` —
  the default filesystem source records `None`, never a fabricated
  device name), and `capabilities` (the evidence kinds the source
  actually ingested, derived from the package — a measured fact, not
  a declared promise).
- Identity rule (Decision 005): a session `SourceRecord` id and the
  `EvidenceSource` it feeds must be coherently linked — no silent
  identity split between `src-XXXX` and `disk:filename`.

## 2. Evidence

What the engine extracted. `evidence/` package.

- `EvidenceItem` (`evidence/schema.py`) — id, kind, source_uri,
  sha256, metadata, provenance. Immutable.
- `EvidencePackage` (`evidence/packages.py`) — deduplicated,
  content-addressed bundle with deterministic ids.
- `ArtifactStore` (`world_ir/artifact_store.py`) — binary payloads
  (points, meshes, depth) live here, content-addressed with SHA256.
  Large binaries NEVER go into JSON rows or WorldIR bodies; WorldIR
  carries `(data_uri, data_hash)` references.

## 3. Reconstruction

`reconstruction/` package.

- `ReconstructionResult` — `ReconstructedPoint` (position, track_id,
  source_evidence_ids, uncertainty), `ReconstructedCameraPose`
  (evidence_id, position, quaternion), `registration_status`
  (`success | partial | failed` — never fabricated).
- `ScaleState` — `metric | relative`; a metric world carries
  `meters_per_unit` and its anchoring provenance. Relative depth/geometry
  must never be silently promoted to metric (metric-by-alignment is
  labeled as such).
- `MeshData` (`reconstruction/meshing/mesh.py`) — **canonical mesh
  representation**: vertices, triangle indices, normals, coordinate
  frame, metadata. COLMAP/OpenMVS are backend representations;
  Blender/glTF/USD are exports (Decision 002).

## 4. Perception

`perception/` package.

- `Detection` (`perception/detection/interface.py`) — label, bbox,
  score, source evidence id, model provenance.
- `SegmentedRegion` — mask tied to source evidence + confidence.
- `ObjectHypothesis` (`perception/instances/lifting.py`) — 3D points,
  centroid, bounds, dimensions, provenance (`INFERRED` for AABB
  approximations), evidence references.
- Multi-view resolution merges hypotheses into one entity when
  geometric + label evidence supports it.

## 5. WorldIR (canonical)

`world_ir/world_v1.py` — the single source of semantic truth.

- `WorldIR` — entities, geometry artifacts, materials, measurements,
  relationships, provenance, confidence, uncertainty, metadata.
- `Entity` — id, kind, transform, observations (append-only history),
  provenance.
- `Geometry` — `type` (MESH/POINTS/…), `data_uri`, `data_hash` into the
  ArtifactStore, vertex/face counts, coordinate frame.
- `Measurement` — value, units, method, provenance
  (`measured | reconstructed | inferred | estimated`).
- Validation (`world_ir/validation.py`) fails compilation loudly on
  broken references, missing artifacts, invalid bounds, frame errors.

## 6. Applications

- Viewer/Studio: consume WorldIR + ArtifactStore only; no independent
  semantic truth.
- Exporters (glTF/USD/Blender): read WorldIR, resolve artifacts,
  emit format-specific representations.

## Units and frames

- All metric geometry is meters, +Y up after frame canonicalization
  (dominant plane → floor), camera frames use the reconstruction
  backend's convention with explicit conversion at the unprojection
  boundary (`reconstruction/depth_to_points.py`).
- Time: sensor-local timestamps are retained; a canonical global time
  model (`t_global = a·t_sensor + b`) is the P1.6 target —
  [`../future/synchronization/TIME_SYNCHRONIZATION.md`](../future/synchronization/TIME_SYNCHRONIZATION.md).
