# Reality Engine Target Architecture — World-Model Compiler

Authoritative baseline: `docs/REALITY_ENGINE_CURRENT_STATUS.md` (reconciled 2026-09-23), `docs/ARCHITECTURE.md` (V11 module dependency rules), `docs/TECHNOLOGY_DECOMPOSITION.md`, `docs/research/capability-fusion-matrix.csv`, `docs/research/INTEGRATION_DECISIONS.md`.

This document states the target architecture as Reality Engine evolves into a system that transforms raw evidence into a persistent, measurable, semantically meaningful, versioned representation of the real world. It refines (does not replace) the existing layered boundaries; nothing here authorizes rewriting a working component.

## Guiding principles (carried from the mission)

1. **Evidence first:** every WorldIR value carries provenance (OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT) and is traceable to source evidence.
2. **Never silently mix coordinate systems:** the frame graph resolves transforms explicitly; ambiguity, disconnected frames, invalid transforms, and inconsistent timestamps are diagnosed, not coerced.
3. **Never overwrite a trusted world:** WorldStore versions are immutable, content-addressed, lineage-tracked, and integrity-verified; updates are explainable candidates.
4. **Provider interchangeability:** SfM/MVS, depth, segmentation, registration are provider interfaces with explicit I/O contracts, capabilities, version info, resource requirements, and error reporting. COLMAP remains the SfM/MVS baseline until a controlled benchmark justifies change.
5. **Honest failure:** backends raise rather than fabricate; absence of data uses `unavailable`/`failed`, never zombie data under a success status.
6. **Do not rewrite what works:** RE's metric scale anchoring, cross-view depth consistency, ICP/GNSS registration, WorldIR v1/v2, WorldStore, exporters, CLI, and API are real and tested; they are RETAINED, not rewritten.

## Layer 1 — Evidence ingestion (existing, RETAIN)

- Folder/file importers, EXIF/GPS, quality, dedup, packages (`evidence.importers`, `evidence.packages`, `evidence.session`).
- MultiSourceSession + full CLI (create/add-source/list/inspect/export).
- Validation, deduplication, evidence manifests, sensor metadata, timestamps, quality assessment.
- Every output traceable to source evidence.

No change required. Gaps noted: large-building stress testing; video walkthrough ingestion (SAM2 temporal tracking is a future trigger, not today).

## Layer 2 — Perception and reconstruction (existing providers, RETAIN + DEFER)

Provider interfaces already defined:
- `IReconstructionBackend` → `ColmapReconstructionBackend` (real, COLMAP on PATH) + `FakeReconstructionBackend` (tests only).
- `IDepthBackend` → `MiDaSDepthBackend` (blocked at runtime: no checkpoint, no GPU).
- `ISegmentationBackend` → `SAMSegmentationBackend` (blocked at runtime: no checkpoint, no GPU).
- Registration: `RegistrationEngine` (GNSS anchor + point-to-point ICP + point-to-plane ICP + covariance).

Decision per `capability-fusion-matrix.csv`:
- COLMAP: RETAIN as SfM/MVS baseline (WRAP as subprocess).
- Open3D point-cloud registration/processing: DEFER (open3d wheel not installable — PyPI unreachable); RE's ICP already satisfies baseline.
- Depth (MiDaS/Depth Anything V2) and segmentation (SAM): DEFER (checkpoints unavailable, no GPU, per-checkpoint licensing unresolved).
- Open3D/GTSAM pose-graph + multi-session alignment: DEFER (not a defined slice; not installed).

Providers must declare capabilities, version, resource requirements, and error reporting; failures are distinguishable from algorithm failures.
## Layer 3 — Canonical WorldIR (existing, STRENGTHEN)

- `world_ir/world_v1.py` (WorldIR V1), `schema_v1.py` (Entity/Geometry/Material/Surface/Component/Relationship/TemporalEvent/CausalRelation/Observation), `coordinates.py` (Frame/Transform), `v2_extensions.py`, `diff.py`, `frame_graph.py`, `serialization.py`, `validation.py`, `incremental.py`, `entity_reid.py`, `artifact_store.py`.
- Stable entity identities, geometry/bounds, transforms/frames, materials/appearance, semantic labels/relationships, observations/evidence, confidence/uncertainty, units/CRS, provenance/history, temporal validity, version/schema.
- Bit-identical serialization round-trip invariant.

Target evolution: schema migrations (not silent format changes), content-addressed hashing of persisted worlds, explicit temporal validity windows. No provider-specific data model may leak into core; adapters + schema migrations only.

## Layer 4 — Registration and world fusion (existing, STRENGTHEN selectively)

- GNSS + local-frame alignment (`register_gnss_anchor`).
- Point-cloud registration: ICP point-to-point, ICP point-to-plane (`register_icp`, `register_icp_point_to_plane`), `ResidualStats` from measured nearest-neighbor correspondences, `RegistrationCovariance` from measured residuals, `AttemptRecord` honesty.
- Coordinate-frame graph: cycle detection, disconnected components, ambiguity detection, timestamp-validity windows, transform composition with uncertainty propagation.

Target evolution (DEFERRED, not a current slice): pose-graph optimization (GTSAM), loop closure, multi-session alignment, landmark co-observation resolver (marked as an honest gap in status), robust outlier rejection beyond ICP inlier fraction.

Safety rule: never silently force incompatible coordinate systems together; return structured diagnostics on ambiguity/disconnected frames/invalid transforms.

## Layer 5 — WorldStore and temporal evolution (existing, STRENGTHEN + EXTEND)

- `worldstore/store.py`: immutable content-addressed versions, `sequence.json` lineage DAG, `_SequenceLock` cross-process mutex, `verify_version` hash re-digest, save/load/list/verify CLI, `lazy_query.py`, `tiles.py`.
- Immutable versions, explicit lineage, integrity checks, survives process restart.

Target evolution: candidate world revisions, geometric/semantic diffs (wire `world_ir/diff.py` more tightly), change detection, evidence-backed update proposals, confidence-aware conflict resolution, rollback/recovery, verified commits + read-back integrity.

## Layer 6 — World compilation and semantic understanding (existing, PARTIAL)

- Compiler: planes -> classify -> promote -> rooms -> validation gate (`reconstruction/orchestrator`, compiler gates).
- Plane/surface detection, room/floor segmentation exist but with named limits (room inference: `PlaneSummary.plane` bug is a pre-existing failure; see KNOWN_LIMITATIONS).
- Scene graph, object relationships, semantic/geometric consistency exist in part.

Distinguish measured facts from inferred geometry; do not manufacture semantic certainty from weak evidence. Pre-existing room-inference/compiler failures must be fixed or explicitly gated, not hidden.

## Layer 7 — Interoperability (existing exporters, EXTEND selectively)

- glTF/USDA (Blender)/CityJSON/CityGML (13 modules) with real-geometry paths + reports.
- Target: round-trip tests where possible; each adapter declares what it preserves/transforms/approximates/loses.
- IFC/USD adapters: DEFER (BIM not a current slice; license mix to verify).
- 3D Tiles: DEFER (geospatial slice not prioritized).

## Layer 8 — Spatial querying and visualization (existing, RETAIN)

- Accelerated `SpatialIndex`: uniform 3D grid + per-cell BVH, nearest(k)/within_radius(r)/within_region(bounds), flat fallback <1000, deterministic ordering; benchmarked at 100/1K/10K/100K entities.
- Frontend: Next.js 16 + Three.js viewer, 12 API proxy routes, desktop + mobile layouts, selection + provenance; offline single-file viewer.
- Target: LOD/streaming for city-scale, measurements UI, timeline, diff views, uncertainty visualization.

## Layer 9 — Engineering automation (REA as research instrument; extend CLI/API diagnostics)

- REA used for source investigation (this cycle); live binary analysis blocked (no Ghidra/IDA/Hopper; Node 26.8.1 below REA's 26.12.1 minimum).
- Controlled developer interface via existing CLI/API: inspect session, run diagnostics, validate reconstruction, compare providers, generate benchmark, inspect world diff, prepare export.
- Write operations on trusted world state remain under explicit validation + integrity checks; no unrestricted agent mutation authority.

## Backward compatibility & migration

- Preserve existing CLI commands, API contracts, WorldIR v1 schemas, WorldStore invariants.
- Schema migrations rather than silent format changes.
- Deterministic tests; timeouts/resource limits/cleanup for subprocesses; interrupted-job and partial-artifact safety.
- No secrets in logs.

