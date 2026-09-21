# Capabilities Manifest

What the Reality Engine can do **today**, verified 2026-09-14.
Companion to `IMPLEMENTATION_STATUS.md` (component detail) and
`PENDING_IMPLEMENTATION.md` (what is missing). A capability listed here
has been executed against real inputs unless marked otherwise.

## Compile a capture into a world

```
reality compile ./capture            # one command, full spine
```

Input: a capture directory (images/video + manifest). Output: WorldIR
JSON + persisted artifacts + report. Backend injection via
`REALITY_TEST_BACKEND` supports deterministic offline runs.

- SfM camera poses (COLMAP), metric scale from measurement priors
- TRUE dense MVS (COLMAP patch_match_stereo → stereo_fusion): wired
  as a backend continuation (`dense_mvs=True`) — runs in the same
  workspace as sparse SfM, probes the binary's dense capability and
  per-stage CLI options, parses fused.ply into canonical points, and
  ingests the cloud through the content-addressed ArtifactStore into
  WorldIR geometry with dense provenance. Real GPU dense run on the
  committed real-photo dataset recorded 2026-09-21.
- Monocular depth (MiDaS) → fused point cloud (provenance per view)
- Poisson surface mesh from oriented fused points (CPU COLMAP
  mesher; auto depth/trim adaptation; camera-envelope outlier
  filtering from sparse SfM points)
- WorldIR world: entities (planes from measurements, objects from
  detections), measurements, metadata (scale state, frame)
- glTF export incl. real triangle mesh + point primitives

Verified on a real room capture: 25/28 cameras registered, 15.8 m
honest extent, 240,735-vertex mesh, 71 entities.

## Artifact store

Content-addressed (sha256), integrity-checked storage for meshes,
clouds, reports, exports; artifacts referenced from WorldIR via
`data_uri` + `data_hash`. Deterministic compilation: identical inputs
produce identical artifact hashes.

## Deterministic testing path

Fake reconstruction backend + synthetic fixtures run the full spine
offline (no COLMAP, no torch) — the backbone of the test suite
(1,279 passing at reconciliation; SAM-perception env failure
excluded, tracked).

## Explicitly NOT capabilities yet

- Sensor depth ingestion (16-bit PNG sidecars) — spec written
- Time sync / VIO / cross-source registration — specs written
- Multi-view object identity, tracking, materials — specs written
- Uncertainty propagation, provenance graph, WorldStore, incremental
  compilation — specs written
- GIS, robotics, large-world partitioning, advanced physics beyond
  the rigid-body foundation — specs written

Each "not yet" is specified under `docs/future/` with model, failure
modes, and acceptance criteria.
