# Large Worlds

Status: MISSING (2026-09-14) — deferred (ROADMAP P5+)

## Purpose

Keep compilation and rendering correct and bounded as worlds grow
from one room to a building/site/neighborhood: memory-bounded
compilation, spatial indexing, streaming, LOD.

## Current state

Everything assumes one session, one world, one frame; meshes load
whole in the viewer (a 240k-vertex room mesh is fine; a building is
not). No spatial index beyond ad hoc k-d trees in processing stages.

## Scope when implemented

1. **Spatial partitioning.** World → tiles/chunks (grid in world
   frame); entities indexed spatially (R-tree-class); WorldIR gains a
   partitioning metadata section without breaking single-chunk worlds.
2. **Compilation memory bounds.** Stages process per-chunk with
   documented peak memory; meshing already scales by depth-limited
   Poisson — chunk first, then mesh per chunk, then stitch/re-weld
   boundary vertices.
3. **LOD.** Per-chunk mesh LOD chain (decimation, Open3D/quadric when
   available; honest fallback = single LOD labeled as such).
4. **Streaming.** Viewer/robot consumers fetch chunks by proximity;
   ArtifactStore's content addressing makes chunk caching natural.

## Rules

- Cross-chunk seams must weld or be explicitly marked (no cracks
  presented as geometry).
- World frame and chunk origin recorded (COORDINATE_FRAMES rules
  apply at every chunk boundary).
- No silent precision loss: metrics of scale/units unchanged.

## Failure modes

- Chunk-boundary features (a wall spanning chunks) → duplicate
  surfaces; weld or assign to one chunk deterministically.
- Index build cost; benchmark before optimizing (§34).

## Acceptance criteria

- Deterministic: two-room fixture compiled as 2 chunks → shared wall
  appears once after weld rule; global queries (bbox) return the
  union correctly.
- Memory: peak memory bounded by chunk size, not world size
  (measured, recorded in BENCHMARKS state).

## Priority

P5+ (after WorldStore; driven by the first site-scale capture).
