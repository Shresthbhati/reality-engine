# HANDOFF

Agent: Claude (city-world-core)
Branch: claude/worldos-incremental-city
Commit: 4779ddc (+ pending benchmark-file commit)
Checkpoint: WORLDOS_CITY_SCALE_SPATIAL_READY
Status: CHECKPOINT_READY (P0 items closed; P1 items partially closed, honestly reported)

## Quality status (per capability, not blanket)

| Capability | IMPLEMENTED | UNIT_VERIFIED | REAL_DATA_VERIFIED | INTEGRATED | PRODUCTION_READY |
|---|---|---|---|---|---|
| Tile membership (boundary semantics) | YES (pre-existing, now pinned down) | YES (14 tests) | N/A (pure math, no data dependency) | YES | YES -- deterministic floor-based hashing, no known defect |
| Incremental update: add/update entity | YES | YES | NO (synthetic fixtures only) | PARTIAL (WorldStore integration proven; no production caller) | NO |
| Incremental update: delete entity | YES (new this checkpoint) | YES (5 tests) | NO | PARTIAL | NO |
| Spatial invalidation (rebuilt vs invalidated tiles) | YES | YES | NO | YES (reuses existing SpatialTiles) | NO -- see performance limitation below |
| Tile artifact integrity | **NOT APPLICABLE** | N/A | N/A | N/A | **NO -- no per-tile artifacts exist in WorldStore at all (see Architecture Audited)** |
| Lazy/partial world loading | **NOT IMPLEMENTED** | N/A | N/A | N/A | **NO -- WorldStore.load_version() and SpatialIndex both require the full world in memory** |
| Query correctness after update | YES | YES (4 tests) | NO | YES (uses real production SpatialIndex) | PARTIAL -- correct, but requires a full SpatialIndex rebuild per update, not incremental |
| City-scale benchmark | YES | N/A (a benchmark, not a test) | **NO -- SYNTHETIC only** | N/A | N/A |
| Coordinate-frame preservation | YES | YES | N/A | YES | YES |
| Determinism (repeated update) | YES | YES | N/A | YES | YES |

**No capability in this checkpoint is claimed PRODUCTION_READY on the
strength of unit tests alone.** The two capabilities marked NOT
APPLICABLE / NOT IMPLEMENTED are real, load-bearing gaps, not merely
unverified -- see Architecture Audited and Known Limitations. The
benchmark below makes the performance gap concrete with real numbers,
not hand-waving.

## Architecture audited

Followed the actual production path (code and tests, not documentation
claims):

```
WorldIR (world_ir/world_v1.py)
  |
  v
SpatialTiles (world_ir/spatial_tiles.py) -- used ONLY by
  world_ir/incremental.py (this checkpoint's own module) and its tests.
  |
  v  [SEPARATE structure -- not the same index]
SpatialIndex (engine/scene_graph/spatial_index.py) -- the REAL
  production query path, reached via sdk/reality.py's spatial_index().
  Built ONCE from a full WorldIR snapshot ("Does not observe later
  mutations to `world`" -- its own docstring). No incremental
  maintenance path exists.
  |
  v
WorldStore (worldstore/store.py) -- ONE JSON blob per version, the
  ENTIRE serialized WorldIR. Zero "tile" references anywhere in this
  file (grepped, confirmed). No per-tile persistence, no tile
  manifests, no per-tile artifact addressing.
```

Key findings, each verified by grep/read of actual code, not assumed:

1. **`world_ir.spatial_tiles.SpatialTiles` and `engine.scene_graph.
   spatial_index.SpatialIndex` are two independent spatial structures**,
   not one shared system. `apply_incremental_update()` (this
   checkpoint's own prior work) uses `SpatialTiles` for its invalidation
   report; the SDK's actual `spatial_index()` query path uses
   `SpatialIndex`, which has no awareness of `SpatialTiles` at all.
   Neither is a second NEW system introduced here -- both already
   existed; this audit is the first documented cross-reference between
   them.
2. **`SpatialIndex` requires the whole `WorldIR` in memory and rebuilds
   from scratch on every construction.** There is no lazy/partial
   query path anywhere in the codebase today.
3. **`WorldStore` persists exactly one artifact per version: the
   entire serialized world.** There is no tile-level granularity to
   corrupt, orphan, or verify independently -- "tile artifact
   integrity" as a distinct concern does not exist in the current
   architecture. The existing whole-world atomicity/corruption
   guarantees (from the prior WORLDSTORE_ATOMICITY checkpoint) are the
   only integrity guarantee that applies.

## Tile semantics (boundary behavior, pinned down and tested)

`SpatialTiles._key_of(position) = (floor(x/tile_size), floor(y/tile_size), floor(z/tile_size))`.
Each tile is a HALF-OPEN interval per axis: `[i*tile_size, (i+1)*tile_size)`.
A position exactly at a tile's lower boundary belongs to THAT tile; a
position at the upper boundary belongs to the NEXT tile. Floor (not
truncation) is used, so negative coordinates bucket correctly and
symmetrically with positive ones (`-0.1` -> tile `-1`, not tile `0`).
Verified deterministic under repeated construction from identical
inputs, at city-scale-magnitude coordinates (500,000+ units), and for
geometry-centroid-derived positions (no transform) as well as explicit
transform positions. `tests/test_spatial_tiles_boundary.py` (14 tests)
is the executable specification of this; nothing here was invented --
it is the EXISTING `_key_of`/`_resolve_position`/`_geometry_centroid`
logic, tested and documented, not changed.

## Invalidation semantics (extended this checkpoint)

`apply_incremental_update()` now supports deletion (`removed_entities`
parameter) alongside the existing add/update path:
- A deleted entity's OLD tile is reported in `rebuilt_tile_ids` (real
  content removal, not just a relationship flag).
- Entities the deleted one was `PART_OF`/`SUPPORTS`/etc. are added to
  `affected_entity_ids` (their relationship now dangles) but are NOT
  themselves mutated -- same "report, don't fabricate re-derivation"
  contract as the add/update path.
- Raises `ValueError` if an id appears in both `updated_entities` and
  `removed_entities` in the same call (ambiguous intent, refused rather
  than guessed).
- An id in `removed_entities` that doesn't exist in `base_world` is
  silently ignored (nothing to delete), matching `affected_closure`'s
  existing "unknown ids are dropped" convention.

Geometry expansion crossing a tile boundary: tested explicitly
(`TestGeometryExpansionCrossesTileBoundary`) -- the owning entity is
marked `affected` (its geometry changed under it) without being
mutated itself, since re-deriving the entity's own fields from new
geometry bounds is domain logic this function does not have.

## Boundary behavior (P0 test matrix)

All in `tests/test_spatial_tiles_boundary.py` and
`tests/test_world_ir_apply_incremental_update.py`:
- tile center, exact lower boundary, just-below boundary, exact zero
- corner of a tile (all three axes near their upper boundary)
- negative coordinates (floor semantics, not truncation)
- negative exact boundary
- very large (city-UTM-scale) coordinates, precision preserved
- repeated construction from identical data -> identical tile
  assignment (the explicit "no float-noise flicker" requirement)
- a mathematically-equal-but-differently-computed position lands in
  the same tile as the stored one

**Not covered in this checkpoint**: multiple distinct coordinate
frames feeding the SAME `SpatialTiles`/`SpatialIndex` simultaneously.
`WorldIR.coordinate_frame` is a single world-level field; per-entity
frame mixing would require `world_ir/frame_graph.py` (a separate,
pre-existing module this checkpoint did not modify or extend) to
resolve entities into one common frame BEFORE tiling. Flagging as a
real, not-yet-tested gap rather than claiming coverage that doesn't
exist.

## Artifact integrity

**Not applicable to the current architecture, and this is a genuine
finding, not a dodge.** WorldStore has no tile-level artifacts to
corrupt, orphan, or verify -- confirmed via `grep -n "tile"
worldstore/store.py` returning zero matches. The mandated scenarios
(orphaned artifacts, stale tile references, missing tile manifests,
incorrect hashes at tile granularity) cannot occur because there are
no per-tile artifacts. The one integrity guarantee that DOES apply
(whole-world atomic write, corrupted-record detection, failed-save
leaves the prior version untouched) was already verified in the prior
WORLDSTORE_ATOMICITY checkpoint and is re-confirmed here by
`TestFailureLeavesV1Untouched`/`TestV1Immutability` in
`tests/test_world_ir_apply_incremental_update.py`. If per-tile
artifact storage becomes a real requirement, that is a WorldStore
architecture change (out of "use and improve the existing
implementation" scope) and should be scoped as its own task.

## Lazy world access / query behavior

**Not implemented.** Audited, not fabricated: `WorldStore.load_version()`
deserializes the entire world JSON blob every time; `SpatialIndex`
(the real production query structure) requires the full `WorldIR`
already in memory and indexes every entity on construction. There is
no code path anywhere that loads "only the tiles a query touches."
`tests/test_query_after_incremental_update.py` proves query
CORRECTNESS after an update (a fresh `SpatialIndex` over `new_world`
gives the right answer), but every one of those tests still builds
the index over the WHOLE world -- it does not exercise or claim any
lazy-loading capability. Memory/time cost of full loads at scale is
measured honestly in the benchmark below instead.

## Benchmark results (SYNTHETIC BENCHMARK)

`benchmarks/city_scale_spatial_bench.py` -- **SYNTHETIC, generated
fixture data, NOT real city capture.** Do not cite as real-world
performance. A single localized change (moving one entity within its
own tile) is applied at every scale; `tiles_invalidated`/`tiles_rebuilt`
staying at 1 regardless of world size (while `tiles_reused` grows) IS
the correctness proof -- localization works. The TIME columns are
where the honest bad news is.

| scale | entities | tiles | store size | save | load | index build | query | **incremental update** | incremental save |
|---|---|---|---|---|---|---|---|---|---|
| ROOM | 20 | 4 | 9.7 KB | 8.4 ms | 18.6 ms | 0.3 ms | 0.12 ms | 0.3 ms | 8.6 ms |
| BUILDING | 200 | 8 | 88 KB | 71.8 ms | 57.0 ms | 4.0 ms | 1.04 ms | 2.7 ms | 92.4 ms |
| STREET | 2,000 | 80 | 874 KB | 204 ms | 191 ms | 32 ms | 1.73 ms | 23.4 ms | 535 ms |
| BLOCK | 5,000 | 200 | 2.19 MB | 3.41 s | 4.52 s | 727 ms | 20.6 ms | 801 ms | 6.28 s |
| MULTI_BLOCK | 20,000 | 800 | 8.79 MB | 20.1 s | 11.2 s | 1.36 s | 27.0 ms | 1.27 s | 17.1 s |
| CITY_REPRESENTATIVE | 100,000 | 4,000 | 44.1 MB | 45.0 s | 52.9 s | 16.6 s | 156 ms | 10.7 s | 137.4 s |

(All times `time.perf_counter`, single run per scale, tracemalloc active
during measurement -- tracemalloc itself adds overhead, so these are
upper bounds, not best-case numbers; still real, not estimated.)

Tile counts confirm proportional growth (4 -> 4,000 tiles, matching the
20x->100,000x entity growth at roughly constant density by design).
`tiles_invalidated`/`tiles_rebuilt` = 1 at every single scale --
**localization is real: a one-entity change never touches more than
its own tile, at any world size tested.**

**Honest reading of the time columns, per the "do not claim
performance improvement unless measured" mandate**: incremental
update time itself grows from 0.3ms (ROOM) to **10.7 SECONDS**
(CITY_REPRESENTATIVE) for touching exactly ONE entity out of 100,000 --
confirming, with real numbers this time (not just the prior
checkpoint's smaller-scale estimate), that `apply_incremental_update`'s
wall-clock cost is dominated by its O(n) steps (the `dict()` copy and
the two full `SpatialTiles` rebuilds), NOT by the size of the actual
change. **This is the single most important, concrete, numeric finding
of this checkpoint: the correctness of localization is proven; the
performance of localization is proven NOT to hold at city scale in the
current implementation.** Save/load times show the same pattern
(WorldStore's whole-world-blob design) -- 45s to save a 100K-entity
world regardless of what changed.

## Determinism results

Verified directly (not merely asserted): the same `apply_incremental_update()`
call, given byte-identical inputs, run twice, produces:
- identical `changed_entity_ids`
- identical `invalidated_tile_ids`
- identical `rebuilt_tile_ids`
- identical `new_world.version`
- identical `new_world.coordinate_frame`

Also verified: `SpatialTiles` tile assignment is stable under 20
repeated constructions from the same entity data (no float-noise
flicker), and `TestDeterministicInvalidation` in the incremental-update
test suite covers the same property at the `apply_incremental_update`
level.

## Coordinate-frame tests

`TestCoordinateFramePreserved` confirms `new_world.coordinate_frame ==
base_world.coordinate_frame` after an incremental update (trivially
true via the shallow-copy design, but pinned down explicitly rather
than left implicit). Full multi-frame-graph interaction (local/session/
site/geographic/city frames simultaneously) is NOT tested this
checkpoint -- see Known Limitations.

## Query correctness after updates

`tests/test_query_after_incremental_update.py` (4 tests), using the
REAL production `engine.scene_graph.spatial_index.SpatialIndex`, not a
test double:
- moved entity is found at its new position in a V2 index, and the V1
  index (built before the update) still reports the OLD position --
  proving the two are genuinely independent snapshots, not a shared
  mutable structure that could leak V2 state backward
- deleted entity is unreachable in the V2 index
- newly added entity is reachable in the V2 index
- an unrelated entity's query result is byte- and object-identical
  before and after an unrelated update

## Known limitations

- **No incremental spatial-index maintenance, and it costs real time.**
  Every query after an update requires a full `SpatialIndex(new_world)`
  rebuild -- 16.6 seconds at 100,000 entities per the benchmark above.
  Query correctness is solved; query COST at real city scale is not.
- **No per-tile WorldStore persistence.** Save/load cost scales with
  total world size (45s/53s at 100K entities) regardless of how
  localized a change is.
- **`apply_incremental_update` itself is O(world size), measured**:
  10.7s at 100K entities for a single-entity change. Confirms and
  quantifies (with real numbers, not estimates) the prior
  WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY checkpoint's documented
  limitation. The concrete next task: replace the `dict(base_world.entities)`
  full copy and the two from-scratch `SpatialTiles` builds with a
  persistent/incrementally-maintained structure that only touches the
  changed/affected ids.
- **Multi-coordinate-frame tiling is untested.** `SpatialTiles`/
  `SpatialIndex` both assume all entities are already in one common
  frame; interaction with `world_ir/frame_graph.py`'s multi-frame
  graph was not exercised this checkpoint.
- **Benchmark is synthetic.** Entity distribution (uniform grid across
  generated 80m blocks) is a simplification; real capture data is
  denser/sparser unevenly and has real geometry (not point transforms
  only). Numbers characterize code scaling behavior, not real-city
  throughput. tracemalloc overhead inflates the peak-memory numbers
  somewhat versus an unprofiled run.

## Ownership boundary

Owned and touched: WorldIR, WorldStore, spatial indexing/tiling,
incremental world updates, coordinate-frame preservation (not the
frame graph module itself), world query infrastructure directly
coupled to WorldOS. NOT touched: Mobile, Desktop UI, reconstruction
algorithms, CI, integration orchestration. No defects were found in
those subsystems during this checkpoint.

## Antigravity consumption instructions

No change to the existing `apply_incremental_update()` call pattern
except the new optional `removed_entities` parameter:

```python
result = apply_incremental_update(
    current_world, new_or_updated_entities,
    updated_geometries=new_or_updated_geometries,
    removed_entities=deleted_entity_ids,   # new this checkpoint
)
stored = world_store.save_version(result.new_world, parent=current_version_id)

# Query correctness after the update requires a FULL rebuild -- there
# is no incremental index maintenance, and per the benchmark above
# this is NOT cheap at city scale:
from engine.scene_graph.spatial_index import SpatialIndex
fresh_index = SpatialIndex(result.new_world)
```

`result.rebuilt_tile_ids` (genuinely rebuilt content) vs
`result.invalidated_tile_ids` (rebuilt + relationship-flagged) tells a
viewport exactly which tiles need re-render vs. re-check.
`result.removed_entity_ids` tells a renderer which entity handles must
be dropped. **Do not call this pipeline "fast" in a UI/product
context at city scale** -- the benchmark shows single-entity updates
taking 10+ seconds at 100K entities; surface a progress/async state,
don't assume sub-second responsiveness.

To run the synthetic benchmark: `python -m benchmarks.city_scale_spatial_bench`.

## Collision notes with other Claude worktree

None. `agent/claude-city-world-core`'s dedicated worktree remains
stale relative to `main`. This checkpoint touches `world_ir/incremental.py`,
adds three new test files, and adds one new benchmark file -- no
overlap with any other active worktree's files.
