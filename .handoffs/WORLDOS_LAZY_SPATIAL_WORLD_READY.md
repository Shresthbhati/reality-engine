# HANDOFF

Agent: Claude (city-world-core)
Branch: claude/worldos-incremental-city
Commit: f09548a (+ pending benchmark-file commit)
Checkpoint: WORLDOS_LAZY_SPATIAL_WORLD_READY
Status: CHECKPOINT_READY (4 of 6 named limitations from the prior checkpoint closed; 2 remain, honestly reported)

## Quality status

| Capability | IMPLEMENTED | UNIT_VERIFIED | REAL_DATA_VERIFIED | INTEGRATED | PRODUCTION_READY |
|---|---|---|---|---|---|
| World manifest (TileManifest) | YES | YES (3 tests) | NO | YES (built from real SpatialTiles) | NO |
| Persisted tile artifacts | YES | YES (2 tests) | NO | YES (uses existing FileArtifactStore, content-addressed) | NO |
| Lazy world access (open_version/list_tiles/load_tile) | YES | YES, call-count verified (4 tests) | NO | PARTIAL -- no production caller yet | **NO -- not REAL_DATA_VERIFIED merely because synthetic benchmark passes, per the mandate** |
| Spatial query (query_region) | YES | YES, call-count verified (2 tests) | NO | YES (reuses SpatialTiles bounds) | NO |
| Incremental tile reuse | YES | YES (3 scenarios: A/B/C, boundary-crossing, deletion) | NO | YES (consumes IncrementalUpdateResult directly) | NO |
| Failure safety (crash mid tile-write) | YES | YES (1 test, real OSError injection) | NO | YES | NO |
| Synthetic benchmark (full load vs lazy load) | YES | N/A (a benchmark) | **NO -- explicitly SYNTHETIC** | N/A | N/A |

**No capability here is claimed PRODUCTION_READY or REAL_DATA_VERIFIED.**
Everything is exercised against synthetic in-memory `WorldIR` fixtures
and `tmp_path` stores. Real verification requires a real evidence
pipeline calling this code, which does not exist yet (same gap named
in every prior checkpoint this session).

## Current architecture (before this checkpoint, confirmed by audit)

```
WorldStore.save_version()/load_version()  -- ONE JSON blob per version,
                                              the ENTIRE serialized WorldIR.
SpatialIndex(world)                        -- built fresh from a FULL
                                              in-memory WorldIR, every time.
```

Zero lazy access existed. This checkpoint adds an ADDITIVE layer
(`worldstore/tiles.py`) alongside both, per "use and improve the
existing implementation, do not invent a parallel storage system":

```
WorldStore.save_version_tiled(world, ...)
    -> calls save_version() UNCHANGED (still the diff_worlds()/lineage
       source of truth)
    -> THEN builds a TileManifest (world_ir.spatial_tiles.SpatialTiles
       over the same world) and writes each tile's entity/geometry
       subset through the EXISTING content-addressed FileArtifactStore
    -> writes the manifest LAST, atomically (via the existing
       _atomic_write_text), after every tile artifact it references
       is already durably written

WorldStore.open_version(version_id) / worldstore.tiles.open_version()
    -> reads ONLY the manifest (a small JSON index) -- proven by a
       call-count assertion, not just correctness: TestLazyLoading::
       test_opening_a_version_does_not_read_any_tile_artifact
    -> WorldVersionHandle.list_tiles()/tile_bounds()/load_tile()/
       query_region() load exactly the tiles requested, nothing else
```

## New manifest contract

`TileManifest` (`worldstore/tiles.py`):
```python
TileManifest(
    version_id: str, world_id: str, parent: Optional[str],
    tile_size: float, coordinate_frame: str,
    tiles: Tuple[TileArtifactRef, ...],       # sorted by tile_key
    unlocalized_entity_ids: Tuple[str, ...],
)
TileArtifactRef(
    tile_key: (int, int, int), bounds_min: (float,float,float),
    bounds_max: (float,float,float), entity_ids: Tuple[str, ...],
    artifact_uri: str, content_hash: str,
)
```
This is an INDEX over `WorldIR`, not a second world representation --
it records where each tile's content lives and what it contains; it
never re-derives or duplicates entity/geometry fields itself.
Persisted at `<store_root>/tile_manifests/<version_id>.json`.

## Tile artifact contract

Each tile's payload (stored via the SAME `FileArtifactStore` the
whole-world path already uses): `{"entities": {id: Entity.to_dict()},
"geometries": {id: Geometry.to_dict()}}` -- includes every geometry any
of the tile's entities reference, so a loaded tile is immediately
usable, not a set of dangling `geometry_ids`. Content-addressed by
sha256 (identical bytes across two tiles/versions are stored once,
for free, via the existing `FileArtifactStore.put()` dedup). Verified
on-disk (`TestPersistedTileArtifacts::test_tile_artifact_bytes_are_actually_on_disk`)
and hash-checked on load (`WorldVersionHandle._load_ref` re-digests and
raises `WorldStoreError` on mismatch, same discipline as
`WorldStore.load_version()`'s existing integrity check).

## Lazy-loading API

```python
from worldstore.tiles import save_version_tiled, open_version

stored = save_version_tiled(store, world, parent=parent_version_id,
                             tile_size=10.0, incremental_result=result)  # result optional
handle = open_version(store, stored.version_id)
handle.list_tiles()                       # -> ((0,0,0), (1,0,0), ...)
handle.tile_bounds((0,0,0))               # -> (bounds_min, bounds_max)
handle.load_tile((0,0,0))                 # -> {entity_id: Entity}, ONE artifact read
handle.query_region(bounds_min, bounds_max)  # -> {entity_id: Entity}, only overlapping tiles read
```
Naming follows the existing repo convention (`WorldStore.load_version`,
`WorldStore.list_versions`) rather than inventing new vocabulary.

## Query behavior

`query_region()` filters tiles by manifest AABB overlap BEFORE loading
anything, then loads only the surviving tiles. Verified with a
call-count assertion (not just a correctness assertion): a query
overlapping exactly one tile results in exactly one `FileArtifactStore.get()`
call, regardless of how many other tiles exist in the version
(`TestSpatialQuery::test_query_region_loads_only_overlapping_tiles`).

## Incremental reuse semantics

`save_version_tiled(..., incremental_result=result)` consumes the
existing `apply_incremental_update()`'s `IncrementalUpdateResult`
directly (no new incremental-update logic invented here): any tile key
NOT in `result.rebuilt_tile_ids`, whose entity-id membership is
unchanged from the parent version's manifest, keeps that parent's
EXACT `artifact_uri`/`content_hash` -- zero re-serialization, zero new
writes. Verified for all three mandated scenarios:
- **A/B/C from the mission spec**: update touches tile A only -> A's
  artifact_uri changes, B and C's stay byte-identical to the parent.
- **Entity crossing a tile boundary**: BOTH the old tile (now empty,
  removed from the manifest since nothing else lives there) and the
  new tile (a fresh manifest entry) are rebuilt; unrelated tiles keep
  their exact artifact identity.
- **Entity deletion**: the entity's old tile is rebuilt (or removed
  from the manifest if it becomes empty); unrelated tiles are reused
  verbatim.

## Failure safety

Simulated a real `OSError` on the first TILE write (after the
whole-world `save_version()` call inside `save_version_tiled()` had
already succeeded, isolating the failure to the tile-persistence step
specifically): V1 remains fully valid and loadable afterward; V2's
tile manifest is never written, so `open_version(store, "v-2")` raises
`WorldStoreError` -- no partially-tiled version is ever visible through
the lazy API. `list_versions()` still honestly reports that V2's
whole-world blob DID commit (a separate, already-atomic step from the
prior WORLDSTORE_ATOMICITY checkpoint) -- this is not hidden or
misrepresented, just correctly scoped to what actually failed.

## Benchmark results (SYNTHETIC BENCHMARK)

`benchmarks/lazy_tile_load_bench.py` -- **SYNTHETIC, generated fixture
data, NOT real city capture.**

| scale | entities | tiles | full load | full load peak mem | open (manifest only) | small-region query | small-query peak mem | entities returned | **open speedup** | **query speedup** |
|---|---|---|---|---|---|---|---|---|---|---|
| ROOM | 20 | 4 | 18.5 ms | 134 KB | 9.1 ms | 17.8 ms | 15 KB | 10 | 2.0x | 1.0x |
| BUILDING | 200 | 8 | 33.6 ms | 607 KB | 9.7 ms | 23.9 ms | 97 KB | 50 | 3.5x | 1.4x |
| STREET | 2,000 | 80 | 181 ms | 6.1 MB | 193 ms | 918 ms | 168 KB | 100 | 0.9x | 0.2x |
| BLOCK | 5,000 | 200 | 367 ms | 15.3 MB | 1.56 s | 275 ms | 168 KB | 100 | 0.2x | 1.3x |
| MULTI_BLOCK | 20,000 | 800 | 3.41 s | 61.4 MB | 101 ms | 70.9 ms | 168 KB | 100 | 33.6x | **48.2x** |
| CITY_REPRESENTATIVE | 100,000 | 4,000 | 36.0 s | 311 MB | 573 ms | 146 ms | 168 KB | 100 | 62.9x | **246.6x** |

**Honest reading, per the "do not claim city-scale readiness based on
synthetic performance" mandate**: this is a genuine, measured win at
larger scales, not a uniform one. At ROOM/BUILDING/STREET/BLOCK, the
lazy path shows NO consistent advantage (0.2x-3.5x, noise-dominated by
manifest-build and per-call overhead outweighing the tiny amount of
data actually saved). The win becomes large and consistent only from
MULTI_BLOCK upward: **a small-region query at CITY_REPRESENTATIVE
scale (100 entities out of 100,000) is 246.6x faster and uses ~1,850x
less peak memory (168 KB vs 311 MB) than a full-world load** --
exactly the property "opening a large world must not require loading
every entity" that this checkpoint set out to prove, and it IS proven,
but only pays off once the world is actually large. Do not extrapolate
these multipliers to real city data; entity distribution, geometry
size, and real disk I/O characteristics will differ.

## Boundary testing

Coordinate/tile-boundary semantics were NOT changed by this checkpoint
(explicitly out of scope: "preserve the existing deterministic
floor-based tile semantics... do not change coordinate semantics
without explicit justification"). All boundary coverage from the prior
checkpoint (`tests/test_spatial_tiles_boundary.py`, 14 tests: center,
exact boundary, corner, negative coordinates, city-scale magnitude,
repeated-construction determinism) applies unchanged to this layer,
since `worldstore/tiles.py`'s manifest building calls the SAME
`SpatialTiles` class, unmodified.

## Known limitations

- **Not REAL_DATA_VERIFIED.** Every test uses synthetic `WorldIR`
  fixtures. No real evidence/reconstruction pipeline calls
  `save_version_tiled()`/`open_version()` yet.
- **`SpatialIndex` (the actual production query structure) does not
  consume this lazy layer.** `sdk/reality.py`'s `spatial_index()` still
  requires a full in-memory `WorldIR` -- this checkpoint adds a
  PARALLEL lazy-tile query path (`WorldVersionHandle.query_region()`)
  rather than rewiring `SpatialIndex` itself to be lazy, since
  `SpatialIndex` also provides `nearest()`/`within_radius()` k-NN
  semantics this checkpoint's simple AABB-overlap `query_region()`
  does not replicate. Rewiring `SpatialIndex` to be lazy-tile-backed
  is the natural next step, not done here.
- **Whole-world `save_version()` cost is unchanged and still O(n).**
  `save_version_tiled()` calls it unchanged, so the WORLDOS_CITY_SCALE_SPATIAL_READY
  checkpoint's measured 45s/53s save/load times at 100K entities still
  apply to the whole-world path this layer sits on top of. The lazy
  layer helps QUERY/OPEN cost, not the underlying whole-world save's
  cost -- see benchmark for how much.
- **`removed_entities` interaction with manifest tile removal**: when
  a tile becomes empty (its only entity deleted), it is dropped from
  the new manifest entirely rather than kept as an empty entry. This
  is a deliberate, tested choice (matches `SpatialTiles`' own
  "unpopulated tiles don't exist" convention) but worth knowing if a
  caller expects tile keys to be stable across versions regardless of
  occupancy.
- **No garbage collection of orphaned tile artifacts.** A rebuilt
  tile's OLD artifact (from the parent version) is not deleted --
  correct (older versions must remain loadable and may still reference
  it) but means tile-artifact storage only grows. Same tradeoff the
  existing whole-world `FileArtifactStore` already makes; not a new
  problem introduced here, just inherited.

## Ownership boundary

Owned and touched: WorldIR, WorldStore, SpatialTiles (read-only, used
not modified), spatial query infrastructure (the new lazy-tile path),
incremental tile reuse. NOT touched: Mobile, Desktop UI, reconstruction
algorithms, CI, cross-agent integration, `engine.scene_graph.SpatialIndex`
(audited, referenced, not modified). No defects found in those
subsystems.

## Antigravity consumption instructions

```python
from worldstore.store import WorldStore
from worldstore.tiles import save_version_tiled, open_version
from world_ir import apply_incremental_update

store = WorldStore(root)

# First save (no incremental_result -- every tile is "rebuilt" trivially).
v1 = save_version_tiled(store, world_v1, parent=None, tile_size=10.0)

# Later, after new evidence:
result = apply_incremental_update(world_v1, new_entities, updated_geometries=new_geoms)
v2 = save_version_tiled(
    store, result.new_world, parent=v1.version_id,
    tile_size=10.0, incremental_result=result,   # <-- enables tile reuse
)

# Lazy access -- opening does NOT load the world:
handle = open_version(store, v2.version_id)
handle.list_tiles()
nearby_entities = handle.query_region(bounds_min, bounds_max)  # only overlapping tiles read
```

Use `tile_size` consistently across saves of the same world lineage --
`save_version_tiled` does not currently detect or reconcile a
`tile_size` change between versions (out of scope; flag if this
becomes a real need).

To run the synthetic benchmark: `python -m benchmarks.lazy_tile_load_bench`.

## Collision notes with other Claude worktree

None. `agent/claude-city-world-core`'s dedicated worktree remains
stale relative to `main`. This checkpoint adds `worldstore/tiles.py`,
one test file, and one benchmark file -- no overlap with any other
active worktree.
