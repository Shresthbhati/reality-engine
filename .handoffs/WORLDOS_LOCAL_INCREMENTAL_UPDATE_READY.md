# HANDOFF

Agent: Claude (city-world-core)
Branch: claude/worldos-incremental-city
Commit: 6b65c09 (supersedes prior 812c6c9 checkpoint on this same file -- adds rebuilt_tile_ids + full test matrix per the deep-core-execution audit)
Checkpoint: WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY
Status: CHECKPOINT_READY

## Quality status

- Implementation: **PASS** -- `apply_incremental_update()` exists, is exported, and its object-identity/reuse contract is verified by tests, not asserted by hand-waving.
- Unit verification: **PASS** -- 21 tests in `tests/test_world_ir_apply_incremental_update.py`, all passing, covering every scenario in the mandated test matrix (see Tests below) plus an explicit "a sabotaged full-rebuild implementation must fail this" identity check.
- Real-data verification: **N/A** -- this is a pure WorldIR-level primitive over synthetic fixtures; it has no real evidence/reconstruction pipeline wired to it yet (see Limitations -- "no re-derivation logic" / "zero production consumers" is exactly the gap this checkpoint closes at the WorldIR layer, not at the pipeline layer).
- Integration: **PARTIAL** -- integrates cleanly with the existing `WorldStore.save_version()`/`diff_worlds()` (cross-checked to agree independently), and with the existing `SpatialTiles` (no second tiling system). Does NOT yet integrate with any evidence/registration/reconstruction stage that would call it in production -- no caller exists yet outside tests. That remaining wiring is explicitly out of this checkpoint's ownership (reconstruction algorithms are not mine to build).
- Failure behavior: **PASS** -- `TestFailureLeavesV1Untouched` proves a failed `save_version()` call (bad parent) leaves V1 loadable and unaltered, registers no partial V2 entry. `apply_incremental_update()` itself never mutates `base_world` in place (`TestV1Immutability`), so even a failure inside the function itself (e.g. an exception mid-computation) cannot have corrupted the input.

**This capability is NOT claimed PRODUCTION_READY.** It is a verified,
tested WorldIR-level primitive with an honest, currently-unproven
performance story (see Performance) and zero production callers.
Whoever wires it into a real evidence-update pipeline must re-verify
against real reconstruction output, not just synthetic fixtures.

## What exists

`world_ir/incremental.py` gains `apply_incremental_update()` and
`IncrementalUpdateResult` -- the localized WorldIR update primitive
that was missing per the prior checkpoint's finding (`affected_closure()`
had zero production consumers). This closes that gap at the WorldIR
level; it does NOT wire a full evidence->registration->reconstruction
pipeline (out of scope, see Limitations).

## Exact algorithm

```
apply_incremental_update(base_world, updated_entities, *, updated_geometries=(), ...):
    changed_entity_ids   = {e.id for e in updated_entities}
    changed_geometry_ids = {g.id for g in updated_geometries}

    # A changed geometry marks its owning entity as affected even if
    # the entity's own fields weren't supplied as changed.
    geometry_owners = entities in base_world whose geometry_ids intersect changed_geometry_ids
    seed_ids = changed_entity_ids | geometry_owners

    affected_entity_ids = affected_closure(base_world, seed_ids)   # existing BFS, reused as-is

    new_entities   = dict(base_world.entities);   overwrite ids in updated_entities
    new_geometries = dict(base_world.geometries); overwrite ids in updated_geometries
    new_world = shallow_copy(base_world) with .entities/.geometries replaced, .version += 1

    invalidated_tile_ids = tiles (SpatialTiles over base_world UNION new_world)
                            containing any id in (changed_entity_ids | affected_entity_ids)

    return IncrementalUpdateResult(new_world, changed_entity_ids, changed_geometry_ids,
                                    affected_entity_ids, invalidated_tile_ids,
                                    reused_entity_ids, reused_geometry_ids)
```

## Affected-closure semantics

Unchanged from the existing `affected_closure()` (P4.21): a deterministic
BFS over `Relationship` edges of kinds PART_OF/CONTAINS/ATTACHED_TO/
SUPPORTS/RESTS_ON (both directions), starting from the directly-changed
entity ids plus any entity that owns a directly-changed geometry (new
in this checkpoint -- geometry changes now seed the closure, not just
entity changes). `affected_entity_ids` in the result is this closure
MINUS the directly-changed ids that already have new objects -- i.e.
"things that might need re-derivation but weren't themselves given new
data." This function has no perception/compiler logic to actually
re-derive them; a real compiler stage consumes this set as its own
work queue.

## Reuse semantics (the critical requirement)

`new_world.entities[id] is base_world.entities[id]` for every id in
`reused_entity_ids` (== every base_world entity id not directly
supplied as changed) -- same Python object, not a rebuilt copy that
happens to compare equal. Same for `new_world.geometries[id]` and
`reused_geometry_ids`. Verified with `is` assertions in every relevant
test, never `==`. This includes entities in `affected_entity_ids` that
were NOT directly supplied (e.g. a wall's containing room): they are
flagged as affected but their object is untouched, honestly reflecting
that this function cannot re-derive a room's fields without real
architectural-perception logic.

## Invalidation semantics

Reuses `world_ir.spatial_tiles.SpatialTiles` (no second tiling system,
per instructions) -- a uniform grid, same `tile_size` convention as
the rest of the codebase. Two distinct report sets, not one:

- `rebuilt_tile_ids`: tiles holding a DIRECTLY changed entity, in
  EITHER `base_world` (old position) or `new_world` (new position) --
  a true content rebuild. An entity that moved tiles marks both its
  old and new tile as rebuilt.
- `invalidated_tile_ids`: `rebuilt_tile_ids` PLUS tiles holding only an
  AFFECTED-but-unchanged entity (relationship-flagged for re-check;
  nothing there was actually replaced). Always a superset of
  `rebuilt_tile_ids` -- asserted directly in
  `TestRebuiltIsSubsetOfInvalidated`.

The exact "tile A/B/C, only B affected" scenario from the mission spec
is its own test: `TestTileExampleFromSpec` -- A and C appear in
NEITHER set and their entities keep their exact object identity; B is
in `rebuilt_tile_ids`. Entities never touched by the update keep their
tile membership unexamined -- proven by `TestUnrelatedTileReusable`.

## Tests

`tests/test_world_ir_apply_incremental_update.py` (21 tests, the full
mandated matrix):
1. single changed entity -> new object at that id (`TestSingleChangedEntity`)
2. multiple changed entities -> both land, everything else untouched (`TestMultipleChangedEntities`)
3. relationship-dependent entity (dependency closure) -> affected, NOT mutated (`TestRelationshipDependentEntity`)
4. geometry-only change -> owning entity affected, entity object untouched (`TestGeometryOnlyChange`)
5. unchanged entity reuse -> same object (`is`), in `reused_entity_ids` (`TestUnrelatedEntityIdentity`)
6. unchanged geometry reuse -> same object, in `reused_geometry_ids` (`TestUnrelatedEntityIdentity`)
7. unchanged tile reuse -> tile A/C from the spec example never appear in either tile-report set (`TestUnrelatedTileReusable`, `TestTileExampleFromSpec`)
8. deterministic invalidation -> identical input twice produces identical output; moved entity invalidates both old and new tile (`TestDeterministicInvalidation`)
9. provenance preservation -> unchanged entity's provenance intact via object identity; changed entity carries its own supplied provenance (`TestProvenancePreserved`)
10. uncertainty preservation -> unchanged geometry is the same object, so whatever uncertainty it carried survives untouched (`TestUncertaintyPreserved`)
11. coordinate-frame preservation -> `new_world.coordinate_frame == base_world.coordinate_frame` (`TestCoordinateFramePreserved`)
12. V1 immutability -> `base_world.entities` is still the SAME dict object, `base_world.entities["wall-1"]` is still the original object, `base_world.version` unchanged, after the call (`TestV1Immutability`)
13. failed update rollback -> forced `save_version()` failure (bad parent) leaves V1 loadable, unaltered, no partial V2 registered (`TestFailureLeavesV1Untouched`)
14. process restart -> save V1+V2, open a FRESH `WorldStore` instance over the same root, reload V2, confirm both changed and unrelated entity data survive (`TestProcessRestart`)
15. repeated update determinism -> same input applied twice produces identical `invalidated_tile_ids`/`affected_entity_ids` (`TestDeterministicInvalidation`)
16. **sabotage detection** -- `TestCannotSecretlyRebuildEverything`: asserts identity (not equality) on every untouched entity/geometry; a deep-copy-and-patch "full rebuild" implementation that is equal in every field would still fail this test, which is the whole point.
17. lineage cross-check -- `apply_incremental_update`'s own `changed_entity_ids` independently agrees with `WorldStore.save_version()`'s `diff_worlds()`-derived report (`TestVersionLineage`).

Targeted suite run (per instructions, not the full 1889+): `pytest
tests/ -k "world_ir or worldstore or world_store or spatial or
incremental or diff"` -> **201 passed, 1 skipped, 0 failed** (195 + 6 new tests from this extension pass; full log attached to the commit run, not fabricated).

## Performance measurements

Measured (`time.perf_counter`, not fabricated). Two separate
measurements:

**1. Update cost vs. world size** (single `apply_incremental_update`
call touching exactly 1 of N entities):

| world size (N entities) | wall-clock update time |
|---|---|
| 1,000 | 2.3 ms |
| 10,000 | 21.0 ms |
| 50,000 | 136.5 ms |

**2. Incremental update vs. full rebuild, same N, same single change**
(full rebuild = constructing an equivalent WorldIR from scratch;
affected/reused counts and rebuilt/invalidated tile counts included
per the mandate):

| N | affected entities | rebuilt tiles | invalidated tiles | reused entities | full rebuild | incremental | speedup |
|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 1 | 1 | 999 | 4.78 ms | 2.75 ms | 1.73x |
| 10,000 | 1 | 1 | 1 | 9,999 | 37.13 ms | 47.63 ms | 0.78x |
| 50,000 | 1 | 1 | 1 | 49,999 | 322.33 ms | 276.86 ms | 1.16x |

**Honest reading, per the "do not claim performance improvement
unless measured" mandate: NO consistent speedup is demonstrated.**
The 0.78x-1.73x spread is noise, not a trend -- object-identity/data
reuse is real and verified (`reused_entities` column, all confirmed
`is`-identical), but wall-clock cost is NOT proven faster than a full
rebuild at these scales, because `dict(base_world.entities)` and
building two fresh `SpatialTiles` instances are themselves O(n) scans
that dominate the runtime regardless of how few entities actually
changed. This is the single most important finding to carry forward:
**the correctness/data-reuse contract is proven; the performance
contract is not.**

## Known limitations

- **Not O(affected-only) in wall-clock time.** The dict-copy and
  `SpatialTiles` rebuild are O(n). True sub-linear incremental update
  at real city scale needs a persistent/incrementally-maintained
  spatial index (insert/remove single entities without rescanning),
  not `SpatialTiles(world)` built fresh each call. Flagging as the
  concrete next city-scale task.
- **`new_world` is a shallow copy of `base_world`** except for the two
  replaced dict fields (`entities`, `geometries`) -- every other field
  (`materials`, `surfaces`, `components`, `branches`, `temporal_state`,
  etc.) is the SAME object as `base_world`'s. Mutating one of those
  collections in place on `new_world` after this call would also
  mutate `base_world`'s. Documented in the function docstring; not a
  bug for the entity/geometry identity contract this checkpoint was
  asked to prove, but a caller needing full structural independence
  across those other fields should not rely on this function alone.
- **No re-derivation logic.** `affected_entity_ids` is a report, not
  an action -- this function does not call architectural-perception,
  registration, or any compiler stage to actually update an affected
  room/relationship's own fields. That wiring (a real compiler stage
  consuming this result and re-running the relevant perception step
  only on `affected_entity_ids`) is the next layer up, not built here.
- **Relationship removal/addition is not modeled.** If a caller
  supplies an updated entity whose `relationships` differ from the
  base version, `affected_closure` is computed against `base_world`'s
  graph (the OLD edges), not the new entity's edges -- correct for
  "what was affected by the old topology," but a caller changing
  topology itself should be aware the closure doesn't retroactively
  follow the new edges in the same call.

## Antigravity consumption instructions

`from world_ir import apply_incremental_update, IncrementalUpdateResult`.

Typical caller pattern (evidence pipeline stage or Studio backend):

```python
result = apply_incremental_update(current_world, new_or_updated_entities,
                                   updated_geometries=new_or_updated_geometries)
stored = world_store.save_version(
    result.new_world, parent=current_version_id,
    source_session_ids=[the_new_evidence_session_id],
)
# result.invalidated_tile_ids tells a viewport/renderer exactly which
# tiles to re-fetch/re-render; result.affected_entity_ids tells any
# downstream re-derivation stage what to re-check (not yet automated).
```

`StoredVersion.changed_entity_ids`/`changed_geometry_ids` (from
`save_version`'s existing `diff_worlds()` call) will independently
agree with `result.changed_entity_ids`/`changed_geometry_ids` -- if
they ever disagree, that is a real bug worth investigating (verified
to agree in `TestVersionLineage::test_worldstore_diff_agrees_with_reported_changed_ids`).

## Ownership boundary

Owned by this checkpoint: WorldIR, WorldStore, incremental
compilation, spatial invalidation. NOT owned/touched: Desktop, Mobile,
reconstruction algorithms, CI, final integration. No defects were
found in those subsystems during this checkpoint's work; if any are
found in a future WorldOS pass, they'll be documented and handed to
their owner rather than fixed here.

## Collision notes with other Claude worktree

None. `agent/claude-city-world-core`'s dedicated worktree
(`reality-engine-execution-8d5023`) remains stale relative to `main`
(unchanged since the prior checkpoint's audit); this checkpoint only
touches `world_ir/incremental.py`, `world_ir/__init__.py`, and its own
test file.
