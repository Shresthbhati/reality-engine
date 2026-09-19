# HANDOFF

Agent: Claude (city-world-core)
Branch: claude/worldos-incremental-city
Commit: 812c6c9
Checkpoint: WORLDOS_LOCAL_INCREMENTAL_UPDATE_READY
Status: CHECKPOINT_READY

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
the rest of the codebase. `invalidated_tile_ids` is the union of every
tile a changed-or-affected entity occupies in EITHER `base_world`
(old position) or `new_world` (new position); an entity that moved
tiles invalidates both, verified by
`TestDeterministicInvalidation::test_moved_entity_invalidates_both_old_and_new_tile`.
Entities never touched by the update keep their tile membership
unexamined -- proven by `TestUnrelatedTileReusable`.

## Tests

`tests/test_world_ir_apply_incremental_update.py` (15 tests, all the
required scenarios):
1. single changed entity -> new object at that id
2. relationship-dependent entity -> affected, NOT mutated (same object)
3. geometry-only change -> owning entity affected, entity object untouched
4. unrelated entity -> same object (`is`), in `reused_entity_ids`, NOT in `affected_entity_ids`
5. unrelated geometry -> same object, in `reused_geometry_ids`
6. unrelated tile -> not invalidated; touched tile -> invalidated
7. deterministic invalidation -> same input produces same output twice; moved entity invalidates both tiles
8. provenance preserved -> unchanged entity's provenance field identical (via object identity); changed entity carries its own supplied provenance
9. uncertainty preserved -> unchanged geometry is the same object, so whatever uncertainty/observations it carried survive untouched
10. V1 -> V2 lineage -> version increments, world id stable, WorldStore's independently-computed `diff_worlds()` result agrees exactly with `apply_incremental_update`'s own `changed_entity_ids` report (two independent computations cross-checked)
11. process restart -> save V1 and V2, open a FRESH `WorldStore` instance over the same root, reload V2, confirm both the changed and unrelated entity data survive
12. failure leaves V1 untouched -> force `save_version()` to fail (bad parent id), assert V1 still loads exactly as saved and no V2 entry was registered

Targeted suite run (per instructions, not the full 1889+): `pytest
tests/ -k "world_ir or worldstore or world_store or spatial or
incremental or diff"` -> **195 passed, 1 skipped, 0 failed**.

## Performance measurements

Measured (`time.perf_counter`, not fabricated), single `apply_incremental_update`
call touching exactly 1 of N entities:

| world size (N entities) | wall-clock update time |
|---|---|
| 1,000 | 2.3 ms |
| 10,000 | 21.0 ms |
| 50,000 | 136.5 ms |

Object-identity reuse verified at every scale (sampled 200 unrelated
entities, all `is`-identical pre/post-update).

**Honest reading of these numbers**: entity/geometry DATA is genuinely
reused (zero recomputation), but wall-clock cost still scales
~linearly with world size, because `dict(base_world.entities)` and
building two fresh `SpatialTiles` instances (`base_world` + `new_world`)
are both O(n) scans. This is NOT the same claim as "O(1) in world
size" -- see Limitations.

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

## Collision notes with other Claude worktree

None. `agent/claude-city-world-core`'s dedicated worktree remains
stale relative to `main` (unchanged since the prior checkpoint's
audit); this checkpoint only touches `world_ir/incremental.py`,
`world_ir/__init__.py`, and its own new test file.
