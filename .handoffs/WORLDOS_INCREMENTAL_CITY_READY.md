# HANDOFF

Agent: Claude (city-world-core)
Branch: claude/worldos-incremental-city (pushed to origin; fresh branch off origin/main since the prior claude/reality-engine-setup-e2a7f8 was already merged via PR #55)
Commit: e53ec07
Checkpoint: WORLDOS_INCREMENTAL_CITY_READY (partial -- P0 WorldStore concurrency only; see below)
Status: CHECKPOINT_READY (concurrency); the incremental-pipeline-wiring and spatial-invalidation items are audited and reported below but NOT implemented in this checkpoint

## Audit findings (code > tests > handoffs > docs, per instructions)

1. **`world_ir.incremental.affected_closure()` has zero consumers.** Grepped
   the whole repo outside tests/`__init__` exports: nothing in
   `apps/cli/main.py`, `sdk/reality.py`, or any compiler stage calls it.
   "Incremental compilation" is currently dead code from the pipeline's
   perspective -- every real `save_version()` call in the codebase is a
   full-world compile-and-save, not a localized update. This is the
   single biggest gap against the mission's stated lifecycle
   (`World V1 + new evidence -> ... -> localized update -> World V2`)
   and is NOT fixed by this checkpoint -- flagged as the next task.
2. **`agent/claude-city-world-core`'s dedicated worktree**
   (`reality-engine-execution-8d5023`, commit `a7a015c`) is stale
   relative to `main` (0 unique commits, main is ahead) -- its
   spatial-index/frame-graph work is already merged into `main`
   independently. Did not touch that worktree or branch.
3. **WorldStore concurrency (this checkpoint's fix).** See below.

## What exists (this checkpoint)

`worldstore/store.py`'s `save_version()` performs a read-modify-write
on `sequence.json` (read current order -> append new version id ->
atomic write) with zero synchronization. Reproduced the race directly:
20 concurrent threads calling `save_version()` against the same store
root produced **15/20 unhandled `PermissionError` (WinError 5)** from
`os.replace()` racing on the destination path -- not a benign
eventually-consistent race, an outright crash surfaced to the caller.
Runs that didn't crash risked silent lost updates (two writers both
reading the same `order` list, the second write clobbering the
first's appended id).

Fixed with `_SequenceLock`: an `os.mkdir()`-based mutex (directory
creation is atomic on both POSIX and Windows NTFS -- exactly one
concurrent caller succeeds, everyone else gets `FileExistsError`,
which is what makes it a real mutex). `save_version()` now holds this
lock for the entire sequence read-modify-write. Acquisition polls with
a bounded timeout (default 30s) and raises `WorldStoreError` explicitly
on timeout, naming the lock path -- a stuck lock from a crashed holder
is never silently stolen; it needs explicit operator cleanup.

## Public contracts

`WorldStore.save_version()`, `.load_version()`, `.list_versions()`,
`.parents()`, `.ancestors()`, `.verify_version()`: unchanged signatures
and return types. New: `worldstore.store._SequenceLock` (private,
context-manager, `__init__(root, timeout=30.0, poll_interval=0.02)`).
No new public API surface -- concurrency safety is transparent to
existing callers.

## Concurrency behavior (verified, not asserted)

- **In-process thread race** (`tests/test_world_store_concurrency.py::TestConcurrentThreadWriters`,
  25 threads): 0 errors, 0 lost writes, exact id set recovered.
- **Real multi-process race** (`TestConcurrentProcessWriters`, 8
  separate OS processes via `subprocess.Popen`, no shared GIL/memory --
  the actual "writer A + writer B" scenario from two independent
  CLI/pipeline invocations): 0 errors, 0 lost writes.
- **Lock semantics** (`TestSequenceLock`): mutual exclusion (second
  acquirer blocks until first releases), explicit timeout error (not a
  hang, not a silent steal), lock directory always cleaned up after use.
- Manually re-ran the original 20-thread reproduction pre/post-fix:
  pre-fix 15/20 `PermissionError`; post-fix (and at 30 threads) 0
  errors, 0 lost writes, no `.sequence.lock` leftover.

## Invalidation semantics

Not addressed in this checkpoint -- audited but not implemented (see
"Not done" below). `world_ir.spatial_tiles.py` /
`engine/scene_graph/spatial_tiling.py` provide tile/chunk query
infrastructure (`P13-01`, already `DONE` per `.agent/TASKS.yaml`) but
there is no invalidation hook wired to WorldStore versioning -- saving
a new version does not currently mark which spatial tiles the diff
touched. `StoredVersion.changed_entity_ids`/`changed_geometry_ids`
(already existing, from `world_ir.diff.diff_worlds`) is the raw
material for this; mapping those ids to tile ids via
`SpatialTiling`/`spatial_tiles.py` and exposing "which tiles this
version invalidated" is the concrete next step, not yet built.

## Incremental semantics

Not addressed in this checkpoint. `affected_closure()` (relationship-graph
BFS from changed entities) exists and is tested in isolation
(`tests/test_world_ir_incremental.py`) but has no caller that:
1. takes a base `WorldIR` + new/updated entities from fresh evidence,
2. computes the affected closure,
3. produces a new `WorldIR` where only the affected entities/geometries
   are new objects and everything else is the *same* object (proving
   "unchanged remains unchanged," not just byte-equal after a full
   rebuild),
4. saves it via `WorldStore.save_version()` with a report of exactly
   which entity/tile ids were touched.

This is the real "localized update" the mission describes and is the
single most valuable next task -- see Next consumer / instructions below.

## Tests

- `tests/test_world_store_concurrency.py` (6 tests, new).
- `tests/test_world_store.py` + `tests/test_world_store_atomicity.py`:
  20 total, all still pass (no regression from the lock).
- Broader targeted suite (`pytest tests/ -k "world_store or worldstore
  or world_ir or spatial or incremental or diff"`): 177 passed, 3
  failed, 1 skipped. **The 3 failures are pre-existing**
  (`tests/test_session_store.py::TestIncrementalMultiSource::*`,
  `EvidencePackage.merge_from` missing -- unrelated to WorldStore,
  already tracked separately, not touched by this checkpoint's files).

## Benchmarks

None added this checkpoint -- no city-scale benchmark work done (P1
in the instructions, correctly deferred behind the P0 items).

## Known limitations

- `_SequenceLock` only guards `sequence.json`'s read-modify-write, not
  the whole `save_version()` call. This is deliberate: the version
  record write (`_atomic_write_text(path, ...)`) is independently safe
  per-version (each version id gets its own file, and the `path.exists()`
  immutability check + atomic write mean two writers with *different*
  version ids never collide there; two writers with the *same* explicit
  version id would race on that check, but that's a caller error
  (reusing an id), not a concurrency bug in the store).
- Lock timeout default (30s) is a guess, not benchmarked against
  realistic save latency at city scale (large `WorldIR.to_dict()`
  serialization could plausibly take longer than 30s for very large
  worlds) -- worth revisiting once real city-scale save timings exist.
- No fairness guarantee: `os.mkdir()` spin-poll doesn't guarantee FIFO
  ordering among waiters, only mutual exclusion. Not a problem for
  correctness, could matter for latency under heavy contention.

## Next consumer

Antigravity (frontend/studio, already consumes `WorldStore` per
`.handoffs/DESKTOP_BACKEND_INTEGRATED.md`) benefits immediately and
transparently -- no integration change needed, existing
`WorldStore(root).save_version(...)` calls are now safe under
concurrent writers (e.g. multiple capture sessions saving to the same
project store).

## Exact integration instructions

Nothing to wire up for this checkpoint -- drop-in fix, same public API.

For whoever picks up the **incremental pipeline wiring** (the
higher-value item this checkpoint didn't get to): start from
`world_ir/incremental.py::affected_closure()` and
`worldstore/store.py::save_version()`'s existing `diff_worlds()` call
(it already computes `changed_entity_ids`/`changed_geometry_ids` on
every save -- that's the signal to feed `affected_closure()` on the
*next* save). The missing piece is a function like
`apply_incremental_update(base_world, new_entities, ...) -> WorldIR`
that mutates only the affected closure and leaves every other entity
as the same object reference, plus a test that asserts object identity
(`is`, not `==`) on untouched entities to prove no whole-world rebuild
happened.

## Collision notes with other Claude worktree

None. `agent/claude-city-world-core`'s worktree
(`reality-engine-execution-8d5023`) is stale relative to `main` (see
audit findings above) and untouched by this checkpoint.
