# HANDOFF

Agent: Claude (city-world-core, parallel checkpoint)
Branch: claude/reality-engine-setup-e2a7f8 (pushed to origin)
Commit: 0bb2e0a (prior checkpoint: 1075f3f, CityGML exporter)
Checkpoint: WorldStore atomic-write / crash-recovery contract (P11-01)
Status: CHECKPOINT_READY

## What exists

`worldstore/store.py`'s `WorldStore.save_version()` now writes both the
per-version JSON record and the `sequence.json` ordering index via a
new `_atomic_write_text()` helper (temp file in the same directory,
`fsync`, then `os.replace()` — atomic on POSIX and Windows NTFS).
Previously both were written with plain `Path.write_text()`, so a
process crash mid-write could leave a truncated JSON file on disk;
every subsequent `_record()`/`list_versions()` call against that store
would then raise a raw `json.JSONDecodeError`, effectively wedging the
*entire* store's history behind one corrupted file.

`_record()` and the new `_read_sequence()` helper now catch
`json.JSONDecodeError` and re-raise as `WorldStoreError` naming the
specific version/index that's corrupted, distinguishable from the
existing "unknown version" `WorldStoreError` (missing file vs.
corrupted file are different failure modes; a caller retrying a typo'd
version id shouldn't be told the store is corrupted).

This closes the "crash-during-commit recovery test" item that was
listed as required-but-unsatisfied verification for P11-01 in
`.agent/TASKS.yaml`.

## Problem solved

WorldStore had no atomicity guarantee on its two JSON writes per save.
"No partially committed world state" (the multi-agent contract's
WorldStore incremental-update requirement) was violated: a crash could
corrupt store state in a way that wasn't just "this one write failed"
but "every future read of this store now throws an unhandled parser
exception."

## Interface/API changed

None — `WorldStore`, `StoredVersion`, `save_version()`,
`load_version()`, `list_versions()`, `parents()`, `ancestors()`,
`verify_version()` all keep their existing signatures and return
types. Purely an internal write-mechanism and error-type hardening.
`WorldStoreError` is still the only exception type callers need to
catch; it just now also covers "corrupted on-disk record" alongside
"unknown version" and "version already exists."

## Files changed

- `worldstore/store.py` — `_atomic_write_text()` added; `save_version()`,
  `_record()`, `list_versions()` updated to use it / raise explicit
  `WorldStoreError` on corruption; new `_read_sequence()` helper.
- `.agent/TASKS.yaml` — P11-01 verification/completion notes updated.

## Tests added

`tests/test_world_store_atomicity.py` (7 tests):
- no temp files left behind after a normal save
- version record + sequence index parse cleanly after a normal save
- corrupted version record -> `WorldStoreError` (not `JSONDecodeError`), from both `load_version()` and `list_versions()`
- corrupted sequence index -> `WorldStoreError`, from both `list_versions()` and a subsequent `save_version()`
- unknown-version vs corrupted-version remain distinguishable failure modes
- a failed save (bad `parent`) registers no orphan sequence entry and leaves no version file on disk

## Tests executed

- `tests/test_world_store.py` + `tests/test_world_store_atomicity.py`: 15 passed.
- Full worldstore/world_ir/spatial/frame_graph/diff-relevant suite (`pytest tests/ -k "world_store or worldstore or world_ir or spatial or frame_graph or diff"`, excluding 5 pre-existing-broken unrelated collection errors — see Known risks): **182 passed, 1 skipped, 0 failed.**

## Dependencies

None new. Uses only `os`/`pathlib`/`json`/`uuid` (all already imported
or stdlib).

## Known limitations

- Concurrent writers across *processes* to the same store root can
  still race on `sequence.json`'s read-modify-write (`_read_sequence()`
  -> append -> `_atomic_write_text()` is not itself a compare-and-swap).
  Two processes saving concurrently could each read the same sequence,
  append their own version, and one write clobbers the other's
  ordering entry -- the version *file itself* is never corrupted (still
  atomically written and independently loadable), but it could be
  missing from the ordering. `list_versions()` already tolerates a
  version file not present in the recorded order (appends it in sorted
  order), so no version is ever silently dropped from listing -- just
  the save-order ranking could be off for a version lost from the
  sequence this way. Not fixed here (would need a file lock or a
  different sequence representation, e.g. one file per version with a
  timestamp instead of a shared index) -- flagging for whoever owns
  concurrent-writer WorldStore access next.

## Known risks

- Five test files fail to *collect* on this branch due to an unrelated
  pre-existing `ModuleNotFoundError: No module named 'engine.physics'`
  (`tests/test_cross_session_registration.py`,
  `test_depth_consistency.py`, `test_landmark_registration.py`,
  `test_multi_session_alignment.py`,
  `test_orchestrator_consistency.py`, all importing
  `engine.physics.math3`). This predates this checkpoint and is
  unrelated to WorldStore -- spun off as a separate task
  (`task_c3d0314d`) rather than fixed here, since it's outside this
  checkpoint's scope and risks touching files another worktree may own.

## Next consumer

Antigravity (frontend/studio integration) and whoever next touches
WorldStore persistence (e.g. the P13/P14 city-scale streaming-compile
work, which will save/load many more versions and is more exposed to
this failure mode).

## Exact integration instructions

Nothing to wire up -- this is a drop-in internal hardening of the
existing `WorldStore` class. Any code already calling
`WorldStore(root).save_version(...)` / `.load_version(...)` /
`.list_versions()` gets the atomicity guarantee for free. The only
externally-visible change is that a corrupted on-disk record now
raises `WorldStoreError` (a `ValueError` subclass) instead of an
unhandled `json.JSONDecodeError` -- any caller that was already
catching `WorldStoreError` broadly is unaffected; a caller that was
specifically catching `json.JSONDecodeError` around WorldStore calls
(none found in this repo) would need to catch `WorldStoreError`
instead.

## Collision notes with other Claude worktree

None. `agent/claude-city-world-core`'s active worktree
(`reality-engine-execution-8d5023`, commit `a7a015c`) touches
`engine/scene_graph/spatial_index.py`, `spatial_tiling.py`,
`spatial_common.py`, and `world_ir/frame_graph.py` -- none of which this
checkpoint touches. Note: `a7a015c`'s spatial-index/frame-graph work
already appears to be present on `main` (and hence in this worktree)
under a separate, already-merged commit lineage -- `main` is 16 commits
ahead of `agent/claude-city-world-core` with zero unique commits in the
other direction, so that branch is stale relative to `main`, not
ahead. Whoever integrates should rebase `agent/claude-city-world-core`
onto `main` before merging rather than merging its stale snapshot.
