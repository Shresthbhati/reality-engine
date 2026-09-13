# cli

`reality` -- a headless command-line client of `sdk.reality`. Every
subcommand is a direct call into the SDK; the CLI adds no logic of its
own beyond argument parsing and file I/O.

```
python -m apps.cli.main ingest <photos-folder> -o package.json
python -m apps.cli.main reconstruct package.json -o world.json [--colmap-binary PATH] [--gpu] [--no-real-geometry]
python -m apps.cli.main validate world.json
python -m apps.cli.main diff before.json after.json
python -m apps.cli.main export world.json --format gltf|usda|blender -o out.file
python -m apps.cli.main physics world.json
python -m apps.cli.main query nearest world.json <x> <y> <z> [--k N]
python -m apps.cli.main query contents world.json <entity-id>
```

`query nearest`/`query contents` are thin CLI callers of
`sdk.reality.spatial_index()`/`sdk.reality.scene_graph()` -- real
spatial/relationship query engines over a compiled WorldIR, not stubs.

`ingest` and `reconstruct` are split because `reconstruction.ReconstructionResult`
has no stable serialization format of its own (by design -- see
`reconstruction/backend/interface.py`): `reconstruct` runs orchestration
and compilation as one step and writes the resulting WorldIR JSON.
`reconstruct` refuses (non-zero exit, full backend attempt log on
stderr) rather than fabricate a world when no backend can honestly
produce geometry -- e.g. no COLMAP install and no canned test data.

### Real geometry artifacts

`reconstruct` stores real plane point-cloud geometry by default, using a
`FileArtifactStore` (`world_ir/artifact_store.py`) rooted at
`<output>.artifacts/` -- e.g. `-o world.json` writes real geometry
artifacts to `world.json.artifacts/` alongside it. This is a real,
persistent on-disk store (content-addressed, sharded like Git's object
store), so it survives across separate CLI invocations/processes --
unlike an in-memory store, which would die with the `reconstruct`
process and be useless to a later `export` run.

Pass `--no-real-geometry` to skip this (no `.artifacts/` directory is
created) and reproduce the previous behavior exactly: a smaller/faster
world.json with no `Geometry.data_uri` set on any geometry.

`export --format gltf` automatically reconnects to `<world>.artifacts/`
if that directory exists next to the world file being exported, so a
`reconstruct` -> `export --format gltf` round-trip (even across two
separate CLI invocations) emits real per-entity meshes instead of the
placeholder cube. `usda`/`blender` exports don't consume the artifact
store yet, so this has no effect on them. If `<world>.artifacts/`
doesn't exist (e.g. the world was reconstructed with
`--no-real-geometry`, or predates this feature), gltf export falls back
to the placeholder-cube behavior exactly as before.

Tests: `tests/test_cli.py`.
