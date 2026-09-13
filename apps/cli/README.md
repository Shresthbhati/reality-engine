# cli

`reality` -- a headless command-line client of `sdk.reality`. Every
subcommand is a direct call into the SDK; the CLI adds no logic of its
own beyond argument parsing and file I/O.

```
python -m apps.cli.main ingest <photos-folder> -o package.json
python -m apps.cli.main reconstruct package.json -o world.json [--colmap-binary PATH] [--gpu]
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

Tests: `tests/test_cli.py`.
