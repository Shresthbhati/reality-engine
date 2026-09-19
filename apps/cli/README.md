# cli

`reality` -- a headless command-line client of `sdk.reality`. Every
subcommand is a direct call into the SDK/engine; the CLI adds no logic of
its own beyond argument parsing and file I/O.

`pip install .` (or `pip install -e .`) installs `reality` as a real
console-script entry point (`[project.scripts]` in `pyproject.toml`,
pointing at `apps.cli.main:main`) -- run it directly, no `python -m`
needed, and no repo checkout on `PYTHONPATH` required once installed:

```
reality ingest <photos-folder> -o package.json
reality session create <id> -o <session-dir>
reality session add-source <session-dir> <path>
reality session list <session-dir>
reality session inspect <session-dir>
reality session inspect-source <session-dir> <source-id>
reality session export-package <session-dir> -o package.json
reality compile <dataset> [-o pipeline_out] [--no-depth] [--no-mesh]
reality reconstruct package.json -o world.json [--colmap-binary PATH] [--gpu] [--no-real-geometry]
reality validate world.json
reality diff before.json after.json
reality export world.json --format gltf|usda|blender -o out.file
reality register <source> <target> -o result.json --from-frame A --to-frame B [--anchors a.json]
reality query nearest world.json <x> <y> <z> [--k N]
reality query contents world.json <entity-id>
reality store save world.json --store <dir> [--parent V] [--version-id V]
reality store load --store <dir> --version V -o world.json
reality store list --store <dir>
reality store verify --store <dir> --version V
reality inspect world.json [--entity <id>]
reality viewer --worldir worldir.json --points points.ply --cameras cameras.json -o viewer.html
```

The `python -m apps.cli.main ...` form (equivalent, useful when running
from a checkout without installing) still works identically.

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

`export --format gltf|usda|blender` automatically reconnects to
`<world>.artifacts/` if that directory exists next to the world file
being exported, so a `reconstruct` -> `export` round-trip (even across
two separate CLI invocations, any of the three formats) emits real
per-entity geometry instead of the placeholder cube. If
`<world>.artifacts/` doesn't exist (e.g. the world was reconstructed
with `--no-real-geometry`, or predates this feature), every format
falls back to its placeholder-shape behavior exactly as before.

Tests: `tests/test_cli.py`, `tests/test_cli_vertical_slice.py` (integration),
`tests/test_vertical_slice_e2e.py` (end-to-end, fake backend).
