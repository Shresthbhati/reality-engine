# Reality Engine

Reality Engine turns captured evidence (overlapping photos of a place) into a versioned 3D world. Photos go through
COLMAP-based reconstruction into a canonical **WorldIR** with per-entity provenance and uncertainty; adding more photos
improves the *same* world as a new version you can diff, query and export (glTF / USD / Blender / IFC). A FastAPI
engine and a Next.js **Studio** sit on top of the same pipeline the `reality` CLI drives.

What is verified, and against what, is in [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) and
`reality verification --runtime`. Read those before relying on any capability.

## 60-second start: photos in, 3D world out

Needs Python 3.10+, Node 20+ (for the Studio) and [COLMAP](https://github.com/colmap/colmap/releases) on `PATH`.

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
reality verification --runtime                      # which optional tools this machine can run

# a dataset is a folder with manifest.json + images/ (see QUICKSTART.md for the layout)
reality compile path/to/dataset -o out/             # capture dataset -> pipeline -> WorldIR + artifacts + exports
reality validate out/worldir.json                  # check the compiled WorldIR
reality export out/worldir.json --format gltf -o out/world.gltf   # also usda, blender, cityjson, citygml
```

`reality compile` flags: `-o/--output` (default `<dataset>/pipeline_out`), `--colmap-binary`, `--no-depth`, `--no-mesh`.
Run `reality --help` for the rest (`ingest`, `session`, `reconstruct`, `diff`, `query`, `store`, `inspect`, `viewer`, ...).
The step-by-step, photos-only walk-through (no CLI needed) is [`QUICKSTART.md`](QUICKSTART.md).

## Launch the Studio

```bash
uvicorn apps.api.main:app --port 8100               # engine API; health: http://localhost:8100/api/health
cd frontend && npm ci && npm run build && npm start # Studio on http://localhost:3000  (npm run dev while developing)
```

On Windows, `launch_ui.bat` installs frontend dependencies if needed and opens the Studio on port 3000; it does not
start the API, so run `uvicorn` first.

## API token

Set `REALITY_API_TOKEN=<secret>` before starting the engine to require `Authorization: Bearer <secret>` on every
endpoint except `/api/health`. Give the Studio's server the same variable `REALITY_API_TOKEN` (the Next.js proxy routes
read it server-side and never ship it to the browser). `NEXT_PUBLIC_API_TOKEN` is an optional fallback that *is* visible
to the browser, so use it only for a local, trusted setup. With no token set, only loopback clients are served.
Single shared token, no users or roles; see `docs/KNOWN_LIMITATIONS.md`.

## Tests

```bash
pytest -m "not slow"                  # default suite; what CI runs on every push
pytest -m slow                        # long-running (real-model downloads, GPU); CI runs it nightly
cd frontend && npx tsc --noEmit && npm run lint && npm run test:contract
```

Markers are defined in `pyproject.toml` (`slow`, `physics`, ...).

## Project layout

| Path | What it is |
| --- | --- |
| `apps/api/` | FastAPI engine (`apps.api.main:app`), including `auth.py` |
| `apps/cli/` | The `reality` command-line interface |
| `apps/viewer/` | Self-contained offline viewer builder (`reality viewer`) |
| `frontend/` | Next.js Studio (UI plus `src/app/api/*` proxy routes) |
| `evidence/` | Sessions, datasets and session merging |
| `reconstruction/`, `registration/` | Calibration, SfM/MVS, meshing; cross-source alignment |
| `world_ir/`, `worldstore/` | Canonical WorldIR; versioned persistence |
| `provenance/`, `uncertainty/` | Evidence-first provenance; uncertainty propagation |
| `exporters/` | glTF, USD, Blender, CityJSON/CityGML, IFC bridge |
| `perception/`, `synthetic/`, `procedural/` | Optional depth/segmentation; synthetic fixtures; procedural generation |
| `sdk/` | Public Python SDK (`sdk/reality.py`) |
| `tests/` | Python test suite |
| `docs/` | Architecture, status, roadmap and per-subsystem specs |

`apps/capture/` and `apps/studio/` are placeholders ("Not yet implemented").

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) -- how the system is built
- [`QUICKSTART.md`](QUICKSTART.md) -- install, start, add photos, refine, export
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) -- what is and is not verified
- [`docs/implementation/ROADMAP.md`](docs/implementation/ROADMAP.md) -- prioritised pending work
