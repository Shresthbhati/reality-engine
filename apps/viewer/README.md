# Reality Studio Viewer (web)

Minimum inspection surface for compiled worlds (campaign Phase 21): a
dependency-light Three.js viewer that consumes **only real pipeline
artifacts** produced by `scripts/run_vertical_slice.py`:

| artifact        | content                                                    |
| --------------- | ---------------------------------------------------------- |
| `worldir.json`  | canonical WorldIR: entities, geometry bounds, provenance, confidence, observations |
| `points.ply`    | real reconstructed points (binary float32 xyz, meters)     |
| `cameras.json`  | registered camera poses (camera-to-world wxyz quaternions) |

WorldIR remains canonical; this viewer is derived state and never writes back.

## Use it

```bash
# 1. produce artifacts (see scripts/run_vertical_slice.py)
python scripts/run_vertical_slice.py datasets/room_capture

# 2. build ONE self-contained HTML (library + app + your data embedded)
python apps/viewer/build_viewer.py \
    --worldir datasets/room_capture/pipeline_out/worldir.json \
    --points  datasets/room_capture/pipeline_out/points.ply \
    --cameras datasets/room_capture/pipeline_out/cameras.json \
    --out     datasets/room_capture/pipeline_out/viewer.html

# 3. open viewer.html in any modern browser -- offline, no server
```

Without `--worldir/--points/--cameras` the shell embeds only the app +
library; the page then fetches artifacts from `./` (e.g. copy the built
HTML into `pipeline_out/`) or accepts drag-and-drop files.

## What it shows (and deliberately does not)

- **3D world**: point cloud (height-colored), plane entities as
  translucent boxes sized from their real bounds, registered camera
  frustums from the true poses.
- **Entity list**: filter by id/type, sorted by confidence.
- **Inspector = "where did this come from?"**: confidence bar,
  provenance tag (OBSERVED/INFERRED/RECONSTRUCTED/...), position,
  bounds extent, geometry confidence, and every persisted observation
  (e.g. `plane_promotion` with inlier count, extent, method).
- **World panel**: scale state + anchoring note, reconstruction
  backend/status, depth stage facts -- the pipeline's honest report.
- **Truth visualization**: low-confidence entities render fainter;
  oversized (extent > 4x the robust point-cloud extent) depth-noise
  planes are hidden by default behind the `oversized` toggle -- noise
  stays inspectable without burying the scene. Outlier points are
  never cropped away.

Not implemented (honest gaps): real mesh rendering (WorldIR does not
store vertex buffers yet), materials, measurements UI, timeline,
diff/branch views. When WorldIR gains real mesh artifacts, this viewer
should render them instead of bounds boxes.

## Layout

- `index.html` + `src/main.js` -- served mode (import map -> `vendor/`).
- `build_viewer.py` -- emits the single-file offline build (blob-URL
  import map; base64-embedded artifacts).
- `vendor/` -- vendored three.js r170 module + OrbitControls (MIT;
  pinned, no CDN/network at runtime).

Verification: built against the real fixture run (58 entities, 121k
points, 25 cameras) and inspected in-browser -- selection, inspector,
provenance, layer toggles, and framing all exercised (see PR #11).
