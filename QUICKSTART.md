# Reality Engine -- Quickstart

Photos in, a 3D world out. Add more photos and the **same** world improves; every improvement is a new version you can
inspect, compare and export. You do not need to know how it works to use it (see
[`docs/DEVELOPER_ARCHITECTURE.md`](docs/DEVELOPER_ARCHITECTURE.md) if you want to).

## 1. Install

You need Python 3.10+, Node 20+ and [COLMAP](https://github.com/colmap/colmap/releases) on your `PATH` (the engine that
works out where each photo was taken; a CUDA build is faster, a CPU build works).

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"           # API, tests, IFC export (everything the app needs)
cd frontend && npm ci && cd ..
```

Optional, for depth and object recognition (large download; the app runs without it and says so when a stage is skipped):

```bash
pip install -e ".[perception]"    # torch, torchvision, segment-anything
```

Check what this machine can run:

```bash
reality verification --runtime    # colmap / ifcopenshell / torch: AVAILABLE or UNAVAILABLE
```

## 2. Start

Two terminals:

```bash
uvicorn apps.api.main:app --port 8100              # the engine (reconstruction runs inside it; no other service)
cd frontend && npm run build && npm start          # the Studio on http://localhost:3000
```

(`npm run dev` instead of build/start while developing.) `http://localhost:8100/api/health` answers `{"status":"ok"}`
when the engine is up. Data lives under `./data/` (database, uploaded photos, worlds); delete it to start empty.

## 3. Open the Studio and add photos

Open <http://localhost:3000>. Drop 5-10 photos of the same place, taken while walking around it (overlapping views),
onto the drop zone, or use **Add evidence**. That is the whole action: the first upload creates your World and starts
reconstruction by itself.

## 4. Watch it reconstruct

The strip at the bottom shows the state ("Building...", then "Partially complete" or "Ready"). When it finishes the model
appears in the 3D view. Orbit with the left mouse button, pan with the right, zoom with the wheel; **F** frames the model.
The panel tells you how many photos were placed, how confident the result is, and what would help most (for example "walk
around to the back").

## 5. Inspect the World

Open **Details** at the bottom. You see: which photos are placed and which are waiting (and why), what was observed vs
deduced vs unknown, whether the surface is sparse or dense, and how this version was built.

## 6. Add more photos

Drop more photos onto the **same** World. Nothing else to choose: the engine decides whether to extend the existing model
or rebuild it, and keeps the one that is better *as a world*. Photos that cannot be placed yet are kept and tried again as
more evidence arrives. If the new photos do not improve the world, your current version is kept and the panel says so.

## 7. See what changed, and look back

* **What changed in this version** counts what was preserved, refined, extended, partly reproduced, split, merged,
  regrouped, ambiguous, removed or new.
* The version buttons **V1 / V2 / V3** (next to the state) open earlier versions. Opening one never changes it; the panel
  shows which photos that version was built from, which were waiting, and which came later.

## 8. Export

Use **Export** and pick a format: glTF, USD (USDA), IFC, CityGML, CityJSON or a Blender script. Or from a terminal:

```bash
curl -X POST http://localhost:8100/api/worlds/<world-id>/export -H "Content-Type: application/json" -d '{"format":"gltf"}'
```

The response gives a `download_url`. Add `"version": "<version-id>"` to export an earlier version. An export lists any
entity it could not write and why; it never writes an empty file.

## If something goes wrong

* **"Could not finish"** -- your photos are kept and the previous model is unchanged; press **Try again**.
* **Photos not placed** -- they need more overlap with the others; the panel says which and what to shoot.
* **`reality verification --runtime` says `colmap` UNAVAILABLE** -- install COLMAP and put it on `PATH`, then restart the engine.

## Try the full journey without a camera

```bash
python scripts/fetch_south_building.py                 # downloads real sample photos once
python scripts/demo_journey.py --out demo_out --keep   # 6 -> +4 -> +10 photos = V1, V2, V3, then exports
```

It prints every measured number and writes `demo_out/demo_metrics.json`.
