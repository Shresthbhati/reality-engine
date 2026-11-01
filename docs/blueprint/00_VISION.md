# 00 — Vision

## One sentence

**Reality Engine turns photographs (and later video and sensor data) of a real
place into one accurate, measurable, editable and persistent 3D world that knows
what it saw, what it guessed, and what it still needs.**

## What it is not

- Not a photogrammetry viewer that shows a pretty point cloud.
- Not a room planner that drops catalogue furniture into a box.
- Not an object detector that labels pictures.
- Not a generator of plausible-looking scenes. A realistic scene with invented
  geometry, wrong dimensions or duplicated furniture is a **failure**.

## Who uses it

| User | Wants |
|------|-------|
| Homeowner / renter | Photograph a room, get a 3D model, measure it, export it. |
| Designer / architect | Accurate existing-conditions model to design on top of. |
| Property / facilities | Persistent record of a building that improves over visits. |
| Developer / agent (later) | Query a world: "where is the nearest socket?", "how wide is this corridor?" |

## The user journey (the product in 8 steps)

1. **Capture** — upload photos now; add more any time, across sessions.
2. **Understand** — the system finds objects, surfaces, openings, image quality.
3. **Relate** — it works out which photos show the same room and which
   detections are the same physical object.
4. **Reconstruct** — cameras, geometry, scale (when there is a scale source),
   surfaces, objects, relationships.
5. **Validate** — it checks consistency, flags conflicts and missing regions.
6. **Build the world** — one persistent world: architecture, objects, materials,
   dimensions, evidence, confidence.
7. **Improve** — it asks for specific photos ("photograph the back of the sofa
   from the window side"), registers them, and updates only what changed.
8. **Use** — navigate, inspect, measure, correct, save, version, export.

## The three views of the same world

1. **Real** — image-derived appearance. The primary view.
2. **Wireframe** — the actual geometry: edges, boundaries; duplicates are obvious.
3. **Evidence** — observed / partial / occluded / inferred / generated /
   conflicted / unknown, as an overlay. Never replaces real materials.

These are display modes of one world, not three reconstructions.

## Scale

Room → large room → apartment → house → building → property → street.
Same world model, same identity system, same evidence rules at every scale.
The difference is only hierarchy depth, entity count and compute budget.
**v1 targets a single room up to a whole apartment.** Buildings, outdoor and
streets come later on the same contracts.

## The proof: v1 acceptance demo (living room)

The product is "real" when this runs end to end in the browser, with no manual steps:

1. Create a world "Living Room".
2. Upload the living-room photo set (8–30 photos).
3. System reports they show **one** room and recovers camera positions.
4. It builds one coherent room: walls, floor, ceiling, window and door openings.
5. It reconstructs the sofa, coffee table, rug, media console, TV, window and
   main plants — **each exactly once**, matched across all photos.
6. Real, Wireframe and Evidence views all work on that one world.
7. Clicking the sofa shows its class, dimensions (or "relative scale"),
   confidence vector, and the source photos that support it.
8. The system issues at least one **capture request** for a poorly observed region.
9. Uploading that photo registers it into the same world and updates only the
   affected region; a new version appears in history.
10. Close the app, reopen; the world is exactly as left.
11. Export GLB; it opens in an external viewer (e.g. Blender) with correct
    transforms, hierarchy and textures.
12. Rename, duplicate, archive, restore and delete (to trash) the world all work.

> The living-room set has no ground-truth scale. It tests identity, coherence and
> workflow. **Dimensional accuracy is tested separately** on tape-measured rooms
> (see [18_VALIDATION_AND_BENCHMARKS.md](18_VALIDATION_AND_BENCHMARKS.md)).

## Non-goals for v1 (on purpose)

- Vehicles or people as entities.
- Live/real-time capture (upload and guided capture only).
- Multi-user workspaces and real accounts (single local user; auth is designed
  for, not built — see [13](13_BACKEND_AND_PERSISTENCE.md)).
- City/street scale, IFC/BIM-grade export, learned geometry completion.
- Training our own models. v1 integrates and benchmarks existing ones.

## Long-term direction

A general spatial-intelligence platform: a world that other software and AI
agents can query ("which parts of the apartment were never captured?", "what
changed between visits?"). Everything in v1 is chosen so that this direction
needs no rewrite.
