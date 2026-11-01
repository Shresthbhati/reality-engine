# Rooms -> houses -> neighbourhoods (plan, 2026-10-31)

Goal: capture rooms as small worlds, merge sessions into a world, merge worlds into bigger worlds (house, street,
neighbourhood), anchor them to the real map, and keep only structure (never people or vehicles).

## Decisions (settled here, do not relitigate)
1. **Hierarchy.** Room = a world built from sessions. House / street / neighbourhood = a *composed world*: a new
   world version whose entities are the source worlds' entities, ids namespaced `<source_world>/<id>`, rigidly placed
   (translate x/z + yaw) from a Site. Sources are never modified; re-composing the same Site replaces the composed world.
   Existing types are enough: ROOM < STOREY < BUILDING < PLOT < BLOCK < LOCALITY < DISTRICT (world_ir/schema_v1.py).
2. **Precedence on overlap:** captured > imported map data > generated. A generated entity that overlaps a captured or
   imported one is dropped on the next compose.
3. **Imagined fill (driveway, street, corridor).** Generated, never asserted: `custom_properties.generated_by`,
   state ESTIMATED, confidence <= 0.3, labelled "Generated (not captured)". Sized by user-given parameters; replaced
   when real evidence for that area arrives.
4. **Geography.** A Site has an optional anchor (lat, lon, rotation_deg). The site frame is x = east, z = south
   (north is "back" on the grid). OpenStreetMap context (buildings with height/levels tags, roads, sidewalks, rail/metro
   and stations, bridges, steps, street furniture, trees) is imported as its own world with provenance "OpenStreetMap
   (ODbL)" and per-entity confidence from tag completeness. Missing tags -> estimated and flagged, never silent.
5. **Structure only.** Keep: buildings, walls, barriers/barricades (always), steps, stations, bridges, poles/lamps,
   bins, post boxes, benches, signs, discrete trees. Drop: people, vehicles (car, bus, truck, motorbike, bicycle),
   animals, loose foliage/undergrowth. Applied as image masks before reconstruction and as a filter on detections.
6. **Detector.** One interface (`Detector.detect(image, queries) -> [Detection]`); backends: OWL-ViT open-vocabulary
   via `transformers` (needs a ~600 MB weight download: user's go-ahead), torchvision COCO (~160 MB), and an injected
   fake for tests. No backend installed => the product says so; it never invents detections.
7. **Video** is a frame source: sample sharp, well-spaced keyframes (OpenCV), then the normal photo pipeline.

## Work packages (disjoint files)
A  world_ir/compose.py, world_ir/connectors.py, routes_sites.py compose + anchor endpoints, new tables
B  geo/osm_context.py, apps/api/routes_geo.py (+ recorded Overpass fixture, offline tests)
C1 perception/detection/* (interface, structure policy, masks, backends)
C2 perception/instances/furniture_from_detections.py (2D detections -> oriented 3D boxes on the floor, ESTIMATED sizes)
D  reconstruction/video_frames.py (+ synthetic-video test)
E  frontend: Sites compose / anchor / map-context UI, imported + generated styling in tree and viewport

## Honest limits
- Imagined fill is a placeholder shape, not a reconstruction.
- OSM coverage of street furniture in India is sparse; photos must supply it.
- Curved landmarks (domes, ornament) are still not reconstructable from photos (registry item LAND-1).
- Neighbour houses from the street: massing only (footprint + height) unless interior/side views are captured.
