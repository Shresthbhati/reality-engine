# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-auto-recon-sprint (from origin/main 310ceff)
Checkpoint: AUTO_RECON_TOPOLOGY_READY
Status: CHECKPOINT_READY

## Mission

Automatic Reconstruction + World Compiler sprint (P0 topological
coherence, P0 rooms/corridors, P1 windows/stairs association):
turn the orphaned room/building graph measurements into a consumed,
canonical building topology in WorldIR.

## What changed

1. perception/architecture/corridors.py (NEW) -- first corridor
   inference in the engine. Measured, three-gate discipline:
   elongation >= CORRIDOR_MIN_ELONGATION (3x), short axis <=
   CORRIDOR_MAX_WIDTH_M (2.5 m), >= CORRIDOR_MIN_DOORS (2) measured
   openings. connects_both_sides is measured from distinct door
   walls; single-side service decays confidence (never fabricated
   symmetry). Refusal is None, never a guess. CorridorFit carries
   only measured quantities.

2. perception/architecture/topology.py (NEW) -- the promotion write
   path for the building graph (previously tests-only):
   - promote_building_topology: STOREY + CORRIDOR additive
     EntityTypes in world_ir/schema_v1.py; BUILDING CONTAINS STOREY
     CONTAINS ROOM CONTAINS parts (+ inverse PART_OF), room
     custom_properties carry measured dimensions/area/openings/
     adjacency, room confidence = min(part confidences), hard
     refusal (TopologyError) when boundary parts were never promoted
     (no dangling topology), atomic validation before any write.
   - link_stairs_to_storeys: StaircaseFit's measured vertical span
     (support centroid +/- n*rise/2 -- derived arithmetic on
     measured quantities, documented in-code) links every storey
     floor it spans, ADJACENT_TO edges with derived_from metadata;
     refuses when the stair entity was never promoted.
   - associate_windows_to_rooms: a WINDOW entity joins the room whose
     boundary contains its measured wall (wall_plane_id ->
     wall-<plane_id> element membership); recorded in the room's
     window_ids properties + ADJACENT_TO edge; unmatched windows
     report empty room_ids (recorded absence).

3. perception/architecture/registry.py -- corridor class registered
   (FitKind.POINT_CLOUD, min_extent_m=2.0); kind is registry-declared,
   not a call-site magic string.

4. perception/architecture/promotion.py -- entity write made
   polymorphic over the entity container (EntityRegistry.add with
   relationship-integrity enforcement, or plain dict for the older
   test convention) via promotion._store_entity; shared by topology.

5. world_ir/schema_v1.py -- EntityType.STOREY and EntityType.CORRIDOR
   added additively (same backward-compatible pattern as the P7-03
   expansion; old worlds load unchanged).

## Tests

tests/test_topology_coherence.py (NEW, 13 red-first tests):
- corridor: detected on an 8x1.2 double-loaded fixture with measured
  length/width/elongation; square room refuses; sealed long room
  refuses; single-side doors degrade confidence + connects_both_sides
  False; deterministic.
- promotion: building/storey/room entities + CONTAINS/PART_OF edges;
  openings + measured dimensions recorded in properties; refusal on
  missing parts; deterministic across full-world comparison.
- stairs: measured-span links to storeys with metadata; entities
  promoted through the canonical component path first.
- windows: joins room via shared wall element + recorded in room
  properties; unmatched window reports empty association.
- registry-declared corridor class.

Cluster verification (this machine, Python 3.14):
- tests/test_topology_coherence.py           13 passed
- tests/test_column_beam_detectors.py        19 passed
- tests/test_window_roof_detectors.py        16 passed
- tests/test_architectural_perception.py      6 passed
- tests/test_room_building_graph.py          11 passed
- tests/test_stair_detector.py               10 passed
- tests/test_room_inference.py               20 passed
No existing test was weakened; the two-storey fixtures construct
elements explicitly because classify_planes is documented single-room
scope (same convention as the multi-room test in
tests/test_room_building_graph.py).

## Known limitations

- Corridor classification runs at the room-graph level: it counts
  distinct door walls rather than resolving each wall's geometric
  side (wall orientation does not ride on RoomGraph); a corridor
  with doors only on its two END caps and none on its sides would
  still count 2 "distinct walls" -- recorded as a refinement item.
- Stair span uses the fit's centroid + measured rhythm (the fit does
  not carry absolute first/last band heights); documented in-code.
- Windows associate through wall-plane membership; windows on walls
  shared by two rooms associate with both (recorded, not guessed).
- Facades/roads/curbs/infrastructure/terrain/vegetation remain
  unimplemented (no fixture; skipped, not faked).
