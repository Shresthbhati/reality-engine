# Interior Compiler: Convergence Audit & Canonical Path

Status: living document. Written from a direct code audit (file:line
citations below); update it when the pipeline changes rather than
letting it drift into fiction.

## 1. The canonical pipeline (as it exists today)

```
ReconstructionResult (evidence -> reconstruction)
  -> detect_planes                      perception/geometry/planes.py
  -> classify_planes                    perception/geometry/orientation.py
  -> promote_plane_to_entity            evidence/promote_planes.py         -> WALL/FLOOR/CEILING entities
  -> detect_rooms + promote_room_to_entity
                                         evidence/promote_rooms.py          -> ROOM entities (evidence-side)
  -> build_room_graph                   perception/architecture/room_graph.py -> RoomGraph[]
  -> detect_corridors                   perception/architecture/corridor.py   -> CorridorGraph[]
  -> detect_window (per wall)           perception/architecture/windows.py    -> WindowFit[]
  -> detect_stairs                      perception/architecture/stairs.py     -> StaircaseFit
  -> build_building_graph               perception/architecture/room_graph.py -> BuildingGraph
  -> promote windows/stairs/corridors   engine/compiler/world_compiler.py (inline)  -> WINDOW/STAIRS/CORRIDOR entities
  -> promote_building_topology (opt-in) perception/architecture/topology.py   -> ROOM(topology-side)/STOREY/BUILDING entities
  -> link_stairs_to_storeys (opt-in)    perception/architecture/topology.py   -> STAIRS<->STOREY ADJACENT_TO edges
  -> associate_windows_to_rooms         perception/architecture/topology.py   -> WINDOW<->ROOM ADJACENT_TO edges
  -> InteriorSpaceGraph.from_building    perception/architecture/space_graph.py -> world.metadata["interior_space_graph"]
  -> validate_world_ir                  world_ir/validation.py                 (gate; ERROR blocks the compile)
  -> WorldStore.save_version / load_version   worldstore/store.py
```

Single entry point: `engine.compiler.world_compiler.compile_reconstruction_to_world`
(`engine/compiler/world_compiler.py:190`). Everything above runs inside
one function call; there is no second competing top-level entry point.

## 2. Representation-by-representation: who creates it, who consumes it

| Representation | Created by | Consumed by | Canonical identity |
|---|---|---|---|
| `RoomGraph` | `perception/architecture/room_graph.py:109` `build_room_graph` | `corridor.py::detect_corridors`, `build_building_graph`, `topology.py::promote_building_topology`, `InteriorSpaceGraph.from_building` | in-memory only, `room_id` (e.g. `room-graph-plane-002`); never itself persisted |
| `DetectedRoom` (evidence-side) | `evidence/promote_rooms.py:478` `detect_rooms` | `world_compiler.py` (promotes directly, independent of `RoomGraph`) | in-memory only, keyed by floor `plane_id` |
| `BuildingGraph` | `room_graph.py:180` `build_building_graph` | `topology.py::promote_building_topology`, `InteriorSpaceGraph.from_building` | in-memory only, `building_id` |
| `CorridorGraph` | `corridor.py:67` `detect_corridors` (internally harmonizes with the older `corridors.py::detect_corridor`) | `world_compiler.py` (promotes to WorldIR), `InteriorSpaceGraph.from_building` | in-memory only, `corridor_id` |
| `InteriorSpaceGraph` | `space_graph.py:26` `from_building()` (pure consumer, no re-detection) | nothing downstream today except `world.metadata["interior_space_graph"]` (JSON blob) | in-memory + serialized dict; **not entity-id-addressable** in WorldIR |
| WorldIR `Entity(type=ROOM)` (evidence-side) | `evidence/promote_rooms.py::promote_room_to_entity`, id `f"{room_prefix}-{floor.plane_id}"` (e.g. `room-plane-002`) | WorldStore, Reality Studio, `promote_building_topology` (via dedup match, see 3) | **This is the canonical room identity going forward.** |
| WorldIR `Entity(type=ROOM)` (topology-side) | `perception/architecture/topology.py::_room_entity`, id `f"room-{index:03d}"` | WorldStore, Reality Studio | Reused/merged into the evidence-side entity when the two overlap >=50% on boundary parts (see 3); only mints a new id when no match exists |
| WorldIR `Entity(type=STOREY)` | `topology.py::promote_building_topology`, id `f"storey-{si:02d}"` | WorldStore, `InteriorSpaceGraph` metadata (informational only) | canonical; only creator |
| WorldIR `Entity(type=LEVEL)` | `perception/architecture/promotion.py::promote_interior_graph_to_world` | **nothing** -- unwired | **dead code**, see 4 |
| WorldIR `Entity(type=BUILDING)` | `topology.py::promote_building_topology` | WorldStore | canonical; only creator |
| WorldIR `Entity(type=WINDOW)` | `world_compiler.py` inline (step 6) | `topology.py::associate_windows_to_rooms` | canonical; only creator. Now also carries a `PART_OF` edge to its host wall entity (added; see 5) |
| WorldIR `Entity(type=STAIRS)` | `world_compiler.py` inline (step 7) | `topology.py::link_stairs_to_storeys` | canonical; only creator |
| WorldIR `Entity(type=CORRIDOR)` | `world_compiler.py` inline (step 8) | WorldStore, `InteriorSpaceGraph` metadata | canonical; only creator |

## 3. Duplication found, and what was done about it

**Confirmed and fixed this pass:** when `CompileOptions.promote_building=True`,
`evidence/promote_rooms.py` and `perception/architecture/topology.py`
both promoted a `ROOM` entity for the same physical room -- reproduced
live: 5 `ROOM` entities for a 2-room synthetic scene. Fixed in
`perception/architecture/topology.py::_find_existing_room_entity`
(matches a `RoomGraph` room against an already-promoted `ROOM` entity by
>=50% Jaccard overlap of `CONTAINS`-boundary parts, and reuses it
instead of minting a duplicate). Verified: 5 -> 4 room entities on the
canonical scene; the residual 4 (not 2) is because the two detectors
*disagree* on room composition for the corridor-adjacent space in this
scene (Jaccard < 0.5) -- a genuine algorithm-level disagreement between
the wall-ring tracer (`evidence/promote_rooms.py`) and the enclosure
grouper (`room_graph.py::build_room_graph`), not a wiring bug. See 6,
"Open decision."

**Confirmed, NOT touched (deliberately, per "do not delete anything
merely because it looks duplicated" and the scope limits of this
pass):**

- `perception/architecture/corridors.py::detect_corridor` (older
  "auto-recon sprint P0" room-reclassification heuristic) is not dead:
  `corridor.py:296-336` calls it internally as a harmonization fallback
  for any `RoomGraph` the geometric detector's own pass misses. Only
  `tests/test_topology_coherence.py` exercises it directly. Leave as
  is; it is a genuine fallback layer inside the canonical detector, not
  a second canonical path.
- `perception/architecture/promotion.py::promote_interior_graph_to_world`
  is a third, fully independent room/building promotion path (uses
  `EntityType.LEVEL` instead of `STOREY` for the same concept) with
  **zero callers** anywhere in the compiler or its tests. It duplicates
  what `topology.py::promote_building_topology` and `world_compiler.py`'s
  inline promotion already do, more completely and with the id scheme
  actually used everywhere else. It is dead code, not a competing
  source of truth in practice (nothing ever calls it), but it should be
  either wired to nothing on purpose (delete) or explicitly marked
  deprecated -- deferred to a follow-up pass since deleting it is a
  larger, separately-reviewable decision (mission explicitly said "do
  not delete anything merely because it looks duplicated" without first
  establishing the dependency graph -- this document is that
  establishment; the deletion itself is not part of this pass).
- `EntityType.STOREY` and `EntityType.LEVEL` both exist in
  `world_ir/schema_v1.py` for the same concept ("one building floor").
  Only `STOREY` is ever produced by the live pipeline (`LEVEL` is
  produced only by the dead `promote_interior_graph_to_world` above).
  Not removed from the enum in this pass (removing an enum value is a
  schema change with a blast radius outside the interior compiler);
  flagged so a future pass doesn't reintroduce `LEVEL` as if it were
  live.
- `world_ir/entity.py` + `world_ir/world.py` (a legacy `Entity`/`WorldIR`
  pair with `type: str` and an `EntityRegistry` container) still exists
  and is exercised directly by `tests/test_topology_coherence.py`
  (imports `from world_ir.world import WorldIR`), while every other
  interior test and the compiler itself use the canonical
  `world_ir.world_v1.WorldIR` (re-exported as `world_ir.WorldIR`). Not a
  competing source of truth for the *compiler's own output* (the
  compiler never touches the legacy pair), but `topology.py`'s
  `promote_building_topology`/`_find_existing_room_entity` had to be
  written to tolerate both `world.entities` shapes (plain dict vs.
  `EntityRegistry`) because that one test file uses the legacy pair
  directly against the same promotion functions the live compiler uses.

## 4. Where information could be lost (checked, mostly not an issue)

- **`InteriorSpaceGraph` is a JSON blob in `world.metadata`, not entity
  data.** It survives a `WorldStore` save/load byte-for-byte (verified,
  `tests/test_interior_invariants.py::TestCanonicalCompileEndToEnd::test_worldstore_roundtrip_preserves_topology`)
  because `WorldIR.to_dict()`/`from_dict()` round-trip `metadata`
  wholesale. It is not independently re-validated on load and its room
  ids are **not guaranteed to match** the promoted `ROOM` entity ids
  (it is built from `RoomGraph`/`CorridorGraph` before the dedup-aware
  entity promotion runs). Reality Studio should treat it as a
  *secondary, informational* index for fast adjacency/reachability
  queries, never as the source of truth for "does this room exist" --
  that answer lives in `world.entities`.
- **Confidence/uncertainty are populated for corridors/windows/stairs**
  (measured formulas in `corridors.py`, `corridor.py`, `windows.py`,
  `stairs.py`) and faithfully copied into `Entity.confidence` at
  promotion time -- not diluted, not fabricated.
- **`RoomGraph.confidence`/`.boundary_completeness`/`.ceiling_evidence`
  are declared fields `build_room_graph` never actually computes**
  (`perception/architecture/room_graph.py:510-518` always constructs
  `confidence=1.0, boundary_completeness=1.0, ceiling_evidence="measured"`
  regardless of real enclosure quality). Not fixed in this pass
  (changing detector-internal confidence semantics is a bigger, more
  invasive change than schema/wiring fixes); flagged as an open item.
- **Doorway `RoomOpening`s never get `evidence_ids` populated** (window
  openings do, `room_graph.py:325-338` vs. `room_graph.py:317-323`) --
  a real traceability gap, not fixed in this pass for the same reason.

## 5. Invariants added this pass (`world_ir/validation.py`)

Composed into the existing `validate_world_ir()` gate (already run
before every compile returns, `world_compiler.py`'s "validation gate"
step):

- `_check_interior_boundary_references`: a `ROOM`/`CORRIDOR` entity's
  `custom_properties["boundary_element_ids"]` must all reference
  entities that exist. (The `CONTAINS`/`PART_OF` relationship copy of
  the same fact was already checked generically by `world.validate()`;
  this checks the informational duplicate directly so the two can't
  silently desync.)
- `_check_single_storey_membership`: a `ROOM`/`CORRIDOR` cannot be
  `CONTAINS`-owned by two different `STOREY` entities at once
  ("impossible containment" / "entity belonging to two incompatible
  levels").
- A `WINDOW` entity is now wired to its host `WALL` entity by a real
  `PART_OF`/`CONTAINS` relationship pair (`engine/compiler/world_compiler.py`,
  window-promotion step), not only by a raw `wall_plane_id` string in
  `custom_properties`. Before this, "window references a nonexistent
  host" could not even be detected -- there was no graph edge to check.
  Now it is caught for free by the existing generic dangling-relationship
  check in `world.validate()`.

Already present before this pass (confirmed by direct code reading, not
re-implemented): non-finite/inverted geometry bounds, non-finite/singular
transforms, duplicate entity identity (type+name+geometry fingerprint),
non-finite or impossible measurements, confidence-outside-[0,1] and
UNKNOWN-provenance-with-high-confidence, RECONSTRUCTED-without-trace
(warning), and all dangling-relationship/geometry/material/surface/
component references (`world_ir/world_v1.py::validate`). Duplicate
canonical *ids* are structurally impossible in the live pipeline: both
the plain-dict `world_v1.WorldIR.entities` (assignment-based) and the
legacy `EntityRegistry.add()` either overwrite silently (dict) or raise
`DuplicateEntityError` (registry) -- the dict path relies on the
compiler's own `if x not in world.entities` idempotency guards, which
were audited and are present at every promotion site.

Also fixed this pass, orthogonal to invariants but part of the same
"no silently discarded information" discipline the mission asks for:
six `except Exception: pass` blocks in `world_compiler.py` around
window/stair detection and corridor/building topology promotion now
record genuine failures into `CompileDiagnostics.interior_warnings`
(new field) instead of discarding them -- while still treating each
detector's own documented `FitRefused` "honest no-detection" signal as
silent, since that is expected behavior, not an error.

## 6. Room-detector reconciliation (resolved)

**Decision:** `evidence/promote_rooms.py`'s wall-ring-closure detector
is authoritative over `perception/architecture/room_graph.py::build_room_graph`'s
enclosure grouper whenever both run against the same world (i.e. always,
inside `engine.compiler.world_compiler`, since evidence-side detection
runs first for every compile). Rationale: ring closure requires a
*closed* boundary loop of walls resting on the candidate floor; the
enclosure grouper only requires bounding-box proximity, which is
measurably coarser -- on the canonical two-room scene it produced a
10-boundary-part "room" that the ring tracer correctly refused to
recognize as one enclosed space.

**Mechanism** (`perception/architecture/topology.py::promote_building_topology`):
`RoomGraph` still feeds corridor detection, storey/building grouping,
and `InteriorSpaceGraph` construction unchanged -- it remains the
structural scaffold. But when this function is called with evidence-side
`ROOM` entities already present in the world, a `RoomGraph` room with no
matching promoted entity (`_find_existing_room_entity` returns `None`,
i.e. <50% boundary overlap with anything evidence-side already found) is
no longer independently promoted as a second, unverified `ROOM` entity.
It is recorded in `TopologyPromotionResult.unmatched_room_ids` -- a
refusal, not a fabrication -- and surfaced by the compiler into
`CompileDiagnostics.interior_warnings` so the disagreement stays visible
rather than silently becoming an extra room. When no evidence-side `ROOM`
entities exist yet (the standalone use of `promote_building_topology`,
e.g. `tests/test_topology_coherence.py`'s 13 tests), there is no
stricter detector to defer to, so an unmatched room still mints its own
entity exactly as before -- that usage is unchanged.

**Verified:** on the canonical two-room scene, the promoted `ROOM`
entity count now equals evidence-side detection exactly (3, matching
`diag.rooms_detected`) instead of drifting to 4-5 depending on partial
overlap luck, and the one RoomGraph grouping the ring tracer didn't
corroborate (`room-002`, the corridor-adjacent over-merge) is reported
by name in `interior_warnings` rather than silently becoming a phantom
room. `tests/test_interior_invariants.py::TestCanonicalCompileEndToEnd::test_full_compile_does_not_duplicate_rooms_across_detectors`
pins both facts. All 13 `test_topology_coherence.py` tests (the
standalone-usage path) still pass unchanged.

## 7. Real-data proof

No real *interior* capture is committed to this repository (the
fixtures referenced by `tests/integration/test_system_runtime_proof.py`
as `datasets/real_room_capture_worldir/` and
`datasets/room_capture/pipeline_out/` are locally-generated and
gitignored; `scripts/generate_room_dataset.py` is honestly labeled
"synthetic," not real capture). The strongest real capture actually
committed is `datasets/south_building/` (32 real photographs, real
committed COLMAP ground truth: `cameras.txt`/`images.txt`/`points3D.txt`,
49,608 real 3D points) -- an outdoor courtyard/facade dataset, not an
interior scene.

`tests/test_interior_invariants.py::TestRealCaptureProof` runs this
real data through the full compiler (`promote_building=True`) and
asserts: zero validation errors, zero silent `interior_warnings`, and
**zero fabricated rooms** -- the room detector's `NO_CLOSED_RING`
refusal fires correctly on real geometry (no wall rests on a candidate
floor, because there is no enclosed room in a courtyard), then the
resulting `WorldIR` round-trips through `WorldStore` cleanly. This is
the honest real-data proof available: it proves the pipeline handles
real, non-synthetic point clouds without crashing, without fabricating
structure, and without validation errors -- it does not and cannot
prove "the compiler finds a real room," because no real room dataset
exists in this repository yet.
