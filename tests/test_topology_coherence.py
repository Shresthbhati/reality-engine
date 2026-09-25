"""Building-topology coherence (AUTO RECON SPRINT P0).

The graph spine (room_graph.py) produced rooms + storeys, but nothing
consumed it: no corridors existed anywhere in the engine, windows never
associated with rooms, stairs never linked to storey transitions, and
rooms/storeys never became WorldIR entities. This suite pins the new
topology layer:

  perception/architecture/corridors.py  - measured corridor inference
  perception/architecture/topology.py   - entity promotion of the
      building graph (building -> storeys -> rooms -> parts), opening
      association, and stair-to-storey transition edges
"""

from __future__ import annotations

import pytest

from perception.architecture.classify import (
    ArchitecturalElement,
    PlaneInput,
    classify_planes,
)
from perception.architecture.parametric import FitRefused
from perception.architecture.room_graph import (
    build_building_graph,
    build_room_graph,
)
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Provenance,
    RelationshipKind,
)
from world_ir.world import WorldIR

UP = (0.0, 0.0, 1.0)


# ------------------------------------------------------------------
# Shared plane fixtures (same conventions as test_room_building_graph)
# ------------------------------------------------------------------


def _grid(a_range, b_range, step=0.1):
    a0, a1 = a_range
    b0, b1 = b_range
    n_a = int(round((a1 - a0) / step))
    n_b = int(round((b1 - b0) / step))
    for i in range(n_a + 1):
        for j in range(n_b + 1):
            yield a0 + i * step, b0 + j * step


def _floor_plane(z=0.0, pid="floor", x_span=3.0, y_span=2.5):
    pts = [(x, y, z) for x, y in _grid((0.0, x_span), (0.0, y_span))]
    return PlaneInput(
        plane_id=pid, normal=(0.0, 0.0, 1.0), centroid=(x_span / 2, y_span / 2, z),
        bounds_min=(0.0, 0.0, z), bounds_max=(x_span, y_span, z),
        inlier_positions=tuple(pts),
    )


def _ceiling_plane(z=2.4, pid="ceiling", x_span=3.0, y_span=2.5):
    pts = [(x, y, z) for x, y in _grid((0.0, x_span), (0.0, y_span))]
    return PlaneInput(
        plane_id=pid, normal=(0.0, 0.0, 1.0), centroid=(x_span / 2, y_span / 2, z),
        bounds_min=(0.0, 0.0, z), bounds_max=(x_span, y_span, z),
        inlier_positions=tuple(pts),
    )


def _wall_plane(pid, normal, a_lo, a_hi, z_lo=0.0, z_hi=2.4, step=0.02,
                gaps=(), offset=None):
    """An axis-aligned wall; `gaps` are (a_lo, a_hi, z_top) door-sized
    rectangles removed from the coverage grid. `offset` places the
    wall plane: (fixed_value, axis) where axis 0 = x-fixed (wall runs
    along y) and axis 1 = y-fixed (wall runs along x)."""
    runs_along_x = normal[1] != 0.0  # normal +/-y -> wall along x
    pts = []
    for a, z in _grid((a_lo, a_hi), (z_lo, z_hi), step=step):
        if runs_along_x:
            fixed = offset[0] if offset else 0.0
            pt = (a, fixed, z)
        else:
            fixed = offset[0] if offset else 0.0
            pt = (fixed, a, z)
        skip = False
        for g_lo, g_hi, g_top in gaps:
            if g_lo <= a <= g_hi and z <= g_top:
                skip = True
                break
        if not skip:
            pts.append(pt)
    return PlaneInput(
        plane_id=pid, normal=normal,
        centroid=_centroid(pts), bounds_min=_bounds(pts)[0],
        bounds_max=_bounds(pts)[1], inlier_positions=tuple(pts),
    )


def _centroid(pts):
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


def _bounds(pts):
    return (
        tuple(min(p[i] for p in pts) for i in range(3)),
        tuple(max(p[i] for p in pts) for i in range(3)),
    )


def _room_planes_with(walls, floor=None, ceiling=None):
    return [floor or _floor_plane(), ceiling or _ceiling_plane()] + walls



# ------------------------------------------------------------------
# P0: corridor inference (measured circulation, not "long rectangle")
# ------------------------------------------------------------------


def _corridor_planes():
    """A long thin room: 8 m x 1.2 m enclosure, openings (doors) on
    BOTH long walls (a corridor connects rooms on its sides)."""
    long_wall_a = _wall_plane(
        "wall-long-a", (0.0, -1.0, 0.0), 0.0, 8.0, offset=(0.0,),
        gaps=((1.0, 1.5, 2.0), (4.0, 4.5, 2.0), (6.5, 7.0, 2.0)),
    )
    long_wall_b = _wall_plane(
        "wall-long-b", (0.0, 1.0, 0.0), 0.0, 8.0, offset=(1.2,),
        gaps=((2.5, 3.0, 2.0), (5.5, 6.0, 2.0)),
    )
    end_a = _wall_plane("wall-end-a", (1.0, 0.0, 0.0), 0.0, 1.2,
                        offset=(0.0,))
    end_b = _wall_plane("wall-end-b", (-1.0, 0.0, 0.0), 0.0, 1.2,
                        offset=(8.0,))
    return [long_wall_a, long_wall_b, end_a, end_b], \
        _floor_plane(x_span=8.0, y_span=1.2), _ceiling_plane(x_span=8.0, y_span=1.2)


class TestCorridorInference:
    def test_detects_elongated_2_sided_connecting_room(self):
        from perception.architecture.corridors import detect_corridor

        walls, floor, ceiling = _corridor_planes()
        elements = classify_planes([floor, ceiling] + walls, UP)
        rooms = build_room_graph(elements, up=UP, planes=[floor, ceiling] + walls)
        assert len(rooms) == 1
        c = detect_corridor(rooms[0])
        assert c is not None
        assert c.kind == "corridor"
        # Measured: long axis is x (8 m vs 1.2 m).
        assert c.length_m == pytest.approx(8.0, abs=0.1)
        assert c.width_m == pytest.approx(1.2, abs=0.1)
        assert c.elongation == pytest.approx(8.0 / 1.2, abs=0.2)
        # Doors on BOTH long walls -> both sides connect.
        assert c.n_doors >= 2
        assert c.connects_both_sides is True
        assert 0.0 < c.confidence <= 1.0

    def test_refuses_square_room(self):
        from perception.architecture.corridors import detect_corridor

        walls = [
            _wall_plane("wall-front", (0.0, -1.0, 0.0), 0.0, 3.0),
            _wall_plane("wall-back", (0.0, 1.0, 0.0), 0.0, 3.0),
            _wall_plane("wall-side", (-1.0, 0.0, 0.0), 0.0, 2.5),
            _wall_plane("wall-side2", (1.0, 0.0, 0.0), 0.0, 2.5),
        ]
        floor, ceiling = _floor_plane(), _ceiling_plane()
        elements = classify_planes([floor, ceiling] + walls, UP)
        rooms = build_room_graph(elements, up=UP, planes=[floor, ceiling] + walls)
        assert len(rooms) == 1
        assert detect_corridor(rooms[0]) is None  # honest refusal

    def test_refuses_long_room_without_doors(self):
        from perception.architecture.corridors import detect_corridor

        # 8 x 1.2 but NO openings: sealed box is not a corridor.
        walls = [
            _wall_plane("wall-long-a", (0.0, -1.0, 0.0), 0.0, 8.0,
                        offset=(0.0,)),
            _wall_plane("wall-long-b", (0.0, 1.0, 0.0), 0.0, 8.0,
                        offset=(1.2,)),
        ]
        floor = _floor_plane(x_span=8.0, y_span=1.2)
        ceiling = _ceiling_plane(x_span=8.0, y_span=1.2)
        elements = classify_planes([floor, ceiling] + walls, UP)
        rooms = build_room_graph(elements, up=UP, planes=[floor, ceiling] + walls)
        if rooms:
            assert detect_corridor(rooms[0]) is None

    def test_single_side_doors_degrade_confidence(self):
        from perception.architecture.corridors import detect_corridor

        walls, floor, ceiling = _corridor_planes()
        # Rebuild long wall A with no gaps -> doors only on side B.
        walls = [
            _wall_plane("wall-long-a", (0.0, -1.0, 0.0), 0.0, 8.0,
                        offset=(0.0,)),
            _wall_plane(
                "wall-long-b", (0.0, 1.0, 0.0), 0.0, 8.0, offset=(1.2,),
                gaps=((2.5, 3.0, 2.0), (5.5, 6.0, 2.0)),
            ),
            walls[2], walls[3],
        ]
        floor = _floor_plane(x_span=8.0, y_span=1.2)
        ceiling = _ceiling_plane(x_span=8.0, y_span=1.2)
        elements = classify_planes([floor, ceiling] + walls, UP)
        rooms = build_room_graph(elements, up=UP, planes=[floor, ceiling] + walls)
        assert rooms, "fixture must still enclose"
        c = detect_corridor(rooms[0])
        if c is not None:
            assert c.connects_both_sides is False
            assert c.confidence < 0.85

# ------------------------------------------------------------------
# P0: topology promotion (building -> storeys -> rooms -> WorldIR)
# ------------------------------------------------------------------


def _world_with_parts(rooms_planes):
    """WorldIR pre-populated with wall/floor/ceiling entities promoted
    per the canonical promote_planes pattern, keyed to element ids the
    room graph produces (wall-<plane_id> etc.)."""
    world = WorldIR(id="w-test")
    for p in rooms_planes:
        role = (
            "wall" if p.plane_id.startswith("wall")
            else "floor" if p.plane_id == "floor"
            else "ceiling"
        )
        etype = {
            "wall": EntityType.WALL, "floor": EntityType.FLOOR,
            "ceiling": EntityType.CEILING,
        }[role]
        world.entities.add(Entity(
            id=f"{role}-{p.plane_id}", type=etype,
            name=f"{role} {p.plane_id}", confidence=0.9,
            provenance=Provenance.RECONSTRUCTED,
        ))
    # Floor/ceiling entities for both storeys (the graph uses explicit
    # element ids for these, built in _two_storey_planes).
    for eid, etype in [
        ("floor-floor", EntityType.FLOOR),
        ("ceiling-ceiling", EntityType.CEILING),
        ("floor-floor-2", EntityType.FLOOR),
        ("ceiling-ceiling-2", EntityType.CEILING),
    ]:
        if eid not in world.entities:
            world.entities.add(Entity(
                id=eid, type=etype, name=eid, confidence=0.9,
                provenance=Provenance.RECONSTRUCTED,
            ))
    return world


def _two_storey_planes():
    """Room at z=0..2.4 and a second stacked room at z=2.8..5.2.
    Floors/ceilings are constructed directly with explicit roles (the
    global classifier is single-room scope -- same convention as the
    multi-room test in test_room_building_graph)."""
    lower = _room_planes_with([
        _wall_plane("wall-front", (0.0, -1.0, 0.0), 0.0, 3.0,
                    gaps=((1.0, 1.6, 2.0),)),
        _wall_plane("wall-back", (0.0, 1.0, 0.0), 0.0, 3.0,
                    offset=(2.5,)),
        _wall_plane("wall-side", (-1.0, 0.0, 0.0), 0.0, 2.5,
                    offset=(0.0,)),
        _wall_plane("wall-side2", (1.0, 0.0, 0.0), 0.0, 2.5,
                    offset=(3.0,)),
    ])
    dz = 2.8
    upper = []
    for p in lower:
        if not p.plane_id.startswith("wall"):
            continue  # walls only: slabs have explicit elements
        upper.append(PlaneInput(
            plane_id=p.plane_id + "-2", normal=p.normal,
            centroid=tuple(c + (dz if k == 2 else 0.0) for k, c in enumerate(p.centroid)),
            bounds_min=tuple(c + (dz if k == 2 else 0.0) for k, c in enumerate(p.bounds_min)),
            bounds_max=tuple(c + (dz if k == 2 else 0.0) for k, c in enumerate(p.bounds_max)),
            inlier_positions=tuple((x, y, z + dz) for x, y, z in p.inlier_positions),
        ))
    return [p for p in lower if p.plane_id.startswith("wall")] + upper


def _two_storey_elements():
    """ArchitecturalElements for both storeys: walls from the plane
    fixtures + explicit floor/ceiling elements (the global classifier
    is single-room scope)."""
    walls = _two_storey_planes()
    els_walls = [
        ArchitecturalElement(
            element_id=f"wall-{p.plane_id}", element_type="wall",
            source_plane_id=p.plane_id, reason="fixture wall",
            bounds_min=p.bounds_min, bounds_max=p.bounds_max,
        )
        for p in walls
    ]
    els_slabs = [
        ArchitecturalElement(
            element_id="floor-floor", element_type="floor",
            source_plane_id="floor", reason="fixture storey 1 floor",
            bounds_min=(0.0, 0.0, 0.0), bounds_max=(3.0, 2.5, 0.0),
        ),
        ArchitecturalElement(
            element_id="ceiling-ceiling", element_type="ceiling",
            source_plane_id="ceiling", reason="fixture storey 1 ceiling",
            bounds_min=(0.0, 0.0, 2.4), bounds_max=(3.0, 2.5, 2.4),
        ),
        ArchitecturalElement(
            element_id="floor-floor-2", element_type="floor",
            source_plane_id="floor-2", reason="fixture storey 2 floor",
            bounds_min=(0.0, 0.0, 2.8), bounds_max=(3.0, 2.5, 2.8),
        ),
        ArchitecturalElement(
            element_id="ceiling-ceiling-2", element_type="ceiling",
            source_plane_id="ceiling-2", reason="fixture storey 2 ceiling",
            bounds_min=(0.0, 0.0, 5.2), bounds_max=(3.0, 2.5, 5.2),
        ),
    ]
    return els_walls + els_slabs


def _two_storey_world_planes():
    """PlaneInputs for all six surfaces of both storeys (for building
    the part entities in the world)."""
    lower = _room_planes_with([
        _wall_plane("wall-front", (0.0, -1.0, 0.0), 0.0, 3.0,
                    gaps=((1.0, 1.6, 2.0),)),
        _wall_plane("wall-back", (0.0, 1.0, 0.0), 0.0, 3.0,
                    offset=(2.5,)),
        _wall_plane("wall-side", (-1.0, 0.0, 0.0), 0.0, 2.5,
                    offset=(0.0,)),
        _wall_plane("wall-side2", (1.0, 0.0, 0.0), 0.0, 2.5,
                    offset=(3.0,)),
    ])
    dz = 2.8
    upper = []
    for p in lower:
        if not p.plane_id.startswith("wall"):
            continue
        upper.append(PlaneInput(
            plane_id=p.plane_id + "-2", normal=p.normal,
            centroid=tuple(c + (dz if k == 2 else 0.0) for k, c in enumerate(p.centroid)),
            bounds_min=tuple(c + (dz if k == 2 else 0.0) for k, c in enumerate(p.bounds_min)),
            bounds_max=tuple(c + (dz if k == 2 else 0.0) for k, c in enumerate(p.bounds_max)),
            inlier_positions=tuple((x, y, z + dz) for x, y, z in p.inlier_positions),
        ))
    return [p for p in lower if p.plane_id.startswith("wall")] + upper


class TestBuildingTopologyPromotion:
    def test_building_storey_room_entities_and_edges(self):
        from perception.architecture.topology import promote_building_topology

        planes = _two_storey_planes()
        elements = _two_storey_elements()
        rooms = build_room_graph(elements, up=UP, planes=planes)
        assert len(rooms) == 2
        building = build_building_graph(rooms, up=UP)
        world = _world_with_parts(_two_storey_world_planes())
        assert building is not None
        result = promote_building_topology(building, rooms, world)
        ids = {e.id for e in world.entities}
        # Building, two storeys, two rooms all become entities.
        assert result.building_id in ids
        assert len(result.storey_ids) == 2
        assert len(result.room_ids) == 2
        for rid in result.room_ids:
            assert world.entities.get(rid).type == EntityType.ROOM
        for sid in result.storey_ids:
            assert world.entities.get(sid).type == EntityType.STOREY
        assert world.entities.get(result.building_id).type == EntityType.BUILDING
        # CONTAINS edges: building -> storey -> room.
        b = world.entities.get(result.building_id)
        assert any(
            r.kind == RelationshipKind.CONTAINS and r.target_id in result.storey_ids
            for r in b.relationships
        )
        s1 = world.entities.get(result.storey_ids[0])
        assert any(
            r.kind == RelationshipKind.CONTAINS and r.target_id in result.room_ids
            for r in s1.relationships
        )
        # Rooms point back at their boundary parts (PART_OF).
        room = world.entities.get(result.room_ids[0])
        assert any(r.kind == RelationshipKind.PART_OF for r in room.relationships)

    def test_room_parts_and_openings_recorded(self):
        from perception.architecture.topology import promote_building_topology

        planes = _two_storey_planes()
        elements = _two_storey_elements()
        rooms = build_room_graph(elements, up=UP, planes=planes)
        building = build_building_graph(rooms, up=UP)
        world = _world_with_parts(_two_storey_world_planes())
        result = promote_building_topology(building, rooms, world)

        room0 = rooms[0]
        ent = world.entities.get(result.room_ids[0])
        # Openings measured on the walls ride into the entity properties.
        props = ent.custom_properties
        assert "openings" in props
        assert props["dimensions_m"]["z"] == pytest.approx(2.4, abs=0.05)
        # The room's CONTAINS targets all exist (registry integrity).
        for r in ent.relationships:
            assert r.target_id in world.entities

    def test_refuses_room_with_missing_parts(self):
        from perception.architecture.topology import promote_building_topology

        planes = _two_storey_planes()
        elements = classify_planes(planes, UP)
        rooms = build_room_graph(elements, up=UP, planes=planes)
        building = build_building_graph(rooms, up=UP)
        world = WorldIR(id="w-empty")  # no parts promoted
        with pytest.raises(Exception):
            promote_building_topology(building, rooms, world)

# ------------------------------------------------------------------
# P1: stairs <-> storey transitions + window association
# ------------------------------------------------------------------


def _stair_points(rise=0.16, going=0.28, n=6):
    pts = []
    for step in range(n + 1):
        z = step * rise
        for i in range(8):
            for j in range(8):
                x = step * going + (i / 7) * going
                y = -(0.6) + (j / 7) * 1.2
                pts.append((x, y, z))
    return pts


class TestStairStoreyTransition:
    def test_stairs_link_storeys_they_span(self):
        from perception.architecture.topology import (
            promote_building_topology,
            link_stairs_to_storeys,
        )

        planes = _two_storey_planes()
        elements = _two_storey_elements()
        rooms = build_room_graph(elements, up=UP, planes=planes)
        building = build_building_graph(rooms, up=UP)
        world = _world_with_parts(_two_storey_world_planes())
        result = promote_building_topology(building, rooms, world)

        fit = detect_stairs_fixture()
        # Promote the fit through the canonical component path first.
        from perception.architecture.components import ComponentObservation
        from perception.architecture.promotion import promote_component_to_entity
        obs = ComponentObservation(
            segment_id="seg-stairs-1", arch_class="stairs", fit=fit,
            evidence_ids=("ev-stairs-1",), confidence=fit.confidence,
            accepted=True, position=fit.position,
        )
        promote_component_to_entity(obs, world, "stairs-001")
        links = link_stairs_to_storeys([fit], building, world)
        # The stair spans z=0..0.96 -> storey 1 (z=0) only is too strict?
        # Contract: link EVERY storey whose floor height falls within the
        # stair's measured vertical span (inclusive of both ends' bands).
        assert links, "a stair spanning a storey floor must link it"
        assert links[0].stair_entity_id.startswith("stairs-")
        assert all(sid in world.entities for sid in links[0].storey_ids)
        # Entity edges recorded, ADJACENT_TO with derivation metadata.
        s_ent = world.entities.get(links[0].stair_entity_id)
        assert any(r.kind == RelationshipKind.ADJACENT_TO for r in s_ent.relationships)
        assert all(
            "derived_from" in r.metadata
            for r in s_ent.relationships
            if r.kind == RelationshipKind.ADJACENT_TO
        )


def detect_stairs_fixture():
    from perception.architecture.stairs import detect_stairs

    return detect_stairs(_stair_points())


class TestWindowRoomAssociation:
    def test_window_joins_room_via_shared_wall(self):
        from perception.architecture.topology import (
            promote_building_topology,
            associate_windows_to_rooms,
        )

        # Single room, front wall carries a window opening (sill 0.9).
        walls = [
            _wall_plane("wall-front", (0.0, -1.0, 0.0), 0.0, 3.0,
                        gaps=((1.4, 2.6, 0.9),), z_lo=0.0),
            _wall_plane("wall-back", (0.0, 1.0, 0.0), 0.0, 3.0),
            _wall_plane("wall-side", (-1.0, 0.0, 0.0), 0.0, 2.5),
            _wall_plane("wall-side2", (1.0, 0.0, 0.0), 0.0, 2.5),
        ]
        # Patch the front wall: window gap only above sill 0.9 (not a
        # floor-reaching door). Build coverage manually.
        front_pts = []
        for x, z in _grid((0.0, 3.0), (0.0, 2.4), step=0.02):
            if 1.4 <= x <= 2.6 and 0.9 <= z <= 1.9:
                continue
            front_pts.append((x, 0.0, z))
        from perception.architecture.classify import PlaneInput
        front = PlaneInput(
            plane_id="wall-front", normal=(0.0, -1.0, 0.0),
            centroid=(1.5, 0.0, 1.2), bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(3.0, 0.0, 2.4), inlier_positions=tuple(front_pts),
        )
        planes = [front] + walls[1:] + [_floor_plane(), _ceiling_plane()]
        elements = classify_planes(planes, UP)
        rooms = build_room_graph(elements, up=UP, planes=planes)
        assert len(rooms) == 1
        building = build_building_graph(rooms, up=UP)
        world = _world_with_parts(planes)
        result = promote_building_topology(building, rooms, world)

        # Promote the measured window the canonical way.
        from perception.architecture.windows import detect_window
        win_fit = detect_window(front, up=UP, floor_height=0.0)
        win_ent = Entity(
            id="window-001", type=EntityType.WINDOW, name="window",
            confidence=win_fit.confidence, provenance=Provenance.INFERRED,
            custom_properties={"wall_plane_id": win_fit.wall_plane_id},
        )
        world.entities.add(win_ent)

        assoc = associate_windows_to_rooms([win_ent], rooms, world)
        assert assoc and assoc[0].window_id == "window-001"
        assert assoc[0].room_ids == (result.room_ids[0],)
        # The room entity records the window in its properties.
        rent = world.entities.get(result.room_ids[0])
        assert "window-001" in rent.custom_properties.get("window_ids", [])

    def test_window_without_room_overlap_refuses(self):
        from perception.architecture.topology import (
            associate_windows_to_rooms,
        )

        from perception.architecture.room_graph import RoomGraph, RoomOpening
        world = WorldIR(id="w2")
        win = Entity(
            id="window-x", type=EntityType.WINDOW, name="w",
            confidence=0.8, provenance=Provenance.INFERRED,
        )
        world.entities.add(win)
        # A room on the far side of the world.
        room = RoomGraph(
            room_id="room-far",
            boundary_element_ids=("wall-a",),
            bounds_min=(100.0, 100.0, 0.0), bounds_max=(103.0, 102.5, 2.4),
            dimensions_m={"x": 3.0, "y": 2.5, "z": 2.4}, floor_area_m2=7.5,
        )
        assoc = associate_windows_to_rooms([win], [room], world)
        assert len(assoc) == 1 and assoc[0].room_ids == ()


# ------------------------------------------------------------------
# Determinism + honesty gates across the new layer
# ------------------------------------------------------------------


class TestDeterminismAndHonesty:
    def test_corridor_detection_deterministic(self):
        from perception.architecture.corridors import detect_corridor

        walls, floor, ceiling = _corridor_planes()
        elements = classify_planes([floor, ceiling] + walls, UP)
        rooms = build_room_graph(elements, up=UP, planes=[floor, ceiling] + walls)
        c1 = detect_corridor(rooms[0])
        c2 = detect_corridor(rooms[0])
        assert (c1.to_dict() if c1 else None) == (c2.to_dict() if c2 else None)

    def test_promotion_deterministic(self):
        from perception.architecture.topology import promote_building_topology

        planes = _two_storey_planes()
        elements = _two_storey_elements()
        rooms = build_room_graph(elements, up=UP, planes=planes)
        building = build_building_graph(rooms, up=UP)

        def run():
            w = _world_with_parts(_two_storey_world_planes())
            res = promote_building_topology(building, rooms, w)
            return res, w

        r1, w1 = run()
        r2, w2 = run()
        assert r1.to_dict() == r2.to_dict()
        assert [e.to_dict() for e in w1.entities] == \
               [e.to_dict() for e in w2.entities]

    def test_corridor_kind_is_registry_declared(self):
        # The corridor class must be registry-declared, not a magic
        # string at the call site.
        from perception.architecture.registry import get_default_registry

        reg = get_default_registry()
        assert reg.has("corridor")
