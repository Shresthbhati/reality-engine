"""T-3.5: a room whose walls form a closed outline gets that outline as its footprint (L-shapes, notches), not the convex hull.

Ground truth is a hand-built layout in the solver's camera frame (x right, y DOWN, z forward, camera at the origin, 1.5 m above the
floor): an L-shaped plan of 6 x 3 m plus a 3 x 3 m arm, area 27 m2. The convex hull of the same walls is larger (it fills the notch).
"""
import pytest

from engine.pipeline.single_image_layout import add_layout_room
from perception.layout.surface_layout import Plane, SurfaceLayout, Wall
from world_ir import Entity, EntityType, WorldIR

# (inner normal, tangent, s_min, s_max, d) in the (x, z) plan, camera at the origin
L_WALLS = [
    ((0, 1), (1, 0), -1.5, 4.5, 2.0),
    ((-1, 0), (0, 1), -2.0, 1.0, 4.5),
    ((0, -1), (-1, 0), -4.5, -1.5, 1.0),
    ((-1, 0), (0, 1), 1.0, 4.0, 1.5),
    ((0, -1), (-1, 0), -1.5, 1.5, 4.0),
    ((1, 0), (0, -1), -4.0, 2.0, 1.5),
]


def _layout(walls):
    ws = tuple(Wall(Plane((n[0], 0.0, n[1]), d), (t[0], 0.0, t[1]), a, b, 1000, 0.01) for n, t, a, b, d in walls)
    return SurfaceLayout(camera_height_m=1.5, room_height_m=2.6, walls=ws)


def _room(walls):
    layout = _layout(walls)
    world = WorldIR(id="w", name="w", main_branch_id="b")
    eids = {}
    for i in range(len(walls)):
        eids[f"layout-wall-{i}"] = f"wall-{i}"
        world.entities[f"wall-{i}"] = Entity(id=f"wall-{i}", type=EntityType.WALL, name="wall")
    facts = add_layout_room(world, layout, eids, pre="t", confidence_ceiling=0.5)
    return world.entities["t-room-01"].custom_properties, facts


def test_an_l_shaped_room_keeps_its_notch():
    props, facts = _room(L_WALLS)
    assert facts["created"] and len(props["footprint_polygon_m"]) == 6
    assert props["floor_area_m2"] == pytest.approx(27.0, abs=0.3)
    assert props["visible_part_only"] is False


def test_partly_seen_walls_still_fall_back_to_the_visible_hull():
    props, _ = _room(L_WALLS[:2])  # two walls cannot close an outline
    assert props["visible_part_only"] is True
    assert "convex hull" in props["footprint_closure"]


def test_a_plain_rectangle_closes_to_four_corners():
    rect = [((0, 1), (1, 0), -2.5, 2.5, 2.0), ((-1, 0), (0, 1), -2.0, 3.0, 2.5),
            ((0, -1), (-1, 0), -2.5, 2.5, 3.0), ((1, 0), (0, -1), -3.0, 2.0, 2.5)]
    props, _ = _room(rect)
    assert len(props["footprint_polygon_m"]) == 4
    assert props["floor_area_m2"] == pytest.approx(25.0, abs=0.2)
