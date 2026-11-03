"""Unit tests for Street and Urban Environment Profiler (Wave 2, Block 14)."""

import math
import numpy as np
import pytest

from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, RelationshipKind
from perception.environment.urban_profiler import (
    CarriagewayProfile,
    CurbProfile,
    PedestrianCrossing,
    SidewalkProfile,
    StreetCorridorProfile,
    StreetProfileRefused,
    profile_street_corridor,
    promote_street_corridor,
)


def _generate_synthetic_street_corridor(
    road_width: float = 7.0,
    sw_left_width: float = 2.0,
    sw_right_width: float = 2.0,
    curb_height: float = 0.15,
    corridor_length: float = 20.0,
    seed: int = 42,
) -> list:
    """Generate 3D points representing a complete street corridor cross-section.

    Lateral u:
      [-sw_left_width - road_width/2, -road_width/2] -> Left sidewalk (elevation +curb_height)
      [-road_width/2, +road_width/2] -> Carriageway (elevation ~0)
      [+road_width/2, +road_width/2 + sw_right_width] -> Right sidewalk (elevation +curb_height)
    Longitudinal v: [0, corridor_length]
    """
    rng = np.random.default_rng(seed)
    pts = []

    half_rw = road_width / 2.0

    # 1. Road points
    for _ in range(80):
        u = rng.uniform(-half_rw, half_rw)
        v = rng.uniform(0, corridor_length)
        # Parabolic crown: apex at u=0, drops towards gutters
        w = -0.02 * abs(u) + rng.normal(0, 0.005)
        # Lateral along X, longitudinal along Y, vertical Z
        pts.append((u, v, w))

    # 2. Left sidewalk points
    for _ in range(40):
        u = rng.uniform(-half_rw - sw_left_width, -half_rw)
        v = rng.uniform(0, corridor_length)
        w = curb_height + rng.normal(0, 0.005)
        pts.append((u, v, w))

    # 3. Right sidewalk points
    for _ in range(40):
        u = rng.uniform(half_rw, half_rw + sw_right_width)
        v = rng.uniform(0, corridor_length)
        w = curb_height + rng.normal(0, 0.005)
        pts.append((u, v, w))

    return pts


def test_profile_street_corridor_success():
    pts = _generate_synthetic_street_corridor()
    profile = profile_street_corridor(pts)

    assert isinstance(profile, StreetCorridorProfile)
    assert profile.point_count == len(pts)
    assert profile.corridor_length_m > 15.0

    # Carriageway checks
    assert 6.0 <= profile.carriageway.width_m <= 8.5
    assert profile.carriageway.lane_count == 2
    assert profile.carriageway.cross_fall_pct > 0.0

    # Sidewalk & Curb checks
    assert profile.left_curb is not None
    assert 0.10 <= profile.left_curb.height_m <= 0.25
    assert profile.right_curb is not None
    assert 0.10 <= profile.right_curb.height_m <= 0.25

    assert profile.left_sidewalk is not None
    assert 1.5 <= profile.left_sidewalk.width_m <= 3.0
    assert profile.right_sidewalk is not None
    assert 1.5 <= profile.right_sidewalk.width_m <= 3.0

    # Pedestrian crossing
    assert len(profile.crossings) >= 1
    crossing = profile.crossings[0]
    assert crossing.span_m > 5.0
    assert crossing.has_curb_ramps is True


def test_profile_street_corridor_furniture_association():
    pts = _generate_synthetic_street_corridor()

    # Create dummy lamppost on the right sidewalk verge (u = 4.0, v = 10.0)
    lp = Entity(
        id="ent-lamppost-1",
        type=EntityType.STRUCTURE,
        name="Lamppost",
        custom_properties={"position": [4.0, 10.0, 0.15]},
    )
    # Create dummy tree on the left sidewalk verge (u = -4.0, v = 5.0)
    tree = Entity(
        id="ent-tree-1",
        type=EntityType.VEGETATION,
        name="Tree",
        custom_properties={"position": [-4.0, 5.0, 0.15]},
    )

    profile = profile_street_corridor(pts, furniture_entities=[lp, tree])
    assert "ent-lamppost-1" in profile.furnishing_entity_ids
    assert "ent-tree-1" in profile.furnishing_entity_ids


def test_promote_street_corridor_world_ir():
    pts = _generate_synthetic_street_corridor()
    world = WorldIR()

    # Pre-add tree in world
    tree = Entity(
        id="ent-tree-1",
        type=EntityType.VEGETATION,
        name="Tree",
        custom_properties={"position": [-4.0, 5.0, 0.15]},
    )
    world.entities[tree.id] = tree

    profile = profile_street_corridor(pts, furniture_entities=[tree])
    root = promote_street_corridor(profile, world)

    assert root.id in world.entities
    assert root.type == EntityType.STREET
    assert "measurements" in root.custom_properties
    assert "right_of_way_width_m" in root.custom_properties["measurements"]

    # Check child entities in world
    types = [e.type for e in world.entities.values()]
    assert EntityType.ROAD in types
    assert EntityType.SIDEWALK in types
    assert EntityType.CURB in types

    # Check relationships
    all_relationships = []
    for ent in world.entities.values():
        all_relationships.extend(ent.relationships)

    part_of_edges = [r for r in all_relationships if r.kind == RelationshipKind.PART_OF]
    assert len(part_of_edges) >= 4  # Road, 2 sidewalks, 2 curbs, plus tree

    adj_edges = [r for r in all_relationships if r.kind == RelationshipKind.ADJACENT_TO]
    assert len(adj_edges) >= 2  # Sidewalks adjacent to carriageway


def test_profile_street_corridor_refusal_insufficient_points():
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    with pytest.raises(StreetProfileRefused):
        profile_street_corridor(pts)


def test_profile_street_corridor_refusal_narrow_span():
    # Only 1.0m width points
    pts = [(float(x) * 0.05, float(y), 0.0) for x in range(10) for y in range(5)]
    with pytest.raises(StreetProfileRefused):
        profile_street_corridor(pts)
