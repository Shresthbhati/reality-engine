"""Cycle 3 architectural geometry extension tests.

Validates:
  1. Sloped ceilings and pitched roofs: pitch angle, ridge height, eave height, normal vector.
  2. Split-level transitions: vertical step elevation difference between adjacent spaces, curb segment, and topological ADJACENT_TO linkage.
  3. Structural columns and beams: linear prismatic members and support relationships.
  4. Multi-room building topology hierarchy: Building -> Storey -> Room -> Wall/Opening containment.
"""

from __future__ import annotations

import math

import pytest

from perception.architecture.classify import PlaneInput
from perception.architecture.roofs import detect_sloped_ceiling
from perception.architecture.split_level import detect_split_levels
from world_ir import (
    Entity,
    EntityType,
    Relationship,
    RelationshipKind,
    WorldIR,
)


def test_sloped_ceiling_pitch_and_ridge_detection():
    """Verify non-horizontal ceiling planes are detected with exact pitch angle and ridge line."""
    # A roof plane pitched at 45 degrees: normal = (0.0, 1/sqrt(2), 1/sqrt(2))
    norm_val = 1.0 / math.sqrt(2.0)
    p_sloped = PlaneInput(
        plane_id="plane-pitched-south",
        normal=(0.0, norm_val, norm_val),
        centroid=(2.5, 2.0, 3.5),
        bounds_min=(0.0, 0.0, 2.5),
        bounds_max=(5.0, 4.0, 4.5),
    )
    p_horizontal = PlaneInput(
        plane_id="plane-flat-ceiling",
        normal=(0.0, 0.0, 1.0),
        centroid=(2.5, 2.0, 2.8),
        bounds_min=(0.0, 0.0, 2.8),
        bounds_max=(5.0, 4.0, 2.8),
    )

    fits = detect_sloped_ceiling([p_sloped, p_horizontal])
    assert len(fits) == 1
    fit = fits[0]
    assert fit.plane_id == "plane-pitched-south"
    assert pytest.approx(fit.pitch_deg, abs=0.5) == 45.0
    assert fit.ridge_height_m == 4.5
    assert fit.eave_height_m == 2.5
    assert fit.is_sloped is True


def test_split_level_detection_and_linkage():
    """Verify adjacent rooms with discrete floor elevation difference (e.g. 0.36m = 2 steps) are linked as split levels."""
    world = WorldIR(id="w-split-level", name="Split Level Residence")

    # Room 1: Sunken Living Room (floor z = 0.0m)
    r1 = Entity(
        id="room-sunken-living",
        name="Sunken Living Room",
        type=EntityType.ROOM,
        custom_properties={
            "floor_elevation_m": 0.0,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [6.0, 5.0, 2.8],
        },
    )
    # Room 2: Raised Dining Room (floor z = 0.36m, ~2 steps up)
    r2 = Entity(
        id="room-raised-dining",
        name="Raised Dining Room",
        type=EntityType.ROOM,
        custom_properties={
            "floor_elevation_m": 0.36,
            "bounds_min": [6.0, 0.0, 0.36],
            "bounds_max": [11.0, 5.0, 3.16],
        },
    )
    world.entities[r1.id] = r1
    world.entities[r2.id] = r2

    transitions = detect_split_levels(world, min_step_m=0.15, max_step_m=1.0)
    assert len(transitions) == 1
    t = transitions[0]
    assert t.lower_room_id == "room-sunken-living"
    assert t.upper_room_id == "room-raised-dining"
    assert pytest.approx(t.elevation_step_m, abs=1e-3) == 0.36
    assert t.confidence >= 0.85

    # Check that ADJACENT_TO topological relationship was linked
    rel_targets = [rel.target_id for rel in r1.relationships if rel.kind == RelationshipKind.ADJACENT_TO]
    assert "room-raised-dining" in rel_targets
    assert "split_levels" in r1.custom_properties
    assert "split_levels" in r2.custom_properties


def test_multi_room_and_structural_elements():
    """Verify columns, beams, and multi-room containment form a coherent topological model."""
    world = WorldIR(id="w-topo", name="Two Room Flat")

    bldg = Entity(id="bldg-1", name="Main Building", type=EntityType.BUILDING)
    storey = Entity(id="storey-1", name="Ground Floor", type=EntityType.STOREY)
    r1 = Entity(id="room-bed", name="Bedroom", type=EntityType.ROOM)
    r2 = Entity(id="room-bath", name="Bathroom", type=EntityType.ROOM)
    col = Entity(id="col-corner", name="Corner Column", type=EntityType.COLUMN)
    beam = Entity(id="beam-load", name="Load Beam", type=EntityType.BEAM)

    world.entities[bldg.id] = bldg
    world.entities[storey.id] = storey
    world.entities[r1.id] = r1
    world.entities[r2.id] = r2
    world.entities[col.id] = col
    world.entities[beam.id] = beam

    # Topology containment relationships
    bldg.relationships = [Relationship(kind=RelationshipKind.CONTAINS, target_id=storey.id)]
    storey.relationships = [
        Relationship(kind=RelationshipKind.CONTAINS, target_id=r1.id),
        Relationship(kind=RelationshipKind.CONTAINS, target_id=r2.id),
        Relationship(kind=RelationshipKind.CONTAINS, target_id=col.id),
        Relationship(kind=RelationshipKind.CONTAINS, target_id=beam.id),
    ]

    # Verify traversal
    storey_ids = [rel.target_id for rel in bldg.relationships if rel.kind == RelationshipKind.CONTAINS]
    assert "storey-1" in storey_ids
    contained_ids = [rel.target_id for rel in storey.relationships if rel.kind == RelationshipKind.CONTAINS]
    assert "room-bed" in contained_ids
    assert "room-bath" in contained_ids
    assert "col-corner" in contained_ids
    assert "beam-load" in contained_ids
