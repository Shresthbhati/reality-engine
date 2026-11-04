"""Unit tests for Topology Revalidation after Merges (Blueprint Task 5.4).

Verifies:
1. Valid merged room topology passes geometric revalidation with zero errors.
2. Unclosed room boundaries resulting from bad merges are detected and flagged.
3. Penetrating / self-intersecting walls are detected and flagged.
4. Over-sized openings that exceed host wall lengths are flagged.
5. Incompatible elevation steps across shared partition walls are flagged.
6. Failures are cleanly recorded in the report and metadata, never hidden.
"""

from __future__ import annotations

import pytest

from perception.architecture.topology import revalidate_merged_topology
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType


def _build_valid_room_world() -> WorldIR:
    """Build a world with a valid closed rectangular room (4x4m)."""
    world = WorldIR(id="world-valid-room")

    world.entities["room-01"] = Entity(
        id="room-01",
        name="Valid Room",
        type=EntityType.ROOM,
        confidence=0.9,
        provenance=Provenance.INFERRED,
        custom_properties={
            "boundary_element_ids": ["w-north", "w-east", "w-south", "w-west"],
            "floor_area_m2": 16.0,
            "floor_elevation_m": 0.0,
        },
    )

    # 4 walls forming a closed loop [0, 4] x [0, 4]
    world.entities["w-north"] = Entity(
        id="w-north",
        type=EntityType.WALL,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "start_point": [0.0, 4.0, 0.0],
            "end_point": [4.0, 4.0, 0.0],
            "bounds_min": [0.0, 4.0, 0.0],
            "bounds_max": [4.0, 4.0, 2.5],
            "room_id": "room-01",
        },
    )

    world.entities["w-east"] = Entity(
        id="w-east",
        type=EntityType.WALL,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "start_point": [4.0, 4.0, 0.0],
            "end_point": [4.0, 0.0, 0.0],
            "bounds_min": [4.0, 0.0, 0.0],
            "bounds_max": [4.0, 4.0, 2.5],
            "room_id": "room-01",
        },
    )

    world.entities["w-south"] = Entity(
        id="w-south",
        type=EntityType.WALL,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "start_point": [4.0, 0.0, 0.0],
            "end_point": [0.0, 0.0, 0.0],
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [4.0, 0.0, 2.5],
            "room_id": "room-01",
        },
    )

    world.entities["w-west"] = Entity(
        id="w-west",
        type=EntityType.WALL,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "start_point": [0.0, 0.0, 0.0],
            "end_point": [0.0, 4.0, 0.0],
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [0.0, 4.0, 2.5],
            "room_id": "room-01",
        },
    )

    # Valid door on south wall (width 0.9m on 4.0m wall)
    world.entities["door-01"] = Entity(
        id="door-01",
        type=EntityType.DOOR,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "host_wall_id": "w-south",
            "width_m": 0.9,
            "height_m": 2.1,
        },
    )

    return world


def test_valid_topology_revalidation():
    """Valid merged topology passes with zero errors."""
    world = _build_valid_room_world()
    report = revalidate_merged_topology(world)

    assert report.is_valid is True
    assert report.error_count == 0
    assert report.checked_rooms == 1
    assert report.checked_walls == 4
    assert report.checked_openings == 1
    assert world.metadata["topology_revalidation"]["is_valid"] is True


def test_bad_merge_unclosed_boundary():
    """A bad merge with a missing wall or gap between endpoints is flagged."""
    world = _build_valid_room_world()
    # Break West wall endpoint so it leaves a 1.0m gap
    world.entities["w-west"].custom_properties["end_point"] = [0.0, 3.0, 0.0]

    report = revalidate_merged_topology(world)
    assert report.is_valid is False
    assert report.error_count >= 1

    issue_codes = [i.code for i in report.issues]
    assert "unclosed_room_boundary" in issue_codes
    assert "w-west" in [eid for i in report.issues for eid in i.element_ids]


def test_bad_merge_penetrating_walls():
    """A bad merge that causes walls to intersect / penetrate is flagged."""
    world = _build_valid_room_world()
    # Add an intersecting partition diagonal from (-1, 2) to (5, 2) crossing East and West walls
    world.entities["w-bad-diag"] = Entity(
        id="w-bad-diag",
        type=EntityType.WALL,
        confidence=0.8,
        provenance=Provenance.INFERRED,
        custom_properties={
            "start_point": [-1.0, 2.0, 0.0],
            "end_point": [5.0, 2.0, 0.0],
            "bounds_min": [-1.0, 2.0, 0.0],
            "bounds_max": [5.0, 2.0, 2.5],
            "room_id": "room-01",
        },
    )
    world.entities["room-01"].custom_properties["boundary_element_ids"].append("w-bad-diag")

    report = revalidate_merged_topology(world)
    assert report.is_valid is False
    issue_codes = [i.code for i in report.issues]
    assert "penetrating_walls" in issue_codes


def test_bad_merge_opening_exceeds_host_wall():
    """An over-sized opening exceeding host wall dimensions is flagged."""
    world = _build_valid_room_world()
    # Make door width 6.0m on a 4.0m wall
    world.entities["door-01"].custom_properties["width_m"] = 6.0

    report = revalidate_merged_topology(world)
    assert report.is_valid is False
    issue_codes = [i.code for i in report.issues]
    assert "opening_exceeds_host_wall" in issue_codes


def test_bad_merge_elevation_step_without_stair():
    """Incompatible floor elevation across shared partition without stair is flagged."""
    world = _build_valid_room_world()

    # Add second adjacent room sharing w-north, but with a 0.75m elevation step
    world.entities["room-02"] = Entity(
        id="room-02",
        name="Upper Loft",
        type=EntityType.ROOM,
        confidence=0.9,
        provenance=Provenance.INFERRED,
        custom_properties={
            "boundary_element_ids": ["w-north", "w2-east", "w2-north", "w2-west"],
            "floor_elevation_m": 0.75,  # 0.75m > 0.30m threshold
        },
    )

    report = revalidate_merged_topology(world)
    assert report.is_valid is False
    issue_codes = [i.code for i in report.issues]
    assert "elevation_step_without_stair" in issue_codes
