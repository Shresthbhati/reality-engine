"""Unit tests for Architecture-to-Contract output adapters (Blueprint Task 5.2).

Verifies:
1. Planes, rooms, and openings become Space, Surface, and Opening nodes in the contract hierarchy.
2. Invariant 1 (explicit evidence_state) and Invariant 7 (Opening -> Surface -> Space) are satisfied.
3. Hierarchy tree structure (World -> Building -> Floor -> Space -> Surface -> Opening) has zero structural issues.
4. Glass/glazing is classified as an Opening (window/glazing), not an opaque wall (Scenario A14).
5. Furniture occluding a wall preserves wall continuity as a single surface with OCCLUDED/OBSERVED spans (Scenario A6).
6. Round-trip serialization preserves hierarchy and surface definitions.
"""

from __future__ import annotations

import json
import pytest

from perception.architecture.adapters import (
    adapt_architecture_to_contract,
    classify_glazing_opening,
    evaluate_wall_continuity_under_occlusion,
)
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from world_ir.v2_extensions import Hierarchy, HierarchyNode
from world_ir.validation import validate_world_ir


def _build_test_world() -> WorldIR:
    """Build a synthetic test WorldIR with storey, room, walls, door, window, and furniture."""
    world = WorldIR(id="world-test-01", main_branch_id="branch-main")

    # Storey
    world.entities["storey-01"] = Entity(
        id="storey-01",
        name="Ground Floor",
        type=EntityType.LEVEL,
        confidence=0.95,
        provenance=Provenance.OBSERVED,
        custom_properties={"elevation_m": 0.0, "evidence_state": "OBSERVED"},
    )

    # Room
    world.entities["room-101"] = Entity(
        id="room-101",
        name="Living Room",
        type=EntityType.ROOM,
        confidence=0.90,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "storey_id": "storey-01",
            "boundary_element_ids": ["wall-north", "wall-south", "wall-east", "wall-west", "floor-01", "ceil-01"],
            "evidence_state": "OBSERVED",
        },
    )

    # Walls / Floor / Ceiling (Surfaces)
    surfaces = [
        ("wall-north", EntityType.WALL, "wall", 0.92),
        ("wall-south", EntityType.WALL, "wall", 0.91),
        ("wall-east", EntityType.WALL, "wall", 0.90),
        ("wall-west", EntityType.WALL, "wall", 0.93),
        ("floor-01", EntityType.FLOOR, "floor", 0.98),
        ("ceil-01", EntityType.CEILING, "ceiling", 0.85),
    ]
    for sid, stype, kind, conf in surfaces:
        world.entities[sid] = Entity(
            id=sid,
            name=f"Surface {sid}",
            type=stype,
            confidence=conf,
            provenance=Provenance.OBSERVED,
            custom_properties={
                "room_id": "room-101",
                "kind": kind,
                "evidence_state": "OBSERVED",
            },
        )

    # Openings: Door on north wall, Window on east wall
    world.entities["door-01"] = Entity(
        id="door-01",
        name="Entry Door",
        type=EntityType.DOOR,
        confidence=0.94,
        provenance=Provenance.INFERRED,
        custom_properties={
            "host_wall_id": "wall-north",
            "kind": "door",
            "width_m": 0.9,
            "height_m": 2.1,
            "evidence_state": "OBSERVED",
        },
    )

    world.entities["window-01"] = Entity(
        id="window-01",
        name="Garden Window",
        type=EntityType.WINDOW,
        confidence=0.88,
        provenance=Provenance.INFERRED,
        custom_properties={
            "host_wall_id": "wall-east",
            "kind": "window",
            "width_m": 1.2,
            "height_m": 1.4,
            "sill_height_m": 0.9,
            "evidence_state": "OBSERVED",
        },
    )

    # Object in room: Sofa
    world.entities["sofa-01"] = Entity(
        id="sofa-01",
        name="Living Room Sofa",
        type=EntityType.STRUCTURE,
        confidence=0.85,
        provenance=Provenance.OBSERVED,
        custom_properties={"room_id": "room-101", "evidence_state": "OBSERVED"},
    )

    return world


def test_adapt_architecture_to_contract_hierarchy():
    """Verify that architecture entities adapt into typed Hierarchy nodes adhering to contract."""
    world = _build_test_world()
    res = adapt_architecture_to_contract(world)

    # Building, floor, space, surfaces, openings nodes created
    assert res.building_node_id is not None
    assert len(res.floor_node_ids) == 1
    assert len(res.space_node_ids) == 1
    assert len(res.surface_node_ids) == 6
    assert len(res.opening_node_ids) == 2

    # Check hierarchy tree integrity
    hierarchy = world.hierarchy
    problems = hierarchy.problems(world.entities)
    assert problems == [], f"Hierarchy structural problems: {problems}"

    # Verify building -> floor -> space -> surface -> opening links
    b_node = hierarchy.nodes[res.building_node_id]
    assert b_node.level == "building"
    assert b_node.parent_id is None

    f_node = hierarchy.nodes[res.floor_node_ids[0]]
    assert f_node.level == "floor"
    assert f_node.parent_id == b_node.id
    assert f_node.entity_id == "storey-01"

    s_node = hierarchy.nodes[res.space_node_ids[0]]
    assert s_node.level == "space"
    assert s_node.parent_id == f_node.id
    assert s_node.entity_id == "room-101"
    assert s_node.kind == "room"

    # Every surface hangs off space
    for sf_id in res.surface_node_ids:
        sf_node = hierarchy.nodes[sf_id]
        assert sf_node.level == "surface"
        assert sf_node.parent_id == s_node.id
        assert sf_node.evidence_state in ("OBSERVED", "INFERRED")

    # Every opening hangs off a surface
    for op_id in res.opening_node_ids:
        op_node = hierarchy.nodes[op_id]
        assert op_node.level == "opening"
        host_surface_node = hierarchy.nodes[op_node.parent_id]
        assert host_surface_node.level == "surface"

    # Sofa is placed in the space
    assert hierarchy.entity_space.get("sofa-01") == s_node.id


def test_invariants_1_and_7_enforcement():
    """Verify that adapted world satisfies Invariant 1 and Invariant 7 in validation."""
    world = _build_test_world()
    adapt_architecture_to_contract(world)

    report = validate_world_ir(world)
    # Check that there are no invariant 1 or invariant 7 errors
    inv_issues = [i for i in report.issues if "invariant_1" in i.code or "invariant_7" in i.code]
    assert inv_issues == [], f"Invariant issues found: {inv_issues}"


def test_hierarchy_roundtrip_serialization():
    """Ensure adapted hierarchy and surfaces survive JSON round-trip serialization."""
    world = _build_test_world()
    adapt_architecture_to_contract(world)

    json_str = world.to_json()
    world2 = WorldIR.from_json(json_str)

    assert world2.hierarchy is not None
    assert len(world2.hierarchy.nodes) == len(world.hierarchy.nodes)
    assert world2.hierarchy.entity_space == world.hierarchy.entity_space
    assert set(world2.surfaces.keys()) == set(world.surfaces.keys())


def test_scenario_a14_glazing_vs_opaque_wall():
    """Scenario A14: Glass is an opening, not an opaque wall."""
    # Glass observation
    kind, props = classify_glazing_opening(
        "plane-window-glass",
        is_translucent=True,
        specular_sheen=0.90,
        material_kind="glazing",
        dimensions=(1.5, 1.2),
    )
    assert kind == "window"
    assert props["glazing"] is True
    assert props["is_opaque"] is False
    assert props["evidence_state"] == "OBSERVED"

    # Solid concrete wall
    wall_kind, wall_props = classify_glazing_opening(
        "plane-concrete-wall",
        is_translucent=False,
        specular_sheen=0.05,
        material_kind="concrete",
    )
    assert wall_kind == "wall"
    assert wall_props["glazing"] is False
    assert wall_props["is_opaque"] is True


def test_scenario_a06_wall_continuity_under_occlusion():
    """Scenario A6: Furniture occluding a wall leaves the wall continuous."""
    # 6.0m wall with a wardrobe (2.0m to 4.0m) occluding the middle
    res = evaluate_wall_continuity_under_occlusion(
        wall_span_length_m=6.0,
        occluder_spans=[(2.0, 4.0)],
        wall_id="wall-south",
    )

    assert res.is_continuous is True
    assert len(res.spans) == 3
    # Visible left
    assert res.spans[0].evidence_state == "OBSERVED"
    assert res.spans[0].start_m == 0.0 and res.spans[0].end_m == 2.0
    # Occluded middle behind wardrobe
    assert res.spans[1].evidence_state == "OCCLUDED"
    assert res.spans[1].start_m == 2.0 and res.spans[1].end_m == 4.0
    # Visible right
    assert res.spans[2].evidence_state == "OBSERVED"
    assert res.spans[2].start_m == 4.0 and res.spans[2].end_m == 6.0

    assert res.observed_fraction == pytest.approx(4.0 / 6.0)
    assert res.overall_evidence_state == "PARTIALLY_OBSERVED"
