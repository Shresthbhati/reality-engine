"""Unit tests for CAD / DXF Floor Plan Ingestion (Wave 3, Block 17)."""

import pytest
from world_ir import WorldIR
from world_ir.schema_v1 import EntityType, RelationshipKind
from evidence.cad_importer import (
    CadFloorPlan,
    CadImportRefused,
    parse_dxf_text,
    promote_cad_floorplan,
)


def _generate_synthetic_dxf(scale_multiplier: float = 1.0) -> str:
    """Generate a minimal valid DXF R12 string representing a rectangular room with a door."""
    lines = [
        "0", "SECTION",
        "2", "ENTITIES",
        # Wall 1: (0, 0) to (5, 0)
        "0", "LINE",
        "8", "WALL",
        "10", f"{0.0 * scale_multiplier:.2f}",
        "20", f"{0.0 * scale_multiplier:.2f}",
        "11", f"{5.0 * scale_multiplier:.2f}",
        "21", f"{0.0 * scale_multiplier:.2f}",
        # Wall 2: (5, 0) to (5, 4)
        "0", "LINE",
        "8", "WALL",
        "10", f"{5.0 * scale_multiplier:.2f}",
        "20", f"{0.0 * scale_multiplier:.2f}",
        "11", f"{5.0 * scale_multiplier:.2f}",
        "21", f"{4.0 * scale_multiplier:.2f}",
        # Wall 3: (5, 4) to (0, 4)
        "0", "LINE",
        "8", "WALL",
        "10", f"{5.0 * scale_multiplier:.2f}",
        "20", f"{4.0 * scale_multiplier:.2f}",
        "11", f"{0.0 * scale_multiplier:.2f}",
        "21", f"{4.0 * scale_multiplier:.2f}",
        # Wall 4: (0, 4) to (0, 0)
        "0", "LINE",
        "8", "WALL",
        "10", f"{0.0 * scale_multiplier:.2f}",
        "20", f"{4.0 * scale_multiplier:.2f}",
        "11", f"{0.0 * scale_multiplier:.2f}",
        "21", f"{0.0 * scale_multiplier:.2f}",
        # Door: (2.0, 0) to (2.9, 0)
        "0", "LINE",
        "8", "DOOR",
        "10", f"{2.0 * scale_multiplier:.2f}",
        "20", f"{0.0 * scale_multiplier:.2f}",
        "11", f"{2.9 * scale_multiplier:.2f}",
        "21", f"{0.0 * scale_multiplier:.2f}",
        # Room text label: "BEDROOM 1" at (2.5, 2.0)
        "0", "TEXT",
        "8", "ROOM",
        "10", f"{2.5 * scale_multiplier:.2f}",
        "20", f"{2.0 * scale_multiplier:.2f}",
        "1", "BEDROOM 1",
        "0", "ENDSEC",
        "0", "EOF",
    ]
    return "\n".join(lines)


def test_parse_dxf_meter_units():
    dxf_str = _generate_synthetic_dxf(scale_multiplier=1.0)
    plan = parse_dxf_text(dxf_str)

    assert isinstance(plan, CadFloorPlan)
    assert len(plan.walls) == 4
    assert len(plan.doors) == 1
    assert len(plan.annotations) == 1
    assert plan.annotations[0].label == "BEDROOM 1"
    assert plan.unit_scale_to_m == 1.0

    # Bounds check
    assert plan.bounds_min == (0.0, 0.0)
    assert plan.bounds_max == (5.0, 4.0)


def test_parse_dxf_millimeter_units_auto_detect():
    # Coordinates in thousands of millimeters (e.g. 5000mm x 4000mm)
    dxf_str = _generate_synthetic_dxf(scale_multiplier=1000.0)
    plan = parse_dxf_text(dxf_str)

    assert plan.unit_scale_to_m == 0.001
    assert plan.bounds_min == (0.0, 0.0)
    assert plan.bounds_max == (5.0, 4.0)


def test_promote_cad_floorplan_to_world():
    dxf_str = _generate_synthetic_dxf(scale_multiplier=1.0)
    plan = parse_dxf_text(dxf_str)

    world = WorldIR()
    created_ids = promote_cad_floorplan(plan, world, nominal_wall_height_m=3.0)

    assert len(created_ids) >= 6  # 1 room + 4 walls + 1 door
    for cid in created_ids:
        assert cid in world.entities

    # Verify Room Entity
    room = [e for e in world.entities.values() if e.type == EntityType.ROOM][0]
    assert "BEDROOM 1" in room.name
    assert room.custom_properties["area_m2"] == pytest.approx(20.0, abs=0.1)
    assert "wall_height_m" in room.custom_properties["measurements"]

    # Verify Wall Entities and relationships
    walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
    assert len(walls) == 4
    for w in walls:
        assert w.custom_properties["room_id"] == room.id
        part_of = [r for r in w.relationships if r.kind == RelationshipKind.PART_OF]
        assert len(part_of) == 1
        assert part_of[0].target_id == room.id


def test_parse_dxf_refusal_empty():
    with pytest.raises(CadImportRefused):
        parse_dxf_text("")


def test_parse_dxf_refusal_no_geometry():
    # Only header without entities
    dxf_str = "0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF"
    with pytest.raises(CadImportRefused):
        parse_dxf_text(dxf_str)
