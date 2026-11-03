"""Tests for Reconstruction Completeness and Coverage Heatmap Engine (Wave 4, Block 32)."""

from __future__ import annotations

import pytest

from planning.completeness import (
    RoomCompletenessReport,
    SceneCompletenessReport,
    evaluate_reconstruction_completeness,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Provenance, Vector3
from world_ir.world_v1 import WorldIR


def test_complete_room_evaluation():
    world = WorldIR()

    # Room
    room = Entity(id="rm-1", name="Master Bedroom", type=EntityType.ROOM, custom_properties={"area_m2": 20.0})
    world.entities[room.id] = room

    # Fully observed Floor
    fl_geom = Geometry(id="g-fl", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=0, z=0), bounds_max=Vector3(x=5, y=0.1, z=4))
    fl = Entity(id="ent-fl", name="Floor", type=EntityType.FLOOR, geometry_ids=["g-fl"], provenance=Provenance.OBSERVED,
                custom_properties={"room_id": "rm-1"})
    world.geometries["g-fl"] = fl_geom
    world.entities["ent-fl"] = fl

    # Fully observed Ceiling
    cl = Entity(id="ent-cl", name="Ceiling", type=EntityType.CEILING, provenance=Provenance.OBSERVED,
                custom_properties={"room_id": "rm-1"})
    world.entities["ent-cl"] = cl

    # 4 Fully observed Walls
    for i in range(4):
        w = Entity(
            id=f"ent-w-{i}",
            name=f"Wall {i}",
            type=EntityType.WALL,
            provenance=Provenance.OBSERVED,
            custom_properties={"room_id": "rm-1", "observed_fraction": 1.0, "dimensions_m": [4.5, 2.7, 0.15]},
        )
        world.entities[w.id] = w

    report = evaluate_reconstruction_completeness(world)

    assert isinstance(report, SceneCompletenessReport)
    assert report.total_spaces == 1
    assert report.total_area_m2 == pytest.approx(20.0)
    assert report.overall_completeness_pct >= 95.0
    assert len(report.missing_elements) == 0

    rm_rep = report.room_reports[0]
    assert rm_rep.observed_wall_fraction == pytest.approx(1.0)
    assert rm_rep.observed_floor_fraction >= 0.90
    assert rm_rep.observed_ceiling_fraction >= 0.90


def test_incomplete_room_with_unobserved_spans():
    world = WorldIR()

    room = Entity(id="rm-corridor", name="Entry Corridor", type=EntityType.CORRIDOR, custom_properties={"area_m2": 8.0})
    world.entities[room.id] = room

    # Wall 1: observed only 50%, with 1.5m unobserved span
    w1 = Entity(
        id="w-1",
        name="North Wall",
        type=EntityType.WALL,
        provenance=Provenance.INFERRED,
        custom_properties={
            "room_id": "rm-corridor",
            "observed_fraction": 0.50,
            "dimensions_m": [4.0, 2.7, 0.15],
            "unobserved_spans_m": [[2.0, 3.5]],
        },
    )
    # Floor observed
    fl = Entity(id="fl-1", name="Floor", type=EntityType.FLOOR, provenance=Provenance.OBSERVED,
                custom_properties={"room_id": "rm-corridor"})
    world.entities[w1.id] = w1
    world.entities[fl.id] = fl
    # Note: ceiling is missing entirely!

    report = evaluate_reconstruction_completeness(world)

    assert report.overall_completeness_pct < 65.0
    assert len(report.missing_elements) >= 1
    assert any("unobserved wall span" in m for m in report.missing_elements)
    assert any("Ceiling surface unobserved" in m for m in report.missing_elements)

    # Check coverage heatmap
    assert "w-1" in report.coverage_heatmap
    assert report.coverage_heatmap["w-1"] == pytest.approx(0.50)
