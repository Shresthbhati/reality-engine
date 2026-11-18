"""Tests for the Automated Measured-Room Benchmark Evaluator."""

from __future__ import annotations

import pytest

from benchmarks.measured_rooms.evaluator import (
    DEV_ROOMS,
    HOLDOUT_ROOMS,
    evaluate_room_benchmark,
    load_room_ground_truth,
)
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)


def test_holdout_room_gating_enforces_blind_evaluation():
    """Rule 10: Never inspect blind holdout ground truth during development tuning."""
    for holdout in HOLDOUT_ROOMS:
        with pytest.raises(PermissionError) as exc_info:
            load_room_ground_truth(holdout, allow_holdout=False)
        assert "blind holdout" in str(exc_info.value).lower()

    # Explicit authorization succeeds
    for holdout in HOLDOUT_ROOMS:
        gt = load_room_ground_truth(holdout, allow_holdout=True)
        assert gt["room"]["id"] in ("room-05", "room-06")


def test_development_rooms_load_without_holdout_gate():
    for dev_room in DEV_ROOMS:
        gt = load_room_ground_truth(dev_room, allow_holdout=False)
        assert "room" in gt
        assert "walls" in gt
        assert "openings" in gt


def test_evaluator_scores_perfect_world_with_passing_verdict():
    gt = load_room_ground_truth("room_01_living_studio")

    world = WorldIR(id="world-test-room-01", name="Perfect Dev Room 1")

    # Add room entity
    r_geom = Geometry(
        id="geom-room-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.2, 4.1, 2.65),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.99,
    )
    world.geometries[r_geom.id] = r_geom
    r_ent = Entity(
        id="room-01",
        name="Living Studio",
        type=EntityType.ROOM,
        geometry_ids=[r_geom.id],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.99,
        custom_properties={
            "camera_stations": ["cam_01", "cam_02", "cam_03"],
            "height_m": 2.65,
            "floor_area_m2": 21.32,
            "floor_dimension_u_m": 5.2,
            "floor_dimension_v_m": 4.1,
        },
    )
    world.entities[r_ent.id] = r_ent

    # Add walls
    for w in gt["walls"]:
        wid = w["id"]
        start = w["start"]
        end = w["end"]
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
        # give wall slight 15cm thickness
        g = Geometry(
            id=f"geom-{wid}",
            type=GeometryType.BOX,
            bounds_min=Vector3(min_x, min_y, 0.0),
            bounds_max=Vector3(max_x + 0.15, max_y + 0.15, 2.65),
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.95,
        )
        world.geometries[g.id] = g
        e = Entity(
            id=wid,
            name=wid,
            type=EntityType.WALL,
            geometry_ids=[g.id],
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.95,
            custom_properties={"length_m": float(w["length_m"])},
        )
        world.entities[e.id] = e
        e.custom_properties["position"] = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, 0.0)

    # Add openings (doors and windows)
    for op in gt["openings"]:
        op_id = op["id"]
        kind = op["kind"]
        ent_type = EntityType.DOOR if kind == "door" else EntityType.WINDOW
        c = op["center"]
        w_m, h_m = op["width_m"], op["height_m"]
        g = Geometry(
            id=f"geom-{op_id}",
            type=GeometryType.BOX,
            bounds_min=Vector3(c[0] - w_m / 2, c[1] - 0.05, c[2] - h_m / 2),
            bounds_max=Vector3(c[0] + w_m / 2, c[1] + 0.05, c[2] + h_m / 2),
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.90,
        )
        world.geometries[g.id] = g
        e = Entity(
            id=op_id,
            name=op_id,
            type=ent_type,
            geometry_ids=[g.id],
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.90,
            custom_properties={
                "width_m": float(w_m),
                "height_m": float(h_m),
                "position": c,
                "position_m": c,
            },
        )
        world.entities[e.id] = e
        e.custom_properties["position"] = tuple(c)

    # Add fixtures
    for fix in gt.get("fixtures", []):
        fid = fix["id"]
        pos = fix["position_m"]
        g = Geometry(
            id=f"geom-{fid}",
            type=GeometryType.BOX,
            bounds_min=Vector3(pos[0] - 0.04, pos[1] - 0.02, pos[2] - 0.04),
            bounds_max=Vector3(pos[0] + 0.04, pos[1] + 0.02, pos[2] + 0.04),
            provenance=Provenance.INFERRED,
            confidence=0.88,
        )
        world.geometries[g.id] = g
        e = Entity(
            id=fid,
            name=fid,
            type=EntityType.STRUCTURE,
            semantic_labels=[f"electrical.{fix['fixture_type']}"],
            geometry_ids=[g.id],
            provenance=Provenance.INFERRED,
            confidence=0.88,
            custom_properties={"position_m": pos},
        )
        world.entities[e.id] = e
        e.custom_properties["position"] = tuple(pos)

    metrics = evaluate_room_benchmark(world, gt)

    assert metrics.meets_tolerances is True
    assert metrics.room_id == "room-01"
    assert metrics.wall_error_pct <= 1.0
    assert metrics.ceiling_height_error_m <= 0.02
    assert metrics.topology_precision == 1.0
    assert metrics.topology_recall == 1.0
    assert metrics.same_camera_exclusion_adherence == 1.0
    assert metrics.failed_checks == 0
    assert metrics.passed_checks == metrics.total_checks


def test_evaluator_detects_dimensional_violations():
    gt = load_room_ground_truth("room_01_living_studio")

    world = WorldIR(id="world-test-violating", name="Violating Dev Room 1")

    # Add room entity with height 3.10m instead of 2.65m (45cm violation)
    r_geom = Geometry(
        id="geom-room-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.2, 4.1, 3.10),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.99,
    )
    world.geometries[r_geom.id] = r_geom
    r_ent = Entity(
        id="room-01",
        name="Living Studio",
        type=EntityType.ROOM,
        geometry_ids=[r_geom.id],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.99,
        custom_properties={"height_m": 3.10},
    )
    world.entities[r_ent.id] = r_ent

    # Wall south with 5.6m instead of 5.2m (40cm violation)
    g = Geometry(
        id="geom-wall-south",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.6, 0.15, 3.10),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )
    world.geometries[g.id] = g
    e = Entity(
        id="wall-south",
        name="wall-south",
        type=EntityType.WALL,
        geometry_ids=[g.id],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )
    world.entities[e.id] = e

    metrics = evaluate_room_benchmark(world, gt)

    assert metrics.meets_tolerances is False
    assert metrics.failed_checks > 0
    assert metrics.ceiling_height_error_m >= 0.40
    # Missing other walls
    assert len(metrics.missing_elements) > 0
