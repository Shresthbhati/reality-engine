"""Tests for Temporal Multi-Epoch Alignment and Diffing (Epic E07-1)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from world_ir.temporal import (
    TemporalAlignmentResult,
    TemporalChangeKind,
    TemporalDiffReport,
    TemporalEntityChange,
    TemporalEpoch,
    align_temporal_epochs,
    diff_temporal_epochs,
)


def _make_world(world_id: str, entities: list[Entity], geometries: list[Geometry] | None = None) -> WorldIR:
    world = WorldIR(id=world_id)
    for e in entities:
        world.entities[e.id] = e
    if geometries:
        for g in geometries:
            world.geometries[g.id] = g
    return world


def test_temporal_alignment_rigid_registration():
    # 4 static structural walls forming a room
    # Epoch A is canonical frame
    walls_a = [
        Entity(id="wall-n", name="North Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 5.0, "z": 0.0}}),
        Entity(id="wall-s", name="South Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": -5.0, "z": 0.0}}),
        Entity(id="wall-e", name="East Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 5.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-w", name="West Wall", type=EntityType.STRUCTURE, transform={"position": {"x": -5.0, "y": 0.0, "z": 0.0}}),
    ]
    world_a = _make_world("world-epoch-1", walls_a)
    epoch_a = TemporalEpoch(epoch_id="epoch-2025-01", timestamp_iso="2025-01-15T10:00:00Z", world=world_a)

    # Epoch B has global drift: translated by (2.0, 3.0, 0.5)
    t_drift = np.array([2.0, 3.0, 0.5])
    walls_b = [
        Entity(id="wall-n", name="North Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0 + t_drift[0], "y": 5.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        Entity(id="wall-s", name="South Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0 + t_drift[0], "y": -5.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        Entity(id="wall-e", name="East Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 5.0 + t_drift[0], "y": 0.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        Entity(id="wall-w", name="West Wall", type=EntityType.STRUCTURE, transform={"position": {"x": -5.0 + t_drift[0], "y": 0.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
    ]
    world_b = _make_world("world-epoch-2", walls_b)
    epoch_b = TemporalEpoch(epoch_id="epoch-2025-06", timestamp_iso="2025-06-20T10:00:00Z", world=world_b)

    alignment = align_temporal_epochs(epoch_a, epoch_b)
    assert alignment.converged is True
    assert alignment.num_anchors_used == 4
    assert alignment.residual_error_m < 1e-4

    # Translation should negate the drift: p_a = p_b + t => t ~ -t_drift
    assert math.isclose(alignment.translation[0], -t_drift[0], abs_tol=1e-3)
    assert math.isclose(alignment.translation[1], -t_drift[1], abs_tol=1e-3)
    assert math.isclose(alignment.translation[2], -t_drift[2], abs_tol=1e-3)


def test_temporal_furniture_movement_detection_with_threshold():
    # Base walls
    walls = [
        Entity(id="wall-1", name="Wall 1", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-2", name="Wall 2", type=EntityType.STRUCTURE, transform={"position": {"x": 5.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-3", name="Wall 3", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 5.0, "z": 0.0}}),
    ]

    # Chair moved by 25cm (0.25m >= 0.05m -> MOVED)
    chair_a = Entity(id="chair-1", name="Office Chair", type=EntityType.STRUCTURE, transform={"position": {"x": 2.0, "y": 2.0, "z": 0.0}})
    chair_b = Entity(id="chair-1", name="Office Chair", type=EntityType.STRUCTURE, transform={"position": {"x": 2.25, "y": 2.0, "z": 0.0}})

    # Table moved by only 2cm (0.02m < 0.05m -> UNCHANGED, filters noise)
    table_a = Entity(id="table-1", name="Coffee Table", type=EntityType.STRUCTURE, transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}})
    table_b = Entity(id="table-1", name="Coffee Table", type=EntityType.STRUCTURE, transform={"position": {"x": 1.015, "y": 1.01, "z": 0.0}})

    world_a = _make_world("w-1", walls + [chair_a, table_a])
    world_b = _make_world("w-2", walls + [chair_b, table_b])

    epoch_a = TemporalEpoch(epoch_id="ep-1", timestamp_iso="2025-01-01T00:00:00Z", world=world_a)
    epoch_b = TemporalEpoch(epoch_id="ep-2", timestamp_iso="2025-02-01T00:00:00Z", world=world_b)

    report = diff_temporal_epochs(epoch_a, epoch_b, movement_threshold_m=0.05)

    assert len(report.moved) == 1
    moved_chair = report.moved[0]
    assert moved_chair.entity_id == "chair-1"
    assert math.isclose(moved_chair.displacement_m, 0.25, abs_tol=1e-3)
    assert moved_chair.change_kind == TemporalChangeKind.MOVED

    unchanged_ids = [c.entity_id for c in report.unchanged]
    assert "table-1" in unchanged_ids
    assert "wall-1" in unchanged_ids


def test_temporal_addition_and_removal():
    walls = [
        Entity(id="wall-1", name="Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-2", name="Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 4.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-3", name="Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 4.0, "z": 0.0}}),
    ]
    desk_removed = Entity(id="desk-old", name="Old Desk", type=EntityType.STRUCTURE, transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}})
    sofa_added = Entity(id="sofa-new", name="New Sofa", type=EntityType.STRUCTURE, transform={"position": {"x": 2.0, "y": 2.0, "z": 0.0}})

    world_a = _make_world("w-a", walls + [desk_removed])
    world_b = _make_world("w-b", walls + [sofa_added])

    epoch_a = TemporalEpoch(epoch_id="t1", timestamp_iso="2025-01-01T00:00:00Z", world=world_a)
    epoch_b = TemporalEpoch(epoch_id="t2", timestamp_iso="2025-03-01T00:00:00Z", world=world_b)

    report = diff_temporal_epochs(epoch_a, epoch_b)

    assert len(report.removed) == 1
    assert report.removed[0].entity_id == "desk-old"
    assert report.removed[0].change_kind == TemporalChangeKind.REMOVED

    assert len(report.added) == 1
    assert report.added[0].entity_id == "sofa-new"
    assert report.added[0].change_kind == TemporalChangeKind.ADDED


def test_temporal_structural_renovation_and_surface_damage():
    # Base wall with structural alteration (renovation)
    wall_old = Entity(
        id="wall-1",
        name="Living Room Partition",
        type=EntityType.STRUCTURE,
        geometry_ids=["geom-old-wall"],
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"finish": "plaster_white"}
    )
    wall_renovated = Entity(
        id="wall-1",
        name="Living Room Partition",
        type=EntityType.STRUCTURE,
        geometry_ids=["geom-new-wall"],  # Altered geometry
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"finish": "wood_paneling", "renovated": True}
    )

    # Column with surface spalling / cracks (damage)
    column_intact = Entity(
        id="col-1",
        name="Support Column",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 3.0, "y": 0.0, "z": 0.0}},
        custom_properties={"condition": "nominal", "integrity_score": 1.0}
    )
    column_damaged = Entity(
        id="col-1",
        name="Support Column",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 3.0, "y": 0.0, "z": 0.0}},
        custom_properties={"condition": "cracked", "damage_detected": True, "integrity_score": 0.45, "condition_notes": "shear crack observed"}
    )

    # Another static anchor for alignment
    anchor = Entity(id="anchor-1", name="Core", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 5.0, "z": 0.0}})

    world_a = _make_world("w-orig", [wall_old, column_intact, anchor])
    world_b = _make_world("w-alter", [wall_renovated, column_damaged, anchor])

    epoch_a = TemporalEpoch(epoch_id="epoch-baseline", timestamp_iso="2025-01-01T00:00:00Z", world=world_a)
    epoch_b = TemporalEpoch(epoch_id="epoch-inspection", timestamp_iso="2025-09-01T00:00:00Z", world=world_b)

    report = diff_temporal_epochs(epoch_a, epoch_b)

    assert len(report.damaged) == 1
    assert report.damaged[0].entity_id == "col-1"
    assert report.damaged[0].change_kind == TemporalChangeKind.DAMAGED
    assert "shear crack" in report.damaged[0].details

    assert len(report.renovated) == 1
    assert report.renovated[0].entity_id == "wall-1"
    assert report.renovated[0].change_kind == TemporalChangeKind.RENOVATED


def test_temporal_diff_summary_and_serialization():
    walls = [
        Entity(id="w-1", name="Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}),
        Entity(id="w-2", name="Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 5.0, "y": 0.0, "z": 0.0}}),
        Entity(id="w-3", name="Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 5.0, "z": 0.0}}),
    ]
    c1 = Entity(id="chair", name="Chair", type=EntityType.STRUCTURE, transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}})
    c2 = Entity(id="chair", name="Chair", type=EntityType.STRUCTURE, transform={"position": {"x": 1.5, "y": 1.0, "z": 0.0}})

    world_a = _make_world("wa", walls + [c1])
    world_b = _make_world("wb", walls + [c2])

    epoch_a = TemporalEpoch("e1", "2025-01-01T00:00:00Z", world_a)
    epoch_b = TemporalEpoch("e2", "2025-02-01T00:00:00Z", world_b)

    report = diff_temporal_epochs(epoch_a, epoch_b)
    summary = report.summary()

    assert summary["from_epoch_id"] == "e1"
    assert summary["to_epoch_id"] == "e2"
    assert summary["moved_count"] == 1
    assert summary["max_displacement_m"] >= 0.49

    rep_dict = report.to_dict()
    assert "summary" in rep_dict
    assert "alignment" in rep_dict
    assert "changes" in rep_dict
    assert len(rep_dict["changes"]) == 4
