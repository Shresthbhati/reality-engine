"""Unit tests for Detail-Aware Adaptive Inspection Zoom (Wave 2, Block 16)."""

from dataclasses import dataclass
import numpy as np
import pytest

from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, Provenance
from perception.entities.attributes import MechanismType, MountingType
from perception.detail.adaptive_inspection import (
    AdaptiveInspectionReport,
    FastenerInspection,
    InspectionRefused,
    InspectionViewSelection,
    apply_adaptive_inspection_to_world,
    perform_adaptive_inspection,
    select_inspection_views,
)


@dataclass
class SimpleCamera:
    id: str
    center: list


def _generate_synthetic_switchboard_points(
    width: float = 0.15,
    height: float = 0.08,
    depth: float = 0.01,
    n_points: int = 60,
    seed: int = 42,
) -> list:
    """Generate 3D points representing a flush-mounted switchboard on a wall at z=1.2m."""
    rng = np.random.default_rng(seed)
    pts = []

    # Wall planar points with small noise
    for _ in range(n_points - 4):
        x = rng.uniform(-width / 2, width / 2)
        y = 0.0 + rng.normal(0, 0.0005)  # Wall along Y=0
        z = 1.2 + rng.uniform(-height / 2, height / 2)
        pts.append((x, y, z))

    # Add 2 fastener / screw screwhead points slightly protruding (+2.5mm)
    pts.append((-width / 2 + 0.015, 0.0025, 1.2))
    pts.append((width / 2 - 0.015, 0.0025, 1.2))

    return pts


def test_select_inspection_views():
    cameras = [
        SimpleCamera(id="cam-far", center=[0.0, 5.0, 1.2]),
        SimpleCamera(id="cam-close", center=[0.0, 0.6, 1.2]),
        SimpleCamera(id="cam-glancing", center=[3.0, 0.1, 1.2]),
    ]
    target = (0.0, 0.0, 1.2)
    normal = (0.0, 1.0, 0.0)  # Facing +Y

    selections = select_inspection_views(target, cameras, surface_normal=normal)
    assert len(selections) == 3

    # Closest perpendicular camera should have highest score
    assert selections[0].camera_id == "cam-close"
    assert selections[0].distance_m < 1.0
    assert selections[0].incidence_angle_deg < 5.0
    assert selections[0].effective_gsd_mm < 1.0


def test_perform_adaptive_inspection_success():
    pts = _generate_synthetic_switchboard_points()

    ent = Entity(
        id="ent-switch-1",
        type=EntityType.STRUCTURE,
        name="Electrical Switchboard",
        confidence=0.75,
        custom_properties={"entity_class": "electrical.switchboard"},
    )

    cameras = [SimpleCamera(id="cam-close", center=[0.0, 0.5, 1.2])]
    report = perform_adaptive_inspection(ent, pts, cameras=cameras, surface_normal=(0.0, 1.0, 0.0))

    assert isinstance(report, AdaptiveInspectionReport)
    assert report.entity_id == "ent-switch-1"
    assert report.detail_level in ("L3", "L4")
    assert report.rms_surface_residual_mm < 5.0
    assert len(report.selected_views) == 1

    # Check fasteners detected
    assert len(report.fasteners) >= 1
    assert report.fasteners[0].fastener_type == "screw"

    # Check inferred attributes
    assert report.inferred_attributes is not None
    assert report.inferred_attributes.mounting == MountingType.WALL_MOUNTED
    assert report.inferred_attributes.mechanism == MechanismType.ROCKER


def test_apply_adaptive_inspection_to_world():
    world = WorldIR()
    ent = Entity(
        id="ent-switch-1",
        type=EntityType.STRUCTURE,
        name="Electrical Switchboard",
        confidence=0.75,
        custom_properties={"entity_class": "electrical.switchboard"},
    )
    world.entities[ent.id] = ent

    pts = _generate_synthetic_switchboard_points()
    report = perform_adaptive_inspection(ent, pts)

    apply_adaptive_inspection_to_world(report, world, "ent-switch-1")

    updated = world.entities["ent-switch-1"]
    assert "adaptive_inspection" in updated.custom_properties
    assert "lod_detail_level" in updated.custom_properties
    assert updated.custom_properties["lod_detail_level"] in ("L3", "L4")
    assert "fine_attributes" in updated.custom_properties


def test_perform_adaptive_inspection_refusal():
    ent = Entity(id="ent-empty", type=EntityType.STRUCTURE)
    sparse_pts = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.0)]
    with pytest.raises(InspectionRefused):
        perform_adaptive_inspection(ent, sparse_pts)
