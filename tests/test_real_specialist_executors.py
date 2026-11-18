"""Unit and integration tests for Real Specialist Refinement Executors (Cycle 2 Phase 1).

Covers:
- REDETECT_CROP: Pinhole camera projection, 2D ROI bounding box calculation, image crop extraction, honest provenance.
- SEGMENT: Real binary mask contour extraction, polygon area calculation, honest visual vs non-visual differentiation.
- REASSOCIATE: Same-camera mutual exclusion, appearance cosine similarity (DINOv2), multi-view geometric consistency, calibrated confidence.
- MORE_VIEWS: Active viewpoint planning, observation angle analysis, expected information gain, room boundary clamping (>= 0.3m wall margin).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.quality.autonomous_correction_loop import RefinementKind, RefinementTask
from perception.specialists.more_views import compute_viewpoint_guidance, more_views_specialist
from perception.specialists.reassociate import (
    check_same_camera_exclusion,
    compute_appearance_similarity,
    compute_geometric_consistency,
    reassociate_specialist,
)
from perception.specialists.redetect_crop import extract_entity_crop, redetect_crop_specialist
from perception.specialists.segment import extract_mask_contour, segment_specialist
from world_ir import Entity, EntityType, Transform, WorldIR
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


def _create_test_camera(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
    width: int = 640,
    height: int = 480,
    fx: float = 500.0,
    fy: float = 500.0,
) -> PinholeCamera:
    """Helper to create a calibrated pinhole camera."""
    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )
    # Rotation about Z-axis
    rad = math.radians(yaw_deg)
    quat = Quat(w=math.cos(rad / 2.0), x=0.0, y=0.0, z=math.sin(rad / 2.0))
    extrinsics = CameraExtrinsics(
        position=Vec3(position[0], position[1], position[2]),
        rotation=quat,
    )
    return PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)


# ==============================================================================
# Phase 1A: REDETECT_CROP Tests
# ==============================================================================

def test_extract_entity_crop_pinhole_projection():
    """Verify 3D bounding box projects onto camera image and slices ROI array."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:300, 200:400] = [255, 128, 64]  # Colored patch

    cam = _create_test_camera(position=(0.0, 0.0, 0.0))
    # Entity directly in front of camera along optical axis Z=2.0m
    entity_pos = Vec3(0.0, 0.0, 2.0)
    entity_extent = (0.4, 0.4, 0.4)

    crop, bbox = extract_entity_crop(img, cam, entity_pos, entity_extent, margin_ratio=0.1)
    assert crop is not None
    assert bbox is not None
    u0, v0, u1, v1 = bbox
    assert 0 <= u0 < u1 <= 640
    assert 0 <= v0 < v1 <= 480
    assert crop.shape[0] == (v1 - v0)
    assert crop.shape[1] == (u1 - u0)


def test_redetect_crop_with_real_image_and_honest_provenance():
    """Verify specialist extracts real crop and tags basis='real_visual_crop_inference'."""
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-door-1",
        name="Front Door",
        type=EntityType.DOOR,
        transform={"position": [0.0, 0.0, 2.5]},
        custom_properties={"attributes": {"handle_type": {"value": None, "basis": "unknown"}}},
    )
    world.entities[ent.id] = ent

    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    cam = _create_test_camera(position=(0.0, 0.0, 0.0))

    task = RefinementTask(
        kind=RefinementKind.REDETECT_CROP,
        target="ent-door-1:handle_type",
        expected_gain=0.7,
        cost_s=0.2,
        metadata={
            "entity_id": "ent-door-1",
            "attribute": "handle_type",
            "image": img,
            "camera": cam,
            "evidence_id": "ev-photo-001",
        },
    )

    result = redetect_crop_specialist(task, world)
    assert result is True
    assert ent.extra_fields["redetected_crop"] is True
    assert "crop_metadata" in ent.extra_fields
    assert ent.extra_fields["crop_metadata"]["is_real_crop"] is True
    assert ent.extra_fields["crop_metadata"]["source_evidence_id"] == "ev-photo-001"

    attr_info = ent.extra_fields["attributes"]["handle_type"]
    assert attr_info["value"] == "lever"
    assert attr_info["basis"] == "real_visual_crop_inference"
    assert attr_info["p"] >= 0.90


def test_redetect_crop_fallback_honest_unverified_prior():
    """When no real image/camera exists, specialist must record 'heuristic_prior_unverified'."""
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-door-2",
        name="Side Door",
        type=EntityType.DOOR,
        custom_properties={"attributes": {"lock_type": {"value": None, "basis": "unknown"}}},
    )
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.REDETECT_CROP,
        target="ent-door-2:lock_type",
        expected_gain=0.5,
        cost_s=0.1,
        metadata={"entity_id": "ent-door-2", "attribute": "lock_type"},
    )

    result = redetect_crop_specialist(task, world)
    assert result is True
    attr_info = ent.extra_fields["attributes"]["lock_type"]
    assert attr_info["basis"] == "heuristic_prior_unverified"
    assert "crop_metadata" not in ent.extra_fields


# ==============================================================================
# Phase 1B: SEGMENT Tests
# ==============================================================================

def test_extract_mask_contour_accuracy():
    """Verify 2D contour polygon extraction and area from a rectangular binary mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True  # 40x40 square = 1600 px

    pts, area = extract_mask_contour(mask)
    assert len(pts) >= 4
    # Bounding extents of contour points should match mask coordinates
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert min(xs) >= 29.0 and max(xs) <= 70.0
    assert min(ys) >= 19.0 and max(ys) <= 60.0
    assert 1400.0 <= area <= 1700.0


def test_segment_specialist_real_mask_contour():
    """Verify segment specialist processes real binary mask and tags is_visual_mask=True."""
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(id="ent-table", name="Dining Table", type=EntityType.FURNITURE)
    world.entities[ent.id] = ent

    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 40:160] = 255

    task = RefinementTask(
        kind=RefinementKind.SEGMENT,
        target="ent-table",
        expected_gain=0.8,
        cost_s=0.25,
        metadata={"entity_id": "ent-table", "mask": mask, "confidence": 0.96},
    )

    result = segment_specialist(task, world)
    assert result is True
    assert ent.extra_fields["has_mask"] is True
    assert ent.extra_fields["is_visual_mask"] is True

    seg = ent.extra_fields["segmentation"]
    assert seg["is_visual_mask"] is True
    assert seg["backend"] == "real_mask_contour_extraction"
    assert seg["area_px"] > 1000.0
    assert len(seg["contour"]) >= 4

    attr = ent.extra_fields["attributes"]["segmentation"]
    assert attr["basis"] == "real_mask_contour_extraction"
    assert attr["p"] == 0.96


def test_segment_specialist_fallback_honest_non_visual():
    """When no visual mask is available, specialist must honestly tag non-visual approximation."""
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(id="ent-bed", name="Bed", type=EntityType.FURNITURE)
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.SEGMENT,
        target="ent-bed",
        expected_gain=0.4,
        cost_s=0.05,
        metadata={"entity_id": "ent-bed"},
    )

    result = segment_specialist(task, world)
    assert result is True
    assert ent.extra_fields["is_visual_mask"] is False
    seg = ent.extra_fields["segmentation"]
    assert seg["backend"] == "bounding_box_approximation_non_visual"
    assert seg["confidence"] <= 0.50

    attr = ent.extra_fields["attributes"]["segmentation"]
    assert attr["basis"] == "bounding_box_approximation_non_visual"
    assert attr["value"] == "bounding_box_approximation"


# ==============================================================================
# Phase 1C: REASSOCIATE Tests
# ==============================================================================

def test_same_camera_mutual_exclusion():
    """Enforce physical law: two observations in the same camera cannot be merged."""
    # Collision: same evidence_id
    obs_conflict = [
        {"evidence_id": "frame_001", "class_label": "chair"},
        {"evidence_id": "frame_001", "class_label": "stool"},
    ]
    assert check_same_camera_exclusion(obs_conflict) is False

    # Distinct camera sources: valid
    obs_valid = [
        {"evidence_id": "frame_001", "class_label": "chair"},
        {"evidence_id": "frame_002", "class_label": "chair"},
    ]
    assert check_same_camera_exclusion(obs_valid) is True


def test_reassociate_specialist_same_camera_rejection():
    """Specialist must reject merging detections from the same camera frame."""
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(id="ent-chair", name="Chair", type=EntityType.UNKNOWN)
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.REASSOCIATE,
        target="ent-chair",
        expected_gain=0.6,
        cost_s=0.1,
        metadata={
            "entity_id": "ent-chair",
            "observations": [
                {"evidence_id": "cam_view_1", "label": "chair"},
                {"evidence_id": "cam_view_1", "label": "armchair"},
            ],
        },
    )

    result = reassociate_specialist(task, world)
    assert result is False
    assert ent.extra_fields.get("reassociation_error") == "same_camera_mutual_exclusion_violated"


def test_reassociate_multiview_geometric_and_appearance_consensus():
    """Verify multi-view reassociation using DINOv2 cosine similarity and camera geometry."""
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-sofa",
        name="Living Unit",
        type=EntityType.UNKNOWN,
        transform={"position": [0.0, 0.0, 2.0]},
    )
    world.entities[ent.id] = ent

    cam1 = _create_test_camera(position=(-0.5, 0.0, 0.0), yaw_deg=0.0)
    cam2 = _create_test_camera(position=(0.5, 0.0, 0.0), yaw_deg=0.0)

    # Compute actual projections of the entity 3D position onto both cameras
    target_pos = Vec3(0.0, 0.0, 2.0)
    proj1 = cam1.project(target_pos)
    proj2 = cam2.project(target_pos)
    assert proj1 is not None and proj2 is not None

    # Identical normalized feature embeddings -> similarity = 1.0
    desc1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    desc2 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    assert pytest.approx(compute_appearance_similarity(desc1, desc2), rel=1e-3) == 1.0

    task = RefinementTask(
        kind=RefinementKind.REASSOCIATE,
        target="ent-sofa",
        expected_gain=0.8,
        cost_s=0.15,
        metadata={
            "entity_id": "ent-sofa",
            "observations": [
                {
                    "evidence_id": "cam_1",
                    "camera": cam1,
                    "point_2d": proj1,
                    "descriptor": desc1,
                    "label": "sofa",
                    "confidence": 0.92,
                },
                {
                    "evidence_id": "cam_2",
                    "camera": cam2,
                    "point_2d": proj2,
                    "descriptor": desc2,
                    "label": "sofa",
                    "confidence": 0.88,
                },
            ],
            "context_hints": {"room_type": "living_room"},
        },
    )

    result = reassociate_specialist(task, world)
    assert result is True
    assert ent.extra_fields["resolved_class"] == "sofa"
    assert ent.extra_fields["reassociated_by"] == "multi_view_geometric_appearance_consensus"
    assert ent.extra_fields["appearance_similarity"] >= 0.95
    attr = ent.extra_fields["attributes"]["semantic_class"]
    assert attr["basis"] == "multi_view_geometric_appearance_consensus"
    assert attr["p"] >= 0.85


# ==============================================================================
# Phase 1D: MORE_VIEWS Guidance Tests
# ==============================================================================

def test_compute_viewpoint_guidance_unobserved_sector_gain():
    """Verify higher information gain is assigned to unobserved angular sectors."""
    entity_pos = (2.0, 2.0, 0.0)
    # Existing camera at angle 0 deg (pos: 3.5, 2.0)
    existing_cameras = [(3.5, 2.0, 0.0)]

    poses = compute_viewpoint_guidance(
        entity_pos=entity_pos,
        existing_cameras=existing_cameras,
        standoff_dist=1.5,
        num_sectors=4,  # 0, 90, 180, 270 deg
    )

    assert len(poses) == 4
    # Sector 0 has 1 observation -> expected gain = 0.5
    # Sectors 90, 180, 270 have 0 observations -> expected gain = 1.0
    gains = {p["azimuth_deg"]: p["expected_gain"] for p in poses}
    assert gains[0.0] < gains[180.0]
    assert gains[180.0] == 1.0
    assert gains[90.0] == 1.0


def test_compute_viewpoint_guidance_room_boundary_clamping():
    """Verify recommended viewpoints remain strictly within room floor boundaries (>= 0.3m margin)."""
    # Room bounds: x in [0.0, 4.0], y in [0.0, 4.0]
    room_bounds = (0.0, 0.0, 4.0, 4.0)
    wall_margin = 0.3
    # Entity near east wall at x=3.8
    entity_pos = (3.8, 2.0, 0.0)

    # Standoff 1.5m at azimuth 0 deg would place camera at x=5.3 (outside the room)
    poses = compute_viewpoint_guidance(
        entity_pos=entity_pos,
        room_bounds=room_bounds,
        standoff_dist=1.5,
        wall_margin=wall_margin,
        num_sectors=4,
    )

    # Max allowed X inside room is 4.0 - 0.3 = 3.7m
    for p in poses:
        cam_x, cam_y, _ = p["position"]
        assert 0.3 <= cam_x <= 3.7, f"Camera pose {p['position']} violated room X boundary"
        assert 0.3 <= cam_y <= 3.7, f"Camera pose {p['position']} violated room Y boundary"

    # The pose at azimuth 0 should have been clamped
    east_pose = next(p for p in poses if p["azimuth_deg"] == 0.0)
    assert east_pose["clamped_to_room"] is True
    assert east_pose["position"][0] == 3.7
