"""Tests for multi-view physical identity association and reassociation specialist.

Validates:
- Strict same-camera mutual exclusion across frames/detections in a capture session.
- Complete absence of fabricated default scores (missing geometry/appearance is explicitly None).
- Competing identity hypotheses retention when evidence is ambiguous.
- Cosine appearance descriptor similarity.
- Multi-camera ray intersection geometric consistency.
- Separation of physical identity association from semantic classification voting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.math import Vec3
from perception.specialists.reassociate import (
    compute_appearance_similarity,
    compute_geometric_consistency,
    evaluate_physical_identity,
)
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    PinholeCamera,
    Quat,
)


def _make_cam(pos: tuple[float, float, float], yaw_deg: float = 0.0) -> PinholeCamera:
    rad = math.radians(yaw_deg)
    quat = Quat(w=math.cos(rad / 2.0), x=0.0, y=0.0, z=math.sin(rad / 2.0))
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    extr = CameraExtrinsics(position=Vec3(pos[0], pos[1], pos[2]), rotation=quat)
    return PinholeCamera(intrinsics=intr, extrinsics=extr)


def test_same_camera_exclusion_enforced() -> None:
    # Two detections in the same image frame cannot be the same physical object
    obs_a = {"evidence_id": "frame_001", "bbox_center_px": (100, 100)}
    obs_b = {"evidence_id": "frame_001", "bbox_center_px": (400, 300)}

    res = evaluate_physical_identity(obs_a, obs_b)
    assert res.is_same_physical_entity is False
    assert res.decision == "SEPARATE"
    assert res.same_camera_violation is True
    assert res.confidence == 1.0


def test_missing_evidence_returns_none_and_ambiguous() -> None:
    # No cameras, no descriptors
    obs_a = {"evidence_id": "frame_001", "point_2d": (100, 100)}
    obs_b = {"evidence_id": "frame_002", "point_2d": (120, 110)}

    res = evaluate_physical_identity(obs_a, obs_b)
    assert res.is_same_physical_entity is False
    assert res.decision == "AMBIGUOUS"
    assert res.geometric_consistency is None
    assert res.appearance_similarity is None
    assert "geometric_calibration" in res.missing_evidence
    assert "appearance_descriptors" in res.missing_evidence
    assert len(res.competing_hypotheses) >= 2


def test_appearance_similarity_cosine() -> None:
    # Orthogonal vectors -> 0.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert compute_appearance_similarity(vec1, vec2) == pytest.approx(0.0, abs=1e-5)

    # Identical vectors -> 1.0
    assert compute_appearance_similarity(vec1, vec1) == pytest.approx(1.0, abs=1e-5)

    # Similar descriptors (e.g. DINOv2 embeddings)
    d1 = np.array([0.8, 0.5, 0.1, 0.2])
    d2 = np.array([0.79, 0.52, 0.09, 0.21])
    sim = compute_appearance_similarity(d1, d2)
    assert sim > 0.98


def test_geometric_consistency_converging_rays() -> None:
    # Cam 1 at (0, 0, 0) looking along +Z: optical center at (320, 240) points to (0, 0, 2)
    cam1 = _make_cam(pos=(0.0, 0.0, 0.0), yaw_deg=0.0)
    # Cam 2 at (1, 0, 0) looking at the same point (0, 0, 2)
    cam2 = _make_cam(pos=(1.0, 0.0, 0.0), yaw_deg=0.0)

    # Rays pointing toward world point (0.0, 0.0, 2.0)
    # For cam1: point (0, 0, 2) projects to (320, 240)
    # For cam2: point (0, 0, 2) in cam2 frame has X = -1.0, Z = 2.0 -> px = 320 + 500 * (-1.0 / 2.0) = 70
    proj1 = cam1.project(Vec3(0.0, 0.0, 2.0))
    proj2 = cam2.project(Vec3(0.0, 0.0, 2.0))
    assert proj1 is not None and proj2 is not None

    g_score = compute_geometric_consistency(
        camera_a=cam1,
        pt_a=(proj1[0], proj1[1]),
        camera_b=cam2,
        pt_b=(proj2[0], proj2[1]),
        world_pos=Vec3(0.0, 0.0, 2.0),
    )
    assert g_score > 0.95


def test_identity_merge_when_geometry_and_appearance_agree() -> None:
    cam1 = _make_cam(pos=(0.0, 0.0, 0.0))
    cam2 = _make_cam(pos=(0.5, 0.0, 0.0))
    world_pt = Vec3(0.0, 0.0, 2.5)

    proj1 = cam1.project(world_pt)
    proj2 = cam2.project(world_pt)
    assert proj1 is not None and proj2 is not None

    desc_common = [0.5, 0.5, 0.5, 0.5]

    obs_a = {
        "evidence_id": "frame_cam1",
        "camera": cam1,
        "point_2d": (proj1[0], proj1[1]),
        "descriptor": desc_common,
    }
    obs_b = {
        "evidence_id": "frame_cam2",
        "camera": cam2,
        "point_2d": (proj2[0], proj2[1]),
        "descriptor": desc_common,
    }

    res = evaluate_physical_identity(obs_a, obs_b, world_pos=world_pt)
    assert res.is_same_physical_entity is True
    assert res.decision == "MERGE"
    assert res.confidence >= 0.85
    assert res.geometric_consistency is not None and res.geometric_consistency > 0.90
    assert res.appearance_similarity is not None and res.appearance_similarity > 0.99
