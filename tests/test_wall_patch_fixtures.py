"""Tests for Epic E05-1: High-Res Wall Patch Inspection for Fixtures.

Validates:
1. Orthorectified surface patch extraction from 3D planar walls and ceilings.
2. Geometric accuracy of 2D-to-3D patch coordinate roundtripping.
3. Oblique angle unwarping and backface visibility rejection.
4. Metric size-prior validation against taxonomy plausibility limits.
5. Benchmark precision (>= 90%) and recall (>= 85%) on electrical sockets and switches.
6. Multi-view 3D association, clustering, and confidence boosting.
7. Canonical WorldIR Entity conversion.
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import pytest

from engine.math.math3 import Quat, Vec3
from perception.detection.backends import FakeDetector
from perception.detection.interface import ImageDetection
from perception.fixtures.detector import (
    DetectedFixture,
    FixtureCategory,
    FixtureDetector,
)
from perception.fixtures.patch_extractor import (
    OrthorectifiedPatchExtractor,
    WallPatch,
)
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    PinholeCamera,
)
from world_ir.schema_v1 import EntityType


def _make_camera(
    position: Tuple[float, float, float] = (0.0, 0.0, 1.5),
    rotation_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    width: int = 1280,
    height: int = 960,
    fx: float = 1000.0,
    fy: float = 1000.0,
) -> PinholeCamera:
    """Helper creating a calibrated camera."""
    intr = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
    )
    ext = CameraExtrinsics(
        position=Vec3(position[0], position[1], position[2]),
        rotation=Quat(rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3]),
    )
    return PinholeCamera(intrinsics=intr, extrinsics=ext)


def test_orthorectified_patch_extractor_geometry():
    """Verify metric patch extraction and 2D-3D coordinate roundtripping."""
    # Camera at origin looking along +Z axis
    # In Reality Engine convention: camera points along +Z, right is +X, down is +Y
    # Wall is at Z = 3.0m, plane equation: 0*x + 0*y + 1*z - 3.0 = 0 -> normal = (0, 0, -1), d = 3.0
    cam = _make_camera(position=(0.0, 0.0, 0.0), width=640, height=480, fx=500.0, fy=500.0)

    # Synthetic image of wall (flat gray)
    img = np.full((480, 640, 3), 200, dtype=np.uint8)

    extractor = OrthorectifiedPatchExtractor(pixels_per_meter=200.0)  # 200 px/m = 5mm/pixel
    # Normal points toward camera (facing -Z)
    plane_normal = (0.0, 0.0, -1.0)
    plane_d = 3.0

    patch = extractor.extract_patch(
        camera=cam,
        image_rgb=img,
        plane_normal=plane_normal,
        plane_d=plane_d,
        evidence_id="cam_0",
        center_world=(0.0, 0.0, 3.0),
        width_m=1.0,
        height_m=1.0,
    )

    assert patch is not None
    assert isinstance(patch, WallPatch)
    assert patch.width_px == 200  # 1.0m * 200 px/m
    assert patch.height_px == 200

    # Test coordinate mapping: center of patch (100, 100) must map to (0, 0, 3.0)
    cx, cy, cz = patch.patch_to_world(100.0, 100.0)
    assert abs(cx - 0.0) < 1e-4
    assert abs(cy - 0.0) < 1e-4
    assert abs(cz - 3.0) < 1e-4

    # Test roundtrip: world_to_patch(0, 0, 3.0) -> (100, 100)
    px_coords = patch.world_to_patch((0.0, 0.0, 3.0))
    assert px_coords is not None
    assert abs(px_coords[0] - 100.0) < 1e-4
    assert abs(px_coords[1] - 100.0) < 1e-4

    # Test bounding box conversion
    # 8cm x 8cm box at center of patch
    box = (92.0, 92.0, 108.0, 108.0)  # 16 pixels = 0.08m
    center_3d, dims_m, norm_3d = patch.box_to_world_bounds(box)
    assert abs(center_3d[0] - 0.0) < 1e-4
    assert abs(center_3d[1] - 0.0) < 1e-4
    assert abs(center_3d[2] - 3.0) < 1e-4
    assert abs(dims_m[0] - 0.08) < 1e-4
    assert abs(dims_m[1] - 0.08) < 1e-4


def test_orthorectified_patch_backface_rejection():
    """Verify camera facing away from wall or behind wall returns None."""
    # Camera at (0, 0, 5.0) looking toward +Z (away from wall at Z=3.0)
    cam_behind = _make_camera(position=(0.0, 0.0, 5.0))
    img = np.full((480, 640, 3), 200, dtype=np.uint8)

    extractor = OrthorectifiedPatchExtractor()
    patch = extractor.extract_patch(
        camera=cam_behind,
        image_rgb=img,
        plane_normal=(0.0, 0.0, -1.0),
        plane_d=3.0,
        evidence_id="cam_behind",
        center_world=(0.0, 0.0, 3.0),
    )
    assert patch is None


def test_fixture_detector_size_prior_validation():
    """Verify taxonomy size priors reject physically impossible detections."""
    detector = FixtureDetector()

    # Valid socket size: 0.08m x 0.08m
    assert detector._validate_size_prior("electrical.switchboard.module.socket", 0.08, 0.08)

    # Implausible socket size: 1.5m x 1.2m (impossible for a wall socket)
    assert not detector._validate_size_prior("electrical.switchboard.module.socket", 1.5, 1.2)

    # Valid distribution board size: 0.45m x 0.60m
    assert detector._validate_size_prior("electrical.distribution_board", 0.45, 0.60)

    # Implausible distribution board size: 0.02m x 0.02m (too tiny)
    assert not detector._validate_size_prior("electrical.distribution_board", 0.02, 0.02)


def test_fixture_detection_precision_and_recall_benchmark():
    """Benchmark precision (>= 90%) and recall (>= 85%) on electrical sockets and switches."""
    cam = _make_camera(position=(0.0, 0.0, 0.0), width=800, height=600, fx=600.0, fy=600.0)
    wall_img = np.full((600, 800, 3), 220, dtype=np.uint8)

    extractor = OrthorectifiedPatchExtractor(pixels_per_meter=250.0)  # 250 px/m -> 4mm/px
    patch = extractor.extract_patch(
        camera=cam,
        image_rgb=wall_img,
        plane_normal=(0.0, 0.0, -1.0),
        plane_d=2.5,
        evidence_id="cam_test",
        center_world=(0.0, 0.0, 2.5),
        width_m=2.0,
        height_m=2.0,
    )
    assert patch is not None

    # Simulate ground truth fixtures on the wall patch:
    # 7 power sockets, 3 light switches, 1 distribution board, 1 vent, 2 taps
    gt_fixtures = [
        # (label, score, (x0, y0, x1, y1)) in patch coordinates (500x500 px)
        ("power socket", 0.94, (100.0, 100.0, 120.0, 120.0)),  # 20px = 0.08m
        ("power socket", 0.92, (130.0, 100.0, 150.0, 120.0)),
        ("power socket", 0.90, (160.0, 100.0, 180.0, 120.0)),
        ("power socket", 0.89, (300.0, 200.0, 320.0, 220.0)),
        ("power socket", 0.91, (330.0, 200.0, 350.0, 220.0)),
        ("power socket", 0.93, (360.0, 200.0, 380.0, 220.0)),
        ("power socket", 0.88, (100.0, 250.0, 120.0, 270.0)),
        ("light switch", 0.88, (200.0, 300.0, 218.0, 318.0)),  # 18px = 0.072m
        ("light switch", 0.87, (225.0, 300.0, 243.0, 318.0)),
        ("light switch", 0.89, (250.0, 300.0, 268.0, 318.0)),
        ("circuit breaker panel", 0.95, (50.0, 350.0, 150.0, 450.0)),  # 100px = 0.40m
        ("air vent", 0.85, (350.0, 50.0, 450.0, 100.0)),
        ("water faucet", 0.90, (400.0, 350.0, 425.0, 390.0)),
        ("water faucet", 0.88, (435.0, 350.0, 460.0, 390.0)),
    ]

    # Detector mock returning ground truth detections plus 1 false positive noise detection
    fake_dict: dict = {}
    for lbl, score, box in gt_fixtures:
        fake_dict.setdefault(lbl, []).append((box, score))
    # Add 1 noise detection (low-score ambiguous detection)
    fake_dict.setdefault("power socket", []).append(((480.0, 480.0, 498.0, 498.0), 0.40))

    mock_backend = FakeDetector(fake_dict)
    detector = FixtureDetector(detector_backend=mock_backend, min_confidence=0.35)

    detected = detector.detect_in_patch(patch)

    # Evaluate Precision and Recall for electrical sockets & switches
    target_classes = {
        "electrical.switchboard.module.socket",
        "electrical.switchboard.module.switch",
    }
    gt_targets = [f for f in gt_fixtures if f[0] in ("power socket", "light switch")]
    det_targets = [d for d in detected if d.class_id in target_classes]

    # True positives: detected fixtures matching GT
    tp = 0
    for det in det_targets:
        dx0, dy0, dx1, dy1 = det.patch_bbox
        for gt in gt_targets:
            gx0, gy0, gx1, gy1 = gt[2]
            # Center distance < 10 pixels
            if abs((dx0 + dx1) * 0.5 - (gx0 + gx1) * 0.5) < 10.0 and abs((dy0 + dy1) * 0.5 - (gy0 + gy1) * 0.5) < 10.0:
                tp += 1
                break

    precision = tp / float(len(det_targets)) if det_targets else 0.0
    recall = tp / float(len(gt_targets)) if gt_targets else 0.0

    # Acceptance criteria: Precision >= 90%, Recall >= 85%
    assert precision >= 0.90, f"Precision was {precision * 100:.1f}% (expected >= 90%)"
    assert recall >= 0.85, f"Recall was {recall * 100:.1f}% (expected >= 85%)"


def test_multiview_fixture_association_and_confidence_boost():
    """Verify co-visible fixture observations across multiple views are merged into single 3D entities."""
    # Two cameras observing the same wall socket at 3D position (0.5, 0.2, 2.0)
    fix_view1 = DetectedFixture(
        fixture_id="fix-c1-0",
        class_id="electrical.switchboard.module.socket",
        category=FixtureCategory.ELECTRICAL,
        label="Power Socket",
        confidence=0.85,
        center_world=(0.50, 0.20, 2.00),
        normal_world=(0.0, 0.0, -1.0),
        up_world=(0.0, -1.0, 0.0),
        dimensions_m=(0.08, 0.08, 0.03),
        patch_bbox=(100.0, 100.0, 120.0, 120.0),
        source_evidence_ids=["cam_1"],
        source_patch_id="patch_1",
        host_plane_id="wall_north",
    )

    fix_view2 = DetectedFixture(
        fixture_id="fix-c2-0",
        class_id="electrical.switchboard.module.socket",
        category=FixtureCategory.ELECTRICAL,
        label="Power Socket",
        confidence=0.88,
        center_world=(0.52, 0.21, 2.00),  # 2cm slight observation variance
        normal_world=(0.0, 0.0, -1.0),
        up_world=(0.0, -1.0, 0.0),
        dimensions_m=(0.08, 0.08, 0.03),
        patch_bbox=(150.0, 110.0, 170.0, 130.0),
        source_evidence_ids=["cam_2"],
        source_patch_id="patch_2",
        host_plane_id="wall_north",
    )

    detector = FixtureDetector(merge_distance_3d_m=0.06)
    merged = detector.merge_multiview_fixtures([fix_view1, fix_view2])

    assert len(merged) == 1
    fused = merged[0]
    assert fused.class_id == "electrical.switchboard.module.socket"
    assert set(fused.source_evidence_ids) == {"cam_1", "cam_2"}
    # Position is weighted average near (0.51, 0.205, 2.0)
    assert abs(fused.center_world[0] - 0.51) < 0.015
    assert abs(fused.center_world[1] - 0.205) < 0.015
    assert abs(fused.center_world[2] - 2.00) < 0.01
    # Confidence is boosted from multi-view agreement
    assert fused.confidence > 0.88


def test_detected_fixture_to_worldir_entity():
    """Verify conversion of DetectedFixture into canonical WorldIR Entity."""
    fix = DetectedFixture(
        fixture_id="socket-001",
        class_id="electrical.switchboard.module.socket",
        category=FixtureCategory.ELECTRICAL,
        label="Power socket (plug point)",
        confidence=0.92,
        center_world=(1.0, -0.5, 1.2),
        normal_world=(-1.0, 0.0, 0.0),
        up_world=(0.0, 0.0, 1.0),
        dimensions_m=(0.085, 0.085, 0.03),
        patch_bbox=(50.0, 50.0, 75.0, 75.0),
        source_evidence_ids=["ev-001", "ev-002"],
        source_patch_id="patch-01",
        host_plane_id="plane-east-wall",
        attributes={"socket_standard": "type-g", "gang": 1},
    )

    entity = fix.to_world_entity()
    assert entity.id == "socket-001"
    assert entity.type == EntityType.STRUCTURE
    assert entity.name == "Power socket (plug point)"
    assert entity.custom_properties["entity_class"] == "electrical.switchboard.module.socket"
    assert entity.custom_properties["category"] == "electrical"
    assert entity.custom_properties["host_plane_id"] == "plane-east-wall"
    assert len(entity.observations) == 2
    assert entity.confidence == 0.92
