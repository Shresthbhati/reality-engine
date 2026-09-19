"""Tests for cross-view depth consistency (reconstruction/consistency.py).

Multi-view consistency is the reliability gap between per-view depth
output and the fused world: two cameras observing the same physical
point must report compatible depths, and contradictory evidence must be
DETECTED and REPORTED, never averaged away or ignored.

Fixtures are deterministic unit-level scenes (explicitly synthetic per
repo rules); the real-data execution record lives in the ledgers.
"""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from reconstruction.consistency import check_depth_consistency

# 15 points on a plane z=5, spanning x in [-2, 2], y in [-1, 1].
SCENE = [(x, y, 5.0) for x in (-2.0, -1.0, 0.0, 1.0, 2.0) for y in (-1.0, 0.0, 1.0)]


def _camera(position, yaw_deg=0.0, evidence_id="cam"):
    """Camera whose optical axis is +z rotated by yaw about +y, with a
    wide enough FOV (fx=200 on 400x300 -> ~45 deg half-angle) to see the
    whole fixture plane from the fixture positions."""
    half = math.radians(yaw_deg) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=math.sin(half), z=0.0).normalized()
    intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=200.0, cy=150.0, width=400, height=300)
    return PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(position=Vec3(*position), rotation=quat),
    )


def _depth_map(camera, evidence_id, scene=SCENE, bias=0.0, all_nan=False):
    """Ground-truth depth map for `camera` over `scene`: project each
    point, write its true camera-frame depth (+ optional bias). Pixels
    with no point stay NaN -- the honest sparse map."""
    values = [[float("nan")] * camera.intrinsics.width for _ in range(camera.intrinsics.height)]
    if not all_nan:
        for point in scene:
            uv = camera.project(Vec3(*point))
            if uv is None:
                continue
            u, v = uv
            ui, vi = int(u), int(v)
            intr = camera.intrinsics
            if 0 <= ui < intr.width and 0 <= vi < intr.height:
                rel = Vec3(*point) - camera.extrinsics.position
                z = camera.extrinsics.rotation.conjugate().rotate(rel).z
                values[vi][ui] = z + bias
    return DepthMap(
        evidence_id=evidence_id, width=camera.intrinsics.width,
        height=camera.intrinsics.height, values=values, unit="meters",
    )


def _two_camera_fixture(bias_b=0.0):
    """Camera A at the origin looking +z; camera B 6 m to the side,
    yawed toward the plane. Mutual ray angles at the shared points span
    roughly 36-60 degrees, so the angle gate is exercisable."""
    cam_a = _camera((0.0, 0.0, 0.0), yaw_deg=0.0)
    # Yaw that points B's +z axis at the scene center (0, 0, 5) from (6, 0, 0).
    yaw_b = math.degrees(math.atan2(-6.0, 5.0))
    cam_b = _camera((6.0, 0.0, 0.0), yaw_deg=yaw_b)
    maps = [
        _depth_map(cam_a, "view-a"),
        _depth_map(cam_b, "view-b", bias=bias_b),
    ]
    return maps, {"view-a": cam_a, "view-b": cam_b}


def test_consistent_views_report_no_contradictions():
    maps, cameras = _two_camera_fixture()
    report = check_depth_consistency(maps, cameras, SCENE, max_angle_deg=90.0)
    assert report.status == "consistent"
    assert report.pairs_checked > 0
    assert report.contradictions == ()
    assert report.max_relative_difference <= 0.10


def test_biased_view_produces_detected_contradictions():
    # 1.0 m depth bias on view-b (5.0 m true depth -> 20% error): the
    # unprojection of view-b's samples lands ~1 m off the true surface,
    # far beyond the 10% contradiction threshold at this scale.
    maps, cameras = _two_camera_fixture(bias_b=1.0)
    report = check_depth_consistency(maps, cameras, SCENE, max_angle_deg=90.0)
    assert report.status == "contradictions"
    assert len(report.contradictions) > 0
    c = report.contradictions[0]
    assert {c.evidence_a, c.evidence_b} == {"view-a", "view-b"}
    # Both measurements are reported, not just a verdict.
    assert c.depth_a > 0 and c.depth_b > 0
    assert c.relative_difference > 0.10
    assert report.max_relative_difference > 0.10


def test_contradiction_records_are_deterministic():
    maps, cameras = _two_camera_fixture(bias_b=1.0)
    r1 = check_depth_consistency(maps, cameras, SCENE, max_angle_deg=90.0)
    r2 = check_depth_consistency(maps, cameras, SCENE, max_angle_deg=90.0)
    assert r1 == r2
    keys = [(c.evidence_a, c.evidence_b, c.point) for c in r1.contradictions]
    assert keys == sorted(keys)


def test_oblique_views_are_inconclusive_not_consistent():
    """When every shared ray pair is excluded by the angle gates, the
    check must say 'inconclusive' -- baseline geometry cannot support
    the verdict. Near-coincident cameras give near-parallel rays, the
    degenerate no-baseline case the min-angle gate exists for."""
    cam_a = _camera((0.0, 0.0, 0.0))
    cam_b = _camera((0.05, 0.0, 0.0))  # 5 cm apart: mutual angles < 1 deg
    maps = [_depth_map(cam_a, "view-a"), _depth_map(cam_b, "view-b")]
    report = check_depth_consistency(maps, {"view-a": cam_a, "view-b": cam_b}, SCENE)
    assert report.status == "inconclusive"
    assert "angle" in report.reason


def test_disjoint_visibility_is_insufficient_evidence():
    """View-b pointed away from the scene: no co-observation, no claim."""
    cam_a = _camera((0.0, 0.0, 0.0))
    cam_b = _camera((6.0, 0.0, 0.0))  # looking +z, scene is behind/beside it
    scene = [(x, 0.0, 5.0) for x in (-2.0, -1.0, 0.0)]
    maps = [_depth_map(cam_a, "view-a", scene=scene), _depth_map(cam_b, "view-b", scene=scene)]
    report = check_depth_consistency(maps, {"view-a": cam_a, "view-b": cam_b}, scene)
    assert report.status == "insufficient_evidence"
    assert report.pairs_checked == 0


def test_relative_depth_maps_are_refused():
    maps, cameras = _two_camera_fixture()
    relative = DepthMap(
        evidence_id="view-a", width=maps[0].width, height=maps[0].height,
        values=maps[0].values, unit="relative",
    )
    with pytest.raises(ValueError):
        check_depth_consistency([relative, maps[1]], cameras, SCENE)


def test_depth_map_without_camera_is_skipped():
    maps, cameras = _two_camera_fixture()
    orphan = _depth_map(cameras["view-a"], "view-orphan")
    report = check_depth_consistency([maps[0], maps[1], orphan], cameras, SCENE, max_angle_deg=90.0)
    assert report.status == "consistent"
    assert report.views_skipped == ("view-orphan",)


def test_no_scene_points_is_insufficient_not_crash():
    maps, cameras = _two_camera_fixture()
    report = check_depth_consistency(maps, cameras, [])
    assert report.status == "insufficient_evidence"


def test_to_dict_roundtrip():
    maps, cameras = _two_camera_fixture(bias_b=1.0)
    report = check_depth_consistency(maps, cameras, SCENE, max_angle_deg=90.0)
    d = report.to_dict()
    assert d["status"] == "contradictions"
    assert d["pairs_checked"] == report.pairs_checked
    assert isinstance(d["contradictions"], list) and d["contradictions"]
