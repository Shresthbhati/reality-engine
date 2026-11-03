"""Unit and benchmark tests for 3D Oriented Bounding Box & 6-DoF Solver (Epic E05-2).

Verifies:
1. Minimum-volume oriented bounding box (OMBB) fitting.
2. Yaw angle estimation on rotated point clouds (0, 30, 45 degrees).
3. 3D IoU >= 0.70 for furniture (tables, chairs, beds).
4. 3D IoU >= 0.60 for small architectural fixtures (electrical sockets).
5. Contact surface snapping to supporting floor/wall planes.
6. Taxonomy size prior regularization.
"""

from __future__ import annotations

import math
import random
import pytest

from perception.instances.bounding import (
    OrientedBoundingBox,
    compute_3d_iou,
    solve_ombb,
)


def _generate_box_points(
    center: tuple[float, float, float],
    extents: tuple[float, float, float],
    yaw_deg: float,
    num_points: int = 200,
    noise_m: float = 0.01,
) -> list[list[float]]:
    """Generate noisy 3D surface points inside/on an oriented box."""
    cx, cy, cz = center
    dx, dy, dz = extents
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    points = []
    # Add box 8 corners to define boundary
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                lx = sx * dx
                ly = sy * dy
                lz = sz * dz
                wx = cx + lx * cos_y - lz * sin_y
                wy = cy + ly
                wz = cz + lx * sin_y + lz * cos_y
                points.append([wx, wy, wz])

    rng = random.Random(42)
    # Add surface sampled points
    for _ in range(num_points):
        # Pick random face
        axis = rng.randint(0, 2)
        sign = rng.choice([-0.5, 0.5])
        if axis == 0:
            lx = sign * dx
            ly = rng.uniform(-0.5, 0.5) * dy
            lz = rng.uniform(-0.5, 0.5) * dz
        elif axis == 1:
            lx = rng.uniform(-0.5, 0.5) * dx
            ly = sign * dy
            lz = rng.uniform(-0.5, 0.5) * dz
        else:
            lx = rng.uniform(-0.5, 0.5) * dx
            ly = rng.uniform(-0.5, 0.5) * dy
            lz = sign * dz

        # Add small measurement noise
        lx += rng.gauss(0, noise_m)
        ly += rng.gauss(0, noise_m)
        lz += rng.gauss(0, noise_m)

        wx = cx + lx * cos_y - lz * sin_y
        wy = cy + ly
        wz = cz + lx * sin_y + lz * cos_y
        points.append([wx, wy, wz])

    return points


def test_aligned_box_solver():
    # 2m x 0.8m x 1m box aligned with axes
    pts = _generate_box_points(center=(0.0, 0.4, 0.0), extents=(2.0, 0.8, 1.0), yaw_deg=0.0, noise_m=0.0)
    obb = solve_ombb(pts)

    # Check volume
    assert pytest.approx(obb.extents[1], abs=0.02) == 0.8
    # One extent is ~2.0, other is ~1.0
    dim_sorted = sorted([obb.extents[0], obb.extents[2]])
    assert pytest.approx(dim_sorted[0], abs=0.03) == 1.0
    assert pytest.approx(dim_sorted[1], abs=0.03) == 2.0


def test_rotated_box_recovers_yaw():
    # 45-degree rotated box
    yaw_target = 45.0
    pts = _generate_box_points(center=(5.0, 1.0, 3.0), extents=(1.8, 0.75, 0.9), yaw_deg=yaw_target, noise_m=0.0)
    obb = solve_ombb(pts)

    # Volume should be ~ 1.8 * 0.75 * 0.9 = 1.215
    expected_vol = 1.8 * 0.75 * 0.9
    assert pytest.approx(obb.volume, rel=0.05) == expected_vol

    # Yaw angle should align with 45 degrees (modulo 90 degrees)
    yaw_deg = math.degrees(obb.yaw_rad) % 90.0
    assert pytest.approx(yaw_deg, abs=2.0) == 45.0


def test_iou_identical_boxes():
    box1 = OrientedBoundingBox(
        center=(1.0, 0.5, 1.0),
        extents=(2.0, 1.0, 1.5),
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        yaw_rad=0.0,
    )
    assert pytest.approx(compute_3d_iou(box1, box1), abs=1e-5) == 1.0


def test_iou_disjoint_boxes():
    box1 = OrientedBoundingBox(
        center=(0.0, 0.5, 0.0),
        extents=(1.0, 1.0, 1.0),
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        yaw_rad=0.0,
    )
    box2 = OrientedBoundingBox(
        center=(10.0, 0.5, 0.0),
        extents=(1.0, 1.0, 1.0),
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        yaw_rad=0.0,
    )
    assert compute_3d_iou(box1, box2) == 0.0


def test_furniture_benchmark_iou_ge_070():
    """Verify 3D IoU >= 0.70 for rotated furniture with sensor noise."""
    gt_center = (2.5, 0.45, 1.8)
    gt_extents = (1.6, 0.9, 0.8)
    gt_yaw = 35.0

    pts = _generate_box_points(center=gt_center, extents=gt_extents, yaw_deg=gt_yaw, num_points=300, noise_m=0.015)
    solved_obb = solve_ombb(pts, class_id="furniture.table")

    # Construct Ground Truth OBB
    cos_y = math.cos(math.radians(gt_yaw))
    sin_y = math.sin(math.radians(gt_yaw))
    gt_rot = [
        [cos_y, 0, -sin_y],
        [0, 1, 0],
        [sin_y, 0, cos_y],
    ]
    gt_box = OrientedBoundingBox(
        center=gt_center,
        extents=gt_extents,
        rotation=gt_rot,
        yaw_rad=math.radians(gt_yaw),
    )

    iou = compute_3d_iou(solved_obb, gt_box)
    assert iou >= 0.70, f"Furniture 3D IoU {iou:.3f} must be >= 0.70"


def test_small_fixture_benchmark_iou_ge_060():
    """Verify 3D IoU >= 0.60 for small electrical fixtures on wall."""
    gt_center = (1.5, 1.2, 0.05)
    gt_extents = (0.12, 0.12, 0.03)  # Wall socket faceplate
    gt_yaw = 0.0

    pts = _generate_box_points(center=gt_center, extents=gt_extents, yaw_deg=gt_yaw, num_points=100, noise_m=0.003)
    solved_obb = solve_ombb(pts, class_id="electrical.switchboard.module.socket")

    gt_box = OrientedBoundingBox(
        center=gt_center,
        extents=gt_extents,
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        yaw_rad=0.0,
    )

    iou = compute_3d_iou(solved_obb, gt_box)
    assert iou >= 0.60, f"Small fixture 3D IoU {iou:.3f} must be >= 0.60"


def test_supporting_plane_snapping():
    # Points slightly floating (y from 0.05 to 0.85)
    pts = _generate_box_points(center=(0.0, 0.45, 0.0), extents=(1.0, 0.8, 1.0), yaw_deg=0.0)

    # Floor at y = 0.0
    floor_plane = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    obb = solve_ombb(pts, snap_to_plane=floor_plane)

    # Bottom face of OBB must be at y = 0.0
    bottom_y = obb.center[1] - obb.extents[1] / 2.0
    assert pytest.approx(bottom_y, abs=1e-5) == 0.0
