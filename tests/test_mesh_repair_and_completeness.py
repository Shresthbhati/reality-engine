"""Tests for mesh repair (Task 4.7) and reconstruction completeness (Tasks 4.9 & 4.10)."""

from __future__ import annotations

import pytest
import numpy as np

from reconstruction.backend.interface import (
    ReconstructionResult,
    ReconstructedCameraPose,
    ReconstructedPoint,
)
from reconstruction.meshing.mesh import MeshData
from reconstruction.meshing.repair import clean_and_repair_mesh, mesh_to_trimesh, trimesh_to_mesh
from reconstruction.completeness import measure_reconstruction_completeness


def test_mesh_repair_cleans_duplicate_and_unreferenced_vertices():
    # Construct a mesh with duplicate vertices and an unreferenced vertex
    # Box with a repeated corner and an isolated vertex
    verts = (
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.0, 1.0, 0.0),  # 2
        (0.0, 0.0, 0.0),  # 3 (duplicate of 0)
        (5.0, 5.0, 5.0),  # 4 (unreferenced)
    )
    faces = (
        (0, 1, 2),
        (3, 1, 2),  # uses duplicate vertex 3
    )
    mesh = MeshData(vertices=verts, faces=faces)
    clean, stats = clean_and_repair_mesh(mesh, fill_holes=False)

    assert stats["initial_vertices"] == 5
    # The duplicate vertex 3 is merged with 0, and unreferenced vertex 4 is removed
    assert len(clean.vertices) == 3
    assert clean.units == "meter"
    assert clean.frame == "world"


def test_mesh_repair_watertight_cube():
    # 8 vertices of a cube, 12 triangles
    verts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    faces = [
        # bottom
        (0, 1, 2), (0, 2, 3),
        # top
        (4, 6, 5), (4, 7, 6),
        # front
        (0, 5, 1), (0, 4, 5),
        # back
        (3, 2, 6), (3, 6, 7),
        # left
        (0, 3, 7), (0, 7, 4),
        # right
        (1, 5, 6), (1, 6, 2),
    ]
    mesh = MeshData(vertices=tuple(verts), faces=tuple(faces))
    clean, stats = clean_and_repair_mesh(mesh, fill_holes=True)

    assert stats["watertight"] is True
    assert len(clean.faces) == 12


def test_measure_reconstruction_completeness():
    poses = [
        ReconstructedCameraPose("cam1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("cam2", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("cam3", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ]
    points = [
        ReconstructedPoint((0.5, 1.0, 0.0), "trk1", ["cam1", "cam2", "cam3"]),
        ReconstructedPoint((1.5, 1.0, 0.0), "trk2", ["cam2", "cam3"]),
        ReconstructedPoint((1.0, 2.0, 0.0), "trk3", ["cam1", "cam2", "cam3"]),
        ReconstructedPoint((1.0, 1.5, 1.0), "trk4", ["cam1", "cam2"]),
    ]
    result = ReconstructionResult(
        points=points,
        camera_poses=poses,
        registration_status="success",
    )

    metrics = measure_reconstruction_completeness(
        result=result,
        total_input_cameras=4,
        scale_metadata={"state": "metric", "basis": "tape_measured_baseline"},
        voxel_size_m=0.5,
    )

    assert metrics.cameras_input == 4
    assert metrics.cameras_registered == 3
    assert metrics.registration_ratio == 0.75
    assert metrics.points_total == 4
    assert metrics.scene_volume_m3 > 0.0
    assert metrics.mean_track_length == 2.5
    assert metrics.redundancy_ratio_3plus == 0.5  # 2 out of 4 points have >= 3 cameras
    assert metrics.scale_state == "metric"
    assert metrics.scale_reliability == "tape_verified"
    assert 0.0 < metrics.completeness_score <= 1.0
