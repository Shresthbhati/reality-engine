"""Phase 6: Coordinate Frame Firewall.

Enforces explicit, strongly-typed coordinate frames across:
capture frame -> sensor frame -> session frame -> registration frame -> world frame -> tile frame.

Asserts:
- Multi-hop coordinate transform chains compose explicitly without implicit axes
- Frame mismatches fail loudly with typed errors (never silent fallback)
- Rigid transform inversion round-trips accurately under rotation and translation
- Spatial index and tiling behave deterministically across negative coordinates, large coordinates, and tile boundaries
"""

from __future__ import annotations

import math
import pytest

from engine.scene_graph.spatial_index import SpatialIndex
from world_ir.coordinates import (
    CoordinateRegistry,
    Frame,
    Transform,
    _mat_mul,
)
from world_ir.schema_v1 import Entity, EntityType, Vector3
from world_ir.spatial_tiles import SpatialTiles
from world_ir.world_v1 import WorldIR


def _make_rotation_z_and_translation(angle_rad: float, tx: float, ty: float, tz: float):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    matrix = (
        (c, -s, 0.0, tx),
        (s,  c, 0.0, ty),
        (0.0, 0.0, 1.0, tz),
        (0.0, 0.0, 0.0, 1.0),
    )
    return matrix


class TestCoordinateFrameFirewall:
    """Verifies coordinate frame integrity and firewall rules."""

    def test_coordinate_graph_chain_resolution(self):
        """Multi-hop path: CAMERA -> SENSOR -> SESSION_LOCAL -> WORLD."""
        registry = CoordinateRegistry()

        # 1. Camera to Sensor (translation 0.05m along x)
        m1 = (
            (1.0, 0.0, 0.0, 0.05),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        t1 = Transform(source_frame=Frame.CAMERA, target_frame=Frame.SENSOR, matrix=m1)
        registry.register(t1)

        # 2. Sensor to Session-Local (rotated 90 deg around z + translated 1m along y)
        m2 = _make_rotation_z_and_translation(math.pi / 2.0, 0.0, 1.0, 0.0)
        t2 = Transform(source_frame=Frame.SENSOR, target_frame=Frame.SESSION_LOCAL, matrix=m2)
        registry.register(t2)

        # 3. Session-Local to World (translated 10m x, 20m y)
        m3 = (
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        t3 = Transform(source_frame=Frame.SESSION_LOCAL, target_frame=Frame.WORLD, matrix=m3)
        registry.register(t3)

        # Resolve CAMERA -> WORLD
        t_cam_to_world = registry.get(Frame.CAMERA, Frame.WORLD)
        assert t_cam_to_world is not None
        assert t_cam_to_world.source_frame == Frame.CAMERA
        assert t_cam_to_world.target_frame == Frame.WORLD

        # Verify point transformation matches sequential application
        pt_cam = (1.0, 0.0, 0.0)
        pt_world_direct = t_cam_to_world.apply(pt_cam)

        pt_step = t3.apply(t2.apply(t1.apply(pt_cam)))
        assert math.isclose(pt_world_direct[0], pt_step[0], abs_tol=1e-6)
        assert math.isclose(pt_world_direct[1], pt_step[1], abs_tol=1e-6)
        assert math.isclose(pt_world_direct[2], pt_step[2], abs_tol=1e-6)

    def test_frame_mismatch_fails_loudly(self):
        """Attempting to compose transforms across mismatched frames must raise ValueError."""
        t_cam_sess = Transform(source_frame=Frame.CAMERA, target_frame=Frame.SESSION_LOCAL)
        t_bldg_world = Transform(source_frame=Frame.BUILDING_LOCAL, target_frame=Frame.WORLD)

        with pytest.raises(ValueError, match="frame mismatch"):
            t_cam_sess.then(t_bldg_world)

    def test_rigid_inversion_and_roundtrip(self):
        """Inverting a rigid transform must recover the exact original point."""
        matrix = _make_rotation_z_and_translation(0.785398, -12.5, 43.2, 5.0)
        t = Transform(source_frame=Frame.SESSION_LOCAL, target_frame=Frame.WORLD, matrix=matrix)
        t_inv = t.inverse()

        assert t_inv.source_frame == Frame.WORLD
        assert t_inv.target_frame == Frame.SESSION_LOCAL

        test_points = [
            (0.0, 0.0, 0.0),
            (10.5, -20.3, 100.0),
            (-500.0, 250.0, -10.0),
        ]
        for p in test_points:
            p_world = t.apply(p)
            p_rec = t_inv.apply(p_world)
            assert math.isclose(p[0], p_rec[0], abs_tol=1e-5)
            assert math.isclose(p[1], p_rec[1], abs_tol=1e-5)
            assert math.isclose(p[2], p_rec[2], abs_tol=1e-5)

    def test_spatial_tiling_boundary_and_negative_coordinates(self):
        """SpatialTiles must handle negative coordinates, large coordinates, and exact boundaries stably."""
        world = WorldIR(id="world-coords", coordinate_frame=Frame.WORLD.value)

        # Entity at exact boundary x=10.0 -> tile (1, 0, 0)
        e_boundary = Entity(
            id="e-boundary",
            type=EntityType.SENSOR,
            transform={"position": {"x": 10.0, "y": 0.0, "z": 0.0}},
        )
        # Entity just inside tile 0 -> tile (0, 0, 0)
        e_inside_0 = Entity(
            id="e-inside-0",
            type=EntityType.SENSOR,
            transform={"position": {"x": 9.999, "y": 0.0, "z": 0.0}},
        )
        # Entity at negative coordinates (-15.0, -25.0, -5.0) -> tile (-2, -3, -1)
        e_negative = Entity(
            id="e-negative",
            type=EntityType.SENSOR,
            transform={"position": {"x": -15.0, "y": -25.0, "z": -5.0}},
        )
        # Entity at large coordinates (10050.0, 500.0, 20000.0) -> tile (1005, 50, 2000)
        e_large = Entity(
            id="e-large",
            type=EntityType.SENSOR,
            transform={"position": {"x": 10050.0, "y": 500.0, "z": 20000.0}},
        )

        for e in (e_boundary, e_inside_0, e_negative, e_large):
            world.entities[e.id] = e

        tiles = SpatialTiles(world, tile_size=10.0)

        # Check tile assignments
        assert tiles.entities_in_tile((1, 0, 0)) == ["e-boundary"]
        assert tiles.entities_in_tile((0, 0, 0)) == ["e-inside-0"]
        assert tiles.entities_in_tile((-2, -3, -1)) == ["e-negative"]
        assert tiles.entities_in_tile((1005, 50, 2000)) == ["e-large"]
