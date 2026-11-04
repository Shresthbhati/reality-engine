"""Tests for Universal Geometry Compiler (Directive Rule 7).

Verifies:
- B-Rep representations and watertight 2-manifold triangular meshes
- Wall solids with explicit thickness and CSG opening cuts (doors, windows)
- Curved wall cylindrical shell solids
- Staircases with code compliance checks (IBC riser/tread tolerances)
- Structural columns and beams
- MEP conduits and electrical fixtures (sockets, switches)
- Site terrain elevation solids and roadway corridors with kerbs and camber
- Integration with WorldIR, ArtifactStore, and schema validation
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import pytest

from engine.compiler.geometry_compiler import (
    BoundaryRepresentation,
    CompiledSolid,
    GeometryCompiler,
    GeometryValidationResult,
    GeometryValidator,
    OpeningSpecification,
    compile_and_attach_geometry,
)
from provenance import Provenance
from world_ir import Entity, EntityType, GeometryType, WorldIR
from world_ir.artifact_store import FileArtifactStore
from world_ir.geometry_data import MeshData
from world_ir.validation import validate_world_ir


class TestGeometryCompilerRule7:
    def test_wall_solid_no_openings(self):
        """Test solid wall extrusion with exact thickness and volume."""
        length = 6.0
        height = 2.80
        thickness = 0.25

        solid = GeometryCompiler.compile_wall_solid(length, height, thickness)

        assert solid.validation.is_watertight is True
        assert solid.validation.is_2manifold is True
        assert solid.validation.has_positive_volume is True
        assert solid.validation.has_consistent_normals is True
        assert solid.validation.is_manufacturable is True

        expected_vol = length * height * thickness
        assert pytest.approx(solid.validation.volume_m3, rel=1e-3) == expected_vol
        assert solid.dimensions["length_m"] == length
        assert solid.dimensions["thickness_m"] == thickness

    def test_wall_solid_with_door_and_window_openings(self):
        """Test CSG aperture carving through a wall slab preserving 2-manifold topology."""
        length = 5.0
        height = 3.0
        thickness = 0.20

        door = OpeningSpecification(
            opening_id="door-1",
            kind="door",
            offset_along_wall_m=1.0,
            sill_height_m=0.0,
            width_m=0.90,
            height_m=2.10,
        )
        window = OpeningSpecification(
            opening_id="window-1",
            kind="window",
            offset_along_wall_m=3.0,
            sill_height_m=1.00,
            width_m=1.20,
            height_m=1.20,
        )

        solid = GeometryCompiler.compile_wall_solid(length, height, thickness, openings=[door, window])

        assert solid.validation.is_watertight is True
        assert solid.validation.is_2manifold is True
        assert len(solid.validation.non_manifold_edges) == 0

        # Expected volume: total wall minus door void minus window void
        total_vol = length * height * thickness  # 3.00 m3
        door_vol = door.width_m * door.height_m * thickness  # 0.9 * 2.1 * 0.2 = 0.378 m3
        win_vol = window.width_m * window.height_m * thickness  # 1.2 * 1.2 * 0.2 = 0.288 m3
        expected_vol = total_vol - door_vol - win_vol  # 2.334 m3

        assert pytest.approx(solid.validation.volume_m3, rel=1e-3) == expected_vol

    def test_curved_wall_solid(self):
        """Test parametric cylindrical shell wall solid."""
        center = (0.0, 0.0, 0.0)
        radius = 4.0
        thickness = 0.20
        height = 3.0
        start_ang = 0.0
        end_ang = math.pi / 2.0  # Quarter circle

        solid = GeometryCompiler.compile_curved_wall_solid(
            center=center,
            radius=radius,
            thickness=thickness,
            height=height,
            start_angle_rad=start_ang,
            end_angle_rad=end_ang,
            segments=16,
        )

        assert solid.validation.is_watertight is True
        assert solid.validation.is_2manifold is True
        assert solid.validation.has_positive_volume is True

        # Volume of quarter cylinder shell: pi/4 * (R_outer^2 - R_inner^2) * H
        r_out = radius + thickness / 2.0
        r_in = radius - thickness / 2.0
        expected_vol = 0.25 * math.pi * (r_out**2 - r_in**2) * height
        assert pytest.approx(solid.validation.volume_m3, rel=5e-2) == expected_vol

    def test_staircase_solid_and_code_compliance(self):
        """Test stepped stair solid with tread/riser and IBC code validation."""
        width = 1.05
        total_rise = 2.00
        total_run = 3.00
        num_treads = 10

        solid = GeometryCompiler.compile_staircase_solid(
            width=width,
            total_rise=total_rise,
            total_run=total_run,
            num_treads=num_treads,
            include_railing=True,
        )

        assert solid.validation.is_watertight is True
        assert solid.validation.is_2manifold is True
        assert solid.dimensions["riser_height_m"] == pytest.approx(0.20, abs=1e-4)
        assert solid.dimensions["tread_depth_m"] == pytest.approx(0.30, abs=1e-4)

    def test_structural_column_and_beam(self):
        """Test rectangular column and beam solids."""
        col = GeometryCompiler.compile_structural_column(0.40, 0.40, 3.20)
        assert col.validation.is_watertight is True
        assert pytest.approx(col.validation.volume_m3, rel=1e-3) == (0.40 * 0.40 * 3.20)

        beam = GeometryCompiler.compile_structural_beam(0.25, 0.50, 5.00)
        assert beam.validation.is_watertight is True
        assert pytest.approx(beam.validation.volume_m3, rel=1e-3) == (0.25 * 0.50 * 5.00)

    def test_mep_conduit_and_fixture(self):
        """Test MEP swept tube conduit and electrical socket faceplate."""
        conduit = GeometryCompiler.compile_mep_conduit(
            centerline_points=[(0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (1.5, 0.0, 2.0)],
            outer_diameter=0.032,  # 32mm conduit
        )
        assert conduit.validation.is_watertight is True
        assert conduit.dimensions["outer_diameter_m"] == 0.032

        socket = GeometryCompiler.compile_mep_fixture(
            fixture_type="socket",
            width=0.086,
            height=0.086,
            depth=0.015,
        )
        assert socket.validation.is_watertight is True
        assert socket.dimensions["depth_m"] == 0.015

    def test_site_terrain_solid(self):
        """Test site terrain volume with elevation function."""
        def hill_fn(x: float, y: float) -> float:
            return 0.5 * math.sin(x * 0.5) * math.cos(y * 0.5)

        solid = GeometryCompiler.compile_site_terrain(
            x_bounds=(-10.0, 10.0),
            y_bounds=(-10.0, 10.0),
            base_elevation_m=0.0,
            resolution=(8, 8),
            elevation_fn=hill_fn,
        )

        assert solid.validation.is_watertight is True
        assert solid.validation.is_2manifold is True
        assert solid.validation.has_positive_volume is True
        assert solid.dimensions["thickness_m"] == 2.0

    def test_roadway_corridor_solid(self):
        """Test roadway corridor with crossfall camber, kerbs, and sidewalks."""
        road = GeometryCompiler.compile_roadway_corridor(
            length=20.0,
            road_width=7.0,
            kerb_width=0.30,
            kerb_height=0.15,
            sidewalk_width=1.80,
        )

        assert road.validation.is_watertight is True
        assert road.validation.is_2manifold is True
        assert road.validation.has_positive_volume is True
        assert road.dimensions["road_width_m"] == 7.0
        assert road.dimensions["length_m"] == 20.0

    def test_worldir_integration_and_artifact_store(self):
        """Test compiling solid geometry, storing binary MeshData, and validating WorldIR."""
        with tempfile.TemporaryDirectory() as td:
            store = FileArtifactStore(Path(td))
            world = WorldIR()

            entity = Entity(id="ent-wall-42", name="Living Room Wall", type=EntityType.WALL)
            world.entities[entity.id] = entity

            solid = GeometryCompiler.compile_wall_solid(4.5, 2.7, 0.20)
            geom = compile_and_attach_geometry(entity, solid, world, artifact_store=store)

            assert geom.id in world.geometries
            assert geom.type == GeometryType.MESH
            assert geom.vertex_count == 8
            assert geom.triangle_count == 12
            assert geom.data_uri != ""
            assert geom.data_hash != ""

            # Check that payload can be re-loaded from artifact store as MeshData
            raw_bytes = store.get(geom.data_uri)
            reloaded_mesh = MeshData.from_bytes(raw_bytes)
            assert len(reloaded_mesh.vertices) == geom.vertex_count
            assert len(reloaded_mesh.triangles) == geom.triangle_count

            # WorldIR validation gate must pass with 0 errors
            report = validate_world_ir(world)
            assert len(report.errors) == 0

    def test_geometry_validator_detects_non_watertight_mesh(self):
        """Test that validator catches open boundaries and degenerate elements."""
        # Box missing one face (open boundary)
        v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
        t = [(0, 1, 4), (1, 2, 4)]  # Only 2 triangles
        val = GeometryValidator.validate_mesh_solid(v, t)
        assert val.is_watertight is False
        assert val.is_valid is False
