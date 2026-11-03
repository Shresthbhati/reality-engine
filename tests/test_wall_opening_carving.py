"""Tests for Epic E04-2: Host-Wall Opening Carving.

Validates:
1. Volumetric negative boolean opening carving in wall slabs (wall_pieces).
2. Host-wall opening carving in IFC export:
   - IfcOpeningElement created for each hosted door/window.
   - IfcRelVoidsElement connects host IfcWall to IfcOpeningElement.
   - IfcRelFillsElement connects IfcOpeningElement to filling (IfcDoor/IfcWindow).
   - Solid clashes between walls and doors/windows eliminated.
3. Byte-level determinism across repeated IFC exports with voided openings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from exporters.ifc_bridge import export_world_to_ifc
from perception.architecture.room_layout import wall_pieces, wall_slab_mesh
from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Relationship, RelationshipKind, Vector3
from world_ir.world_v1 import WorldIR


def _create_room_with_openings() -> WorldIR:
    world = WorldIR()

    # Host Wall: 6m long, 0.2m thick, 2.5m high along X axis
    wall_geom = Geometry(
        id="geom-wall-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(6.0, 0.2, 2.5),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-wall-1"] = wall_geom
    wall = Entity(
        id="wall-1",
        name="South Wall",
        type=EntityType.WALL,
        geometry_ids=["geom-wall-1"],
        provenance=Provenance.OBSERVED,
    )
    wall.custom_properties["wall_slab"] = {
        "start_xyz_m": [0.0, 0.0, 0.0],
        "end_xyz_m": [6.0, 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0],
        "outward": [0.0, 1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "length_m": 6.0,
        "height_m": 2.5,
        "thickness_m": 0.2,
        "openings": [
            {"entity_id": "door-1", "kind": "door", "offset_m": 1.0, "sill_m": 0.0, "width_m": 0.9, "height_m": 2.1},
            {"entity_id": "win-1", "kind": "window", "offset_m": 3.0, "sill_m": 0.9, "width_m": 1.5, "height_m": 1.2},
        ],
    }
    world.entities["wall-1"] = wall

    # Door: 0.9m wide, 2.1m high, 0.06m thick, hosted in wall-1 at offset 1.0m
    door_geom = Geometry(
        id="geom-door-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(1.0, 0.07, 0.0),
        bounds_max=Vector3(1.9, 0.13, 2.1),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-door-1"] = door_geom
    door = Entity(
        id="door-1",
        name="Entry Door",
        type=EntityType.DOOR,
        geometry_ids=["geom-door-1"],
        provenance=Provenance.OBSERVED,
    )
    door.custom_properties["in_wall"] = {
        "wall_id": "wall-1",
        "offset_m": 1.0,
        "sill_m": 0.0,
        "width_m": 0.9,
        "height_m": 2.1,
    }
    door.relationships.append(Relationship(kind=RelationshipKind.PART_OF, target_id="wall-1"))
    wall.relationships.append(Relationship(kind=RelationshipKind.CONTAINS, target_id="door-1"))
    world.entities["door-1"] = door

    # Window: 1.5m wide, 1.2m high, 0.08m thick, sill at 0.9m, offset 3.0m
    win_geom = Geometry(
        id="geom-win-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(3.0, 0.06, 0.9),
        bounds_max=Vector3(4.5, 0.14, 2.1),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-win-1"] = win_geom
    win = Entity(
        id="win-1",
        name="Living Room Window",
        type=EntityType.WINDOW,
        geometry_ids=["geom-win-1"],
        provenance=Provenance.OBSERVED,
    )
    win.custom_properties["in_wall"] = {
        "wall_id": "wall-1",
        "offset_m": 3.0,
        "sill_m": 0.9,
        "width_m": 1.5,
        "height_m": 1.2,
    }
    win.relationships.append(Relationship(kind=RelationshipKind.PART_OF, target_id="wall-1"))
    wall.relationships.append(Relationship(kind=RelationshipKind.CONTAINS, target_id="win-1"))
    world.entities["win-1"] = win

    return world


class TestHostWallOpeningCarving:
    def test_ifc_export_creates_openings_voids_and_fills(self, tmp_path: Path):
        world = _create_room_with_openings()
        out_path = tmp_path / "model.ifc"
        report = export_world_to_ifc(world, out_path)

        assert out_path.exists()
        f = ifcopenshell.open(str(out_path))

        # 1 wall, 1 door, 1 window
        walls = f.by_type("IfcWall")
        doors = f.by_type("IfcDoor")
        windows = f.by_type("IfcWindow")
        openings = f.by_type("IfcOpeningElement")
        assert len(walls) == 1
        assert len(doors) == 1
        assert len(windows) == 1
        assert len(openings) == 2

        # Voids: both openings must void the host wall
        voids = f.by_type("IfcRelVoidsElement")
        assert len(voids) == 2
        for v in voids:
            assert v.RelatingBuildingElement == walls[0]
            assert v.RelatedOpeningElement in openings

        # Fills: door and window must fill their respective openings
        fills = f.by_type("IfcRelFillsElement")
        assert len(fills) == 2
        filling_elements = {fill.RelatedBuildingElement for fill in fills}
        assert filling_elements == {doors[0], windows[0]}

    def test_standalone_opening_voids_host_wall_without_filling(self, tmp_path: Path):
        world = WorldIR()
        wall_geom = Geometry(
            id="gw", type=GeometryType.BOX,
            bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(5, 0.2, 2.5)
        )
        world.geometries["gw"] = wall_geom
        world.entities["wall-pass"] = Entity(id="wall-pass", name="Wall", type=EntityType.WALL, geometry_ids=["gw"])

        # Pass-through opening
        op_geom = Geometry(
            id="go", type=GeometryType.BOX,
            bounds_min=Vector3(2.0, 0.0, 0.8), bounds_max=Vector3(3.2, 0.2, 2.0)
        )
        world.geometries["go"] = op_geom
        op_ent = Entity(id="op-pass", name="Pass Through", type=EntityType.OPENING, geometry_ids=["go"])
        op_ent.relationships.append(Relationship(kind=RelationshipKind.PART_OF, target_id="wall-pass"))
        world.entities["op-pass"] = op_ent

        out_path = tmp_path / "passthrough.ifc"
        export_world_to_ifc(world, out_path)

        f = ifcopenshell.open(str(out_path))
        openings = f.by_type("IfcOpeningElement")
        assert len(openings) == 1
        voids = f.by_type("IfcRelVoidsElement")
        assert len(voids) == 1
        assert voids[0].RelatingBuildingElement.is_a("IfcWall")
        assert voids[0].RelatedOpeningElement == openings[0]
        # No door or window filling this pass-through
        fills = f.by_type("IfcRelFillsElement")
        assert len(fills) == 0

    def test_ifc_export_is_strictly_deterministic(self, tmp_path: Path):
        world = _create_room_with_openings()
        p1 = tmp_path / "run1.ifc"
        p2 = tmp_path / "run2.ifc"

        export_world_to_ifc(world, p1)
        export_world_to_ifc(world, p2)

        bytes1 = p1.read_bytes()
        bytes2 = p2.read_bytes()
        assert bytes1 == bytes2
        assert hashlib.sha256(bytes1).hexdigest() == hashlib.sha256(bytes2).hexdigest()

    def test_wall_pieces_merges_overlapping_openings(self):
        # Two overlapping windows horizontally: 2.0..3.5 and 2.5..4.0
        pieces = wall_pieces(
            length=6.0,
            height=2.5,
            openings=[
                {"offset_m": 2.0, "width_m": 1.5, "sill_m": 0.9, "height_m": 1.2},
                {"offset_m": 2.5, "width_m": 1.5, "sill_m": 0.9, "height_m": 1.2},
            ],
        )
        # Should cleanly merge into:
        # 1. Left wall pier: (0.0, 2.0, 0.0, 2.5)
        # 2. Sill: (2.0, 4.0, 0.0, 0.9)
        # 3. Lintel: (2.0, 4.0, 2.1, 2.5)
        # 4. Right wall pier: (4.0, 6.0, 0.0, 2.5)
        assert len(pieces) == 4
        assert pieces[0] == (0.0, 2.0, 0.0, 2.5)
        assert pieces[1] == (2.0, 4.0, 0.0, 0.9)
        assert pieces[2] == (2.0, 4.0, 2.1, 2.5)
        assert pieces[3] == (4.0, 6.0, 0.0, 2.5)

    def test_wall_pieces_floor_to_ceiling_opening_splits_wall(self):
        # Full-height opening from 2.0 to 3.0 m
        pieces = wall_pieces(
            length=6.0,
            height=2.5,
            openings=[{"offset_m": 2.0, "width_m": 1.0, "sill_m": 0.0, "height_m": 2.5}],
        )
        assert len(pieces) == 2
        assert pieces[0] == (0.0, 2.0, 0.0, 2.5)
        assert pieces[1] == (3.0, 6.0, 0.0, 2.5)

    def test_wall_slab_mesh_volume_matches_wall_minus_openings(self):
        # 6m x 2.5m x 0.2m wall with a 1m x 2m door
        verts, faces = wall_slab_mesh(
            origin_xyz=(0, 0, 0),
            dir_xyz=(1, 0, 0),
            out_xyz=(0, 1, 0),
            up_xyz=(0, 0, 1),
            length=6.0,
            height=2.5,
            thickness=0.2,
            openings=[{"offset_m": 1.5, "width_m": 1.0, "sill_m": 0.0, "height_m": 2.0}],
        )
        assert len(verts) > 0 and len(faces) > 0

        # Calculate signed volume
        import numpy as np
        v = np.array(verts)
        vol = sum(float(np.dot(v[a], np.cross(v[b], v[c]))) for a, b, c in faces) / 6.0
        expected_vol = (6.0 * 2.5 * 0.2) - (1.0 * 2.0 * 0.2)
        assert vol == pytest.approx(expected_vol, rel=1e-3)
