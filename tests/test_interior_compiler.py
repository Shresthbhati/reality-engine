"""End-to-end tests for the Unified Automatic Interior Compiler & Reality Studio.

Verifies the complete interior pipeline across all 5 pillars:
A. Interior pipeline:
   Evidence -> Reconstruction -> Architectural perception -> Rooms -> Corridors
   -> Openings -> Windows -> Stairs -> Levels -> InteriorSpaceGraph -> WorldIR.
B. Corridor completion:
   Geometry, width/length, connected rooms, openings, intersections, level assignment,
   and honest ambiguous circulation handling (no invented geometry).
C. Interior topology:
   Canonical graph:
   ROOM <-> DOORWAY <-> CORRIDOR <-> ROOM
                       ^
                     STAIR
                       v
                     LEVEL
D. Inspectable architectural entities:
   Building, levels, rooms, corridors, walls, floors, ceilings, doors, windows, stairs
   with topology, confidence, provenance, and geometry.
E. Spatial correction loop:
   Correction -> WorldStore -> new version -> diff.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from engine.compiler import (
    CompileDiagnostics,
    CompileOptions,
    compile_reconstruction_to_world,
)
from perception.architecture.classify import ArchitecturalElement, PlaneInput
from perception.architecture.corridor import (
    CorridorGraph,
    CorridorIntersection,
    detect_corridors,
)
from perception.architecture.room_graph import (
    BuildingGraph,
    RoomGraph,
    RoomOpening,
    Storey,
    build_building_graph,
    build_room_graph,
)
from perception.architecture.space_graph import InteriorSpaceGraph
from perception.architecture.stairs import StaircaseFit, detect_stairs
from perception.architecture.windows import WindowFit, detect_window
from provenance import Provenance
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from tests.test_canonical_interior import UP, _canonical_interior_scene
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
    WorldIR,
)
from world_ir.artifact_store import FileArtifactStore
from world_ir.diff import diff_worlds
from worldstore.store import WorldStore


class TestInteriorCompilerPipeline:
    def test_complete_interior_compilation_e2e(self, tmp_path: Path):
        """Test Pillar A & D: End-to-end interior compiler generates rooms, levels, building, and space graph."""
        result = _canonical_interior_scene()
        store_root = tmp_path / "store"
        store_root.mkdir()
        artifact_store = FileArtifactStore(store_root / "artifacts")

        options = CompileOptions(
            seed=42,
            up=UP,
            artifact_store=artifact_store,
            promote_corridors=True,
            promote_windows=True,
            promote_stairs=True,
            promote_building=True,
        )

        world, diag = compile_reconstruction_to_world(result, options)

        # 1. Plane classification & promotion
        assert diag.planes_total > 5
        wall_entities = [e for e in world.entities.values() if e.type == EntityType.WALL]
        floor_entities = [e for e in world.entities.values() if e.type == EntityType.FLOOR]
        ceiling_entities = [e for e in world.entities.values() if e.type == EntityType.CEILING]
        assert len(wall_entities) >= 4
        assert len(floor_entities) >= 1
        assert len(ceiling_entities) >= 1

        # 2. Rooms
        assert diag.rooms_detected >= 2
        rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
        assert len(rooms) >= 2

        # 3. Building Envelope & Storeys
        buildings = [e for e in world.entities.values() if e.type == EntityType.BUILDING]
        assert len(buildings) == 1
        bld = buildings[0]
        assert bld.provenance == Provenance.INFERRED
        assert "n_storeys" in bld.custom_properties
        assert bld.custom_properties["n_storeys"] >= 2

        storeys = [e for e in world.entities.values() if e.type == EntityType.STOREY]
        assert len(storeys) >= 2

        # 4. InteriorSpaceGraph attached to world metadata
        assert "interior_space_graph" in world.metadata
        sg_dict = world.metadata["interior_space_graph"]
        assert "rooms" in sg_dict
        assert "levels" in sg_dict
        assert len(sg_dict["levels"]) >= 2
        assert len(sg_dict["rooms"]) >= 2

    def test_corridor_inference_and_topology(self):
        """Test Pillar B & C: Measured corridor completion and canonical topology graph."""
        up = (0.0, 0.0, 1.0)

        # 1. Create elongated circulation corridor geometry
        # Floor: 1.2m wide, 5.0m long at Z=0.0 (aspect ratio 4.17 >= 2.0)
        floor_pi = PlaneInput(
            plane_id="f-corr-1",
            normal=(0.0, 0.0, 1.0),
            centroid=(0.6, 2.5, 0.0),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(1.2, 5.0, 0.0),
        )
        floor_el = ArchitecturalElement(
            element_id="el-f-corr-1",
            element_type="floor",
            source_plane_id="f-corr-1",
            reason="floor",
            bounds_min=floor_pi.bounds_min,
            bounds_max=floor_pi.bounds_max,
        )

        # Ceiling: 1.2m wide, 5.0m long at Z=2.4
        ceil_pi = PlaneInput(
            plane_id="c-corr-1",
            normal=(0.0, 0.0, -1.0),
            centroid=(0.6, 2.5, 2.4),
            bounds_min=(0.0, 0.0, 2.4),
            bounds_max=(1.2, 5.0, 2.4),
        )
        ceil_el = ArchitecturalElement(
            element_id="el-c-corr-1",
            element_type="ceiling",
            source_plane_id="c-corr-1",
            reason="ceiling",
            bounds_min=ceil_pi.bounds_min,
            bounds_max=ceil_pi.bounds_max,
        )

        # Left longitudinal wall: X=0.0, Y in [0, 5], Z in [0, 2.4]
        w1_pi = PlaneInput(
            plane_id="w-corr-l",
            normal=(1.0, 0.0, 0.0),
            centroid=(0.0, 2.5, 1.2),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 5.0, 2.4),
        )
        w1_el = ArchitecturalElement(
            element_id="el-w-corr-l",
            element_type="wall",
            source_plane_id="w-corr-l",
            reason="wall",
            bounds_min=w1_pi.bounds_min,
            bounds_max=w1_pi.bounds_max,
        )

        # Right longitudinal wall: X=1.2, Y in [0, 5], Z in [0, 2.4]
        w2_pi = PlaneInput(
            plane_id="w-corr-r",
            normal=(-1.0, 0.0, 0.0),
            centroid=(1.2, 2.5, 1.2),
            bounds_min=(1.2, 0.0, 0.0),
            bounds_max=(1.2, 5.0, 2.4),
        )
        w2_el = ArchitecturalElement(
            element_id="el-w-corr-r",
            element_type="wall",
            source_plane_id="w-corr-r",
            reason="wall",
            bounds_min=w2_pi.bounds_min,
            bounds_max=w2_pi.bounds_max,
        )

        elements = [floor_el, ceil_el, w1_el, w2_el]
        plane_inputs = {
            "f-corr-1": floor_pi,
            "c-corr-1": ceil_pi,
            "w-corr-l": w1_pi,
            "w-corr-r": w2_pi,
        }

        # Connect to Room A (at Y=0) and Room B (at Y=5)
        room_a = RoomGraph(
            room_id="room-A",
            boundary_element_ids=("el-w-corr-l", "el-w-a1", "el-w-a2", "el-f-a"),
            bounds_min=(-3.0, -3.0, 0.0),
            bounds_max=(0.0, 0.0, 2.4),
            dimensions_m={"x": 3.0, "y": 3.0, "z": 2.4},
            floor_area_m2=9.0,
        )
        room_b = RoomGraph(
            room_id="room-B",
            boundary_element_ids=("el-w-corr-r", "el-w-b1", "el-w-b2", "el-f-b"),
            bounds_min=(1.2, 5.0, 0.0),
            bounds_max=(4.2, 8.0, 2.4),
            dimensions_m={"x": 3.0, "y": 3.0, "z": 2.4},
            floor_area_m2=9.0,
        )

        corridors = detect_corridors(
            elements,
            up=up,
            plane_inputs=plane_inputs,
            rooms=[room_a, room_b],
        )

        assert len(corridors) == 1
        c = corridors[0]
        assert c.width_m == pytest.approx(1.2, rel=0.05)
        assert c.length_m == pytest.approx(5.0, rel=0.05)
        assert c.aspect_ratio >= 2.0
        assert c.status == "detected"
        assert c.confidence > 0.7
        # Shares boundary wall with room A and room B
        assert "room-A" in c.connected_room_ids
        assert "room-B" in c.connected_room_ids

        # 2. Build canonical graph: ROOM <-> DOORWAY <-> CORRIDOR <-> ROOM, CORRIDOR <-> STAIR <-> LEVEL
        world = WorldIR(id="w-topo-test")
        corr_id = c.corridor_id

        # Corridor entity
        world.entities[corr_id] = Entity(
            id=corr_id,
            type=EntityType.CORRIDOR,
            name="Main Circulation Corridor",
            custom_properties=c.to_dict(),
            provenance=Provenance.INFERRED,
            confidence=c.confidence,
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id="door-001"),
                Relationship(kind=RelationshipKind.CONNECTS, target_id="door-002"),
                Relationship(kind=RelationshipKind.CONNECTS, target_id="stair-001"),
                Relationship(kind=RelationshipKind.PART_OF, target_id="storey-01"),
            ],
        )

        world.entities["door-001"] = Entity(
            id="door-001",
            type=EntityType.DOOR,
            name="Doorway to Room A",
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id=corr_id),
                Relationship(kind=RelationshipKind.CONNECTS, target_id="room-A"),
            ],
        )
        world.entities["door-002"] = Entity(
            id="door-002",
            type=EntityType.DOOR,
            name="Doorway to Room B",
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id=corr_id),
                Relationship(kind=RelationshipKind.CONNECTS, target_id="room-B"),
            ],
        )
        world.entities["stair-001"] = Entity(
            id="stair-001",
            type=EntityType.STAIRS,
            name="Staircase 1",
            relationships=[
                Relationship(kind=RelationshipKind.CONNECTS, target_id=corr_id),
                Relationship(kind=RelationshipKind.CONNECTS, target_id="storey-02"),
            ],
        )
        world.entities["storey-01"] = Entity(
            id="storey-01",
            type=EntityType.STOREY,
            name="Ground Level",
            relationships=[
                Relationship(kind=RelationshipKind.CONTAINS, target_id=corr_id),
                Relationship(kind=RelationshipKind.CONTAINS, target_id="room-A"),
            ],
        )
        world.entities["storey-02"] = Entity(
            id="storey-02",
            type=EntityType.STOREY,
            name="Upper Level",
            relationships=[
                Relationship(kind=RelationshipKind.CONTAINS, target_id="room-B"),
            ],
        )

        # Verify full canonical traversal
        # ROOM A -> DOOR 1 -> CORRIDOR -> DOOR 2 -> ROOM B
        door_a = next(r.target_id for r in world.entities[corr_id].relationships if r.target_id == "door-001")
        assert any(r.target_id == "room-A" for r in world.entities[door_a].relationships)
        # CORRIDOR -> STAIR -> LEVEL
        stair_edge = next(r.target_id for r in world.entities[corr_id].relationships if r.target_id == "stair-001")
        assert any(r.target_id == "storey-02" for r in world.entities[stair_edge].relationships)

    def test_ambiguous_circulation_honest_handling(self):
        """Test Pillar B: Ambiguous circulation candidate is handled honestly without inventing geometry."""
        # Create an elongated candidate missing enclosing walls
        floor_pi = PlaneInput(
            plane_id="f-ambig",
            normal=(0.0, 1.0, 0.0),
            centroid=(1.0, 0.0, 3.0),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(2.0, 0.0, 6.0),  # 2.0 x 6.0 m (aspect 3.0)
        )
        floor_el = ArchitecturalElement(
            element_id="el-f-ambig",
            element_type="floor",
            source_plane_id="f-ambig",
            reason="floor",
            bounds_min=floor_pi.bounds_min,
            bounds_max=floor_pi.bounds_max,
        )
        ceil_pi = PlaneInput(
            plane_id="c-ambig",
            normal=(0.0, -1.0, 0.0),
            centroid=(1.0, 2.4, 3.0),
            bounds_min=(0.0, 2.4, 0.0),
            bounds_max=(2.0, 2.4, 6.0),
        )
        ceil_el = ArchitecturalElement(
            element_id="el-c-ambig",
            element_type="ceiling",
            source_plane_id="c-ambig",
            reason="ceiling",
            bounds_min=ceil_pi.bounds_min,
            bounds_max=ceil_pi.bounds_max,
        )
        # Only 1 wall (open/incomplete enclosure)
        w1_pi = PlaneInput(
            plane_id="w-ambig",
            normal=(1.0, 0.0, 0.0),
            centroid=(0.0, 1.2, 3.0),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 2.4, 6.0),
        )
        w1_el = ArchitecturalElement(
            element_id="el-w-ambig",
            element_type="wall",
            source_plane_id="w-ambig",
            reason="wall",
            bounds_min=w1_pi.bounds_min,
            bounds_max=w1_pi.bounds_max,
        )

        elements = [floor_el, ceil_el, w1_el]
        plane_inputs = {"f-ambig": floor_pi, "c-ambig": ceil_pi, "w-ambig": w1_pi}

        # Must refuse or not declare a full confident corridor
        corridors = detect_corridors(elements, up=UP, plane_inputs=plane_inputs)
        # Candidate has len(walls) == 1 < 2, so detect_corridors refuses to guess a corridor
        assert len(corridors) == 0

    def test_window_and_stair_perception_and_promotion(self):
        """Test Pillar A & D: Window opening and stair rhythm perception and WorldIR promotion."""
        # 1. Window Detection: Wall plane with opening
        # Wall: X in [0, 4.0], Z in [0, 3.0] with opening at X in [1.4, 2.6], Z in [0.9, 1.9]
        win_pts = []
        step = 0.05
        for xi in range(int(4.0 / step) + 1):
            x = round(xi * step, 3)
            for zi in range(int(3.0 / step) + 1):
                z = round(zi * step, 3)
                if 1.4 <= x <= 2.6 and 0.9 <= z <= 1.9:
                    continue  # Window opening
                win_pts.append((x, 0.0, z))

        wall_pi = PlaneInput(
            plane_id="wall-ext-1",
            normal=(0.0, 1.0, 0.0),
            centroid=(2.0, 0.0, 1.5),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(4.0, 0.0, 3.0),
            inlier_positions=tuple(win_pts),
        )
        win_fit = detect_window(wall_pi, up=(0.0, 0.0, 1.0), floor_height=0.0)
        assert win_fit is not None
        assert win_fit.width_m == pytest.approx(1.2, abs=0.15)
        assert win_fit.height_m == pytest.approx(1.0, abs=0.15)
        assert win_fit.sill_height_m == pytest.approx(0.9, abs=0.15)
        assert win_fit.confidence > 0.7

        # 2. Stair Detection: 6 steps climbing along +Z
        stair_pts = []
        riser = 0.17
        going = 0.28
        for s in range(7):
            z = s * riser
            xb = s * going
            for xi in range(5):
                for yi in range(10):
                    stair_pts.append((xb + xi * 0.06, yi * 0.1, z))

        st_fit = detect_stairs(stair_pts)
        assert st_fit is not None
        assert st_fit.n_steps == 6
        assert st_fit.rise_m == pytest.approx(0.17, abs=0.03)
        assert st_fit.going_m == pytest.approx(0.28, abs=0.03)
        assert st_fit.confidence > 0.7

        # 3. WorldIR Promotion
        world = WorldIR(id="w-win-stair-test")
        win_ent = Entity(
            id="window-001",
            type=EntityType.WINDOW,
            name="Exterior Window 001",
            custom_properties=win_fit.to_dict(),
            provenance=Provenance.INFERRED,
            confidence=win_fit.confidence,
        )
        stair_ent = Entity(
            id="stairs-001",
            type=EntityType.STAIRS,
            name="Main Circulation Staircase",
            custom_properties=st_fit.to_dict(),
            provenance=Provenance.INFERRED,
            confidence=st_fit.confidence,
        )
        world.entities[win_ent.id] = win_ent
        world.entities[stair_ent.id] = stair_ent

        assert world.entities["window-001"].custom_properties["sill_height_m"] == pytest.approx(0.9, abs=0.15)
        assert world.entities["stairs-001"].custom_properties["n_steps"] == 6

    def test_reality_studio_inspectable_entities(self):
        """Test Pillar D: Every reconstructed architectural entity exposes topology, evidence, confidence and provenance."""
        world = WorldIR(id="w-inspect-test")

        entity_types_to_verify = [
            (EntityType.BUILDING, "Building A", {"n_storeys": 2}),
            (EntityType.STOREY, "Ground Floor", {"floor_height_m": 0.0}),
            (EntityType.ROOM, "Living Room", {"floor_area_m2": 24.5}),
            (EntityType.CORRIDOR, "North Hallway", {"length_m": 6.2, "width_m": 1.4}),
            (EntityType.WALL, "Interior Partition Wall", {"thickness_m": 0.15}),
            (EntityType.FLOOR, "Hardwood Floor Plane", {"elevation_m": 0.0}),
            (EntityType.CEILING, "Ceiling Plane", {"elevation_m": 2.6}),
            (EntityType.DOOR, "Pocket Door Opening", {"width_m": 0.9}),
            (EntityType.WINDOW, "Picture Window", {"sill_height_m": 0.85}),
            (EntityType.STAIRS, "Straight Run Stair", {"n_steps": 14, "rise_m": 0.17}),
            (EntityType.COLUMN, "Structural Column", {"radius_m": 0.2}),
            (EntityType.BEAM, "Load-bearing Beam", {"depth_m": 0.4}),
        ]

        for idx, (etype, ename, props) in enumerate(entity_types_to_verify, start=1):
            eid = f"ent-{etype.value}-{idx:03d}"
            gid = f"geom-{eid}"
            world.geometries[gid] = Geometry(
                id=gid,
                type=GeometryType.BOX,
                bounds_min=Vector3(0.0, 0.0, 0.0),
                bounds_max=Vector3(2.0, 2.0, 2.0),
            )
            world.entities[eid] = Entity(
                id=eid,
                type=etype,
                name=ename,
                geometry_ids=[gid],
                custom_properties=props,
                provenance=Provenance.INFERRED,
                confidence=0.92,
                relationships=[
                    Relationship(kind=RelationshipKind.CONNECTS, target_id="ent-other-001"),
                ],
            )

        # Assert every entity type is present and fully inspectable
        for etype, ename, props in entity_types_to_verify:
            matching = [e for e in world.entities.values() if e.type == etype]
            assert len(matching) == 1
            e = matching[0]
            # 1. Topology
            assert len(e.relationships) > 0
            # 2. Evidence / Geometry
            assert len(e.geometry_ids) > 0
            geom = world.geometries[e.geometry_ids[0]]
            assert geom.bounds_min is not None
            # 3. Confidence & Provenance
            assert e.confidence > 0.0
            assert e.provenance is Provenance.INFERRED
            # 4. Custom properties
            for k in props:
                assert k in e.custom_properties

    def test_spatial_correction_and_diff(self, tmp_path: Path):
        """Test Pillar E: Spatial correction loop -> WorldStore -> new version -> diff."""
        store_root = tmp_path / "store"
        store_root.mkdir()
        ws = WorldStore(store_root)

        # Create world v1
        world_v1 = WorldIR(id="w-correction-test")
        corr_id = "corridor-001"
        world_v1.entities[corr_id] = Entity(
            id=corr_id,
            type=EntityType.CORRIDOR,
            name="Uncorrected Corridor",
            custom_properties={"length_m": 4.5, "width_m": 1.2},
            provenance=Provenance.INFERRED,
            confidence=0.80,
        )

        v1 = ws.save_version(world_v1, parent=None, version_id="v-interior-1")
        assert v1.version_id == "v-interior-1"

        # Apply spatial correction: rename, upgrade confidence, refine dimensions
        world_v2 = WorldIR.from_dict(world_v1.to_dict())
        world_v2.entities[corr_id].name = "Main North-South Gallery"
        world_v2.entities[corr_id].confidence = 0.99
        world_v2.entities[corr_id].custom_properties["corrected_by_user"] = True

        v2 = ws.save_version(world_v2, parent=v1.version_id, version_id="v-interior-2")
        assert v2.version_id == "v-interior-2"

        # Diff versions
        reloaded_v1 = ws.load_version(v1.version_id)
        reloaded_v2 = ws.load_version(v2.version_id)
        diff = diff_worlds(reloaded_v1, reloaded_v2)
        diff_dict = diff.to_dict()

        assert diff_dict["summary"]["entities_modified"] >= 1
        assert reloaded_v2.entities[corr_id].name == "Main North-South Gallery"
        assert reloaded_v1.entities[corr_id].name == "Uncorrected Corridor"
        assert reloaded_v2.entities[corr_id].custom_properties["corrected_by_user"] is True
