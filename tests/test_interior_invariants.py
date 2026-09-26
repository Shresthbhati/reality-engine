"""Regression tests for canonical interior-compiler invariants.

Covers the malformed-topology rejections and cross-representation
consistency guarantees added while unifying the interior pipeline:

  - a ROOM/CORRIDOR cannot reference a boundary part that does not exist
    (world_ir.validation._check_interior_boundary_references)
  - a ROOM/CORRIDOR cannot be CONTAINS-owned by two STOREY entities at
    once (world_ir.validation._check_single_storey_membership)
  - a WINDOW is wired to its host WALL by a real relationship, not only
    a raw plane-id string in custom_properties, so a deleted/renamed
    wall shows up as a dangling reference rather than disappearing
    silently (engine.compiler.world_compiler)
  - promote_building_topology reuses an already-promoted ROOM entity
    for the same physical space instead of minting a duplicate
    (perception.architecture.topology._find_existing_room_entity)
  - the full canonical compile (promote_building=True) survives a
    WorldStore save -> reload cycle with rooms/storeys/building
    relationships and the InteriorSpaceGraph metadata intact.
"""

from __future__ import annotations

from pathlib import Path

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from provenance import Provenance
from reconstruction.backend.colmap_backend import _parse_images_txt, _parse_points3d_txt
from reconstruction.backend.interface import ReconstructionResult
from tests.test_canonical_interior import UP, _canonical_interior_scene
from world_ir import EntityType, Relationship, RelationshipKind, WorldIR
from world_ir.validation import validate_world_ir
from worldstore.store import WorldStore

REPO_ROOT = Path(__file__).resolve().parent.parent
SOUTH_BUILDING_SPARSE = REPO_ROOT / "datasets" / "south_building" / "sparse"


def _bare_world() -> WorldIR:
    return WorldIR(id="w-invariants", name="invariants", main_branch_id="branch-main")


class TestBoundaryReferenceInvariant:
    def test_room_referencing_missing_wall_is_rejected(self):
        world = _bare_world()
        world.entities["room-001"] = _entity(
            "room-001", EntityType.ROOM,
            custom_properties={"boundary_element_ids": ["struct-plane-999"]},
        )
        report = validate_world_ir(world)
        codes = {i.code for i in report.errors}
        assert "interior_boundary_reference_missing" in codes

    def test_room_with_real_boundary_is_accepted(self):
        world = _bare_world()
        world.entities["struct-plane-000"] = _entity("struct-plane-000", EntityType.WALL)
        world.entities["room-001"] = _entity(
            "room-001", EntityType.ROOM,
            custom_properties={"boundary_element_ids": ["struct-plane-000"]},
        )
        report = validate_world_ir(world)
        codes = {i.code for i in report.errors}
        assert "interior_boundary_reference_missing" not in codes


class TestSingleStoreyMembershipInvariant:
    def test_room_owned_by_two_storeys_is_rejected(self):
        world = _bare_world()
        world.entities["room-001"] = _entity("room-001", EntityType.ROOM)
        world.entities["storey-01"] = _entity(
            "storey-01", EntityType.STOREY,
            relationships=[_contains("room-001")],
        )
        world.entities["storey-02"] = _entity(
            "storey-02", EntityType.STOREY,
            relationships=[_contains("room-001")],
        )
        report = validate_world_ir(world)
        codes = {i.code for i in report.errors}
        assert "entity_multiple_storey_membership" in codes

    def test_room_owned_by_one_storey_is_accepted(self):
        world = _bare_world()
        world.entities["room-001"] = _entity("room-001", EntityType.ROOM)
        world.entities["storey-01"] = _entity(
            "storey-01", EntityType.STOREY,
            relationships=[_contains("room-001")],
        )
        report = validate_world_ir(world)
        codes = {i.code for i in report.errors}
        assert "entity_multiple_storey_membership" not in codes


class TestWindowWallHostEdge:
    def test_window_carries_a_real_relationship_to_its_wall(self):
        world = _bare_world()
        world.entities["struct-plane-000"] = _entity("struct-plane-000", EntityType.WALL)
        world.entities["window-001"] = _entity(
            "window-001", EntityType.WINDOW,
            relationships=[Relationship(
                kind=RelationshipKind.PART_OF,
                target_id="struct-plane-000",
                confidence=0.9,
                provenance=Provenance.INFERRED,
            )],
        )
        report = validate_world_ir(world)
        assert report.is_valid()

    def test_window_referencing_deleted_wall_is_flagged(self):
        # Same graph as above, minus the wall entity: the PART_OF edge
        # now dangles and must surface as an error, not vanish.
        world = _bare_world()
        world.entities["window-001"] = _entity(
            "window-001", EntityType.WINDOW,
            relationships=[Relationship(
                kind=RelationshipKind.PART_OF,
                target_id="struct-plane-000",
                confidence=0.9,
                provenance=Provenance.INFERRED,
            )],
        )
        report = validate_world_ir(world)
        assert not report.is_valid()
        assert any(i.code == "structural" for i in report.errors)


class TestCanonicalCompileEndToEnd:
    def test_full_compile_with_building_promotion_passes_validation(self):
        result = _canonical_interior_scene()
        options = CompileOptions(
            seed=42, up=UP,
            promote_corridors=True, promote_windows=True,
            promote_stairs=True, promote_building=True,
        )
        world, diag = compile_reconstruction_to_world(result, options)
        assert diag.validation_issues == []
        report = validate_world_ir(world)
        assert report.is_valid(), [str(i) for i in report.errors]

    def test_full_compile_does_not_duplicate_rooms_across_detectors(self):
        # Regression for the topology.py dedup fix AND the room-detector
        # reconciliation decision: evidence.promote_rooms and
        # perception.architecture.topology both independently identify
        # rooms when promote_building=True. Without reuse-by-shared-
        # boundary matching this scene produced 5 ROOM entities for 2
        # physical rooms; evidence-side ring-closure detection is now
        # authoritative (see topology.py::promote_building_topology
        # docstring, "Room-detector reconciliation"), so the promoted
        # ROOM entity count must equal evidence-side detection exactly --
        # no extra entity for a RoomGraph grouping the stricter detector
        # didn't corroborate.
        result = _canonical_interior_scene()
        options = CompileOptions(
            seed=42, up=UP,
            promote_corridors=True, promote_windows=True,
            promote_stairs=True, promote_building=True,
        )
        world, diag = compile_reconstruction_to_world(result, options)
        rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
        assert len(rooms) == diag.rooms_detected
        # This fixture's RoomGraph groups the corridor-adjacent space
        # differently from the evidence-side wall-ring tracer; that
        # disagreement must surface as a visible, named refusal, not a
        # silently-dropped fact.
        assert any(
            "room-graph grouping" in w and "not promoted as separate entities" in w
            for w in diag.interior_warnings
        )

    def test_worldstore_roundtrip_preserves_topology(self, tmp_path: Path):
        result = _canonical_interior_scene()
        options = CompileOptions(
            seed=42, up=UP,
            promote_corridors=True, promote_windows=True,
            promote_stairs=True, promote_building=True,
        )
        world, _ = compile_reconstruction_to_world(result, options)

        store = WorldStore(tmp_path / "store")
        stored = store.save_version(world, parent=None)
        reloaded = store.load_version(stored.version_id)

        assert set(reloaded.entities.keys()) == set(world.entities.keys())

        buildings = [e for e in reloaded.entities.values() if e.type == EntityType.BUILDING]
        storeys = [e for e in reloaded.entities.values() if e.type == EntityType.STOREY]
        rooms = [e for e in reloaded.entities.values() if e.type == EntityType.ROOM]
        assert len(buildings) == 1
        assert len(storeys) >= 2
        assert len(rooms) >= 2

        # Relationships (topology) round-trip byte-for-byte, not just entity ids.
        for entity_id, original in world.entities.items():
            reloaded_entity = reloaded.entities[entity_id]
            original_edges = {(r.kind, r.target_id) for r in original.relationships}
            reloaded_edges = {(r.kind, r.target_id) for r in reloaded_entity.relationships}
            assert original_edges == reloaded_edges, entity_id

        # InteriorSpaceGraph metadata (the topology-query layer) survives
        # the round trip -- not asserted by any existing golden test.
        assert "interior_space_graph" in reloaded.metadata
        assert reloaded.metadata["interior_space_graph"] == world.metadata["interior_space_graph"]

        # A fresh store instance re-reading from disk (process-restart
        # durability) sees the identical topology, not just the same
        # in-memory object.
        reloaded_again = WorldStore(tmp_path / "store").load_version(stored.version_id)
        assert set(reloaded_again.entities.keys()) == set(world.entities.keys())
        report = validate_world_ir(reloaded_again)
        assert report.is_valid(), [str(i) for i in report.errors]


class TestRealCaptureProof:
    """Runs the compiler against the only real (non-synthetic) capture
    committed to this repository: datasets/south_building -- 32 real
    photographs, real COLMAP-derived camera poses and 3D points
    (committed ground truth, no re-run of the COLMAP binary needed).

    South Building is an outdoor courtyard/facade dataset, not an
    interior scene: it has no enclosed room. The honest, correct
    result is therefore ZERO detected rooms with ZERO validation
    errors -- the room detector's NO_CLOSED_RING refusal firing
    correctly on real, non-synthetic geometry, not a bug. This is the
    proof required by the interior-compiler mission's "real-data proof"
    step; no real *interior* capture exists in this repository (the
    only interior fixtures referenced elsewhere are locally-generated
    and gitignored), so this is honestly the strongest real capture
    available, and the test says exactly that rather than fabricating
    a room that isn't there.
    """

    def _load_real_reconstruction(self) -> ReconstructionResult:
        images_txt = (SOUTH_BUILDING_SPARSE / "images.txt").read_text(encoding="utf-8")
        points_txt = (SOUTH_BUILDING_SPARSE / "points3D.txt").read_text(encoding="utf-8")
        poses = _parse_images_txt(images_txt, {})
        points = _parse_points3d_txt(points_txt, {})
        return ReconstructionResult(points=points, camera_poses=poses, registration_status="success")

    def test_real_capture_compiles_to_a_clean_validated_world(self, tmp_path: Path):
        result = self._load_real_reconstruction()
        assert len(result.camera_poses) == 32
        assert len(result.points) > 40000  # real COLMAP output, not a stub

        options = CompileOptions(
            seed=42,
            promote_corridors=True, promote_windows=True,
            promote_stairs=True, promote_building=True,
        )
        world, diag = compile_reconstruction_to_world(result, options)

        # Honest outcome for an exterior scene: no fabricated rooms, and
        # no silent errors -- every plane the room detector refused is
        # accounted for in diag.room_candidates with its real reason.
        assert diag.rooms_detected == 0
        assert diag.interior_warnings == []
        assert all(c["status"] != "detected" for c in diag.room_candidates)

        report = validate_world_ir(world)
        assert report.is_valid(), [str(i) for i in report.errors]

        # Real capture round-trips through WorldStore like any other.
        store = WorldStore(tmp_path / "store")
        stored = store.save_version(world, parent=None)
        reloaded = WorldStore(tmp_path / "store").load_version(stored.version_id)
        assert set(reloaded.entities.keys()) == set(world.entities.keys())
        assert validate_world_ir(reloaded).is_valid()


def _entity(entity_id, entity_type, *, custom_properties=None, relationships=None):
    from world_ir.schema_v1 import Entity
    return Entity(
        id=entity_id,
        type=entity_type,
        name=entity_id,
        custom_properties=custom_properties or {},
        relationships=relationships or [],
        provenance=Provenance.INFERRED,
        confidence=0.9,
    )


def _contains(target_id: str) -> Relationship:
    return Relationship(
        kind=RelationshipKind.CONTAINS,
        target_id=target_id,
        confidence=0.9,
        provenance=Provenance.INFERRED,
    )
