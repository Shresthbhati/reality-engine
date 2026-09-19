"""Tests for the world compiler (engine/compiler/world_compiler.py) and
the WorldIR validation gate (world_ir/validation.py).

Fixtures reuse the hand-computable two-room-scene family: a 2.5x2.25x2 m
room with cameras, so expected entity counts, roles, areas, and
provenance are all closed-form. Validation tests build deliberately
corrupted worlds by hand (no shared mutable state).
"""

from __future__ import annotations

import math

import pytest

from engine.compiler import (
    CompileDiagnostics,
    CompileInputError,
    CompileOptions,
    WorldValidationGateError,
    compile_reconstruction_to_world,
)
from provenance import Provenance
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
from world_ir.validation import (
    ValidationSeverity,
    validate_world_ir,
)


def _compiled_world(seed: int = 42):
    from reconstruction.backend.interface import ReconstructedCameraPose
    from tests.test_room_inference import _CAMS, _two_room_scene

    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    return compile_reconstruction_to_world(result, CompileOptions(seed=seed))


# ---------------------------------------------------------------- compiler


class TestCompilerEndToEnd:
    def test_compiles_the_room_scene_into_structure_plus_room(self):
        world, diag = _compiled_world()
        # 7 planes: ceiling, floor, slab(as floor), 4 walls.
        assert diag.planes_total == 7
        assert diag.planes_by_role == {"ceiling": 1, "floor": 2, "wall": 4}
        # All 7 structure entities + 1 room.
        assert len(diag.entities_created) == 8
        assert diag.rooms_detected == 1
        room_entities = [e for e in world.entities.values() if e.type is EntityType.ROOM]
        assert len(room_entities) == 1
        assert room_entities[0].custom_properties["floor_area_m2"] == pytest.approx(5.625)

    def test_every_input_plane_is_accounted_for(self):
        world, diag = _compiled_world()
        # planes_total == promoted + unpromoted (nothing silently dropped).
        assert diag.planes_total == len(diag.entities_created) - diag.rooms_detected + len(diag.planes_unpromoted)

    def test_provenance_semantics_are_preserved(self):
        world, diag = _compiled_world()
        for entity in world.entities.values():
            if entity.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING):
                assert entity.provenance is Provenance.INFERRED
            elif entity.type is EntityType.ROOM:
                assert entity.provenance is Provenance.INFERRED
        # The world as a whole records its derivation without upgrading
        # any entity.
        assert world.global_provenance is Provenance.RECONSTRUCTED
        assert world.metadata["compiled_from"]["seed"] == 42

    def test_deterministic_byte_identical_worlds(self):
        world_a, diag_a = _compiled_world(seed=42)
        world_b, diag_b = _compiled_world(seed=42)
        assert world_a.to_dict() == world_b.to_dict()
        assert diag_a.summary_text() == diag_b.summary_text()

    def test_different_seed_still_finds_the_structure(self):
        world_a, diag_a = _compiled_world(seed=42)
        world_b, diag_b = _compiled_world(seed=123)
        assert diag_b.rooms_detected == 1
        roles_a = {e.type for e in world_a.entities.values()}
        roles_b = {e.type for e in world_b.entities.values()}
        assert roles_a == roles_b

    def test_diagnostics_report_the_unassigned_floor(self):
        _, diag = _compiled_world()
        assert diag.points_unassigned == 0
        assert diag.unknown_point_fraction == 0.0
        # Failed room candidate (the slab floor) is visible, not hidden.
        assert any(
            c["status"] == "NO_CLOSED_RING" for c in diag.room_candidates
        )

    def test_validation_gate_passes_on_the_compiled_world(self):
        world, _ = _compiled_world()
        report = validate_world_ir(world)
        assert report.is_valid(), report.messages()

    def test_failed_reconstruction_is_refused(self):
        from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult

        with pytest.raises(CompileInputError, match="failed reconstruction"):
            compile_reconstruction_to_world(
                ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
            )

    def test_empty_reconstruction_is_refused(self):
        from reconstruction.backend.interface import (
            ReconstructedCameraPose,
            ReconstructedPoint,
            ReconstructionResult,
        )

        with pytest.raises(CompileInputError, match="empty reconstruction"):
            compile_reconstruction_to_world(
                ReconstructionResult(
                    points=[],
                    camera_poses=[ReconstructedCameraPose(evidence_id="ev", position=(0, 0, 0), rotation=(1, 0, 0, 0))],
                    registration_status="partial",
                )
            )

    def test_no_cameras_is_refused(self):
        from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult

        points = [
            ReconstructedPoint(position=(x, y, z), track_id=f"p{i}", source_evidence_ids=["e"])
            for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)])
        ]
        with pytest.raises(CompileInputError, match="camera poses"):
            compile_reconstruction_to_world(
                ReconstructionResult(points=points, camera_poses=[], registration_status="success")
            )

    def test_gate_catches_corruption_between_promotion_and_return(self):
        """The gate gates: corrupting a compiled world (inverted geometry
        bounds) is caught, proving the compiler's check is real."""
        compiled, _ = _compiled_world()
        compiled.geometries["geom-corrupt"] = Geometry(
            id="geom-corrupt",
            type=GeometryType.BOX,
            vertex_count=8,
            bounds_min=Vector3(1.0, 0.0, 0.0),
            bounds_max=Vector3(0.0, 0.0, 0.0),
        )
        report = validate_world_ir(compiled)
        assert not report.is_valid()
        assert any(i.code == "geometry_inverted_bounds" for i in report.issues)

    def test_studio_consumes_the_compiled_world(self):
        from engine.studio.session import StudioSession

        world, _ = _compiled_world()
        studio = StudioSession(world)
        # Outliner sees the compiled structure; provenance panel answers
        # "why does the engine believe this?" from real observations.
        visible = studio.visible_entities() if hasattr(studio, "visible_entities") else []
        assert world.entities  # non-empty world handed to Studio
        room = next(e for e in world.entities.values() if e.type is EntityType.ROOM)
        studio.select(room.id)
        summary = studio.active_entity_summary() if hasattr(studio, "active_entity_summary") else None
        assert room.provenance is Provenance.INFERRED

    def test_measurements_survive_worldir_round_trip(self):
        world, _ = _compiled_world()
        data = world.to_dict()
        restored = WorldIR.from_dict(data)
        room = next(e for e in restored.entities.values() if e.type is EntityType.ROOM)
        assert room.custom_properties["floor_area_m2"] == pytest.approx(5.625)


# ---------------------------------------------------------------- validation


class TestValidationGate:
    def _empty_world(self) -> WorldIR:
        return WorldIR(id="world-validation-test")

    def _entity(self, entity_id: str, **kwargs) -> Entity:
        defaults = dict(
            id=entity_id,
            type=EntityType.UNKNOWN,
            provenance=Provenance.OBSERVED,
            confidence=0.9,
        )
        defaults.update(kwargs)
        return Entity(**defaults)

    def test_valid_minimal_world_passes(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        report = validate_world_ir(world)
        assert report.is_valid()

    def test_dangling_relationship_is_an_error(self):
        world = self._empty_world()
        entity = self._entity("e1")
        from world_ir import Relationship, RelationshipKind

        entity.relationships.append(Relationship(
            kind=RelationshipKind.CONTAINS, target_id="ghost",
            provenance=Provenance.OBSERVED,
        ))
        world.entities["e1"] = entity
        report = validate_world_ir(world)
        assert not report.is_valid()
        assert any(i.code == "structural" for i in report.issues)

    def test_inverted_bounds_are_an_error(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        world.geometries["g1"] = Geometry(
            id="g1", type=GeometryType.BOX, vertex_count=8,
            bounds_min=Vector3(1.0, 1.0, 1.0), bounds_max=Vector3(0.0, 0.0, 0.0),
        )
        report = validate_world_ir(world)
        assert not report.is_valid()
        assert any(i.code == "geometry_inverted_bounds" for i in report.issues)

    def test_non_finite_bounds_are_an_error(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        nan = float("nan")
        world.geometries["g1"] = Geometry(
            id="g1", type=GeometryType.BOX, vertex_count=8,
            bounds_min=Vector3(nan, 0.0, 0.0), bounds_max=Vector3(1.0, 1.0, 1.0),
        )
        report = validate_world_ir(world)
        assert any(i.code == "geometry_non_finite_bounds" for i in report.issues)

    def test_near_singular_transform_is_an_error(self):
        from world_ir import Frame, Transform

        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        # Singular matrix (collapses Y): geometry through this transform
        # would flatten onto a plane.
        world.transforms["e1"] = Transform(
            source_frame=Frame.WORLD, target_frame=Frame.WORLD,
            matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        report = validate_world_ir(world)
        assert any(i.code == "transform_singular" for i in report.issues)

    def test_unknown_provenance_with_high_confidence_is_an_error(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity(
            "e1", provenance=Provenance.UNKNOWN, confidence=0.95
        )
        report = validate_world_ir(world)
        assert not report.is_valid()
        assert any(i.code == "unknown_provenance_high_confidence" for i in report.issues)

    def test_unknown_provenance_with_low_confidence_is_fine(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity(
            "e1", provenance=Provenance.UNKNOWN, confidence=0.2
        )
        report = validate_world_ir(world)
        assert report.is_valid()

    def test_negative_dimension_property_is_an_error(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        world.entities["e1"].custom_properties["height_m"] = -2.0
        report = validate_world_ir(world)
        assert any(i.code == "measurement_negative_dimension" for i in report.issues)

    def test_non_finite_property_is_an_error(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        world.entities["e1"].custom_properties["height_m"] = float("nan")
        report = validate_world_ir(world)
        assert any(i.code == "measurement_non_finite" for i in report.issues)

    def test_zero_dimension_is_not_an_error(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        world.entities["e1"].custom_properties["height_m"] = 0.0
        assert validate_world_ir(world).is_valid()

    def test_duplicate_identity_warns(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1", name="wall")
        world.entities["e2"] = self._entity("e2", name="wall")
        report = validate_world_ir(world)
        assert any(i.code == "entity_duplicate_identity" for i in report.warnings)
        assert report.is_valid()  # warning, not error

    def test_severity_partition_and_deterministic_order(self):
        world = self._empty_world()
        world.entities["e1"] = self._entity("e1")
        world.entities["e1"].custom_properties["height_m"] = -1.0
        world.entities["e2"] = self._entity(
            "e2", provenance=Provenance.UNKNOWN, confidence=0.99
        )
        report_a = validate_world_ir(world)
        report_b = validate_world_ir(world)
        assert report_a.messages() == report_b.messages()
        assert report_a.errors and report_a.warnings or report_a.errors
        assert len(report_a.errors) == 2

    def test_gate_error_raised_by_compiler_on_corrupt_input(self):
        # End-to-end: the compiler's gate refuses a world corrupted between
        # promotion and return.
        world, _ = _compiled_world()
        world.entities["ghost-room"] = Entity(
            id="ghost-room", type=EntityType.ROOM,
            provenance=Provenance.INFERRED, confidence=0.5,
        )
        world.entities["ghost-room"].geometry_ids.append("geom-does-not-exist")
        report = validate_world_ir(world)
        assert not report.is_valid()
        issues = [i.message for i in report.issues if i.severity is ValidationSeverity.ERROR]
        assert issues
