"""Tests for the WorldIR -> physics compiler (engine/physics/world_compiler.py).

Fixtures: synthetic hand-built WorldIRs with closed-form geometry (boxes of
known extents and materials carrying known measurements) so every expected
mass, provenance label, and skip reason is hand-computable -- plus the real
end-to-end chain (reconstructed room scene -> WorldIR -> physics -> a body
at rest on a compiled floor).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.physics

try:
    from engine.physics.world_compiler import (
        CompiledBody,
        PhysicsCompileReport,
        compile_entity,
        compile_world_to_physics,
    )
except ImportError:
    pytest.skip("engine.physics module not available - requires reality-engine-child", allow_module_level=True)
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Measurement,
    PhysicalProperties,
    Vector3,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _box_world(
    geometry_type=GeometryType.BOX,
    bmin=(0.0, 0.0, 0.0),
    bmax=(2.0, 2.0, 2.0),
    entity_type="unknown",
    material_class="wood",
    density=None,
    entity_provenance=Provenance.RECONSTRUCTED,
) -> WorldIR:
    """One entity, one geometry, optional material with optional density."""
    world = WorldIR(
        id="w-test",
        name="test",
        main_branch_id="b-test",
    )
    geometry = Geometry(
        id="geom-1",
        type=geometry_type,
        bounds_min=Vector3(*bmin),
        bounds_max=Vector3(*bmax),
        provenance=entity_provenance,
        confidence=0.9,
    )
    material = None
    if material_class is not None:
        props = PhysicalProperties()
        if density is not None:
            props.density = Measurement(
                value=density, unit="kilogram_per_meter_cubed", precision=1.0,
                provenance=Provenance.OBSERVED, confidence=0.95,
            )
        material = Material(
            id="mat-1",
            class_name=material_class,
            properties=props,
            provenance=Provenance.ESTIMATED,
            confidence=0.7,
        )
        world.materials["mat-1"] = material
    entity = Entity(
        id="ent-1",
        type=EntityType(entity_type),
        name="test entity",
        geometry_ids=["geom-1"],
        material_ids=["mat-1"] if material is not None else [],
        provenance=entity_provenance,
        confidence=0.85,
    )
    world.geometries["geom-1"] = geometry
    world.entities["ent-1"] = entity
    return world


def _room_world():
    """The real end-to-end source: reconstructed room scene -> WorldIR."""
    from tests.test_room_inference import _CAMS, _two_room_scene
    from reconstruction.backend.interface import ReconstructedCameraPose
    from engine.compiler.world_compiler import CompileOptions, compile_reconstruction_to_world

    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    world, _diag = compile_reconstruction_to_world(result, CompileOptions(seed=7))
    return world


# ---------------------------------------------------------------------------
# Entity-level compilation
# ---------------------------------------------------------------------------


class TestCompileEntity:
    def test_box_entity_mass_derived_and_recorded(self):
        """2x2x2 m box, observed density 800: mass = 800 * 8 = 6400 kg."""
        world = _box_world(density=800.0)
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert skip is None
        assert record.mass_kg == pytest.approx(6400.0)
        assert record.density_source == "observed"
        assert record.kind == "dynamic"
        assert any("density(800 kg/m^3, observed)" in n for n in record.notes)

    def test_no_density_uses_class_estimate_labelled(self):
        world = _box_world(material_class="steel", density=None)
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert record.density_source == "estimated"
        # 7850 kg/m^3 * 8 m^3
        assert record.mass_kg == pytest.approx(7850.0 * 8.0)
        assert any("class 'steel'" in n for n in record.notes)

    def test_no_material_falls_back_and_flags(self):
        world = _box_world(material_class=None)
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert record.density_source == "fallback"
        assert any("flagged" in n for n in record.notes)

    def test_missing_geometry_is_skipped_with_reason(self):
        world = _box_world()
        world.entities["ent-1"].geometry_ids = []
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert record is None and body is None
        assert skip.reason_code == "NO_GEOMETRY"

    def test_unsupported_geometry_is_skipped_not_guessed(self):
        world = _box_world(geometry_type=GeometryType.POINTCLOUD)
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert skip.reason_code == "UNSUPPORTED_GEOMETRY"
        assert "pointcloud" in skip.reason

    def test_missing_bounds_is_skipped(self):
        world = _box_world()
        world.geometries["geom-1"].bounds_min = None
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert skip.reason_code == "BAD_BOUNDS"

    def test_structure_types_compile_static(self):
        world = _box_world(entity_type="floor", material_class="concrete", density=2400.0)
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert record.kind == "static"
        assert record.mass_kg is None
        assert body.mass == 0.0

    def test_dynamic_body_id_derives_from_entity_id(self):
        world = _box_world()
        record, body, skip = compile_entity(world.entities["ent-1"], world)
        assert body.id == "phys-ent-1"


# ---------------------------------------------------------------------------
# Whole-world compile
# ---------------------------------------------------------------------------


class TestCompileWorld:
    def test_room_scene_compiles_planes_and_skips_room(self):
        phys, report = compile_world_to_physics(_room_world(), seed=7)
        assert len(report.compiled) == 7
        assert report.skipped_ids == ["room-plane-000"]  # ROOM: no geometry
        assert report.static_count == 7 and report.dynamic_count == 0
        assert all(c.shape_kind == "plane" for c in report.compiled)
        assert all("infinite-plane approximation" in " ".join(c.notes) for c in report.compiled)

    def test_compile_is_deterministic(self):
        w = _room_world()
        p1, r1 = compile_world_to_physics(w, seed=7)
        p2, r2 = compile_world_to_physics(w, seed=7)
        assert r1.to_dict() == r2.to_dict()
        assert [b.id for b in p1.bodies.values()] == [b.id for b in p2.bodies.values()]

    def test_report_serializes(self):
        _, report = compile_world_to_physics(_room_world(), seed=7)
        d = report.to_dict()
        assert d["static_count"] == 7
        assert d["compiled"][0]["body_id"].startswith("phys-")
        assert d["skipped"][0]["reason_code"] == "NO_GEOMETRY"


# ---------------------------------------------------------------------------
# End-to-end: compiled world actually simulates
# ---------------------------------------------------------------------------


class TestCompiledWorldSimulates:
    """Simulation tests run on a FLOOR-ONLY compiled world. The full room
    compiles every plane to an infinite zero-thickness sheet; dense sheets
    make indoor spawns degenerate (a box between two sheets gets ejected).
    That over-restraint is a documented approximation of the plane path,
    pinned separately below; the floor-only world proves the BRIDGE."""

    def _floor_only_world(self) -> WorldIR:
        world = _box_world(
            geometry_type=GeometryType.PLANE,
            bmin=(0.0, 0.0, 0.0),
            bmax=(2.5, 0.0, 2.25),
            entity_type="floor",
            material_class="concrete",
            density=2400.0,
        )
        return world

    def test_dynamic_box_falls_and_rests_on_compiled_floor(self):
        """A dynamic box added above a compiled floor plane must fall,
        contact it, and come to rest ON it -- proof the bridge produces a
        real, simulatable world from WorldIR."""
        try:
            from engine.physics.backend.simple_backend import SimpleRigidBodyBackend
            from engine.physics.math3 import Vec3
            from engine.physics.collision.shapes import Box
            from engine.physics.materials.material import PhysicsMaterial
            from engine.physics.rigid.body import RigidBody
        except ImportError:
            pytest.skip("engine.physics module not available - requires reality-engine-child")
        phys, report = compile_world_to_physics(self._floor_only_world(), seed=7)
        assert report.static_count == 1

        phys.add_body(RigidBody(
            id="crate",
            shape=Box(half_extents=Vec3(0.2, 0.2, 0.2)),
            material=PhysicsMaterial(
                name="crate", density=600.0,
                friction_static=0.5, friction_dynamic=0.45, restitution=0.2,
            ),
            mass=40.0,
            position=Vec3(1.25, 0.6, 1.0),
        ))

        backend = SimpleRigidBodyBackend()
        for _ in range(600):  # 6 s at 100 Hz through the backend interface
            backend.step(phys, 0.01)

        crate = phys.get_body("crate")
        assert crate.position.y == pytest.approx(0.2, abs=0.05), (
            f"crate must rest on the compiled floor plane, got y={crate.position.y}"
        )
        assert abs(crate.linear_velocity.y) < 0.05
        # NOTE (observed, engine characteristic): the body is at rest but
        # stays AWAKE -- the integrator applies gravity before the sleep
        # timer check each step, so a resting-on-plane body never
        # accumulates still-time. Sleep evaluation order is an engine-level
        # concern, deliberately not patched in the compiler milestone.

    def test_infinite_plane_over_restrains_at_wall(self):
        """PIN of the documented approximation (spec sec 29: simplifications
        that preserve the causal abstraction, explicitly documented): a
        compiled wall is an INFINITE plane, so a slider pushed toward it is
        stopped even though real reconstruction geometry has finite extent.
        Asserted so a future extent-enforcing plane consciously flips it."""
        try:
            from engine.physics.backend.simple_backend import SimpleRigidBodyBackend
            from engine.physics.math3 import Vec3
            from engine.physics.collision.shapes import Box
            from engine.physics.materials.material import PhysicsMaterial
            from engine.physics.rigid.body import RigidBody
        except ImportError:
            pytest.skip("engine.physics module not available - requires reality-engine-child")
        # Wall-only world: one vertical sheet at z=0.
        wall_world = _box_world(
            geometry_type=GeometryType.PLANE,
            bmin=(0.0, 0.0, 0.0),
            bmax=(2.5, 2.0, 0.0),
            entity_type="wall",
            material_class="concrete",
        )
        phys, _ = compile_world_to_physics(wall_world, seed=7)
        phys.add_body(RigidBody(
            id="slider",
            shape=Box(half_extents=Vec3(0.1, 0.1, 0.1)),
            material=PhysicsMaterial(
                name="slider", density=600.0,
                friction_static=0.3, friction_dynamic=0.25, restitution=0.1,
            ),
            mass=5.0,
            position=Vec3(1.0, 0.5, 1.0),
            linear_velocity=Vec3(0.0, 0.0, -1.0),  # toward the wall sheet
        ))
        backend = SimpleRigidBodyBackend()
        for _ in range(300):  # 3 s at 1 m/s: real gap would let it pass z<0
            backend.step(phys, 0.01)
        slider = phys.get_body("slider")
        assert slider.position.z > 0.10, (
            "infinite-plane approximation must stop the slider at the wall "
            "sheet; if this flips, extent enforcement landed"
        )
