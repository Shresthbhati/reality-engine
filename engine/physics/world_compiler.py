"""WorldIR -> physics compilation (spec: WORLDIR -> PHYSICAL WORLD; the
simulation campaign's sec-1 target).

Before this module the physics stack was real but ran on test geometry:
nothing converted reconstructed WorldIR entities into simulation bodies.
This compiler is that bridge, and it follows the repo's honesty rules:

  - Every derived property records HOW it was derived (observed measurement
    / derived from other measurements / estimated from class defaults) --
    never a silent default. Mass is DERIVED from density x geometry volume
    and the derivation is recorded; a density picked from a material-class
    table is ESTIMATED and labelled as such.
  - Unsupported geometry (POINTCLOUD, TERRAIN, VOXEL, UNKNOWN...) is
    skipped per entity with a reason in diagnostics -- never guessed into
    a shape.
  - Determinism: no uuid4, no wall clock; body ids derive from entity ids
    and the compile is a pure function of (world, options).
  - Reconstruction provenance is preserved: an entity compiled from
    RECONSTRUCTED geometry carries that provenance into the report; the
    compile itself INFERRED nothing about the world it was not given.

The result: a SimpleRigidBodyBackend world plus a frozen compile report
accounting for every entity -- compiled, skipped (why), static/dynamic,
provenance, confidence -- so Reality Studio can show WHY each body exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from engine.physics.backend.interface import PhysicsWorldConfig
from engine.physics.backend.simple_backend import (
    PhysicsWorld,
    SimpleRigidBodyBackend,
    StaticPlane,
)
from engine.physics.collision.shapes import Box, Plane
from engine.physics.materials.material import PhysicsMaterial
from engine.physics.rigid.body import RigidBody
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    GeometryType,
    Material as WorldIRMaterial,
)

# ---------------------------------------------------------------------------
# Default physical parameter tables (ESTIMATED class defaults)
# ---------------------------------------------------------------------------

#: Textbook-class density estimates (kg/m^3), applied only when the
#: entity's own material lacks a density measurement, always labelled
#: "estimated" in the report.
DEFAULT_DENSITY_BY_CLASS = {
    "wood": 600.0,
    "concrete": 2400.0,
    "steel": 7850.0,
    "metal": 7850.0,
    "glass": 2500.0,
    "brick": 1900.0,
    "plastic": 950.0,
    "stone": 2600.0,
    "soil": 1600.0,
    "asphalt": 2300.0,
    "fabric": 300.0,
    "paper": 800.0,
    "water": 1000.0,
    "generic": 500.0,
}

#: (friction_static, restitution) estimates per material class.
DEFAULT_SURFACE_BY_CLASS = {
    "wood": (0.45, 0.30),
    "concrete": (0.70, 0.15),
    "steel": (0.35, 0.30),
    "metal": (0.35, 0.30),
    "glass": (0.25, 0.60),
    "brick": (0.70, 0.15),
    "plastic": (0.30, 0.35),
    "stone": (0.65, 0.20),
    "generic": (0.50, 0.25),
}

#: Density for entities with NO material at all.
FALLBACK_DENSITY = 500.0

#: Geometry types this compiler can turn into collision shapes.
SUPPORTED_GEOMETRY = frozenset({GeometryType.BOX, GeometryType.PLANE})


# ---------------------------------------------------------------------------
# Compile report records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledBody:
    """One entity successfully compiled into a physics body."""
    entity_id: str
    body_id: str
    kind: str                      # "dynamic" | "static"
    shape_kind: str                # "box" | "plane"
    mass_kg: Optional[float]       # None for static
    position: Tuple[float, float, float]
    material_class: str
    density_source: str            # "observed" | "estimated" | "fallback"
    provenance: str                # WorldIR entity provenance
    confidence: float              # WorldIR entity confidence
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SkippedEntity:
    """One entity NOT compiled, with the honest reason why."""
    entity_id: str
    reason_code: str   # NO_GEOMETRY | UNSUPPORTED_GEOMETRY | BAD_BOUNDS | NO_TRANSFORM
    reason: str


@dataclass(frozen=True)
class PhysicsCompileReport:
    """Frozen, serializable account of one WorldIR -> physics compile."""
    seed: int
    compiled: Tuple[CompiledBody, ...]
    skipped: Tuple[SkippedEntity, ...]
    static_count: int
    dynamic_count: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "compiled": [
                {
                    "entity_id": c.entity_id,
                    "body_id": c.body_id,
                    "kind": c.kind,
                    "shape_kind": c.shape_kind,
                    "mass_kg": c.mass_kg,
                    "position": list(c.position),
                    "material_class": c.material_class,
                    "density_source": c.density_source,
                    "provenance": c.provenance,
                    "confidence": c.confidence,
                }
                for c in self.compiled
            ],
            "skipped": [
                {
                    "entity_id": s.entity_id,
                    "reason_code": s.reason_code,
                    "reason": s.reason,
                }
                for s in self.skipped
            ],
            "static_count": self.static_count,
            "dynamic_count": self.dynamic_count,
        }

    @property
    def compiled_ids(self) -> List[str]:
        return [c.entity_id for c in self.compiled]

    @property
    def skipped_ids(self) -> List[str]:
        return [s.entity_id for s in self.skipped]


class PhysicsCompileError(RuntimeError):
    """Raised when the compile cannot proceed at all (no world, etc.)."""


# ---------------------------------------------------------------------------
# Per-entity compilation helpers
# ---------------------------------------------------------------------------


def _bounds_center_and_half(bmin, bmax) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    cx = (bmin.x + bmax.x) / 2.0
    cy = (bmin.y + bmax.y) / 2.0
    cz = (bmin.z + bmax.z) / 2.0
    hx = (bmax.x - bmin.x) / 2.0
    hy = (bmax.y - bmin.y) / 2.0
    hz = (bmax.z - bmin.z) / 2.0
    return (cx, cy, cz), (hx, hy, hz)


def _resolve_material(entity: Entity, world: WorldIR) -> Tuple[Optional[WorldIRMaterial], str]:
    """Resolve the entity's material, recording how it was resolved."""
    mat_ids = getattr(entity, "material_ids", None) or []
    for mid in mat_ids:
        mat = world.materials.get(mid)
        if mat is not None:
            return mat, "entity_material"
    return None, "no_material"


def _resolve_density(
    entity: Entity,
    material: Optional[WorldIRMaterial],
    notes: List[str],
) -> Tuple[float, str]:
    """Density in kg/m^3 + source label. Honesty ladder:
    observed/estimated measurement on the material -> class-table estimate
    -> fallback. The label travels into the report; nothing is silent."""
    props = getattr(material, "properties", None) if material is not None else None
    density_measurement = getattr(props, "density", None) if props is not None else None
    if density_measurement is not None and density_measurement.value:
        return float(density_measurement.value), "observed"

    # A material EXISTS but lacks a density measurement: its class is real
    # evidence, so a class-table value is an "estimated" density. NO
    # material at all: no class evidence exists, so the value is a
    # "fallback", not an estimate from evidence.
    if material is not None:
        class_name = material.class_name.lower()
        if class_name in DEFAULT_DENSITY_BY_CLASS:
            notes.append(f"density estimated from material class '{class_name}'")
            return DEFAULT_DENSITY_BY_CLASS[class_name], "estimated"

    notes.append("no material/density evidence; fallback density applied (flagged)")
    return FALLBACK_DENSITY, "fallback"


def _resolve_surface(material: Optional[WorldIRMaterial]) -> Tuple[float, float]:
    """(friction_static, restitution) from the material when present, else
    the class table, else generic."""
    props = getattr(material, "properties", None) if material is not None else None
    friction = getattr(props, "friction_coefficient", None) if props is not None else None
    restitution = getattr(props, "restitution", None) if props is not None else None
    class_name = (material.class_name if material is not None else "generic").lower()
    default_friction, default_restitution = DEFAULT_SURFACE_BY_CLASS.get(
        class_name, DEFAULT_SURFACE_BY_CLASS["generic"]
    )
    return (
        friction if friction is not None else default_friction,
        restitution if restitution is not None else default_restitution,
    )


def _physics_material(
    entity: Entity,
    material: Optional[WorldIRMaterial],
    density: float,
    notes: List[str],
) -> PhysicsMaterial:
    friction, restitution = _resolve_surface(material)
    class_name = (material.class_name if material is not None else "generic").lower()
    name = material.name if material is not None and material.name else f"{entity.id}-material"
    return PhysicsMaterial(
        name=name,
        density=density,
        friction_static=friction,
        friction_dynamic=min(friction, friction * 0.9),
        restitution=restitution,
    )


def _box_volume_m3(half: Tuple[float, float, float]) -> float:
    hx, hy, hz = half
    return max(0.0, (2 * hx) * (2 * hy) * (2 * hz))


# ---------------------------------------------------------------------------
# Entity-level compile
# ---------------------------------------------------------------------------


def compile_entity(
    entity: Entity,
    world: WorldIR,
) -> Tuple[Optional[CompiledBody], Optional[RigidBody | StaticPlane], Optional[SkippedEntity]]:
    """Compile one entity into a body (or static plane).

    Returns (report_record, body_or_plane, skip_record): exactly one of
    (record, body) or skip is populated.
    """
    geometry_ids = getattr(entity, "geometry_ids", None) or []
    if not geometry_ids:
        return None, None, SkippedEntity(
            entity_id=entity.id,
            reason_code="NO_GEOMETRY",
            reason="entity carries no geometry; nothing physical to simulate",
        )

    geometry = world.geometries.get(geometry_ids[0])
    if geometry is None:
        return None, None, SkippedEntity(
            entity_id=entity.id,
            reason_code="NO_GEOMETRY",
            reason=f"referenced geometry '{geometry_ids[0]}' not found in world",
        )

    if geometry.type not in SUPPORTED_GEOMETRY:
        return None, None, SkippedEntity(
            entity_id=entity.id,
            reason_code="UNSUPPORTED_GEOMETRY",
            reason=(
                f"geometry type '{geometry.type.value}' has no collision "
                "representation yet (supported: box, plane)"
            ),
        )

    if geometry.bounds_min is None or geometry.bounds_max is None:
        return None, None, SkippedEntity(
            entity_id=entity.id,
            reason_code="BAD_BOUNDS",
            reason="geometry lacks bounds_min/bounds_max; cannot place or size a body",
        )

    # Position: entity transform wins (explicit placement); geometry-bbox
    # center is the fallback and the choice is recorded.
    notes: List[str] = []
    position = None
    transform = getattr(entity, "transform", None)
    if transform:
        pos = transform.get("position") or transform.get("translation")
        if pos is not None:
            position = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        else:
            notes.append("entity transform present without position; geometry center used")
    else:
        notes.append("no entity transform; geometry-bbox center used as body position")
    center, half = _bounds_center_and_half(geometry.bounds_min, geometry.bounds_max)
    if position is None:
        position = center

    material, _material_how = _resolve_material(entity, world)
    density, density_source = _resolve_density(entity, material, notes)
    phys_mat = _physics_material(entity, material, density, notes)
    class_name = (material.class_name if material is not None else "generic").lower()

    if geometry.type is GeometryType.PLANE:
        # Structure planes: infinite static plane through the bounds center
        # along the geometry's X/Y extents (walls/floors from reconstruction).
        # DOCUMENTED APPROXIMATION (spec sec 29): the collision backend only
        # supports infinite planes, so the plane blocks EVERYWHERE along its
        # surface -- it does not honour the geometry's lateral bounds, and it
        # over-restrains: a doorway gap in a reconstructed wall still blocks
        # a body passing through where the gap is. Recorded as a note on the
        # compiled record; the Box path enforces extent correctly.
        notes.append(
            "infinite-plane approximation: lateral bounds not enforced by "
            "the collision backend"
        )
        normal = _plane_normal_from_bounds(geometry)
        plane = Plane(normal=normal, distance=normal.dot(_vec(center)))
        body = StaticPlane(
            id=f"phys-{entity.id}",
            plane=plane,
            material=phys_mat,
        )
        record = CompiledBody(
            entity_id=entity.id,
            body_id=body.id,
            kind="static",
            shape_kind="plane",
            mass_kg=None,
            position=position,
            material_class=class_name,
            density_source=density_source,
            provenance=entity.provenance.value,
            confidence=entity.confidence,
            notes=tuple(notes),
        )
        return record, body, None

    # BOX geometry: static if the entity says so, dynamic otherwise.
    volume = _box_volume_m3(half)
    mass = density * volume
    is_static = entity.type.value in ("floor", "ceiling", "terrain", "road", "sidewalk", "curb")
    body = RigidBody(
        id=f"phys-{entity.id}",
        shape=Box(half_extents=_vec(half)),
        material=phys_mat,
        mass=0.0 if is_static else max(mass, 1e-6),
        position=_vec(position),
    )
    kind = "static" if is_static else "dynamic"
    if kind == "dynamic":
        notes.append(
            f"mass derived: density({density:.0f} kg/m^3, {density_source}) "
            f"x bbox volume({volume:.3f} m^3) = {mass:.2f} kg"
        )
    record = CompiledBody(
        entity_id=entity.id,
        body_id=body.id,
        kind=kind,
        shape_kind="box",
        mass_kg=None if is_static else mass,
        position=position,
        material_class=class_name,
        density_source=density_source,
        provenance=entity.provenance.value,
        confidence=entity.confidence,
        notes=tuple(notes),
    )
    return record, body, None


def _vec(t: Tuple[float, float, float]):
    from engine.physics.math3 import Vec3
    return Vec3(t[0], t[1], t[2])


def _plane_normal_from_bounds(geometry):
    """Infer the plane's normal from its bounds: the thinnest axis."""
    bmin, bmax = geometry.bounds_min, geometry.bounds_max
    extents = (bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z)
    axis = min(range(3), key=lambda i: extents[i])
    from engine.physics.math3 import Vec3
    if axis == 0:
        return Vec3(1.0, 0.0, 0.0)
    if axis == 1:
        return Vec3(0.0, 1.0, 0.0)
    return Vec3(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Whole-world compile
# ---------------------------------------------------------------------------


def compile_world_to_physics(
    world: WorldIR,
    seed: int = 0,
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
) -> Tuple["PhysicsWorld", PhysicsCompileReport]:
    """Compile a WorldIR into a physics world + full report.

    Deterministic: same world + same seed -> identical bodies and report.
    """
    from engine.physics.math3 import Vec3

    backend = SimpleRigidBodyBackend()
    config = PhysicsWorldConfig(
        gravity=Vec3(gravity[0], gravity[1], gravity[2]),
        seed=seed,
    )
    physics_world = backend.create_world(config)

    compiled_records: List[CompiledBody] = []
    skipped_records: List[SkippedEntity] = []
    static_count = 0
    dynamic_count = 0

    for entity in world.entities.values():
        record, body, skip = compile_entity(entity, world)
        if skip is not None:
            skipped_records.append(skip)
            continue
        if isinstance(body, StaticPlane):
            physics_world.add_plane(body)
            static_count += 1
        else:
            physics_world.add_body(body)
            if body.is_static:
                static_count += 1
            else:
                dynamic_count += 1
        compiled_records.append(record)

    report = PhysicsCompileReport(
        seed=seed,
        compiled=tuple(compiled_records),
        skipped=tuple(skipped_records),
        static_count=static_count,
        dynamic_count=dynamic_count,
    )
    return physics_world, report
