"""WorldIR -> Physics compiler bridge (simulation campaign, item 1).

Before this module, `engine/physics/` (rigid bodies, materials, collision
shapes, fracture, fluids, fire, structural, disasters -- a large, real
physics stack) had zero references to WorldIR anywhere in the repo, and
`evidence/`/`engine/compiler/` had zero references to physics. Every
physics golden scene was hand-built `RigidBody` instances; nothing
connected a reconstructed world to the simulator it's meant to feed.
This module is that bridge.

For each WorldIR Entity with real geometry bounds (BOX/PLANE, whichever
type has `bounds_min`/`bounds_max` set -- see world_ir/schema_v1.py), it
derives a `RigidBody`:

  - collision shape: an axis-aligned `Box` sized from the entity's own
    AABB (real data, same source `exporters/blender/exporter.py` uses),
    each axis floored at 1 cm so a degenerate (zero-thickness) plane
    bound still produces a usable, non-fabricated shape;
  - staticness: entities of a structural/boundary EntityType (WALL,
    FLOOR, CEILING, ROOF, COLUMN, BEAM, STRUCTURE, BUILDING, TERRAIN,
    ROAD, CURB, SIDEWALK, INFRASTRUCTURE) compile to mass=0 (static,
    never integrated) -- the same default any physics engine gives a
    level's fixed geometry, made explicit rather than assumed by every
    caller separately. This is an INFERRED classification from the
    entity's semantic type, not an observed physical fact, and is
    recorded as such in the diagnostics;
  - mass for everything else: `density(material) * AABB volume` -- a
    real derived quantity (ESTIMATED provenance), never an arbitrary
    constant;
  - material: read from `entity.custom_properties["physics_material"]`
    when it names a `CANONICAL_MATERIALS` entry (real evidence-backed
    choice); otherwise falls back to a named default at explicitly low
    confidence -- never silently presented as equivalent to a real
    material observation.

Entities with no geometry, or geometry with no real bounds (the same
"AABB not set" case `world_ir/validation.py` flags as an error for
BOX/PLANE types), are skipped with an explicit diagnostic rather than
defaulted into a fake unit-cube body -- Phase 1's "invalid or
insufficient physical properties must produce explicit diagnostics
rather than silent defaults."

Not built here (explicitly out of scope for this bridge): inertia
tensors beyond what `RigidBody.__post_init__` already derives from
shape+mass, constraints/joints, support-relationship inference from the
scene graph, structural connectivity -- those are later, larger
simulation-campaign items (collision expansion, structural graph) that
consume this bridge's output rather than belong in it.

Additive, opt-in enhancement: true infinite-plane statics. Box bodies
(above) are the default and already collide correctly -- this is a real
enhancement, not a bug fix. `engine/physics/backend/simple_backend.py`'s
`PhysicsWorld` also supports true infinite `Plane` statics via
`add_plane()`, which nothing populated until now. `derive_plane_for_entity`
derives a `Plane` from a static entity's real AABB (thinnest-axis normal,
center-based distance -- see its docstring for the exact convention), and
`build_stepped_physics_world(..., use_planes_for_statics=True)` uses it to
add certain static entities as real Planes instead of thick Box slabs. The
default (`use_planes_for_statics=False`) reproduces prior behavior exactly:
no plane derivation happens unless a caller opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR

from engine.physics.backend.interface import PhysicsWorldConfig
from engine.physics.backend.simple_backend import PhysicsWorld, SimpleRigidBodyBackend, StaticPlane
from engine.physics.collision.shapes import Box, Plane
from engine.physics.materials.material import CANONICAL_MATERIALS, PhysicsMaterial
from engine.physics.math3 import Vec3
from engine.physics.rigid.body import RigidBody

#: Entity types treated as structurally static (infinite mass) by
#: default -- fixed boundary/load-bearing elements a reconstructed scene
#: implies are immovable absent contrary evidence. This is a semantic
#: inference, not a structural-engineering claim (see PHASE 12 of the
#: simulation campaign, which is separate, larger work).
_STATIC_ENTITY_TYPES = frozenset({
    EntityType.WALL, EntityType.FLOOR, EntityType.CEILING, EntityType.ROOF,
    EntityType.STRUCTURE, EntityType.BUILDING, EntityType.TERRAIN,
    EntityType.COLUMN, EntityType.BEAM, EntityType.ROAD, EntityType.CURB,
    EntityType.SIDEWALK, EntityType.INFRASTRUCTURE,
})

#: Used only when the entity carries no physics-material evidence at
#: all. Concrete is a reasonable order-of-magnitude default for
#: reconstructed structural geometry, but the low confidence below is
#: the honest signal that this is a fallback, not an observation.
_DEFAULT_MATERIAL_NAME = "concrete"
_DEFAULT_MATERIAL_CONFIDENCE = 0.1

#: Floor for any derived AABB axis, matching exporters/blender/exporter.py's
#: rule: a degenerate (zero-thickness) plane bound must still produce a
#: usable, visible/collidable shape rather than a zero-volume one.
_MIN_DIMENSION_M = 0.01


class PhysicsCompileStatus(str, Enum):
    COMPILED = "compiled"
    SKIPPED_NO_GEOMETRY = "skipped_no_geometry"
    SKIPPED_NO_BOUNDS = "skipped_no_bounds"


@dataclass(frozen=True)
class EntityPhysicsResult:
    entity_id: str
    status: PhysicsCompileStatus
    body: Optional[RigidBody] = None
    #: "custom_properties" when a real physics_material was named on the
    #: entity, "default" when the low-confidence fallback was used, ""
    #: when nothing was compiled at all.
    material_source: str = ""
    mass_provenance: Provenance = Provenance.UNKNOWN
    confidence: float = 0.0
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "status": self.status.value,
            "body": self.body.to_dict() if self.body is not None else None,
            "material_source": self.material_source,
            "mass_provenance": self.mass_provenance.value,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PhysicsCompileDiagnostics:
    results: Tuple[EntityPhysicsResult, ...] = ()

    @property
    def compiled(self) -> Tuple[EntityPhysicsResult, ...]:
        return tuple(r for r in self.results if r.status is PhysicsCompileStatus.COMPILED)

    @property
    def skipped(self) -> Tuple[EntityPhysicsResult, ...]:
        return tuple(r for r in self.results if r.status is not PhysicsCompileStatus.COMPILED)

    def bodies(self) -> Dict[str, RigidBody]:
        """entity_id -> RigidBody for every entity that actually compiled."""
        return {r.entity_id: r.body for r in self.compiled}

    def to_dict(self) -> dict:
        return {
            "compiled_count": len(self.compiled),
            "skipped_count": len(self.skipped),
            "results": [r.to_dict() for r in self.results],
        }


def _entity_geometry_with_bounds(world: WorldIR, entity: Entity):
    """The first geometry attached to `entity` with real AABB bounds set,
    or None. Type-agnostic (BOX/PLANE/etc.) -- what matters here is
    whether real bounds exist, same test exporters/blender/exporter.py
    uses for "is there real data to size a shape from"."""
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.bounds_min is not None and geom.bounds_max is not None:
            return geom
    return None


def _material_for(entity: Entity) -> Tuple[PhysicsMaterial, str, float]:
    name = entity.custom_properties.get("physics_material")
    if isinstance(name, str) and name in CANONICAL_MATERIALS:
        return CANONICAL_MATERIALS[name], "custom_properties", entity.confidence
    return (
        CANONICAL_MATERIALS[_DEFAULT_MATERIAL_NAME],
        "default",
        min(entity.confidence, _DEFAULT_MATERIAL_CONFIDENCE),
    )


def compile_entity_physics(entity: Entity, world: WorldIR) -> EntityPhysicsResult:
    """Compile one WorldIR entity into a RigidBody, or an explicit
    skip diagnostic when there is nothing real to derive one from."""
    geometry = _entity_geometry_with_bounds(world, entity)
    if geometry is None:
        if entity.geometry_ids:
            return EntityPhysicsResult(
                entity_id=entity.id,
                status=PhysicsCompileStatus.SKIPPED_NO_BOUNDS,
                notes=("entity has geometry but no geometry with real bounds_min/bounds_max "
                       "-- refusing to fabricate a collision shape",),
            )
        return EntityPhysicsResult(
            entity_id=entity.id,
            status=PhysicsCompileStatus.SKIPPED_NO_GEOMETRY,
            notes=("entity has no geometry at all -- nothing to derive a physics body from",),
        )

    bmin, bmax = geometry.bounds_min, geometry.bounds_max
    dx = max(_MIN_DIMENSION_M, bmax.x - bmin.x)
    dy = max(_MIN_DIMENSION_M, bmax.y - bmin.y)
    dz = max(_MIN_DIMENSION_M, bmax.z - bmin.z)
    center = Vec3((bmin.x + bmax.x) / 2.0, (bmin.y + bmax.y) / 2.0, (bmin.z + bmax.z) / 2.0)
    shape = Box(half_extents=Vec3(dx / 2.0, dy / 2.0, dz / 2.0))

    material, material_source, material_confidence = _material_for(entity)

    notes = []
    if entity.type in _STATIC_ENTITY_TYPES:
        mass = 0.0
        mass_provenance = Provenance.INFERRED
        notes.append(f"entity type {entity.type.value} classified static (INFERRED, not observed)")
    else:
        volume = dx * dy * dz
        mass = material.density * volume
        mass_provenance = Provenance.ESTIMATED
        notes.append(
            f"mass = {material.name} density {material.density} kg/m^3 x AABB volume "
            f"{volume:.4f} m^3 = {mass:.3f} kg (ESTIMATED, not observed)"
        )

    if material_source == "default":
        notes.append(
            f"entity carries no 'physics_material' evidence -- defaulted to "
            f"'{material.name}' at confidence {material_confidence:.2f}"
        )

    body = RigidBody(id=f"phys-{entity.id}", shape=shape, material=material, mass=mass, position=center)

    return EntityPhysicsResult(
        entity_id=entity.id,
        status=PhysicsCompileStatus.COMPILED,
        body=body,
        material_source=material_source,
        mass_provenance=mass_provenance,
        confidence=min(entity.confidence, material_confidence),
        notes=tuple(notes),
    )


def derive_plane_for_entity(entity: Entity, world: WorldIR) -> Optional[Plane]:
    """Derive a true infinite `Plane` for `entity`, or `None` when there is
    nothing real to derive one from.

    Convention (thinnest-AABB-axis heuristic, center-based distance):
      - Only entities of a `_STATIC_ENTITY_TYPES` type with real
        BOX/PLANE geometry bounds (same gating as `compile_entity_physics`,
        via `_entity_geometry_with_bounds`) produce a plane. Everything
        else returns None -- never fabricated.
      - The normal is a unit vector along whichever axis (X, Y, or Z) has
        the smallest extent in `bounds_max - bounds_min`. A real wall/
        floor/ceiling AABB from evidence/promote_planes.py is always thin
        along its true normal axis by construction of how planes are
        fit, so this is a deterministic, real-data-derived choice, not a
        guess. Ties (a genuine cube) are broken by X, then Y, then Z.
      - `distance` is computed as `normal.dot(center)` where `center` is
        the AABB's own center, so the derived plane passes exactly
        through the AABB's center along the thin axis. No sign
        convention beyond "positive along the chosen axis" is asserted.

    Example: floor AABB (0,0,0)-(4,0.1,4) -> extents (4, 0.1, 4) -> Y is
    thinnest -> normal (0,1,0), distance = 1*0.05 = 0.05.
    """
    if entity.type not in _STATIC_ENTITY_TYPES:
        return None
    geometry = _entity_geometry_with_bounds(world, entity)
    if geometry is None:
        return None

    bmin, bmax = geometry.bounds_min, geometry.bounds_max
    extents = (bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z)
    thin_axis = min(range(3), key=lambda i: extents[i])  # ties -> first in X/Y/Z order
    center = Vec3((bmin.x + bmax.x) / 2.0, (bmin.y + bmax.y) / 2.0, (bmin.z + bmax.z) / 2.0)
    normal = [Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)][thin_axis]
    distance = normal.dot(center)
    return Plane(normal=normal, distance=distance)


def compile_physics_world(world: WorldIR) -> PhysicsCompileDiagnostics:
    """Compile every entity in `world` into physics, in deterministic
    (sorted entity id) order. Never mutates `world`."""
    results = tuple(compile_entity_physics(world.entities[eid], world) for eid in sorted(world.entities))
    return PhysicsCompileDiagnostics(results=results)


def build_stepped_physics_world(
    diagnostics: PhysicsCompileDiagnostics,
    config: Optional[PhysicsWorldConfig] = None,
    *,
    world: Optional[WorldIR] = None,
    use_planes_for_statics: bool = False,
) -> PhysicsWorld:
    """Load every compiled body into a real, steppable `PhysicsWorld`
    (engine/physics/backend/simple_backend.py) -- the actual gap this
    bridge exists to close: a compiled reconstructed room's walls/floor
    (static) and debris (dynamic) now collide with each other through
    the engine's real broadphase/narrowphase/contact-solver stack, not
    just integrate independently. Bodies are added in the diagnostics'
    already-deterministic (sorted entity id) order.

    `use_planes_for_statics=False` (the default) reproduces prior
    behavior exactly: every compiled body, static or dynamic, is added
    via `world.add_body()`, and `world.planes` stays empty.

    `use_planes_for_statics=True` opts into the additive enhancement: for
    each static compiled entity (`result.body.mass == 0`) for which
    `derive_plane_for_entity(world.entities[result.entity_id], world)`
    returns a real `Plane`, that entity is added as a `StaticPlane` via
    `world.add_plane()` instead of a Box body. Every other entity
    (dynamic, or static with no derivable plane) is still added as a
    body exactly as before. Requires `world` (the same WorldIR the
    diagnostics were compiled from) when enabled.
    """
    physics_world = SimpleRigidBodyBackend().create_world(config or PhysicsWorldConfig())
    for result in diagnostics.compiled:
        plane = None
        if use_planes_for_statics and result.body.mass == 0:
            if world is None:
                raise ValueError("use_planes_for_statics=True requires the source WorldIR ('world')")
            plane = derive_plane_for_entity(world.entities[result.entity_id], world)
        if plane is not None:
            physics_world.add_plane(StaticPlane(id=result.entity_id, plane=plane, material=result.body.material))
        else:
            physics_world.add_body(result.body)
    return physics_world
