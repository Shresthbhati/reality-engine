"""WorldIR validation gate (spec sec 25 WORLD VALIDATION).

Composes the world's own structural self-check (`WorldIR.validate()` --
dangling entity/geometry/material/surface/component references, temporal
and branch consistency) with deeper geometric and provenance checks
that self-check does not perform:

  - invalid geometry: empty bounds, inverted bounds, non-finite values;
  - impossible transforms: non-finite matrices, near-singular ones;
  - duplicate entity identity: two entities with the same
    (type, name, geometry) fingerprint -- a real duplicate, not a
    naming convention;
  - invalid measurements: non-finite or negative values, non-positive
    precision, confidence outside [0, 1], impossible (negative)
    dimensions (zero allowed: absence of extent is not corruption);
  - provenance/confidence consistency: UNKNOWN-provenance entities must
    not claim high confidence (0.9+) -- claiming to not know while
    claiming certainty is a spec violation, not a style choice;
  - reconstruction traceability: RECONSTRUCTED entities must carry
    their build trace (session/backend/config) -- WARNING, so
    pre-stamp worlds stay loadable while new drops are flagged;
  - coordinate-frame consistency for transforms that reference entities.

Blueprint invariants 1-7 (docs/blueprint/11_WORLDIR_CONTRACT.md section 3) are
enforced by `_check_invariants`, each failure an ERROR with an `invariant_N_*`
code so a commit gate (`validate_world_ir(...).is_valid()`) blocks on it. They
map onto this WorldIR as follows (this schema has no Space/Opening/Surface
records of its own):
  evidence_state -> `custom_properties["evidence_state"]` when set, else the
                    entity's `Provenance` (GENERATED == PRIOR_GENERATED);
  Surface        -> WALL/FLOOR/CEILING entities (and `world.surfaces`);
  Opening        -> DOOR/WINDOW/OPENING entities; Space -> ROOM/CORRIDOR;
  observation_id -> `entity.observations`, `custom_properties["observation_ids"]`
                    or an association decision in the world's ledger;
  bounds         -> the union of the entity's geometry bounds (world frame).

Severity matters: ERROR blocks the world compiler's gate (and any
consumer that refuses corrupted state); WARNING is reported and
returned, never silently swallowed.

Deterministic: issues are collected in canonical order (entities sorted
by id, checks in fixed order), so the same world always yields the same
report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from provenance import Provenance
from world_ir import WorldIR
from world_ir.observations import AssociationState, LEDGER_KEY, ledger_of
from world_ir.schema_v1 import EntityType, RelationshipKind


class ValidationSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(frozen=True)
class WorldValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity is ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity is ValidationSeverity.WARNING]

    def is_valid(self) -> bool:
        return not self.errors

    def messages(self) -> List[str]:
        return [str(i) for i in self.issues]

    def render(self) -> str:
        """Readable multi-line report for a blocked commit: errors first, then warnings."""
        if not self.issues:
            return "world validation: OK"
        head = "world validation: " + ("OK with warnings" if self.is_valid() else f"BLOCKED, {len(self.errors)} error(s)")
        return "\n".join([head, *(f"  {i}" for i in (*self.errors, *self.warnings))])


def _iter_bounds(geometry):
    yield geometry.bounds_min
    yield geometry.bounds_max


def _check_geometry(world: WorldIR, issues: List[ValidationIssue]) -> None:
    for geometry_id in sorted(world.geometries):
        geometry = world.geometries[geometry_id]
        for vector, label in ((geometry.bounds_min, "bounds_min"), (geometry.bounds_max, "bounds_max")):
            if vector is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="geometry_missing_bounds",
                    message=f"geometry {geometry_id} has no {label}",
                ))
                continue
            coords = (vector.x, vector.y, vector.z)
            if any(not math.isfinite(c) for c in coords):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="geometry_non_finite_bounds",
                    message=f"geometry {geometry_id} has non-finite {label} {coords}",
                ))
        if geometry.bounds_min is None or geometry.bounds_max is None:
            continue
        lo = (geometry.bounds_min.x, geometry.bounds_min.y, geometry.bounds_min.z)
        hi = (geometry.bounds_max.x, geometry.bounds_max.y, geometry.bounds_max.z)
        if any(h < l for h, l in zip(hi, lo)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="geometry_inverted_bounds",
                message=(
                    f"geometry {geometry_id} has inverted bounds: "
                    f"max {hi} < min {lo} on at least one axis"
                ),
            ))


def _check_transforms(world: WorldIR, issues: List[ValidationIssue]) -> None:
    for entity_id in sorted(world.transforms):
        transform = world.transforms[entity_id]
        if entity_id not in world.entities:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="transform_unknown_entity",
                message=f"transform references unknown entity {entity_id}",
            ))
            continue
        matrix = transform.matrix
        flat = [value for row in matrix for value in row] if isinstance(matrix[0], (list, tuple)) else list(matrix)
        if any(not math.isfinite(float(v)) for v in flat):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="transform_non_finite",
                message=f"transform of entity {entity_id} contains non-finite values",
            ))
            continue
        # Near-singular rigid transform: last-row guard + determinant on
        # the upper-left 3x3. A degenerate matrix would collapse world
        # geometry onto a plane/point when applied.
        if len(matrix) == 4:
            bottom = matrix[3]
            if any(abs(float(v)) > 1e-9 for v in bottom[:3]) or abs(float(bottom[3]) - 1.0) > 1e-9:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="transform_not_affine",
                    message=f"transform of entity {entity_id} has a non-affine bottom row {bottom}",
                ))
            a = matrix[0][:3]
            b = matrix[1][:3]
            c = matrix[2][:3]
            det = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            if abs(det) < 1e-9:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="transform_singular",
                    message=f"transform of entity {entity_id} is near-singular (det={det:.3e})",
                ))


def _check_duplicate_identity(world: WorldIR, issues: List[ValidationIssue]) -> None:
    seen = {}
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        fingerprint = (
            entity.type.value if hasattr(entity.type, "value") else str(entity.type),
            entity.name,
            tuple(sorted(entity.geometry_ids)),
        )
        if fingerprint in seen:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="entity_duplicate_identity",
                message=(
                    f"entities {seen[fingerprint]} and {entity_id} share type/name/geometry "
                    "-- possible duplicate"
                ),
            ))
        else:
            seen[fingerprint] = entity_id


def _check_measurements(world: WorldIR, issues: List[ValidationIssue]) -> None:
    # Measurements ride on entities as (key, Measurement) pairs in
    # custom_properties; values may also be plain scalars promoted by
    # earlier layers. Both shapes are validated where applicable.
    #
    # Coordinate-like properties are positions IN the reconstruction's
    # world frame, not magnitudes: the frame's origin is arbitrary
    # (COLMAP puts it wherever the SfM solution landed -- a real
    # capture legitimately measured its ground storey at -1.42 m), so
    # a negative value is honest data, not corruption. Declaring these
    # keys here is deliberate: a new position-like property must be
    # consciously classified, and everything else ending in _m/_m2/_m3
    # stays an extent where negative means a broken measurement.
    COORDINATE_LIKE_PROPERTY_KEYS = {
        "floor_height_m",     # storey elevation (measured, frame-relative)
        "elevation_m",       # storey level elevation
        "sill_height_m",     # opening sill elevation
        "head_height_m",     # opening head elevation
    }
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        for key, value in entity.custom_properties.items():
            if not isinstance(value, float) and not isinstance(value, int):
                continue
            if not math.isfinite(float(value)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="measurement_non_finite",
                    message=f"entity {entity_id} property {key} is non-finite",
                ))
                continue
            # Dimension-like keys must be non-negative; zero is legal
            # (absence of extent is not corruption). Coordinate-like
            # keys are exempt: their sign carries the arbitrary frame.
            if (key.endswith(("_m", "_m2", "_m3"))
                    and key not in COORDINATE_LIKE_PROPERTY_KEYS
                    and float(value) < 0.0):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="measurement_negative_dimension",
                    message=f"entity {entity_id} property {key} is negative ({value})",
                ))


def _check_provenance_confidence(world: WorldIR, issues: List[ValidationIssue]) -> None:
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        provenance = entity.provenance
        confidence = entity.confidence
        # Non-numeric confidence must be an explicit ERROR, never a
        # TypeError escaping the validator (which would turn a malformed
        # world into a crashed gate instead of a refused one).
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="confidence_out_of_range",
                message=f"entity {entity_id} confidence {confidence!r} outside [0, 1]",
            ))
            continue
        if provenance is Provenance.UNKNOWN and confidence >= 0.9:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="unknown_provenance_high_confidence",
                message=(
                    f"entity {entity_id} claims UNKNOWN provenance with confidence "
                    f"{confidence:.2f} -- cannot know nothing and be certain"
                ),
            ))


def _check_reconstruction_trace(world: WorldIR, issues: List[ValidationIssue]) -> None:
    """Every RECONSTRUCTED entity must carry its build trace (session,
    backend, configuration) in custom_properties.reconstruction -- stamped
    by the reconstruction job before persistence. Without it the entity
    cannot be traced back to its capture/session/evidence and stage, so
    a reload cannot prove where it came from. WARNING (not ERROR): worlds
    persisted before stamping existed remain loadable, but any new
    reconstruction that drops its trace is flagged, and the job grader
    treats warnings as PARTIAL rather than SUCCEEDED."""
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if entity.provenance is not Provenance.RECONSTRUCTED:
            continue
        trace = entity.custom_properties.get("reconstruction") if isinstance(
            entity.custom_properties, dict) else None
        if not isinstance(trace, dict) or not trace.get("session_id"):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="reconstruction_untraced",
                message=(
                    f"entity {entity_id} claims RECONSTRUCTED provenance but "
                    "carries no reconstruction trace (session/backend/config) "
                    "-- untraceable after reload"
                ),
            ))


def _check_interior_boundary_references(world: WorldIR, issues: List[ValidationIssue]) -> None:
    """ROOM/CORRIDOR entities record their boundary parts (walls, floors,
    ceilings) two ways: as CONTAINS relationships (checked generically by
    world.validate()'s dangling-relationship pass) and, for entities
    promoted by perception.architecture.topology, as a plain id list in
    custom_properties["boundary_element_ids"]. The two can desync if a
    future caller edits one without the other; check the informational
    copy directly rather than trusting it matches the graph edges."""
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if entity.type not in (EntityType.ROOM, EntityType.CORRIDOR):
            continue
        boundary_ids = entity.custom_properties.get("boundary_element_ids")
        if not isinstance(boundary_ids, (list, tuple)):
            continue
        for bid in boundary_ids:
            if bid not in world.entities:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="interior_boundary_reference_missing",
                    message=(
                        f"{entity.type.value} {entity_id} references boundary "
                        f"part {bid!r} which does not exist in the world"
                    ),
                ))


def _check_single_storey_membership(world: WorldIR, issues: List[ValidationIssue]) -> None:
    """A ROOM/CORRIDOR promoted onto two different STOREY entities via
    CONTAINS would be physically impossible (it cannot occupy two
    building levels at once) and is evidence of a topology-promotion bug
    (e.g. running promote_building_topology twice, or a level-assignment
    conflict), not a legitimate multi-level space."""
    owners: dict = {}
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        if entity.type is not EntityType.STOREY:
            continue
        for rel in entity.relationships:
            if rel.kind is not RelationshipKind.CONTAINS:
                continue
            target = world.entities.get(rel.target_id)
            if target is None or target.type not in (EntityType.ROOM, EntityType.CORRIDOR):
                continue
            owners.setdefault(rel.target_id, set()).add(entity_id)
    for member_id, storey_ids in owners.items():
        if len(storey_ids) > 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="entity_multiple_storey_membership",
                message=(
                    f"entity {member_id} is CONTAINS-owned by "
                    f"{len(storey_ids)} storeys ({sorted(storey_ids)}) -- "
                    "impossible containment, a space cannot belong to two levels"
                ),
            ))


_STRONG_STATES = {"OBSERVED", "PARTIALLY_OBSERVED"}
_WEAK_STATES = {"INFERRED", "GENERATED", "PRIOR_GENERATED"}
VALID_EVIDENCE_STATES = (
    {p.value for p in Provenance}
    | {"OBSERVED", "PARTIALLY_OBSERVED", "OCCLUDED", "UNOBSERVED", "INFERRED", "PRIOR_GENERATED", "CONFLICTED", "UNKNOWN"}
)
_SURFACE_TYPES = {EntityType.WALL, EntityType.FLOOR, EntityType.CEILING}
_OPENING_TYPES = {EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING}
_SPACE_TYPES = {EntityType.ROOM, EntityType.CORRIDOR}
_METRIC_UNITS = {"m", "m2", "m3", "mm", "cm", "km", "meter", "meters", "metre", "metres"}
_RELATIVE_SCALE_STATES = {"", "relative", "unknown"}
_DUPLICATE_IOU = 0.5
#: Invariant 7, "each Surface references a Space": the current compiler emits floor/ceiling planes that no room lists
#: (observed: a compiled world with a room but an unlisted floor plane). Blocking on that would make real captures
#: uncommittable, so it is a WARNING until the hierarchy (blueprint 5.1) links them. Flip to True to make it block.
SURFACE_WITHOUT_SPACE_BLOCKS = False
#: Invariant 6 on architectural surfaces (wall/floor/ceiling): the real South Building capture compiles three
#: overlapping floor patches (AABB IoU 50-69%), and thin plane boxes make IoU a weak duplicate signal. Surface
#: duplicates are a WARNING until plane merging lands; every other class blocks. Flip to True to block surfaces too.
SURFACE_DUPLICATE_BLOCKS = False
#: association states that explicitly say two entities are NOT one thing (exempt from the duplicate guard)
_DISTINCT_STATES = {AssociationState.DIFFERENT_ENTITY, AssociationState.PHYSICAL_INTERSECTION}


def _err(issues: List[ValidationIssue], code: str, message: str) -> None:
    issues.append(ValidationIssue(ValidationSeverity.ERROR, code, message))


def _props(entity) -> dict:
    return entity.custom_properties if isinstance(entity.custom_properties, dict) else {}


def _entity_states(entity) -> Set[str]:
    """Every evidence-state claim the entity makes, as upper-case strings."""
    states = set()
    if isinstance(entity.provenance, Provenance):
        states.add(entity.provenance.value)
    if isinstance(_props(entity).get("evidence_state"), str):
        states.add(_props(entity)["evidence_state"].upper())
    if entity.statement_state is not None:
        states.add(getattr(entity.statement_state, "value", str(entity.statement_state)).upper())
    return states


def _entity_aabb(world: WorldIR, entity) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    lo, hi = None, None
    for gid in entity.geometry_ids:
        g = world.geometries.get(gid)
        if g is None or g.bounds_min is None or g.bounds_max is None:
            continue
        gl, gh = (g.bounds_min.x, g.bounds_min.y, g.bounds_min.z), (g.bounds_max.x, g.bounds_max.y, g.bounds_max.z)
        if not all(math.isfinite(v) for v in gl + gh):
            continue
        lo = gl if lo is None else tuple(min(a, b) for a, b in zip(lo, gl))
        hi = gh if hi is None else tuple(max(a, b) for a, b in zip(hi, gh))
    return None if lo is None else (lo, hi)


def _iou(a, b) -> float:
    inter = 1.0
    for i in range(3):
        inter *= max(0.0, min(a[1][i], b[1][i]) - max(a[0][i], b[0][i]))
    vol = lambda box: math.prod(max(0.0, box[1][i] - box[0][i]) for i in range(3))
    union = vol(a) + vol(b) - inter
    return inter / union if union > 0 else 0.0


def _metric_unit_paths(value, path: str):
    if isinstance(value, dict):
        unit = value.get("unit")
        if isinstance(unit, str) and unit.strip().lower() in _METRIC_UNITS:
            yield path, unit
        for k, v in value.items():
            yield from _metric_unit_paths(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for n, v in enumerate(value):
            yield from _metric_unit_paths(v, f"{path}[{n}]")


def _linked(world: WorldIR, a_id: str, a, predicate) -> bool:
    """True if `a` has a relationship (either direction) with an entity satisfying `predicate`."""
    for rel in a.relationships:
        t = world.entities.get(rel.target_id)
        if t is not None and predicate(t):
            return True
    return any(predicate(o) and any(r.target_id == a_id for r in o.relationships) for o in world.entities.values())


def _check_cross_entity_consistency(world: WorldIR, issues: List[ValidationIssue]) -> None:
    """Automated cross-entity consistency checks (Task 7.5):
    - Circular support relationships (e.g. A SUPPORTS B and B SUPPORTS A)
    - Vertical resting consistency (RESTS_ON / SUPPORTS geometric alignment)
    - Containment disjointness (CONTAINS targets located far outside parent)
    - Opening height consistency (opening top/bottom bounds relative to host wall)
    - Attachment disconnection (ATTACHED_TO entities separated by > 2.0m)
    """
    ids = sorted(world.entities)

    # 1. Check for cyclic vertical support
    support_edges: Dict[str, Set[str]] = {}
    for eid in ids:
        e = world.entities[eid]
        for rel in e.relationships:
            if rel.kind is RelationshipKind.SUPPORTS and rel.target_id in world.entities:
                support_edges.setdefault(eid, set()).add(rel.target_id)
            elif rel.kind is RelationshipKind.RESTS_ON and rel.target_id in world.entities:
                support_edges.setdefault(rel.target_id, set()).add(eid)

    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    cycles_found: Set[Tuple[str, str]] = set()

    def _dfs_cycle(node: str, path: List[str]):
        visited.add(node)
        rec_stack.add(node)
        for neighbour in sorted(support_edges.get(node, ())):
            if neighbour not in visited:
                _dfs_cycle(neighbour, path + [neighbour])
            elif neighbour in rec_stack:
                cycle_pair = tuple(sorted([node, neighbour]))
                if cycle_pair not in cycles_found:
                    cycles_found.add(cycle_pair)
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        "cross_entity_cyclic_support",
                        f"cyclic vertical support relationship detected between {node} and {neighbour}",
                    ))
        rec_stack.remove(node)

    for eid in sorted(support_edges):
        if eid not in visited:
            _dfs_cycle(eid, [eid])

    # 2. Check vertical support alignment & horizontal overlap
    for supporter_id, targets in sorted(support_edges.items()):
        supporter = world.entities[supporter_id]
        box_s = _entity_aabb(world, supporter)
        for target_id in sorted(targets):
            target = world.entities[target_id]
            box_t = _entity_aabb(world, target)
            if box_s is not None and box_t is not None:
                lo_s, hi_s = box_s
                lo_t, hi_t = box_t
                # Supported entity bottom significantly below supporter base (e.g. sunk > 0.25m)
                if lo_t[2] < lo_s[2] - 0.25:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        "cross_entity_vertical_support_inconsistent",
                        f"entity {target_id} rests on {supporter_id} but sits {lo_s[2] - lo_t[2]:.2f}m below its base",
                    ))
                # Supported entity bottom floating far above supporter top (> 0.5m)
                elif lo_t[2] > hi_s[2] + 0.50:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        "cross_entity_vertical_support_inconsistent",
                        f"entity {target_id} rests on {supporter_id} but floats {lo_t[2] - hi_s[2]:.2f}m above its top",
                    ))
                # Horizontal separation (dx or dy > 0.5m)
                dx = max(0.0, max(lo_s[0], lo_t[0]) - min(hi_s[0], hi_t[0]))
                dy = max(0.0, max(lo_s[1], lo_t[1]) - min(hi_s[1], hi_t[1]))
                if dx > 0.5 or dy > 0.5:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        "cross_entity_vertical_support_inconsistent",
                        f"entity {target_id} rests on {supporter_id} but they are horizontally separated by {max(dx, dy):.2f}m",
                    ))

    # 3. Check Opening vs Host Wall bounds
    for eid in ids:
        e = world.entities[eid]
        if e.type in _OPENING_TYPES:
            host_id = _props(e).get("host_wall_id")
            if not host_id:
                for rel in e.relationships:
                    if rel.kind is RelationshipKind.ATTACHED_TO and rel.target_id in world.entities:
                        if world.entities[rel.target_id].type is EntityType.WALL:
                            host_id = rel.target_id
                            break
            if host_id and host_id in world.entities:
                host = world.entities[host_id]
                box_op = _entity_aabb(world, e)
                box_w = _entity_aabb(world, host)
                if box_op is not None and box_w is not None:
                    lo_op, hi_op = box_op
                    lo_w, hi_w = box_w
                    if hi_op[2] > hi_w[2] + 0.20:
                        issues.append(ValidationIssue(
                            ValidationSeverity.WARNING,
                            "cross_entity_opening_exceeds_wall_height",
                            f"opening {eid} top ({hi_op[2]:.2f}m) exceeds host wall {host_id} top ({hi_w[2]:.2f}m)",
                        ))
                    if lo_op[2] < lo_w[2] - 0.20:
                        issues.append(ValidationIssue(
                            ValidationSeverity.WARNING,
                            "cross_entity_opening_exceeds_wall_height",
                            f"opening {eid} bottom ({lo_op[2]:.2f}m) extends below host wall {host_id} base ({lo_w[2]:.2f}m)",
                        ))

    # 4. Check Containment consistency
    for eid in ids:
        e = world.entities[eid]
        box_p = _entity_aabb(world, e)
        for rel in e.relationships:
            if rel.kind is RelationshipKind.CONTAINS and rel.target_id in world.entities:
                child = world.entities[rel.target_id]
                box_c = _entity_aabb(world, child)
                if box_p is not None and box_c is not None:
                    lo_p, hi_p = box_p
                    lo_c, hi_c = box_c
                    margin = 0.5
                    is_disjoint = (
                        lo_c[0] > hi_p[0] + margin or hi_c[0] < lo_p[0] - margin or
                        lo_c[1] > hi_p[1] + margin or hi_c[1] < lo_p[1] - margin or
                        lo_c[2] > hi_p[2] + margin or hi_c[2] < lo_p[2] - margin
                    )
                    if is_disjoint:
                        issues.append(ValidationIssue(
                            ValidationSeverity.WARNING,
                            "cross_entity_containment_disjoint",
                            f"entity {rel.target_id} is CONTAINS-owned by {eid} but its geometry is disjoint from {eid}",
                        ))

    # 5. Check Attachment proximity
    for eid in ids:
        e = world.entities[eid]
        box_a = _entity_aabb(world, e)
        for rel in e.relationships:
            if rel.kind is RelationshipKind.ATTACHED_TO and rel.target_id in world.entities:
                target = world.entities[rel.target_id]
                box_b = _entity_aabb(world, target)
                if box_a is not None and box_b is not None:
                    dx = max(0.0, max(box_a[0][0], box_b[0][0]) - min(box_a[1][0], box_b[1][0]))
                    dy = max(0.0, max(box_a[0][1], box_b[0][1]) - min(box_a[1][1], box_b[1][1]))
                    dz = max(0.0, max(box_a[0][2], box_b[0][2]) - min(box_a[1][2], box_b[1][2]))
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > 2.0:
                        issues.append(ValidationIssue(
                            ValidationSeverity.WARNING,
                            "cross_entity_attachment_disconnected",
                            f"entity {eid} claims ATTACHED_TO {rel.target_id} but they are {dist:.2f}m apart",
                        ))


def _check_invariants(world: WorldIR, issues: List[ValidationIssue]) -> None:
    ids = sorted(world.entities)
    try:
        ledger_obs, ledger_dec = ledger_of(world)
    except (KeyError, TypeError, ValueError) as exc:
        _err(issues, "invariant_5_ledger_unreadable", f"{LEDGER_KEY} in world.metadata cannot be read: {exc!r}")
        ledger_obs, ledger_dec = (), ()
    obs_ids = {o.observation_id for o in ledger_obs} | set(world.observations)
    decided_entities = {d.entity_id for d in ledger_dec if d.entity_id}

    # 1. every Entity/Surface has an evidence_state from the vocabulary
    for eid in ids:
        e = world.entities[eid]
        if not isinstance(e.provenance, Provenance):
            _err(issues, "invariant_1_no_evidence_state", f"entity {eid} has no valid evidence state (provenance={e.provenance!r})")
        if "evidence_state" in _props(e) and str(_props(e)["evidence_state"]).upper() not in VALID_EVIDENCE_STATES:
            _err(issues, "invariant_1_no_evidence_state", f"entity {eid} evidence_state {_props(e)['evidence_state']!r} is not a known state")
    for sid in sorted(world.surfaces):
        state = (world.surfaces[sid].custom_attributes or {}).get("evidence_state")
        if not isinstance(state, str) or state.upper() not in VALID_EVIDENCE_STATES:
            _err(issues, "invariant_1_no_evidence_state", f"surface {sid} has no valid evidence_state (got {state!r})")

    # 2. OBSERVED / PARTIALLY_OBSERVED entities cite at least one observation
    for eid in ids:
        e = world.entities[eid]
        strong = _entity_states(e) & _STRONG_STATES
        if strong and not (e.observations or _props(e).get("observation_ids") or eid in decided_entities):
            _err(issues, "invariant_2_observed_without_observation", f"entity {eid} is {sorted(strong)[0]} but cites no observation_id")

    # 3. INFERRED / PRIOR_GENERATED is never also OBSERVED
    for eid in ids:
        states = _entity_states(world.entities[eid])
        if states & _STRONG_STATES and states & _WEAK_STATES:
            _err(issues, "invariant_3_inferred_labelled_observed",
                 f"entity {eid} is labelled both {sorted(states & _STRONG_STATES)} and {sorted(states & _WEAK_STATES)}")

    # 4. metric units only when the world's scale is not relative
    scale = world.metadata.get("scale") if isinstance(world.metadata, dict) else None
    scale_state = str(scale.get("state") or "").lower() if isinstance(scale, dict) else ""
    if scale_state in _RELATIVE_SCALE_STATES:
        for eid in ids:
            for path, unit in _metric_unit_paths(_props(world.entities[eid]), "custom_properties"):
                _err(issues, "invariant_4_metric_unit_in_relative_world",
                     f"entity {eid} {path} uses metric unit {unit!r} but the world scale is "
                     f"{scale_state or 'undeclared'} (relative worlds use model units)")

    # 5. ids unique (key == id, no duplicates) and every referenced id exists
    #    (entity/geometry/material/surface/component/relationship references: world.validate() -> "structural")
    for label, table in (("entity", world.entities), ("geometry", world.geometries), ("material", world.materials),
                         ("surface", world.surfaces), ("component", world.components), ("observation", world.observations)):
        for key in sorted(table):
            if getattr(table[key], "id", key) != key:
                _err(issues, "invariant_5_id_mismatch", f"{label} stored under key {key!r} has id {table[key].id!r}")
    seen_obs: Set[str] = set()
    for o in ledger_obs:
        if o.observation_id in seen_obs:
            _err(issues, "invariant_5_duplicate_id", f"ledger observation id {o.observation_id!r} appears twice")
        seen_obs.add(o.observation_id)
    for d in ledger_dec:
        if d.observation_id not in seen_obs:
            _err(issues, "invariant_5_dangling_reference", f"association decision cites unknown observation {d.observation_id!r}")
    for eid in ids:
        for oid in _props(world.entities[eid]).get("observation_ids") or ():
            if oid not in obs_ids:
                _err(issues, "invariant_5_dangling_reference", f"entity {eid} cites unknown observation {oid!r}")

    # 6. duplicate guard: same class, >50% bounds IoU, no decision saying they differ
    distinct: Set[frozenset] = set()
    for d in ledger_dec:
        if d.state in _DISTINCT_STATES and d.entity_id:
            distinct.update(frozenset((d.entity_id, other)) for other, _ in d.candidates)
    for eid in ids:
        distinct.update(frozenset((eid, other)) for other in _props(world.entities[eid]).get("distinct_from") or ())
    groups: Dict[tuple, list] = {}
    for eid in ids:
        e = world.entities[eid]
        box = _entity_aabb(world, e)
        if box is not None:
            groups.setdefault((e.type, tuple(sorted(e.semantic_labels))), []).append((eid, box))
    for (etype, _labels), members in groups.items():
        severity = (ValidationSeverity.ERROR if etype not in _SURFACE_TYPES or SURFACE_DUPLICATE_BLOCKS
                    else ValidationSeverity.WARNING)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (a, ba), (b, bb) = members[i], members[j]
                iou = _iou(ba, bb)
                if iou > _DUPLICATE_IOU and frozenset((a, b)) not in distinct:
                    issues.append(ValidationIssue(
                        severity, "invariant_6_duplicate_entity",
                        f"entities {a} and {b} are the same class and overlap {iou:.0%} (IoU) with no "
                        "DIFFERENT_ENTITY / PHYSICAL_INTERSECTION decision"))

    # 7. each Opening references a Surface; each Surface references a Space
    #    Enforced once the world models that level at all (task 5.1 adds the hierarchy): a compiled world with planes
    #    but no rooms yet (partial capture) would otherwise be un-committable.
    is_surface = lambda t: t.type in _SURFACE_TYPES
    is_space = lambda t: t.type in _SPACE_TYPES
    has_surfaces = any(is_surface(e) for e in world.entities.values())
    has_spaces = any(is_space(e) for e in world.entities.values())
    for eid in ids:
        e = world.entities[eid]
        cp = _props(e)
        if e.type in _OPENING_TYPES and has_surfaces:
            host = cp.get("host_wall_id") or cp.get("surface_id")
            if not ((host in world.entities and is_surface(world.entities[host])) or _linked(world, eid, e, is_surface)):
                _err(issues, "invariant_7_opening_without_surface", f"opening {eid} ({e.type.value}) references no surface (wall/floor/ceiling)")
        elif e.type in _SURFACE_TYPES and has_spaces:
            in_boundary = any(eid in (_props(r).get("boundary_element_ids") or ()) for r in world.entities.values() if is_space(r))
            if not ((cp.get("space_id") or cp.get("room_id")) in world.entities or in_boundary or _linked(world, eid, e, is_space)):
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR if SURFACE_WITHOUT_SPACE_BLOCKS else ValidationSeverity.WARNING,
                    "invariant_7_surface_without_space", f"surface {eid} ({e.type.value}) references no space (room/corridor)"))


def validate_world_ir(world: WorldIR) -> WorldValidationReport:
    """Full validation: structural self-check + geometric/provenance
    depth. Deterministic issue ordering (sorted entity/geometry ids,
    fixed check order)."""
    issues: List[ValidationIssue] = []

    # 1. The world's own structural self-check (dangling references,
    #    temporal/branch consistency) -- compose, don't duplicate.
    for message in world.validate():
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="structural",
            message=message,
        ))

    _check_geometry(world, issues)
    _check_transforms(world, issues)
    _check_duplicate_identity(world, issues)
    _check_measurements(world, issues)
    _check_provenance_confidence(world, issues)
    _check_reconstruction_trace(world, issues)
    _check_interior_boundary_references(world, issues)
    _check_single_storey_membership(world, issues)
    _check_cross_entity_consistency(world, issues)
    _check_confidence_vectors(world, issues)
    _check_invariants(world, issues)

    return WorldValidationReport(issues=issues)


def _check_confidence_vectors(world: WorldIR, issues: List[ValidationIssue]) -> None:
    """Validate ConfidenceVector on each entity (Task 7.1).
    Every dimension must be either None or a float in [0.0, 1.0].
    """
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        cv = getattr(entity, "confidence_vector", None)
        if cv is None:
            continue
        for dim in (
            "classification", "identity", "position", "orientation",
            "dimensions", "geometry", "completeness", "material",
            "colour", "topology", "cross_view",
        ):
            val = getattr(cv, dim, None)
            if val is not None:
                if not isinstance(val, (int, float)) or math.isnan(val) or not (0.0 <= float(val) <= 1.0):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="invalid_confidence_vector_value",
                        message=(
                            f"entity {entity_id} confidence_vector.{dim} ({val}) "
                            "must be None or float in [0.0, 1.0]"
                        ),
                    ))
