"""Geometric relationship inference for WorldIR entities.

Computes semantic and spatial relationships directly from 3D geometry:
- supported_by / supports (gravity support, horizontal planar contact)
- mounted_on / attached_to (proximity and vertical overlap against wall surfaces)
- part_of / contains (bounding containment and architectural component membership)
- inside / contains (spatial enclosure within rooms, spaces, or containers)

Implements Blueprint Task 8.4 and Blueprint 11 (WorldIR Contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR

Point3 = Tuple[float, float, float]
Bounds3 = Tuple[Point3, Point3]


@dataclass(frozen=True)
class GeometricRelationshipOptions:
    """Tolerances and parameters for geometric relationship inference."""
    vertical_support_tolerance_m: float = 0.15
    vertical_support_penetration_m: float = 0.08
    wall_mount_tolerance_m: float = 0.20
    min_horizontal_overlap_ratio: float = 0.10
    containment_threshold: float = 0.85
    infer_reciprocals: bool = True
    default_confidence: float = 0.85


@dataclass(frozen=True)
class InferredRelationship:
    """A single geometrically inferred relationship edge between two entities."""
    source_id: str
    target_id: str
    kind: RelationshipKind
    confidence: float
    basis: str = "geometric"
    reason: str = ""

    def to_relationship(self) -> Relationship:
        return Relationship(
            kind=self.kind,
            target_id=self.target_id,
            confidence=self.confidence,
            provenance=Provenance.INFERRED,
            metadata={"basis": self.basis, "reason": self.reason},
        )


def get_entity_bounds(world: WorldIR, entity: Entity) -> Optional[Bounds3]:
    """Extract axis-aligned bounding box (min_pt, max_pt) for an entity.

    Sources in order:
    1. Geometries referenced in entity.geometry_ids
    2. custom_properties['bounding_box'] or custom_properties['bounds']
    3. custom_properties['dimensions'] combined with transform position
    4. Derived from parametric components if available
    """
    # 1. Geometries
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.bounds_min is not None and geom.bounds_max is not None:
            return (
                (float(geom.bounds_min.x), float(geom.bounds_min.y), float(geom.bounds_min.z)),
                (float(geom.bounds_max.x), float(geom.bounds_max.y), float(geom.bounds_max.z)),
            )

    # 2. Custom properties explicit bounds
    cp = entity.custom_properties or {}
    if "bounding_box" in cp and isinstance(cp["bounding_box"], (list, tuple)) and len(cp["bounding_box"]) == 2:
        b0, b1 = cp["bounding_box"]
        return (
            (float(b0[0]), float(b0[1]), float(b0[2])),
            (float(b1[0]), float(b1[1]), float(b1[2])),
        )
    if "bounds" in cp and isinstance(cp["bounds"], (list, tuple)) and len(cp["bounds"]) == 2:
        b0, b1 = cp["bounds"]
        return (
            (float(b0[0]), float(b0[1]), float(b0[2])),
            (float(b1[0]), float(b1[1]), float(b1[2])),
        )

    # 3. Position + Dimensions
    pos: Optional[Point3] = None
    if entity.transform and "position" in entity.transform:
        p = entity.transform["position"]
        pos = (float(p["x"]), float(p["y"]), float(p["z"]))

    dims: Optional[Tuple[float, float, float]] = None
    if "dimensions" in cp and isinstance(cp["dimensions"], (list, tuple, dict)):
        d = cp["dimensions"]
        if isinstance(d, (list, tuple)) and len(d) == 3:
            dims = (float(d[0]), float(d[1]), float(d[2]))
        elif isinstance(d, dict) and "width" in d and "depth" in d and "height" in d:
            w = d["width"].get("value") if isinstance(d["width"], dict) else d["width"]
            dp = d["depth"].get("value") if isinstance(d["depth"], dict) else d["depth"]
            h = d["height"].get("value") if isinstance(d["height"], dict) else d["height"]
            if w is not None and dp is not None and h is not None:
                dims = (float(w), float(dp), float(h))

    if pos is not None and dims is not None:
        return (
            (pos[0] - dims[0] * 0.5, pos[1] - dims[1] * 0.5, pos[2] - dims[2] * 0.5),
            (pos[0] + dims[0] * 0.5, pos[1] + dims[1] * 0.5, pos[2] + dims[2] * 0.5),
        )

    # 4. Parametric component union
    param = cp.get("parametric")
    if isinstance(param, dict) and "components" in param and isinstance(param["components"], list):
        comp_boxes = []
        for c in param["components"]:
            if isinstance(c, dict) and "box_min" in c and "box_max" in c:
                comp_boxes.append((c["box_min"], c["box_max"]))
        if comp_boxes:
            min_x = min(b[0][0] for b in comp_boxes)
            min_y = min(b[0][1] for b in comp_boxes)
            min_z = min(b[0][2] for b in comp_boxes)
            max_x = max(b[1][0] for b in comp_boxes)
            max_y = max(b[1][1] for b in comp_boxes)
            max_z = max(b[1][2] for b in comp_boxes)
            return ((float(min_x), float(min_y), float(min_z)), (float(max_x), float(max_y), float(max_z)))

    return None


def get_entity_centroid(world: WorldIR, entity: Entity) -> Optional[Point3]:
    """Get the 3D centroid of an entity."""
    bounds = get_entity_bounds(world, entity)
    if bounds is not None:
        b0, b1 = bounds
        return (
            (b0[0] + b1[0]) * 0.5,
            (b0[1] + b1[1]) * 0.5,
            (b0[2] + b1[2]) * 0.5,
        )
    if entity.transform and "position" in entity.transform:
        p = entity.transform["position"]
        return (float(p["x"]), float(p["y"]), float(p["z"]))
    return None


def _horizontal_overlap_ratio(a: Bounds3, b: Bounds3) -> float:
    """Calculate the ratio of XY horizontal intersection area over min(area(A), area(B))."""
    (a_min, a_max), (b_min, b_max) = a, b
    ix_min = max(a_min[0], b_min[0])
    ix_max = min(a_max[0], b_max[0])
    iy_min = max(a_min[1], b_min[1])
    iy_max = min(a_max[1], b_max[1])

    if ix_max <= ix_min or iy_max <= iy_min:
        return 0.0

    inter_area = (ix_max - ix_min) * (iy_max - iy_min)
    area_a = max(0.001, (a_max[0] - a_min[0]) * (a_max[1] - a_min[1]))
    area_b = max(0.001, (b_max[0] - b_min[0]) * (b_max[1] - b_min[1]))
    return inter_area / min(area_a, area_b)


def _point_in_xy_box(pt: Point3, b: Bounds3) -> bool:
    """True if point pt's (x, y) is within bounds b."""
    (b_min, b_max) = b
    return b_min[0] <= pt[0] <= b_max[0] and b_min[1] <= pt[1] <= b_max[1]


def _containment_ratio(inner: Bounds3, outer: Bounds3) -> float:
    """Volume intersection over inner volume."""
    (i_min, i_max), (o_min, o_max) = inner, outer
    ix_min = max(i_min[0], o_min[0])
    ix_max = min(i_max[0], o_max[0])
    iy_min = max(i_min[1], o_min[1])
    iy_max = min(i_max[1], o_max[1])
    iz_min = max(i_min[2], o_min[2])
    iz_max = min(i_max[2], o_max[2])

    if ix_max <= ix_min or iy_max <= iy_min or iz_max <= iz_min:
        return 0.0

    inter_vol = (ix_max - ix_min) * (iy_max - iy_min) * (iz_max - iz_min)
    inner_vol = max(1e-6, (i_max[0] - i_min[0]) * (i_max[1] - i_min[1]) * (i_max[2] - i_min[2]))
    return inter_vol / inner_vol


def _is_wall(entity: Entity) -> bool:
    """Check if entity represents a wall."""
    if entity.type == EntityType.WALL:
        return True
    labels = [l.lower() for l in entity.semantic_labels]
    return any("wall" in l for l in labels) or "architecture/wall" in labels


def _is_floor(entity: Entity) -> bool:
    """Check if entity represents a floor."""
    if entity.type == EntityType.FLOOR:
        return True
    labels = [l.lower() for l in entity.semantic_labels]
    return any("floor" in l for l in labels) or "architecture/floor" in labels


def _is_space(entity: Entity) -> bool:
    """Check if entity represents a room or space container."""
    if entity.type in (EntityType.ROOM, EntityType.CORRIDOR):
        return True
    labels = [l.lower() for l in entity.semantic_labels]
    return any("room" in l or "space" in l for l in labels)


def _is_opening(entity: Entity) -> bool:
    """Check if entity is an opening (door, window, passage)."""
    if entity.type in (EntityType.DOOR, EntityType.WINDOW, EntityType.OPENING):
        return True
    labels = [l.lower() for l in entity.semantic_labels]
    return any("door" in l or "window" in l or "opening" in l for l in labels)


def _distance_to_box(pt: Point3, b: Bounds3) -> float:
    """Euclidean distance from a 3D point to an AABB."""
    (b_min, b_max) = b
    dx = max(0.0, b_min[0] - pt[0], pt[0] - b_max[0])
    dy = max(0.0, b_min[1] - pt[1], pt[1] - b_max[1])
    dz = max(0.0, b_min[2] - pt[2], pt[2] - b_max[2])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _horizontal_distance_boxes(a: Bounds3, b: Bounds3) -> float:
    """Shortest XY distance between two bounding boxes."""
    (a_min, a_max), (b_min, b_max) = a, b
    dx = max(0.0, max(a_min[0] - b_max[0], b_min[0] - a_max[0]))
    dy = max(0.0, max(a_min[1] - b_max[1], b_min[1] - a_max[1]))
    return float(np.hypot(dx, dy))


def infer_geometric_relationships(
    world: WorldIR,
    options: Optional[GeometricRelationshipOptions] = None,
) -> List[InferredRelationship]:
    """Compute geometric relationships between entities in a WorldIR scene.

    Infers:
    1. supported_by (and supports / rests_on)
    2. mounted_on (and attached_to)
    3. part_of (and contains)
    4. inside (and contains)
    """
    opts = options or GeometricRelationshipOptions()
    results: List[InferredRelationship] = []

    # Pre-extract bounds for all entities
    bounds_map: Dict[str, Bounds3] = {}
    centroids: Dict[str, Point3] = {}
    for eid, ent in world.entities.items():
        b = get_entity_bounds(world, ent)
        if b is not None:
            bounds_map[eid] = b
            c = get_entity_centroid(world, ent)
            if c is not None:
                centroids[eid] = c

    entities = list(world.entities.values())

    # 1. part_of from explicit component_ids
    for ent in entities:
        for cid in ent.component_ids:
            if cid in world.entities:
                results.append(InferredRelationship(
                    source_id=cid,
                    target_id=ent.id,
                    kind=RelationshipKind.PART_OF,
                    confidence=1.0,
                    basis="structural_component",
                    reason=f"{cid} declared in component_ids of {ent.id}",
                ))
                if opts.infer_reciprocals:
                    results.append(InferredRelationship(
                        source_id=ent.id,
                        target_id=cid,
                        kind=RelationshipKind.CONTAINS,
                        confidence=1.0,
                        basis="structural_component",
                        reason=f"{ent.id} contains declared component {cid}",
                    ))

    # 2. Geometric pairwise checks
    for i, a in enumerate(entities):
        if a.id not in bounds_map:
            continue
        a_bounds = bounds_map[a.id]
        a_cent = centroids[a.id]

        for j, b in enumerate(entities):
            if i == j or b.id not in bounds_map:
                continue
            b_bounds = bounds_map[b.id]
            b_cent = centroids[b.id]

            # ---- A: inside / contains (Rooms & Spaces) ----
            if _is_space(b) and not _is_space(a):
                # Check if A's centroid is inside B's bounding volume
                (b_min, b_max) = b_bounds
                if (b_min[0] <= a_cent[0] <= b_max[0] and
                    b_min[1] <= a_cent[1] <= b_max[1] and
                    b_min[2] - 0.1 <= a_cent[2] <= b_max[2] + 0.1):
                    results.append(InferredRelationship(
                        source_id=a.id,
                        target_id=b.id,
                        kind=RelationshipKind.INSIDE,
                        confidence=0.95,
                        basis="geometric_containment",
                        reason=f"{a.id} centroid inside space {b.id}",
                    ))
                    if opts.infer_reciprocals:
                        results.append(InferredRelationship(
                            source_id=b.id,
                            target_id=a.id,
                            kind=RelationshipKind.CONTAINS,
                            confidence=0.95,
                            basis="geometric_containment",
                            reason=f"Space {b.id} contains {a.id}",
                        ))
                    continue

            # ---- B: part_of (Openings in Walls or Strong 3D Containment) ----
            if _is_opening(a) and _is_wall(b):
                # Opening inside wall
                h_dist = _horizontal_distance_boxes(a_bounds, b_bounds)
                if h_dist <= 0.15:
                    # Check vertical overlap
                    if max(a_bounds[0][2], b_bounds[0][2]) < min(a_bounds[1][2], b_bounds[1][2]):
                        results.append(InferredRelationship(
                            source_id=a.id,
                            target_id=b.id,
                            kind=RelationshipKind.PART_OF,
                            confidence=0.90,
                            basis="geometric_opening_host",
                            reason=f"Opening {a.id} embedded in wall {b.id}",
                        ))
                        if opts.infer_reciprocals:
                            results.append(InferredRelationship(
                                source_id=b.id,
                                target_id=a.id,
                                kind=RelationshipKind.CONTAINS,
                                confidence=0.90,
                                basis="geometric_opening_host",
                                reason=f"Wall {b.id} contains opening {a.id}",
                            ))
                        continue

            cont_ratio = _containment_ratio(a_bounds, b_bounds)
            if cont_ratio >= opts.containment_threshold and not _is_space(b):
                results.append(InferredRelationship(
                    source_id=a.id,
                    target_id=b.id,
                    kind=RelationshipKind.PART_OF,
                    confidence=float(min(0.99, cont_ratio)),
                    basis="geometric_box_containment",
                    reason=f"{a.id} volume is {cont_ratio:.2f} contained within {b.id}",
                ))
                if opts.infer_reciprocals:
                    results.append(InferredRelationship(
                        source_id=b.id,
                        target_id=a.id,
                        kind=RelationshipKind.CONTAINS,
                        confidence=float(min(0.99, cont_ratio)),
                        basis="geometric_box_containment",
                        reason=f"{b.id} contains {a.id} ({cont_ratio:.2f})",
                    ))
                continue

            # ---- C: mounted_on / attached_to (Wall Mounting) ----
            if _is_wall(b) and not _is_wall(a) and not _is_space(a) and not _is_floor(a):
                h_dist = _horizontal_distance_boxes(a_bounds, b_bounds)
                if h_dist <= opts.wall_mount_tolerance_m:
                    # Check vertical overlap with wall
                    z_ov = min(a_bounds[1][2], b_bounds[1][2]) - max(a_bounds[0][2], b_bounds[0][2])
                    if z_ov > 0.05 and b_bounds[0][2] <= a_cent[2] <= b_bounds[1][2]:
                        conf = max(0.5, 1.0 - (h_dist / opts.wall_mount_tolerance_m) * 0.4)
                        results.append(InferredRelationship(
                            source_id=a.id,
                            target_id=b.id,
                            kind=RelationshipKind.MOUNTED_ON,
                            confidence=float(conf),
                            basis="geometric_wall_proximity",
                            reason=f"{a.id} within {h_dist:.3f}m of wall {b.id}",
                        ))
                        if opts.infer_reciprocals:
                            results.append(InferredRelationship(
                                source_id=b.id,
                                target_id=a.id,
                                kind=RelationshipKind.ATTACHED_TO,
                                confidence=float(conf),
                                basis="geometric_wall_proximity",
                                reason=f"Wall {b.id} hosts mounted {a.id}",
                            ))
                        continue

            # ---- D: supported_by / supports (Gravity Contact) ----
            # B is below A, supporting A
            if not _is_space(a) and not _is_space(b) and not _is_wall(a):
                z_diff = a_bounds[0][2] - b_bounds[1][2]  # bottom of A minus top of B
                if -opts.vertical_support_penetration_m <= z_diff <= opts.vertical_support_tolerance_m:
                    # Check horizontal overlap
                    ov_ratio = _horizontal_overlap_ratio(a_bounds, b_bounds)
                    if ov_ratio >= opts.min_horizontal_overlap_ratio or _point_in_xy_box(a_cent, b_bounds):
                        conf = max(0.5, 0.95 - abs(z_diff) * 2.0)
                        results.append(InferredRelationship(
                            source_id=a.id,
                            target_id=b.id,
                            kind=RelationshipKind.SUPPORTED_BY,
                            confidence=float(conf),
                            basis="geometric_gravity_support",
                            reason=f"{a.id} bottom resting on {b.id} top (gap: {z_diff:.3f}m, xy_ov: {ov_ratio:.2f})",
                        ))
                        if opts.infer_reciprocals:
                            results.append(InferredRelationship(
                                source_id=b.id,
                                target_id=a.id,
                                kind=RelationshipKind.SUPPORTS,
                                confidence=float(conf),
                                basis="geometric_gravity_support",
                                reason=f"{b.id} supports {a.id}",
                            ))

    return results


def apply_geometric_relationships(
    world: WorldIR,
    options: Optional[GeometricRelationshipOptions] = None,
) -> int:
    """Infer geometric relationships and attach them directly to entities in the world.

    Skips relationships that already exist on the entity to prevent duplicates.
    Returns the count of newly added relationships.
    """
    inferred = infer_geometric_relationships(world, options)
    added_count = 0

    for rel in inferred:
        entity = world.entities.get(rel.source_id)
        if entity is None:
            continue

        # Check if already declared
        exists = any(
            existing.target_id == rel.target_id and existing.kind == rel.kind
            for existing in entity.relationships
        )
        if not exists:
            entity.relationships.append(rel.to_relationship())
            added_count += 1

    return added_count
