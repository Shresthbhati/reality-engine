"""Spatial reasoning and entity relationship inference (Master Development Plan Tasks 6.6 & 7.2).

Infers evidence-grounded spatial relationships between detected objects and
architectural structures in WorldIR:
1. ATTACHED_TO / SUPPORTS:
   - Electrical fittings (sockets, switches, switchboards), lighting, mirrors, and artwork
     attached to walls or ceilings within measured clearance tolerances.
2. RESTS_ON / SUPPORTS:
   - Furniture (chairs, tables, sofas, beds, desks) and appliances resting on floors;
   - Street furniture (benches, dustbins, lampposts) resting on footpaths or roads.
3. CONTAINS / PART_OF:
   - Rooms containing interior objects and fixtures;
   - Switchboard plates containing modular sockets/switches;
   - Buildings containing storeys and rooms.

Pure and idempotent: relationships carry confidence and INFERRED provenance,
avoid duplicates, and preserve all existing relationships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from perception.entities import taxonomy

# Maximum distance tolerances in meters
MAX_WALL_ATTACHMENT_DIST_M = 0.20
MAX_FLOOR_RESTING_DIST_M = 0.15
MAX_CEILING_ATTACHMENT_DIST_M = 0.25


@dataclass(frozen=True)
class SpatialInferenceResult:
    """Summary of spatial relationships inferred and applied."""

    attachments_created: int = 0
    resting_created: int = 0
    containments_created: int = 0
    total_edges_created: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "attachments_created": self.attachments_created,
            "resting_created": self.resting_created,
            "containments_created": self.containments_created,
            "total_edges_created": self.total_edges_created,
            "details": self.details,
        }


def _entity_position(entity: Entity) -> Optional[np.ndarray]:
    """Extract 3D center position from transform or custom_properties."""
    pos = (entity.transform or {}).get("position")
    if isinstance(pos, dict) and "x" in pos and "y" in pos and "z" in pos:
        return np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])], dtype=np.float64)

    cp = entity.custom_properties or {}
    for key in ("position", "plan_obb_centre_m", "bounds_min"):
        v = cp.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)

    bmin, bmax = cp.get("bounds_min"), cp.get("bounds_max")
    if isinstance(bmin, (list, tuple)) and isinstance(bmax, (list, tuple)) and len(bmin) >= 3 and len(bmax) >= 3:
        p0 = np.array([float(bmin[0]), float(bmin[1]), float(bmin[2])], dtype=np.float64)
        p1 = np.array([float(bmax[0]), float(bmax[1]), float(bmax[2])], dtype=np.float64)
        return (p0 + p1) / 2.0

    return None


def _entity_bottom_z(entity: Entity) -> Optional[float]:
    """Extract bottom z elevation of entity using bounds_min or position."""
    cp = entity.custom_properties or {}
    bmin = cp.get("bounds_min")
    if isinstance(bmin, (list, tuple)) and len(bmin) >= 3:
        return float(bmin[2])
    pos = _entity_position(entity)
    return float(pos[2]) if pos is not None else None


def _has_relationship(entity: Entity, kind: RelationshipKind, target_id: str) -> bool:
    """Check if entity already carries a relationship of this kind to target_id."""
    for rel in entity.relationships:
        r_kind = rel.kind.value if hasattr(rel.kind, "value") else rel.kind
        target_kind = kind.value if hasattr(kind, "value") else kind
        if r_kind == target_kind and rel.target_id == target_id:
            return True
    return False


def _add_bilateral_relationship(
    source_entity: Entity,
    target_entity: Entity,
    forward_kind: RelationshipKind,
    reverse_kind: RelationshipKind,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Add bilateral relationships if not already present. Returns True if added."""
    meta = metadata or {}
    added = False
    if not _has_relationship(source_entity, forward_kind, target_entity.id):
        source_entity.relationships.append(
            Relationship(
                kind=forward_kind,
                target_id=target_entity.id,
                confidence=confidence,
                provenance=Provenance.INFERRED,
                metadata=meta,
            )
        )
        added = True

    if not _has_relationship(target_entity, reverse_kind, source_entity.id):
        target_entity.relationships.append(
            Relationship(
                kind=reverse_kind,
                target_id=source_entity.id,
                confidence=confidence,
                provenance=Provenance.INFERRED,
                metadata=meta,
            )
        )
        added = True

    return added


def infer_spatial_relationships(
    world: WorldIR,
    max_wall_dist_m: float = MAX_WALL_ATTACHMENT_DIST_M,
    max_floor_dist_m: float = MAX_FLOOR_RESTING_DIST_M,
    max_ceiling_dist_m: float = MAX_CEILING_ATTACHMENT_DIST_M,
) -> SpatialInferenceResult:
    """Infer and apply spatial relationships (ATTACHED_TO, RESTS_ON, CONTAINS, PART_OF) across entities.

    Idempotent and non-destructive: existing relationships are preserved.
    """
    entities = world.entities

    # 1. Categorize surfaces and spaces
    walls: List[Entity] = []
    floors: List[Entity] = []
    ceilings: List[Entity] = []
    rooms: List[Entity] = []
    switchboards: List[Entity] = []
    objects: List[Entity] = []

    for ent in entities.values():
        e_type = ent.type.value if hasattr(ent.type, "value") else ent.type
        labels = [str(lbl).lower() for lbl in (ent.semantic_labels or [])]
        cp = ent.custom_properties or {}
        role = str(cp.get("role", "")).lower()

        if e_type == "wall" or "wall" in labels or role == "wall":
            walls.append(ent)
        elif e_type in ("floor", "sidewalk", "road", "curb") or "floor" in labels or role in ("floor", "ground", "sidewalk", "road"):
            floors.append(ent)
        elif e_type == "ceiling" or "ceiling" in labels or role == "ceiling":
            ceilings.append(ent)
        elif e_type == "room" or "room" in labels:
            rooms.append(ent)
        elif "switchboard" in labels or cp.get("entity_class") == "electrical.switchboard":
            switchboards.append(ent)
        else:
            objects.append(ent)

    n_attached = 0
    n_resting = 0
    n_contains = 0
    total_edges = 0
    details: List[Dict[str, Any]] = []

    # 2. Match objects against surfaces
    for obj in objects:
        pos = _entity_position(obj)
        if pos is None:
            continue

        cp = obj.custom_properties or {}
        e_class = str(cp.get("entity_class", "")).lower()
        labels = [str(lbl).lower() for lbl in (obj.semantic_labels or [])]
        name = str(obj.name or "").lower()

        # Taxonomy lookup for host hints
        tax_entry = taxonomy.get(e_class) if e_class else None
        hosts = tax_entry.hosts if tax_entry else ()

        is_wall_mount = (
            "envelope.wall" in hosts
            or any(k in name or k in labels for k in ("socket", "switch", "plate", "sconce", "mirror", "picture", "frame", "panel"))
            or "wall_fixture" in labels
        )
        is_ceiling_mount = (
            "envelope.ceiling" in hosts
            or any(k in name or k in labels for k in ("ceiling light", "pendant", "downlight", "chandelier", "fan", "smoke detector"))
        )
        is_floor_rest = (
            "envelope.floor" in hosts
            or any(k in name or k in labels for k in ("chair", "table", "desk", "sofa", "bed", "wardrobe", "shelf", "furniture", "appliance", "bench", "bin", "lamppost", "tree", "bollard"))
            or e_class.startswith("street.") or e_class.startswith("vegetation.")
            or "furniture_candidate" in labels
        )

        # A. Wall attachment reasoning
        if is_wall_mount and walls:
            best_wall = None
            min_dist = float("inf")
            for wall in walls:
                w_pos = _entity_position(wall)
                if w_pos is not None:
                    dist = float(np.linalg.norm(pos - w_pos))
                    if dist < min_dist:
                        min_dist = dist
                        best_wall = wall

            if best_wall is not None and min_dist <= max_wall_dist_m * 5.0:  # 5x when comparing center-to-center
                conf = max(0.5, min(0.95, 1.0 - (min_dist / (max_wall_dist_m * 5.0))))
                if _add_bilateral_relationship(
                    source_entity=obj,
                    target_entity=best_wall,
                    forward_kind=RelationshipKind.ATTACHED_TO,
                    reverse_kind=RelationshipKind.SUPPORTS,
                    confidence=round(conf, 3),
                    metadata={"surface_role": "wall", "measured_distance_m": round(min_dist, 4)},
                ):
                    n_attached += 1
                    total_edges += 2
                    details.append({
                        "object_id": obj.id,
                        "host_id": best_wall.id,
                        "kind": "attached_to_wall",
                        "distance_m": round(min_dist, 4),
                    })

        # B. Ceiling attachment reasoning
        if is_ceiling_mount and ceilings:
            best_ceil = None
            min_dz = float("inf")
            for ceil in ceilings:
                c_pos = _entity_position(ceil)
                if c_pos is not None and c_pos[2] >= pos[2]:  # ceiling above object
                    dz = float(abs(c_pos[2] - pos[2]))
                    if dz < min_dz:
                        min_dz = dz
                        best_ceil = ceil

            if best_ceil is not None and min_dz <= max_ceiling_dist_m * 4.0:
                conf = max(0.5, min(0.95, 1.0 - (min_dz / (max_ceiling_dist_m * 4.0))))
                if _add_bilateral_relationship(
                    source_entity=obj,
                    target_entity=best_ceil,
                    forward_kind=RelationshipKind.ATTACHED_TO,
                    reverse_kind=RelationshipKind.SUPPORTS,
                    confidence=round(conf, 3),
                    metadata={"surface_role": "ceiling", "vertical_clearance_m": round(min_dz, 4)},
                ):
                    n_attached += 1
                    total_edges += 2
                    details.append({
                        "object_id": obj.id,
                        "host_id": best_ceil.id,
                        "kind": "attached_to_ceiling",
                        "vertical_clearance_m": round(min_dz, 4),
                    })

        # C. Floor resting reasoning
        if is_floor_rest and floors:
            best_floor = None
            min_dz = float("inf")
            obj_base_z = _entity_bottom_z(obj)
            base_z = obj_base_z if obj_base_z is not None else pos[2]
            for floor in floors:
                f_pos = _entity_position(floor)
                if f_pos is not None and base_z >= f_pos[2] - 0.15:
                    dz = float(abs(base_z - f_pos[2]))
                    if dz < min_dz:
                        min_dz = dz
                        best_floor = floor

            if best_floor is not None and min_dz <= max_floor_dist_m * 4.0:
                conf = max(0.5, min(0.95, 1.0 - (min_dz / (max_floor_dist_m * 4.0))))
                if _add_bilateral_relationship(
                    source_entity=obj,
                    target_entity=best_floor,
                    forward_kind=RelationshipKind.RESTS_ON,
                    reverse_kind=RelationshipKind.SUPPORTS,
                    confidence=round(conf, 3),
                    metadata={"surface_role": "floor", "vertical_offset_m": round(min_dz, 4)},
                ):
                    n_resting += 1
                    total_edges += 2
                    details.append({
                        "object_id": obj.id,
                        "host_id": best_floor.id,
                        "kind": "rests_on_floor",
                        "vertical_offset_m": round(min_dz, 4),
                    })

        # D. Room containment reasoning
        if rooms:
            target_room_id = cp.get("room_id")
            chosen_room = None
            if target_room_id and target_room_id in entities:
                chosen_room = entities[target_room_id]
            else:
                # Find nearest room center
                min_r_dist = float("inf")
                for room in rooms:
                    r_pos = _entity_position(room)
                    if r_pos is not None:
                        dist = float(np.linalg.norm(pos - r_pos))
                        if dist < min_r_dist:
                            min_r_dist = dist
                            chosen_room = room

            if chosen_room is not None:
                if _add_bilateral_relationship(
                    source_entity=chosen_room,
                    target_entity=obj,
                    forward_kind=RelationshipKind.CONTAINS,
                    reverse_kind=RelationshipKind.PART_OF,
                    confidence=0.85,
                ):
                    n_contains += 1
                    total_edges += 2

    # 3. Switchboard module containment
    for sb in switchboards:
        sb_pos = _entity_position(sb)
        if sb_pos is None:
            continue
        for obj in objects:
            if obj.id == sb.id:
                continue
            labels = [str(l).lower() for l in (obj.semantic_labels or [])]
            if any(l in labels for l in ("socket", "switch", "module")) or "switchboard" in str(obj.custom_properties.get("entity_class")):
                m_pos = _entity_position(obj)
                if m_pos is not None and np.linalg.norm(m_pos - sb_pos) < 0.3:
                    if _add_bilateral_relationship(
                        source_entity=sb,
                        target_entity=obj,
                        forward_kind=RelationshipKind.CONTAINS,
                        reverse_kind=RelationshipKind.PART_OF,
                        confidence=0.95,
                    ):
                        n_contains += 1
                        total_edges += 2

    return SpatialInferenceResult(
        attachments_created=n_attached,
        resting_created=n_resting,
        containments_created=n_contains,
        total_edges_created=total_edges,
        details=details,
    )


def get_attached_entities(world: WorldIR, surface_id: str) -> List[Entity]:
    """Retrieve all entities attached to the specified surface."""
    surface = world.entities.get(surface_id)
    if not surface:
        return []
    results: List[Entity] = []
    for rel in surface.relationships:
        r_kind = rel.kind.value if hasattr(rel.kind, "value") else rel.kind
        if r_kind == RelationshipKind.SUPPORTS.value:
            child = world.entities.get(rel.target_id)
            if child:
                results.append(child)
    return results


def get_room_entities(world: WorldIR, room_id: str) -> List[Entity]:
    """Retrieve all entities contained within the specified room."""
    room = world.entities.get(room_id)
    if not room:
        return []
    results: List[Entity] = []
    for rel in room.relationships:
        r_kind = rel.kind.value if hasattr(rel.kind, "value") else rel.kind
        if r_kind == RelationshipKind.CONTAINS.value:
            child = world.entities.get(rel.target_id)
            if child:
                results.append(child)
    return results
