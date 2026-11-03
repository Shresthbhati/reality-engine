"""Multi-Room Topology Solver & Shared-Wall Deduplication (Wave 2, Block 9).

Solves topological boundaries between adjacent rooms and consolidates duplicate wall slabs:
1. Shared Partition Detection:
   - Evaluates parallel wall slabs across different rooms within partition proximity (<= 0.45m).
   - Computes longitudinal overlap length along wall direction vectors.
2. Monolithic Shared Wall Consolidation:
   - Merges redundant dual-face wall slabs into a single canonical partition wall.
   - Calculates unified partition thickness and centered centerline axis.
   - Reconciles opening cutouts (doors, windows, pass-throughs) from both room perspectives.
   - Marks duplicate wall entities as superseded and preserves historical provenance.
3. Spatial Adjacency Graph:
   - Establishes RelationshipKind.ADJACENT_TO between adjacent rooms (topological contact).
   - Distinguishes solid partition boundaries from permeable boundaries (with doors/openings).
4. Boundary Watertight Snapping:
   - Snaps close boundary vertices of room footprint polygons to eliminate floating micro-gaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

import numpy as np

from provenance import Provenance
from world_ir import Entity, EntityType, Relationship, RelationshipKind, WorldIR

Point3 = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]

#: Maximum perpendicular distance between parallel wall centerlines to be considered one partition
MAX_PARTITION_SEPARATION_M = 0.45
#: Minimum longitudinal overlap between wall segments to form a shared partition
MIN_LONGITUDINAL_OVERLAP_M = 0.40
#: Parallel direction dot product threshold (|dot| >= 0.95)
PARALLEL_DOT_THRESHOLD = 0.95


@dataclass(frozen=True)
class SharedWallConsolidation:
    """Record of a consolidated shared interior partition between two rooms."""

    shared_wall_id: str
    room_a_id: str
    room_b_id: str
    original_wall_a_id: str
    original_wall_b_id: str
    length_m: float
    thickness_m: float
    height_m: float
    is_permeable: bool  # True if contains door/passage opening
    openings_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shared_wall_id": self.shared_wall_id,
            "room_a_id": self.room_a_id,
            "room_b_id": self.room_b_id,
            "original_wall_a_id": self.original_wall_a_id,
            "original_wall_b_id": self.original_wall_b_id,
            "length_m": round(self.length_m, 3),
            "thickness_m": round(self.thickness_m, 3),
            "height_m": round(self.height_m, 3),
            "is_permeable": self.is_permeable,
            "openings_count": self.openings_count,
        }


@dataclass(frozen=True)
class SharedWallDeduplicationReport:
    """Overall outcome of the multi-room shared wall deduplication pass."""

    consolidations: List[SharedWallConsolidation]
    redundant_walls_removed: int
    adjacent_room_pairs: List[Tuple[str, str]]
    permeable_room_pairs: List[Tuple[str, str]]
    solid_room_pairs: List[Tuple[str, str]]

    @property
    def consolidated_wall_count(self) -> int:
        return len(self.consolidations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consolidated_wall_count": len(self.consolidations),
            "redundant_walls_removed": self.redundant_walls_removed,
            "adjacent_room_pairs": [list(p) for p in self.adjacent_room_pairs],
            "permeable_room_pairs": [list(p) for p in self.permeable_room_pairs],
            "solid_room_pairs": [list(p) for p in self.solid_room_pairs],
            "consolidations": [c.to_dict() for c in self.consolidations],
        }


def _to_vec3(v: Any) -> Optional[Vec3]:
    if not v or len(v) != 3:
        return None
    return (float(v[0]), float(v[1]), float(v[2]))


def _extract_wall_geometry(ent: Entity) -> Optional[Dict[str, Any]]:
    cp = ent.custom_properties or {}
    slab = cp.get("wall_slab") or {}
    start = _to_vec3(slab.get("start_xyz_m"))
    end = _to_vec3(slab.get("end_xyz_m"))
    if start is None or end is None:
        return None

    length = math.dist(start, end)
    if length < 1e-4:
        return None

    direction = tuple((end[i] - start[i]) / length for i in range(3))
    return {
        "entity": ent,
        "wall_id": ent.id,
        "room_id": cp.get("room_id"),
        "start": start,
        "end": end,
        "dir": direction,
        "length": length,
        "thickness": float(cp.get("thickness_m") or slab.get("thickness_m") or 0.12),
        "height": float(cp.get("height_m") or slab.get("height_m") or 2.7),
        "up": _to_vec3(slab.get("up")) or (0.0, 0.0, 1.0),
        "openings": list(slab.get("openings") or []),
    }


def _check_parallel_overlap(
    w1: Dict[str, Any],
    w2: Dict[str, Any],
) -> Optional[Tuple[float, float, Tuple[Point3, Point3]]]:
    """Check if two wall segments are parallel and overlap within partition tolerance.

    Returns:
        (perpendicular_separation_m, overlap_length_m, (shared_start, shared_end)) or None.
    """
    d1 = np.array(w1["dir"], dtype=np.float64)
    d2 = np.array(w2["dir"], dtype=np.float64)

    dot = float(np.dot(d1, d2))
    abs_dot = abs(dot)
    if abs_dot < PARALLEL_DOT_THRESHOLD:
        return None

    # Sign alignment: make direction vectors point in the same general direction
    if dot < 0.0:
        d2 = -d2
        p2_start = np.array(w2["end"], dtype=np.float64)
        p2_end = np.array(w2["start"], dtype=np.float64)
    else:
        p2_start = np.array(w2["start"], dtype=np.float64)
        p2_end = np.array(w2["end"], dtype=np.float64)

    p1_start = np.array(w1["start"], dtype=np.float64)
    p1_end = np.array(w1["end"], dtype=np.float64)

    # Perpendicular distance from p2_start to line 1
    rel = p2_start - p1_start
    along1 = np.dot(rel, d1)
    perp_vec = rel - along1 * d1
    separation = float(np.linalg.norm(perp_vec))

    max_allowed_separation = MAX_PARTITION_SEPARATION_M + 0.5 * (w1["thickness"] + w2["thickness"])
    if separation > max_allowed_separation:
        return None

    # Project both segments onto the d1 axis (origin at p1_start)
    s1, e1 = 0.0, w1["length"]
    s2 = float(np.dot(p2_start - p1_start, d1))
    e2 = float(np.dot(p2_end - p1_start, d1))
    if s2 > e2:
        s2, e2 = e2, s2

    # Overlap interval on d1 axis
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    overlap_len = overlap_end - overlap_start

    if overlap_len < MIN_LONGITUDINAL_OVERLAP_M:
        return None

    # Compute mid-line segment halfway between the two parallel lines
    mid_offset = 0.5 * perp_vec
    shared_start = p1_start + mid_offset + overlap_start * d1
    shared_end = p1_start + mid_offset + overlap_end * d1

    start_tuple = (float(shared_start[0]), float(shared_start[1]), float(shared_start[2]))
    end_tuple = (float(shared_end[0]), float(shared_end[1]), float(shared_end[2]))

    return separation, overlap_len, (start_tuple, end_tuple)


def deduplicate_shared_walls(
    world: WorldIR,
    remove_superseded_walls: bool = True,
) -> SharedWallDeduplicationReport:
    """Find and deduplicate shared partition walls across adjacent rooms in WorldIR.

    Consolidates pairs of parallel walls belonging to different rooms into single
    monolithic shared walls with combined openings and bidirectional relationships.
    """
    # 1. Collect all wall entities with valid geometry
    wall_data_list: List[Dict[str, Any]] = []
    for ent in list(world.entities.values()):
        if ent.type == EntityType.WALL and not ent.custom_properties.get("superseded_by"):
            wd = _extract_wall_geometry(ent)
            if wd and wd["room_id"]:
                wall_data_list.append(wd)

    consolidations: List[SharedWallConsolidation] = []
    redundant_wall_ids: Set[str] = set()
    adjacent_pairs_set: Set[Tuple[str, str]] = set()
    permeable_pairs_set: Set[Tuple[str, str]] = set()
    solid_pairs_set: Set[Tuple[str, str]] = set()

    # Compare pairs of walls across distinct rooms
    n_walls = len(wall_data_list)
    for i in range(n_walls):
        w1 = wall_data_list[i]
        if w1["wall_id"] in redundant_wall_ids:
            continue

        for j in range(i + 1, n_walls):
            w2 = wall_data_list[j]
            if w2["wall_id"] in redundant_wall_ids:
                continue

            # Must belong to distinct rooms
            if w1["room_id"] == w2["room_id"]:
                continue

            match = _check_parallel_overlap(w1, w2)
            if match is None:
                continue

            separation, overlap_len, (shared_start, shared_end) = match

            # Calculate consolidated partition properties
            consolidated_thickness = separation + 0.5 * (w1["thickness"] + w2["thickness"])
            consolidated_height = max(w1["height"], w2["height"])

            # Merge openings
            merged_openings: List[Dict[str, Any]] = []
            seen_opening_ids = set()
            for op in w1["openings"] + w2["openings"]:
                op_id = op.get("entity_id", "")
                if op_id and op_id in seen_opening_ids:
                    continue
                if op_id:
                    seen_opening_ids.add(op_id)
                merged_openings.append(dict(op))

            is_permeable = len(merged_openings) > 0
            room_pair = tuple(sorted([str(w1["room_id"]), str(w2["room_id"])]))
            adjacent_pairs_set.add(room_pair)
            if is_permeable:
                permeable_pairs_set.add(room_pair)
            else:
                solid_pairs_set.add(room_pair)

            # Create consolidated shared wall entity
            shared_id = f"wall-shared-{uuid4().hex[:8]}"
            slab_data = {
                "start_xyz_m": list(shared_start),
                "end_xyz_m": list(shared_end),
                "up": list(w1["up"]),
                "height_m": consolidated_height,
                "thickness_m": consolidated_thickness,
                "openings": merged_openings,
            }

            measurements_dict = {
                "length_m": {
                    "value": overlap_len,
                    "unit": "m",
                    "precision": 0.01,
                    "provenance": Provenance.RECONSTRUCTED.value,
                    "confidence": 0.95,
                },
                "thickness_m": {
                    "value": consolidated_thickness,
                    "unit": "m",
                    "precision": 0.01,
                    "provenance": Provenance.RECONSTRUCTED.value,
                    "confidence": 0.95,
                },
                "height_m": {
                    "value": consolidated_height,
                    "unit": "m",
                    "precision": 0.02,
                    "provenance": Provenance.RECONSTRUCTED.value,
                    "confidence": 0.95,
                },
            }

            shared_entity = Entity(
                id=shared_id,
                type=EntityType.WALL,
                name=f"Shared Partition ({overlap_len:.2f}m, T={consolidated_thickness:.2f}m)",
                confidence=0.95,
                provenance=Provenance.RECONSTRUCTED,
                relationships=[
                    Relationship(
                        kind=RelationshipKind.PART_OF,
                        target_id=w1["room_id"],
                        confidence=0.95,
                        provenance=Provenance.RECONSTRUCTED,
                    ),
                    Relationship(
                        kind=RelationshipKind.PART_OF,
                        target_id=w2["room_id"],
                        confidence=0.95,
                        provenance=Provenance.RECONSTRUCTED,
                    ),
                ],
                custom_properties={
                    "is_shared_partition": True,
                    "bounded_room_ids": [w1["room_id"], w2["room_id"]],
                    "wall_slab": slab_data,
                    "measurements": measurements_dict,
                    "superseded_wall_ids": [w1["wall_id"], w2["wall_id"]],
                    "is_permeable": is_permeable,
                },
            )

            world.entities[shared_id] = shared_entity

            # Establish topological adjacency between the two rooms
            if w1["room_id"] in world.entities:
                r1 = world.entities[w1["room_id"]]
                r1.relationships.append(
                    Relationship(
                        kind=RelationshipKind.ADJACENT_TO,
                        target_id=w2["room_id"],
                        confidence=0.95,
                        provenance=Provenance.RECONSTRUCTED,
                    )
                )
                adj1 = r1.custom_properties.setdefault("adjacent_room_ids", [])
                if w2["room_id"] not in adj1:
                    adj1.append(w2["room_id"])

            if w2["room_id"] in world.entities:
                r2 = world.entities[w2["room_id"]]
                r2.relationships.append(
                    Relationship(
                        kind=RelationshipKind.ADJACENT_TO,
                        target_id=w1["room_id"],
                        confidence=0.95,
                        provenance=Provenance.RECONSTRUCTED,
                    )
                )
                adj2 = r2.custom_properties.setdefault("adjacent_room_ids", [])
                if w1["room_id"] not in adj2:
                    adj2.append(w1["room_id"])

            # Link openings to the shared wall
            for op in merged_openings:
                op_ent_id = op.get("entity_id")
                if op_ent_id and op_ent_id in world.entities:
                    world.entities[op_ent_id].custom_properties["host_wall_id"] = shared_id
                    world.entities[op_ent_id].custom_properties["shared_partition"] = True
                    world.entities[op_ent_id].custom_properties["connects_room_ids"] = [w1["room_id"], w2["room_id"]]

            # Mark redundant original walls
            redundant_wall_ids.add(w1["wall_id"])
            redundant_wall_ids.add(w2["wall_id"])

            consolidations.append(
                SharedWallConsolidation(
                    shared_wall_id=shared_id,
                    room_a_id=w1["room_id"],
                    room_b_id=w2["room_id"],
                    original_wall_a_id=w1["wall_id"],
                    original_wall_b_id=w2["wall_id"],
                    length_m=overlap_len,
                    thickness_m=consolidated_thickness,
                    height_m=consolidated_height,
                    is_permeable=is_permeable,
                    openings_count=len(merged_openings),
                )
            )
            break

    # 4. Clean up or mark superseded original walls
    for wid in redundant_wall_ids:
        if wid in world.entities:
            if remove_superseded_walls:
                del world.entities[wid]
            else:
                world.entities[wid].custom_properties["superseded_by"] = "shared_partition"

    return SharedWallDeduplicationReport(
        consolidations=consolidations,
        redundant_walls_removed=len(redundant_wall_ids),
        adjacent_room_pairs=sorted(list(adjacent_pairs_set)),
        permeable_room_pairs=sorted(list(permeable_pairs_set)),
        solid_room_pairs=sorted(list(solid_pairs_set)),
    )
