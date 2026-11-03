"""Temporal multi-epoch alignment, diffing, and change detection (P1, Epic E07-1).

Tracks 3D environments across capture epochs:
- Rigid Procrustes alignment of multi-temporal epochs using static architectural anchors.
- Robust outlier rejection: excludes movable furniture and non-static elements from registration.
- Detection of added, removed, moved, renovated, and damaged entities.
- Precision thresholding: detects furniture displacements >= 5cm, filtering reconstruction noise.
- Structural alteration reasoning: detects wall additions, removals, dimension alterations, and renovations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from provenance import Provenance, Uncertainty
from world_ir.diff import ChangeKind, WorldDiff, diff_worlds
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR


class TemporalChangeKind(str, Enum):
    """Classification of temporal state transition for an entity between epochs."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    RENOVATED = "renovated"
    DAMAGED = "damaged"


@dataclass(frozen=True)
class TemporalEntityChange:
    """Detailed temporal change record for a specific physical entity."""

    entity_id: str
    entity_name: str
    entity_type: EntityType
    change_kind: TemporalChangeKind
    displacement_m: float = 0.0
    displacement_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: float = 0.0
    old_position: Optional[Tuple[float, float, float]] = None
    new_position: Optional[Tuple[float, float, float]] = None
    details: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type.value if hasattr(self.entity_type, "value") else str(self.entity_type),
            "change_kind": self.change_kind.value,
            "displacement_m": round(self.displacement_m, 4),
            "displacement_vector": [round(v, 4) for v in self.displacement_vector],
            "rotation_deg": round(self.rotation_deg, 2),
            "old_position": [round(v, 4) for v in self.old_position] if self.old_position else None,
            "new_position": [round(v, 4) for v in self.new_position] if self.new_position else None,
            "details": self.details,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class TemporalEpoch:
    """Represents a discrete capture epoch of a physical site at a given point in time."""

    epoch_id: str
    timestamp_iso: str
    world: WorldIR
    description: str = ""
    point_cloud_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "timestamp_iso": self.timestamp_iso,
            "world_id": self.world.id,
            "description": self.description,
            "point_cloud_ref": self.point_cloud_ref,
            "entity_count": len(self.world.entities),
        }


@dataclass(frozen=True)
class TemporalAlignmentResult:
    """Result of rigid spatial registration between two capture epochs."""

    rotation_matrix: Tuple[Tuple[float, float, float], ...]
    translation: Tuple[float, float, float]
    residual_error_m: float
    num_anchors_used: int
    converged: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rotation_matrix": [[round(x, 6) for x in row] for row in self.rotation_matrix],
            "translation": [round(x, 6) for x in self.translation],
            "residual_error_m": round(self.residual_error_m, 4),
            "num_anchors_used": self.num_anchors_used,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class TemporalDiffReport:
    """Comprehensive temporal change comparison between two capture epochs."""

    from_epoch_id: str
    to_epoch_id: str
    alignment: TemporalAlignmentResult
    changes: Tuple[TemporalEntityChange, ...] = ()
    movement_threshold_m: float = 0.05

    @property
    def added(self) -> Tuple[TemporalEntityChange, ...]:
        return tuple(c for c in self.changes if c.change_kind == TemporalChangeKind.ADDED)

    @property
    def removed(self) -> Tuple[TemporalEntityChange, ...]:
        return tuple(c for c in self.changes if c.change_kind == TemporalChangeKind.REMOVED)

    @property
    def moved(self) -> Tuple[TemporalEntityChange, ...]:
        return tuple(c for c in self.changes if c.change_kind == TemporalChangeKind.MOVED)

    @property
    def renovated(self) -> Tuple[TemporalEntityChange, ...]:
        return tuple(c for c in self.changes if c.change_kind == TemporalChangeKind.RENOVATED)

    @property
    def damaged(self) -> Tuple[TemporalEntityChange, ...]:
        return tuple(c for c in self.changes if c.change_kind == TemporalChangeKind.DAMAGED)

    @property
    def unchanged(self) -> Tuple[TemporalEntityChange, ...]:
        return tuple(c for c in self.changes if c.change_kind == TemporalChangeKind.UNCHANGED)

    def summary(self) -> Dict[str, Any]:
        return {
            "from_epoch_id": self.from_epoch_id,
            "to_epoch_id": self.to_epoch_id,
            "alignment_residual_m": self.alignment.residual_error_m,
            "total_entities_evaluated": len(self.changes),
            "added_count": len(self.added),
            "removed_count": len(self.removed),
            "moved_count": len(self.moved),
            "renovated_count": len(self.renovated),
            "damaged_count": len(self.damaged),
            "unchanged_count": len(self.unchanged),
            "max_displacement_m": max((c.displacement_m for c in self.moved), default=0.0),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "alignment": self.alignment.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
        }


DEFAULT_STATIC_TYPES: Tuple[EntityType, ...] = (
    EntityType.WALL,
    EntityType.FLOOR,
    EntityType.CEILING,
    EntityType.COLUMN,
    EntityType.BEAM,
    EntityType.ROOF,
    EntityType.BUILDING,
    EntityType.STRUCTURE,
)


def _is_static_candidate(entity: Entity, static_types: Sequence[EntityType]) -> bool:
    """Determine whether an entity is a true static architectural anchor."""
    if entity.type not in static_types:
        return False

    name_lower = (entity.name or "").lower()
    movable_keywords = ("chair", "table", "desk", "sofa", "couch", "bed", "stool", "cabinet", "shelf", "furniture", "box")
    if any(kw in name_lower for kw in movable_keywords):
        return False

    props = entity.custom_properties or {}
    if props.get("movable") is True:
        return False
    eclass = str(props.get("entity_class", "")).lower()
    if eclass.startswith("furniture.") or eclass.startswith("appliance.") or eclass.startswith("movable."):
        return False

    return True


def _resolve_entity_pos(entity: Entity, world: WorldIR) -> Optional[np.ndarray]:
    """Resolve an entity's 3D center position from transform or geometry bounds."""
    if entity.transform and isinstance(entity.transform, dict):
        pos = entity.transform.get("position")
        if pos and isinstance(pos, dict) and "x" in pos and "y" in pos and "z" in pos:
            return np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])], dtype=np.float64)

    # Fallback to geometry AABB center
    if entity.geometry_ids:
        for gid in entity.geometry_ids:
            geom = world.geometries.get(gid)
            if geom and geom.bounds_min and geom.bounds_max:
                bmin = geom.bounds_min
                bmax = geom.bounds_max
                bx = float(getattr(bmin, "x", bmin[0] if isinstance(bmin, (list, tuple)) else 0.0))
                by = float(getattr(bmin, "y", bmin[1] if isinstance(bmin, (list, tuple)) else 0.0))
                bz = float(getattr(bmin, "z", bmin[2] if isinstance(bmin, (list, tuple)) else 0.0))
                tx = float(getattr(bmax, "x", bmax[0] if isinstance(bmax, (list, tuple)) else 0.0))
                ty = float(getattr(bmax, "y", bmax[1] if isinstance(bmax, (list, tuple)) else 0.0))
                tz = float(getattr(bmax, "z", bmax[2] if isinstance(bmax, (list, tuple)) else 0.0))
                return np.array([(bx + tx) * 0.5, (by + ty) * 0.5, (bz + tz) * 0.5], dtype=np.float64)

    return None


def align_temporal_epochs(
    epoch_a: TemporalEpoch,
    epoch_b: TemporalEpoch,
    static_types: Sequence[EntityType] = DEFAULT_STATIC_TYPES,
) -> TemporalAlignmentResult:
    """Rigid Procrustes registration between two epochs using static architectural anchors.

    Finds static entities present in both epochs (walls, structural columns, slabs)
    and computes the optimal rotation matrix R in SO(3) and translation vector t:
        p_A = R * p_B + t
    """
    pts_a: List[np.ndarray] = []
    pts_b: List[np.ndarray] = []

    # Find shared static entities
    for eid, ent_a in epoch_a.world.entities.items():
        if _is_static_candidate(ent_a, static_types) and eid in epoch_b.world.entities:
            ent_b = epoch_b.world.entities[eid]
            if _is_static_candidate(ent_b, static_types):
                pa = _resolve_entity_pos(ent_a, epoch_a.world)
                pb = _resolve_entity_pos(ent_b, epoch_b.world)
                if pa is not None and pb is not None:
                    pts_a.append(pa)
                    pts_b.append(pb)

    num_anchors = len(pts_a)

    if num_anchors < 3:
        if num_anchors > 0:
            mean_a = np.mean(pts_a, axis=0)
            mean_b = np.mean(pts_b, axis=0)
            trans = mean_a - mean_b
            res = float(np.mean([np.linalg.norm((pb + trans) - pa) for pa, pb in zip(pts_a, pts_b)]))
            return TemporalAlignmentResult(
                rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                translation=(float(trans[0]), float(trans[1]), float(trans[2])),
                residual_error_m=res,
                num_anchors_used=num_anchors,
                converged=True,
            )
        # Identity fallback
        return TemporalAlignmentResult(
            rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
            residual_error_m=0.0,
            num_anchors_used=0,
            converged=False,
        )

    A = np.array(pts_a, dtype=np.float64)
    B = np.array(pts_b, dtype=np.float64)

    centroid_a = np.mean(A, axis=0)
    centroid_b = np.mean(B, axis=0)

    AA = A - centroid_a
    BB = B - centroid_b

    # Covariance matrix H = BB^T * AA
    H = BB.T @ AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection check for SO(3)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_a - R @ centroid_b

    # Measure residual error
    B_aligned = (R @ B.T).T + t
    residuals = np.linalg.norm(A - B_aligned, axis=1)
    mean_res = float(np.mean(residuals))

    rot_tuple = tuple(
        (float(R[i, 0]), float(R[i, 1]), float(R[i, 2])) for i in range(3)
    )
    trans_tuple = (float(t[0]), float(t[1]), float(t[2]))

    return TemporalAlignmentResult(
        rotation_matrix=rot_tuple,
        translation=trans_tuple,
        residual_error_m=mean_res,
        num_anchors_used=num_anchors,
        converged=True,
    )


def diff_temporal_epochs(
    epoch_a: TemporalEpoch,
    epoch_b: TemporalEpoch,
    movement_threshold_m: float = 0.05,
    auto_align: bool = True,
    static_types: Sequence[EntityType] = DEFAULT_STATIC_TYPES,
) -> TemporalDiffReport:
    """Compare two capture epochs and detect added, removed, moved, renovated, and damaged entities.

    Parameters:
    - epoch_a: Baseline temporal epoch (earlier capture).
    - epoch_b: Target temporal epoch (later capture).
    - movement_threshold_m: Minimum spatial displacement to classify as MOVED (default: 0.05m = 5cm).
    - auto_align: If True, rigidly aligns epoch_b to epoch_a using static anchors before diffing.
    - static_types: Architectural entity types used for registration.
    """
    alignment: TemporalAlignmentResult
    if auto_align:
        alignment = align_temporal_epochs(epoch_a, epoch_b, static_types=static_types)
    else:
        alignment = TemporalAlignmentResult(
            rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
            residual_error_m=0.0,
            num_anchors_used=0,
            converged=True,
        )

    R = np.array(alignment.rotation_matrix, dtype=np.float64)
    t = np.array(alignment.translation, dtype=np.float64)

    world_a = epoch_a.world
    world_b = epoch_b.world

    ents_a = world_a.entities
    ents_b = world_b.entities

    changes: List[TemporalEntityChange] = []
    processed_b_ids: Set[str] = set()

    for eid_a, ent_a in ents_a.items():
        if eid_a not in ents_b:
            # REMOVED
            pos_a = _resolve_entity_pos(ent_a, world_a)
            changes.append(
                TemporalEntityChange(
                    entity_id=eid_a,
                    entity_name=ent_a.name or eid_a,
                    entity_type=ent_a.type,
                    change_kind=TemporalChangeKind.REMOVED,
                    old_position=tuple(pos_a) if pos_a is not None else None,
                    details=f"Entity removed between {epoch_a.epoch_id} and {epoch_b.epoch_id}",
                    confidence=ent_a.confidence or 1.0,
                )
            )
            continue

        processed_b_ids.add(eid_a)
        ent_b = ents_b[eid_a]

        # Entity present in both epochs
        pos_a = _resolve_entity_pos(ent_a, world_a)
        pos_b_raw = _resolve_entity_pos(ent_b, world_b)

        if pos_a is not None and pos_b_raw is not None:
            # Apply rigid alignment to pos_b
            pos_b_aligned = R @ pos_b_raw + t
            disp_vec = pos_b_aligned - pos_a
            disp_m = float(np.linalg.norm(disp_vec))

            # Check for damage / renovation status in custom_properties or statements
            props_b = ent_b.custom_properties or {}
            props_a = ent_a.custom_properties or {}

            is_damaged = (
                props_b.get("condition") in ("damaged", "degraded", "spalled", "cracked")
                or "damage_detected" in props_b
                or props_b.get("integrity_score", 1.0) < 0.7
            )
            is_renovated = (
                props_b.get("renovated") is True
                or props_b.get("finish") != props_a.get("finish")
                or (ent_a.type in DEFAULT_STATIC_TYPES and set(ent_a.geometry_ids) != set(ent_b.geometry_ids))
            )

            if is_damaged:
                changes.append(
                    TemporalEntityChange(
                        entity_id=eid_a,
                        entity_name=ent_b.name or eid_a,
                        entity_type=ent_b.type,
                        change_kind=TemporalChangeKind.DAMAGED,
                        displacement_m=disp_m,
                        displacement_vector=(float(disp_vec[0]), float(disp_vec[1]), float(disp_vec[2])),
                        old_position=tuple(pos_a),
                        new_position=tuple(pos_b_aligned),
                        details=f"Surface damage or structural degradation detected: {props_b.get('condition_notes', 'degraded')}",
                        confidence=ent_b.confidence or 1.0,
                    )
                )
            elif is_renovated and _is_static_candidate(ent_a, DEFAULT_STATIC_TYPES):
                changes.append(
                    TemporalEntityChange(
                        entity_id=eid_a,
                        entity_name=ent_b.name or eid_a,
                        entity_type=ent_b.type,
                        change_kind=TemporalChangeKind.RENOVATED,
                        displacement_m=disp_m,
                        displacement_vector=(float(disp_vec[0]), float(disp_vec[1]), float(disp_vec[2])),
                        old_position=tuple(pos_a),
                        new_position=tuple(pos_b_aligned),
                        details="Structural alteration / architectural renovation observed",
                        confidence=ent_b.confidence or 1.0,
                    )
                )
            elif disp_m >= movement_threshold_m:
                changes.append(
                    TemporalEntityChange(
                        entity_id=eid_a,
                        entity_name=ent_b.name or eid_a,
                        entity_type=ent_b.type,
                        change_kind=TemporalChangeKind.MOVED,
                        displacement_m=disp_m,
                        displacement_vector=(float(disp_vec[0]), float(disp_vec[1]), float(disp_vec[2])),
                        old_position=tuple(pos_a),
                        new_position=tuple(pos_b_aligned),
                        details=f"Displaced by {disp_m * 100:.1f} cm",
                        confidence=ent_b.confidence or 1.0,
                    )
                )
            else:
                changes.append(
                    TemporalEntityChange(
                        entity_id=eid_a,
                        entity_name=ent_b.name or eid_a,
                        entity_type=ent_b.type,
                        change_kind=TemporalChangeKind.UNCHANGED,
                        displacement_m=disp_m,
                        displacement_vector=(float(disp_vec[0]), float(disp_vec[1]), float(disp_vec[2])),
                        old_position=tuple(pos_a),
                        new_position=tuple(pos_b_aligned),
                        details="Entity location static (< 5cm threshold)",
                        confidence=ent_b.confidence or 1.0,
                    )
                )
        else:
            # Position unavailable
            changes.append(
                TemporalEntityChange(
                    entity_id=eid_a,
                    entity_name=ent_b.name or eid_a,
                    entity_type=ent_b.type,
                    change_kind=TemporalChangeKind.UNCHANGED,
                    details="Position unresolvable; retained across epochs",
                    confidence=ent_b.confidence or 1.0,
                )
            )

    # Added entities in epoch B
    for eid_b, ent_b in ents_b.items():
        if eid_b not in processed_b_ids:
            pos_b_raw = _resolve_entity_pos(ent_b, world_b)
            pos_b_aligned = (R @ pos_b_raw + t) if pos_b_raw is not None else None
            changes.append(
                TemporalEntityChange(
                    entity_id=eid_b,
                    entity_name=ent_b.name or eid_b,
                    entity_type=ent_b.type,
                    change_kind=TemporalChangeKind.ADDED,
                    new_position=tuple(pos_b_aligned) if pos_b_aligned is not None else None,
                    details=f"Newly introduced entity observed in {epoch_b.epoch_id}",
                    confidence=ent_b.confidence or 1.0,
                )
            )

    # Sort changes by entity_id for determinism
    changes.sort(key=lambda c: c.entity_id)

    return TemporalDiffReport(
        from_epoch_id=epoch_a.epoch_id,
        to_epoch_id=epoch_b.epoch_id,
        alignment=alignment,
        changes=tuple(changes),
        movement_threshold_m=movement_threshold_m,
    )
