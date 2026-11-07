"""Architectural Fixtures: Sockets, Switches, Light Fixtures, Radiators (T-3.4).

Provides measured 3D placement of architectural fixtures on their host wall planes
with measured height above floor slab, satisfying the +-5 cm accuracy tolerance
on measured rooms for architectural and interior survey deliverables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from provenance import Provenance
from world_ir.schema_v1 import Entity, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR

Point3 = tuple[float, float, float]
Bounds3 = tuple[Point3, Point3]

#: Maximum perpendicular distance (m) to associate a fixture with a wall plane.
MAX_WALL_FIXTURE_DIST_M = 0.35

#: Accuracy tolerance for reported dimensions on measured rooms (5 cm).
FIXTURE_HEIGHT_TOLERANCE_M = 0.05

#: Standard nominal height ranges above floor (m) for verification guidance.
NOMINAL_FIXTURE_HEIGHTS = {
    "socket": (0.15, 0.45),        # Standard baseboard/low sockets
    "socket_counter": (0.95, 1.25),# Countertop/appliance sockets
    "switch": (0.90, 1.30),        # Wall switches
    "light_wall": (1.60, 2.20),    # Wall sconces / accent lights
    "radiator": (0.05, 0.40),      # Heating panels / baseboard radiators
    "thermostat": (1.20, 1.60),    # Climate controls
}


@dataclass(frozen=True)
class FixturePlacement:
    """A measured architectural fixture placed on a host wall plane (T-3.4)."""

    entity_id: str
    kind: str
    host_wall_id: str | None
    position: Point3
    height_above_floor_m: float
    distance_to_wall_m: float
    confidence: float
    is_wall_mounted: bool = True
    is_floor_standing: bool = False
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "host_wall_id": self.host_wall_id,
            "position": list(self.position),
            "height_above_floor_m": round(self.height_above_floor_m, 3),
            "distance_to_wall_m": round(self.distance_to_wall_m, 4),
            "confidence": round(self.confidence, 3),
            "is_wall_mounted": self.is_wall_mounted,
            "is_floor_standing": self.is_floor_standing,
            "evidence_ids": list(self.evidence_ids),
        }


def classify_fixture(entity: Entity) -> str | None:
    """Classify an entity as a specific architectural fixture kind or None if not a fixture.

    Recognizes:
    - socket (power sockets, duplex outlets, data ports)
    - switch (light switches, switchboards)
    - light_fixture (wall sconces, vanity lights, accent lights)
    - radiator (radiator heaters, HVAC baseboard units)
    - thermostat (climate controls)
    """
    labels = [str(lbl).lower() for lbl in (entity.semantic_labels or [])]
    name = str(entity.name or "").lower()
    cp = entity.custom_properties or {}
    e_class = str(cp.get("entity_class") or cp.get("class_name") or cp.get("category") or "").lower()

    # Check sockets
    if any(k in e_class for k in ("socket", "outlet")) or any(
        k in name or k in labels for k in ("socket", "outlet", "power_point", "receptacle")
    ):
        return "socket"

    # Check switches
    if any(k in e_class for k in ("switch", "switchboard")) or any(
        k in name or k in labels for k in ("switch", "light_switch", "switchboard")
    ):
        return "switch"

    # Check radiators / heaters
    if any(k in e_class for k in ("radiator", "heater", "hvac.radiator")) or any(
        k in name or k in labels for k in ("radiator", "heater", "convector", "baseboard_heater")
    ):
        return "radiator"

    # Check wall light fixtures
    if any(k in e_class for k in ("sconce", "wall_light", "light_fixture")) or any(
        k in name or k in labels for k in ("sconce", "wall_light", "wall light", "vanity light", "picture light")
    ):
        return "light_fixture"

    # Check thermostats
    if "thermostat" in e_class or any(k in name or k in labels for k in ("thermostat", "climate_control")):
        return "thermostat"

    # Generic fixture fallback
    e_type = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
    if e_type == "fixture" or "fixture" in labels or e_class.startswith("fixture"):
        return "fixture"

    return None


def _get_entity_position(entity: Entity) -> Point3 | None:
    """Extract (x, y, z) position from entity transform or custom properties."""
    tr = entity.transform or {}
    pos = tr.get("position")
    if isinstance(pos, dict) and "x" in pos and "y" in pos and "z" in pos:
        return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    cp = entity.custom_properties or {}
    for key in ("position", "plan_obb_centre_m"):
        v = cp.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return (float(v[0]), float(v[1]), float(v[2]))

    bmin, bmax = cp.get("bounds_min"), cp.get("bounds_max")
    if isinstance(bmin, (list, tuple)) and isinstance(bmax, (list, tuple)) and len(bmin) >= 3 and len(bmax) >= 3:
        return (
            (float(bmin[0]) + float(bmax[0])) / 2.0,
            (float(bmin[1]) + float(bmax[1])) / 2.0,
            (float(bmin[2]) + float(bmax[2])) / 2.0,
        )

    bb = cp.get("bounding_box")
    if isinstance(bb, (list, tuple)) and len(bb) == 2:
        b0, b1 = bb
        return (
            (float(b0[0]) + float(b1[0])) / 2.0,
            (float(b0[1]) + float(b1[1])) / 2.0,
            (float(b0[2]) + float(b1[2])) / 2.0,
        )
    return None


def _wall_segment(wall: Entity) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Extract (x0, y0), (x1, y1) start and end coordinates of a wall line."""
    cp = wall.custom_properties or {}

    # 1. Check explicit endpoints
    p0 = cp.get("start_point_m") or cp.get("p0")
    p1 = cp.get("end_point_m") or cp.get("p1")
    if isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple)):
        return ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))

    # 2. Check bounding box or bounds
    bb = cp.get("bounding_box") or cp.get("bounds")
    if isinstance(bb, (list, tuple)) and len(bb) == 2:
        bmin, bmax = bb[0], bb[1]
        dx = abs(float(bmax[0]) - float(bmin[0]))
        dy = abs(float(bmax[1]) - float(bmin[1]))
        if dx >= dy:
            # Wall runs mainly along X
            y_mid = (float(bmin[1]) + float(bmax[1])) / 2.0
            return ((float(bmin[0]), y_mid), (float(bmax[0]), y_mid))
        else:
            # Wall runs mainly along Y
            x_mid = (float(bmin[0]) + float(bmax[0])) / 2.0
            return ((x_mid, float(bmin[1])), (x_mid, float(bmax[1])))

    # 3. Check plan OBB
    centre = cp.get("plan_obb_centre_m")
    size = cp.get("plan_obb_size_m")
    angle = float(cp.get("plan_obb_angle_rad", 0.0))
    if isinstance(centre, (list, tuple)) and isinstance(size, (list, tuple)):
        cx, cy = float(centre[0]), float(centre[1])
        length = max(float(size[0]), float(size[1]))
        dx = (length / 2.0) * math.cos(angle)
        dy = (length / 2.0) * math.sin(angle)
        return ((cx - dx, cy - dy), (cx + dx, cy + dy))

    # 4. Fallback from position and normal
    pos = _get_entity_position(wall)
    if pos is not None:
        normal = cp.get("normal", [0.0, 1.0, 0.0])
        nx, ny = float(normal[0]), float(normal[1])
        # Perpendicular tangent
        tx, ty = -ny, nx
        length = float(cp.get("length_m", 3.0))
        return (
            (pos[0] - (length / 2.0) * tx, pos[1] - (length / 2.0) * ty),
            (pos[0] + (length / 2.0) * tx, pos[1] + (length / 2.0) * ty),
        )

    return None


def point_to_segment_distance(
    pt: tuple[float, float],
    seg_p0: tuple[float, float],
    seg_p1: tuple[float, float],
) -> tuple[float, float]:
    """Compute (perpendicular_dist_m, projection_param_t) from point to 2D line segment."""
    x0, y0 = seg_p0
    x1, y1 = seg_p1
    dx = x1 - x0
    dy = y1 - y0
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-9:
        dist = math.hypot(pt[0] - x0, pt[1] - y0)
        return dist, 0.0

    t = ((pt[0] - x0) * dx + (pt[1] - y0) * dy) / seg_len_sq
    # Clamp projection slightly beyond bounds (e.g. corner proximity margin)
    clamped_t = max(-0.05, min(1.05, t))
    proj_x = x0 + clamped_t * dx
    proj_y = y0 + clamped_t * dy
    dist = math.hypot(pt[0] - proj_x, pt[1] - proj_y)
    return dist, clamped_t


def find_host_wall(
    fixture_pos: Point3,
    walls: Sequence[Entity],
    max_dist_m: float = MAX_WALL_FIXTURE_DIST_M,
) -> tuple[Entity | None, float]:
    """Find the closest host wall plane to a fixture position using planar segment distance.

    Returns:
        (best_wall_entity, perpendicular_distance_m)
    """
    best_wall: Entity | None = None
    min_dist = float("inf")

    pt_xy = (fixture_pos[0], fixture_pos[1])
    pt_z = fixture_pos[2]

    for wall in walls:
        # Check vertical span of wall
        w_cp = wall.custom_properties or {}
        bb = w_cp.get("bounding_box") or w_cp.get("bounds")
        if isinstance(bb, (list, tuple)) and len(bb) == 2:
            z_min = float(bb[0][2]) - 0.20
            z_max = float(bb[1][2]) + 0.20
            if not (z_min <= pt_z <= z_max):
                continue

        seg = _wall_segment(wall)
        if seg is not None:
            dist, t = point_to_segment_distance(pt_xy, seg[0], seg[1])
            # Fixture must be within lateral segment bounds (with small corner tolerance)
            if -0.05 <= t <= 1.05 and dist < min_dist:
                min_dist = dist
                best_wall = wall
        else:
            # Point fallback
            w_pos = _get_entity_position(wall)
            if w_pos is not None:
                dist = math.hypot(pt_xy[0] - w_pos[0], pt_xy[1] - w_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_wall = wall

    if best_wall is not None and min_dist <= max_dist_m:
        return best_wall, min_dist

    return None, min_dist


def find_floor_elevation(
    fixture_pos: Point3,
    floors: Sequence[Entity],
    default_z: float = 0.0,
) -> float:
    """Find the reference floor elevation directly below a fixture position."""
    best_floor_z = default_z
    min_dz = float("inf")

    pt_z = fixture_pos[2]

    for floor in floors:
        f_cp = floor.custom_properties or {}
        f_pos = _get_entity_position(floor)
        floor_z = f_pos[2] if f_pos is not None else float(f_cp.get("elevation_m", default_z))

        # Floor must be at or below fixture elevation
        if floor_z <= pt_z + 0.05:
            dz = pt_z - floor_z
            if dz < min_dz:
                min_dz = dz
                best_floor_z = floor_z

    return best_floor_z


def place_fixture_on_wall(
    fixture: Entity,
    walls: Sequence[Entity],
    floor_z: float = 0.0,
    max_dist_m: float = MAX_WALL_FIXTURE_DIST_M,
) -> FixturePlacement | None:
    """Measure and place an architectural fixture on its host wall plane (T-3.4).

    Calculates:
    - Host wall association via orthogonal projection
    - Measured height above floor plane: `height_above_floor_m = fixture.z - floor_z`
    - In-place updates to entity custom_properties and bilateral relationships.
    """
    pos = _get_entity_position(fixture)
    if pos is None:
        return None

    kind = classify_fixture(fixture)
    if kind is None:
        return None

    best_wall, wall_dist = find_host_wall(pos, walls, max_dist_m=max_dist_m)
    height_above_floor = float(pos[2] - floor_z)

    # Confidence calculation based on wall proximity and reasonable height
    prox_conf = max(0.50, min(0.98, 1.0 - (wall_dist / max_dist_m) * 0.45))
    is_wall_mounted = best_wall is not None
    is_floor_standing = kind == "radiator" and height_above_floor <= 0.15

    # Update fixture properties in-place
    fixture.custom_properties = fixture.custom_properties or {}
    fixture.custom_properties["fixture_kind"] = kind
    fixture.custom_properties["height_above_floor_m"] = round(height_above_floor, 3)
    fixture.custom_properties["elevation_measured"] = True
    fixture.custom_properties["elevation_basis"] = "measured_floor_relative"

    if best_wall is not None:
        fixture.custom_properties["host_wall_id"] = best_wall.id
        fixture.custom_properties["distance_to_wall_m"] = round(wall_dist, 4)
        fixture.custom_properties["mount_type"] = "wall_mounted"

        # Bilateral relationship: fixture ATTACHED_TO wall, wall SUPPORTS fixture
        _add_relationship_if_missing(
            source=fixture,
            target_id=best_wall.id,
            kind=RelationshipKind.ATTACHED_TO,
            confidence=round(prox_conf, 3),
            meta={"measured_height_m": round(height_above_floor, 3), "wall_dist_m": round(wall_dist, 4)},
        )
        _add_relationship_if_missing(
            source=best_wall,
            target_id=fixture.id,
            kind=RelationshipKind.SUPPORTS,
            confidence=round(prox_conf, 3),
            meta={"fixture_kind": kind, "fixture_id": fixture.id},
        )
    else:
        fixture.custom_properties["mount_type"] = "unattached"

    return FixturePlacement(
        entity_id=fixture.id,
        kind=kind,
        host_wall_id=best_wall.id if best_wall else None,
        position=pos,
        height_above_floor_m=height_above_floor,
        distance_to_wall_m=wall_dist if best_wall else 999.0,
        confidence=prox_conf if best_wall else 0.50,
        is_wall_mounted=is_wall_mounted,
        is_floor_standing=is_floor_standing,
        evidence_ids=tuple(fixture.custom_properties.get("evidence_ids", ())),
    )


def place_all_fixtures(
    world: WorldIR,
    max_dist_m: float = MAX_WALL_FIXTURE_DIST_M,
) -> list[FixturePlacement]:
    """Place and measure all architectural fixtures in a WorldIR model (T-3.4).

    Identifies all sockets, switches, light fixtures, and radiators, associates them
    with their host wall planes, computes measured heights above floor, and persists
    relationships.
    """
    walls: list[Entity] = []
    floors: list[Entity] = []
    candidate_fixtures: list[Entity] = []

    for ent in world.entities.values():
        e_type = ent.type.value if hasattr(ent.type, "value") else str(ent.type)
        labels = [str(lbl).lower() for lbl in (ent.semantic_labels or [])]
        cp = ent.custom_properties or {}
        role = str(cp.get("role", "")).lower()

        if e_type == "wall" or "wall" in labels or role == "wall":
            walls.append(ent)
        elif e_type in ("floor", "slab") or "floor" in labels or role == "floor":
            floors.append(ent)
        else:
            kind = classify_fixture(ent)
            if kind is not None:
                candidate_fixtures.append(ent)

    placements: list[FixturePlacement] = []
    for fixture in candidate_fixtures:
        pos = _get_entity_position(fixture)
        if pos is None:
            continue
        floor_z = find_floor_elevation(pos, floors, default_z=0.0)
        placement = place_fixture_on_wall(
            fixture=fixture,
            walls=walls,
            floor_z=floor_z,
            max_dist_m=max_dist_m,
        )
        if placement is not None:
            placements.append(placement)

    return placements


def evaluate_fixture_accuracy(
    predicted_placements: Sequence[FixturePlacement],
    ground_truth: Sequence[dict[str, Any]],
    tolerance_m: float = FIXTURE_HEIGHT_TOLERANCE_M,
) -> dict[str, Any]:
    """Evaluate measured fixture positions against ground truth (e.g. 8.6 eval set) (T-3.4).

    Computes:
    - precision, recall, F1-score
    - height error and RMS error (m)
    - fraction of fixtures within the +-5 cm tolerance
    - host wall assignment accuracy
    """
    if not ground_truth:
        return {
            "total_gt": 0,
            "total_pred": len(predicted_placements),
            "recall": 1.0 if not predicted_placements else 0.0,
            "precision": 1.0 if not predicted_placements else 0.0,
            "f1": 1.0 if not predicted_placements else 0.0,
            "within_tolerance_ratio": 1.0,
            "height_rmse_m": 0.0,
            "wall_accuracy": 1.0,
        }

    matched_gt: set[int] = set()
    height_errors: list[float] = []
    pos_errors: list[float] = []
    wall_matches: list[bool] = []
    within_tol_count = 0

    for pred in predicted_placements:
        best_gt_idx = None
        min_3d_dist = float("inf")

        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            gt_kind = str(gt.get("kind", "")).lower()
            if gt_kind and gt_kind != pred.kind:
                continue

            gt_pos = gt["position"]
            dist = math.sqrt(sum((pred.position[k] - gt_pos[k]) ** 2 for k in range(3)))
            if dist < min_3d_dist and dist <= 0.50:  # 50cm matching window
                min_3d_dist = dist
                best_gt_idx = i

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            gt_item = ground_truth[best_gt_idx]

            # Height difference
            gt_h = float(gt_item.get("height_above_floor_m", gt_item["position"][2]))
            h_err = abs(pred.height_above_floor_m - gt_h)
            height_errors.append(h_err)
            pos_errors.append(min_3d_dist)

            if h_err <= tolerance_m:
                within_tol_count += 1

            # Wall assignment
            gt_wall = gt_item.get("host_wall_id")
            if gt_wall:
                wall_matches.append(gt_wall == pred.host_wall_id)

    n_gt = len(ground_truth)
    n_pred = len(predicted_placements)
    n_tp = len(matched_gt)

    recall = n_tp / n_gt if n_gt > 0 else 0.0
    precision = n_tp / n_pred if n_pred > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    height_rmse = math.sqrt(sum(e * e for e in height_errors) / len(height_errors)) if height_errors else 0.0
    within_tol_ratio = within_tol_count / n_tp if n_tp > 0 else 0.0
    wall_acc = sum(wall_matches) / len(wall_matches) if wall_matches else 1.0

    return {
        "total_gt": n_gt,
        "total_pred": n_pred,
        "true_positives": n_tp,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "height_rmse_m": round(height_rmse, 4),
        "within_tolerance_ratio": round(within_tol_ratio, 3),
        "wall_accuracy": round(wall_acc, 3),
        "meets_5cm_tolerance": within_tol_ratio >= 0.90,
    }


def _add_relationship_if_missing(
    source: Entity,
    target_id: str,
    kind: RelationshipKind,
    confidence: float,
    meta: dict[str, Any] | None = None,
) -> None:
    """Helper to append relationship edge if not already present."""
    source.relationships = source.relationships or []
    for r in source.relationships:
        r_kind = r.kind.value if hasattr(r.kind, "value") else r.kind
        target_kind = kind.value if hasattr(kind, "value") else kind
        if r_kind == target_kind and r.target_id == target_id:
            return

    source.relationships.append(
        Relationship(
            kind=kind,
            target_id=target_id,
            confidence=confidence,
            provenance=Provenance.OBSERVED,
            metadata=meta or {},
        )
    )
