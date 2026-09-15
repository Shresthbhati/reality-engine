"""Architectural element classification from plane geometry (P7-03).

Scope (deliberately cut down from the ledger's broad "walls/floors/
ceilings/doors/windows/stairs/columns/beams/roofs/facades/roads/curbs/
infrastructure/terrain/vegetation" line -- see PENDING_IMPLEMENTATION.md
P7-03 and this module's test file for the full rationale):

  - wall / floor / ceiling: classified from plane orientation (normal
    vs. world up) and, for floor-vs-ceiling, relative position among
    the room's horizontal planes -- the same plane hypotheses
    evidence/promote_rooms.py already classifies via orientation.py,
    but produced here as an independent geometry-only classifier
    (perception layer, not the evidence/promotion layer).
  - doorway/opening: a gap in a wall plane's inlier coverage at floor
    height, sized like a door. Detected from the wall's own inlier
    point cloud -- no label-only guessing.

Explicitly OUT of scope, skipped rather than faked:
  - door-leaf/window-as-distinct-from-opening, stairs, columns, beams,
    roofs: this repo has no detector producing evidence for any of
    these (no stair-step detection, no column/beam segmentation), so
    classifying them would be guessing, not classification.
  - facades/roads/curbs/infrastructure/terrain/vegetation: outdoor/
    city-scale concepts with no synthetic or real fixture in this repo
    to classify against.

A plane that does not clearly match wall/floor/ceiling orientation, or
a horizontal plane whose relative position can't be resolved, is
UNKNOWN -- never guessed into the nearest category.

Deterministic: pure float math, no clocks, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

#: A plane counts as a WALL when its normal is nearly perpendicular to
#: world up (|normal . up| small -- the plane itself stands vertical).
WALL_MAX_UP_DOT = 0.35

#: A plane counts as horizontal (floor/ceiling candidate) when its
#: normal is nearly parallel to world up.
HORIZONTAL_MIN_UP_DOT = 0.85

#: Doorway/opening size heuristics (interior residential doors).
DOOR_MIN_WIDTH_M = 0.4
DOOR_MAX_WIDTH_M = 1.6

#: Height band above the floor scanned for wall coverage: a real
#: doorway is empty here (a genuine wall is not), regardless of what
#: happens above head height.
DOOR_SCAN_BAND_M = 0.4

#: Lateral bucket resolution for the wall-coverage scan.
SCAN_BUCKET_M = 0.05

#: Bins within this many buckets of the wall's own lateral edges are
#: excluded from opening detection -- edge dropout (a plane's inlier
#: set naturally thins near its boundary) must not read as a doorway.
EDGE_MARGIN_BUCKETS = 2


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: Vec3) -> Vec3:
    n = math.sqrt(_dot(v, v))
    if n < 1e-12:
        raise ValueError(f"cannot normalize a near-zero vector {v!r}")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class PlaneInput:
    """Minimal plane facts this classifier needs -- deliberately its
    own small type (not evidence.promote_rooms.PlaneSummary) since
    this is an independent perception-layer classifier, not a
    consumer of the room-promotion pipeline."""

    plane_id: str
    normal: Vec3  # unit
    centroid: Vec3
    bounds_min: Vec3
    bounds_max: Vec3
    #: Real inlier positions, required only for doorway detection.
    inlier_positions: Tuple[Vec3, ...] = ()


@dataclass(frozen=True)
class ArchitecturalElement:
    """A classified architectural element (WorldIR EntityType value in
    `element_type`: 'wall' | 'floor' | 'ceiling' | 'door' | 'unknown')."""

    element_id: str
    element_type: str
    source_plane_id: str
    reason: str
    bounds_min: Optional[Vec3] = None
    bounds_max: Optional[Vec3] = None

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "source_plane_id": self.source_plane_id,
            "reason": self.reason,
            "bounds_min": list(self.bounds_min) if self.bounds_min else None,
            "bounds_max": list(self.bounds_max) if self.bounds_max else None,
        }


def classify_planes(
    planes: Sequence[PlaneInput], up: Vec3
) -> List[ArchitecturalElement]:
    """Classify each plane as wall / floor / ceiling / unknown from
    orientation and, for horizontal planes, relative height.

    Never guesses: a plane whose orientation is neither clearly
    vertical nor clearly horizontal (e.g. a sloped roof plane -- out
    of scope, no roof detector exists) is UNKNOWN. A horizontal plane
    whose floor-vs-ceiling role can't be resolved from context is also
    UNKNOWN rather than picked arbitrarily.
    """
    up = _normalize(up)
    results: List[ArchitecturalElement] = []

    horizontal: List[Tuple[PlaneInput, float]] = []  # (plane, height)
    walls: List[PlaneInput] = []

    for plane in planes:
        up_dot = _dot(plane.normal, up)
        height = _dot(plane.centroid, up)
        if abs(up_dot) <= WALL_MAX_UP_DOT:
            walls.append(plane)
            results.append(ArchitecturalElement(
                element_id=f"wall-{plane.plane_id}",
                element_type="wall",
                source_plane_id=plane.plane_id,
                reason=f"near-vertical normal (|n.up|={abs(up_dot):.3f})",
                bounds_min=plane.bounds_min, bounds_max=plane.bounds_max,
            ))
        elif abs(up_dot) >= HORIZONTAL_MIN_UP_DOT:
            horizontal.append((plane, height))
        else:
            results.append(ArchitecturalElement(
                element_id=f"unknown-{plane.plane_id}",
                element_type="unknown",
                source_plane_id=plane.plane_id,
                reason=(
                    f"orientation ambiguous (|n.up|={abs(up_dot):.3f}, "
                    f"neither wall nor floor/ceiling range)"
                ),
                bounds_min=plane.bounds_min, bounds_max=plane.bounds_max,
            ))

    if len(horizontal) >= 2:
        lowest = min(horizontal, key=lambda ph: ph[1])
        highest = max(horizontal, key=lambda ph: ph[1])
        for plane, h in horizontal:
            if plane.plane_id == lowest[0].plane_id and h < highest[1]:
                label, reason = "floor", "lowest horizontal plane in the room"
            elif plane.plane_id == highest[0].plane_id and h > lowest[1]:
                label, reason = "ceiling", "highest horizontal plane in the room"
            else:
                label, reason = "unknown", (
                    "horizontal plane at a mid-height with other horizontal "
                    "planes present -- floor/ceiling role not resolvable"
                )
            results.append(ArchitecturalElement(
                element_id=f"{label}-{plane.plane_id}",
                element_type=label,
                source_plane_id=plane.plane_id,
                reason=reason,
                bounds_min=plane.bounds_min, bounds_max=plane.bounds_max,
            ))
    elif len(horizontal) == 1:
        plane, h = horizontal[0]
        if walls:
            wall_low = min(_dot(w.bounds_min, up) for w in walls)
            wall_high = max(_dot(w.bounds_max, up) for w in walls)
            near_low = abs(h - wall_low) <= abs(h - wall_high)
            if near_low:
                label, reason = "floor", "sits at the walls' low extent"
            else:
                label, reason = "ceiling", "sits at the walls' high extent"
        else:
            label, reason = "unknown", (
                "single horizontal plane with no wall planes to resolve "
                "floor-vs-ceiling against"
            )
        results.append(ArchitecturalElement(
            element_id=f"{label}-{plane.plane_id}",
            element_type=label,
            source_plane_id=plane.plane_id,
            reason=reason,
            bounds_min=plane.bounds_min, bounds_max=plane.bounds_max,
        ))

    return results


def detect_wall_opening(
    wall: PlaneInput, up: Vec3, floor_height: float
) -> Optional[ArchitecturalElement]:
    """Detect a doorway-sized gap in a wall's inlier coverage at floor
    height (this repo's only real evidence source for an opening: the
    wall plane's own point cloud, per scripts/render_room_dataset.py's
    doorway convention -- an opening starting at the floor).

    Returns None (never a guess) when the wall has no inliers, no
    lateral extent, or no interior gap sized like a door.
    """
    if not wall.inlier_positions:
        return None
    up_u = _normalize(up)
    # Lateral axis: perpendicular to both the wall normal and up.
    lateral = _cross(wall.normal, up_u)
    lateral_len = math.sqrt(_dot(lateral, lateral))
    if lateral_len < 1e-9:
        return None  # wall normal parallel to up: not a real wall
    lateral = (lateral[0] / lateral_len, lateral[1] / lateral_len, lateral[2] / lateral_len)

    lat_min = _dot(wall.bounds_min, lateral)
    lat_max = _dot(wall.bounds_max, lateral)
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    span = lat_max - lat_min
    if span < DOOR_MIN_WIDTH_M:
        return None

    n_buckets = max(1, int(math.ceil(span / SCAN_BUCKET_M)))
    covered = [False] * n_buckets
    for pos in wall.inlier_positions:
        height = _dot(pos, up_u)
        if height < floor_height or height > floor_height + DOOR_SCAN_BAND_M:
            continue
        lat = _dot(pos, lateral)
        idx = int((lat - lat_min) / SCAN_BUCKET_M)
        idx = max(0, min(n_buckets - 1, idx))
        covered[idx] = True

    # Find the longest contiguous empty run, excluding edge buckets.
    lo, hi = EDGE_MARGIN_BUCKETS, n_buckets - EDGE_MARGIN_BUCKETS
    best_start, best_len = -1, 0
    run_start, run_len = -1, 0
    for i in range(max(0, lo), min(n_buckets, hi)):
        if not covered[i]:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0

    if best_len <= 0:
        return None
    width = best_len * SCAN_BUCKET_M
    if not (DOOR_MIN_WIDTH_M <= width <= DOOR_MAX_WIDTH_M):
        return None

    gap_lat_lo = lat_min + best_start * SCAN_BUCKET_M
    gap_lat_hi = gap_lat_lo + width

    # Reconstruct world points from (lateral, height) using the wall's
    # own centroid as reference -- robust regardless of which way
    # `lateral` happens to point relative to bounds_min/bounds_max.
    ref_lat = _dot(wall.centroid, lateral)
    ref_height = _dot(wall.centroid, up_u)

    def _point(lat_value: float, height_value: float) -> Vec3:
        return tuple(
            wall.centroid[k]
            + (lat_value - ref_lat) * lateral[k]
            + (height_value - ref_height) * up_u[k]
            for k in range(3)
        )

    corner_a = _point(gap_lat_lo, floor_height)
    corner_b = _point(gap_lat_hi, floor_height + DOOR_SCAN_BAND_M)
    bmin = tuple(min(corner_a[k], corner_b[k]) for k in range(3))
    bmax = tuple(max(corner_a[k], corner_b[k]) for k in range(3))
    return ArchitecturalElement(
        element_id=f"door-{wall.plane_id}",
        element_type="door",
        source_plane_id=wall.plane_id,
        reason=(
            f"{width:.2f} m interior gap in wall coverage at floor height "
            f"(scanned band {DOOR_SCAN_BAND_M} m)"
        ),
        bounds_min=bmin, bounds_max=bmax,
    )
