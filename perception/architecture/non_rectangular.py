"""Non-rectangular room geometry solver and shape classifier (Plan 13.4 / T-3.5).

Detects, regularises, and verifies non-rectangular interior rooms:
- L-shaped rooms (6-vertex orthogonal layout with 1 reflex 270-degree corner)
- T-shaped and U-shaped rooms (8-vertex orthogonal layouts with 2 reflex corners)
- Chamfered corners (45-degree bevels connecting perpendicular walls)
- Bay windows (canted multi-facet projections extending outward from host walls)
- General polygonal rooms

Evaluates dimensional accuracy against the Turnaround Plan Section 1 tolerances:
- Wall lengths: +-2 cm (0.02 m)
- Room area: +-1.0 %
- Ceiling height: +-2 cm (0.02 m)
- Opening width: +-2 cm (0.02 m)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np


class RoomShapeType(str, Enum):
    RECTANGULAR = "rectangular"
    L_SHAPED = "l_shaped"
    T_SHAPED = "t_shaped"
    U_SHAPED = "u_shaped"
    BAY_WINDOW = "bay_window"
    CHAMFERED = "chamfered"
    POLYGONAL = "polygonal"


# Tolerances from 33_TURNAROUND_PLAN.md Section 1
TOLERANCE_WALL_LENGTH_M = 0.02    # +-2 cm
TOLERANCE_ROOM_AREA_PCT = 1.0     # +-1 %
TOLERANCE_CEILING_HEIGHT_M = 0.02 # +-2 cm
TOLERANCE_OPENING_WIDTH_M = 0.02  # +-2 cm


def _signed_area(poly: Sequence[tuple[float, float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))


def _normalize_angle(rad: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (rad + math.pi) % (2 * math.pi) - math.pi


def _interior_angles_deg(poly: Sequence[tuple[float, float]]) -> list[float]:
    """Compute interior angles in degrees for each vertex of a simple polygon.

    Ensures CCW orientation so left turns are interior angles < 180 and reflex turns > 180.
    """
    n = len(poly)
    if n < 3:
        return []
    is_ccw = _signed_area(poly) > 0
    pts = list(poly) if is_ccw else list(reversed(poly))

    angles = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]

        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        # Cross product (v1 x v2) and dot product
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]

        ang = math.atan2(cross, dot)
        if ang < 0:
            ang += 2 * math.pi
        angles.append(math.degrees(ang))

    if not is_ccw:
        angles.reverse()
    if sum(angles) > n * 180.0:
        angles = [360.0 - a for a in angles]
    return angles


def classify_room_shape(
    polygon: Sequence[tuple[float, float]],
    dominant_angle_deg: float | None = None,
    right_angle_tol_deg: float = 12.0,
    chamfer_angle_tol_deg: float = 12.0,
) -> dict[str, Any]:
    """Classify the architectural shape of a room polygon.

    Recognises:
    - Rectangles (4 vertices, 4 right angles)
    - L-shapes (6 vertices, 5 convex 90 deg corners, 1 reflex 270 deg corner)
    - Chamfers (diagonal corner segments beveling a 90 deg corner)
    - Bay windows (outward canted facets ~45 or ~30/60 deg)
    - T-shapes and U-shapes (8 vertices with 2 reflex corners)
    - General polygons
    """
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    n = len(pts)
    if n < 3:
        return {
            "shape_type": RoomShapeType.POLYGONAL.value,
            "vertex_count": n,
            "area_m2": 0.0,
            "is_orthogonal": False,
            "details": {"reason": "fewer than 3 vertices"},
        }

    area = abs(_signed_area(pts))
    angles = _interior_angles_deg(pts)

    # Count corner types
    right_angles = [abs(a - 90.0) <= right_angle_tol_deg for a in angles]
    reflex_right = [abs(a - 270.0) <= right_angle_tol_deg for a in angles]
    chamfer_corners = [abs(a - 135.0) <= chamfer_angle_tol_deg for a in angles]

    num_right = sum(right_angles)
    num_reflex = sum(reflex_right)
    num_chamfer = sum(chamfer_corners)

    # 1. Check for Rectangular
    if n == 4 and num_right == 4:
        # Compute lengths of adjacent sides
        e0 = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        e1 = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
        return {
            "shape_type": RoomShapeType.RECTANGULAR.value,
            "vertex_count": 4,
            "area_m2": round(area, 4),
            "is_orthogonal": True,
            "details": {
                "length_m": round(max(e0, e1), 4),
                "width_m": round(min(e0, e1), 4),
                "aspect_ratio": round(max(e0, e1) / max(min(e0, e1), 1e-4), 3),
            },
        }

    # 2. Check for L-shaped
    if n == 6 and num_right == 5 and num_reflex == 1:
        reflex_idx = reflex_right.index(True)
        # Compute arm extents
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        return {
            "shape_type": RoomShapeType.L_SHAPED.value,
            "vertex_count": 6,
            "area_m2": round(area, 4),
            "is_orthogonal": True,
            "details": {
                "reflex_corner_index": reflex_idx,
                "reflex_vertex": pts[reflex_idx],
                "bounding_extent_u_m": round(bbox_w, 4),
                "bounding_extent_v_m": round(bbox_h, 4),
                "bounding_box_area_m2": round(bbox_w * bbox_h, 4),
                "area_utilization": round(area / max(bbox_w * bbox_h, 1e-4), 3),
            },
        }

    # 3. Check for Bay Window
    # A bay window consists of 3 (or 5) contiguous edges projecting outward from a main baseline
    bay_candidates = _detect_bay_window_projections(pts, angles)
    if bay_candidates:
        return {
            "shape_type": RoomShapeType.BAY_WINDOW.value,
            "vertex_count": n,
            "area_m2": round(area, 4),
            "is_orthogonal": False,
            "details": {
                "bay_windows": bay_candidates,
                "orthogonal_corners": num_right,
            },
        }

    # 4. Check for Chamfered corners
    # A chamfer is an edge between two ~135 degree interior angles connecting orthogonal runs
    chamfer_segments = []
    for i in range(n):
        ang_a = angles[i]
        ang_b = angles[(i + 1) % n]
        if abs(ang_a - 135.0) <= chamfer_angle_tol_deg and abs(ang_b - 135.0) <= chamfer_angle_tol_deg:
            p_a, p_b = pts[i], pts[(i + 1) % n]
            chamfer_len = math.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1])
            chamfer_segments.append({
                "edge_index": i,
                "start": p_a,
                "end": p_b,
                "length_m": round(chamfer_len, 4),
            })

    if chamfer_segments and (num_right >= 3):
        return {
            "shape_type": RoomShapeType.CHAMFERED.value,
            "vertex_count": n,
            "area_m2": round(area, 4),
            "is_orthogonal": False,
            "details": {
                "chamfer_count": len(chamfer_segments),
                "chamfers": chamfer_segments,
                "orthogonal_corners": num_right,
            },
        }

    # 5. Check for T-shaped or U-shaped (8 vertices, 6 convex 90s, 2 reflex 270s)
    if n == 8 and num_right == 6 and num_reflex == 2:
        reflex_indices = [i for i, r in enumerate(reflex_right) if r]
        # If the two reflex corners are adjacent or face each other
        return {
            "shape_type": RoomShapeType.U_SHAPED.value if (reflex_indices[1] - reflex_indices[0]) in (1, 7) else RoomShapeType.T_SHAPED.value,
            "vertex_count": 8,
            "area_m2": round(area, 4),
            "is_orthogonal": True,
            "details": {
                "reflex_indices": reflex_indices,
                "reflex_vertices": [pts[i] for i in reflex_indices],
            },
        }

    # 6. Fallback: general polygonal room
    return {
        "shape_type": RoomShapeType.POLYGONAL.value,
        "vertex_count": n,
        "area_m2": round(area, 4),
        "is_orthogonal": num_right + num_reflex == n,
        "details": {
            "convex_right_angles": num_right,
            "reflex_right_angles": num_reflex,
            "chamfer_angles": num_chamfer,
        },
    }


def _detect_bay_window_projections(
    pts: list[tuple[float, float]],
    angles: list[float],
) -> list[dict[str, Any]]:
    """Detect bay window canted facet projections from polygon edges."""
    n = len(pts)
    if n < 7:  # A bay window requires at least 4 main walls + 3 bay facets = 7 vertices
        return []
    bays = []
    # Search for a 3-edge canted sequence:
    # Edge 1 (canted flank), Edge 2 (front face), Edge 3 (canted flank)
    for i in range(n):
        i0 = i
        i1 = (i + 1) % n
        i2 = (i + 2) % n
        i3 = (i + 3) % n

        # Check if the flanking edges are angled (e.g. interior angles roughly 120-150 deg or reflex equivalent)
        a1 = angles[i1]
        a2 = angles[i2]
        if (120 <= a1 <= 155 or 205 <= a1 <= 240) and (120 <= a2 <= 155 or 205 <= a2 <= 240):
            p0, p1, p2, p3 = pts[i0], pts[i1], pts[i2], pts[i3]
            span = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
            # Projection depth of front face (p1-p2) from the chord line (p0-p3)
            # Normal to chord p0-p3
            chord_dx, chord_dy = p3[0] - p0[0], p3[1] - p0[1]
            chord_len = math.hypot(chord_dx, chord_dy)
            if chord_len > 0.3:
                nx, ny = -chord_dy / chord_len, chord_dx / chord_len
                depth_1 = abs((p1[0] - p0[0]) * nx + (p1[1] - p0[1]) * ny)
                depth_2 = abs((p2[0] - p0[0]) * nx + (p2[1] - p0[1]) * ny)
                proj_depth = (depth_1 + depth_2) / 2.0
                if proj_depth > 0.15:  # At least 15 cm projection into/out of room
                    bays.append({
                        "indices": [i0, i1, i2, i3],
                        "opening_span_m": round(span, 4),
                        "projection_depth_m": round(proj_depth, 4),
                        "front_width_m": round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 4),
                        "left_flank_length_m": round(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 4),
                        "right_flank_length_m": round(math.hypot(p3[0] - p2[0], p3[1] - p2[1]), 4),
                    })
    bays.sort(key=lambda b: b["projection_depth_m"], reverse=True)
    return bays


def regularise_canted_angles(
    raw_angle_rad: float,
    dominant_angle_rad: float | None,
    snap_tol_rad: float = math.radians(6.0),
) -> tuple[float, bool]:
    """Snap a non-Manhattan edge to standard architectural angles (45/135 deg or 30/60 deg).

    Returns (regularised_angle_rad, is_canted_snapped).
    Does NOT mark as Manhattan-snapped, preserving separation of orthogonal vs canted walls.
    """
    if dominant_angle_rad is None:
        return raw_angle_rad, False

    theta = dominant_angle_rad
    # Check 45-degree (pi / 4) octagonal increments
    diff_45 = _normalize_angle(raw_angle_rad - theta)
    k45 = round(diff_45 / (math.pi / 4))
    target_45 = theta + k45 * (math.pi / 4)
    if abs(_normalize_angle(raw_angle_rad - target_45)) <= snap_tol_rad:
        # Check if it is purely orthogonal (multiple of pi/2)
        if k45 % 2 != 0:
            return target_45, True

    # Check 30-degree (pi / 6) hexagonal increments
    diff_30 = _normalize_angle(raw_angle_rad - theta)
    k30 = round(diff_30 / (math.pi / 6))
    target_30 = theta + k30 * (math.pi / 6)
    if abs(_normalize_angle(raw_angle_rad - target_30)) <= snap_tol_rad:
        if k30 % 3 != 0:  # Not a multiple of pi/2
            return target_30, True

    return raw_angle_rad, False


def evaluate_non_rectangular_tolerances(
    measured: dict[str, Any],
    ground_truth: dict[str, Any],
    tolerance_wall_m: float = TOLERANCE_WALL_LENGTH_M,
    tolerance_area_pct: float = TOLERANCE_ROOM_AREA_PCT,
    tolerance_height_m: float = TOLERANCE_CEILING_HEIGHT_M,
    tolerance_opening_m: float = TOLERANCE_OPENING_WIDTH_M,
) -> dict[str, Any]:
    """Evaluate reconstructed non-rectangular room measurements against ground truth.

    Checks:
    - Wall lengths: error <= tolerance_wall_m (2 cm)
    - Floor area: error % <= tolerance_area_pct (1.0 %)
    - Ceiling heights: error <= tolerance_height_m (2 cm)
    - Opening widths: error <= tolerance_opening_m (2 cm)
    """
    results: dict[str, Any] = {
        "meets_tolerances": True,
        "checks": {},
        "summary": [],
    }

    # 1. Room Area Check
    gt_area = ground_truth.get("floor_area_m2")
    meas_area = measured.get("floor_area_m2")
    if gt_area is not None and meas_area is not None:
        area_diff = abs(meas_area - gt_area)
        area_pct = (area_diff / gt_area) * 100.0
        area_ok = area_pct <= tolerance_area_pct
        results["checks"]["floor_area"] = {
            "measured_m2": round(meas_area, 4),
            "ground_truth_m2": round(gt_area, 4),
            "diff_m2": round(area_diff, 4),
            "error_pct": round(area_pct, 3),
            "tolerance_pct": tolerance_area_pct,
            "pass": area_ok,
        }
        if not area_ok:
            results["meets_tolerances"] = False
            results["summary"].append(f"Floor area error {area_pct:.2f}% exceeds {tolerance_area_pct}%")

    # 2. Wall Lengths Check
    gt_walls = ground_truth.get("wall_lengths_m", [])
    meas_walls = measured.get("wall_lengths_m", [])
    if gt_walls and meas_walls:
        sorted_gt = sorted(gt_walls)
        sorted_meas = sorted(meas_walls)
        wall_diffs = []
        for g, m in zip(sorted_gt, sorted_meas):
            diff = abs(m - g)
            wall_diffs.append(diff)

        max_wall_diff = max(wall_diffs) if wall_diffs else 0.0
        median_wall_diff = float(np.median(wall_diffs)) if wall_diffs else 0.0
        rmse_wall_diff = float(np.sqrt(np.mean(np.array(wall_diffs) ** 2))) if wall_diffs else 0.0
        all_walls_ok = all(d <= tolerance_wall_m for d in wall_diffs)
        results["checks"]["wall_lengths"] = {
            "wall_count_measured": len(meas_walls),
            "wall_count_ground_truth": len(gt_walls),
            "max_diff_m": round(max_wall_diff, 4),
            "median_diff_m": round(median_wall_diff, 4),
            "rmse_m": round(rmse_wall_diff, 4),
            "per_wall_diffs_m": [round(d, 4) for d in wall_diffs],
            "tolerance_m": tolerance_wall_m,
            "pass": all_walls_ok,
        }
        if not all_walls_ok:
            results["meets_tolerances"] = False
            results["summary"].append(f"Max wall length diff {max_wall_diff * 100:.1f} cm exceeds {tolerance_wall_m * 100:.1f} cm")

    # 3. Ceiling Heights Check
    gt_heights = ground_truth.get("ceiling_heights_m", [])
    meas_heights = measured.get("ceiling_heights_m", [])
    if gt_heights and meas_heights:
        height_diffs = [abs(m - g) for g, m in zip(sorted(gt_heights), sorted(meas_heights))]
        max_h_diff = max(height_diffs) if height_diffs else 0.0
        heights_ok = all(d <= tolerance_height_m for d in height_diffs)
        results["checks"]["ceiling_heights"] = {
            "measured_m": [round(h, 4) for h in sorted(meas_heights)],
            "ground_truth_m": [round(h, 4) for h in sorted(gt_heights)],
            "max_diff_m": round(max_h_diff, 4),
            "tolerance_m": tolerance_height_m,
            "pass": heights_ok,
        }
        if not heights_ok:
            results["meets_tolerances"] = False
            results["summary"].append(f"Ceiling height diff {max_h_diff * 100:.1f} cm exceeds {tolerance_height_m * 100:.1f} cm")

    # 4. Opening Widths Check
    gt_openings = ground_truth.get("opening_widths_m", [])
    meas_openings = measured.get("opening_widths_m", [])
    if gt_openings and meas_openings:
        opening_diffs = [abs(m - g) for g, m in zip(sorted(gt_openings), sorted(meas_openings))]
        max_o_diff = max(opening_diffs) if opening_diffs else 0.0
        openings_ok = all(d <= tolerance_opening_m for d in opening_diffs)
        results["checks"]["opening_widths"] = {
            "measured_m": [round(w, 4) for w in sorted(meas_openings)],
            "ground_truth_m": [round(w, 4) for w in sorted(gt_openings)],
            "max_diff_m": round(max_o_diff, 4),
            "tolerance_m": tolerance_opening_m,
            "pass": openings_ok,
        }
        if not openings_ok:
            results["meets_tolerances"] = False
            results["summary"].append(f"Opening width diff {max_o_diff * 100:.1f} cm exceeds {tolerance_opening_m * 100:.1f} cm")

    if results["meets_tolerances"]:
        results["summary"].append("All non-rectangular room measurements satisfy Turnaround Plan Section 1 tolerances.")

    return results
