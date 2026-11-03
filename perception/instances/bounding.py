"""3D Oriented Bounding Box & 6-DoF Pose Solver (Epic E05-2).

Computes minimum-volume oriented bounding boxes (OMBB) for 3D point clusters,
constrained by supporting surface contact normal (floor or wall) and taxonomy size priors.
Includes exact 3D polygon-clipped intersection-over-union (IoU) evaluation.

Success criteria:
- 3D IoU >= 0.70 for furniture
- 3D IoU >= 0.60 for small architectural fixtures
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from perception.entities import taxonomy


@dataclass
class OrientedBoundingBox:
    """3D Oriented Bounding Box with 6-DoF pose and metric extents."""
    center: Tuple[float, float, float]
    extents: Tuple[float, float, float]  # (dx, dy, dz) full dimensions along local axes
    rotation: List[List[float]]         # 3x3 orthonormal rotation matrix
    yaw_rad: float                      # Rotation angle around vertical axis (radians)
    contact_normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    volume: float = 0.0
    class_id: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.volume <= 0.0:
            self.volume = self.extents[0] * self.extents[1] * self.extents[2]

    def corners(self) -> List[Tuple[float, float, float]]:
        """Compute the 8 world-space vertices of the oriented bounding box."""
        cx, cy, cz = self.center
        dx, dy, dz = self.extents[0] / 2.0, self.extents[1] / 2.0, self.extents[2] / 2.0
        R = self.rotation

        # Local corners
        local_corners = [
            (-dx, -dy, -dz),
            ( dx, -dy, -dz),
            ( dx, -dy,  dz),
            (-dx, -dy,  dz),
            (-dx,  dy, -dz),
            ( dx,  dy, -dz),
            ( dx,  dy,  dz),
            (-dx,  dy,  dz),
        ]

        world_corners = []
        for lx, ly, lz in local_corners:
            wx = cx + R[0][0] * lx + R[0][1] * ly + R[0][2] * lz
            wy = cy + R[1][0] * lx + R[1][1] * ly + R[1][2] * lz
            wz = cz + R[2][0] * lx + R[2][1] * ly + R[2][2] * lz
            world_corners.append((wx, wy, wz))

        return world_corners

    def footprint_2d(self) -> List[Tuple[float, float]]:
        """Project the 4 bottom corners onto the horizontal XZ plane."""
        c = self.corners()
        # Bottom 4 corners: 0, 1, 2, 3
        return [(c[i][0], c[i][2]) for i in [0, 1, 2, 3]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": list(self.center),
            "extents": list(self.extents),
            "rotation": self.rotation,
            "yaw_deg": math.degrees(self.yaw_rad),
            "contact_normal": list(self.contact_normal),
            "volume": self.volume,
            "class_id": self.class_id,
            "confidence": self.confidence,
        }


# ============================================================================
# Vector and Geometry Helpers
# ============================================================================

def _normalize(v: Sequence[float]) -> Tuple[float, float, float]:
    norm = math.hypot(v[0], v[1], v[2])
    if norm == 0.0 or not math.isfinite(norm):
        return (0.0, 1.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _convex_hull_2d(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Monotone chain 2D convex hull algorithm."""
    unique_pts = sorted(list(set(points)))
    if len(unique_pts) <= 2:
        return unique_pts

    def cross_2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in unique_pts:
        while len(lower) >= 2 and cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(unique_pts):
        while len(upper) >= 2 and cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# ============================================================================
# 3D Minimum-Volume Oriented Bounding Box Solver
# ============================================================================

def solve_ombb(
    points: Sequence[Sequence[float]],
    contact_normal: Sequence[float] = (0.0, 1.0, 0.0),
    class_id: Optional[str] = None,
    snap_to_plane: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> OrientedBoundingBox:
    """Solve minimum-volume oriented bounding box constrained by contact normal and taxonomy priors."""
    if not points or len(points) == 0:
        raise ValueError("Cannot solve bounding box for empty point cloud")

    up = _normalize(contact_normal)

    # Establish orthonormal basis (u1, up, u2)
    # Choose arbitrary vector not collinear with up
    ref = (0.0, 0.0, 1.0) if abs(up[2]) < 0.9 else (1.0, 0.0, 0.0)
    u1 = _normalize(_cross(up, ref))
    u2 = _normalize(_cross(up, u1))

    # Project points onto basis:
    # 2D coordinates on contact plane: (s1, s2)
    # 1D height along contact normal: h
    coords_2d: List[Tuple[float, float]] = []
    heights: List[float] = []

    for p in points:
        p_vec = (float(p[0]), float(p[1]), float(p[2]))
        s1 = _dot(p_vec, u1)
        s2 = _dot(p_vec, u2)
        h = _dot(p_vec, up)
        coords_2d.append((s1, s2))
        heights.append(h)

    # 1D height extents along contact normal
    min_h = min(heights)
    max_h = max(heights)
    h_extent = max(max_h - min_h, 0.01)

    # 2D Convex Hull on projected plane
    hull = _convex_hull_2d(coords_2d)
    if len(hull) < 3:
        # Fallback to AABB on projected plane
        s1_vals = [c[0] for c in coords_2d]
        s2_vals = [c[1] for c in coords_2d]
        min_s1, max_s1 = min(s1_vals), max(s1_vals)
        min_s2, max_s2 = min(s2_vals), max(s2_vals)
        best_angle = 0.0
        best_e1 = max(max_s1 - min_s1, 0.01)
        best_e2 = max(max_s2 - min_s2, 0.01)
        best_c1 = (min_s1 + max_s1) / 2.0
        best_c2 = (min_s2 + max_s2) / 2.0
    else:
        # Rotating calipers search over hull edge orientations
        min_area = float("inf")
        best_angle = 0.0
        best_e1 = 0.0
        best_e2 = 0.0
        best_c1 = 0.0
        best_c2 = 0.0

        num_edges = len(hull)
        candidate_angles = []
        for i in range(num_edges):
            p_cur = hull[i]
            p_nxt = hull[(i + 1) % num_edges]
            edge_dx = p_nxt[0] - p_cur[0]
            edge_dy = p_nxt[1] - p_cur[1]
            if edge_dx != 0.0 or edge_dy != 0.0:
                angle = math.atan2(edge_dy, edge_dx)
                candidate_angles.append(angle % (math.pi / 2.0))

        # Also add discrete samples for fine search
        candidate_angles.extend([i * (math.pi / 36.0) for i in range(18)])

        for theta in set(candidate_angles):
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Project hull points onto rotated 2D axes
            t1_vals = [p[0] * cos_t + p[1] * sin_t for p in hull]
            t2_vals = [-p[0] * sin_t + p[1] * cos_t for p in hull]

            t1_min, t1_max = min(t1_vals), max(t1_vals)
            t2_min, t2_max = min(t2_vals), max(t2_vals)

            len1 = t1_max - t1_min
            len2 = t2_max - t2_min
            area = len1 * len2

            if area < min_area:
                min_area = area
                best_angle = theta
                best_e1 = max(len1, 0.01)
                best_e2 = max(len2, 0.01)
                best_t1_mid = (t1_min + t1_max) / 2.0
                best_t2_mid = (t2_min + t2_max) / 2.0
                best_c1 = best_t1_mid * cos_t - best_t2_mid * sin_t
                best_c2 = best_t1_mid * sin_t + best_t2_mid * cos_t

    # Reconstruct 3D local orientation matrix
    cos_a = math.cos(best_angle)
    sin_a = math.sin(best_angle)

    # Local X axis = cos_a * u1 + sin_a * u2
    loc_x = (
        cos_a * u1[0] + sin_a * u2[0],
        cos_a * u1[1] + sin_a * u2[1],
        cos_a * u1[2] + sin_a * u2[2],
    )
    # Local Z axis = -sin_a * u1 + cos_a * u2
    loc_z = (
        -sin_a * u1[0] + cos_a * u2[0],
        -sin_a * u1[1] + cos_a * u2[1],
        -sin_a * u1[2] + cos_a * u2[2],
    )
    # Local Y axis = up
    loc_y = up

    rot_matrix = [
        [loc_x[0], loc_y[0], loc_z[0]],
        [loc_x[1], loc_y[1], loc_z[1]],
        [loc_x[2], loc_y[2], loc_z[2]],
    ]

    # Calculate center in world coordinates
    center_plane = (
        best_c1 * u1[0] + best_c2 * u2[0],
        best_c1 * u1[1] + best_c2 * u2[1],
        best_c1 * u1[2] + best_c2 * u2[2],
    )

    # Supporting surface snapping
    if snap_to_plane is not None:
        plane_orig, plane_norm = snap_to_plane
        p_norm = _normalize(plane_norm)
        # Snap min height flush to supporting plane
        dist_to_plane = _dot(
            (center_plane[0] - plane_orig[0], center_plane[1] - plane_orig[1], center_plane[2] - plane_orig[2]),
            p_norm,
        )
        h_center = dist_to_plane + h_extent / 2.0
    else:
        h_center = (min_h + max_h) / 2.0

    world_center = (
        center_plane[0] + h_center * up[0],
        center_plane[1] + h_center * up[1],
        center_plane[2] + h_center * up[2],
    )

    extents = (best_e1, h_extent, best_e2)

    # Taxonomy prior validation and regularization
    if class_id:
        try:
            eclass = taxonomy.get(class_id)
            min_size, max_size = eclass.size_m
            max_ext = max(extents)
            if not taxonomy.size_plausible(class_id, max_ext):
                # Clamp sparse noisy outliers toward taxonomy plausible limits
                scale_factor = max(min_size / max_ext, min(1.0, (max_size * 1.2) / max_ext))
                extents = (extents[0] * scale_factor, extents[1] * scale_factor, extents[2] * scale_factor)
        except KeyError:
            pass

    return OrientedBoundingBox(
        center=world_center,
        extents=extents,
        rotation=rot_matrix,
        yaw_rad=best_angle,
        contact_normal=up,
        class_id=class_id,
    )


# ============================================================================
# 3D Oriented Bounding Box IoU Calculation
# ============================================================================

def _polygon_area_2d(poly: Sequence[Tuple[float, float]]) -> float:
    """Shoelace formula for 2D polygon area."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def _clip_polygon_2d(subject_poly: List[Tuple[float, float]], clip_poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman polygon clipping algorithm in 2D."""
    output = subject_poly

    for i in range(len(clip_poly)):
        input_list = output
        output = []
        if not input_list:
            break

        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % len(clip_poly)]

        def inside(p):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= 0

        def intersection(p1, p2):
            dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
            dp = (p1[0] - p2[0], p1[1] - p2[1])
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = p1[0] * p2[1] - p1[1] * p2[0]
            denom = dc[0] * dp[1] - dc[1] * dp[0]
            if abs(denom) < 1e-12:
                return p1
            return ((n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom)

        s = input_list[-1]
        for e in input_list:
            if inside(e):
                if not inside(s):
                    output.append(intersection(s, e))
                output.append(e)
            elif inside(s):
                output.append(intersection(s, e))
            s = e

    return output


def compute_3d_iou(box1: OrientedBoundingBox, box2: OrientedBoundingBox) -> float:
    """Compute exact 3D Intersection-over-Union (IoU) between two oriented bounding boxes."""
    # 1. Height overlap along contact vertical axis (Y axis)
    y1_min = box1.center[1] - box1.extents[1] / 2.0
    y1_max = box1.center[1] + box1.extents[1] / 2.0

    y2_min = box2.center[1] - box2.extents[1] / 2.0
    y2_max = box2.center[1] + box2.extents[1] / 2.0

    h_overlap = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    if h_overlap <= 0.0:
        return 0.0

    # 2. 2D Footprint intersection in XZ plane
    poly1 = box1.footprint_2d()
    poly2 = box2.footprint_2d()

    # Ensure counter-clockwise orientation for clipping
    if (poly1[1][0] - poly1[0][0]) * (poly1[2][1] - poly1[1][1]) - (poly1[1][1] - poly1[0][1]) * (poly1[2][0] - poly1[1][0]) < 0:
        poly1 = list(reversed(poly1))
    if (poly2[1][0] - poly2[0][0]) * (poly2[2][1] - poly2[1][1]) - (poly2[1][1] - poly2[0][1]) * (poly2[2][0] - poly2[1][0]) < 0:
        poly2 = list(reversed(poly2))

    clipped = _clip_polygon_2d(poly1, poly2)
    area_int = _polygon_area_2d(clipped)

    vol_int = area_int * h_overlap
    vol1 = box1.volume
    vol2 = box2.volume

    vol_union = vol1 + vol2 - vol_int
    if vol_union <= 0.0:
        return 0.0

    return min(1.0, max(0.0, vol_int / vol_union))
