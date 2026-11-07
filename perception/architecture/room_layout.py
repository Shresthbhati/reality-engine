"""Room layout solver: a regularised, closed wall loop from the measured footprint (plan task S2) plus the pure
geometry the solid-wall promotion needs (slab meshes, view coverage).

Input is what ``room_semantics.refine_interior_semantics`` measured: the footprint polygon in plan coordinates (a, b),
the per-edge wall heights, and (optionally) the reconstructed wall points. Output is one ordered wall list.

Rules (Decision 023 -- nothing is invented, everything estimated is labelled):

* Edges within ``SNAP_TOL_DEG`` of the dominant Manhattan direction are snapped to it; edges that disagree are KEPT
  non-Manhattan (``snapped=False``), never forced.
* Adjacent collinear edges are merged -- unless their measured heights differ (a step in the wall top is a fact).
* A wall is OBSERVED only when reconstructed points actually cover it (``observed_fraction``); a run of the footprint
  that no point supports is a segment inferred to CLOSE the loop: ``provenance="INFERRED"``, low confidence, and the
  unsupported spans are listed (``unobserved_spans_m``) so an absent surface is never read as an opening.
* Thickness: no pipeline stage sees both faces of a wall, so it is the labelled default ``DEFAULT_THICKNESS_M``
  (``thickness_basis="default_unmeasured"``). The footprint edge is the face the cameras saw (the inner face); the slab
  extends outward from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.architecture.layout_evidence import LAYOUT_BASIS_CONFIDENCE, layout_support
from perception.architecture.non_rectangular import classify_room_shape, regularise_canted_angles

#: edges within this angle of a dominant direction are snapped to it
SNAP_TOL_DEG = 8.0
#: adjacent parallel edges closer than this (perpendicular offset) are one wall
MERGE_OFFSET_M = 0.10
#: a regularised corner may move at most this far from the measured vertex, otherwise the measured vertex is kept
MAX_CORNER_SHIFT_M = 0.75
#: labelled default (no pipeline stage observes both faces of a wall); not a measurement
DEFAULT_THICKNESS_M = 0.10
#: wall-point support is evaluated in bins of this length along the wall
SUPPORT_BIN_M = 0.25
SUPPORT_MIN_POINTS = 3
SUPPORT_BAND_M = 0.15
#: a wall with less point support than this is an inferred closing segment
OBSERVED_MIN_FRACTION = 0.5
#: an unsupported run shorter than this is noise, not a reportable span
MIN_SPAN_M = 0.5


@dataclass(frozen=True)
class LayoutWall:
    index: int
    a: Tuple[float, float]
    b: Tuple[float, float]
    height_m: Optional[float]
    height_basis: str                      # "measured" | "none"
    thickness_m: float
    thickness_basis: str                   # "default_unmeasured"
    provenance: str                        # "OBSERVED" | "INFERRED"
    confidence: float
    observed_fraction: Optional[float]     # None = no wall points were supplied, nothing is claimed
    unobserved_spans_m: Tuple[Tuple[float, float], ...]
    snapped: bool
    canted: bool = False
    #: Which observation made this wall OBSERVED: "points" (point cloud), "layout_image" (a projected single-image
    #: floor boundary, plan F4) or "points+layout_image"; "none" when it is an inferred closing segment.
    support_basis: str = "none"
    #: Share of the wall covered by projected layout boundaries (None = no layout evidence was supplied).
    layout_support_fraction: Optional[float] = None

    @property
    def length_m(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


@dataclass
class RoomLayout:
    walls: List[LayoutWall]
    polygon: List[Tuple[float, float]]
    closed: bool
    dominant_angle_deg: Optional[float]
    manhattan_fraction: float              # share of the outline length snapped to the dominant directions
    clockwise: bool = False
    shape_type: Optional[str] = None
    shape_details: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    def inward(self, wall: LayoutWall) -> Tuple[float, float]:
        """Unit plan normal pointing into the room."""
        dx, dy = wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]
        n = math.hypot(dx, dy) or 1.0
        left = (-dy / n, dx / n)
        return (left[0], left[1]) if not self.clockwise else (-left[0], -left[1])


def _signed_area(poly: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def dominant_angle(poly: Sequence[Sequence[float]]) -> Tuple[Optional[float], float]:
    """(angle in radians in [-pi/4, pi/4), strength 0..1) of the length-weighted dominant orthogonal direction."""
    z, tot = 0j, 0.0
    for i in range(len(poly)):
        dx, dy = poly[(i + 1) % len(poly)][0] - poly[i][0], poly[(i + 1) % len(poly)][1] - poly[i][1]
        length = math.hypot(dx, dy)
        z += length * complex(math.cos(4 * math.atan2(dy, dx)), math.sin(4 * math.atan2(dy, dx)))
        tot += length
    if tot < 1e-9 or abs(z) < 1e-9:
        return None, 0.0
    ang = math.atan2(z.imag, z.real) / 4.0
    if abs(math.degrees(ang)) <= 1.0:
        ang = 0.0
    return ang, abs(z) / tot


def _intersect(p1, d1, p2, d2) -> Optional[Tuple[float, float]]:
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-3:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _support(a, b, pts_ab: Optional[np.ndarray]):
    """(observed_fraction, unobserved spans in metres along a->b) from wall points; (None, ()) when none supplied."""
    if pts_ab is None:
        return None, ()
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return 0.0, ()
    u = d / length
    rel = pts_ab - a
    s = rel @ u
    off = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    s = s[(off < SUPPORT_BAND_M) & (s > -0.05) & (s < length + 0.05)]
    nb = max(1, int(math.ceil(length / SUPPORT_BIN_M)))
    cnt = np.bincount(np.clip((s / SUPPORT_BIN_M).astype(int), 0, nb - 1), minlength=nb)
    occ = cnt >= SUPPORT_MIN_POINTS
    spans, start = [], None
    for i in range(nb + 1):
        empty = i < nb and not occ[i]
        if empty and start is None:
            start = i
        elif not empty and start is not None:
            s0, s1 = start * SUPPORT_BIN_M, min(length, i * SUPPORT_BIN_M)
            if s1 - s0 >= MIN_SPAN_M:
                spans.append((round(s0, 3), round(s1, 3)))
            start = None
    return round(float(occ.mean()), 4), tuple(spans)


def solve_room_layout(
    polygon_ab: Sequence[Sequence[float]],
    edge_heights_m: Optional[Sequence[Optional[float]]] = None,
    wall_points_ab: Optional[np.ndarray] = None,
    *,
    snap_tol_deg: float = SNAP_TOL_DEG,
    merge_offset_m: float = MERGE_OFFSET_M,
    thickness_m: float = DEFAULT_THICKNESS_M,
    layout_edges_ab: Optional[Sequence[Sequence[Sequence[float]]]] = None,
) -> RoomLayout:
    """Regularise ``polygon_ab`` into a closed ordered wall list. See the module docstring for the rules."""
    poly = [(float(p[0]), float(p[1])) for p in polygon_ab]
    n = len(poly)
    warnings: List[str] = []
    if n < 3:
        return RoomLayout([], poly, False, None, 0.0, warnings=["footprint has fewer than 3 vertices"])
    heights = list(edge_heights_m) if edge_heights_m is not None else [None] * n
    if len(heights) != n:
        warnings.append(f"edge heights ({len(heights)}) do not match footprint edges ({n}): heights ignored")
        heights = [None] * n
    clockwise = _signed_area(poly) < 0
    theta, strength = dominant_angle(poly)

    # ---- one record per measured edge: line (point + unit direction), start vertex, snapped flag ----
    edges = []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        if length < 1e-6:
            continue
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        snapped = False
        canted = False
        if theta is not None:
            k = round((ang - theta) / (math.pi / 2))
            if abs(math.degrees(ang - theta - k * math.pi / 2)) <= snap_tol_deg:
                ang, snapped = theta + k * math.pi / 2, True
            else:
                ang, canted = regularise_canted_angles(ang, theta)
        u_dir = np.array([math.cos(ang), math.sin(ang)], float)
        n_dir = np.array([-u_dir[1], u_dir[0]], float)
        mid_pt = np.array([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], float)
        if wall_points_ab is not None and len(wall_points_ab) > 0 and (snapped or canted):
            rel = wall_points_ab - mid_pt
            s = rel @ u_dir
            off = rel @ n_dir
            inliers = (np.abs(off) < SUPPORT_BAND_M) & (s > -0.05) & (s < length + 0.05)
            if np.sum(inliers) >= SUPPORT_MIN_POINTS * 2:
                delta = float(np.median(off[inliers]))
                if abs(delta) <= merge_offset_m:
                    mid_pt = mid_pt + delta * n_dir
        edges.append({"mid": (float(mid_pt[0]), float(mid_pt[1])), "dir": (float(u_dir[0]), float(u_dir[1])),
                      "len": length, "start": p, "snapped": snapped, "canted": canted, "h": heights[i]})
    if len(edges) < 3:
        return RoomLayout([], poly, False, theta and math.degrees(theta), 0.0, warnings=["fewer than 3 usable edges"])

    # ---- merge adjacent collinear edges (same line, compatible heights) ----
    def mergeable(e, f) -> bool:
        cross = e["dir"][0] * f["dir"][1] - e["dir"][1] * f["dir"][0]
        if abs(cross) > math.sin(math.radians(2.0)) or e["dir"][0] * f["dir"][0] + e["dir"][1] * f["dir"][1] <= 0:
            return False
        off = abs((f["mid"][0] - e["mid"][0]) * e["dir"][1] - (f["mid"][1] - e["mid"][1]) * e["dir"][0])
        if off > merge_offset_m:
            return False
        return e["h"] is None or f["h"] is None or abs(e["h"] - f["h"]) <= 0.05

    changed = True
    while changed and len(edges) > 3:
        changed = False
        for i in range(len(edges)):
            e, f = edges[i], edges[(i + 1) % len(edges)]
            if mergeable(e, f):
                w = e["len"] + f["len"]
                mid = tuple((e["mid"][k] * e["len"] + f["mid"][k] * f["len"]) / w for k in (0, 1))
                merged = {"mid": mid, "dir": e["dir"], "len": w, "start": e["start"],
                          "snapped": e["snapped"] and f["snapped"],
                          "canted": e.get("canted", False) and f.get("canted", False),
                          "h": e["h"] if e["h"] is not None else f["h"]}
                j = (i + 1) % len(edges)
                if j == 0:                       # merged across the seam
                    edges = [merged] + edges[1:i]
                else:
                    edges[i:j + 1] = [merged]
                changed = True
                break

    # ---- corners = intersections of consecutive lines; keep the measured vertex when ill-conditioned or far ----
    m = len(edges)
    verts = []
    for i in range(m):
        prev, cur = edges[i - 1], edges[i]
        x = _intersect(prev["mid"], prev["dir"], cur["mid"], cur["dir"])
        v0 = cur["start"]
        if x is None or math.hypot(x[0] - v0[0], x[1] - v0[1]) > MAX_CORNER_SHIFT_M:
            x = v0
            warnings.append(f"corner {i}: kept the measured vertex (regularised corner ill-conditioned or too far)")
        verts.append(x)

    walls: List[LayoutWall] = []
    snapped_len, tot_len = 0.0, 0.0
    for i in range(m):
        a, b = verts[i], verts[(i + 1) % m]
        e = edges[i]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1e-6:
            continue
        frac, spans = _support(a, b, wall_points_ab)
        # Plan F4: a projected single-image floor boundary is a weaker, independent observation of the same wall.
        # It only ever RAISES an unsupported wall to OBSERVED (with a lower confidence cap); points remain the authority.
        lfrac = layout_support(a, b, layout_edges_ab) if layout_edges_ab else None
        if frac is None:
            if lfrac is not None and lfrac >= OBSERVED_MIN_FRACTION:
                prov, conf = "OBSERVED", round(LAYOUT_BASIS_CONFIDENCE - 0.1 + 0.1 * lfrac, 3)
                basis = "layout_image"
            else:
                prov, conf, basis = "INFERRED", 0.3, "none"
        elif frac >= OBSERVED_MIN_FRACTION:
            prov, conf, basis = "OBSERVED", round(0.5 + 0.45 * frac, 3), "points"
        else:
            prov, conf, basis = "INFERRED", round(0.15 + 0.3 * frac, 3), "points"
            if lfrac is not None and lfrac >= OBSERVED_MIN_FRACTION:
                prov, conf, basis = "OBSERVED", round(min(0.45, conf + 0.15 * lfrac), 3), "points+layout_image"
        h = e["h"]
        walls.append(LayoutWall(index=len(walls), a=(float(a[0]), float(a[1])), b=(float(b[0]), float(b[1])),
                                height_m=None if h is None else float(h), height_basis="measured" if h is not None else "none",
                                thickness_m=thickness_m, thickness_basis="default_unmeasured", provenance=prov,
                                confidence=conf, observed_fraction=frac, unobserved_spans_m=spans, snapped=e["snapped"],
                                canted=e.get("canted", False),
                                support_basis=basis, layout_support_fraction=lfrac))
        tot_len += length
        snapped_len += length if e["snapped"] else 0.0
    closed = len(walls) >= 3 and all(
        math.hypot(walls[i].b[0] - walls[(i + 1) % len(walls)].a[0], walls[i].b[1] - walls[(i + 1) % len(walls)].a[1]) < 1e-6
        for i in range(len(walls)))
    shape_meta = classify_room_shape([w.a for w in walls], dominant_angle_deg=None if theta is None else round(math.degrees(theta), 3))
    return RoomLayout(walls=walls, polygon=[w.a for w in walls], closed=closed,
                      dominant_angle_deg=None if theta is None else round(math.degrees(theta), 3),
                      manhattan_fraction=round(snapped_len / tot_len, 4) if tot_len else 0.0, clockwise=clockwise,
                      shape_type=shape_meta.get("shape_type"), shape_details=shape_meta.get("details"),
                      warnings=warnings)


# ---------------------------------------------------------------------------------------------------------------
# slab geometry (pure; world frame)
# ---------------------------------------------------------------------------------------------------------------

def wall_pieces(length: float, height: float, openings: Sequence[dict]) -> List[Tuple[float, float, float, float]]:
    """Axis-aligned (s0, s1, v0, v1) boxes that tile a wall rectangle minus its rectangular openings.

    ``openings``: dicts with ``offset_m``, ``width_m``, ``sill_m``, ``height_m`` (in-wall rectangle)."""
    holes = []
    for o in openings:
        s0, s1 = max(0.0, o["offset_m"]), min(length, o["offset_m"] + o["width_m"])
        v0, v1 = max(0.0, o["sill_m"]), min(height, o["sill_m"] + o["height_m"])
        if s1 - s0 > 1e-4 and v1 - v0 > 1e-4:
            holes.append((s0, s1, v0, v1))
    cuts = sorted({0.0, length, *[c for h in holes for c in h[:2]]})
    raw_pieces = []
    for s0, s1 in zip(cuts, cuts[1:]):
        if s1 - s0 < 1e-4:
            continue
        mid = (s0 + s1) / 2
        spans = sorted((h[2], h[3]) for h in holes if h[0] <= mid <= h[1])
        v = 0.0
        for h0, h1 in spans:
            if h0 > v + 1e-4:
                raw_pieces.append((s0, s1, v, h0))
            v = max(v, h1)
        if height > v + 1e-4:
            raw_pieces.append((s0, s1, v, height))

    if not raw_pieces:
        return []

    from collections import defaultdict
    by_span = defaultdict(list)
    for s0, s1, v0, v1 in raw_pieces:
        by_span[(round(v0, 6), round(v1, 6))].append((s0, s1, v0, v1))
    merged = []
    for (v0, v1), slist in by_span.items():
        slist.sort(key=lambda x: x[0])
        cur_s0, cur_s1 = slist[0][0], slist[0][1]
        for s0, s1, _, _ in slist[1:]:
            if abs(s0 - cur_s1) < 1e-4:
                cur_s1 = max(cur_s1, s1)
            else:
                merged.append((cur_s0, cur_s1, v0, v1))
                cur_s0, cur_s1 = s0, s1
        merged.append((cur_s0, cur_s1, v0, v1))
    return sorted(merged, key=lambda x: (x[0], x[2]))


_BOX_QUADS = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def box_mesh(corners: Sequence[Sequence[float]]):
    """(vertices, outward-wound triangles) for a convex hexahedron given as bottom quad 0-3 then top quad 4-7."""
    c = np.asarray(corners, float)
    centre = c.mean(axis=0)
    tris = []
    for q in _BOX_QUADS:
        for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
            nrm = np.cross(c[tri[1]] - c[tri[0]], c[tri[2]] - c[tri[0]])
            if float(nrm @ (c[tri[0]] - centre)) < 0:
                tri = (tri[0], tri[2], tri[1])
            tris.append(tri)
    return [tuple(map(float, p)) for p in c], tris


def _merge(parts):
    verts, faces = [], []
    for v, f in parts:
        base = len(verts)
        verts += v
        faces += [(a + base, b + base, c + base) for a, b, c in f]
    return verts, faces


def wall_slab_mesh(origin_xyz, dir_xyz, out_xyz, up_xyz, length, height, thickness, openings: Sequence[dict]):
    """Triangle mesh of an extruded wall slab with its openings carved through (world frame, metres).

    ``origin_xyz`` is the wall start at floor level on the INNER face; the slab extends ``thickness`` along ``out_xyz``."""
    o, d, n, u = (np.asarray(x, float) for x in (origin_xyz, dir_xyz, out_xyz, up_xyz))
    parts = []
    for s0, s1, v0, v1 in wall_pieces(length, height, openings):
        parts.append(box_mesh([o + s * d + t * n + v * u for (s, t, v) in
                               ((s0, 0, v0), (s1, 0, v0), (s1, thickness, v0), (s0, thickness, v0),
                                (s0, 0, v1), (s1, 0, v1), (s1, thickness, v1), (s0, thickness, v1))]))
    return _merge(parts)


def ear_clip(poly: Sequence[Sequence[float]]) -> List[Tuple[int, int, int]]:
    """Counter-clockwise triangles (vertex index triples) of a simple polygon; [] when it is not simple."""
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    if _signed_area(poly) < 0:
        idx.reverse()

    def orient(o, a, b):
        return (poly[a][0] - poly[o][0]) * (poly[b][1] - poly[o][1]) - (poly[a][1] - poly[o][1]) * (poly[b][0] - poly[o][0])

    out: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < n * n:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if orient(a, b, c) <= 1e-12:
                continue
            if any(p not in (a, b, c) and orient(a, b, p) >= -1e-12 and orient(b, c, p) >= -1e-12 and orient(c, a, p) >= -1e-12
                   for p in idx):
                continue
            out.append((a, b, c))
            idx.pop(k)
            break
        else:
            return []
    if len(idx) == 3:
        out.append((idx[0], idx[1], idx[2]))
    return out


def polygon_slab_mesh(poly_xyz: Sequence[Sequence[float]], up_xyz, thickness: float, direction: int, plan_ab: Sequence[Sequence[float]]):
    """Prism over a planar polygon: top surface = ``poly_xyz``, extruded ``thickness`` along ``direction`` * up
    (-1 = down, a floor; +1 = up, a ceiling). ``plan_ab`` is the same polygon in plan coordinates (for triangulation)."""
    tris = ear_clip(plan_ab)
    if not tris:
        return [], []
    u = np.asarray(up_xyz, float)
    top = [np.asarray(p, float) for p in poly_xyz]
    bot = [p + direction * thickness * u for p in top]
    ccw = _signed_area(plan_ab) > 0
    verts = [tuple(map(float, p)) for p in top + bot]
    n = len(top)
    faces = []
    for a, b, c in tris:      # ear_clip is CCW in plan and (e1, e2, up) is right-handed: (a, b, c) faces +up
        room_side_up = direction < 0                         # a floor's room-facing surface faces up, a ceiling's faces down
        faces.append((a, b, c) if room_side_up else (a, c, b))
        faces.append((n + a, n + c, n + b) if room_side_up else (n + a, n + b, n + c))
    for i in range(n):
        j = (i + 1) % n
        q = (i, j, n + j, n + i)
        faces += [(q[0], q[1], q[2]), (q[0], q[2], q[3])] if ccw == (direction > 0) else [(q[0], q[2], q[1]), (q[0], q[3], q[2])]
    return verts, faces


# ---------------------------------------------------------------------------------------------------------------
# camera view coverage: which part of a wall could a camera have seen at all
# ---------------------------------------------------------------------------------------------------------------

#: ASSUMED -- the pipeline carries no intrinsics. Deliberately narrower than a phone camera so that a region is only
#: called "viewed" when it was clearly in front of a lens; occlusion is ignored (labelled in the output).
ASSUMED_HALF_FOV_DEG = 30.0
VIEW_MAX_RANGE_M = 8.0
VIEW_CELL_M = 0.25
VIEW_MODEL = {"half_fov_deg": ASSUMED_HALF_FOV_DEG, "max_range_m": VIEW_MAX_RANGE_M, "occlusion": "ignored",
              "basis": "assumed (no intrinsics in the reconstruction)"}


def quat_forward(q) -> np.ndarray:
    """World direction of the camera +Z axis for a (w, x, y, z) camera-to-world quaternion (ReconstructedCameraPose)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def wall_view_grid(origin_xyz, dir_xyz, in_xyz, up_xyz, length, height, camera_poses) -> np.ndarray:
    """bool [n_v, n_s]: True where some camera, on the room side of the wall and within range, had the cell inside its
    assumed view cone. Empty array when there are no cameras."""
    ns, nv = max(1, int(math.ceil(length / VIEW_CELL_M))), max(1, int(math.ceil(height / VIEW_CELL_M)))
    grid = np.zeros((nv, ns), bool)
    if not camera_poses:
        return grid
    o, d, i_, u = (np.asarray(x, float) for x in (origin_xyz, dir_xyz, in_xyz, up_xyz))
    s = (np.arange(ns) + 0.5) * VIEW_CELL_M
    v = (np.arange(nv) + 0.5) * VIEW_CELL_M
    cells = o + s[None, :, None] * d + v[:, None, None] * u            # [nv, ns, 3]
    cos_min = math.cos(math.radians(ASSUMED_HALF_FOV_DEG))
    for pose in camera_poses:
        p = np.asarray(pose.position, float)
        if float((p - o) @ i_) < 0.2:
            continue                                                    # behind / inside the wall plane
        rel = cells - p
        dist = np.linalg.norm(rel, axis=2)
        cosang = (rel @ quat_forward(pose.rotation)) / np.maximum(dist, 1e-9)
        grid |= (dist <= VIEW_MAX_RANGE_M) & (cosang >= cos_min)
    return grid
