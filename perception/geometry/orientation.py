"""Orientation classification of detected planes (spec sec 17 GEOMETRIC
REASONING: walls/floors/ceilings -- the first automatic assignment of
WALL/FLOOR/CEILING ontology types in this repo; before this pass the
extended EntityType vocabulary existed but, per
docs/REALITY_ENGINE_AUDIT.md, "no code anywhere in this repo assigns
these types automatically").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from provenance import Uncertainty
from .planes import DetectedPlane, flip_normal_toward

#: A plane is "vertical" (wall) when its normal is within this many
#: radians of the horizontal plane (i.e. |n . up| <= sin(10 deg)).
WALL_NORMAL_TILT_RAD = math.radians(10.0)

#: A plane is "horizontal-ish" (floor/ceiling candidate) when its normal
#: is within this many radians of the up vector.
HORIZONTAL_NORMAL_TILT_RAD = math.radians(15.0)


class OrientationError(ValueError):
    pass


@dataclass(frozen=True)
class OrientedPlane:
    """A DetectedPlane plus its classified role and a camera-facing normal."""

    plane: DetectedPlane
    role: str  # "floor" | "ceiling" | "wall" | "unknown"
    normal: Tuple[float, float, float]  # unit, camera-facing when role != unknown
    d: float
    tilt_from_role_rad: float  # how far the raw normal was from the ideal for this role
    uncertainty: Uncertainty = field(default_factory=Uncertainty)


def classify_plane(
    plane: DetectedPlane,
    camera_positions: Sequence[Tuple[float, float, float]],
    up: Optional[Tuple[float, float, float]] = None,
    observer_positions: Optional[Sequence[Tuple[float, float, float]]] = None,
    plane_centroid: Optional[Tuple[float, float, float]] = None,
) -> OrientedPlane:
    """Classify one plane against camera positions and an up vector."""
    if not camera_positions:
        raise OrientationError(
            "classify_plane requires at least one camera position -- "
            "floor/ceiling disambiguation is defined by which side the cameras were on"
        )

    n_len = math.sqrt(sum(c * c for c in plane.normal))
    if not math.isfinite(n_len) or n_len == 0.0:
        raise OrientationError(f"plane {plane.plane_id} has degenerate normal {plane.normal!r}")
    raw_normal = (plane.normal[0] / n_len, plane.normal[1] / n_len, plane.normal[2] / n_len)
    raw_d = plane.d / n_len  # keep n . p + d = 0 the same zero-set after normalizing

    # Which side of the plane were the cameras on?
    # 1. Observers: cameras that observed the plane
    # 2. Centroid: closest camera to plane's inlier centroid
    # 3. Global: centroid of all cameras
    if observer_positions:
        side_cams = list(observer_positions)
        camera_side = (
            sum(c[0] for c in side_cams) / len(side_cams),
            sum(c[1] for c in side_cams) / len(side_cams),
            sum(c[2] for c in side_cams) / len(side_cams),
        )
    elif plane_centroid is not None and len(camera_positions) > 1:
        camera_side = min(
            camera_positions,
            key=lambda c: (
                (c[0] - plane_centroid[0]) ** 2
                + (c[1] - plane_centroid[1]) ** 2
                + (c[2] - plane_centroid[2]) ** 2
            ),
        )
    else:
        side_cams = list(camera_positions)
        camera_side = (
            sum(c[0] for c in side_cams) / len(side_cams),
            sum(c[1] for c in side_cams) / len(side_cams),
            sum(c[2] for c in side_cams) / len(side_cams),
        )

    if up is None:
        return OrientedPlane(
            plane=plane,
            role="unknown",
            normal=raw_normal,
            d=raw_d,
            tilt_from_role_rad=float("nan"),
            uncertainty=Uncertainty(
                confidence=0.0,
                note="no up vector supplied: horizontal/vertical classification undefined; refusing to guess",
            ),
        )

    up_unit = _as_unit(up)

    # Camera-facing orientation of the raw plane.
    normal, d = flip_normal_toward(raw_normal, raw_d, camera_side)

    dot = normal[0] * up_unit[0] + normal[1] * up_unit[1] + normal[2] * up_unit[2]
    tilt = math.acos(max(-1.0, min(1.0, dot)))

    if tilt <= HORIZONTAL_NORMAL_TILT_RAD:
        # Normal points up => cameras above the plane => floor.
        return OrientedPlane(
            plane=plane, role="floor", normal=normal, d=d,
            tilt_from_role_rad=tilt,
            uncertainty=Uncertainty(
                confidence=max(0.0, 1.0 - tilt / HORIZONTAL_NORMAL_TILT_RAD),
                note=f"horizontal plane, normal up (tilt {math.degrees(tilt):.1f} deg from up)",
            ),
        )
    if abs(math.pi - tilt) <= HORIZONTAL_NORMAL_TILT_RAD:
        # Normal points down => cameras below the visible face => ceiling.
        return OrientedPlane(
            plane=plane, role="ceiling", normal=normal, d=d,
            tilt_from_role_rad=abs(math.pi - tilt),
            uncertainty=Uncertainty(
                confidence=max(0.0, 1.0 - abs(math.pi - tilt) / HORIZONTAL_NORMAL_TILT_RAD),
                note=f"horizontal plane, normal down (tilt {math.degrees(abs(math.pi - tilt)):.1f} deg from down)",
            ),
        )
    if abs(dot) <= math.sin(WALL_NORMAL_TILT_RAD):
        # Near-vertical: normal is horizontal-ish; already camera-facing.
        return OrientedPlane(
            plane=plane, role="wall", normal=normal, d=d,
            tilt_from_role_rad=abs(math.pi / 2.0 - tilt),
            uncertainty=Uncertainty(
                confidence=max(0.0, 1.0 - abs(math.pi / 2.0 - tilt) / WALL_NORMAL_TILT_RAD),
                note=f"vertical plane (tilt {math.degrees(abs(math.pi / 2.0 - tilt)):.1f} deg from vertical)",
            ),
        )

    return OrientedPlane(
        plane=plane, role="unknown", normal=raw_normal, d=raw_d,
        tilt_from_role_rad=float("nan"),
        uncertainty=Uncertainty(
            confidence=0.0,
            note=f"tilt {math.degrees(tilt):.1f} deg from up matches no role within tolerance",
        ),
    )


def observer_positions_by_plane(planes, result) -> dict:
    """plane_id -> positions of the registered cameras that observed the plane's inlier points."""
    pose_of = {c.evidence_id: c.position for c in result.camera_poses}
    seen_by = {p.track_id: p.source_evidence_ids for p in result.points}
    out = {}
    for pl in planes:
        ids = {e for tid in pl.inlier_ids for e in (seen_by.get(tid) or ())}
        pos = [pose_of[e] for e in sorted(ids) if e in pose_of]
        if pos:
            out[pl.plane_id] = pos
    return out


def classify_planes(
    planes: List[DetectedPlane],
    camera_positions: Sequence[Tuple[float, float, float]],
    up: Optional[Tuple[float, float, float]] = None,
    observers: Optional[dict] = None,
    plane_centroids: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> List[OrientedPlane]:
    """Classify every plane; deterministic (input order preserved)."""
    observers = observers or {}
    plane_centroids = plane_centroids or {}
    return [
        classify_plane(
            p, camera_positions, up,
            observer_positions=observers.get(p.plane_id),
            plane_centroid=plane_centroids.get(p.plane_id),
        )
        for p in planes
    ]


#: A floor/ceiling is flat: the 5th..95th percentile spread of its inlier heights, over its lateral extent, stays below this.
#: (Measured: depth-smeared "ceilings" in a room spanned 0.5-0.76 units vertically over ~3 laterally, i.e. ~0.2.)
FLATNESS_MAX = 0.08
#: Fragments of one flat level: heights within this share of the vertical extent of all horizontal planes.
LEVEL_HEIGHT_TOL_SHARE = 0.02


def consolidate_horizontal_roles(
    oriented: Sequence[OrientedPlane],
    up: Tuple[float, float, float],
    camera_positions: Sequence[Tuple[float, float, float]],
    positions_by_id: Dict[str, Tuple[float, float, float]],
    observers: Optional[Dict[str, Sequence[Tuple[float, float, float]]]] = None,
) -> Tuple[List[OrientedPlane], Dict[str, int]]:
    """Per-plane sanity for floor/ceiling roles; ``classify_plane`` only checks which side of a plane the cameras are on.

    That is enough to call every downward-facing slab that dense monocular depth smears across a room a "ceiling"
    (12 of them in a room with one). Two checks, then one clean-up; distinct flat levels (storeys, mezzanines) are never
    removed:
      1. flatness: inlier heights spread (5th..95th percentile) over more than FLATNESS_MAX of the plane's lateral
         extent marks a plane tilted/smeared. Tilt only RANKS: when a flat candidate exists for the role, the tilted
         ones become ``unknown``; when none does, the best-supported tilted one stays (depth tilts real floors too);
      2. physical: a ceiling lies above every camera that OBSERVED it and a floor below them (the cameras are inside the
         room; per plane, so a multi-storey capture stays valid) -> else ``unknown``. Without observer data, all cameras;
      3. fragments of ONE level (same role, heights within LEVEL_HEIGHT_TOL_SHARE of the vertical extent) merge into the
         strongest plane, which takes their inliers.
    Demoted planes are ``unknown`` with the reason recorded (never dropped silently). Returns
    ``(planes, {"demoted": n, "merged": n})``."""
    import dataclasses

    import numpy as np

    u = np.asarray(_as_unit(up), dtype=float)
    cam_h = [float(np.dot(c, u)) for c in camera_positions]
    out = list(oriented)
    stats = {"demoted": 0, "merged": 0}

    def demote(i: int, why: str) -> None:
        out[i] = dataclasses.replace(out[i], role="unknown", uncertainty=Uncertainty(confidence=0.0, note=why))
        stats["demoted"] += 1

    height: Dict[int, float] = {}
    support: Dict[int, int] = {}
    tilted: Dict[int, bool] = {}
    tilt_note: Dict[int, str] = {}
    spread_of: Dict[int, float] = {}
    for i, op in enumerate(oriented):
        if op.role not in ("floor", "ceiling"):
            continue
        pts = np.array([positions_by_id[t] for t in op.plane.inlier_ids if t in positions_by_id], dtype=float)
        if len(pts) < 3:
            continue
        hs = pts @ u
        lateral = pts - np.outer(hs, u)
        extent = float(max(np.ptp(lateral @ axis) for axis in _orthonormal_pair(u))) or 1e-9
        spread = float(np.percentile(hs, 95) - np.percentile(hs, 5))
        tilted[i] = spread / extent > FLATNESS_MAX
        spread_of[i] = spread
        tilt_note[i] = f"height spread {spread:.2f} over {extent:.2f}"
        seen = [float(np.dot(c, u)) for c in (observers or {}).get(op.plane.plane_id, ())] or cam_h
        h = float(np.mean(hs))
        if seen and ((op.role == "ceiling" and h <= max(seen)) or (op.role == "floor" and h >= min(seen))):
            demote(i, f"faces the cameras but is not beyond them (height {h:.2f}, observing cameras "
                      f"{min(seen):.2f}..{max(seen):.2f}): a surface inside the room, not its {op.role}")
            continue
        height[i], support[i] = h, len(op.plane.inlier_ids)

    if not height:
        return out, stats
    # Tilt is a ranking signal, not a veto: monocular depth tilts real floors too (a real floor measured 0.88 high over
    # 4.5 across). With any flat candidate, tilted ones are clutter; with none, the best-supported tilted one IS the surface.
    for role in ("ceiling", "floor"):
        cand = [i for i in height if out[i].role == role]
        flat = [i for i in cand if not tilted[i]]
        best_tilted = [] if flat or not cand else [max(cand, key=lambda i: support[i])]
        for i in cand:
            if tilted[i] and i not in best_tilted:
                demote(i, f"{role} candidate is tilted or depth-smeared ({tilt_note[i]}) and "
                          + ("a flat one exists" if flat else "a better-supported one exists"))
    tol =max(LEVEL_HEIGHT_TOL_SHARE * (max(list(height.values()) + cam_h) - min(list(height.values()) + cam_h)), 1e-9)
    for role in ("ceiling", "floor"):
        # Greedy from the best-supported plane: everything within its noise band is the same surface. The band is the
        # larger of the fixed level tolerance and the plane's own vertical spread -- per-view monocular depth gives each
        # photo a slightly different copy of one floor, so copies closer than the surface's own roughness are not levels.
        idx = sorted((i for i in height if out[i].role == role), key=lambda i: height[i])
        chains: List[List[int]] = []                    # heights that follow each other within the fixed tolerance
        for i in idx:
            if chains and height[i] - height[chains[-1][-1]] <= tol:
                chains[-1].append(i)
            else:
                chains.append([i])
        for c in chains:
            c.sort(key=lambda i: -support[i])           # lead = best-supported member
        pending = sorted(chains, key=lambda c: -support[c[0]])
        levels: List[List[int]] = []
        while pending:
            first = pending.pop(0)
            band = max(tol, spread_of.get(first[0], 0.0))
            near = [c for c in pending if abs(height[c[0]] - height[first[0]]) <= band]
            pending = [c for c in pending if c not in near]
            levels.append(first + [i for c in near for i in c])
        for lv in levels:
            if len(lv) < 2:
                continue
            lead = lv[0]
            union = sorted({t for i in lv for t in out[i].plane.inlier_ids})
            out[lead] = dataclasses.replace(out[lead], plane=dataclasses.replace(out[lead].plane, inlier_ids=union))
            for i in lv:
                if i != lead:
                    out[i] = dataclasses.replace(out[i], role="unknown", uncertainty=Uncertainty(
                        confidence=0.0, note=f"fragment of the {role} at height {height[lead]:.2f}; merged into "
                                             f"{out[lead].plane.plane_id}"))
                    stats["merged"] += 1

    # B09 support-plane separation rule: separate a storey's real floor from table tops, steps and shelves.
    floor_indices = [i for i in range(len(out)) if out[i].role == "floor"]
    wall_planes = [op for op in out if op.role == "wall"]
    if len(floor_indices) > 1 and len(wall_planes) >= 2:
        from perception.geometry.support_planes import split_floor_candidates

        wall_base_heights = []
        for wp in wall_planes:
            w_pts = [positions_by_id[t] for t in wp.plane.inlier_ids if t in positions_by_id]
            if w_pts:
                wall_base_heights.append(float(np.min(np.asarray(w_pts) @ u)))

        cand_heights = []
        cand_areas = []
        cand_wall_bases = []
        for idx in floor_indices:
            h = height.get(idx, float(np.dot(out[idx].normal, u) * -out[idx].d))
            cand_heights.append(h)
            pts = [positions_by_id[t] for t in out[idx].plane.inlier_ids if t in positions_by_id]
            if len(pts) >= 3:
                pts_arr = np.asarray(pts, dtype=float)
                hs_p = pts_arr @ u
                lateral = pts_arr - np.outer(hs_p, u)
                axes = _orthonormal_pair(u)
                ptp0 = float(np.ptp(lateral @ axes[0]))
                ptp1 = float(np.ptp(lateral @ axes[1]))
                area = max(ptp0 * ptp1, 0.01)
            else:
                area = 0.01
            cand_areas.append(area)
            if wall_base_heights:
                # 0.25m tolerance accounts for grid quantization, baseboards, and wall-base separation
                wb_count = sum(1 for wbh in wall_base_heights if abs(wbh - h) <= 0.25)
                cand_wall_bases.append(wb_count)
            else:
                cand_wall_bases.append(0)

        sorted_pairs = sorted(enumerate(cand_heights), key=lambda x: x[1])
        storeys: List[List[int]] = []
        for orig_pos, h in sorted_pairs:
            if not storeys or h - cand_heights[storeys[-1][-1]] > 1.8:
                storeys.append([orig_pos])
            else:
                storeys.append(storeys.pop() + [orig_pos])

        for storey_positions in storeys:
            st_heights = [cand_heights[p] for p in storey_positions]
            st_areas = [cand_areas[p] for p in storey_positions]
            st_bases = [cand_wall_bases[p] for p in storey_positions]
            st_footprint = max(max(st_areas), 1e-6)
            st_roles = split_floor_candidates(
                st_heights,
                st_areas,
                st_footprint,
                st_bases,
            )
            for p, r in zip(storey_positions, st_roles):
                if r == "support_surface":
                    target_idx = floor_indices[p]
                    out[target_idx] = dataclasses.replace(
                        out[target_idx],
                        role="support_surface",
                        uncertainty=Uncertainty(
                            confidence=0.7,
                            note="horizontal support surface (table top, shelf, step), not storey floor slab",
                        ),
                    )
    return out, stats


#: Wall fragments of one wall: horizontal normals within this angle ...
WALL_AZIMUTH_TOL_RAD = math.radians(12.0)
#: ... standing within the larger of this share of the scene's lateral extent and the lead fragment's own thickness ...
WALL_OFFSET_TOL_SHARE = 0.03
#: ... and touching along the wall: the gap between their runs is at most this share of the scene's lateral extent.
WALL_GAP_SHARE = 0.10


def consolidate_vertical_roles(
    oriented: Sequence[OrientedPlane],
    up: Tuple[float, float, float],
    positions_by_id: Dict[str, Tuple[float, float, float]],
) -> Tuple[List[OrientedPlane], Dict[str, int]]:
    """Fragments of ONE wall become one wall. Dense depth cuts a single wall into several patches (one per view); the
    wall-role planes that share a direction, stand on the same line (within the better-supported patch's own thickness)
    and touch along it are merged into the best-supported one, which takes their inliers. Opposite walls, perpendicular
    walls and walls that merely are parallel but apart are never merged. Merged-away planes become ``unknown`` with the
    reason recorded. Returns ``(planes, {"merged": n})``."""
    import dataclasses

    import numpy as np

    u = np.asarray(_as_unit(up), dtype=float)
    out = list(oriented)
    stats = {"merged": 0}
    info: Dict[int, dict] = {}
    for i, op in enumerate(oriented):
        if op.role != "wall":
            continue
        pts = np.array([positions_by_id[t] for t in op.plane.inlier_ids if t in positions_by_id], dtype=float)
        n = np.asarray(op.normal, dtype=float)
        n_h = n - (n @ u) * u
        norm = float(np.linalg.norm(n_h))
        if len(pts) < 3 or norm < 1e-6:
            continue
        info[i] = {"pts": pts, "n": n_h / norm, "support": len(op.plane.inlier_ids)}
    if len(info) < 2:
        return out, stats
    allp = np.vstack([v["pts"] for v in info.values()])
    lateral = allp - np.outer(allp @ u, u)
    scene = float(max(np.ptp(lateral @ axis) for axis in _orthonormal_pair(u))) or 1e-9

    pending = sorted(info, key=lambda i: -info[i]["support"])
    while pending:
        lead = pending.pop(0)
        L = info[lead]
        t = np.cross(u, L["n"])
        lead_off = L["pts"] @ L["n"]
        band = max(WALL_OFFSET_TOL_SHARE * scene, float(np.percentile(lead_off, 95) - np.percentile(lead_off, 5)))
        s_lead = L["pts"] @ t
        lo, hi = float(np.percentile(s_lead, 2)), float(np.percentile(s_lead, 98))
        members = []
        for j in pending:
            J = info[j]
            if math.acos(max(-1.0, min(1.0, float(J["n"] @ L["n"])))) > WALL_AZIMUTH_TOL_RAD:
                continue
            if abs(float(np.median(J["pts"] @ L["n"])) - float(np.median(lead_off))) > band:
                continue
            s_j = J["pts"] @ t
            j0, j1 = float(np.percentile(s_j, 2)), float(np.percentile(s_j, 98))
            if max(j0, lo) - min(j1, hi) > WALL_GAP_SHARE * scene:
                continue
            members.append(j)
            lo, hi = min(lo, j0), max(hi, j1)         # the merged wall grows: later fragments may touch the new run
        if not members:
            continue
        pending = [j for j in pending if j not in members]
        union = sorted({t_ for i in [lead] + members for t_ in out[i].plane.inlier_ids})
        out[lead] = dataclasses.replace(out[lead], plane=dataclasses.replace(out[lead].plane, inlier_ids=union))
        for j in members:
            out[j] = dataclasses.replace(out[j], role="unknown", uncertainty=Uncertainty(
                confidence=0.0, note=f"fragment of the wall {out[lead].plane.plane_id} (same direction, same line, "
                                     f"touching); merged into it"))
            stats["merged"] += 1
    return out, stats


def _orthonormal_pair(u):
    """Two unit vectors spanning the plane orthogonal to the unit vector ``u``."""
    import numpy as np

    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)

def _as_unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in v))
    if length == 0.0 or not math.isfinite(length):
        raise OrientationError(f"up vector must be non-degenerate, got {v!r}")
    return (v[0] / length, v[1] / length, v[2] / length)
