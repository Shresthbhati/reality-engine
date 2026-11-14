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
    (12 of them in a room with one). Two checks, then one clean-up; distinct levels (storeys, mezzanines) are never
    removed, because each check is a property of ONE plane, not a vote between levels:
      1. flatness: a real floor/ceiling is flat. Inlier heights spread (5th..95th percentile) over more than
         FLATNESS_MAX of the plane's lateral extent -> tilted or smeared -> ``unknown``;
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
        if spread / extent > FLATNESS_MAX:
            demote(i, f"horizontal-facing but not flat (height spread {spread:.2f} over {extent:.2f}): "
                      "tilted or depth-smeared surface, not a floor or ceiling")
            continue
        seen = [float(np.dot(c, u)) for c in (observers or {}).get(op.plane.plane_id, ())] or cam_h
        h = float(np.mean(hs))
        if seen and ((op.role == "ceiling" and h <= max(seen)) or (op.role == "floor" and h >= min(seen))):
            demote(i, f"faces the cameras but is not beyond them (height {h:.2f}, observing cameras "
                      f"{min(seen):.2f}..{max(seen):.2f}): a surface inside the room, not its {op.role}")
            continue
        height[i], support[i] = h, len(op.plane.inlier_ids)

    if not height:
        return out, stats
    tol = max(LEVEL_HEIGHT_TOL_SHARE * (max(list(height.values()) + cam_h) - min(list(height.values()) + cam_h)), 1e-9)
    for role in ("ceiling", "floor"):
        idx = sorted((i for i in height if out[i].role == role), key=lambda i: height[i])
        levels: List[List[int]] = []
        for i in idx:
            if levels and height[i] - height[levels[-1][-1]] <= tol:
                levels[-1].append(i)
            else:
                levels.append([i])
        for lv in levels:
            if len(lv) < 2:
                continue
            lead = max(lv, key=lambda i: support[i])
            union = sorted({t for i in lv for t in out[i].plane.inlier_ids})
            out[lead] = dataclasses.replace(out[lead], plane=dataclasses.replace(out[lead].plane, inlier_ids=union))
            for i in lv:
                if i != lead:
                    out[i] = dataclasses.replace(out[i], role="unknown", uncertainty=Uncertainty(
                        confidence=0.0, note=f"fragment of the {role} at height {height[lead]:.2f}; merged into "
                                             f"{out[lead].plane.plane_id}"))
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
