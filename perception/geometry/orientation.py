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


def _as_unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in v))
    if length == 0.0 or not math.isfinite(length):
        raise OrientationError(f"up vector must be non-degenerate, got {v!r}")
    return (v[0] / length, v[1] / length, v[2] / length)
