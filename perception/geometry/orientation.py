"""Orientation classification of detected planes (spec sec 17 GEOMETRIC
REASONING: walls/floors/ceilings -- the first automatic assignment of
WALL/FLOOR/CEILING ontology types in this repo; before this pass the
extended EntityType vocabulary existed but, per
docs/REALITY_ENGINE_AUDIT.md, "no code anywhere in this repo assigns
these types automatically").

How it decides (no LLM, no ML, no guessing):
  - FLOOR: horizontal-ish plane whose normal points UP (toward the side
    the cameras were on -- cameras are above floors and below ceilings in
    any real capture).
  - CEILING: horizontal-ish plane whose normal points DOWN (away from the
    camera side -- a ceiling's visible face faces the floor).
  - WALL: near-vertical plane (normal within WALL_NORMAL_TILT_RAD of the
    horizontal plane; flip sign so the normal faces the cameras, which is
    the face people photograph).
  - UNKNOWN: everything else (sloped, degenerate, or unclassifiable).
    Unclassifiable is an answer here, not a failure: a tilted roof plane
    is honestly UNKNOWN rather than mislabeled.

`up` is a caller-supplied parameter defaulting to +Z with the documented
constraint that callers who have a real gravity vector (IMU, geodetic
up, or a known-convention world frame) pass it. When up is None the
classification is explicitly impossible for horizontal-vs-vertical
disambiguation... except via camera positions, which is exactly what the
floor/ceiling camera-side rule uses -- so the None case only downgrades
WALL detection confidence and still refuses to guess, reported as
UNKNOWN, never silently treated as +Z.
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
    """A DetectedPlane plus its classified role and a camera-facing normal.

    normal is flipped toward `camera_side` (the mean camera position when
    available) so "normal points at the observer" is true for the planes
    people actually photograph; d is negated along with it so the plane
    equation normal . p + d = 0 stays equivalent.
    """

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
    plane_centroid: Optional[Tuple[float, float, float]] = None,
) -> OrientedPlane:
    """Classify one plane against camera positions and an up vector.

    camera_positions must be non-empty: without any camera there is no
    camera-side to orient toward and no floor/ceiling disambiguation, and
    guessing either would violate the no-fabrication rule.

    up=None is the honest-unavailable case: horizontal/vertical
    classification is still possible ONLY via the up-down camera rule,
    which itself needs `up` to define "horizontal" -- so with up=None
    every plane is UNKNOWN with an explanatory note. This is a refusal to
    guess, surfaced, not a crash.

    plane_centroid (optional): the mean position of this plane's own real
    inlier points (e.g. from perception.geometry.planes.plane_centroids).
    When supplied, the plane is oriented toward whichever camera is
    spatially closest to it in full 3D, not the mean of every camera in
    the capture -- see the "Orientation reference point" note below for
    why this matters for multi-room/multi-storey captures. Omitting it
    keeps the original global-mean behavior unchanged, so every existing
    caller that does not pass it sees no behavior change.
    """
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

    # Orientation reference point for flip_normal_toward. The mean of
    # EVERY camera in the whole capture is only a valid "which side is
    # the camera on" reference when every camera is genuinely on the
    # same side of this plane -- true for a single-room capture, false
    # once a capture spans multiple rooms or storeys: a plane can have
    # cameras from an entirely different room/floor sitting closer to
    # its GLOBAL average than its own room's cameras are, and the mean
    # can land on the wrong side of the plane entirely (e.g. a storey
    # boundary with half the capture's cameras above it, half below).
    # Either way the plane gets misclassified against its own local
    # evidence (floor read as ceiling or vice versa).
    #
    # When the caller supplies plane_centroid (this plane's own real
    # inlier points' mean position -- not just its infinite normal/d
    # equation), the camera spatially closest to it in full 3D is used
    # instead: the camera that actually stood nearest this specific
    # plane when it was captured, regardless of which room or storey
    # other cameras in the capture belong to. Without a centroid (the
    # caller has not resolved inlier positions, or none were passed) the
    # original global mean is used unchanged, so no existing caller's
    # behavior changes by omission.
    if plane_centroid is not None and len(camera_positions) > 1:
        camera_side = min(
            camera_positions,
            key=lambda c: (
                (c[0] - plane_centroid[0]) ** 2
                + (c[1] - plane_centroid[1]) ** 2
                + (c[2] - plane_centroid[2]) ** 2
            ),
        )
    else:
        camera_side = (
            sum(c[0] for c in camera_positions) / len(camera_positions),
            sum(c[1] for c in camera_positions) / len(camera_positions),
            sum(c[2] for c in camera_positions) / len(camera_positions),
        )

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
        # Flip to camera-facing (down-facing normal already faces the
        # cameras in a normal room capture, so normal/d stay as flipped).
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


def classify_planes(
    planes: List[DetectedPlane],
    camera_positions: Sequence[Tuple[float, float, float]],
    up: Optional[Tuple[float, float, float]] = None,
    plane_centroids: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> List[OrientedPlane]:
    """Classify every plane; deterministic (input order preserved).

    plane_centroids (optional): plane_id -> mean inlier position, e.g.
    from perception.geometry.planes.plane_centroids(result, planes). See
    classify_plane's docstring for why this matters once a capture spans
    more than one room or storey.
    """
    return [
        classify_plane(
            p, camera_positions, up,
            plane_centroid=(plane_centroids or {}).get(p.plane_id),
        )
        for p in planes
    ]


def _as_unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in v))
    if length == 0.0 or not math.isfinite(length):
        raise OrientationError(f"up vector must be non-degenerate, got {v!r}")
    return (v[0] / length, v[1] / length, v[2] / length)
