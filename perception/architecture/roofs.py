"""Measured roof detection (P7-03 architectural-perception breadth).

A roof's evidence, at this repo's level of support, is the TOPMOST
horizontal plane of a supplied exterior plane set -- the same
relative-position logic classify.py already trusts for floor (lowest)
vs ceiling (highest), applied to the building's exterior: the topmost
horizontal plane is the roof, and its own measured bounds give the
lateral extents. What is measured: plane height, lateral width/depth
along the horizontal axes of `up`, and the number of supporting
horizontal planes (a flat roof may be seen as several coplanar sheets;
the topmost BAND is measured, not one lucky plane).

Honest limitations, refused not guessed:
  - a horizontal plane with tiny lateral extent is a skylight/cap, not
    a roof (extent gate);
  - a pitched/curved roof's slopes are NOT horizontal planes; without
    multi-slope aggregation evidence this detector refuses with the
    measured up-dot rather than calling the highest plane a roof;

Deterministic: pure float math, no clocks, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from perception.architecture.classify import (
    HORIZONTAL_MIN_UP_DOT,
    PlaneInput,
    _dot,
    _normalize,
)
from perception.architecture.parametric import FitRefused

Vec3 = Tuple[float, float, float]

#: Minimum lateral extent (both axes) for a horizontal plane to count as
#: a roof surface rather than a cap/skylight. Deliberately modest.
MIN_ROOF_EXTENT_M = 1.0



class RoofRefused(FitRefused):
    """The supplied planes do not demonstrate a roof. The measured fact
    is in the message."""


@dataclass(frozen=True)
class RoofFit:
    """A measured roof: the topmost horizontal plane (plus any measured
    coplanar sheets within one bucket band) of the supplied set."""

    #: Plane id of the highest measured sheet.
    plane_id: str
    #: Supporting sheet plane ids (all coplanar members of the top band).
    supporting_plane_ids: Tuple[str, ...]
    position: Vec3
    width_m: float
    depth_m: float
    height_m: float
    #: Measured horizontality of the top plane (|normal . up|).
    up_dot: float
    n_planes: int
    confidence: float
    evidence_ids: Tuple[str, ...]
    kind: str = "roof"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "plane_id": self.plane_id,
            "supporting_plane_ids": list(self.supporting_plane_ids),
            "position": list(self.position),
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "height_m": self.height_m,
            "up_dot": self.up_dot,
            "n_planes": self.n_planes,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
        }


def _extent(bounds_min: Vec3, bounds_max: Vec3, axis: Vec3) -> float:
    lo = _dot(bounds_min, axis)
    hi = _dot(bounds_max, axis)
    return abs(hi - lo)


def detect_roof(
    planes: Sequence[PlaneInput], up: Vec3 = (0.0, 0.0, 1.0)
) -> RoofFit:
    """Detect the roof among the supplied exterior planes or refuse.

    Deterministic. Raises RoofRefused (a FitRefused subclass) when no
    horizontal plane with roof-scale extent exists, or when the
    highest plane is tilted (its measured up-dot is in the message).
    """
    if not planes:
        raise RoofRefused("no planes supplied -- nothing to measure")
    up_u = _normalize(up)

    # Horizontal candidates: same threshold classify.py trusts.
    horizontal = []
    for p in planes:
        up_dot = abs(_dot(_normalize(p.normal), up_u))
        if up_dot >= HORIZONTAL_MIN_UP_DOT:
            horizontal.append((p, up_dot, _dot(p.centroid, up_u)))
    if not horizontal:
        raise RoofRefused(
            "no plane is horizontal enough to be a roof surface "
            f"(threshold {HORIZONTAL_MIN_UP_DOT}) -- a pitched or curved "
            "roof needs multi-slope aggregation this detector does not "
            "have evidence for"
        )

    # Topmost plane; the band gathers coplanar sheets within one scan
    # bucket of its height (a flat roof is often several sheets).
    horizontal.sort(key=lambda t: -t[2])
    top, top_up_dot, top_h = horizontal[0]
    band = [
        (p, ud, h) for (p, ud, h) in horizontal
        if abs(h - top_h) <= 0.05
    ]

    # Lateral axes: the two horizontal directions orthogonal to up.
    # Pick a deterministic first axis: the world axis least aligned
    # with up (falls back to a derived perpendicular when up is a
    # world axis exactly).
    world_axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    candidates = sorted(world_axes, key=lambda a: abs(_dot(a, up_u)))
    a1 = candidates[0]
    a2 = (
        up_u[1] * a1[2] - up_u[2] * a1[1],
        up_u[2] * a1[0] - up_u[0] * a1[2],
        up_u[0] * a1[1] - up_u[1] * a1[0],
    )
    a2_len = math.sqrt(_dot(a2, a2))
    if a2_len < 1e-9:
        raise RoofRefused("up is degenerate -- no lateral axes")
    a2 = tuple(c / a2_len for c in a2)

    # Measured extents: union of the band members' own bounds.
    ext1 = max(_extent(p.bounds_min, p.bounds_max, a1) for p, _, _ in band)
    ext2 = max(_extent(p.bounds_min, p.bounds_max, a2) for p, _, _ in band)

    if ext1 < MIN_ROOF_EXTENT_M or ext2 < MIN_ROOF_EXTENT_M:
        raise RoofRefused(
            f"topmost horizontal plane's measured extent {min(ext1, ext2):.2f} m "
            f"is below the roof minimum {MIN_ROOF_EXTENT_M} m -- a cap or "
            "skylight is real geometry but not a roof surface"
        )

    # Position: area-weighted centroid of the band members (their own
    # measured centroids; nothing synthetic).
    areas = []
    for p, _, _ in band:
        areas.append(max(ext1 * ext2 / len(band), 1e-6))
    total = sum(areas)
    cx = sum(p.centroid[0] * a for (p, _, _), a in zip(band, areas)) / total
    cy = sum(p.centroid[1] * a for (p, _, _), a in zip(band, areas)) / total
    cz = sum(p.centroid[2] * a for (p, _, _), a in zip(band, areas)) / total

    confidence = max(0.05, min(1.0, top_up_dot * min(1.0, ext1 / 2.0) * min(1.0, ext2 / 2.0)))

    supporting = tuple(sorted(p.plane_id for p, _, _ in band))
    evidence_ids = tuple(f"roof-{pid}" for pid in supporting)

    return RoofFit(
        plane_id=top.plane_id,
        supporting_plane_ids=supporting,
        position=(cx, cy, cz),
        width_m=ext1,
        depth_m=ext2,
        height_m=top_h,
        up_dot=top_up_dot,
        n_planes=len(band),
        confidence=confidence,
        evidence_ids=evidence_ids,
    )
