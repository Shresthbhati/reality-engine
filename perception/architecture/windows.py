"""Measured window detection (P7-03 architectural-perception breadth).

A window's evidence is an EDGE-BOUNDED OPENING in a wall plane's own
inlier coverage that does NOT reach the floor: lateral neighbors are
covered, coverage resumes above, and the empty rectangle is window-
sized. The detector measures exactly that from the wall's real points
and refuses when the points do not demonstrate it -- the same
refusal-is-a-fact discipline as stairs/columns/beams (FitRefused, never
best-effort geometry).

Relationship to classify.detect_wall_opening: that function measures
the FLOOR-height band (doors); this module scans the FULL wall face,
finds interior empty rectangles, and gates them by measured size + sill
height. An opening reaching the floor is a door-shaped fact and is
refused here with the measured sill in the message -- never reclassified
silently.

Buckets: SCAN_BUCKET_M (0.05 m) reused from classify.py so both opening
detectors share one lateral resolution. Size gates are interior-
residential-window-wide (0.4-3.0 m width, 0.4-2.6 m height, sill
0.2-2.5 m) -- deliberately generous, not scene-tuned. Deterministic:
pure float math, no clocks, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from perception.architecture.classify import (
    DOOR_MAX_WIDTH_M,
    PlaneInput,
    SCAN_BUCKET_M,
    _dot,
    _normalize,
)
from perception.architecture.parametric import FitRefused

Vec3 = Tuple[float, float, float]

#: Window size gates (interior windows). Wide on purpose.
WIN_MIN_WIDTH_M = 0.4
WIN_MAX_WIDTH_M = 3.0
WIN_MIN_HEIGHT_M = 0.4
WIN_MAX_HEIGHT_M = 2.6

#: A window's sill sits above the floor by at least this much and its
#: head stays within reach of a wall top band. An opening with a lower
#: sill is a door-shaped fact, not a window.
MIN_SILL_M = 0.2
MAX_SILL_M = 2.5



class WindowRefused(FitRefused):
    """The wall's points do not demonstrate a window-sized, edge-bounded
    opening. The measured fact is in the message."""


@dataclass(frozen=True)
class WindowFit:
    """A measured window: the opening's real geometry measured from the
    wall plane's own inlier coverage."""

    wall_plane_id: str
    #: Center of the measured opening (world frame).
    position: Vec3
    width_m: float
    height_m: float
    #: Sill height above the caller's floor reference.
    sill_height_m: float
    bounds_min: Vec3
    bounds_max: Vec3
    #: Wall inlier points bordering the measured opening (support).
    n_points: int
    confidence: float
    evidence_ids: Tuple[str, ...]
    kind: str = "window"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "wall_plane_id": self.wall_plane_id,
            "position": list(self.position),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_height_m": self.sill_height_m,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "n_points": self.n_points,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
        }


def _coverage_grid(
    inlier_positions: Sequence[Vec3],
    lateral: Vec3,
    up_u: Vec3,
    lat_min: float,
    lat_max: float,
    z_min: float,
    z_max: float,
):
    """Boolean coverage grid over the wall face (lateral x height)."""
    span = lat_max - lat_min
    zspan = z_max - z_min
    nx = max(1, int(math.ceil(span / SCAN_BUCKET_M)))
    nz = max(1, int(math.ceil(zspan / SCAN_BUCKET_M)))
    grid = [[False] * nx for _ in range(nz)]
    for pos in inlier_positions:
        lat = _dot(pos, lateral)
        h = _dot(pos, up_u)
        ix = int((lat - lat_min) / SCAN_BUCKET_M)
        iz = int((h - z_min) / SCAN_BUCKET_M)
        ix = max(0, min(nx - 1, ix))
        iz = max(0, min(nz - 1, iz))
        grid[iz][ix] = True
    return grid, nx, nz


def _largest_empty_rectangle(grid, nx: int, nz: int):
    """Largest all-empty axis-aligned rectangle in the boolean grid.
    Classic histogram method per row; deterministic. Returns
    (ix0, iz0, ix1, iz1) in grid indices or None."""
    heights = [0] * nx
    best = None
    best_area = 0
    for iz in range(nz):
        for ix in range(nx):
            heights[ix] = 0 if grid[iz][ix] else heights[ix] + 1
        stack = []
        for ix in range(nx + 1):
            h = heights[ix] if ix < nx else 0
            start = ix
            while stack and stack[-1][1] >= h:
                sx, sh = stack.pop()
                area = sh * (ix - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, iz - sh + 1, ix, iz + 1)
                start = sx
            stack.append((start, h))
    return best




def detect_window(
    wall: PlaneInput, up: Vec3 = (0.0, 0.0, 1.0), floor_height: float = 0.0
) -> WindowFit:
    """Detect a window in a wall plane's real inlier coverage or refuse.

    The wall must be provided with `inlier_positions` (the classifier's
    own PlaneInput carries them). Raises WindowRefused (a FitRefused
    subclass) when the measured structure does not demonstrate a
    window-sized, edge-bounded opening. Deterministic.
    """
    if not wall.inlier_positions:
        raise WindowRefused(
            f"wall plane {wall.plane_id} has no inlier positions -- an "
            "opening cannot be measured without the wall's own points"
        )
    up_u = _normalize(up)
    lateral = (
        wall.normal[1] * up_u[2] - wall.normal[2] * up_u[1],
        wall.normal[2] * up_u[0] - wall.normal[0] * up_u[2],
        wall.normal[0] * up_u[1] - wall.normal[1] * up_u[0],
    )
    lateral_len = math.sqrt(_dot(lateral, lateral))
    if lateral_len < 1e-9:
        raise WindowRefused(
            f"wall plane {wall.plane_id} normal is parallel to up -- "
            "not a wall, no lateral axis to scan"
        )
    lateral = tuple(c / lateral_len for c in lateral)

    lat_min = _dot(wall.bounds_min, lateral)
    lat_max = _dot(wall.bounds_max, lateral)
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    z_min = _dot(wall.bounds_min, up_u)
    z_max = _dot(wall.bounds_max, up_u)
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    grid, nx, nz = _coverage_grid(
        wall.inlier_positions, lateral, up_u, lat_min, lat_max, z_min, z_max
    )
    rect = _largest_empty_rectangle(grid, nx, nz)
    if rect is None:
        raise WindowRefused(
            f"wall plane {wall.plane_id} coverage has no interior gap"
        )
    ix0, iz0, ix1, iz1 = rect
    # A gap whose edge touches the wall face boundary is a boundary of the
    # measured face, not an edge-BOUNDED interior opening (missing data
    # looks identical to a real opening edge there). The BOTTOM edge is
    # the exception: a gap reaching it is the door-shaped fact the sill
    # gate below measures, so bottom-touch falls through to that gate.
    interior = ix0 > 0 and ix1 < nx and iz1 < nz
    gap_w = (ix1 - ix0) * SCAN_BUCKET_M
    gap_h = (iz1 - iz0) * SCAN_BUCKET_M

    if not interior:
        raise WindowRefused(
            "measured gap reaches the wall face boundary -- the face edge "
            "is not evidence of an opening's edge; refusing to guess a "
            "window where data may simply be missing"
        )
    if gap_w < WIN_MIN_WIDTH_M or gap_w > WIN_MAX_WIDTH_M:
        raise WindowRefused(
            f"measured opening width {gap_w:.2f} m outside window band "
            f"[{WIN_MIN_WIDTH_M}, {WIN_MAX_WIDTH_M}] m"
        )
    if gap_h < WIN_MIN_HEIGHT_M or gap_h > WIN_MAX_HEIGHT_M:
        raise WindowRefused(
            f"measured opening height {gap_h:.2f} m outside window band "
            f"[{WIN_MIN_HEIGHT_M}, {WIN_MAX_HEIGHT_M}] m"
        )

    sill = z_min + iz0 * SCAN_BUCKET_M - floor_height
    if sill < MIN_SILL_M:
        raise WindowRefused(
            f"measured opening reaches the floor band (sill {sill:.2f} m "
            f"above floor < {MIN_SILL_M} m) -- a door-shaped fact, not a "
            "window; use the door detector"
        )
    if sill > MAX_SILL_M:
        raise WindowRefused(
            f"measured sill {sill:.2f} m above floor exceeds the window "
            f"band ({MAX_SILL_M} m)"
        )

    # World-frame geometry of the opening, reconstructed through the
    # wall's own centroid (same technique as classify.detect_wall_opening).
    centroid_lat = _dot(wall.centroid, lateral)
    centroid_h = _dot(wall.centroid, up_u)
    gap_lat_lo = lat_min + ix0 * SCAN_BUCKET_M
    gap_lat_hi = gap_lat_lo + gap_w
    gap_z_lo = z_min + iz0 * SCAN_BUCKET_M
    gap_z_hi = gap_z_lo + gap_h

    def _point(lat_value: float, height_value: float) -> Vec3:
        return tuple(
            wall.centroid[k]
            + (lat_value - centroid_lat) * lateral[k]
            + (height_value - centroid_h) * up_u[k]
            for k in range(3)
        )

    corner_a = _point(gap_lat_lo, gap_z_lo)
    corner_b = _point(gap_lat_hi, gap_z_hi)
    bounds_min = tuple(min(corner_a[k], corner_b[k]) for k in range(3))
    bounds_max = tuple(max(corner_a[k], corner_b[k]) for k in range(3))
    center = (
        (corner_a[0] + corner_b[0]) / 2.0,
        (corner_a[1] + corner_b[1]) / 2.0,
        (corner_a[2] + corner_b[2]) / 2.0,
    )

    # Support: the wall inliers adjacent to the opening (the bounding
    # ring's coverage is what makes the gap edge-BOUNDED rather than an
    # artifact of missing data). Confidence: coverage density around the
    # gap, discounted by nothing else -- measured, documented.
    ring_needed = 2 * ((ix1 - ix0) + (iz1 - iz0))
    ring_have = 0
    for pos in wall.inlier_positions:
        lat = _dot(pos, lateral)
        h = _dot(pos, up_u)
        ix = int((lat - lat_min) / SCAN_BUCKET_M)
        iz = int((h - z_min) / SCAN_BUCKET_M)
        if ix0 - 1 <= ix <= ix1 and (iz == iz0 - 1 or iz == iz1):
            ring_have += 1
        elif iz0 - 1 <= iz <= iz1 and (ix == ix0 - 1 or ix == ix1):
            ring_have += 1
    ring_frac = ring_have / max(1, ring_needed)
    confidence = max(0.05, min(1.0, ring_frac))

    evidence_ids = tuple(
        f"win-{wall.plane_id}-{i}" for i in range(max(1, ring_have))
    )

    return WindowFit(
        wall_plane_id=wall.plane_id,
        position=center,
        width_m=gap_w,
        height_m=gap_h,
        sill_height_m=sill,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        n_points=ring_have,
        confidence=confidence,
        evidence_ids=evidence_ids,
    )
