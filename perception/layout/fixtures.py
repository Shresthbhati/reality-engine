"""Electrical and heating fixtures (sockets, switches, radiators, ceiling lights) placed on the solved room (task T-3.4).

A detector gives a 2D box; the solved layout gives the walls and the ceiling as metric planes. The ray through the box centre meets
the host surface at one point, which is the fixture's wall, its position along that wall and its height above the floor. No depth
under the box is trusted (a socket is a few centimetres wide: its own depth is noise); only the plane the ray hits.

A detection is refused, with the reason, when no wall is behind it, when the height is implausible for that kind (a "socket"
near the ceiling is a detection error) or when the ceiling is needed and unknown. Nothing is guessed to make a fixture appear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.layout.surface_layout import SurfaceLayout

#: kind -> (host surface, plausible height range above the floor in metres)
KINDS: Dict[str, Tuple[str, Optional[Tuple[float, float]]]] = {
    "socket": ("wall", (0.05, 1.6)),
    "switch": ("wall", (0.6, 1.8)),
    "radiator": ("wall", (0.0, 1.2)),
    "ceiling_light": ("ceiling", None),
}
#: a ray may land this far beyond the observed extent of a wall (the wall continues past what was seen)
WALL_EXTENT_SLACK_M = 0.3


@dataclass(frozen=True)
class Fixture:
    kind: str
    wall_index: Optional[int]  # None for a ceiling fixture
    s: Optional[float]  # metres along the wall's tangent (reference frame); None for a ceiling fixture
    h: float  # height above the floor, metres
    width_m: float
    height_m: float
    score: float
    position: Tuple[float, float, float]  # reference (first photo) camera frame: x right, y DOWN, z forward
    depth_m: float


def place_fixtures(detections: Sequence[Tuple[str, float, Tuple[float, float, float, float]]], layout: SurfaceLayout, *,
                   hfov_deg: float, image_size: Tuple[int, int], R: Optional[np.ndarray] = None,
                   t: Optional[np.ndarray] = None) -> Tuple[List[Fixture], List[dict]]:
    """``detections``: [(kind, score, (x0, y0, x1, y1) pixels)] from one photo whose pose in the reference frame is (R, t)."""
    placed: List[Fixture] = []
    refused: List[dict] = []
    if layout.camera_height_m is None:
        return placed, [{"kind": k, "reason": "no solved layout to place fixtures on"} for k, _, _ in detections]
    w, h = image_size
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    rot = np.eye(3) if R is None else np.asarray(R, float)
    org = np.zeros(3) if t is None else np.asarray(t, float)
    up = np.asarray(layout.up)
    cam_h = float(layout.camera_height_m)
    room_h = layout.room_height_m

    def height_of(p: np.ndarray) -> float:
        return float(p @ up + cam_h)

    def depth_of(p: np.ndarray) -> float:
        return float((rot.T @ (p - org))[2])

    for kind, score, (x0, y0, x1, y1) in detections:
        if kind not in KINDS:
            refused.append({"kind": kind, "reason": f"unknown fixture kind {kind!r}"})
            continue
        host, limits = KINDS[kind]
        u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        d = rot @ np.array([(u - w / 2.0) / f, (v - h / 2.0) / f, 1.0])

        if host == "ceiling":
            if room_h is None:
                refused.append({"kind": kind, "reason": "the ceiling height is unknown, so a ceiling fixture cannot be placed"})
                continue
            ud = float(up @ d)
            if ud <= 1e-6:
                refused.append({"kind": kind, "reason": "the detection's ray does not rise toward a ceiling"})
                continue
            sc = (room_h - cam_h - float(up @ org)) / ud
            if sc <= 0:
                refused.append({"kind": kind, "reason": "the detection's ray meets the ceiling behind the camera"})
                continue
            p = org + sc * d
            z = depth_of(p)
            placed.append(Fixture(kind, None, None, float(room_h), (x1 - x0) / f * z, (y1 - y0) / f * z, float(score),
                                  tuple(float(c) for c in p), z))
            continue

        best = None
        for wi, wl in enumerate(layout.walls):
            n = np.asarray(wl.plane.normal)
            denom = float(n @ d)
            if denom >= -1e-6:
                continue
            sc = -(wl.plane.d + float(n @ org)) / denom
            if sc <= 0:
                continue
            p = org + sc * d
            s = float(p @ np.asarray(wl.tangent))
            hgt = height_of(p)
            top = (room_h if room_h is not None else 3.0) + 0.1
            if not (wl.s_min - WALL_EXTENT_SLACK_M <= s <= wl.s_max + WALL_EXTENT_SLACK_M) or not (-0.1 <= hgt <= top):
                continue
            if best is None or sc < best[0]:
                best = (sc, wi, s, hgt, p)
        if best is None:
            refused.append({"kind": kind, "reason": "no wall behind the detection (its ray misses every solved wall)"})
            continue
        _, wi, s, hgt, p = best
        lo, hi = limits  # type: ignore[misc]
        if not (lo <= hgt <= hi):
            refused.append({"kind": kind, "reason": f"height {hgt:.2f} m is outside the plausible {lo:g}-{hi:g} m range for a {kind}"})
            continue
        z = depth_of(p)
        placed.append(Fixture(kind, wi, s, max(0.0, hgt), (x1 - x0) / f * z, (y1 - y0) / f * z, float(score),
                              tuple(float(c) for c in p), z))
    return placed, refused
