"""Room-aware plane detection: floor, ceiling and walls from height / axis histograms.

Greedy RANSAC (``detect_planes``) is fragile on a furnished room: furniture tops become dozens of "floors" and plain
walls, which carry few points, are lost (measured on 18 real living-room photos: 31 floor fragments, 1 wall, no room).
With 'up' known (the frame is canonicalised first) a room is far more constrained than a bag of planes:

  * the floor is the LOWEST strong horizontal peak below the cameras, the ceiling the highest above them;
  * walls are vertical, mostly orthogonal, and lie OUTSIDE the cameras (the photographer stands inside the room).

This returns a ``PlaneDetection`` the unchanged compile pipeline (refine -> classify -> rooms) consumes, or None when the
cloud does not look like a room (the caller then uses RANSAC). It claims nothing it did not find: a missing ceiling or
wall is simply absent.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from perception.geometry.planes import DetectedPlane, PlaneDetection
from provenance import Uncertainty

BIN = 0.05                 # histogram bin (working units)
BAND = 1.5 * BIN           # half-width of a plane's inlier band: MVS noise is well above 2 cm
MIN_FRACTION = 0.012       # a plane needs this share of the (outlier-free) points to count


def core_mask(P: np.ndarray) -> np.ndarray:
    """Points inside the main body of the cloud: the 10-90 percentile box grown by 100% each way. Dense MVS leaves
    floaters tens of units out that would otherwise stretch every histogram."""
    lo, hi = np.percentile(P, 10, axis=0), np.percentile(P, 90, axis=0)
    pad = hi - lo
    return np.all((P >= lo - pad) & (P <= hi + pad), axis=1)


def _peaks(counts: np.ndarray, min_count: float) -> List[int]:
    sm = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    return [i for i in range(len(sm)) if sm[i] >= min_count and sm[i] >= sm[max(0, i - 1)] and sm[i] > sm[min(len(sm) - 1, i + 1)]]


def _fit(points: np.ndarray, normal: np.ndarray, d: float) -> Tuple[float, int]:
    dist = points @ normal + d
    return float(math.sqrt(float(np.mean(dist * dist)))) if len(dist) else 0.0, int(len(dist))


#: Handheld interior photographs are taken at roughly chest-to-eye height (nominal, not measured).
NOMINAL_EYE_HEIGHT = 1.45


def eye_height_factor(result) -> Optional[float]:
    """Factor that makes the median camera height above the detected floor equal NOMINAL_EYE_HEIGHT (frame must be
    canonical, +Y up). A far better scale prior than viewing distance, but still a prior: None when no floor shows."""
    cams = np.asarray([c.position for c in result.camera_poses], dtype=float)
    if len(result.points) < 200 or len(cams) < 3:
        return None
    P = np.asarray([p.position for p in result.points], dtype=float)
    med = np.median(cams, axis=0)
    mad = np.median(np.abs(cams - med), axis=0) + 1e-9
    good = np.all(np.abs(cams - med) <= 4.0 * mad + 0.5, axis=1)
    cam_y = float(np.median(cams[good][:, 1])) if good.sum() >= 3 else float(np.median(cams[:, 1]))
    y = P[core_mask(P), 1]
    edges = np.arange(y.min(), y.max() + BIN, BIN)
    if len(edges) < 6:
        return None
    hy, _ = np.histogram(y, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    # the floor is the lowest strong sheet clearly under the cameras; scale-free thresholds (share of points only)
    for i in _peaks(hy.astype(float), max(MIN_FRACTION * len(y), 25)):
        if centres[i] < cam_y:
            h = cam_y - float(centres[i])
            return NOMINAL_EYE_HEIGHT / h if h > 1e-6 else None
    return None


def detect_room_planes(result, up: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> Optional[PlaneDetection]:
    pts = list(result.points)
    cams = np.asarray([c.position for c in result.camera_poses], dtype=float)
    if len(pts) < 200 or len(cams) < 3:
        return None
    P = np.asarray([p.position for p in pts], dtype=float)
    ids = [p.track_id for p in pts]
    core = core_mask(P)                                               # drop floaters before measuring anything
    # cameras: robust to a badly registered one
    med = np.median(cams, axis=0)
    mad = np.median(np.abs(cams - med), axis=0) + 1e-9
    good = np.all(np.abs(cams - med) <= 4.0 * mad + 0.5, axis=1)
    cams = cams[good] if good.sum() >= 3 else cams
    cam_y = float(np.median(cams[:, 1]))
    min_count = max(MIN_FRACTION * core.sum(), 25)

    y = P[core, 1]
    edges = np.arange(y.min(), y.max() + BIN, BIN)
    if len(edges) < 6:
        return None
    hy, _ = np.histogram(y, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    below = [i for i in _peaks(hy.astype(float), min_count) if centres[i] < cam_y - 0.3]
    above = [i for i in _peaks(hy.astype(float), min_count) if centres[i] > cam_y + 0.3]
    if not below:
        return None
    floor_y = float(centres[below[0]])                                # lowest strong sheet below the cameras
    ceil_y = float(centres[above[-1]]) if above and centres[above[-1]] - floor_y > 1.0 else None
    top = ceil_y if ceil_y is not None else float(np.percentile(P[core, 1], 98))
    band = core & (P[:, 1] > floor_y + 0.25 * (top - floor_y)) & (P[:, 1] < top - 0.1 * (top - floor_y))
    if band.sum() < 100:
        return None

    # dominant horizontal direction: the rotation that makes the xz marginals sharpest (walls pile up on two axes)
    xz = P[band][:, [0, 2]]
    best_t, best_s = 0.0, -1.0
    for deg in range(0, 90, 2):
        t = math.radians(deg)
        u = xz[:, 0] * math.cos(t) + xz[:, 1] * math.sin(t)
        v = -xz[:, 0] * math.sin(t) + xz[:, 1] * math.cos(t)
        s = sum(float(np.sum(np.histogram(a, bins=np.arange(a.min(), a.max() + BIN, BIN))[0].astype(float) ** 2))
                for a in (u, v))
        if s > best_s:
            best_t, best_s = t, s
    c, s_ = math.cos(best_t), math.sin(best_t)
    axes = [np.array([c, 0.0, s_]), np.array([-s_, 0.0, c])]

    planes: List[Tuple[str, np.ndarray, float]] = [("floor", np.array(up, float), -floor_y)]
    if ceil_y is not None:
        planes.append(("ceiling", np.array(up, float), -ceil_y))
    for k, ax in enumerate(axes):
        coord = P[band] @ ax
        cam_c = cams @ ax
        e = np.arange(coord.min(), coord.max() + BIN, BIN)
        if len(e) < 6:
            continue
        h, _ = np.histogram(coord, bins=e)
        cc = (e[:-1] + e[1:]) / 2
        pk = _peaks(h.astype(float), max(MIN_FRACTION * band.sum() * 0.5, 20))
        low = [i for i in pk if cc[i] < cam_c.min() - 0.1]
        high = [i for i in pk if cc[i] > cam_c.max() + 0.1]
        for side, cand in (("a", low), ("b", high)):
            if cand:
                # the outermost peak that still has real support is the wall (nearer ones are furniture faces)
                i = cand[0] if side == "a" else cand[-1]
                planes.append((f"wall{k}{side}", ax, -float(cc[i])))

    detected: List[DetectedPlane] = []
    claimed = np.zeros(len(pts), dtype=bool)
    for seq, (_, n, d) in enumerate(planes):
        dist = np.abs(P @ n + d)
        m = (dist <= BAND) & ~claimed
        if n[1] == 0.0:                                               # a wall only spans the floor..ceiling slab
            m &= (P[:, 1] >= floor_y - BAND) & (P[:, 1] <= top + BAND)
        if m.sum() < 25:
            continue
        claimed |= m
        rms, cnt = _fit(P[m], n, d)
        detected.append(DetectedPlane(
            plane_id=f"plane-{seq:03d}", normal=tuple(float(v) for v in n), d=float(d),
            inlier_ids=sorted(ids[i] for i in np.flatnonzero(m)), inlier_rms_distance_m=rms,
            uncertainty=Uncertainty(confidence=min(1.0, cnt / (cnt + 10.0)),
                                    note=f"room-prior plane: histogram peak, {cnt} pts, rms {rms:.3f}")))
    if not any(abs(p.normal[1]) > 0.9 for p in detected) or sum(1 for p in detected if abs(p.normal[1]) < 0.1) < 2:
        return None                                                    # not room-like: let RANSAC decide
    detected.sort(key=lambda p: (-len(p.inlier_ids), p.plane_id))
    detected = [DetectedPlane(plane_id=f"plane-{i:03d}", normal=p.normal, d=p.d, inlier_ids=p.inlier_ids,
                              inlier_rms_distance_m=p.inlier_rms_distance_m, uncertainty=p.uncertainty)
                for i, p in enumerate(detected)]
    assigned = {pid for p in detected for pid in p.inlier_ids}
    return PlaneDetection(planes=detected, unassigned_point_ids=sorted(set(ids) - assigned), points_total=len(pts))
