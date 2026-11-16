"""Clean room footprints without forcing right angles (BUILD_SPEC B10).

Wall segments (2-D, floor plane) -> dominant directions (length-weighted circular histogram) -> snap only when within
5 deg (old buildings are not square) -> merge collinear fragments -> polygonize. A room whose outline does not close
within 0.3 m is reported ``partial`` with the gap, never closed by invention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SNAP_DEG = 5.0
PEAK_SHARE = 0.15
COLLINEAR_OFFSET_M = 0.05
COLLINEAR_GAP_M = 0.20
CLOSURE_PARTIAL_M = 0.30


def _angles_lengths(seg: np.ndarray):
    d = seg[:, 1] - seg[:, 0]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0, np.hypot(d[:, 0], d[:, 1])


def dominant_directions(seg: np.ndarray) -> np.ndarray:
    """Peak angles (deg in [0,180)) of segments ``(n,2,2)``: length-weighted 1-degree histogram, 5-degree smoothing."""
    ang, ln = _angles_lengths(seg)
    hist = np.bincount(np.minimum(ang.astype(int), 179), weights=ln, minlength=180)
    sm = sum(np.roll(hist, s) for s in range(-2, 3))
    peak = (sm >= np.roll(sm, 1)) & (sm > np.roll(sm, -1)) & (sm >= PEAK_SHARE * hist.sum())
    refined: list[float] = []
    for b in np.flatnonzero(peak):                      # refine each plateau to the length-weighted mean of its segments
        near = np.abs((ang - (b + 0.5) + 90.0) % 180.0 - 90.0) <= SNAP_DEG
        if not near.any():
            continue
        off = (ang[near] - (b + 0.5) + 90.0) % 180.0 - 90.0
        p = float(((b + 0.5) + np.average(off, weights=ln[near])) % 180.0)
        if all(abs((p - q + 90.0) % 180.0 - 90.0) > SNAP_DEG for q in refined):
            refined.append(p)
    return np.asarray(refined, dtype=float)


def snap(seg: np.ndarray, peaks: np.ndarray, tol_deg: float = SNAP_DEG) -> np.ndarray:
    """Rotate each segment about its midpoint onto the nearest peak when within ``tol_deg``; otherwise keep its angle."""
    out = np.asarray(seg, dtype=float).copy()
    ang, ln = _angles_lengths(out)
    for i in range(len(out)):
        if ln[i] == 0 or len(peaks) == 0:
            continue
        diff = (peaks - ang[i] + 90.0) % 180.0 - 90.0
        j = int(np.argmin(np.abs(diff)))
        if abs(diff[j]) > tol_deg:
            continue
        t = np.radians(ang[i] + diff[j])
        u = np.array([np.cos(t), np.sin(t)])
        if np.dot(u, out[i, 1] - out[i, 0]) < 0:
            u = -u
        mid = out[i].mean(axis=0)
        out[i] = [mid - u * ln[i] / 2, mid + u * ln[i] / 2]
    return out


def merge_collinear(seg: np.ndarray, angle_tol_deg: float = SNAP_DEG) -> np.ndarray:
    """Merge same-direction segments offset < 5 cm with gaps < 20 cm: group, then one sweep along the direction."""
    seg = np.asarray(seg, dtype=float)
    used = np.zeros(len(seg), bool); out = []
    for i in range(len(seg)):
        if used[i]:
            continue
        d = seg[i, 1] - seg[i, 0]; u = d / np.hypot(*d); nrm = np.array([-u[1], u[0]])
        grp = [i]
        for j in range(i + 1, len(seg)):
            if used[j]:
                continue
            e = seg[j, 1] - seg[j, 0]
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(e / np.hypot(*e), u)), 0.0, 1.0)))
            if ang <= angle_tol_deg and abs(np.dot(seg[j, 0] - seg[i, 0], nrm)) < COLLINEAR_OFFSET_M:
                grp.append(j)
        t = np.sort(np.array([[np.dot(seg[k, 0] - seg[i, 0], u), np.dot(seg[k, 1] - seg[i, 0], u)] for k in grp]), axis=1)
        order = np.argsort(t[:, 0]); t = t[order]; ids = [grp[o] for o in order]
        lo, hi, members = t[0, 0], t[0, 1], [ids[0]]
        for (a, b), k in zip(t[1:], ids[1:]):
            if a - hi < COLLINEAR_GAP_M:
                hi = max(hi, b); members.append(k)
            else:
                out.append([seg[i, 0] + u * lo, seg[i, 0] + u * hi]); used[members] = True
                lo, hi, members = a, b, [k]
        out.append([seg[i, 0] + u * lo, seg[i, 0] + u * hi]); used[members] = True
    return np.asarray(out)


@dataclass(frozen=True)
class Footprint:
    polygon: np.ndarray        # (m,2); empty when nothing closed
    closure_gap_m: float
    status: str                # "closed" | "partial"


def footprint(seg: np.ndarray) -> Footprint:
    """Snap, merge, extend each wall end up to the closure tolerance, polygonize; ``partial`` if no face closes."""
    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize, unary_union

    seg = np.asarray(seg, dtype=float)
    s = merge_collinear(snap(seg, dominant_directions(seg)))
    ext = []
    for a, b in s:
        u = (b - a) / np.hypot(*(b - a))
        ext.append([tuple(a - u * CLOSURE_PARTIAL_M), tuple(b + u * CLOSURE_PARTIAL_M)])
    faces = list(polygonize(unary_union(MultiLineString(ext))))
    if faces:
        f = max(faces, key=lambda p: p.area)
        return Footprint(np.asarray(f.exterior.coords)[:-1], 0.0, "closed")
    ends = s.reshape(-1, 2)
    dist = np.linalg.norm(ends[:, None] - ends[None, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    same = np.arange(len(ends)) ^ 1                       # an endpoint's own other end is not a gap
    dist[np.arange(len(ends)), same] = np.inf
    return Footprint(np.empty((0, 2)), float(dist.min(axis=1).max()), "partial")
