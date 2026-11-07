"""Room layout from ONE photograph: metric depth + per-pixel surface labels -> floor, ceiling, walls, windows, doors.

Why this exists: sparse structure-from-motion needs many geometrically consistent, well-overlapping photos. With few,
low-overlap or AI-generated photos it yields a few hundred noisy points and the point-plane stage promotes noise as
structure (five "floors" at different heights, a floor above the ceiling). A learned depth model plus a learned surface
segmenter gives every photo thousands of labelled 3D points, so the room shell no longer depends on matching.

Pure numpy/scipy; the models live in ``perception.layout.learned_models`` so this solver is testable on synthetic rooms
with exact ground truth. Camera frame: x right, y down, z forward (OpenCV). Plane form: ``normal . p + d = 0`` with a
unit normal facing the camera, so ``d`` is the camera's distance to the plane.

Honesty contract: every stage that cannot be justified REFUSES (recorded in ``refused``) instead of guessing. A floor that
is not planar, a ceiling that is not a horizontal plane, a wall with too little support all produce no surface, never a
plausible-looking invented one. Window and door geometry is taken from where the pixel RAYS meet the host wall plane, never
from the depth under the glass (a depth model sees through windows to the outside).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

Vec = Tuple[float, float, float]

#: RANSAC inlier tolerance for planes, metres (indoor metric depth is good to a few centimetres on flat surfaces)
PLANE_TOL_M = 0.05
#: monocular depth error grows with distance, so the inlier band is PLANE_TOL_M + this fraction of the point's depth
DEPTH_TOL_FRACTION = 0.02
#: a floor/ceiling candidate needs this many labelled pixels, otherwise it is not evidence
MIN_FLOOR_PX = 400
MIN_CEILING_PX = 200
#: the floor normal may be at most this far from the camera's vertical before the floor label is treated as wrong
MAX_UP_TILT_DEG = 40.0
#: ceiling points must sit within this median absolute deviation of one height to count as a horizontal plane
CEILING_MAD_M = 0.12
ROOM_HEIGHT_RANGE_M = (1.8, 6.0)
MIN_WALL_SUPPORT_PX = 150
#: below this visible floor extent a plane fit cannot pin down the tilt, so heights are flagged approximate
MIN_FLOOR_SPAN_M = 1.0
MAX_WALLS = 6
#: a wall must carry at least this share of all wall evidence, lie within a plausible distance and be flat enough
WALL_MIN_SHARE = 0.05
WALL_DIST_RANGE_M = (0.4, 8.0)  # indoor metric depth is unreliable beyond ~8 m
#: candidates are merged only when the union is still this flat; a wall is rejected outright above WALL_REJECT_RMS_M
WALL_MAX_RMS_M = 0.09
WALL_REJECT_RMS_M = 0.25
#: two wall candidates this close in orientation and offset are one wall bent by depth error
WALL_DUP_ANGLE_DEG = 10.0
WALL_DUP_OFFSET_M = 0.30
OPENING_MIN_PX = 60
OPENING_WALL_DIST_M = 0.25
#: plausible metric extents; an opening outside them is a segmentation error and is refused with the reason
WINDOW_LIMITS = {"width": (0.3, 4.5), "height": (0.3, 2.7), "max_sill": 1.6}
DOOR_LIMITS = {"width": (0.5, 3.5), "height": (1.6, 3.0), "max_sill": 0.3}
MAX_FIT_POINTS = 4000


@dataclass(frozen=True)
class Plane:
    normal: Vec
    d: float  # camera distance to the plane (normal faces the camera)


@dataclass(frozen=True)
class Wall:
    plane: Plane
    tangent: Vec  # horizontal unit vector along the wall
    s_min: float  # extent along the tangent, metres
    s_max: float
    support_px: int
    rms_m: float


@dataclass(frozen=True)
class Opening:
    kind: str  # "window" | "door"
    wall_index: int
    s_min: float
    s_max: float
    h_min: float  # metres above the floor
    h_max: float
    support_px: int


@dataclass(frozen=True)
class SurfaceLayout:
    up: Vec = (0.0, -1.0, 0.0)  # unit vector, camera frame, pointing up
    camera_height_m: Optional[float] = None
    room_height_m: Optional[float] = None
    floor_rms_m: Optional[float] = None
    walls: Tuple[Wall, ...] = ()
    openings: Tuple[Opening, ...] = ()
    refused: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = field(default=())


def backproject(depth: np.ndarray, hfov_deg: float) -> np.ndarray:
    """(H, W) z-depth in metres -> (H, W, 3) camera-frame points. Principal point at the image centre."""
    return _rays(*depth.shape, hfov_deg) * depth[..., None]


def _rays(h: int, w: int, hfov_deg: float) -> np.ndarray:
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - w / 2.0) / f, (v - h / 2.0) / f, np.ones((h, w))], axis=-1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _fit_plane_svd(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, float(-n @ c)


def _subsample(pts: np.ndarray, rng: np.random.Generator, cap: int = MAX_FIT_POINTS) -> np.ndarray:
    return pts if len(pts) <= cap else pts[rng.choice(len(pts), cap, replace=False)]


def _band(z: np.ndarray, tol: float) -> np.ndarray:
    return tol + DEPTH_TOL_FRACTION * np.abs(z)


def _ransac_plane(pts: np.ndarray, rng: np.random.Generator, tol: float, iters: int = 300, z: Optional[np.ndarray] = None):
    if len(pts) < 3:
        return None
    band = _band(pts[:, 2] if z is None else z, tol)
    best_mask, best_count = None, 0
    for _ in range(iters):
        i, j, k = rng.choice(len(pts), 3, replace=False)
        n = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        mask = np.abs(pts @ n - n @ pts[i]) <= band
        cnt = int(mask.sum())
        if cnt > best_count:
            best_mask, best_count = mask, cnt
    if best_mask is None or best_count < 3:
        return None
    n, d = _fit_plane_svd(pts[best_mask])
    mask = np.abs(pts @ n + d) <= band
    n, d = _fit_plane_svd(pts[mask])
    return n, d


def _horizontal_basis(up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(a - (a @ up) * up)
    return e1, np.cross(up, e1)


def _ransac_line_2d(q: np.ndarray, rng: np.random.Generator, band: np.ndarray, iters: int = 250):
    best, best_count = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(q), 2, replace=False)
        t = q[j] - q[i]
        norm = np.linalg.norm(t)
        if norm < 0.2:  # two samples a few centimetres apart define a direction by noise alone
            continue
        n = np.array([-t[1], t[0]]) / norm
        mask = np.abs(q @ n - n @ q[i]) <= band
        cnt = int(mask.sum())
        if cnt > best_count:
            best, best_count = mask, cnt
    if best is None:
        return None
    mask = best
    n = off = None
    for _ in range(2):  # total-least-squares refit, then re-collect
        pts = q[mask]
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        n = vt[-1]
        off = float(n @ c)
        mask = np.abs(q @ n - off) <= band
    return n, off


def _finalise_wall(wall_pts: np.ndarray, members: np.ndarray, e1: np.ndarray, e2: np.ndarray, up: np.ndarray) -> Wall:
    """Total-least-squares line through the member points in the horizontal projection -> a camera-facing wall plane."""
    pm = wall_pts[members]
    q = np.stack([pm @ e1, pm @ e2], axis=-1)
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n2 = vt[-1]
    normal3 = n2[0] * e1 + n2[1] * e2
    dist = -float(n2 @ c)  # normal . p = off  ->  normal . p + d = 0 with d = -off
    if dist < 0:
        normal3, dist = -normal3, -dist
    tangent = _unit(np.cross(up, normal3))
    s = pm @ tangent
    rms = float(np.sqrt(np.mean((pm @ normal3 + dist) ** 2)))
    return Wall(Plane(tuple(map(float, normal3)), float(dist)), tuple(map(float, tangent)),
                float(np.percentile(s, 1)), float(np.percentile(s, 99)), int(len(members)), rms)


@dataclass
class ViewInput:
    """One photo's depth + surface masks and its camera pose in the REFERENCE frame (``p_ref = R @ p_cam + t``)."""
    depth: np.ndarray
    masks: Mapping[str, np.ndarray]
    hfov_deg: float = 65.0
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    name: str = "view"

    def pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.eye(3) if self.R is None else np.asarray(self.R, float)), (np.zeros(3) if self.t is None else np.asarray(self.t, float))


#: a non-reference photo whose own floor normal disagrees with the reference's by more than this is mis-registered
MAX_VERTICAL_DISAGREEMENT_DEG = 8.0
#: openings on one wall are one when their spans overlap or are separated by less than this, with matching heights
OPENING_ABUT_M = 0.12
OPENING_HEIGHT_TOL_M = 0.25


def _own_up(depth: np.ndarray, floor_mask: np.ndarray, hfov_deg: float, seed: int) -> Optional[np.ndarray]:
    """The floor normal of ONE photo in its own camera frame (facing the camera), or None without enough floor."""
    valid = np.isfinite(depth) & (depth > 0.05) & floor_mask
    if int(valid.sum()) < MIN_FLOOR_PX:
        return None
    pts = backproject(depth, hfov_deg)[valid]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(len(pts), MAX_FIT_POINTS), replace=False)
    fit = _ransac_plane(pts[idx], rng, PLANE_TOL_M)
    if fit is None:
        return None
    n, d = fit
    return -n if d < 0 else n


def _own_wall_normal(depth: np.ndarray, wall_mask: np.ndarray, hfov_deg: float, seed: int) -> Optional[np.ndarray]:
    """The dominant wall plane's normal of ONE photo in its own camera frame, or None without enough wall."""
    valid = np.isfinite(depth) & (depth > 0.05) & wall_mask
    if int(valid.sum()) < MIN_WALL_SUPPORT_PX * 3:
        return None
    pts = backproject(depth, hfov_deg)[valid]
    z = depth[valid]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(len(pts), MAX_FIT_POINTS), replace=False)
    fit = _ransac_plane(pts[idx], rng, PLANE_TOL_M, z=z[idx])
    return None if fit is None else fit[0]


#: a wall seen by a photo without floor must still be vertical in the reference frame, within this many degrees
MAX_WALL_TILT_DEG = 10.0


def _merge_openings(ops: List[Opening]) -> List[Opening]:
    """The same window seen by several photos is ONE window.

    Spans on the same wall and kind merge when they overlap, or abut (a gap under OPENING_ABUT_M) with matching heights: a
    photo's frame clips a window, so different photos see different fragments of it. The merged span is the UNION (clipping can
    only underestimate a width); heights are support-weighted.
    """
    out: List[Opening] = []
    for o in ops:
        for k, m in enumerate(out):
            if m.kind != o.kind or m.wall_index != o.wall_index:
                continue
            gap = max(m.s_min, o.s_min) - min(m.s_max, o.s_max)  # negative when the spans overlap
            h_overlap = min(m.h_max, o.h_max) - max(m.h_min, o.h_min)
            similar_height = abs(m.h_min - o.h_min) < OPENING_HEIGHT_TOL_M and abs(m.h_max - o.h_max) < OPENING_HEIGHT_TOL_M
            if gap <= OPENING_ABUT_M and h_overlap > 0 and similar_height:
                w1, w2 = m.support_px, o.support_px
                tot = float(w1 + w2)
                out[k] = Opening(m.kind, m.wall_index, min(m.s_min, o.s_min), max(m.s_max, o.s_max),
                                 (m.h_min * w1 + o.h_min * w2) / tot, (m.h_max * w1 + o.h_max * w2) / tot, w1 + w2)
                break
        else:
            out.append(o)
    return out


def solve_layout(
    depth: np.ndarray,
    masks: Mapping[str, np.ndarray],
    hfov_deg: float = 65.0,
    seed: int = 0,
) -> SurfaceLayout:
    """Solve the room shell visible in one photo. ``masks`` keys: floor, ceiling, wall, window, door (bool HxW)."""
    return solve_layout_views([ViewInput(depth, masks, hfov_deg)], seed=seed)


def solve_layout_views(views: Sequence[ViewInput], seed: int = 0) -> SurfaceLayout:
    """Solve ONE room from several registered photos: their labelled points are pooled in the first photo's frame.

    The first view is the reference (its camera is the origin, wall normals face it). Every other view must agree with the
    reference about which way is up; one that does not is mis-registered and is left out with a note, never blended in.
    """
    rng = np.random.default_rng(seed)
    ref = views[0]
    notes: List[str] = []
    # ---- per-view point clouds in the reference frame, with a vertical-agreement check on every non-reference view ----
    kept: List[Tuple[ViewInput, np.ndarray, np.ndarray, np.ndarray]] = []  # (view, pts_ref, valid, rays_cam)
    ref_up = _own_up(ref.depth, ref.masks.get("floor", np.zeros(ref.depth.shape, bool)), ref.hfov_deg, seed)
    for k, v in enumerate(views):
        R, t = v.pose()
        if k > 0 and ref_up is not None:
            own = _own_up(v.depth, v.masks.get("floor", np.zeros(v.depth.shape, bool)), v.hfov_deg, seed)
            if own is not None:
                ang = math.degrees(math.acos(float(np.clip((R @ own) @ ref_up, -1, 1))))
                if ang > MAX_VERTICAL_DISAGREEMENT_DEG:
                    notes.append(f"{v.name}: left out, its vertical disagrees with the reference by {ang:.0f} degrees (mis-registered)")
                    continue
            else:  # too little floor to verify the vertical: its dominant wall must come out vertical once placed
                wn = _own_wall_normal(v.depth, v.masks.get("wall", np.zeros(v.depth.shape, bool)), v.hfov_deg, seed)
                if wn is None:
                    notes.append(f"{v.name}: left out, neither floor nor wall visible enough to verify its vertical against the reference")
                    continue
                tilt = math.degrees(math.asin(float(np.clip(abs((R @ wn) @ ref_up), 0, 1))))
                if tilt > MAX_WALL_TILT_DEG:
                    notes.append(f"{v.name}: left out, its wall is {tilt:.0f} degrees off vertical in the reference frame, so its vertical "
                                 "disagrees with the reference (mis-registered)")
                    continue
        cam_pts = backproject(v.depth, v.hfov_deg)
        valid = np.isfinite(v.depth) & (v.depth > 0.05)
        kept.append((v, cam_pts @ R.T + t, valid, _rays(*v.depth.shape, v.hfov_deg)))

    def pooled(key: str) -> Tuple[np.ndarray, np.ndarray]:
        pts, zs = [], []
        for v, pr, valid, _ in kept:
            m = v.masks.get(key, np.zeros(v.depth.shape, bool)) & valid
            pts.append(pr[m])
            zs.append(v.depth[m])
        return (np.concatenate(pts), np.concatenate(zs)) if pts else (np.zeros((0, 3)), np.zeros(0))

    floor_pts, floor_z = pooled("floor")
    if len(floor_pts) < MIN_FLOOR_PX:
        return SurfaceLayout(refused=(f"floor: only {len(floor_pts)} labelled pixels (need {MIN_FLOOR_PX}); "
                                      "without a floor there is no vertical reference, so no layout is produced",))
    sel = np.arange(len(floor_pts)) if len(floor_pts) <= MAX_FIT_POINTS else rng.choice(len(floor_pts), MAX_FIT_POINTS, replace=False)
    fit = _ransac_plane(floor_pts[sel], rng, PLANE_TOL_M, z=floor_z[sel])
    if fit is None:
        return SurfaceLayout(refused=("floor: no plane could be fitted to the floor-labelled points",))
    up, d = fit
    if d < 0:  # the normal must face the reference camera
        up, d = -up, -d
    tilt = math.degrees(math.acos(float(np.clip(up @ np.array([0.0, -1.0, 0.0]), -1, 1))))
    if tilt > MAX_UP_TILT_DEG:
        return SurfaceLayout(refused=(f"floor: fitted floor is {tilt:.0f} degrees from the camera's vertical "
                                      f"(limit {MAX_UP_TILT_DEG:g}); the floor label is probably wrong",))

    def height_of(p: np.ndarray) -> np.ndarray:  # metres above the floor plane (the reference camera sits at d)
        return p @ up + d

    floor_rms = float(np.sqrt(np.mean(height_of(floor_pts) ** 2)))
    refused: List[str] = []
    fp = floor_pts
    # the tilt is pinned down by the NARROW in-plane extent (second principal axis), not the wide one
    along = (fp - fp.mean(axis=0)) @ np.linalg.svd(fp - fp.mean(axis=0), full_matrices=False)[2][1]
    span = float(np.percentile(along, 98) - np.percentile(along, 2))
    if span < MIN_FLOOR_SPAN_M:
        notes.append(f"floor spans only {span:.1f} m in this photo: the up vector is weakly constrained, treat heights as approximate")

    # ---- ceiling: a horizontal plane at one height, or nothing ---------------------------------------------------
    room_height: Optional[float] = None
    ceil_pts, _ = pooled("ceiling")
    if len(ceil_pts) >= MIN_CEILING_PX:
        hs = height_of(ceil_pts)
        med = float(np.median(hs))
        mad = float(np.median(np.abs(hs - med)))
        if mad > CEILING_MAD_M:
            refused.append(f"ceiling: labelled points are not one horizontal plane (MAD {mad:.2f} m > {CEILING_MAD_M} m)")
        elif not (ROOM_HEIGHT_RANGE_M[0] <= med <= ROOM_HEIGHT_RANGE_M[1]):
            refused.append(f"ceiling: height {med:.2f} m is outside the plausible {ROOM_HEIGHT_RANGE_M} m range")
        else:
            room_height = med
    else:
        notes.append("ceiling not visible in this photo")

    # ---- walls: vertical planes are lines in the horizontal projection -------------------------------------------
    walls: List[Wall] = []
    e1, e2 = _horizontal_basis(up)
    wall_pts, wall_z = pooled("wall")
    if room_height is not None:  # keep points between floor and ceiling (drops hanging clutter)
        hh = height_of(wall_pts)
        keep_h = (hh > -0.1) & (hh < room_height + 0.1)
        wall_pts, wall_z = wall_pts[keep_h], wall_z[keep_h]
    if len(wall_pts) >= MIN_WALL_SUPPORT_PX:
        q_all = np.stack([wall_pts @ e1, wall_pts @ e2], axis=-1)
        band_all = _band(wall_z, PLANE_TOL_M)
        keep = np.arange(len(q_all))
        min_support = max(MIN_WALL_SUPPORT_PX, int(WALL_MIN_SHARE * len(q_all)))
        groups: List[np.ndarray] = []
        for _ in range(MAX_WALLS):
            if len(keep) < MIN_WALL_SUPPORT_PX:
                break
            sub = keep if len(keep) <= MAX_FIT_POINTS else rng.choice(keep, MAX_FIT_POINTS, replace=False)
            got = _ransac_line_2d(q_all[sub], rng, band_all[sub])
            if got is None:
                break
            n2, off = got
            inl = np.abs(q_all[keep] @ n2 - off) <= band_all[keep]
            if int(inl.sum()) < MIN_WALL_SUPPORT_PX:
                break
            groups.append(keep[inl])
            keep = keep[~inl]
        # monocular depth bends a flat wall by a few degrees, which RANSAC reads as two walls: merge near-duplicates
        merged = True
        while merged:
            merged = False
            fin = [_finalise_wall(wall_pts, g, e1, e2, up) for g in groups]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    ni, nj = np.array(fin[i].plane.normal), np.array(fin[j].plane.normal)
                    if ni @ nj > math.cos(math.radians(WALL_DUP_ANGLE_DEG)) and abs(fin[i].plane.d - fin[j].plane.d) < WALL_DUP_OFFSET_M:
                        union = np.concatenate([groups[i], groups[j]])
                        if _finalise_wall(wall_pts, union, e1, e2, up).rms_m <= WALL_MAX_RMS_M:  # one wall only if still flat
                            groups[i] = union
                            del groups[j]
                            merged = True
                            break
                if merged:
                    break
        for g in groups:
            wl = _finalise_wall(wall_pts, g, e1, e2, up)
            if wl.support_px < min_support:
                notes.append(f"dropped a weak wall plane ({wl.support_px} px, under {WALL_MIN_SHARE:.0%} of wall evidence)")
            elif not (WALL_DIST_RANGE_M[0] <= wl.plane.d <= WALL_DIST_RANGE_M[1]):
                notes.append(f"dropped an implausible wall plane at {wl.plane.d:.2f} m")
            elif wl.rms_m > WALL_REJECT_RMS_M:
                notes.append(f"dropped a non-planar wall candidate (rms {wl.rms_m:.2f} m)")
            else:
                walls.append(wl)
        walls.sort(key=lambda x: -x.support_px)
        if not walls:
            refused.append("walls: wall-labelled points did not form any plausible vertical plane")
    else:
        notes.append("no wall visible in this photo")

    # ---- openings: geometry from where each photo's pixel RAYS meet the host wall plane ---------------------------
    openings: List[Opening] = []
    if walls:
        from scipy import ndimage

        found: List[Opening] = []
        for v, pr, valid, rays_cam in kept:
            R, t = v.pose()
            wall_all = v.masks.get("wall", np.zeros(v.depth.shape, bool)) & valid
            for kind in ("window", "door"):
                m = v.masks.get(kind)
                if m is None or not m.any():
                    continue
                lab, count = ndimage.label(m)
                for comp_id in range(1, count + 1):
                    comp = lab == comp_id
                    if int(comp.sum()) < OPENING_MIN_PX:
                        continue
                    ring = ndimage.binary_dilation(comp, iterations=6) & ~comp & wall_all
                    if int(ring.sum()) < 20:
                        refused.append(f"{kind}: no surrounding wall pixels to anchor it to a wall")
                        continue
                    ring_pts = pr[ring]
                    dists = [float(np.median(np.abs(ring_pts @ np.array(wl.plane.normal) + wl.plane.d))) for wl in walls]
                    wi = int(np.argmin(dists))
                    if dists[wi] > OPENING_WALL_DIST_M:
                        refused.append(f"{kind}: its surrounding wall pixels fit no detected wall plane (nearest {dists[wi]:.2f} m)")
                        continue
                    wl = walls[wi]
                    wn = np.array(wl.plane.normal)
                    dirs = rays_cam[comp] @ R.T  # ray directions in the reference frame, from the camera at t
                    denom = dirs @ wn
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sc = -(wl.plane.d + float(wn @ t)) / denom
                    ok = (denom < -1e-6) & (sc > 0)
                    if int(ok.sum()) < OPENING_MIN_PX // 2:
                        continue
                    p = t + dirs[ok] * sc[ok][:, None]
                    sv = p @ np.array(wl.tangent)
                    hv = height_of(p)
                    found.append(Opening(kind, wi, float(np.percentile(sv, 3)), float(np.percentile(sv, 97)),
                                         max(0.0, float(np.percentile(hv, 3))), float(np.percentile(hv, 97)), int(comp.sum())))
        for op in _merge_openings(found):
            lim = WINDOW_LIMITS if op.kind == "window" else DOOR_LIMITS
            wd, ht = op.s_max - op.s_min, op.h_max - op.h_min
            why = (f"width {wd:.2f} m" if not lim["width"][0] <= wd <= lim["width"][1] else
                   f"height {ht:.2f} m" if not lim["height"][0] <= ht <= lim["height"][1] else
                   f"sill {op.h_min:.2f} m above the floor" if op.h_min > lim["max_sill"] else None)
            if why:
                refused.append(f"{op.kind}: implausible {why} (a label error, not a {op.kind}); not reported")
                continue
            openings.append(op)

    if len(kept) > 1:
        notes.append(f"solved from {len(kept)} registered photographs")
    return SurfaceLayout(
        up=tuple(map(float, up)), camera_height_m=float(d), room_height_m=room_height, floor_rms_m=floor_rms,
        walls=tuple(walls), openings=tuple(openings), refused=tuple(refused), notes=tuple(notes),
    )
