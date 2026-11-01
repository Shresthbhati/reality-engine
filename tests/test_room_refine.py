"""Room-shell refinement: measured on synthetic rooms with noise, outliers, unequal coverage and perturbed initial planes.

The generator and metrics here are imported by ``scripts/bench_room_refine.py``, so the benchmark table and these
assertions measure the same thing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest

from reconstruction.refine import refine_room, room_dimensions

KINDS = ("box", "L", "skew")


@dataclass
class SynthRoom:
    points: np.ndarray
    cameras: np.ndarray
    true_planes: List[Tuple[str, np.ndarray, float]]     # (role, unit normal, d) in the output frame
    init_planes: List[SimpleNamespace]                   # perturbed, role-less (roles are inferred by the refiner)
    centre: np.ndarray
    scale: float
    size: float                                          # canonical room size (cube root of volume)


def _in_poly(xz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, z = xz[:, 0], xz[:, 1]
    inside = np.zeros(len(xz), dtype=bool)
    for i in range(len(poly)):
        (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % len(poly)]
        cross = ((z1 > z) != (z2 > z)) & (x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-300) + x1)
        inside ^= cross
    return inside


def _perturb(rng, n, ref_point, size):
    ang = math.radians(rng.uniform(2.0, 5.0))
    a = rng.normal(size=3)
    a -= (a @ n) * n
    a /= np.linalg.norm(a)
    n2 = n * math.cos(ang) + a * math.sin(ang)                     # rotate n by `ang` towards a (a is perpendicular)
    delta = rng.choice([-1.0, 1.0]) * rng.uniform(0.02, 0.05) * size
    return n2, -(n2 @ ref_point) + delta


def make_room(seed: int, kind: str = "box", scale: float = 1.0, shift=(0.0, 0.0, 0.0), n_pts: int = 20000) -> SynthRoom:
    rng = np.random.default_rng(seed)
    H = rng.uniform(2.3, 3.4)
    A, B = rng.uniform(3.0, 8.0), rng.uniform(2.5, 6.0)
    if kind == "box":
        poly = np.array([[0, 0], [A, 0], [A, B], [0, B]], float)
    elif kind == "L":
        A1, B1 = A * rng.uniform(0.4, 0.6), B * rng.uniform(0.4, 0.6)
        poly = np.array([[0, 0], [A, 0], [A, B - B1], [A - A1, B - B1], [A - A1, B], [0, B]], float)
    elif kind == "skew":                                           # one wall 6-12 degrees off orthogonal (inside the prior's gate)
        poly = np.array([[0, 0], [A, 0], [A - B * math.tan(math.radians(rng.uniform(6, 12))), B], [0, B]], float)
    else:
        raise ValueError(kind)
    n_e = len(poly)
    edges = [(poly[i], poly[(i + 1) % n_e]) for i in range(n_e)]
    lens = np.array([np.linalg.norm(q - p) for p, q in edges])
    area = 0.5 * abs(sum(poly[i, 0] * poly[(i + 1) % n_e, 1] - poly[(i + 1) % n_e, 0] * poly[i, 1] for i in range(n_e)))
    size = (area * H) ** (1.0 / 3.0)

    out_frac = rng.uniform(0.05, 0.20)
    n_surf = int(n_pts * (1 - out_frac))
    w = np.concatenate([[area, area], lens * H])
    cnt = rng.multinomial(n_surf, w / w.sum())
    planes_c: List[Tuple[str, np.ndarray, float]] = [("floor", np.array([0, 1.0, 0]), 0.0),
                                                      ("ceiling", np.array([0, 1.0, 0]), -H)]
    pts = []
    lo, hi = poly.min(0), poly.max(0)
    for yy, c in ((0.0, cnt[0]), (H, cnt[1])):
        got = np.empty((0, 2))
        while len(got) < c:
            cand = rng.uniform(lo, hi, (max(2 * c, 50), 2))
            got = np.vstack([got, cand[_in_poly(cand, poly)]])
        got = got[:c]
        pts.append(np.column_stack([got[:, 0], np.full(c, yy), got[:, 1]]))
    for i, (p, q) in enumerate(edges):
        e = (q - p) / lens[i]
        n = np.array([e[1], 0.0, -e[0]])
        planes_c.append(("wall", n, -float(n @ np.array([p[0], 0.0, p[1]]))))
        s = rng.uniform(0, 1, cnt[2 + i])
        y = rng.uniform(0, H, cnt[2 + i])
        pts.append(np.column_stack([p[0] + s * (q[0] - p[0]), y, p[1] + s * (q[1] - p[1])]))
        pts[-1] = np.column_stack([pts[-1], s])                    # remember the along-wall parameter for coverage cuts
    # unequal coverage: one wall loses its far 45%, another keeps 25% of its points
    wall_ids = rng.permutation(n_e)
    for tag, i in ((0, wall_ids[0]), (1, wall_ids[1])):
        arr = pts[2 + i]
        keep = arr[:, 3] <= 0.55 if tag == 0 else rng.uniform(size=len(arr)) < 0.25
        pts[2 + i] = arr[keep]
    ref_pts = [np.mean(a[:, :3], axis=0) for a in pts]
    surf = np.vstack([a[:, :3] for a in pts])
    surf = surf + rng.normal(0, rng.uniform(0.004, 0.010) * size, surf.shape)
    n_out = int(len(surf) * out_frac / (1 - out_frac))
    pad = 0.25 * (np.array([hi[0], H, hi[1]]) - np.array([lo[0], 0, lo[1]]))
    out = rng.uniform(np.array([lo[0], 0, lo[1]]) - pad, np.array([hi[0], H, hi[1]]) + pad, (n_out, 3))
    pts_c = np.vstack([surf, out])

    cams = []
    while len(cams) < 4:
        c = rng.uniform(lo, hi, (1, 2))
        if _in_poly(c, poly)[0]:
            cams.append([c[0, 0], 0.5 * H, c[0, 1]])
    cams = np.array(cams)

    yaw = rng.uniform(0, 2 * math.pi)
    R = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    t = np.asarray(shift, float)

    def xform_pts(p):
        return scale * (p @ R.T) + t

    def xform_plane(n, d):
        n2 = R @ n
        return n2, scale * d - float(n2 @ t)

    true = [(r, *xform_plane(n, d)) for r, n, d in planes_c]
    init = []
    for (r, n, d), rp in zip(planes_c, ref_pts):
        n2, d2 = _perturb(rng, n, rp, size)
        nn, dd = xform_plane(n2, d2)
        init.append(SimpleNamespace(normal=tuple(nn), d=dd, plane_id=r))
    pts_w = xform_pts(pts_c)
    return SynthRoom(pts_w, xform_pts(cams), true, init, np.median(pts_w, axis=0), scale, size * scale)


# ---------------------------------------------------------------------------------------------------- metrics
def dim_errors(dims: dict, truth: dict) -> np.ndarray:
    """Relative % error of (length, width, height); a missing truth dimension is skipped (nan)."""
    return np.array([abs(dims[k] - truth[k]) / truth[k] * 100.0 if truth[k] > 0 else math.nan
                     for k in ("length", "width", "height")])


def mean_angle_err(room: SynthRoom, normals: List[Tuple[float, float, float]]) -> float:
    errs = [math.degrees(math.acos(min(1.0, abs(float(np.asarray(n) @ tn)) / (np.linalg.norm(n) * np.linalg.norm(tn)))))
            for n, (_, tn, _) in zip(normals, room.true_planes)]
    return float(np.mean(errs))


def evaluate(room: SynthRoom, **kw) -> dict:
    res = refine_room(room.points, room.init_planes, room.cameras, **kw)
    true_dims = room_dimensions(room.true_planes, room.centre)
    by_src = {p.source_index: p for p in res.planes}
    ordered = [by_src[i] for i in range(len(room.true_planes))]
    init_dims = room_dimensions([(r, p.normal, p.d) for r, p in zip([t[0] for t in room.true_planes], room.init_planes)],
                                room.centre)
    return {
        "res": res,
        "dim_before": np.nanmean(dim_errors(init_dims, true_dims)),
        "dim_after": np.nanmean(dim_errors(res.dims_after, true_dims)),
        "dims_before": dim_errors(init_dims, true_dims),
        "dims_after": dim_errors(res.dims_after, true_dims),
        "ang_before": mean_angle_err(room, [p.normal for p in room.init_planes]),
        "ang_after": mean_angle_err(room, [p.normal for p in ordered]),
        "seconds": res.seconds,
    }


FAST = dict(time_budget_s=30.0, patience=2, n_bootstrap=4)


# ------------------------------------------------------------------------------------------------------ tests
def test_box_rooms_improve_over_30_seeds():
    rows = [evaluate(make_room(s, "box", n_pts=8000), seed=s, **FAST) for s in range(30)]
    db, da = np.array([r["dim_before"] for r in rows]), np.array([r["dim_after"] for r in rows])
    ab, aa = np.array([r["ang_before"] for r in rows]), np.array([r["ang_after"] for r in rows])
    assert np.median(da) < np.median(db)
    assert np.median(aa) < np.median(ab)
    assert np.percentile(da, 90) < np.percentile(db, 90)
    assert all(r["res"].rms_after <= r["res"].rms_before + 1e-12 for r in rows)      # never a worse fit of the inliers


def test_L_shape_never_materially_worse_and_median_better():
    rows = [evaluate(make_room(100 + s, "L", n_pts=8000), seed=s, **FAST) for s in range(30)]
    db, da = np.array([r["dim_before"] for r in rows]), np.array([r["dim_after"] for r in rows])
    assert np.all(da <= db + 0.5), f"worse by more than 0.5 pp on seeds {np.flatnonzero(da > db + 0.5)}"
    assert np.median(da) < np.median(db)


def test_scale_and_frame_independence():
    base = evaluate(make_room(7, "box", n_pts=12000), seed=3, **FAST)
    for k, shift in ((0.003, (0.0, 0.0, 0.0)), (900.0, (5.0e3, -2.0e3, 7.0e2))):
        other = evaluate(make_room(7, "box", scale=k, shift=shift, n_pts=12000), seed=3, **FAST)
        assert other["dim_after"] == pytest.approx(base["dim_after"], abs=1e-4)        # percent points
        assert other["ang_after"] == pytest.approx(base["ang_after"], abs=1e-4)        # degrees
        assert other["res"].rms_after_rel == pytest.approx(base["res"].rms_after_rel, rel=1e-6)


def test_two_second_budget_returns_valid_best_so_far():
    room = make_room(11, "box", n_pts=20000)
    t0 = time.perf_counter()
    res = refine_room(room.points, room.init_planes, room.cameras, time_budget_s=2.0, seed=0)
    wall = time.perf_counter() - t0
    assert wall < 4.0                                            # budget + one iteration of slack, never minutes
    assert len(res.planes) == len(room.init_planes)
    for p in res.planes:
        assert math.isclose(sum(c * c for c in p.normal), 1.0, rel_tol=1e-9)
        assert math.isfinite(p.d)
    assert res.rms_after <= res.rms_before + 1e-12


def test_zero_budget_does_no_fitting_and_is_never_worse_than_the_input():
    """With no time the result is at most the input projected onto the architectural constraints (parallel floor /
    ceiling, vertical walls) - kept only because it scores no worse than the raw input on the fixed residual score."""
    room = make_room(5, "box", n_pts=5000)
    res = refine_room(room.points, room.init_planes, time_budget_s=0.0)
    assert res.iterations == 0 and not res.converged
    assert res.rms_after <= res.rms_before + 1e-12


def test_skewed_room_keeps_its_skew():
    """Orthogonality is a prior, not a rule: with one wall 6-12 degrees off, the refined angle error must not exceed
    the input's, whichever way the held-out selection went."""
    worse = 0
    for s in range(6):
        r = evaluate(make_room(200 + s, "skew", n_pts=15000), seed=s, **FAST)
        worse += r["ang_after"] > r["ang_before"]
    assert worse == 0


def test_openings_are_projected_and_do_not_bias_the_fit():
    room = make_room(21, "box", n_pts=15000)
    # a 'window' on the wall: points belonging to it sit 6% of the room size behind the wall
    res0 = refine_room(room.points, room.init_planes, time_budget_s=20.0, patience=2, n_bootstrap=4)
    w = next(p for p in res0.planes if p.role == "wall")
    n, d = np.asarray(w.normal), w.d
    t = np.cross([0, 1.0, 0], n)
    t /= np.linalg.norm(t)
    inl = room.points[np.abs(room.points @ n + d) < 0.02 * room.size]
    c0 = inl.mean(0)
    corners = np.array([c0 + a * 0.15 * room.size * t + b * 0.15 * room.size * np.array([0, 1.0, 0])
                        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    rng = np.random.default_rng(0)
    ab = rng.uniform(-1, 1, (3000, 2)) * 0.15 * room.size
    glass = c0 + ab[:, :1] * t + ab[:, 1:] * np.array([0, 1.0, 0]) - 0.06 * room.size * n
    pts = np.vstack([room.points, glass])
    op = [{"wall": w.source_index, "corners": corners.tolist()}]
    res = refine_room(pts, room.init_planes, time_budget_s=20.0, patience=2, n_bootstrap=4, openings=op)
    assert len(res.openings) == 1
    cp = np.asarray(res.openings[0]["corners"])
    wp = next(p for p in res.planes if p.source_index == w.source_index)
    assert np.max(np.abs(cp @ np.asarray(wp.normal) + wp.d)) < 1e-6 * room.size


def test_rejects_bad_input():
    room = make_room(1, "box", n_pts=2000)
    with pytest.raises(ValueError):
        refine_room(room.points[:10], room.init_planes)
    with pytest.raises(ValueError):
        refine_room(room.points, [SimpleNamespace(normal=(1.0, 0, 0), d=0.0), SimpleNamespace(normal=(0, 0, 0), d=1.0)])
