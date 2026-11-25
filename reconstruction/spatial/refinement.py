"""Geometric refinement from reconstruction output (consume, do not duplicate, the perception engine).

What this does NOT do, deliberately: feature matching, SfM, bundle adjustment, ICP and loop closure already live
in `reconstruction/sfm`, `reconstruction/refine`, `registration/` (COLMAP / KISS-ICP / pose graph, owned by the
perception side). This module only
  * turns an existing registration result (a `RigidTransform` between named frames) into the 2D plan-view
    transform the association step needs, refusing when the frames differ by more than a rotation about the
    vertical axis (a tilted frame cannot be expressed as a plan-view transform);
  * refines each wall's line against the reconstructed POINT CLOUD with a robust (Huber / MAD-rejected) total
    least squares fit and fuses it with the image-observation line by inverse variance -- but only if the two
    agree; if they disagree beyond their sigmas the point-cloud estimate is kept as a CONFLICT alternative and
    the observation line is left unchanged (the disagreement is surfaced, not averaged);
  * reports surface/plane consistency (inlier count, residual) and which evidence (images) supports the points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from engine.core.rigid_transform import RigidTransform

from .association import line_through
from .model import Belief, Conflict, Kind, SpatialEntity, Status
from .uncertainty import consistent, fuse_inverse_variance


def frame_transform_2d(rt: RigidTransform, tilt_tol_rad: float = 0.02) -> tuple[float, float, float]:
    """RigidTransform (p_to = R p_from + t) -> (theta, tx, ty) about the vertical axis. Raises ValueError if the
    rotation tilts the vertical axis by more than `tilt_tol_rad` (no honest plan-view transform exists)."""
    q = rt.rotation
    w, x, y, z = q.w, q.x, q.y, q.z
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    # rotation of the z axis
    zz = 1 - 2 * (x * x + y * y)
    tilt = math.acos(max(-1.0, min(1.0, zz)))
    if tilt > tilt_tol_rad:
        raise ValueError(f"frame {rt.from_frame!r}->{rt.to_frame!r} tilts the vertical axis by {math.degrees(tilt):.2f} deg: "
                         "not a plan-view transform")
    theta = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return theta, float(rt.translation.x), float(rt.translation.y)


@dataclass(frozen=True)
class RefinementConfig:
    band_m: float = 0.30               # perpendicular search band around the current wall line
    along_margin_m: float = 0.20
    min_points: int = 30
    max_iter: int = 6
    reject_k: float = 3.0              # MAD-sigmas
    n_eff_cap: int = 25                # points are correlated (shared depth errors): cap the effective sample size
    point_floor_m: float = 0.01
    k_agree: float = 3.0
    offset_floor_m: float = 0.03


@dataclass
class RefinementReport:
    entities: list[SpatialEntity]
    stats: dict[str, dict] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)


def _robust_line(P: np.ndarray, cfg: RefinementConfig):
    """Huber-weighted TLS with MAD rejection. Returns (centroid, unit dir, inlier mask, rms) or None."""
    keep = np.ones(len(P), bool)
    for _ in range(cfg.max_iter):
        Q = P[keep]
        if len(Q) < cfg.min_points:
            return None
        c = Q.mean(0)
        _, vec = np.linalg.eigh(np.cov((Q - c).T))
        u = vec[:, 1]
        r = (P - c) @ np.array([-u[1], u[0]])
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        s = max(mad, 1e-4)
        new = np.abs(r) <= cfg.reject_k * s
        if (new == keep).all():
            break
        keep = new
    Q = P[keep]
    c = Q.mean(0)
    _, vec = np.linalg.eigh(np.cov((Q - c).T))
    u = vec[:, 1]
    r = (Q - c) @ np.array([-u[1], u[0]])
    return c, u, keep, float(np.sqrt(np.mean(r * r)))


def refine_walls_with_points(
    entities: Sequence[SpatialEntity], points_xyz: np.ndarray, point_evidence: Sequence[Sequence[str]] | None = None,
    cfg: RefinementConfig = RefinementConfig(),
) -> RefinementReport:
    pts = np.asarray(points_xyz, float).reshape(-1, 3)
    out, stats, conflicts = [], {}, []
    for e in entities:
        if e.kind is not Kind.WALL:
            out.append(e)
            continue
        fit = line_through(e.a, e.b)
        t = (pts[:, 0] - fit.c[0]) * fit.u[0] + (pts[:, 1] - fit.c[1]) * fit.u[1]
        d = -(pts[:, 0] - fit.c[0]) * fit.u[1] + (pts[:, 1] - fit.c[1]) * fit.u[0]
        sel = (np.abs(d) <= cfg.band_m) & (t >= fit.t_lo - cfg.along_margin_m) & (t <= fit.t_hi + cfg.along_margin_m)
        if e.z0 is not None and e.z1 is not None:
            sel &= (pts[:, 2] >= e.z0 - 0.2) & (pts[:, 2] <= e.z1 + 0.2)
        idx = np.nonzero(sel)[0]
        res = _robust_line(pts[idx][:, :2], cfg) if len(idx) >= cfg.min_points else None
        if res is None:
            stats[e.entity_id] = {"used": False, "reason": f"{len(idx)} candidate points (< {cfg.min_points}) or no consensus"}
            out.append(e)
            continue
        c, u, keep, rms = res
        if u[0] < 0 or (abs(u[0]) < 1e-12 and u[1] < 0):
            u = -u
        theta_p = math.atan2(u[1], u[0]) % math.pi
        n_in = int(keep.sum())
        n_eff = min(n_in, cfg.n_eff_cap)
        s_d = max(rms / math.sqrt(n_eff), cfg.point_floor_m)
        tin = (pts[idx][:, :2][keep] - c) @ u
        s_th = max(rms / (math.sqrt(n_eff) * max(float(tin.std()), 1e-6)), 1e-5)
        nrm_p = np.array([-u[1], u[0]])
        off_p = float(c @ nrm_p)
        nrm_o = (-fit.u[1], fit.u[0])
        off_o = fit.c[0] * nrm_o[0] + fit.c[1] * nrm_o[1]
        if (nrm_p @ np.array(nrm_o)) < 0:
            off_p = -off_p
        s_d_o = e.beliefs["position"].sigma
        s_th_o = e.beliefs["orientation"].sigma
        ok_off = consistent((off_o, s_d_o), (off_p, s_d), cfg.k_agree, cfg.offset_floor_m)
        dth = ((theta_p - fit.theta + math.pi / 2) % math.pi) - math.pi / 2
        ok_th = consistent((0.0, s_th_o), (dth, s_th), cfg.k_agree, 0.02)
        ev_extra = sorted({x for i in idx[keep] for x in (point_evidence[i] if point_evidence else ())})
        stats[e.entity_id] = {"used": True, "inliers": n_in, "candidates": int(len(idx)), "rms_m": rms,
                              "offset_delta_m": off_p - off_o, "angle_delta_rad": dth, "n_eff": n_eff}
        if ok_off is False or ok_th is False:
            conflicts.append(Conflict(
                "pointcloud_disagrees_with_observations", (e.entity_id,), e.evidence_ids,
                f"{e.entity_id}: point-cloud line differs from the image-observation line by {off_p - off_o:+.3f} m / "
                f"{math.degrees(dth):+.2f} deg beyond their sigmas; observation line kept, point estimate recorded",
                abs(off_p - off_o), None))
            e2 = replace(e, beliefs=dict(e.beliefs), alternatives=list(e.alternatives) + [
                {"kind": "pointcloud_line", "offset_delta_m": off_p - off_o, "angle_delta_rad": dth, "inliers": n_in}])
            p = e2.beliefs["position"]
            e2.beliefs["position"] = Belief(Status.CONFLICTING, p.support, p.value, p.sigma, p.confidence,
                                            p.note + "; disagrees with the reconstructed point cloud")
            out.append(e2)
            continue
        new_off, new_sd = fuse_inverse_variance([(off_o, s_d_o), (off_p, s_d)])
        new_th, new_sth = fuse_inverse_variance([(0.0, s_th_o), (dth, s_th)])
        # rebuild the wall on the fused line: shift by the offset change, rotate by the angle change about its centre
        mid = np.array([(e.a[0] + e.b[0]) / 2, (e.a[1] + e.b[1]) / 2])
        nrm = np.array(nrm_o)
        shift = (new_off - off_o) * nrm if new_off is not None else 0.0
        ang = new_th if new_th is not None else 0.0
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        a2 = tuple(float(v) for v in (R @ (np.array(e.a) - mid) + mid + shift))
        b2 = tuple(float(v) for v in (R @ (np.array(e.b) - mid) + mid + shift))
        pos = e.beliefs["position"]
        sup = tuple(sorted(set(pos.support) | set(ev_extra)))
        e2 = replace(e, a=a2, b=b2, evidence_ids=tuple(sorted(set(e.evidence_ids) | set(ev_extra))),
                     beliefs=dict(e.beliefs), alternatives=list(e.alternatives))
        e2.beliefs["position"] = Belief(Status.GEOMETRICALLY_INFERRED, sup, pos.value, new_sd, pos.confidence,
                                        f"image observations fused with {n_in} point-cloud inliers (rms {rms * 1000:.1f} mm)")
        o = e.beliefs["orientation"]
        e2.beliefs["orientation"] = Belief(Status.GEOMETRICALLY_INFERRED, sup, fit.theta + (new_th or 0.0), new_sth, o.confidence,
                                           "fused with point-cloud plane fit")
        e2.beliefs["geometry"] = Belief(Status.GEOMETRICALLY_INFERRED, sup, rms,
                                        note=f"surface consistency: {n_in}/{len(idx)} points inside the wall band, rms {rms * 1000:.1f} mm")
        out.append(e2)
    return RefinementReport(out, stats, conflicts)


def points_from_result(result) -> tuple[np.ndarray, list[list[str]]]:
    """Adapter for reconstruction.backend.interface.ReconstructionResult (perception output, read-only)."""
    P = np.array([p.position for p in result.points], float) if result.points else np.zeros((0, 3))
    return P, [list(p.source_evidence_ids) for p in result.points]
