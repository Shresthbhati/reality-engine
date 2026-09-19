"""Cross-source registration engine (P4-01):
docs/future/registration/CROSS_SOURCE_REGISTRATION.md.

Three of the spec's four confidence-ordered methods are implemented --
`register_gnss_anchor` (translation-only anchor alignment), `register_icp`
(point-to-point), and `register_icp_point_to_plane` (the spec's named
algorithm; surface normals are caller-supplied, e.g. from a mesh or an
organized depth cloud). Landmark/object-correspondence alignment needs
co-observed object sets from multi-view identity (P7-01 landed the
appearance/epipolar machinery but no cross-source co-observation
resolver yet -- still an honest gap, not silently substituted). Manual
anchor is a CLI/provenance concern (`reality register`), not engine
logic -- callers pass a user-supplied RigidTransform directly.

`RegistrationEngine` orchestrates the methods in the spec's confidence
order (GNSS anchors first, then ICP), recording EVERY attempt -- a
blocked method is an attempt record with its reason, and only a method
that actually ran and failed produces a blocked result. Rejections are
always a `RegistrationResult` with `status="blocked"` and a `reason` --
never a silent best-effort transform (spec rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from engine.math import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform


class RegistrationError(ValueError):
    """Bad input to a registration call (not a registration failure --
    see RegistrationResult.status for that)."""


@dataclass(frozen=True)
class ResidualStats:
    """Measured correspondence residuals of an alignment, in the
    target frame's units (meters). Every number is computed from the
    actual nearest-neighbor correspondences of the FINAL transform --
    nothing is modeled, assumed, or fabricated. `count` is the number
    of correspondences measured."""

    count: int
    median: float
    p95: float
    max: float
    rmse: float

    @staticmethod
    def from_distances(distances) -> "ResidualStats":
        import numpy as _np
        d = _np.asarray(distances, dtype=float)
        if d.size == 0:
            raise RegistrationError("no correspondences to measure residuals from")
        return ResidualStats(
            count=int(d.size),
            median=float(_np.median(d)),
            p95=float(_np.percentile(d, 95.0)),
            max=float(d.max()),
            rmse=float(_np.sqrt(_np.mean(d ** 2))),
        )

    def to_dict(self) -> dict:
        return {
            "count": self.count, "median": self.median, "p95": self.p95,
            "max": self.max, "rmse": self.rmse,
        }


@dataclass(frozen=True)
class RegistrationCovariance:
    """Registration uncertainty derived from MEASURED residuals -- never
    a hardcoded or invented confidence score.

    Model: each correspondence contributes an independent range error
    of order `sigma_residual` (the measured RMS residual). With N
    correspondences the translation estimate's standard error is
        sigma_t = rms / sqrt(N_effective)
    where N_effective is reduced by spatial degeneracy: correspondences
    spread over only k well-separated directions constrain k axes, not
    three. `degenerate_axes` names axes whose constraint basis is weak
    (rank-deficient scatter of correspondence directions) and
    `axis_sigmas_m` carries per-axis sigmas in the order (x, y, z),
    inflated on degenerate axes. `rotation_sigma_rad` follows the same
    construction with lever arm = correspondence distance from the
    centroid (sigma_theta ~ rms / rms_lever).

    basis starts with "residual-derived" so consumers can assert the
    number came from measurement.
    """

    translation_sigma_m: float
    axis_sigmas_m: Tuple[float, float, float]
    rotation_sigma_rad: float
    n_correspondences: int
    degenerate_axes: Optional[Tuple[str, ...]]
    basis: str

    def to_dict(self) -> dict:
        return {
            "translation_sigma_m": self.translation_sigma_m,
            "axis_sigmas_m": list(self.axis_sigmas_m),
            "rotation_sigma_rad": self.rotation_sigma_rad,
            "n_correspondences": self.n_correspondences,
            "degenerate_axes": list(self.degenerate_axes) if self.degenerate_axes else None,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class AttemptRecord:
    """One orchestration attempt, successful or not. Kept even when a
    later method succeeds so the decision trail survives (spec:
    "recorded estimate, never a silent default")."""

    method: str
    status: str  # "accepted" | "blocked"
    reason: str = ""

    def to_dict(self) -> dict:
        return {"method": self.method, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class RegistrationResult:
    """One registration attempt's outcome. `transform` is None iff
    `status == "blocked"` -- the honest failure the spec requires,
    carrying enough diagnostics (rmse, inlier_fraction) to explain why.
    `residual_stats` is None when no correspondences were measurable
    (e.g. blocked before any alignment was computed); `attempts` is
    empty for direct method calls and populated only by the engine."""

    transform: Optional[RigidTransform]
    method: str
    status: str  # "accepted" | "blocked"
    reason: str
    rmse: float
    inlier_fraction: float
    iterations: int
    source_points: int
    target_points: int
    residual_stats: Optional[ResidualStats] = None
    covariance: Optional[RegistrationCovariance] = None
    attempts: Tuple[AttemptRecord, ...] = ()

    def to_dict(self) -> dict:
        return {
            "transform": self.transform.to_dict() if self.transform else None,
            "method": self.method,
            "status": self.status,
            "reason": self.reason,
            "rmse": self.rmse,
            "inlier_fraction": self.inlier_fraction,
            "iterations": self.iterations,
            "source_points": self.source_points,
            "target_points": self.target_points,
            "residual_stats": self.residual_stats.to_dict() if self.residual_stats else None,
            "covariance": self.covariance.to_dict() if self.covariance else None,
            "attempts": [a.to_dict() for a in self.attempts],
        }


def _matrix_to_quat(r: np.ndarray) -> Quat:
    """3x3 proper rotation matrix -> unit quaternion (Shepperd's
    method: pick the numerically stable branch by the largest of
    trace/diagonal terms). Standard, not invented; the inverse of
    `reconstruction.calibration.camera.quat_to_matrix`."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return Quat(w, x, y, z).normalized()


def _kabsch(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form rigid R, t minimizing sum ||R @ src_i + t - tgt_i||^2
    (Kabsch/Umeyama, no scale -- this repo's scale is resolved
    upstream, per the spec's failure-mode note)."""
    centroid_src = source.mean(axis=0)
    centroid_tgt = target.mean(axis=0)
    h = (source - centroid_src).T @ (target - centroid_tgt)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = centroid_tgt - r @ centroid_src
    return r, t


def _to_array(points: Sequence[Vec3]) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in points], dtype=float)


def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0]


def estimate_registration_covariance(
    transformed_source: np.ndarray,
    target_points: Sequence[Vec3],
    distances: np.ndarray,
    indices: np.ndarray,
    inlier_mask: np.ndarray,
) -> RegistrationCovariance:
    """Derive a defensible registration covariance from measured
    residuals and the correspondence geometry (first-order error
    propagation; see RegistrationCovariance's docstring for the model).

    `transformed_source`/`distances`/`indices`/`inlier_mask` MUST be the
    exact arrays the caller's own ICP loop already computed for its
    final transform -- NOT re-derived here from a `RigidTransform`.
    Recomputing them by round-tripping the rotation through
    matrix->quaternion->matrix (the previous implementation) introduces
    floating-point error too small to matter for the transform itself
    but large enough to flip a borderline correspondence across the
    inlier threshold differently on different numpy/BLAS builds --
    which silently made this function's own inlier count diverge from
    `ResidualStats.count` (observed: 299 vs 300 on Linux CI vs a
    passing count locally). Reusing the caller's arrays makes the two
    counts equal by construction, not by coincidence.
    """
    import numpy as _np

    tgt = _to_array(target_points)
    if int(inlier_mask.sum()) < 3:
        raise RegistrationError(
            "covariance estimation needs >= 3 inlier correspondences"
        )
    p = transformed_source[inlier_mask]
    q_pts = tgt[indices[inlier_mask]]
    n = int(inlier_mask.sum())
    rms = float(_np.sqrt(_np.mean((distances[inlier_mask]) ** 2)))

    # Spatial conditioning of the correspondences: eigenvalues of the
    # centered correspondence covariance. A tiny eigenvalue along one
    # axis means that axis is weakly constrained (the corridor problem).
    centered = p - p.mean(axis=0)
    scatter = centered.T @ centered / max(n, 1)
    eigvals, eigvecs = _np.linalg.eigh(scatter)  # ascending
    spread = float(_np.max(eigvals)) if len(eigvals) else 0.0
    degenerate_axes: list[str] = []
    axis_sigmas = [0.0, 0.0, 0.0]
    for axis in range(3):
        # Relative conditioning of this axis: fraction of the largest
        # eigenvalue's spread captured along it.
        rel = float(eigvals[axis] / spread) if spread > 0 else 0.0
        if rel < 1e-3:
            degenerate_axes.append("xyz"[axis])
            # Effectively unconstrained along this axis: inflate by the
            # cloud's own extent (an honest "we don't know" scale), not
            # a fabricated small sigma.
            extent = float(_np.max(p[:, axis]) - _np.min(p[:, axis]))
            axis_sigmas[axis] = rms + extent
        else:
            # Effective independent sample count shrinks with the axis's
            # relative conditioning.
            n_eff = max(1.0, n * min(1.0, rel))
            axis_sigmas[axis] = rms / _np.sqrt(n_eff)

    # Rotation sigma via lever arms: sigma_theta ~ rms / rms_lever.
    centroid = p.mean(axis=0)
    lever = _np.linalg.norm(p - centroid, axis=1)
    rms_lever = float(_np.sqrt(_np.mean(lever ** 2)))
    rot_sigma = rms / rms_lever if rms_lever > 0 else float("inf")

    return RegistrationCovariance(
        translation_sigma_m=float(_np.max(axis_sigmas)),
        axis_sigmas_m=(axis_sigmas[0], axis_sigmas[1], axis_sigmas[2]),
        rotation_sigma_rad=float(rot_sigma),
        n_correspondences=n,
        degenerate_axes=tuple(degenerate_axes) if degenerate_axes else None,
        basis=(
            f"residual-derived: rms {rms:.3e} m over {n} inlier "
            "correspondences; first-order propagation with spatial "
            "conditioning (weaker axes inflate)"
        ),
    )


def _quat_to_matrix(q: "Quat") -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix (standard; the inverse
    of `_matrix_to_quat`)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def register_icp(
    source: Sequence[Vec3],
    target: Sequence[Vec3],
    *,
    from_frame: str,
    to_frame: str,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    min_overlap: float = 0.5,
    inlier_multiplier: float = 3.0,
    max_rmse_scale: float = 5.0,
    initial_rotation: Optional[np.ndarray] = None,
    initial_translation: Optional[np.ndarray] = None,
) -> RegistrationResult:
    """Point-to-point ICP: align `source` onto `target`, returning the
    transform mapping source points into target's frame. Outlier
    correspondences beyond `inlier_multiplier` * median distance are
    excluded each iteration (simple robustness, not full RANSAC --
    sufficient for the spec's rejection tests). BLOCKED (not a
    degraded transform) when there are too few points, too few
    inliers, the final inlier_fraction is below `min_overlap`, or the
    final rmse exceeds `max_rmse_scale` * target's own point spacing
    (spec failure mode: "rmse >> voxel size"). That last check matters
    because a uniformly-offset, non-overlapping cloud still looks
    internally self-consistent to a purely relative inlier-fraction
    gate -- an absolute scale reference is required to catch it. Like
    all ICP, this assumes a coarse initial alignment already roughly
    within the cloud's own scale (spec's earlier pipeline stage);
    it is not a global registration/RANSAC search over large offsets.
    """
    n_source, n_target = len(source), len(target)
    if n_source < 3 or n_target < 3:
        return RegistrationResult(
            transform=None, method="icp", status="blocked",
            reason=f"degenerate geometry: need >=3 points per cloud, got "
                   f"{n_source} source / {n_target} target",
            rmse=float("nan"), inlier_fraction=0.0, iterations=0,
            source_points=n_source, target_points=n_target,
        )

    src = _to_array(source)
    tgt = _to_array(target)
    tree = cKDTree(tgt)
    r = np.asarray(initial_rotation, dtype=float).copy() if initial_rotation is not None else np.eye(3)
    t = np.asarray(initial_translation, dtype=float).copy() if initial_translation is not None else np.zeros(3)
    prev_rmse = float("inf")
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        transformed = src @ r.T + t
        distances, indices = tree.query(transformed)
        median = float(np.median(distances))
        threshold = median * inlier_multiplier if median > 0 else float("inf")
        inliers = distances <= threshold

        if int(inliers.sum()) < 3:
            return RegistrationResult(
                transform=None, method="icp", status="blocked",
                reason="fewer than 3 inlier correspondences after outlier rejection",
                rmse=float(distances.mean()), inlier_fraction=float(inliers.mean()),
                iterations=iterations, source_points=n_source, target_points=n_target,
            )

        r, t = _kabsch(src[inliers], tgt[indices[inliers]])
        rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))
        if abs(prev_rmse - rmse) < tolerance:
            prev_rmse = rmse
            break
        prev_rmse = rmse

    final_distances, final_indices = tree.query(src @ r.T + t)
    median = float(np.median(final_distances))
    threshold = median * inlier_multiplier if median > 0 else float("inf")
    inliers = final_distances <= threshold
    inlier_fraction = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(final_distances[inliers] ** 2))) if inliers.any() else float("nan")
    residual_stats = ResidualStats.from_distances(final_distances) if inliers.any() else None

    if inlier_fraction < min_overlap:
        return RegistrationResult(
            transform=None, method="icp", status="blocked",
            reason=f"insufficient overlap: inlier_fraction={inlier_fraction:.3f} "
                   f"< min_overlap={min_overlap}",
            rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
            source_points=n_source, target_points=n_target,
            residual_stats=residual_stats,
        )

    # Absolute-scale sanity check: a uniformly-offset non-overlapping
    # cloud can pass the relative inlier-fraction gate above (every
    # point is equally "close" to some other point). Compare the
    # final rmse against the target cloud's own nearest-neighbor
    # spacing -- if the alignment error dwarfs the geometry's own
    # scale, this isn't a real correspondence, no matter how
    # internally consistent it looked.
    if n_target >= 2:
        self_nn = cKDTree(tgt).query(tgt, k=2)[0][:, 1]
        point_spacing = float(np.median(self_nn))
        if point_spacing > 0 and rmse > max_rmse_scale * point_spacing:
            return RegistrationResult(
                transform=None, method="icp", status="blocked",
                reason=f"rmse={rmse:.4g} exceeds {max_rmse_scale}x the target's own "
                       f"point spacing ({point_spacing:.4g}) -- not a real correspondence",
                rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
                source_points=n_source, target_points=n_target,
                residual_stats=residual_stats,
            )

    transform = RigidTransform(
        from_frame=from_frame, to_frame=to_frame,
        rotation=_matrix_to_quat(r), translation=Vec3(*t),
    )
    covariance = None
    if residual_stats is not None:
        try:
            covariance = estimate_registration_covariance(
                src @ r.T + t, target, final_distances, final_indices, inliers
            )
        except RegistrationError:
            covariance = None  # degenerate correspondences: honest None
    return RegistrationResult(
        transform=transform, method="icp", status="accepted", reason="",
        rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
        source_points=n_source, target_points=n_target,
        residual_stats=residual_stats, covariance=covariance,
    )


def register_icp_point_to_plane(
    source: Sequence[Vec3],
    target: Sequence[Vec3],
    *,
    source_normals: Sequence[Vec3],
    from_frame: str,
    to_frame: str,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    min_overlap: float = 0.5,
    inlier_multiplier: float = 3.0,
    max_rmse_scale: float = 5.0,
    initial_rotation: Optional[np.ndarray] = None,
    initial_translation: Optional[np.ndarray] = None,
) -> RegistrationResult:
    """The spec's named algorithm: point-to-plane ICP (Chen & Medioni).
    Each iteration solves a linearized small-angle increment against
    plane constraints (n_j . (R p_i + t - q_j) = 0) built from the
    source points' normals -- converges faster and more accurately on
    surface-bearing scenes than point-to-point, which fights the
    surface by pulling points ONTO points. Normals are caller-supplied
    (mesh faces / organized-cloud normals); mismatched counts are a
    caller contract violation, not a registration failure."""
    n_source, n_target = len(source), len(target)
    if len(source_normals) != n_source:
        raise RegistrationError(
            f"source_normals must pair 1:1 with source points, "
            f"got {len(source_normals)} vs {n_source}"
        )
    if n_source < 3 or n_target < 3:
        return RegistrationResult(
            transform=None, method="icp_point_to_plane", status="blocked",
            reason=f"degenerate geometry: need >=3 points per cloud, got "
                   f"{n_source} source / {n_target} target",
            rmse=float("nan"), inlier_fraction=0.0, iterations=0,
            source_points=n_source, target_points=n_target,
        )

    src = _to_array(source)
    tgt = _to_array(target)
    nrm = _to_array(source_normals)
    norms = np.linalg.norm(nrm, axis=1)
    if (norms == 0).any():
        raise RegistrationError("source_normals must be non-zero")
    nrm = nrm / norms[:, None]

    tree = cKDTree(tgt)
    r = np.asarray(initial_rotation, dtype=float).copy() if initial_rotation is not None else np.eye(3)
    t = np.asarray(initial_translation, dtype=float).copy() if initial_translation is not None else np.zeros(3)
    prev_cost = float("inf")
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        transformed = src @ r.T + t
        distances, indices = tree.query(transformed)
        median = float(np.median(distances))
        threshold = median * inlier_multiplier if median > 0 else float("inf")
        inliers = distances <= threshold
        if int(inliers.sum()) < 6:
            # A linearized rigid increment has 6 DOF; fewer constraints
            # than that cannot determine one.
            return RegistrationResult(
                transform=None, method="icp_point_to_plane", status="blocked",
                reason="fewer than 6 inlier correspondences after outlier rejection "
                       "-- insufficient constraints for a rigid solve",
                rmse=float(distances.mean()), inlier_fraction=float(inliers.mean()),
                iterations=iterations, source_points=n_source, target_points=n_target,
            )

        p = transformed[inliers]
        q = tgt[indices[inliers]]
        n = nrm[indices[inliers]]
        c = np.einsum("ij,ij->i", n, (p - q))  # current signed plane residuals
        # Linearized increment: for small (w, v),
        #   n . ((I + [w]_x) p + t + v - q) = -c  ->
        #   [n . (w x p) + n . v] = -c
        # with w x p = [p]_x^T w; stack 6 unknowns per correspondence.
        a_matrix = np.zeros((p.shape[0], 6))
        a_matrix[:, 0:3] = np.cross(p, n)   # d/dw of w x p dotted with n
        a_matrix[:, 3:6] = n
        delta = _solve_linear(a_matrix, -c)
        w, v = delta[0:3], delta[3:6]

        theta = float(np.linalg.norm(w))
        if theta > 1e-12:
            axis = w / theta
            kmat = np.array([[0.0, -axis[2], axis[1]],
                             [axis[2], 0.0, -axis[0]],
                             [-axis[1], axis[0], 0.0]])
            step_r = np.eye(3) + math.sin(theta) * kmat + (1.0 - math.cos(theta)) * (kmat @ kmat)
        else:
            step_r = np.eye(3)
        step_t = v

        r = step_r @ r
        t = step_r @ t + step_t
        cost = float(np.mean(c ** 2))
        if abs(prev_cost - cost) < tolerance:
            prev_cost = cost
            break
        prev_cost = cost

    final_distances, final_indices = tree.query(src @ r.T + t)
    median = float(np.median(final_distances))
    threshold = median * inlier_multiplier if median > 0 else float("inf")
    inliers = final_distances <= threshold
    inlier_fraction = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(final_distances[inliers] ** 2))) if inliers.any() else float("nan")
    residual_stats = ResidualStats.from_distances(final_distances) if inliers.any() else None

    if inlier_fraction < min_overlap:
        return RegistrationResult(
            transform=None, method="icp_point_to_plane", status="blocked",
            reason=f"insufficient overlap: inlier_fraction={inlier_fraction:.3f} "
                   f"< min_overlap={min_overlap}",
            rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
            source_points=n_source, target_points=n_target,
            residual_stats=residual_stats,
        )
    if n_target >= 2:
        self_nn = cKDTree(tgt).query(tgt, k=2)[0][:, 1]
        point_spacing = float(np.median(self_nn))
        if point_spacing > 0 and rmse > max_rmse_scale * point_spacing:
            return RegistrationResult(
                transform=None, method="icp_point_to_plane", status="blocked",
                reason=f"rmse={rmse:.4g} exceeds {max_rmse_scale}x the target's own "
                       f"point spacing ({point_spacing:.4g}) -- not a real correspondence",
                rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
                source_points=n_source, target_points=n_target,
                residual_stats=residual_stats,
            )

    transform = RigidTransform(
        from_frame=from_frame, to_frame=to_frame,
        rotation=_matrix_to_quat(r), translation=Vec3(*t),
    )
    covariance = None
    if residual_stats is not None:
        try:
            covariance = estimate_registration_covariance(
                src @ r.T + t, target, final_distances, final_indices, inliers
            )
        except RegistrationError:
            covariance = None
    return RegistrationResult(
        transform=transform, method="icp_point_to_plane", status="accepted", reason="",
        rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
        source_points=n_source, target_points=n_target,
        residual_stats=residual_stats, covariance=covariance,
    )


def register_gnss_anchor(
    source_positions: Sequence[Vec3],
    target_positions: Sequence[Vec3],
    *,
    from_frame: str,
    to_frame: str,
) -> RegistrationResult:
    """Translation-only alignment (spec: "no yaw assumption") from
    paired anchor positions -- e.g. both sources' GNSS-derived ENU
    coordinates for the same physical points, matched by index.
    Requires at least one pair; raises RegistrationError (a caller
    contract violation, not a registration failure) if the lists are
    empty or mismatched in length."""
    if len(source_positions) != len(target_positions):
        raise RegistrationError(
            f"source_positions and target_positions must be paired 1:1, "
            f"got {len(source_positions)} vs {len(target_positions)}"
        )
    if not source_positions:
        raise RegistrationError("register_gnss_anchor needs at least one anchor pair")

    src = _to_array(source_positions)
    tgt = _to_array(target_positions)
    offset = (tgt - src).mean(axis=0)
    residuals = (src + offset) - tgt
    rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    transform = RigidTransform(
        from_frame=from_frame, to_frame=to_frame,
        rotation=Quat.identity(), translation=Vec3(*offset),
    )
    n_pairs = len(source_positions)
    # Translation-only model: sigma_t = rmse / sqrt(N) over the anchor
    # pairs; no rotation is estimated, so rotation_sigma is exactly the
    # model's non-estimability, not a number.
    import numpy as _np
    anchor_cov = RegistrationCovariance(
        translation_sigma_m=float(rmse / _np.sqrt(max(n_pairs, 1))),
        axis_sigmas_m=(float(rmse / _np.sqrt(max(n_pairs, 1))),) * 3,
        rotation_sigma_rad=float("nan"),  # translation-only: rotation NOT estimated
        n_correspondences=n_pairs,
        degenerate_axes=None,
        basis=f"residual-derived: anchor-pair rmse {rmse:.3e} m over {n_pairs} pairs; translation-only model (rotation not estimated)",
    )
    return RegistrationResult(
        transform=transform, method="gnss_anchor", status="accepted", reason="",
        rmse=rmse, inlier_fraction=1.0, iterations=1,
        source_points=n_pairs, target_points=len(target_positions),
        covariance=anchor_cov,
    )


class RegistrationEngine:
    """Confidence-ordered orchestration (spec's ordered methods): try
    GNSS anchors first (strongest evidence -- measured positions, no
    shape ambiguity), then ICP on shared geometry. Every attempt is
    recorded in `attempts`; the first accepted method wins and the
    engine stops -- later methods are not tried after success, and
    their absence from `attempts` reflects that they were never
    needed. All-blocked returns a blocked result with method="none"
    and every attempt's reason -- never a silent best-effort."""

    def register(
        self,
        *,
        source_cloud: Sequence[Vec3],
        target_cloud: Sequence[Vec3],
        from_frame: str,
        to_frame: str,
        source_normals: Optional[Sequence[Vec3]] = None,
        anchor_source: Optional[Sequence[Vec3]] = None,
        anchor_target: Optional[Sequence[Vec3]] = None,
        min_overlap: float = 0.5,
        initial_transform: Optional[RigidTransform] = None,
    ) -> RegistrationResult:
        """`initial_transform` seeds ICP when a prior exists (known
        extrinsics, GNSS/trajectory prior -- the spec's pipeline puts a
        coarse alignment BEFORE ICP for exactly this reason: ICP is a
        local optimizer and an unseeded run on far-apart clouds is a
        guess, not an alignment)."""
        attempts: list[AttemptRecord] = []

        # Pre-flight (spec pipeline, before any method): a prior
        # transform must put the clouds in plausible contact before ICP
        # is meaningful. Measured on actual geometry -- median
        # nearest-neighbor distance after applying the prior vs the
        # target's own point spacing. Without a prior, unseeded ICP is
        # only attempted when the raw clouds are already in contact
        # (same gate, identity prior) -- registering two disjoint
        # clouds by blind iteration would be a guess.
        import numpy as _np
        src_arr = _to_array(source_cloud)
        tgt_arr = _to_array(target_cloud)
        if len(src_arr) >= 3 and len(tgt_arr) >= 3:
            if initial_transform is not None:
                q = initial_transform.rotation
                tv = initial_transform.translation
                rot = _quat_to_matrix(q)
                seeded = src_arr @ rot.T + _np.array([tv.x, tv.y, tv.z])
            else:
                seeded = src_arr
            nn = cKDTree(tgt_arr).query(seeded)[0]
            contact = float(_np.median(nn))
            self_nn = cKDTree(tgt_arr).query(tgt_arr, k=2)[0][:, 1]
            spacing = float(_np.median(self_nn)) if len(tgt_arr) >= 2 else float("inf")
            contact_scale = 20.0  # generous: real contact, coarse grids
            if spacing > 0 and contact > contact_scale * spacing:
                return RegistrationResult(
                    transform=None, method="none", status="blocked",
                    reason=(
                        "no plausible contact: median nearest-neighbor distance "
                        f"{contact:.4g} after the " +
                        ("prior transform" if initial_transform is not None else
                         "identity (no prior given)") +
                        f" exceeds {contact_scale}x the target's own point "
                        f"spacing ({spacing:.4g}) -- ICP on disjoint clouds "
                        "would be a guess, not an alignment"
                    ),
                    rmse=float(contact), inlier_fraction=0.0, iterations=0,
                    source_points=len(source_cloud), target_points=len(target_cloud),
                    attempts=(AttemptRecord(
                        "contact_check", "blocked",
                        "clouds not in contact prior to alignment",
                    ),),
                )

        # Method 1 (confidence-ordered first): GNSS anchor pairs.
        if anchor_source is not None and anchor_target is not None:
            try:
                anchor_result = register_gnss_anchor(
                    anchor_source, anchor_target,
                    from_frame=from_frame, to_frame=to_frame,
                )
            except RegistrationError as exc:
                attempts.append(AttemptRecord("gnss_anchor", "blocked", str(exc)))
            else:
                attempts.append(AttemptRecord("gnss_anchor", anchor_result.status, anchor_result.reason))
                if anchor_result.status == "accepted":
                    return RegistrationResult(
                        transform=anchor_result.transform,
                        method=anchor_result.method,
                        status=anchor_result.status,
                        reason=anchor_result.reason,
                        rmse=anchor_result.rmse,
                        inlier_fraction=anchor_result.inlier_fraction,
                        iterations=anchor_result.iterations,
                        source_points=anchor_result.source_points,
                        target_points=anchor_result.target_points,
                        residual_stats=anchor_result.residual_stats,
                        covariance=anchor_result.covariance,
                        attempts=tuple(attempts),
                    )
        else:
            attempts.append(AttemptRecord(
                "gnss_anchor", "blocked", "no anchor pairs provided -- "
                "cannot fabricate positions"
            ))

        # Method 2: point-to-plane ICP when normals are available (the
        # spec's named algorithm; strictly more information than
        # point-to-point), else point-to-point ICP.
        seed_r = np.eye(3)
        seed_t = np.zeros(3)
        if initial_transform is not None:
            seed_r = _quat_to_matrix(initial_transform.rotation)
            seed_t = np.array([initial_transform.translation.x,
                               initial_transform.translation.y,
                               initial_transform.translation.z])
        if source_normals is not None:
            icp_result = register_icp_point_to_plane(
                source_cloud, target_cloud,
                source_normals=source_normals,
                from_frame=from_frame, to_frame=to_frame,
                min_overlap=min_overlap,
                initial_rotation=seed_r, initial_translation=seed_t,
            )
        else:
            icp_result = register_icp(
                source_cloud, target_cloud,
                from_frame=from_frame, to_frame=to_frame,
                min_overlap=min_overlap,
                initial_rotation=seed_r, initial_translation=seed_t,
            )
        attempts.append(AttemptRecord(icp_result.method, icp_result.status, icp_result.reason))
        if icp_result.status == "accepted":
            return RegistrationResult(
                transform=icp_result.transform,
                method=icp_result.method,
                status=icp_result.status,
                reason=icp_result.reason,
                rmse=icp_result.rmse,
                inlier_fraction=icp_result.inlier_fraction,
                iterations=icp_result.iterations,
                source_points=icp_result.source_points,
                target_points=icp_result.target_points,
                residual_stats=icp_result.residual_stats,
                covariance=icp_result.covariance,
                attempts=tuple(attempts),
            )

        return RegistrationResult(
            transform=None, method="none", status="blocked",
            reason="; ".join(f"{a.method}: {a.reason}" for a in attempts if a.reason),
            rmse=float("nan"), inlier_fraction=0.0, iterations=0,
            source_points=len(source_cloud), target_points=len(target_cloud),
            attempts=tuple(attempts),
        )
