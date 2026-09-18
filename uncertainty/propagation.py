from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from uncertainty.quantity import UNKNOWN, Uncertain

#: Central-difference step for numeric pose Jacobians (rotation-vector
#: radians and translation meters). Small enough for O(h^2) accuracy,
#: large enough to avoid subtractive cancellation in float64.
#
# This module carries BOTH P10-02 halves:
#   1. Uncertain scalar algebra + numeric-Jacobian pose/point
#      covariance composition (sum/difference/scale/linear_propagate/
#      compose_pose_covariances/transform_point_covariance).
#   2. The analytic per-point covariance chains (Covariance3,
#      depth_to_world_covariance, propagate_point_through_pose,
#      propagate_chain): closed-form Jacobians for the depth ->
#      unprojection -> pose path, finite-difference-verified, UNKNOWN
#      propagates as UNKNOWN, uncertainty never collapsed into
#      confidence.
_STEP = 1e-5


def _derived(value: float, sigma: Optional[float]) -> Uncertain:
    if sigma is None:
        return Uncertain(value=value, sigma=None, basis=UNKNOWN)
    return Uncertain(value=value, sigma=sigma, basis="derived")


def sum(a: Uncertain, b: Uncertain, covariance: float = 0.0) -> Uncertain:
    """a + b with standard error propagation: var = va + vb + 2cov.
    UNKNOWN/None-sigma inputs propagate to an UNKNOWN output."""
    value = a.value + b.value
    if a.is_unknown() or b.is_unknown():
        return Uncertain(value=value, sigma=None, basis=UNKNOWN)
    var = a.sigma ** 2 + b.sigma ** 2 + 2.0 * covariance
    return _derived(value, math.sqrt(max(0.0, var)))


def difference(a: Uncertain, b: Uncertain) -> Uncertain:
    """a - b: variances add (independence approximation documented in
    the module docstring)."""
    value = a.value - b.value
    if a.is_unknown() or b.is_unknown():
        return Uncertain(value=value, sigma=None, basis=UNKNOWN)
    return _derived(value, math.hypot(a.sigma, b.sigma))


def scale(a: Uncertain, k: float) -> Uncertain:
    """k * a: the absolute spread scales by |k|; the RELATIVE spread
    (sigma/value) is invariant — the property the spec names. k = 0
    annihilates the uncertainty (0*a is exactly 0)."""
    value = k * a.value
    if a.is_unknown():
        return Uncertain(value=value, sigma=None, basis=UNKNOWN)
    return _derived(value, abs(k) * a.sigma)


def _symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def linear_propagate(
    jacobian: np.ndarray, covariance: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Covariance of y = J x given cov(x): Sigma_y = J Sigma J^T.
    `covariance=None` (unknown input uncertainty) propagates as None."""
    if covariance is None:
        return None
    j = np.asarray(jacobian, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (j.shape[1], j.shape[1]):
        raise ValueError(
            f"covariance shape {cov.shape} does not match jacobian input "
            f"dimension {j.shape[1]}"
        )
    if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(j)):
        raise ValueError("jacobian and covariance must be finite")
    return _symmetrize(j @ _symmetrize(cov) @ j.T)


def rotate_covariance(
    covariance: Optional[np.ndarray], rotation: np.ndarray
) -> Optional[np.ndarray]:
    """Rotate a 3x3 covariance block: Sigma' = R Sigma R^T. Rotation
    redistributes variance across axes; it never creates or destroys
    information (determinant preserved up to float error)."""
    if covariance is None:
        return None
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {r.shape}")
    return linear_propagate(r, covariance)


def transform_point_covariance(
    covariance: Optional[np.ndarray], transform: RigidTransform
) -> Optional[np.ndarray]:
    """Covariance of R p + t: the rotation rotates the covariance
    block; translation shifts the mean and does NOT affect the spread."""
    return rotate_covariance(covariance, _quat_to_matrix(transform.rotation))


def _quat_to_matrix(q: Quat) -> np.ndarray:
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_to_rotvec(q: Quat) -> np.ndarray:
    """Unit quaternion -> rotation vector (axis * angle). The atan2
    form is numerically robust near angle 0 and 2pi (acos is not)."""
    v = np.array([q.x, q.y, q.z])
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(s, q.w)
    return v / s * angle


__all__ = [
    "sum", "difference", "scale", "linear_propagate",
    "rotate_covariance", "transform_point_covariance",
]


def _perturb(
    t: RigidTransform, d: np.ndarray, from_frame: str, to_frame: str
) -> RigidTransform:
    """The pose t perturbed by d = (rotation-vector, translation): a
    left-multiplied small rotation and an added translation. This is
    the measurement model whose Jacobian the covariance needs."""
    axis_len = float(np.linalg.norm(d[:3]))
    if axis_len > 0:
        axis = d[:3] / axis_len
        half = math.sin(axis_len / 2.0)
        dq = Quat(
            math.cos(axis_len / 2.0), axis[0] * half,
            axis[1] * half, axis[2] * half,
        ).normalized()
    else:
        dq = Quat.identity()
    return RigidTransform(
        from_frame=from_frame,
        to_frame=to_frame,
        rotation=dq.multiply(t.rotation).normalized(),
        translation=Vec3(
            t.translation.x + float(d[3]),
            t.translation.y + float(d[4]),
            t.translation.z + float(d[5]),
        ),
    )


def _pose_6(pose: RigidTransform) -> np.ndarray:
    return np.concatenate([
        _quat_to_rotvec(pose.rotation),
        [pose.translation.x, pose.translation.y, pose.translation.z],
    ])


def _numeric_jacobian(fn, base: np.ndarray, step: float = _STEP) -> np.ndarray:
    """Central-difference Jacobian of a vector function at `base`."""
    base = np.asarray(base, dtype=float)
    n = base.size
    out_dim = np.asarray(fn(base), dtype=float).size
    j = np.zeros((out_dim, n))
    for i in range(n):
        plus = base.copy()
        minus = base.copy()
        plus[i] += step
        minus[i] -= step
        j[:, i] = (
            np.asarray(fn(plus), dtype=float)
            - np.asarray(fn(minus), dtype=float)
        ) / (2.0 * step)
    return j


def compose_pose_covariances(
    t1: RigidTransform,
    covariance1: Optional[np.ndarray],
    t2: RigidTransform,
    covariance2: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """6x6 covariance of T1 composed with T2 given the two input pose
    covariances. Parameterization: [rotation-vector (3), translation
    (3)]. Method: central-difference Jacobians of the EXACT composition
    (`RigidTransform.compose`), propagated as
    Sigma_c = J1 S1 J1^T + J2 S2 J2^T — first-order, inputs treated as
    independent (module docstring, v1 approximation). An unknown input
    covariance (None) propagates as None: a chain with any unknown pose
    uncertainty has unknown output uncertainty — never zero."""
    composed = t1.compose(t2)  # raises on frame mismatch (a real error)
    if covariance1 is None or covariance2 is None:
        return None
    s1 = np.asarray(covariance1, dtype=float)
    s2 = np.asarray(covariance2, dtype=float)
    for name, s in (("covariance1", s1), ("covariance2", s2)):
        if s.shape != (6, 6):
            raise ValueError(f"{name} must be 6x6, got {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"{name} must be finite")

    def f(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
        p1 = _perturb(t1, d1, t1.from_frame, t1.to_frame)
        p2 = _perturb(t2, d2, t2.from_frame, t2.to_frame)
        return _pose_6(p1.compose(p2))

    zero1 = np.zeros(6)
    zero2 = np.zeros(6)
    j1 = _numeric_jacobian(lambda d: f(d, zero2), zero1)
    j2 = _numeric_jacobian(lambda d: f(zero1, d), zero2)
    sigma_c = j1 @ _symmetrize(s1) @ j1.T + j2 @ _symmetrize(s2) @ j2.T
    return _symmetrize(sigma_c)


__all__ = [
    "sum", "difference", "scale", "linear_propagate",
    "rotate_covariance", "transform_point_covariance",
    "compose_pose_covariances",
]



# ====================================================================
# Analytic per-point covariance chains (analytic-Jacobian P10-02 path)
# ====================================================================
class PropagationError(ValueError):
    """Uncertainty propagation refused."""


@dataclass(frozen=True)
class Covariance3:
    """A 3-axis standard deviation (meters) or an explicit UNKNOWN.

    `sigma` is (sx, sy, sz) in the world frame. `is_unknown=True`
    means the inputs to a propagation stage were missing -- the honest
    output is the REASON, not a number.
    """

    sigma: Optional[Tuple[float, float, float]] = None
    is_unknown: bool = False
    reason: str = ""

    def __post_init__(self):
        if self.is_unknown:
            if not self.reason:
                raise PropagationError(
                    "an unknown covariance must carry the reason it is unknown"
                )
            return
        if self.sigma is None:
            raise PropagationError(
                "Covariance3 requires sigma values or is_unknown=True"
            )
        if any(not math.isfinite(s) or s < 0 for s in self.sigma):
            raise PropagationError(
                f"sigma components must be finite and non-negative, got {self.sigma}"
            )

    @staticmethod
    def from_sigmas(sx: float, sy: float, sz: float) -> "Covariance3":
        return Covariance3(sigma=(sx, sy, sz))

    @staticmethod
    def unknown(reason: str) -> "Covariance3":
        return Covariance3(is_unknown=True, reason=reason)

    def require_sigma(self) -> Tuple[float, float, float]:
        if self.is_unknown or self.sigma is None:
            raise PropagationError(
                f"covariance is UNKNOWN ({self.reason or 'no reason recorded'}) "
                "-- asking for its sigma would fabricate certainty"
            )
        return self.sigma

    def to_dict(self) -> dict:
        if self.is_unknown:
            return {"is_unknown": True, "reason": self.reason}
        return {"is_unknown": False, "sigma": list(self.sigma)}

    def to_provenanced(
        self,
        provenance: Provenance,
        confidence: ConfidenceUncertainty,
    ) -> "object":
        """Bridge onto the WorldIR Provenanced record WITHOUT merging
        the two concepts: the covariance rides in `value`, the
        measured confidence record rides in `uncertainty` unchanged."""
        from provenance import Provenanced

        return Provenanced(
            value=self,
            provenance=provenance,
            uncertainty=confidence,
        )


def _sigma_quat_to_matrix(q: Sequence[float]) -> Tuple[Tuple[float, ...], ...]:
    """(w, x, y, z) -> 3x3 rotation matrix rows (camera-to-world), the
    same convention as reconstruction.calibration.camera.quat_to_matrix."""
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _sigma_rotate_into_world(
    cam_axis_sigmas: Sequence[float], R: Sequence[Sequence[float]]
) -> Tuple[float, float, float]:
    """Camera-axes 1-sigma offsets -> world-axes magnitudes. For
    independent per-camera-axis components, the world-axis variance is
    the row-weighted sum: var_w_i = sum_j R_ij^2 var_c_j."""
    out = []
    for i in range(3):
        var = 0.0
        for j in range(3):
            var += R[i][j] ** 2 * cam_axis_sigmas[j] ** 2
        out.append(math.sqrt(var))
    return tuple(out)


def depth_to_world_covariance(
    camera,
    col: float,
    row: float,
    depth: float,
    sigma_depth_m: Optional[float] = None,
    sigma_pixel: Optional[float] = None,
    sigma_pose_rotation_rad: Optional[float] = None,
    sigma_pose_translation_m: Optional[float] = None,
) -> Covariance3:
    """World-frame point covariance for one depth unprojection through
    `camera` (reconstruction.calibration.camera.PinholeCamera).

    Every sigma parameter is OPTIONAL and defaults to None = UNKNOWN:
    supplying none of them (or leaving out the ones with no measured
    calibration) produces an UNKNOWN covariance naming the missing
    stage, never a guessed number. A known subset propagates from the
    stages that HAVE calibration and reports the rest as unknown --
    partial knowledge is not silently promoted to full knowledge.
    """
    intr = camera.intrinsics
    if intr.k1 or intr.k2 or intr.p1 or intr.p2 or intr.k3:
        # The analytic Jacobian is pinhole-only; a distorted camera's
        # pixel Jacobian is NOT (u-cx)/fx and propagating through the
        # pinhole formula would be a wrong number presented as math.
        raise PropagationError(
            "depth_to_world_covariance implements the pinhole Jacobian; "
            "this camera has Brown-Conrady distortion -- propagate via "
            "finite differences of camera.unproject instead"
        )

    missing = []
    if sigma_depth_m is None:
        missing.append("sigma_depth_m (depth calibration)")
    if sigma_pixel is None:
        missing.append("sigma_pixel (sensor calibration)")
    if sigma_pose_rotation_rad is None:
        missing.append("sigma_pose_rotation_rad (pose covariance)")
    if sigma_pose_translation_m is None:
        missing.append("sigma_pose_translation_m (pose covariance)")
    if missing:
        return Covariance3.unknown(
            "no measured calibration for: " + "; ".join(missing)
        )

    if depth <= 0:
        raise PropagationError(f"depth must be positive, got {depth}")
    if min(sigma_depth_m, sigma_pixel, sigma_pose_rotation_rad,
           sigma_pose_translation_m) < 0:
        raise PropagationError("sigma inputs must be non-negative")

    q = camera.extrinsics.rotation
    R = _sigma_quat_to_matrix((q.w, q.x, q.y, q.z))

    # Camera-axes variances at this pixel/depth. Rotation uncertainty
    # is DIRECTIONAL: a small rotation about a camera axis displaces a
    # point by theta x (axis_hat x p_cam) -- not an isotropic lever.
    # Contributions per camera axis (x, y, z), from rotations about the
    # camera's x and y axes (the standard two-angle pose model):
    #   about cam x: displacement (0, d, -y_c) * sigma_theta
    #   about cam y: displacement (d, 0, x_c) * sigma_theta
    x_c = (col - intr.cx) / intr.fx * depth
    y_c = (row - intr.cy) / intr.fy * depth
    st = sigma_pose_rotation_rad
    var_x = (depth / intr.fx * sigma_pixel) ** 2 \
        + (x_c / depth * sigma_depth_m) ** 2 \
        + (depth * st) ** 2
    var_y = (depth / intr.fy * sigma_pixel) ** 2 \
        + (y_c / depth * sigma_depth_m) ** 2 \
        + (depth * st) ** 2
    var_z = sigma_depth_m ** 2 \
        + (y_c * st) ** 2 + (x_c * st) ** 2
    cam_axis = (math.sqrt(var_x), math.sqrt(var_y), math.sqrt(var_z))

    # Rotate camera-axes sigmas into the world frame.
    world_from_cam = _sigma_rotate_into_world(cam_axis, R)

    # Translation uncertainty is already world-frame per axis.
    world = tuple(
        math.sqrt(w ** 2 + sigma_pose_translation_m ** 2)
        for w in world_from_cam
    )
    return Covariance3.from_sigmas(*world)


def propagate_point_through_pose(
    point_cov: Covariance3,
    point: Tuple[float, float, float],
    rotation_quat: Sequence[float],
    translation: Sequence[float],
    sigma_rotation_rad: float,
    sigma_translation_m: float,
    rotation_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> Covariance3:
    """Propagate a point's covariance through one rigid transform
    (rotation quaternion + translation) with pose uncertainty.

    J_point = R (the rotation carries the covariance). Pose translation
    adds per world axis in quadrature. Pose ROTATION uncertainty is
    directional: a small rotation sigma about `rotation_axis` (unit,
    default world z = yaw) displaces the point by
    sigma_theta * (axis_hat x point) -- variance adds per world axis.
    An isotropic lever here would overstate the displacement along the
    rotation axis itself."""
    if point_cov.is_unknown:
        return point_cov  # UNKNOWN in -> UNKNOWN out, reason preserved
    s = point_cov.require_sigma()
    R = _sigma_quat_to_matrix(rotation_quat)
    out_from_point = _sigma_rotate_into_world(s, R)

    axis_len = math.sqrt(
        rotation_axis[0] ** 2 + rotation_axis[1] ** 2 + rotation_axis[2] ** 2
    )
    if axis_len < 1e-12:
        raise PropagationError("rotation_axis must be non-zero")
    ax = tuple(a / axis_len for a in rotation_axis)
    cross = (
        ax[1] * point[2] - ax[2] * point[1],
        ax[2] * point[0] - ax[0] * point[2],
        ax[0] * point[1] - ax[1] * point[0],
    )
    out = tuple(
        math.sqrt(
            a ** 2 + sigma_translation_m ** 2
            + (c * sigma_rotation_rad) ** 2
        )
        for a, c in zip(out_from_point, cross)
    )
    return Covariance3.from_sigmas(*out)


def propagate_chain(
    depth_sigma_m: Optional[float],
    pose: Covariance3,
    registration: Optional[Covariance3],
) -> Covariance3:
    """Combine the pipeline stages' world-frame sigmas in quadrature.

    UNKNOWN anywhere in the chain -> UNKNOWN (with the reason). This
    is the measurement-level entry point: a downstream consumer asking
    "how well is this point known?" gets one covariance that traces
    the whole chain, or the honest reason it cannot."""
    unknowns = []
    if depth_sigma_m is None:
        unknowns.append("depth")
    if pose.is_unknown:
        unknowns.append(f"pose ({pose.reason})")
    if registration is not None and registration.is_unknown:
        unknowns.append(f"registration ({registration.reason})")
    if unknowns:
        return Covariance3.unknown(
            "chain stage(s) without covariance: " + "; ".join(unknowns)
        )

    depth = (depth_sigma_m, depth_sigma_m, depth_sigma_m)
    p = pose.require_sigma()
    reg = registration.require_sigma() if registration is not None else (0.0, 0.0, 0.0)
    combined = tuple(
        math.sqrt(d ** 2 + pp ** 2 + rr ** 2)
        for d, pp, rr in zip(depth, p, reg)
    )
    return Covariance3.from_sigmas(*combined)
