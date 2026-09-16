from __future__ import annotations

import math
from typing import Optional

import numpy as np

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from uncertainty.quantity import UNKNOWN, Uncertain

#: Central-difference step for numeric pose Jacobians (rotation-vector
#: radians and translation meters). Small enough for O(h^2) accuracy,
#: large enough to avoid subtractive cancellation in float64.
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

