"""Sophus algorithmic extraction: Lie groups SO(3) and SE(3) with Jacobians and Geodesics.

Provides mathematically rigorous Lie group operations for camera pose estimation,
bundle adjustment, and trajectory optimization:
- SO(3): Special Orthogonal Group (3D Rotations)
- SE(3): Special Euclidean Group (Rigid body transformations)
- Exact Taylor series approximations for singularity-free small angles (theta -> 0)
- Left Jacobians, Inverse Jacobians, Adjoint matrices, and Geodesic SLERP/SE(3) interpolation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix (cross-product operator) [v]_\times for a 3-vector."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def vee(omega_hat: np.ndarray) -> np.ndarray:
    """Inverse of hat operator: extracts 3-vector from 3x3 skew-symmetric matrix."""
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], dtype=np.float64)


class SO3:
    """Special Orthogonal group SO(3) represented by a 3x3 orthogonal rotation matrix."""

    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self._matrix = np.eye(3, dtype=np.float64)
        else:
            self._matrix = np.array(matrix, dtype=np.float64).reshape(3, 3)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @classmethod
    def identity(cls) -> SO3:
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def exp(cls, omega: np.ndarray) -> SO3:
        """Exponential map from Lie algebra so(3) to Lie group SO(3) using Rodrigues formula."""
        omega = np.asarray(omega, dtype=np.float64).reshape(3)
        theta_sq = float(np.dot(omega, omega))
        theta = math.sqrt(theta_sq)

        omega_hat = hat(omega)
        omega_hat_sq = omega_hat @ omega_hat

        if theta < 1e-7:
            # Taylor series for small angles
            a = 1.0 - theta_sq / 6.0
            b = 0.5 - theta_sq / 24.0
        else:
            a = math.sin(theta) / theta
            b = (1.0 - math.cos(theta)) / theta_sq

        r_mat = np.eye(3, dtype=np.float64) + a * omega_hat + b * omega_hat_sq
        return cls(r_mat)

    def log(self) -> np.ndarray:
        """Logarithmic map from Lie group SO(3) to Lie algebra so(3)."""
        r = self._matrix
        tr = float(np.trace(r))
        cos_theta = 0.5 * (tr - 1.0)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)

        if theta < 1e-7:
            # Taylor expansion around theta = 0
            factor = 0.5 + (theta ** 2) / 12.0
            diff = r - r.T
            return factor * vee(diff)
        elif math.pi - theta < 1e-5:
            # Near pi singularity: use diagonal entries
            diag = np.diagonal(r)
            k = int(np.argmax(diag))
            v = np.zeros(3, dtype=np.float64)
            v[k] = math.sqrt(max(0.0, (diag[k] + 1.0) / 2.0))
            for j in range(3):
                if j != k and v[k] > 1e-6:
                    v[j] = (r[k, j] + r[j, k]) / (4.0 * v[k])
            v_norm = float(np.linalg.norm(v))
            if v_norm > 1e-12:
                v = v / v_norm
            return theta * v
        else:
            factor = theta / (2.0 * math.sin(theta))
            diff = r - r.T
            return factor * vee(diff)

    @classmethod
    def left_jacobian(cls, omega: np.ndarray) -> np.ndarray:
        """Left Jacobian J_l(omega) mapping tangent vectors to group perturbations."""
        omega = np.asarray(omega, dtype=np.float64).reshape(3)
        theta_sq = float(np.dot(omega, omega))
        theta = math.sqrt(theta_sq)
        omega_hat = hat(omega)

        if theta < 1e-7:
            a = 0.5 - theta_sq / 24.0
            b = 1.0 / 6.0 - theta_sq / 120.0
        else:
            a = (1.0 - math.cos(theta)) / theta_sq
            b = (theta - math.sin(theta)) / (theta_sq * theta)

        return np.eye(3, dtype=np.float64) + a * omega_hat + b * (omega_hat @ omega_hat)

    @classmethod
    def left_jacobian_inverse(cls, omega: np.ndarray) -> np.ndarray:
        """Inverse of Left Jacobian J_l^{-1}(omega)."""
        omega = np.asarray(omega, dtype=np.float64).reshape(3)
        theta_sq = float(np.dot(omega, omega))
        theta = math.sqrt(theta_sq)
        omega_hat = hat(omega)

        if theta < 1e-7:
            b = 1.0 / 12.0 + theta_sq / 720.0
        else:
            b = (1.0 - (theta * math.sin(theta)) / (2.0 * (1.0 - math.cos(theta)))) / theta_sq

        return np.eye(3, dtype=np.float64) - 0.5 * omega_hat + b * (omega_hat @ omega_hat)

    def inverse(self) -> SO3:
        """Inverse rotation (transpose for orthogonal group)."""
        return SO3(self._matrix.T)

    def __matmul__(self, other: Union[SO3, np.ndarray]) -> Union[SO3, np.ndarray]:
        if isinstance(other, SO3):
            return SO3(self._matrix @ other.matrix)
        other_arr = np.asarray(other)
        if other_arr.ndim == 1:
            return self._matrix @ other_arr
        elif other_arr.ndim == 2:
            return (self._matrix @ other_arr.T).T
        return self._matrix @ other_arr


class SE3:
    """Special Euclidean group SE(3) represented by 4x4 rigid transformation matrix."""

    def __init__(self, rotation: Optional[SO3] = None, translation: Optional[np.ndarray] = None):
        self._rotation = rotation if rotation is not None else SO3.identity()
        if translation is None:
            self._translation = np.zeros(3, dtype=np.float64)
        else:
            self._translation = np.asarray(translation, dtype=np.float64).reshape(3)

    @property
    def rotation(self) -> SO3:
        return self._rotation

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @property
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self._rotation.matrix
        mat[:3, 3] = self._translation
        return mat

    @classmethod
    def identity(cls) -> SE3:
        return cls(SO3.identity(), np.zeros(3, dtype=np.float64))

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> SE3:
        mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
        return cls(SO3(mat[:3, :3]), mat[:3, 3])

    @classmethod
    def exp(cls, xi: np.ndarray) -> SE3:
        """Exponential map from se(3) tangent space [upsilon, omega]^T in R^6 to SE(3)."""
        xi = np.asarray(xi, dtype=np.float64).reshape(6)
        upsilon = xi[:3]
        omega = xi[3:]

        so3 = SO3.exp(omega)
        jl = SO3.left_jacobian(omega)
        translation = jl @ upsilon
        return cls(so3, translation)

    def log(self) -> np.ndarray:
        """Logarithmic map from SE(3) to se(3) tangent vector [upsilon, omega]^T in R^6."""
        omega = self._rotation.log()
        jl_inv = SO3.left_jacobian_inverse(omega)
        upsilon = jl_inv @ self._translation
        return np.concatenate([upsilon, omega])

    def inverse(self) -> SE3:
        """Analytical inverse of rigid transform: [R^T, -R^T t]."""
        r_inv = self._rotation.inverse()
        t_inv = - (r_inv.matrix @ self._translation)
        return SE3(r_inv, t_inv)

    def adjoint(self) -> np.ndarray:
        """6x6 Adjoint representation Adj(T) mapping tangent vectors across coordinate frames."""
        r = self._rotation.matrix
        t_hat = hat(self._translation)
        adj = np.zeros((6, 6), dtype=np.float64)
        adj[:3, :3] = r
        adj[:3, 3:] = t_hat @ r
        adj[3:, 3:] = r
        return adj

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an Nx3 point array: p' = R p + t."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            return self._rotation.matrix @ pts + self._translation
        return (self._rotation.matrix @ pts.T).T + self._translation

    @classmethod
    def interpolate(cls, t0: SE3, t1: SE3, alpha: float) -> SE3:
        """Geodesic interpolation between two SE(3) poses (constant screw motion)."""
        alpha = max(0.0, min(1.0, float(alpha)))
        delta = t0.inverse() @ t1
        delta_xi = delta.log()
        scaled_step = cls.exp(alpha * delta_xi)
        return t0 @ scaled_step

    def __matmul__(self, other: Union[SE3, np.ndarray]) -> Union[SE3, np.ndarray]:
        if isinstance(other, SE3):
            new_r = self._rotation @ other.rotation
            new_t = self._rotation.matrix @ other.translation + self._translation
            return SE3(new_r, new_t)
        other_arr = np.asarray(other)
        if other_arr.ndim == 1 and len(other_arr) == 3:
            return self.transform_points(other_arr)
        elif other_arr.ndim == 2 and other_arr.shape[1] == 3:
            return self.transform_points(other_arr)
        elif other_arr.shape == (4, 4):
            return self.matrix @ other_arr
        raise ValueError(f"Unsupported operand shape for SE3 matmul: {other_arr.shape}")
