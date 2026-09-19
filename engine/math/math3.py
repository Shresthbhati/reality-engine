"""Minimal 3D vector/quaternion/matrix math for rigid body dynamics.

No numpy dependency, consistent with the rest of the foundation layer:
this is 3- and 4-element algebra, not worth a heavy dependency. If a
solver that needs real linear algebra (fluids, structural FEM) shows up
later, that's the point to reconsider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def length_sq(self) -> float:
        return self.dot(self)

    def normalized(self) -> "Vec3":
        n = self.length()
        if n < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return self / n

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x, self.y, self.z))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data: dict) -> "Vec3":
        return Vec3(data["x"], data["y"], data["z"])

    @staticmethod
    def zero() -> "Vec3":
        return Vec3(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Mat3:
    """Row-major 3x3 matrix, used for inertia tensors."""

    rows: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    @staticmethod
    def diagonal(xx: float, yy: float, zz: float) -> "Mat3":
        return Mat3(((xx, 0.0, 0.0), (0.0, yy, 0.0), (0.0, 0.0, zz)))

    @staticmethod
    def identity() -> "Mat3":
        return Mat3.diagonal(1.0, 1.0, 1.0)

    def apply(self, v: Vec3) -> Vec3:
        r = self.rows
        return Vec3(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )

    def transpose(self) -> "Mat3":
        r = self.rows
        return Mat3((
            (r[0][0], r[1][0], r[2][0]),
            (r[0][1], r[1][1], r[2][1]),
            (r[0][2], r[1][2], r[2][2]),
        ))

    def multiply(self, other: "Mat3") -> "Mat3":
        a, b = self.rows, other.rows
        return Mat3(tuple(
            tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        ))

    def inverse_diagonal(self) -> "Mat3":
        """Inverse assuming a diagonal matrix (true for all inertia
        tensors this backend constructs -- box/sphere primitives in
        principal-axis form). Zero entries (locked axes / static bodies)
        invert to zero, matching infinite-inertia convention.
        """
        r = self.rows
        return Mat3.diagonal(
            0.0 if r[0][0] == 0.0 else 1.0 / r[0][0],
            0.0 if r[1][1] == 0.0 else 1.0 / r[1][1],
            0.0 if r[2][2] == 0.0 else 1.0 / r[2][2],
        )


@dataclass(frozen=True)
class Quat:
    """Unit quaternion (w, x, y, z) representing orientation."""

    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def identity() -> "Quat":
        return Quat(1.0, 0.0, 0.0, 0.0)

    def normalized(self) -> "Quat":
        n = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if n < 1e-12:
            return Quat.identity()
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def multiply(self, other: "Quat") -> "Quat":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate(self, v: Vec3) -> Vec3:
        qv = Vec3(self.x, self.y, self.z)
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * (2.0 * self.w)) + (uuv * 2.0)

    def integrate(self, angular_velocity: Vec3, dt: float) -> "Quat":
        """First-order quaternion integration assuming `angular_velocity`
        is expressed in the BODY-LOCAL frame: q' = q + 0.5 * q * omega_quat
        * dt, renormalized. This convention (rather than a world-frame
        omega) is what makes a constant diagonal body-space inertia
        tensor valid for the whole simulation without ever rotating it --
        see rigid/integrator.py, which applies Euler's equations in this
        same body-local frame. Standard first-order approximation; fine
        at gameplay timesteps, not meant for high-precision drift-free
        integration over long horizons.
        """
        omega_quat = Quat(0.0, angular_velocity.x, angular_velocity.y, angular_velocity.z)
        delta = self.multiply(omega_quat)
        return Quat(
            self.w + 0.5 * delta.w * dt,
            self.x + 0.5 * delta.x * dt,
            self.y + 0.5 * delta.y * dt,
            self.z + 0.5 * delta.z * dt,
        ).normalized()

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.w, self.x, self.y, self.z))

    def conjugate(self) -> "Quat":
        """For a unit quaternion this IS the inverse rotation: rotating by
        `q.conjugate()` undoes rotating by `q`. Used by camera extrinsics
        (reconstruction/calibration/camera.py) to go world->camera when
        the stored orientation is camera->world, without introducing a
        second "inverse" concept for a case the conjugate already covers
        exactly for rotations (no scale/shear here, unlike Mat4)."""
        return Quat(self.w, -self.x, -self.y, -self.z)

    def to_dict(self) -> dict:
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data: dict) -> "Quat":
        return Quat(data["w"], data["x"], data["y"], data["z"])
