"""Rigid body transform between two named frames.

Core-layer type: both `uncertainty.propagation` (core) and
`reconstruction.calibration.transforms` (peer) consume this, so it
must live in core where neither layer's import boundary is violated.

`RigidTransform(from_frame, to_frame)`: `p_to = R(p_from) + t`.
`rotation` maps directions from `from_frame` into `to_frame`;
`translation` is expressed in `to_frame` coordinates. Frame names are
explicit and travel with the transform -- composition validates
connectivity so a mis-ordered chain fails loudly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from engine.math import Quat, Vec3


class CalibrationTransformError(ValueError):
    """A transform violated the documented frame contract (empty frame
    name, non-unit quaternion, non-finite translation, frame mismatch on
    compose)."""


@dataclass(frozen=True)
class RigidTransform:
    """A rigid body transform between two NAMED frames:
    ``p_to = rotation(p_from) + translation``.

    `rotation` is a unit `Quat` mapping directions from `from_frame`
    into `to_frame`; `translation` is expressed in `to_frame`. Frame
    names are required and non-empty so every transform carries its
    own semantics; `compose`/`inverse` preserve them (an inverse of
    A->B is B->A). Points transform with translation, directions do
    not -- `apply_direction` exists so callers never have to remember
    which is which.
    """

    from_frame: str
    to_frame: str
    rotation: Quat = field(default_factory=Quat.identity)
    translation: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        for name in (self.from_frame, self.to_frame):
            if not isinstance(name, str) or not name:
                raise CalibrationTransformError("frame names must be non-empty strings")
        norm = self.rotation.normalized()
        if (
            abs(norm.w - self.rotation.w) > 1e-9
            or abs(norm.x - self.rotation.x) > 1e-9
            or abs(norm.y - self.rotation.y) > 1e-9
            or abs(norm.z - self.rotation.z) > 1e-9
        ):
            raise CalibrationTransformError("rotation must be a unit quaternion (within 1e-9)")
        for component in (self.translation.x, self.translation.y, self.translation.z):
            if not math.isfinite(component):
                raise CalibrationTransformError("translation components must be finite")

    def apply(self, point_from: Vec3) -> Vec3:
        """Transform a POINT (translation applies)."""
        return self.rotation.rotate(point_from) + self.translation

    def apply_direction(self, direction_from: Vec3) -> Vec3:
        """Transform a DIRECTION (rotation only -- directions have no
        position, so translation must not apply)."""
        return self.rotation.rotate(direction_from)

    def compose(self, outer: "RigidTransform") -> "RigidTransform":
        """`self` maps A->B and `outer` maps B->C; returns A->C.
        Raises when the frames do not connect -- a mis-ordered chain is
        a real error, not something to silently compute anyway."""
        if outer.from_frame != self.to_frame:
            raise CalibrationTransformError(
                f"cannot compose {self.from_frame}->{self.to_frame} with "
                f"{outer.from_frame}->{outer.to_frame}: outer must start at {self.to_frame!r}"
            )
        return RigidTransform(
            from_frame=self.from_frame,
            to_frame=outer.to_frame,
            rotation=outer.rotation.multiply(self.rotation),
            translation=outer.rotation.rotate(self.translation) + outer.translation,
        )

    def inverse(self) -> "RigidTransform":
        """The exact inverse: A->B becomes B->A (unit-quaternion
        conjugate is the exact inverse rotation)."""
        rotation_inv = self.rotation.conjugate()
        return RigidTransform(
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            rotation=rotation_inv,
            translation=rotation_inv.rotate(self.translation * -1.0),
        )

    def to_dict(self) -> dict:
        return {
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "rotation": self.rotation.to_dict(),
            "translation": self.translation.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "RigidTransform":
        return RigidTransform(
            from_frame=data["from_frame"],
            to_frame=data["to_frame"],
            rotation=Quat.from_dict(data["rotation"]),
            translation=Vec3.from_dict(data["translation"]),
        )
