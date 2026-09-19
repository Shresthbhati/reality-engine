"""Coordinate library (spec sec 89 COORDINATES).

Every transform stores source_frame, target_frame, a 4x4 matrix,
timestamp, and uncertainty -- never a bare matrix with implicit meaning.
No numpy dependency: a 4x4 homogeneous matrix is 16 floats, and pulling
in a heavy dependency for that at the foundation layer isn't worth it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from provenance import Uncertainty

Mat4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

IDENTITY_MATRIX: Mat4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class Frame(str, Enum):
    """spec sec 89: supported coordinate frames."""

    CAMERA = "camera"
    SENSOR = "sensor"
    SESSION_LOCAL = "session-local"
    BUILDING_LOCAL = "building-local"
    WORLD = "world"
    ENU = "ENU"
    UTM = "UTM"
    WGS84 = "WGS84"
    ECEF = "ECEF"
    ENGINE_LOCAL = "engine-local"


def _mat_mul(a: Mat4, b: Mat4) -> Mat4:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _mat_inverse_rigid(m: Mat4) -> Mat4:
    """Inverse for a rigid transform (rotation + translation, no scale/shear):
    R^-1 = R^T, t' = -R^T * t. Sufficient for the frames this library
    composes today; a general 4x4 inverse can be added if a non-rigid
    transform is introduced later.
    """
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    t = [m[i][3] for i in range(3)]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    t_new = [-sum(rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    return (
        (rt[0][0], rt[0][1], rt[0][2], t_new[0]),
        (rt[1][0], rt[1][1], rt[1][2], t_new[1]),
        (rt[2][0], rt[2][1], rt[2][2], t_new[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def apply_point(m: Mat4, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    hom = [m[i][0] * x + m[i][1] * y + m[i][2] * z + m[i][3] for i in range(3)]
    return (hom[0], hom[1], hom[2])


@dataclass(frozen=True)
class Transform:
    """spec sec 89: source_frame, target_frame, matrix, timestamp, uncertainty."""

    source_frame: Frame
    target_frame: Frame
    matrix: Mat4 = IDENTITY_MATRIX
    timestamp: Optional[float] = None
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    @staticmethod
    def identity(frame: Frame, timestamp: Optional[float] = None) -> "Transform":
        return Transform(source_frame=frame, target_frame=frame, matrix=IDENTITY_MATRIX, timestamp=timestamp)

    def apply(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        return apply_point(self.matrix, point)

    def inverse(self) -> "Transform":
        return Transform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            matrix=_mat_inverse_rigid(self.matrix),
            timestamp=self.timestamp,
            uncertainty=self.uncertainty,
        )

    def then(self, other: "Transform") -> "Transform":
        """Compose self (A->B) with other (B->C) to get A->C."""
        if self.target_frame != other.source_frame:
            raise ValueError(
                f"cannot compose {self.source_frame}->{self.target_frame} "
                f"with {other.source_frame}->{other.target_frame}: frame mismatch"
            )
        return Transform(
            source_frame=self.source_frame,
            target_frame=other.target_frame,
            matrix=_mat_mul(other.matrix, self.matrix),
            timestamp=other.timestamp if other.timestamp is not None else self.timestamp,
        )

    def to_dict(self) -> dict:
        return {
            "source_frame": self.source_frame.value,
            "target_frame": self.target_frame.value,
            "matrix": [list(row) for row in self.matrix],
            "timestamp": self.timestamp,
            "uncertainty": self.uncertainty.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "Transform":
        matrix = tuple(tuple(float(v) for v in row) for row in data["matrix"])
        return Transform(
            source_frame=Frame(data["source_frame"]),
            target_frame=Frame(data["target_frame"]),
            matrix=matrix,
            timestamp=data.get("timestamp"),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
        )


class CoordinateRegistry:
    """Tracks known transforms between frames and resolves chained lookups
    (e.g. camera -> session-local -> world) by composing registered edges.
    """

    def __init__(self):
        self._edges: dict[tuple[Frame, Frame], Transform] = {}

    def register(self, transform: Transform) -> None:
        self._edges[(transform.source_frame, transform.target_frame)] = transform
        self._edges[(transform.target_frame, transform.source_frame)] = transform.inverse()

    def get(self, source: Frame, target: Frame) -> Optional[Transform]:
        if source == target:
            return Transform.identity(source)
        direct = self._edges.get((source, target))
        if direct is not None:
            return direct
        return self._resolve_path(source, target)

    def _resolve_path(self, source: Frame, target: Frame) -> Optional[Transform]:
        # Breadth-first search over registered edges, composing as we go.
        visited = {source}
        frontier: list[tuple[Frame, Transform]] = [(source, Transform.identity(source))]
        while frontier:
            next_frontier: list[tuple[Frame, Transform]] = []
            for frame, acc in frontier:
                for (a, b), edge in self._edges.items():
                    if a != frame or b in visited:
                        continue
                    composed = acc.then(edge)
                    if b == target:
                        return composed
                    visited.add(b)
                    next_frontier.append((b, composed))
            frontier = next_frontier
        return None
