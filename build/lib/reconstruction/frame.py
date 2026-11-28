"""Frame canonicalization: SfM's arbitrary output frame -> gravity frame.

COLMAP (any SfM) returns geometry in an arbitrary similarity frame: the
model's +Y is whatever the initial image pair decided, not gravity.
Downstream classification (floor/ceiling/wall by camera-side) and every
consumer that assumes "world up is up" need the model rotated into a
gravity-canonical frame first.

Without EXIF gravity/IMU priors, the deterministic indoor prior is:

  the dominant plane (most RANSAC inliers) of a room capture is the
  floor or the ceiling; cameras stand on the room side of the floor and
  below the ceiling. We choose the sign so the camera centers lie on
  the positive side (cameras are above floors; for ceiling-dominant
  captures this heuristic flips and the note says so), then rotate the
  model so that direction is +Y.

Honesty rules:

  - This is a documented geometric heuristic, not measured gravity:
    the applied rotation, its source plane, and inlier support travel
    in the returned record and must be persisted into world metadata.
  - Rotation only: orthogonal, det=+1, no scale, no translation of
    geometry content -- nothing is fabricated, distances are preserved.
  - Deterministic: plane selection ties break by plane_id; the rotation
    is computed in closed form; no clocks, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from perception.geometry.planes import detect_planes
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)

__all__ = [
    "FrameCanonicalization",
    "FrameCanonicalizationError",
    "canonicalize_frame",
]


class FrameCanonicalizationError(ValueError):
    """Up cannot honestly be estimated from this reconstruction."""


@dataclass(frozen=True)
class FrameCanonicalization:
    """The rotation applied and the evidence behind it."""

    #: 3x3 row-major rotation, model_old -> model_canonical (x_new = R x_old).
    rotation: Tuple[Tuple[float, float, float], ...]
    up_source: str            # "dominant_plane"
    source_plane_id: str
    source_inliers: int
    #: One-line provenance: what was rotated and why.
    note: str

    def to_dict(self) -> dict:
        return {
            "rotation": [list(r) for r in self.rotation],
            "up_source": self.up_source,
            "source_plane_id": self.source_plane_id,
            "source_inliers": self.source_inliers,
            "note": self.note,
        }


def _qvec_to_rotmat(q: Tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rotmat_to_qvec(m: np.ndarray) -> Tuple[float, float, float, float]:
    """Shepperd's method, deterministic branch selection; (w, x, y, z)."""
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    if q[0] < 0:  # canonical hemisphere so the same pose hashes the same
        q = -q
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in q)


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Closed-form rotation taking unit vector `source` to unit vector
    `target` (minimal axis-angle via Rodrigues). Degenerate inputs raise;
    the antiparallel case is resolved with a deterministic perpendicular
    axis."""
    s = source / np.linalg.norm(source)
    t = target / np.linalg.norm(target)
    v = np.cross(s, t)
    c = float(np.dot(s, t))
    if np.linalg.norm(v) < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # antiparallel: rotate pi about any axis perpendicular to s
        axis = np.array([1.0, 0.0, 0.0])
        if abs(s[0]) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis - np.dot(axis, s) * s
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def canonicalize_frame(
    result: ReconstructionResult,
    seed: int,
) -> Tuple[ReconstructionResult, FrameCanonicalization]:
    """Rotate a reconstruction so its dominant plane's camera-facing
    normal is +Y. Returns (rotated_result, record). Raises
    FrameCanonicalizationError when no plane exists to estimate from."""
    if not result.camera_poses:
        raise FrameCanonicalizationError(
            "frame canonicalization needs camera poses (the up prior is "
            "'cameras stand on the room side of the dominant plane')"
        )

    detection = detect_planes(result, seed=seed)
    if not detection.planes:
        raise FrameCanonicalizationError(
            "no planes detected -- cannot estimate up from geometry; "
            "supply a gravity prior instead of guessing"
        )

    # Dominant plane: most inliers, ties by plane_id (deterministic).
    dominant = sorted(detection.planes, key=lambda p: (-p.inlier_count, p.plane_id))[0]

    normal = np.array(dominant.normal, dtype=float)
    normal /= np.linalg.norm(normal)
    d = dominant.d

    # Sign so camera centers sit on the positive side (cameras above the
    # floor / below the ceiling -> the room side).
    cams = np.array([p.position for p in result.camera_poses], dtype=float)
    sides = cams @ normal + d
    if float(np.sum(sides > 0)) < float(np.sum(sides < 0)):
        normal = -normal
        d = -d
    if float(np.sum(sides * np.sign(cams @ normal + d)) == 0) and bool(np.any(sides == 0)):
        raise FrameCanonicalizationError(
            f"dominant plane {dominant.plane_id} contains camera centers -- "
            "no camera-side exists to orient up with"
        )

    R = _rotation_from_to(normal, np.array([0.0, 1.0, 0.0]))
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
        raise FrameCanonicalizationError("computed rotation is not proper")

    rotated_points = [
        ReconstructedPoint(
            position=tuple(R @ np.array(p.position)),
            track_id=p.track_id,
            source_evidence_ids=p.source_evidence_ids,
            uncertainty=p.uncertainty,
        )
        for p in sorted(result.points, key=lambda p: p.track_id)
    ]
    rotated_poses = [
        ReconstructedCameraPose(
            evidence_id=p.evidence_id,
            position=tuple(R @ np.array(p.position)),
            rotation=_rotmat_to_qvec(R @ _qvec_to_rotmat(p.rotation)),
            uncertainty=p.uncertainty,
        )
        for p in sorted(result.camera_poses, key=lambda p: p.evidence_id)
    ]

    rotated = ReconstructionResult(
        points=rotated_points,
        camera_poses=rotated_poses,
        registration_status=result.registration_status,
    )
    record = FrameCanonicalization(
        rotation=tuple(tuple(float(v) for v in row) for row in R),
        up_source="dominant_plane",
        source_plane_id=dominant.plane_id,
        source_inliers=dominant.inlier_count,
        note=(
            f"frame canonicalized: up = dominant plane {dominant.plane_id} "
            f"({dominant.inlier_count} inliers, {len(result.camera_poses)} cameras on "
            "its positive side), rotated to +Y; geometric heuristic, not a "
            "measured gravity prior"
        ),
    )
    return rotated, record
