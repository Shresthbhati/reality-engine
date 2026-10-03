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
    "canonicalize_like_head",
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


#: mean-of-unit-vectors norm required for the camera up-vectors to count as
#: one shared direction (0.9 ~ within 25 degrees on average).
CAMERA_UP_MIN_COHERENCE = 0.9
#: dominant-plane up and camera up must differ by more than this before the
#: camera prior overrides the plane prior.
CAMERA_UP_DISAGREE_DEG = 35.0
#: plane-vs-camera-up angle range in which the dominant plane is treated as
#: a vertical wall and gravity is made perpendicular to it.
CAMERA_UP_VERTICAL_PLANE_DEG = (55.0, 125.0)
#: cameras whose orientations all agree within this carry no pose diversity
#: (identical synthetic rotations say nothing about gravity).
CAMERA_DIVERSITY_MIN_DEG = 1.0


def _camera_up_consensus(poses):
    """(mean world-up unit vector | None, coherence, diverse). Poses hold
    camera-to-world (w,x,y,z); the image-up axis is -Y in COLMAP/OpenCV
    camera coordinates (x right, y down, z forward)."""
    if len(poses) < 3:
        return None, 0.0, False
    mats = [_qvec_to_rotmat(p.rotation) for p in poses]
    ups = np.array([-m[:, 1] for m in mats], dtype=float)
    mean = ups.mean(axis=0)
    coherence = float(np.linalg.norm(mean))
    if coherence < 1e-9:
        return None, 0.0, False
    fwd = np.array([m[:, 2] for m in mats], dtype=float)
    spread = max(
        float(np.degrees(np.arccos(np.clip(float(fwd[i] @ fwd[j]), -1.0, 1.0))))
        for i in range(len(fwd)) for j in range(i + 1, len(fwd))
    )
    return mean / coherence, coherence, spread > CAMERA_DIVERSITY_MIN_DEG


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


def _rotate_result(result: ReconstructionResult, R: np.ndarray) -> ReconstructionResult:
    """Apply one proper rotation to every point and camera pose (positions and orientations)."""
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
    return ReconstructionResult(
        points=rotated_points,
        camera_poses=rotated_poses,
        registration_status=result.registration_status,
    )


#: A candidate shares HEAD's raw (pre-canonical) frame when, after HEAD's canonical rotation is applied, its cameras
#: sit where HEAD's cameras sit: mean displacement of the shared cameras below this fraction of their extent.
#: (Measured on real incremental registration: bundle adjustment moves established cameras by ~0.1-0.6 % of the
#: extent; a full re-solve lands in an arbitrary frame, tens of percent away.)
INHERIT_FRAME_MAX_DISPLACEMENT = 0.05
#: fewer shared cameras than this cannot demonstrate that two reconstructions share a frame
INHERIT_FRAME_MIN_COMMON = 3


def canonicalize_like_head(
    result: ReconstructionResult,
    seed: int,
    head_rotation: Optional[Tuple[Tuple[float, float, float], ...]],
    head_cameras: Optional[dict],
) -> Tuple[ReconstructionResult, FrameCanonicalization]:
    """Canonicalize ``result``, REUSING HEAD's recorded canonical rotation when the candidate demonstrably shares
    HEAD's raw frame.

    Why: ``canonicalize_frame`` re-estimates 'up' from each candidate's own cameras and dominant plane, so two
    successive versions of one world -- even when COLMAP's own frame was preserved by incremental registration --
    land in canonical frames that differ by whatever the up estimate moved (measured on real photographs: 7.6 and
    9.9 degrees about the vertical between V1->V2->V3). The world's visible frame must not wander. A candidate that
    is in a DIFFERENT raw frame (a full re-solve) cannot reuse the rotation and is canonicalized on its own.

    Reuse requires evidence, not trust: apply HEAD's rotation, then require >= INHERIT_FRAME_MIN_COMMON shared
    cameras to land within INHERIT_FRAME_MAX_DISPLACEMENT of HEAD's positions. Otherwise (no HEAD frame recorded,
    too few shared cameras, cameras elsewhere) estimate as before."""
    if head_rotation is not None and head_cameras and result.camera_poses:
        try:
            R = np.array(head_rotation, dtype=float)
            if R.shape == (3, 3) and np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
                common = [p for p in result.camera_poses if p.evidence_id in head_cameras]
                if len(common) >= INHERIT_FRAME_MIN_COMMON:
                    mine = np.array([R @ np.array(p.position) for p in common], dtype=float)
                    theirs = np.array([head_cameras[p.evidence_id] for p in common], dtype=float)
                    extent = float(np.sqrt(((theirs - theirs.mean(0)) ** 2).sum(1).mean()))
                    if extent > 1e-12:
                        shift = float(np.linalg.norm(mine - theirs, axis=1).mean()) / extent
                        if shift <= INHERIT_FRAME_MAX_DISPLACEMENT:
                            record = FrameCanonicalization(
                                rotation=tuple(tuple(float(v) for v in row) for row in R),
                                up_source="inherited_from_head",
                                source_plane_id="",
                                source_inliers=0,
                                note=(f"frame inherited from the current version: {len(common)} shared cameras land "
                                      f"within {shift:.1%} of their established positions after its canonical "
                                      "rotation, so this candidate is in the same frame; up was NOT re-estimated"),
                            )
                            return _rotate_result(result, R), record
        except (TypeError, ValueError):
            pass                                   # a malformed stored frame is never trusted: fall through
    return canonicalize_frame(result, seed)


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

    # Second, independent gravity prior: people hold cameras upright, so
    # the mean of the cameras' image-up axes points at gravity-up whatever
    # the scene is. It is consulted only when it can actually carry
    # evidence (>= 3 cameras, genuinely different orientations, mutually
    # coherent) AND it clearly contradicts the dominant-plane prior --
    # measured: 6 real photos of a building facade made the facade the
    # "floor", laying the whole model on its side.
    up_source = "dominant_plane"
    cam_up, coherence, diverse = _camera_up_consensus(result.camera_poses)
    disagreement_deg = None
    if cam_up is not None:
        disagreement_deg = float(np.degrees(np.arccos(np.clip(float(normal @ cam_up), -1.0, 1.0))))
        if (
            diverse
            and coherence >= CAMERA_UP_MIN_COHERENCE
            and disagreement_deg > CAMERA_UP_DISAGREE_DEG
        ):
            if CAMERA_UP_VERTICAL_PLANE_DEG[0] <= disagreement_deg <= CAMERA_UP_VERTICAL_PLANE_DEG[1]:
                # The dominant plane is roughly a WALL relative to camera-up.
                # Cameras pitch up at buildings, which tilts the mean image-up
                # axis toward the facade normal; gravity is perpendicular to a
                # wall, so remove that component (measured: 16 deg of pitch
                # left a real facade "matching no role within tolerance").
                snapped = cam_up - float(cam_up @ normal) * normal
                normal = snapped / np.linalg.norm(snapped)
                up_source = "camera_up_vectors+vertical_plane"
            else:
                normal = cam_up
                up_source = "camera_up_vectors"

    R = _rotation_from_to(normal, np.array([0.0, 1.0, 0.0]))
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
        raise FrameCanonicalizationError("computed rotation is not proper")

    rotated = _rotate_result(result, R)
    record = FrameCanonicalization(
        rotation=tuple(tuple(float(v) for v in row) for row in R),
        up_source=up_source,
        source_plane_id=dominant.plane_id,
        source_inliers=dominant.inlier_count,
        note=(
            (
                f"frame canonicalized: up = mean camera up-vector ({len(result.camera_poses)} "
                f"cameras, coherence {coherence:.2f}); the dominant-plane prior "
                f"({dominant.plane_id}, {dominant.inlier_count} inliers) was "
                f"{disagreement_deg:.0f} deg away and was overridden; rotated to +Y; "
                "geometric heuristic, not a measured gravity prior"
            )
            if up_source.startswith("camera_up_vectors") else
            (
                f"frame canonicalized: up = dominant plane {dominant.plane_id} "
                f"({dominant.inlier_count} inliers, {len(result.camera_poses)} cameras on "
                "its positive side), rotated to +Y; geometric heuristic, not a "
                "measured gravity prior"
            )
        ),
    )
    return rotated, record
