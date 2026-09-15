"""Cross-source registration engine (P4-01):
docs/future/registration/CROSS_SOURCE_REGISTRATION.md.

Two of the spec's four confidence-ordered methods are implemented --
`register_gnss_anchor` (translation-only anchor alignment) and
`register_icp` (point-cloud alignment). The other two need
capabilities this repo doesn't have yet: landmark/object-correspondence
alignment needs multi-view object identity (P2, not built), and manual
anchor is a CLI/provenance concern (`reality register`), not engine
logic -- there's nothing to implement here beyond accepting a
caller-supplied RigidTransform, which callers can already construct
directly. Both are skipped rather than stubbed.

Deviation from the spec's "point-to-plane ICP": this module does
POINT-TO-POINT ICP (Kabsch/Umeyama per iteration). Point-to-plane
needs surface normals, which nothing upstream produces for a bare
point list; point-to-point is the correct algorithm for the data this
repo actually has, and meets the spec's acceptance criterion (known
rigid offset recovered within tolerance). Upgrade to point-to-plane if
a normal-bearing surface representation becomes available and
point-to-point measurably underperforms.

Rejections are always a `RegistrationResult` with `status="blocked"`
and a `reason` -- never a silent best-effort transform (spec rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform


class RegistrationError(ValueError):
    """Bad input to a registration call (not a registration failure --
    see RegistrationResult.status for that)."""


@dataclass(frozen=True)
class RegistrationResult:
    """One registration attempt's outcome. `transform` is None iff
    `status == "blocked"` -- the honest failure the spec requires,
    carrying enough diagnostics (rmse, inlier_fraction) to explain why."""

    transform: Optional[RigidTransform]
    method: str
    status: str  # "accepted" | "blocked"
    reason: str
    rmse: float
    inlier_fraction: float
    iterations: int
    source_points: int
    target_points: int

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
    r = np.eye(3)
    t = np.zeros(3)
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

    if inlier_fraction < min_overlap:
        return RegistrationResult(
            transform=None, method="icp", status="blocked",
            reason=f"insufficient overlap: inlier_fraction={inlier_fraction:.3f} "
                   f"< min_overlap={min_overlap}",
            rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
            source_points=n_source, target_points=n_target,
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
            )

    transform = RigidTransform(
        from_frame=from_frame, to_frame=to_frame,
        rotation=_matrix_to_quat(r), translation=Vec3(*t),
    )
    return RegistrationResult(
        transform=transform, method="icp", status="accepted", reason="",
        rmse=rmse, inlier_fraction=inlier_fraction, iterations=iterations,
        source_points=n_source, target_points=n_target,
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
    return RegistrationResult(
        transform=transform, method="gnss_anchor", status="accepted", reason="",
        rmse=rmse, inlier_fraction=1.0, iterations=1,
        source_points=len(source_positions), target_points=len(target_positions),
    )
