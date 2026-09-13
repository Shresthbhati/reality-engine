"""Depth map -> metric point cloud (capture-to-WorldIR vertical slice).

Ports the audit worktree's `depth_map_to_points` into the main tree and
adds the piece it lacked: making RELATIVE monocular depth (MiDaS) usable
without lying about it.

Honesty rules (spec sec 6/29; matching worktree precedent):

  - A relative (non-metric) depth map is refused for direct unprojection
    (`DepthToPointsError`) -- never silently treated as meters.
  - Metricizing relative depth is a SEPARATE, explicit operation
    (`metricize_relative_depth`) that fits one scale against the SfM
    sparse cloud and REPORTS the fit quality; the output is labeled
    metric-by-alignment, an explicitly documented approximation, never
    sensor-quality depth.
  - Pixels with non-finite/non-positive depth are skipped, never given
    fabricated positions.
  - `stride` is honest pixel decimation, NOT voxel downsampling.

Why scale-from-SfM works here: monocular models predict inverse depth
well up to one unknown scale; the sparse cloud observed by the same
camera fixes that scale per-view. Where the fit is poor (occlusion,
model failure) the diagnostics say so instead of hiding it.

Camera-frame math here is vectorized numpy using the same conventions
as `PinholeCamera` (world_to_camera = R^T (p - C) with camera-to-world
R; depth = z along the optical axis), so results are consistent with
the scalar model to floating-point tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from perception.depth.interface import DepthMap
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from reconstruction.calibration.camera import (
    CameraIntrinsics,
    PinholeCamera,
    camera_from_pose,
    quat_to_matrix,
)


class DepthToPointsError(ValueError):
    pass


def depth_map_to_points(
    depth: DepthMap,
    camera: PinholeCamera,
    stride: int = 1,
) -> List[ReconstructedPoint]:
    """Unproject every valid pixel of a METRIC `depth` map (subsampled
    by `stride`) through `camera` into world-space `ReconstructedPoint`s.

    Raises DepthToPointsError for a relative (non-metric) depth map or a
    non-positive stride -- unprojecting relative depth would silently
    invent metric scale. Skips (never fabricates) pixels with
    non-finite/non-positive depth. track_id is deterministic (evidence
    id + pixel coordinates).
    """
    if depth.unit != "meters":
        raise DepthToPointsError(
            f"depth map for {depth.evidence_id} is unit={depth.unit!r} (relative), not 'meters' -- "
            "unprojecting a relative depth map would silently invent metric scale; "
            "call metricize_relative_depth() first"
        )
    if stride < 1:
        raise DepthToPointsError(f"stride must be >= 1, got {stride}")

    rows = range(0, depth.height, stride)
    cols = range(0, depth.width, stride)

    points: List[ReconstructedPoint] = []
    for row in rows:
        depth_row = depth.values[row]
        for col in cols:
            d = depth_row[col]
            if not math.isfinite(d) or d <= 0.0:
                continue
            world_point = camera.unproject(col + 0.5, row + 0.5, d)
            points.append(ReconstructedPoint(
                position=world_point.as_tuple(),
                track_id=f"depth-{depth.evidence_id}-{row:05d}-{col:05d}",
                source_evidence_ids=[depth.evidence_id],
                uncertainty=depth.uncertainty,
            ))
    return points


@dataclass(frozen=True)
class DepthAlignment:
    """Observed facts of one relative->metric alignment, for provenance."""

    evidence_id: str
    #: median |z_metric - z_sfm| over aligned pixels, in METERS
    residual_median_m: float
    #: fraction of aligned pixels within 2x the median residual
    inlier_fraction: float
    aligned_pixels: int
    #: fitted scale s where metric_depth = s / relative_value
    scale: float
    note: str

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "residual_median_m": self.residual_median_m,
            "inlier_fraction": self.inlier_fraction,
            "aligned_pixels": self.aligned_pixels,
            "scale": self.scale,
            "note": self.note,
        }


def metricize_relative_depth(
    depth: DepthMap,
    camera: PinholeCamera,
    sparse_points_world: Sequence[Sequence[float]],
    max_sparse_points: int = 2000,
) -> Tuple[DepthMap, DepthAlignment]:
    """Convert a RELATIVE depth map to metric 'meters' by fitting ONE
    scale against the SfM sparse points visible in this view.

    Method (documented approximation): project each sparse point into
    this view; where it lands, compare its true camera-frame depth with
    the model's relative value. Monocular relative depth is inverse-
    depth-like, so the fit is z_sfm ~ scale / rel and the scale is the
    median of z_sfm * rel over the aligned pixels. The output is honest
    metric-by-alignment depth: trustworthy where the sparse cloud is
    dense, smooth interpolation elsewhere -- exactly what the returned
    DepthAlignment reports. Raises DepthToPointsError when too few
    sparse points project into the view for a trustworthy fit.
    """
    if depth.unit != "relative":
        raise DepthToPointsError(
            f"metricize_relative_depth expects unit='relative', got {depth.unit!r}"
        )

    rel = np.asarray(depth.values, dtype=float)
    R = np.asarray(quat_to_matrix(camera.extrinsics.rotation), dtype=float)
    C = np.array([
        camera.extrinsics.position.x,
        camera.extrinsics.position.y,
        camera.extrinsics.position.z,
    ], dtype=float)

    pts = np.asarray(sparse_points_world, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise DepthToPointsError(
            f"sparse_points_world must be an (N, 3) array, got shape {pts.shape}"
        )
    if len(pts) > max_sparse_points:
        pts = pts[:: max(1, len(pts) // max_sparse_points)]

    # world -> camera (matches PinholeCamera.world_to_camera)
    pts_cam = (pts - C) @ R
    z = pts_cam[:, 2]
    in_front = z > 1e-6

    intr = camera.intrinsics
    fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
    z_safe = np.where(in_front, z, 1.0)
    u = fx * pts_cam[:, 0] / z_safe + cx
    v = fy * pts_cam[:, 1] / z_safe + cy

    inside = (
        in_front
        & (u >= 0) & (u < depth.width - 1)
        & (v >= 0) & (v < depth.height - 1)
    )
    n_inside = int(inside.sum())

    min_points = 30
    if n_inside < min_points:
        raise DepthToPointsError(
            f"only {n_inside} sparse points project into view {depth.evidence_id} "
            f"(need >= {min_points}) -- cannot honestly metricize this depth map"
        )

    ui = u[inside].astype(int)
    vi = v[inside].astype(int)
    rel_at = rel[vi, ui]
    z_sfm = z[inside]

    valid = np.isfinite(rel_at) & (rel_at > 1e-6) & np.isfinite(z_sfm)
    if int(valid.sum()) < min_points:
        raise DepthToPointsError(
            f"too few valid depth samples in view {depth.evidence_id} "
            f"({int(valid.sum())} of {n_inside})"
        )

    # inverse-depth scale fit: z_sfm ~ scale / rel => scale = median(z_sfm * rel)
    ratios = z_sfm[valid] * rel_at[valid]
    scale = float(np.median(ratios))
    pred = scale / np.maximum(rel_at[valid], 1e-9)
    residuals = np.abs(pred - z_sfm[valid])
    residual_median = float(np.median(residuals))
    inlier_fraction = float(np.mean(residuals <= max(2.0 * residual_median, 1e-9)))

    metric_values = scale / np.maximum(rel, 1e-9)
    metric = DepthMap(
        evidence_id=depth.evidence_id,
        width=depth.width,
        height=depth.height,
        values=metric_values.tolist(),
        unit="meters",
        uncertainty=depth.uncertainty,
    )
    alignment = DepthAlignment(
        evidence_id=depth.evidence_id,
        residual_median_m=residual_median,
        inlier_fraction=inlier_fraction,
        aligned_pixels=int(valid.sum()),
        scale=scale,
        note=(
            f"relative depth metricized by median inverse-depth alignment against "
            f"{int(valid.sum())} SfM sparse points (residual median "
            f"{residual_median * 1000:.1f} mm, inlier fraction {inlier_fraction:.2f}); "
            "correct where the sparse cloud is dense, approximate elsewhere"
        ),
    )
    return metric, alignment


def metricize_all(
    depth_maps: List[DepthMap],
    result: ReconstructionResult,
    intrinsics: CameraIntrinsics,
) -> Tuple[List[DepthMap], List[DepthAlignment], int]:
    """Metricize a batch of relative depth maps against the reconstruction
    they belong to.

    For each depth map, the matching camera pose is looked up by
    evidence_id; maps whose view cannot be honestly aligned (pose
    missing, too few visible sparse points) are SKIPPED and counted in
    the returned failure count -- never silently degraded. Returns
    (metric_maps, alignments, failed_count).
    """
    poses_by_id = {p.evidence_id: p for p in result.camera_poses}
    sparse = np.array([p.position for p in result.points], dtype=float)
    metric_maps: List[DepthMap] = []
    alignments: List[DepthAlignment] = []
    failed = 0
    for dm in depth_maps:
        pose = poses_by_id.get(dm.evidence_id)
        if pose is None:
            failed += 1
            continue
        camera = camera_from_pose(intrinsics, pose)
        try:
            metric, alignment = metricize_relative_depth(dm, camera, sparse)
        except DepthToPointsError:
            failed += 1
            continue
        metric_maps.append(metric)
        alignments.append(alignment)
    return metric_maps, alignments, failed
