"""Metric scale from device depth (plan F2): LiDAR / structured-light / any metric depth sensor.

A phone depth frame is in METRES. The reconstruction's points are in model units. For every reconstructed point that
projects into a posed depth frame, the ratio ``depth_z / model_z`` is one estimate of metres-per-model-unit; the median
over many points and views is the scale. Nothing here guesses: a frame without a pose, a point behind the camera, an
invalid depth pixel or a spread too wide to be one rigid scene is excluded and the estimate says so.

This is the estimator only. It does not invent a depth frame, and it does not apply the scale -- ``scale_from_depth``
does that explicitly, and the caller records ``DEVICE_DEPTH`` as the scale basis (dimension state MEASURED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from reconstruction.scale import DEVICE_DEPTH, ScaleDiagnostics, ScaleState, ScaledReconstruction

#: a LiDAR-like frame must agree this closely; a wider spread means the depth and the model are not one rigid scene
MAX_RATIO_SPREAD = 0.20
#: samples below this cannot support a median anybody should trust
MIN_SAMPLES = 30


@dataclass(frozen=True)
class DepthScaleEstimate:
    meters_per_unit: Optional[float]
    samples: int
    ratio_spread: float
    used_views: Tuple[str, ...]
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return {"method": DEVICE_DEPTH, "meters_per_unit": self.meters_per_unit, "samples": self.samples,
                "ratio_spread": round(self.ratio_spread, 4), "used_views": list(self.used_views),
                "accepted": self.accepted, "reason": self.reason}


def _pose_fields(pose) -> Tuple[str, Sequence[float], Sequence[float]]:
    if isinstance(pose, tuple):
        return str(pose[0]), pose[1], pose[2]
    return str(pose.evidence_id), pose.position, pose.rotation


def _rotation(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def estimate_scale_from_depth(poses: Iterable, points: Iterable[Sequence[float]],
                              depth_by_evidence_id: Dict[str, np.ndarray], intrinsics: Any,
                              *, stride: int = 8, min_samples: int = MIN_SAMPLES,
                              max_spread: float = MAX_RATIO_SPREAD) -> DepthScaleEstimate:
    """Metres-per-model-unit from metric depth frames. ``intrinsics`` is a CameraIntrinsics matching the depth size."""
    fx, fy, cx, cy = float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.cx), float(intrinsics.cy)
    w, h = int(intrinsics.width), int(intrinsics.height)
    # ``points`` may be ReconstructedPoint objects or raw (x, y, z) positions
    raw = [getattr(p, "position", p) for p in points]
    pts = np.asarray(raw, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts):
        return DepthScaleEstimate(None, 0, 0.0, (), False, "no reconstructed points")
    chosen = pts[::max(1, int(stride))]
    ratios: list[float] = []
    used: list[str] = []
    for pose in poses:
        eid, position, rotation = _pose_fields(pose)
        depth = depth_by_evidence_id.get(eid)
        if depth is None:
            continue
        depth = np.asarray(depth, float)
        if depth.ndim != 2 or depth.shape[0] < 2 or depth.shape[1] < 2:
            continue
        R = _rotation(rotation)
        C = np.asarray(position, float)
        rel = chosen - C
        cam = rel @ R                       # R^T (p - C), since R is camera-to-world
        z = cam[:, 2]
        valid = z > 1e-6
        if not valid.any():
            continue
        u = fx * cam[valid, 0] / z[valid] + cx
        v = fy * cam[valid, 1] / z[valid] + cy
        ui = np.rint(u).astype(int)
        vi = np.rint(v).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not inside.any():
            continue
        zz = z[valid][inside]
        dd = depth[np.clip(vi[inside], 0, depth.shape[0] - 1), np.clip(ui[inside], 0, depth.shape[1] - 1)]
        # depth maps whose size differs from the intrinsics are refused, not resampled: a mismatched camera model
        # produces a scale that looks plausible and is wrong
        if depth.shape[1] != w or depth.shape[0] != h:
            continue
        good = np.isfinite(dd) & (dd > 0.0) & np.isfinite(zz)
        if good.any():
            ratios.extend((dd[good] / zz[good]).tolist())
            used.append(eid)
    if len(ratios) < max(3, int(min_samples)):
        return DepthScaleEstimate(None, len(ratios), 0.0, tuple(sorted(set(used))), False,
                                  f"only {len(ratios)} valid depth/model samples (need {min_samples})")
    arr = np.asarray(ratios, float)
    med = float(np.median(arr))
    if not math.isfinite(med) or med <= 0:
        return DepthScaleEstimate(None, len(arr), 0.0, tuple(sorted(set(used))), False,
                                  "median depth/model ratio is not a positive finite number")
    p10, p90 = np.percentile(arr, [10, 90])
    spread = float((p90 - p10) / med)
    accepted = spread <= max_spread
    return DepthScaleEstimate(med if accepted else None, len(arr), spread, tuple(sorted(set(used))), accepted,
                              ("metric depth agrees with the model" if accepted else
                               f"depth/model spread {spread:.3f} > {max_spread:.2f}: not one rigid scene; scale refused"))


def scale_from_depth(result, estimate: DepthScaleEstimate) -> ScaledReconstruction:
    """Apply an ACCEPTED depth-scale estimate to a reconstruction. Raises ValueError for a refused estimate."""
    if not estimate.accepted or not estimate.meters_per_unit or estimate.meters_per_unit <= 0:
        raise ValueError(f"scale cannot be applied: {estimate.reason}")
    mpu = float(estimate.meters_per_unit)
    from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult

    scaled_poses = [ReconstructedCameraPose(evidence_id=p.evidence_id,
                                            position=tuple(float(c) * mpu for c in p.position),
                                            rotation=p.rotation, uncertainty=p.uncertainty)
                    for p in sorted(result.camera_poses, key=lambda p: p.evidence_id)]
    scaled_points = [ReconstructedPoint(position=tuple(float(c) * mpu for c in p.position), track_id=p.track_id,
                                        source_evidence_ids=p.source_evidence_ids, uncertainty=p.uncertainty,
                                        color=p.color)
                     for p in sorted(result.points, key=lambda p: p.track_id)]
    scaled = ReconstructionResult(points=scaled_points, camera_poses=scaled_poses,
                                  registration_status=result.registration_status)
    note = (f"metric scale from {DEVICE_DEPTH}: median metric depth / model z over {estimate.samples} samples in "
            f"{len(estimate.used_views)} frames = {mpu:.6f} m/unit (spread {estimate.ratio_spread:.3f})")
    diag = ScaleDiagnostics(state=ScaleState.METRIC, meters_per_unit=mpu, ratio_count=estimate.samples,
                            ratio_spread=estimate.ratio_spread, note=note, basis=DEVICE_DEPTH,
                            dimension_state="MEASURED")
    return ScaledReconstruction(result=scaled, state=ScaleState.METRIC, meters_per_unit=mpu, diagnostics=diag)
