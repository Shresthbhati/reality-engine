"""Cross-view depth consistency (reliability priority: multi-view depth
consistency -- detect contradictory evidence between depth maps, camera
poses, and scene structure instead of assuming agreement).

Position in the reliability chain (mission):

    per-view depth (IDepthBackend) -> unprojection (depth_to_points)
      -> (this module) cross-view consistency gate
      -> fusion / WorldIR geometry

Method, measured-only:

  - For each scene point, find each view's nearest measured depth
    sample (within `max_sample_distance`). Two views "co-observe" a
    point when both have a nearby sample.
  - Each co-observed ray pair is measured only when the mutual ray
    angle is inside [`min_angle_deg`, `max_angle_deg`]: near-parallel
    rays are degenerate (no baseline), and extremely oblique pairs
    carry depth uncertainty large enough that a 10% disagreement is
    not evidence of contradiction. Excluded pairs are reported as
    `inconclusive`, never as "consistent".
  - Co-visibility is established in PIXEL space, not world space: each
    scene point is projected into a view; if a measured depth sample
    lies within `max_pixel_distance` of that pixel, the view measures
    this point. (World-space association would be circular -- a
    biased depth displaces the unprojected sample, hiding exactly the
    contradiction it should reveal.) Each view's sample is then
    UNPROJECTED to a world point through its own camera. Two views of
    one physical surface must unproject to the same place; a pair
    CONTRADICTS when the world-space disagreement |p_a - p_b| exceeds
    `contradiction_threshold` times the pair's scene scale (mean
    camera range to the two samples; default 10%). Raw depths are NOT
    compared against each other -- cameras at different distances
    legitimately read different depths for the same point.
    Contradiction records carry BOTH measured depths -- never averaged
    away, never silently dropped.

Honesty rules: relative (non-metric) depth maps are refused (a
relative map has no meters to disagree in); depth maps without a
camera pose are skipped and named; no co-visibility at all is
`insufficient_evidence`, not "consistent". Fixtures in the tests are
deterministic unit scenes; real-data runs must be recorded as real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from engine.math import Vec3
from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import (
    PinholeCamera,
    quat_to_matrix,
)
from reconstruction.calibration.camera import (
    _UNDISTORT_ITERATIONS as _UNDISTORT_ITERS,
)
from reconstruction.calibration.camera import _undistort as _undistort_scalar

#: Default relative-depth disagreement considered a contradiction (10%).
DEFAULT_CONTRADICTION_THRESHOLD = 0.10
#: Deterministic cap on depth samples per view (stride-subsampled).
MAX_SAMPLES_PER_VIEW = 4096


@dataclass(frozen=True)
class DepthContradiction:
    """One measured cross-view disagreement: both depths, one point."""

    evidence_a: str
    evidence_b: str
    point: Tuple[float, float, float]  # the scene point both rays target
    depth_a: float  # camera-frame depth (meters) measured by view A
    depth_b: float  # camera-frame depth (meters) measured by view B
    relative_difference: float  # world disagreement / mean pair range

    def to_dict(self) -> dict:
        return {
            "evidence_a": self.evidence_a,
            "evidence_b": self.evidence_b,
            "point": list(self.point),
            "depth_a": self.depth_a,
            "depth_b": self.depth_b,
            "relative_difference": round(self.relative_difference, 6),
        }


@dataclass(frozen=True)
class ConsistencyReport:
    """Outcome of one cross-view depth consistency check.

    status:
      "consistent"           -- measured pairs, none contradict
      "contradictions"       -- at least one measured contradiction
      "inconclusive"         -- co-visibility existed but every pair was
                                excluded by the angle gates
      "insufficient_evidence" -- no scene point co-observed by >=2 views
    """

    status: str
    pairs_checked: int
    contradictions: Tuple[DepthContradiction, ...] = ()
    max_relative_difference: float = 0.0
    views_skipped: Tuple[str, ...] = ()  # depth maps with no camera pose
    reason: str = ""

    def has_contradictions(self) -> bool:
        return bool(self.contradictions)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "pairs_checked": self.pairs_checked,
            "max_relative_difference": round(self.max_relative_difference, 6),
            "views_skipped": list(self.views_skipped),
            "reason": self.reason,
            "contradictions": [c.to_dict() for c in self.contradictions],
        }


def _undistort_vectorized(x_d: np.ndarray, y_d: np.ndarray, cam: PinholeCamera):
    """Vectorized Brown-Conrady inverse (fixed-point), matching
    camera._undistort's model and iteration count exactly."""
    k = cam.intrinsics
    x, y = x_d.copy(), y_d.copy()
    for _ in range(_UNDISTORT_ITERS):
        r2 = x * x + y * y
        radial = 1.0 + k.k1 * r2 + k.k2 * r2 ** 2 + k.k3 * r2 ** 3
        tx = 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x)
        ty = k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y
        x = (x_d - tx) / radial
        y = (y_d - ty) / radial
    return x, y


class _ViewSamples:
    """One view's measured depth samples, keyed by pixel for
    co-visibility and unprojected to world space for comparison."""

    def __init__(self, depth_map: DepthMap, camera: PinholeCamera, max_samples: int):
        values = np.asarray(depth_map.values, dtype=float)
        mask = np.isfinite(values) & (values > 0.0)
        rows, cols = np.nonzero(mask)
        if len(cols) > max_samples:
            stride = int(np.ceil(len(cols) / max_samples))
            rows, cols = rows[::stride], cols[::stride]
        self.z = values[rows, cols]
        # Pixel centers of the measured samples.
        self.pix = np.stack([cols.astype(float) + 0.5,
                             rows.astype(float) + 0.5], axis=1)
        intr = camera.intrinsics
        x_d = (self.pix[:, 0] - intr.cx) / intr.fx
        y_d = (self.pix[:, 1] - intr.cy) / intr.fy
        x, y = _undistort_vectorized(x_d, y_d, camera)
        # Camera-frame points at measured depth (z along optical axis).
        pts_cam = np.stack([x * self.z, y * self.z, self.z], axis=1)
        # Camera-to-world via the extrinsics quaternion.
        r = np.asarray(quat_to_matrix(camera.extrinsics.rotation), dtype=float)
        origin = np.array([
            camera.extrinsics.position.x,
            camera.extrinsics.position.y,
            camera.extrinsics.position.z,
        ], dtype=float)
        self.pts = pts_cam @ r.T + origin
        self.origin = origin
        self.range = np.linalg.norm(self.pts - origin, axis=1)
        self.rays = pts_cam / np.linalg.norm(pts_cam, axis=1, keepdims=True)
        self.pix_tree = cKDTree(self.pix)


def check_depth_consistency(
    depth_maps: Sequence[DepthMap],
    cameras: Dict[str, PinholeCamera],
    scene_points: Sequence[Sequence[float]],
    *,
    min_angle_deg: float = 5.0,
    max_angle_deg: float = 75.0,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    max_pixel_distance: float = 2.0,
    max_samples_per_view: int = MAX_SAMPLES_PER_VIEW,
) -> ConsistencyReport:
    """Check whether metric depth maps from different views agree on the
    same scene points. See module docstring for method and honesty rules.

    Raises ValueError if any depth map is not unit="meters" (a relative
    map has no meters to disagree in; metricize it first).
    """
    for m in depth_maps:
        if m.unit != "meters":
            raise ValueError(
                f"depth map {m.evidence_id!r} is unit={m.unit!r} (relative) -- "
                "cross-view consistency requires metric depth; call "
                "metricize_relative_depth() first"
            )
    if not scene_points:
        return ConsistencyReport(
            status="insufficient_evidence", pairs_checked=0,
            reason="no scene points to check",
        )

    skipped = tuple(m.evidence_id for m in depth_maps if m.evidence_id not in cameras)
    views: Dict[str, _ViewSamples] = {}
    for m in depth_maps:
        cam = cameras.get(m.evidence_id)
        if cam is not None:
            views[m.evidence_id] = _ViewSamples(m, cam, max_samples_per_view)
    if len(views) < 2:
        return ConsistencyReport(
            status="insufficient_evidence", pairs_checked=0,
            views_skipped=skipped,
            reason=f"fewer than two views with cameras and depth samples ({len(views)})",
        )

    view_ids = sorted(views)
    contradictions: List[DepthContradiction] = []
    pairs_checked = 0
    co_observed = 0
    max_rel = 0.0

    for point in scene_points:
        p = Vec3(float(point[0]), float(point[1]), float(point[2]))
        hits: Dict[str, int] = {}
        for eid in view_ids:
            view = views[eid]
            cam = cameras[eid]
            uv = cam.project(p)
            if uv is None:
                continue
            dist, idx = view.pix_tree.query(list(uv))
            if dist <= max_pixel_distance:
                hits[eid] = int(idx)
        if len(hits) < 2:
            continue
        ids = sorted(hits)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                va, vb = views[ids[i]], views[ids[j]]
                ia, ib = hits[ids[i]], hits[ids[j]]
                cos_angle = float(np.clip(np.dot(va.rays[ia], vb.rays[ib]), -1.0, 1.0))
                angle = math.degrees(math.acos(cos_angle))
                co_observed += 1
                if angle < min_angle_deg or angle > max_angle_deg:
                    continue
                pairs_checked += 1
                # World-space disagreement between the two unprojected
                # measurements of the same physical surface, normalized
                # by the pair's scene scale (mean camera range).
                disagreement = float(np.linalg.norm(va.pts[ia] - vb.pts[ib]))
                scale = 0.5 * (float(va.range[ia]) + float(vb.range[ib]))
                rel = disagreement / scale
                max_rel = max(max_rel, rel)
                if rel > contradiction_threshold:
                    a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                    da, db = (va.z[ia], vb.z[ib]) if ids[i] < ids[j] else (vb.z[ib], va.z[ia])
                    contradictions.append(DepthContradiction(
                        evidence_a=a, evidence_b=b,
                        point=(p.x, p.y, p.z),
                        depth_a=float(da), depth_b=float(db),
                        relative_difference=float(rel),
                    ))

    contradictions.sort(key=lambda c: (c.evidence_a, c.evidence_b, c.point))

    if pairs_checked == 0:
        if co_observed == 0:
            return ConsistencyReport(
                status="insufficient_evidence", pairs_checked=0,
                contradictions=tuple(contradictions),
                views_skipped=skipped,
                reason="no scene point is co-observed by two or more views "
                       "within max_pixel_distance",
            )
        return ConsistencyReport(
            status="inconclusive", pairs_checked=0,
            contradictions=tuple(contradictions),
            views_skipped=skipped,
            reason=f"all {co_observed} co-observed ray pairs were excluded by the "
                   f"angle gates (min {min_angle_deg} deg, max {max_angle_deg} deg) -- "
                   "baseline geometry cannot support a consistency verdict",
        )

    if contradictions:
        status = "contradictions"
        reason = (
            f"{len(contradictions)} of {pairs_checked} measured cross-view pairs "
            f"disagree by more than {contradiction_threshold:.0%} relative depth"
        )
    else:
        status = "consistent"
        reason = (
            f"{pairs_checked} cross-view pairs agree within "
            f"{contradiction_threshold:.0%} (max observed {max_rel:.1%})"
        )

    return ConsistencyReport(
        status=status, pairs_checked=pairs_checked,
        contradictions=tuple(contradictions),
        max_relative_difference=max_rel,
        views_skipped=skipped, reason=reason,
    )
