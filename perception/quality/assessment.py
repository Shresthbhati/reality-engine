"""Universal evidence-quality assessment (P7-04, directive items 2 +
32 + 11): measured, domain-agnostic detail capability of the evidence
behind a reconstruction.

Position in the universal pipeline (directive section 3): after dense
reconstruction and BEFORE entity discovery -- every downstream stage
(entity resolution, detail discovery, adaptive compute, capture
feedback) consumes this report's measured facts instead of assuming
capture quality.

Method, domain-agnostic by construction (a column, a pipe, a fender,
and a rock are all just points):

- GSD is MEASURED from real geometry. For every point observed by a
  camera: the actual camera-to-surface distance (euclidean, not
  z-depth) and the observing camera's actual focal length. Per-point
  GSD = distance * 1000 / focal_length (mm/px); the reported GSD is
  the median across observations -- the defensible summary statistic,
  not an assumed average.
- A point is observed by a camera iff it projects INSIDE that
  camera's image bounds. The `source_evidence_ids` metadata is NOT
  treated as the truth (it can be stale or optimistic); the
  projection is. Mismatches between claimed sources and measured
  observations are REPORTED (overclaim_count), never silently
  reconciled.
- Detail tier is DERIVED from measured facts by documented
  thresholds: fine GSD + enough views -> "fine", coarse GSD ->
  "coarse", nothing observed -> "unsupported". Deriving by a
  documented rule is not fabricating: the tier claims what the
  evidence supports, nothing more.
- Capture recommendations (directive sections 32/33 closed loop) are
  derived from the same measured facts, phrased for the Capture
  application.

Honesty rules inherited from the repository conventions: unknown
stays unknown (no observed points -> GSD None, tier "unsupported"),
points observed by zero cameras are reported (unprojectable) not
dropped, and every number in the report is measured or derived by a
documented formula -- nothing is invented.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from engine.math import Vec3
from reconstruction.backend.interface import ReconstructionResult
from reconstruction.calibration.camera import PinholeCamera

#: Detail-tier thresholds (documented, overridable per call). GSD in
#: mm/px; min_views_for_fine = minimum distinct cameras observing a
#: point for the "fine" tier. NOT tuned against real datasets --
#: tuning is deferred until real-capture benchmarks exist (the same
#: honesty as every other threshold in this repo).
DETAIL_TIER_THRESHOLDS = {
    "fine_gsd_mm": 5.0,      # <= 5 mm/px supports fine detail
    "medium_gsd_mm": 25.0,   # <= 25 mm/px supports medium detail
    "min_views_for_fine": 2, # fine detail needs multi-view corroboration
}

#: Fraction of observed points with >= 2 views below which the
#: "additional overlapping views" recommendation fires.
_MIN_MULTI_VIEW_FRACTION = 0.5

#: Fraction of points observed by zero cameras above which the
#: coverage-gap recommendation fires.
_MAX_UNPROJECTABLE_FRACTION = 0.1


@dataclass(frozen=True)
class EvidenceQualityReport:
    """Measured evidence quality for one reconstruction. Every field is
    measured or derived by a documented rule; None means unknown."""

    gsd_mm_per_px: Optional[float]
    detail_tier: str  # "fine" | "medium" | "coarse" | "unsupported"
    observed_fraction: float
    view_counts: Dict[str, int]
    unprojectable_point_ids: Tuple[str, ...]
    #: Count of points whose claimed source_evidence_ids include
    #: cameras that did NOT observe them (projection said out of
    #: bounds / behind the camera plane).
    overclaim_count: int
    #: Measured per-point view-angle diversity (directive section 11:
    #: view angle diversity is an evidence-quality input, previously
    #: unmeasured). For each observed point: the local surface normal
    #: from the point's k nearest neighbors (PCA smallest-eigenvalue
    #: vector, closed-form eigensolver -- the repo's established
    #: approach), then the angular spread of the observing cameras'
    #: ray-to-normal angles: (max - min) in degrees. Median across
    #: observed points. Points with < 2 observers or a degenerate
    #: local neighborhood contribute nothing (excluded from the
    #: median, counted in view_angle_diversity_n). None when no point
    #: has a measurable diversity.
    view_angle_diversity_deg: Optional[float] = None
    view_angle_diversity_n: int = 0
    #: Measured per-point GSD (mm/px), keyed by track id -- the same
    #: distance/focal measurement that produces the scene median,
    #: retained per point so region-level consumers (detail discovery
    #: budgets, per-cell refinement) can budget from THEIR OWN measured
    #: sampling rather than the scene median. Unprojectable points are
    #: absent (no camera observed them -- no GSD exists).
    per_point_gsd_mm: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_point_gsd_mm is None:
            object.__setattr__(self, "per_point_gsd_mm", {})

    def to_dict(self) -> dict:
        return {
            "gsd_mm_per_px": self.gsd_mm_per_px,
            "detail_tier": self.detail_tier,
            "observed_fraction": self.observed_fraction,
            "view_counts": dict(self.view_counts),
            "unprojectable_point_ids": list(self.unprojectable_point_ids),
            "overclaim_count": self.overclaim_count,
            "view_angle_diversity_deg": self.view_angle_diversity_deg,
            "view_angle_diversity_n": self.view_angle_diversity_n,
            "per_point_gsd_mm": dict(self.per_point_gsd_mm),
        }


def _detail_tier(gsd: Optional[float], min_views: int,
                 thresholds: dict) -> str:
    if gsd is None:
        return "unsupported"
    if (gsd <= thresholds["fine_gsd_mm"]
            and min_views >= thresholds["min_views_for_fine"]):
        return "fine"
    if gsd <= thresholds["medium_gsd_mm"]:
        return "medium"
    return "coarse"


def assess_evidence_quality(
    result: ReconstructionResult,
    cameras: Sequence[PinholeCamera],
    thresholds: Optional[dict] = None,
    min_views_for_fine: Optional[int] = None,
) -> EvidenceQualityReport:
    """Measure the detail capability of the evidence behind `result`.

    `cameras` must be the calibrated camera models corresponding to
    `result.camera_poses` (same order), built via
    `reconstruction.calibration.camera.camera_from_pose`. Passing an
    empty sequence refuses (ValueError): GSD is undefined without a
    focal length, and guessing one would fabricate the central number
    this report exists to measure.
    """
    if not cameras:
        if not result.points:
            # Nothing to measure at all: an honest empty report, not an
            # error -- there is no claim to refuse here.
            return EvidenceQualityReport(
                gsd_mm_per_px=None,
                detail_tier="unsupported",
                observed_fraction=0.0,
                view_counts={},
                unprojectable_point_ids=(),
                overclaim_count=0,
            )
        raise ValueError(
            "no calibrated cameras supplied -- GSD cannot be measured "
            "without focal length; refusing to guess intrinsics"
        )
    thresholds = dict(DETAIL_TIER_THRESHOLDS if thresholds is None else thresholds)
    if min_views_for_fine is not None:
        thresholds["min_views_for_fine"] = min_views_for_fine

    view_counts: Dict[str, int] = {}
    unprojectable: List[str] = []
    overclaims = 0
    per_point_gsd: List[float] = []
    per_point_gsd_by_id: Dict[str, float] = {}
    #: point index -> observing camera indices (for the diversity pass).
    obs_by_point: Dict[int, List[int]] = {}

    for p_idx, point in enumerate(result.points):
        px, py, pz = (float(c) for c in point.position)
        views = 0
        seen_by = set()
        for cam_idx, (cam, pose) in enumerate(zip(cameras, result.camera_poses)):
            proj = cam.project(Vec3(px, py, pz))
            if proj is None:
                continue
            u, v = proj
            if 0.0 <= u < cam.intrinsics.width and 0.0 <= v < cam.intrinsics.height:
                views += 1
                seen_by.add(pose.evidence_id)
                obs_by_point.setdefault(p_idx, []).append(cam_idx)
                dx = px - cam.extrinsics.position.x
                dy = py - cam.extrinsics.position.y
                dz = pz - cam.extrinsics.position.z
                distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
                # max(fx, fy): the finest-sampled image axis. For the
                # square pixels every current calibration produces
                # (fx == fy) this is just fx; documenting the choice
                # keeps the formula defensible for anisotropic pixels.
                focal_px = max(cam.intrinsics.fx, cam.intrinsics.fy)
                point_gsd = distance_m * 1000.0 / focal_px
                per_point_gsd.append(point_gsd)
                # Keep the point's FINEST measured GSD (closest/most
                # detailed observation): the budget question is what
                # sampling this evidence supports, and the best
                # observation defines that, not the median of the
                # point's observers.
                prev = per_point_gsd_by_id.get(point.track_id)
                if prev is None or point_gsd < prev:
                    per_point_gsd_by_id[point.track_id] = point_gsd
        if views == 0:
            unprojectable.append(point.track_id)
        else:
            view_counts[point.track_id] = views
            if set(point.source_evidence_ids) - seen_by:
                overclaims += 1

    # Median of REAL per-point observations -- not an assumed average.
    gsd: Optional[float] = (
        statistics.median(per_point_gsd) if per_point_gsd else None
    )
    observed = len(result.points) - len(unprojectable)
    observed_fraction = (observed / len(result.points)) if result.points else 0.0
    min_views = min(view_counts.values()) if view_counts else 0

    diversity_deg, diversity_n = _view_angle_diversity(
        result, cameras, obs_by_point
    )
    return EvidenceQualityReport(
        gsd_mm_per_px=gsd,
        detail_tier=_detail_tier(gsd, min_views, thresholds),
        observed_fraction=observed_fraction,
        view_counts=view_counts,
        unprojectable_point_ids=tuple(sorted(unprojectable)),
        overclaim_count=overclaims,
        per_point_gsd_mm=per_point_gsd_by_id,
        view_angle_diversity_deg=diversity_deg,
        view_angle_diversity_n=diversity_n,
    )


def _view_angle_diversity(
    result: ReconstructionResult,
    cameras: Sequence[PinholeCamera],
    obs_by_point: Dict[int, List[int]],
    k_neighbors: int = 6,
    max_points: int = 512,
) -> Tuple[Optional[float], int]:
    """Measured per-point view-angle diversity (directive section 11:
    previously an unmeasured evidence-quality input).

    For each observed point with >= 2 observing cameras: estimate the
    local surface normal from the point's `k_neighbors` nearest
    neighbors (PCA smallest-eigenvalue eigenvector of the local
    covariance, numpy's eigh on the real symmetric scatter -- the
    repo's established PCA approach), then measure the angular spread
    (max - min, degrees) of the observing cameras' ray-to-normal
    angles: angle between the camera->point ray and the normal, folded
    into [0, 90] via abs(cos) since PCA normals have no orientation.

    Honest limitations (documented, not hidden):
      - Points with a degenerate local neighborhood (collinear or
        coincident kNN: rank-deficient covariance) have no defensible
        normal and contribute nothing -- a fabricated normal would be
        worse than no measurement.
      - Only points with >= 2 observers carry an angular SPREAD; a
        single view direction has nothing to spread.
      - Sampling: if more than `max_points` eligible points exist, a
        deterministic stride sample is measured (fixed stride, no
        randomness) so the report stays affordable on large clouds;
        the sample size is in `view_angle_diversity_n`.

    Returns (median diversity in degrees, number of points measured).
    """
    eligible = sorted(i for i, cs in obs_by_point.items() if len(cs) >= 2)
    if not eligible:
        return None, 0
    if len(eligible) > max_points:
        stride = math.ceil(len(eligible) / max_points)
        eligible = eligible[::stride]

    import numpy as np
    from scipy.spatial import cKDTree

    pts = np.asarray(
        [[float(c) for c in result.points[i].position] for i in eligible]
    )
    tree = cKDTree(pts)
    k = min(k_neighbors, len(pts))
    _, idx = tree.query(pts, k=k, workers=1)
    if k == 1:
        idx = idx.reshape(-1, 1)

    spreads: List[float] = []
    for row, i in enumerate(eligible):
        neigh = pts[idx[row]]
        centered = neigh - neigh.mean(axis=0)
        scatter = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(scatter)
        # Smallest eigenvalue's eigenvector is the normal. A rank-2
        # neighborhood (an exactly planar patch) HAS a defensible
        # normal; degenerate means rank < 2 (collinear/coincident
        # points), where no plane is defined. Relative threshold:
        # floating-point noise on an exactly collinear fixture leaves
        # eigvals[1] at numerical zero, not exact zero.
        if eigvals[1] <= 1e-9 * max(eigvals[2], 1.0):
            continue
        normal = eigvecs[:, 0]
        # pts rows correspond to the (possibly stride-sampled) eligible
        # list; index by row, not by the original point id i.
        px, py, pz = pts[row]
        angles = []
        for c_idx in obs_by_point[i]:
            cam = cameras[c_idx].extrinsics.position
            ray = np.array([px - cam.x, py - cam.y, pz - cam.z])
            norm = float(np.linalg.norm(ray))
            if norm <= 0.0:
                continue
            cos_a = abs(float(np.dot(ray, normal)) / norm)
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))))
        if len(angles) < 2:
            continue
        spreads.append(max(angles) - min(angles))
    if not spreads:
        return None, 0
    return statistics.median(spreads), len(spreads)


def recommend_capture(report: EvidenceQualityReport) -> Dict[str, object]:
    """Derive capture recommendations from a measured quality report
    (closed loop: capture -> reconstruct -> measure -> request more
    capture).

    Returns a dict: "scene_budget" maps to the derived DetailBudget
    (perception/quality/detail_budget.py -- the machine-readable
    justified-detail level and compute tier for the Capture app), and
    every other key is a measured prose recommendation. Each
    recommendation is a measured conclusion from the report's facts --
    no fixed vocabulary of scene types."""
    from perception.quality.detail_budget import detail_budget_for

    recs: Dict[str, object] = {
        "scene_budget": detail_budget_for(report),
    }
    prose: List[str] = []
    if report.detail_tier == "unsupported" and not report.view_counts:
        # Distinguish genuinely empty scenes (nothing was ever
        # reconstructed) from coverage failures (points exist but no
        # camera observed any of them -- the coverage rule below says
        # that); only the former gets this blanket statement.
        if not report.unprojectable_point_ids:
            prose.append(
                "No reconstruction points exist; nothing about capture "
                "quality can be assessed yet."
            )
            recs["no_points"] = prose[0]
            return recs
        # Points exist but none observed: fall through -- the coverage
        # rule below reports it as the coverage failure it is.
    if (report.gsd_mm_per_px is not None
            and report.gsd_mm_per_px > DETAIL_TIER_THRESHOLDS["medium_gsd_mm"]):
        prose.append(
            f"GSD {report.gsd_mm_per_px:.1f} mm/px is coarse; "
            "additional close-range evidence recommended for detail."
        )
    multi = sum(1 for v in report.view_counts.values() if v >= 2)
    total = len(report.view_counts)
    if total and multi / total < _MIN_MULTI_VIEW_FRACTION:
        prose.append(
            f"Only {multi}/{total} points have >= 2 views; additional "
            "overlapping views recommended for reliable matching."
        )
    if (report.unprojectable_point_ids
            and report.observed_fraction < 1.0 - _MAX_UNPROJECTABLE_FRACTION):
        prose.append(
            f"{len(report.unprojectable_point_ids)} points were observed "
            "by zero cameras; coverage gaps detected -- additional "
            "capture positions recommended."
        )
    for i, text in enumerate(prose):
        recs[f"recommendation_{i}"] = text
    return recs
