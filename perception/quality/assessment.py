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

from engine.physics.math3 import Vec3
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

    def to_dict(self) -> dict:
        return {
            "gsd_mm_per_px": self.gsd_mm_per_px,
            "detail_tier": self.detail_tier,
            "observed_fraction": self.observed_fraction,
            "view_counts": dict(self.view_counts),
            "unprojectable_point_ids": list(self.unprojectable_point_ids),
            "overclaim_count": self.overclaim_count,
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

    for point in result.points:
        px, py, pz = (float(c) for c in point.position)
        views = 0
        seen_by = set()
        for cam, pose in zip(cameras, result.camera_poses):
            proj = cam.project(Vec3(px, py, pz))
            if proj is None:
                continue
            u, v = proj
            if 0.0 <= u < cam.intrinsics.width and 0.0 <= v < cam.intrinsics.height:
                views += 1
                seen_by.add(pose.evidence_id)
                dx = px - cam.extrinsics.position.x
                dy = py - cam.extrinsics.position.y
                dz = pz - cam.extrinsics.position.z
                distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
                # max(fx, fy): the finest-sampled image axis. For the
                # square pixels every current calibration produces
                # (fx == fy) this is just fx; documenting the choice
                # keeps the formula defensible for anisotropic pixels.
                focal_px = max(cam.intrinsics.fx, cam.intrinsics.fy)
                per_point_gsd.append(distance_m * 1000.0 / focal_px)
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
    return EvidenceQualityReport(
        gsd_mm_per_px=gsd,
        detail_tier=_detail_tier(gsd, min_views, thresholds),
        observed_fraction=observed_fraction,
        view_counts=view_counts,
        unprojectable_point_ids=tuple(sorted(unprojectable)),
        overclaim_count=overclaims,
    )


def recommend_capture(report: EvidenceQualityReport) -> List[str]:
    """Derive capture recommendations from a measured quality report
    (closed loop: capture -> reconstruct -> measure -> request more
    capture). Each recommendation is a measured conclusion from the
    report's facts -- no fixed vocabulary of scene types."""
    recs: List[str] = []
    if report.detail_tier == "unsupported" and not report.view_counts:
        # Distinguish genuinely empty scenes (nothing was ever
        # reconstructed) from coverage failures (points exist but no
        # camera observed any of them -- the coverage rule below says
        # that); only the former gets this blanket statement.
        if not report.unprojectable_point_ids:
            recs.append(
                "No reconstruction points exist; nothing about capture "
                "quality can be assessed yet."
            )
            return recs
        # Points exist but none observed: fall through -- the coverage
        # rule below reports it as the coverage failure it is.
    if (report.gsd_mm_per_px is not None
            and report.gsd_mm_per_px > DETAIL_TIER_THRESHOLDS["medium_gsd_mm"]):
        recs.append(
            f"GSD {report.gsd_mm_per_px:.1f} mm/px is coarse; "
            "additional close-range evidence recommended for detail."
        )
    multi = sum(1 for v in report.view_counts.values() if v >= 2)
    total = len(report.view_counts)
    if total and multi / total < _MIN_MULTI_VIEW_FRACTION:
        recs.append(
            f"Only {multi}/{total} points have >= 2 views; additional "
            "overlapping views recommended for reliable matching."
        )
    if (report.unprojectable_point_ids
            and report.observed_fraction < 1.0 - _MAX_UNPROJECTABLE_FRACTION):
        recs.append(
            f"{len(report.unprojectable_point_ids)} points were observed "
            "by zero cameras; coverage gaps detected -- additional "
            "capture positions recommended."
        )
    return recs
