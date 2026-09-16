"""Detail pipeline driver (universal-perception directive section 11:
DISCOVERY -> ROI -> LOCAL RECONSTRUCTION -> ADAPTIVE REFINEMENT).

One call executes the whole chain so the detail stages are not
isolated modules consumed only by their own tests:

    assess_evidence_quality (P7-04, measured)
      -> discover_detail (P7-06)
      -> generate_rois (P7-06)
      -> refine_rois (P7-06 refinement executor)
      -> apply_outcomes (status transitions with evidence)

Default evidence resolution: when no `point_lookup` is supplied, the
driver resolves point ids against the reconstruction result's own
points (track_id -> position). A caller with a richer registry
(artifact store, WorldIR entities) passes its own lookup.

Honesty: nothing is fabricated at any stage -- the quality report is
measured, candidates/ROIs carry measured budgets, refinement either
produces measured residuals from resolved evidence or refuses with a
diagnostic. An empty scene yields an honest empty report, not an
error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from perception.detail.discovery import DetailCandidate, discover_detail
from perception.detail.refinement import (
    RefinementOutcome,
    apply_outcomes,
    refine_rois,
)
from perception.detail.roi import RegionOfInterest, generate_rois
from perception.quality.assessment import (
    EvidenceQualityReport,
    assess_evidence_quality,
)
from reconstruction.backend.interface import ReconstructionResult

__all__ = ["DetailPipelineReport", "run_detail_pipeline"]


@dataclass(frozen=True)
class DetailPipelineReport:
    """Every stage's artifacts, in one evidence-backed record."""

    quality: EvidenceQualityReport
    candidates: List[DetailCandidate]
    rois: List[RegionOfInterest]
    outcomes: List[RefinementOutcome]
    summary: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "quality": self.quality.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "rois": [r.to_dict() for r in self.rois],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "summary": dict(self.summary),
        }


def _default_lookup(
    result: ReconstructionResult,
) -> Callable[[str], Optional[Tuple[float, float, float]]]:
    table = {p.track_id: tuple(float(c) for c in p.position)
             for p in result.points}
    return lambda pid: table.get(pid)


def run_detail_pipeline(
    result: ReconstructionResult,
    cameras: Sequence,
    point_lookup: Optional[Callable[[str], Optional[Tuple[float, float, float]]]] = None,
    voxel_size: float = 1.0,
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> DetailPipelineReport:
    """Execute the detail chain over one reconstruction result."""
    quality = assess_evidence_quality(result, cameras)

    # Detail discovery needs the reconstruction's points; the assessor
    # produced an honest empty report for an empty scene -- discovery
    # itself returns [] for empty input.
    candidates = discover_detail(result, quality, voxel_size=voxel_size)
    rois = generate_rois(candidates, voxel_size=voxel_size)

    lookup = point_lookup or _default_lookup(result)
    outcomes = refine_rois(rois, point_lookup=lookup, up=up)
    rois = apply_outcomes(rois, outcomes)

    summary = {
        "n_points": len(result.points),
        "n_candidates": len(candidates),
        "n_rois": len(rois),
        "n_refined": sum(1 for r in rois if r.status == "refined"),
        "n_refused": sum(1 for r in rois if r.status == "refused"),
    }
    return DetailPipelineReport(
        quality=quality,
        candidates=candidates,
        rois=rois,
        outcomes=outcomes,
        summary=summary,
    )
