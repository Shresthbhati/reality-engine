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
from perception.detail.worldir import integrate_detail_outcomes
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
    #: WorldIR integration output (set when build_world_ir=True): the
    #: report records which entities/geometries were added and which
    #: ROIs were refused as facts. None when integration is off.
    integration: Optional[object] = None
    #: The world the refined outcomes were integrated into (the same
    #: object passed in, mutated in place by the integration's
    #: conventions). None when build_world_ir is False or no world was
    #: supplied.
    world_ir: Optional[object] = None

    def to_dict(self) -> dict:
        return {
            "quality": self.quality.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "rois": [r.to_dict() for r in self.rois],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "summary": dict(self.summary),
            "integration": (
                self.integration.to_dict()
                if self.integration is not None
                else None
            ),
        }

    @property
    def refusal_count(self) -> int:
        """ROIs refused as recorded facts (zero when integration is off)."""
        return self.integration.refusal_count if self.integration is not None else 0


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
    build_world_ir: bool = False,
    world=None,
    include_structure: bool = False,
) -> DetailPipelineReport:
    """Execute the detail chain over one reconstruction result.

    With `build_world_ir=True`, refined outcomes are integrated into
    `world` (a WorldIR instance; a fresh one is created when omitted)
    using the repo's provenance conventions -- refused ROIs record a
    fact, never geometry. The integration report rides on the
    returned pipeline report; entity/geometry counts are appended to
    the summary.

    `include_structure=True` seeds ROIs from measured planar
    STRUCTURE cells as well as detail cells (see generate_rois): the
    real room capture proved oriented planar structure is invisible
    to the curvature gate by construction, so a driver without this
    pass-through dead-ends a room's walls.
    """
    quality = assess_evidence_quality(result, cameras)

    # Detail discovery needs the reconstruction's points; the assessor
    # produced an honest empty report for an empty scene -- discovery
    # itself returns [] for empty input.
    candidates = discover_detail(result, quality, voxel_size=voxel_size)
    rois = generate_rois(
        candidates, voxel_size=voxel_size,
        include_structure=include_structure,
    )

    lookup = point_lookup or _default_lookup(result)
    outcomes = refine_rois(rois, point_lookup=lookup, up=up)
    rois = apply_outcomes(rois, outcomes)

    summary = {
        "n_points": len(result.points),
        "n_candidates": len(candidates),
        "n_rois": len(rois),
        "n_refined": sum(1 for r in rois if r.status == "refined"),
        "n_refused": sum(1 for r in rois if r.status == "refused"),
        "n_entities": 0,
    }

    integration = None
    world_ir = None
    if build_world_ir:
        world_ir = world if world is not None else _new_world_ir()
        integration = integrate_detail_outcomes(rois, outcomes, world_ir)
        summary["n_entities"] = len(integration.entity_ids)
        summary["n_geometries"] = len(integration.geometry_ids)
        summary["n_refusals"] = integration.refusal_count

    return DetailPipelineReport(
        quality=quality,
        candidates=candidates,
        rois=rois,
        outcomes=outcomes,
        summary=summary,
        integration=integration,
        world_ir=world_ir,
    )


def _new_world_ir():
    from world_ir import WorldIR

    return WorldIR()
