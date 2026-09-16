"""Architectural reconstruction benchmark (directive sections 30-33,
37): a structure benchmark record + orchestrator that runs the REAL
perception stack over a ReconstructionResult and reports what actually
happened -- registration, geometry, architectural components by class
and phase, relationships, confidence tiers, failure regions, WorldIR
entity count.

Honesty rules (directive section 1: DO NOT FAKE THE RESULT):
  - A structure whose real capture has not happened (CAPTURE_PENDING)
    cannot produce a run. The capture is a genuine external dependency
    -- a human with a camera at the structure -- and is recorded as
    such, never simulated.
  - Every reported number is measured from the actual pipeline output.
    No metric is hard-coded, filled in, or "representative".
  - The record is structure-agnostic: nothing here knows anything
    about Victoria Memorial's specific architecture. The same record
    type serves forts, skyscrapers, ordinary buildings (directive
    section 34's generalization requirement).
  - Unaccepted components are reported as failure regions with their
    recorded reasons -- visible, not dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from perception.architecture.components import (
    ConfidenceTier,
    apply_repetition_priors,
    build_component_observations,
    detect_repetition,
    resolve_components,
)
from perception.architecture.promotion import build_architectural_graph
from perception.architecture.registry import get_default_registry
from perception.architecture.segments import segment_non_plane_points
from perception.architecture.parametric import (
    FitRefused,
    fit_circle,
    fit_cylinder,
    fit_sphere,
)
from reconstruction.backend.interface import ReconstructionResult
from world_ir import WorldIR

UP = (0.0, 0.0, 1.0)


class CaptureStatus(str, Enum):
    """Is there REAL captured evidence for this structure?"""

    CAPTURE_PENDING = "capture_pending"
    CAPTURED = "captured"


#: Map a fit's numeric kind to the registry classes it can propose --
#: mirrored from components._KIND_TO_CLASSES (single source kept in
#: components; re-declared here only for the benchmark's phase report).
def _kind_classes(fit) -> tuple:
    from perception.architecture.components import _KIND_TO_CLASSES, _kind_of
    kind = _kind_of(fit)
    return _KIND_TO_CLASSES.get(kind, ()) if kind else ()


@dataclass
class StructureBenchmark:
    """One real-world structure's benchmark record. Nothing in here
    encodes the structure's architecture -- only its identity, capture
    state, and what a real capture must contain."""

    structure_name: str
    location: str
    capture_status: CaptureStatus
    #: What a real capture must include (directive section 33).
    capture_requirements: List[str] = field(default_factory=list)
    #: Populated by a real capture session; never fabricated.
    capture_session_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "structure_name": self.structure_name,
            "location": self.location,
            "capture_status": self.capture_status.value,
            # Explicit data label (directive sections 11/33): real
            # captured evidence vs synthetic fixture.
            "evidence_label": (
                "real_captured"
                if self.capture_status is CaptureStatus.CAPTURED
                else "no_capture_yet"
            ),
            "capture_requirements": list(self.capture_requirements),
            "capture_session_id": self.capture_session_id,
        }


@dataclass
class FailureRegion:
    """A part of the capture the pipeline could not accept -- reported,
    never hidden."""

    kind: str  # "unaccepted_component" | "undersized_segment" | "fit_refused"
    reason: str
    subject_id: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "reason": self.reason, "subject_id": self.subject_id}


@dataclass
class BenchmarkReport:
    structure_name: str
    registration_status: str
    camera_count: int
    point_count: int
    plane_count: int
    segment_count: int
    components_by_class: Dict[str, int]
    components_by_phase: Dict[int, int]
    relationship_count: int
    tier_distribution: Dict[str, int]
    worldir_entity_count: int
    failure_regions: List[FailureRegion]

    def to_dict(self) -> dict:
        return {
            "structure_name": self.structure_name,
            "registration_status": self.registration_status,
            "camera_count": self.camera_count,
            "point_count": self.point_count,
            "plane_count": self.plane_count,
            "segment_count": self.segment_count,
            "components_by_class": self.components_by_class,
            "components_by_phase": self.components_by_phase,
            "relationship_count": self.relationship_count,
            "tier_distribution": self.tier_distribution,
            "worldir_entity_count": self.worldir_entity_count,
            "failure_regions": [f.to_dict() for f in self.failure_regions],
        }


def run_architectural_benchmark(
    benchmark: StructureBenchmark,
    reconstruction: Optional[ReconstructionResult] = None,
    seed: int = 7,
) -> BenchmarkReport:
    """Run the real perception stack over a real reconstruction result.

    Raises RuntimeError for a CAPTURE_PENDING benchmark (nothing has
    been captured -- there is nothing real to report) and ValueError
    for a CAPTURED benchmark called without its reconstruction output.
    """
    if benchmark.capture_status is not CaptureStatus.CAPTURED:
        raise RuntimeError(
            f"benchmark {benchmark.structure_name!r}: capture status is "
            f"{benchmark.capture_status.value} -- the real capture is an "
            f"external dependency that has not happened; no run exists "
            f"and none may be simulated"
        )
    if reconstruction is None:
        raise ValueError(
            "a CAPTURED benchmark needs its actual ReconstructionResult"
        )

    failure_regions: List[FailureRegion] = []

    # ---- segmentation FIRST, classification second ----
    # Subtracting plane inliers before segmentation would shred curved
    # structures: a cylinder's front strip is locally planar within the
    # plane detector's tolerance (measured: one "wall" claimed the
    # fronts of four separate columns), so plane points are NOT
    # pre-removed. Each spatial segment is classified by what it fits
    # best; plane-flat segments are structural material, not
    # parametric-component candidates.
    segments = segment_non_plane_points(
        reconstruction, plane_inlier_ids=set()
    )

    # ---- per-segment classification ----
    fitted = []
    structural_planes = 0
    for seg in segments:
        if seg.below_min:
            failure_regions.append(FailureRegion(
                kind="undersized_segment",
                reason=f"segment has {len(seg.member_ids)} points "
                       f"(below the fitting minimum)",
                subject_id=seg.segment_id,
            ))
            continue
        plane_rms = _plane_rms(seg.positions)
        fit = _best_fit(seg.positions)
        if fit is None or plane_rms <= fit.rms_residual_m:
            # Flat material (walls/floors): counted as structural
            # planes; their promotion is the compiler's promote_planes
            # path, not a parametric class. Not a failure -- just not
            # this benchmark's component set.
            structural_planes += 1
            continue
        fitted.append((seg, fit))

    # ---- components -> candidates -> priors ----
    observations = build_component_observations(
        [((seg.segment_id,), fit, seg.evidence_ids) for seg, fit in fitted],
        up=UP,
    )
    for obs in observations:
        if not obs.accepted:
            failure_regions.append(FailureRegion(
                kind="unaccepted_component",
                reason=obs.rejection_reason or "class gates not met",
                subject_id=obs.segment_id,
            ))
    candidates = resolve_components(observations)
    priors = detect_repetition(candidates)
    candidates = apply_repetition_priors(candidates, priors)

    # ---- WorldIR promotion + graph ----
    world = WorldIR(id=f"benchmark-{benchmark.structure_name.lower().replace(' ', '-')}")
    build_architectural_graph(candidates, world)

    # ---- measured report ----
    reg = get_default_registry()
    by_class: Dict[str, int] = {}
    by_phase: Dict[int, int] = {}
    for cand in candidates:
        by_class[cand.arch_class] = by_class.get(cand.arch_class, 0) + 1
        phase = reg.get(cand.arch_class).phase
        by_phase[phase] = by_phase.get(phase, 0) + 1
    tiers: Dict[str, int] = {}
    for cand in candidates:
        tier = ConfidenceTier.from_confidence(cand.effective_confidence())
        tiers[tier.value] = tiers.get(tier.value, 0) + 1

    return BenchmarkReport(
        structure_name=benchmark.structure_name,
        registration_status=reconstruction.registration_status,
        camera_count=len(reconstruction.camera_poses),
        point_count=len(reconstruction.points),
        plane_count=structural_planes,
        segment_count=len(segments),
        components_by_class=by_class,
        components_by_phase=by_phase,
        relationship_count=sum(
            len(e.relationships) for e in world.entities.values()
        ),
        tier_distribution=tiers,
        worldir_entity_count=len(world.entities),
        failure_regions=failure_regions,
    )


def _plane_rms(positions) -> float:
    """RMS distance of the segment's points to their best-fit plane --
    exactly sqrt of the smallest covariance eigenvalue. A flat segment
    has a tiny value; a curved shell cannot."""
    from perception.architecture.parametric import _covariance, _eigen_symmetric_3x3
    cov, _ = _covariance(list(positions))
    eigs = _eigen_symmetric_3x3(cov)
    return math.sqrt(max(0.0, eigs[0][0]))


def _best_fit(positions):
    """Try each parametric fit; return the one with the best
    (lowest) rms among those that accept the points, or None if every
    fit refuses. Deterministic order: cylinder, sphere, circle."""
    best = None
    for fit_fn in (fit_cylinder, fit_sphere):
        try:
            if fit_fn is fit_cylinder:
                fit = fit_fn(positions, up=UP)
            else:
                fit = fit_fn(positions)
        except (FitRefused, ValueError):
            continue
        if best is None or fit.rms_residual_m < best.rms_residual_m:
            best = fit
    # Circle fits need a plane normal; derive from the smallest
    # covariance eigenvector by passing normal=None.
    try:
        circ = fit_circle(positions, normal=None)
        if best is None or circ.rms_residual_m < best.rms_residual_m:
            best = circ
    except (FitRefused, ValueError):
        pass
    return best
