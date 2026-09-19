"""Quality-aware compute admission (ROBUSTNESS directive P1).

The orchestrator's input gate is batch-level only (counts, duplicate
ids). The item-level quality signals recorded by the importer (blur /
exposure metrics from `_measure_photo_quality`) are never consulted
before reconstruction spend. This module composes
`reconstruction.robustness.classify_evidence_items` with the
orchestrator -- the orchestrator itself stays untouched (same
discipline as orchestrator_consistency).

Semantics (evidence-retention honesty):
  - ACCEPTED/DEGRADED proceed (degraded carry their measured reason)
  - REJECTED/FAILED are EXCLUDED from reconstruction spend; the
    exclusion is a recorded fact with its reason -- never a silent drop
  - UNRESOLVED proceed WITH diagnostics: absent metrics must not veto
    otherwise-good captures (the evidence stays, flagged)
  - fewer than MIN_IMAGE_EVIDENCE admitted images refuses the batch
    (AdmissionError) naming the counts -- never a silent pass-through

Backend economics note: reconstruction is the expensive stage this
gate protects. Admission is deliberately CHEAP (dict lookups +
comparisons) so it saves spend rather than adding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from reconstruction.backend.interface import IReconstructionBackend, ReconstructionResult
from reconstruction.orchestrator import (
    MIN_IMAGE_EVIDENCE,
    ReconstructionOrchestrator,
)
from reconstruction.robustness import (
    ACCEPTED,
    DEGRADED,
    FAILED,
    REJECTED,
    UNRESOLVED,
    AdmissionReport,
    classify_evidence_items,
)


class AdmissionError(ValueError):
    """The admitted subset cannot support reconstruction."""


@dataclass(frozen=True)
class AdmissionDecision:
    """The recorded outcome of the admission pass over one batch."""

    classification: AdmissionReport
    admitted_ids: List[str] = field(default_factory=list)
    excluded_ids: List[str] = field(default_factory=list)
    proceeded_unresolved: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "classification": self.classification.to_dict(),
            "admitted": self.admitted_ids,
            "excluded": self.excluded_ids,
            "proceeded_unresolved": self.proceeded_unresolved,
        }


def admit_for_reconstruction(
    evidence: List,
    *,
    blur_reject_laplacian: float = 50.0,
    blur_degrade_laplacian: float = 150.0,
    clip_reject_fraction: float = 0.5,
    clip_degrade_fraction: float = 0.25,
) -> Tuple[List, AdmissionDecision]:
    """Split a batch into the admitted subset and the recorded rest.

    Returns (admitted_items, decision). Raises AdmissionError when the
    admitted subset is below the orchestrator's minimum image count.
    """
    classification = classify_evidence_items(
        evidence,
        blur_reject_laplacian=blur_reject_laplacian,
        blur_degrade_laplacian=blur_degrade_laplacian,
        clip_reject_fraction=clip_reject_fraction,
        clip_degrade_fraction=clip_degrade_fraction,
    )
    outcome_by_id = {a.evidence_id: a.outcome for a in classification.admissions}

    admitted: List = []
    excluded: List[str] = []
    unresolved: List[str] = []
    for item in evidence:
        outcome = outcome_by_id.get(item.id)
        if outcome in (REJECTED, FAILED):
            excluded.append(item.id)
        else:
            admitted.append(item)
            if outcome == UNRESOLVED:
                unresolved.append(item.id)
            # ACCEPTED and DEGRADED proceed plainly; degraded reasons
            # live in the classification report carried on the decision.

    image_admitted = [
        i for i in admitted
        if str(getattr(getattr(i, "kind", None), "value", i.kind)).lower()
        in ("photo", "video")
    ]
    if len(image_admitted) < MIN_IMAGE_EVIDENCE:
        counts = classification.counts
        raise AdmissionError(
            "admission left too little usable evidence to reconstruct: "
            f"{len(image_admitted)} admitted image item(s), need >= "
            f"{MIN_IMAGE_EVIDENCE}; excluded={len(excluded)} "
            f"(rejected={counts[REJECTED]}, failed={counts[FAILED]}) "
            f"of {len(evidence)} submitted"
        )

    decision = AdmissionDecision(
        classification=classification,
        admitted_ids=[i.id for i in admitted],
        excluded_ids=excluded,
        proceeded_unresolved=unresolved,
    )
    return admitted, decision


def reconstruct_with_admission(
    backend: IReconstructionBackend,
    evidence: List,
) -> Tuple[object, AdmissionDecision]:
    """Admit -> orchestrate the admitted subset -> record the decision.

    The run's diagnostics carry the admission summary under
    evidence_validation['admitted'] (plus 'submitted' and 'excluded')
    so downstream consumers see the gate without re-deriving it.
    """
    admitted, decision = admit_for_reconstruction(evidence)
    orchestrator = ReconstructionOrchestrator([backend])
    run = orchestrator.run(admitted)

    validation = dict(run.diagnostics.evidence_validation)
    validation["submitted"] = len(evidence)
    validation["admitted"] = len(admitted)
    validation["excluded"] = len(decision.excluded_ids)
    object.__setattr__(run.diagnostics, "evidence_validation", validation)
    return run, decision
