"""Reconstruction robustness classification (ROBUSTNESS directive P0).

Real captures are large, messy, incomplete, and imperfect. This module
sits between evidence import and reconstruction and makes outcomes
EXPLICIT at two levels:

  - item level (`classify_evidence_items`): every piece of evidence ends
    ACCEPTED / REJECTED / DEGRADED / UNRESOLVED / FAILED with a reason.
    Nothing is silently discarded; low-quality evidence is degraded or
    unresolved -- never dropped without a record, never upgraded into
    certainty.

  - run level (`classify_reconstruction_run`): a reconstruction result's
    registration status maps onto the same vocabulary, so a failed run
    can never masquerade as success downstream.

Outcome semantics (exactly five, no sixth "fine-ish" state):
  ACCEPTED    measured metrics within thresholds; usable as-is
  REJECTED    measured disqualifying defect (e.g. hard blur, hard clipping)
  DEGRADED    usable but impaired (borderline metrics); carries the reason
  UNRESOLVED  cannot decide from available data (metrics absent)
  FAILED      measurement itself failed (decode error recorded by importer)

Metric schema mirrors evidence/importers._measure_photo_quality:
  laplacian_variance (blur, higher = sharper), luma_mean,
  clipped_fraction, measured (1.0 when pixel metrics exist).
Thresholds are constructor-injectable so callers with different capture
domains (indoor close-range vs drone altitude) can tune without forking
the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from evidence.session import EvidenceItem

ACCEPTED = "accepted"
REJECTED = "rejected"
DEGRADED = "degraded"
UNRESOLVED = "unresolved"
FAILED = "failed"
OUTCOMES = (ACCEPTED, DEGRADED, UNRESOLVED, FAILED, REJECTED)


#: Default admission thresholds. Blur via variance of the Laplacian on
#: 512px-downscaled luma (importer's normalization): sub-~50 is visibly
#: out of focus, 50-150 is soft. Clipping = fraction of pixels pinned at
#: either luma rail.
DEFAULT_BLUR_REJECT_LAPLACIAN = 50.0
DEFAULT_BLUR_DEGRADE_LAPLACIAN = 150.0
DEFAULT_CLIP_REJECT_FRACTION = 0.5
DEFAULT_CLIP_DEGRADE_FRACTION = 0.25


@dataclass(frozen=True)
class ItemAdmission:
    """One evidence item's explicit outcome. `reason` is always
    non-empty and names the measured value when one exists."""

    evidence_id: str
    outcome: str
    reason: str

    def to_dict(self) -> dict:
        return {"evidence_id": self.evidence_id, "outcome": self.outcome, "reason": self.reason}


@dataclass(frozen=True)
class AdmissionReport:
    """All items, all outcomes, nothing omitted."""

    admissions: List[ItemAdmission] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def counts(self) -> Dict[str, int]:
        counts = {outcome: 0 for outcome in OUTCOMES}
        for a in self.admissions:
            counts[a.outcome] += 1
        return counts

    @property
    def accepted_ids(self) -> List[str]:
        return [a.evidence_id for a in self.admissions if a.outcome == ACCEPTED]

    @property
    def rejected_ids(self) -> List[str]:
        return [a.evidence_id for a in self.admissions if a.outcome == REJECTED]

    @property
    def degraded_ids(self) -> List[str]:
        return [a.evidence_id for a in self.admissions if a.outcome == DEGRADED]

    @property
    def unresolved_ids(self) -> List[str]:
        return [a.evidence_id for a in self.admissions if a.outcome == UNRESOLVED]

    @property
    def failed_ids(self) -> List[str]:
        return [a.evidence_id for a in self.admissions if a.outcome == FAILED]

    def to_dict(self) -> dict:
        return {
            "admissions": [a.to_dict() for a in self.admissions],
            "counts": dict(self.counts),
            "thresholds": dict(self.thresholds),
        }


@dataclass(frozen=True)
class RunAdmission:
    """A reconstruction run's explicit outcome over the same vocabulary."""

    outcome: str
    reason: str
    registration_status: str

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "registration_status": self.registration_status,
        }


def _quality_metrics(item: EvidenceItem) -> Optional[Dict[str, float]]:
    """The importer stores measured pixel metrics under metadata
    'quality'. Absent key -> None (UNRESOLVED path), never a guess."""
    metrics = item.metadata.get("quality")
    if isinstance(metrics, dict):
        return metrics
    return None


def classify_evidence_items(
    items: Sequence[EvidenceItem],
    *,
    blur_reject_laplacian: float = DEFAULT_BLUR_REJECT_LAPLACIAN,
    blur_degrade_laplacian: float = DEFAULT_BLUR_DEGRADE_LAPLACIAN,
    clip_reject_fraction: float = DEFAULT_CLIP_REJECT_FRACTION,
    clip_degrade_fraction: float = DEFAULT_CLIP_DEGRADE_FRACTION,
) -> AdmissionReport:
    """Classify every item; input order preserved; deterministic."""
    thresholds = {
        "blur_reject_laplacian": blur_reject_laplacian,
        "blur_degrade_laplacian": blur_degrade_laplacian,
        "clip_reject_fraction": clip_reject_fraction,
        "clip_degrade_fraction": clip_degrade_fraction,
    }
    admissions: List[ItemAdmission] = []
    for item in items:
        admissions.append(
            _classify_one(item, thresholds)
        )
    return AdmissionReport(admissions=admissions, thresholds=thresholds)


def _classify_one(item: EvidenceItem, thresholds: Dict[str, float]) -> ItemAdmission:
    metrics = _quality_metrics(item)

    if metrics is None:
        return ItemAdmission(
            evidence_id=item.id,
            outcome=UNRESOLVED,
            reason="no pixel-quality metrics recorded; cannot classify",
        )

    if not metrics.get("measured"):
        note = metrics.get("quality_note", "pixel metrics unmeasured")
        return ItemAdmission(
            evidence_id=item.id,
            outcome=FAILED,
            reason=str(note),
        )

    laplacian = metrics.get("laplacian_variance")
    clipped = metrics.get("clipped_fraction")

    # REJECTED beats DEGRADED: report the strongest measured defect.
    blur_reject = (
        isinstance(laplacian, (int, float))
        and laplacian < thresholds["blur_reject_laplacian"]
    )
    clip_reject = (
        isinstance(clipped, (int, float))
        and clipped > thresholds["clip_reject_fraction"]
    )
    if blur_reject or clip_reject:
        detail = []
        if blur_reject:
            detail.append(f"blur: laplacian_variance {laplacian:g} below "
                          f"reject threshold {thresholds['blur_reject_laplacian']:g}")
        if clip_reject:
            detail.append(f"clipping: clipped_fraction {clipped:g} above "
                          f"reject threshold {thresholds['clip_reject_fraction']:g}")
        return ItemAdmission(evidence_id=item.id, outcome=REJECTED, reason="; ".join(detail))

    blur_degrade = (
        isinstance(laplacian, (int, float))
        and laplacian < thresholds["blur_degrade_laplacian"]
    )
    clip_degrade = (
        isinstance(clipped, (int, float))
        and clipped > thresholds["clip_degrade_fraction"]
    )
    if blur_degrade or clip_degrade:
        detail = []
        if blur_degrade:
            detail.append(f"soft blur: laplacian_variance {laplacian:g} below "
                          f"degrade threshold {thresholds['blur_degrade_laplacian']:g}")
        if clip_degrade:
            detail.append(f"partial clipping: clipped_fraction {clipped:g} above "
                          f"degrade threshold {thresholds['clip_degrade_fraction']:g}")
        return ItemAdmission(evidence_id=item.id, outcome=DEGRADED, reason="; ".join(detail))

    return ItemAdmission(
        evidence_id=item.id,
        outcome=ACCEPTED,
        reason="measured metrics within thresholds",
    )


def classify_reconstruction_run(result) -> RunAdmission:
    """Map a ReconstructionResult's registration_status onto the shared
    outcome vocabulary. Unknown statuses refuse to classify rather than
    being silently accepted."""
    status = getattr(result, "registration_status", None)
    if status == "success":
        return RunAdmission(ACCEPTED, "registration_status=success", status)
    if status == "partial":
        return RunAdmission(DEGRADED, "registration_status=partial: result usable but incomplete", status)
    if status == "failed":
        return RunAdmission(FAILED, "registration_status=failed: no trustworthy registration", status)
    return RunAdmission(
        UNRESOLVED,
        f"unknown registration_status {status!r}; refusing to classify",
        status,
    )
