"""RECONSTRUCTION_CONTRACT_V1 (multi-agent execution contract: "A HANDOFF
IS A TESTED INTERFACE").

The reconstruction/pipeline result surface handed to frontend and
integration consumers (Antigravity studio, integration PR #51's WorldStore
bridge, CLI). Two rules define it:

  1. COMPOSE, do not compete: the payload types are exactly the canonical
     stage types this repo already owns (DepthMap, ReconstructedCameraPose,
     SegmentedRegion, FusedEstimate, EvidenceQualityReport,
     ReconstructionResult). This module adds NO new payload shape -- the
     tests prove identity with the canonical classes.

  2. EXPLICIT STATUS, no silent degradation: every stage result carries
     StageStatus -- complete | partial | unavailable | failed. Absence of
     data is never silently upgraded to success, and payload data is
     never carried under a failed/unavailable status (no zombie data).

The per-stage envelopes (DepthResult, PoseResult, ObservationResult,
FusionResult, QualityResult) wrap one canonical payload or an honest
absence. The top-level ReconstructionContract composes the orchestrator's
ReconstructionRun verbatim -- status and error text are never rewritten.

Every envelope serializes with to_dict().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from perception.depth.interface import DepthMap
from perception.quality.assessment import EvidenceQualityReport
from provenance import Uncertainty
from reconstruction.backend.interface import (
    ReconstructionResult as CanonicalReconstructionResult,
    ReconstructedCameraPose,
)
from reconstruction.orchestrator import ReconstructionRun

CONTRACT_VERSION = "1.0"


class StageStatus(str, Enum):
    """The contract's status vocabulary: exactly four states.

    UNAVAILABLE = the stage never ran (dependency/model/backend missing).
    FAILED = the stage ran and failed. PARTIAL = some, not all, of the
    stage's output exists. COMPLETE = the stage's full output exists.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"

    def __str__(self) -> str:  # keep f-strings and to_dict output clean
        return self.value


class ContractError(ValueError):
    """A contract envelope violates the honesty invariants."""


_PAYLOAD_STATUSES = (StageStatus.COMPLETE, StageStatus.PARTIAL)
_EMPTY_STATUSES = (StageStatus.UNAVAILABLE, StageStatus.FAILED)


def _validate_payload_or_absence(status: StageStatus, payload: Any,
                                 payload_name: str) -> None:
    """The shared honesty invariant for every stage envelope.

    complete/partial REQUIRE the payload; unavailable/failed FORBID it
    (data under a failed status is exactly the fabricated-success the
    project conventions forbid).
    """
    if status in _PAYLOAD_STATUSES and payload is None:
        raise ContractError(
            f"status '{status.value}' requires a {payload_name}; "
            f"absence of data must use status 'unavailable' or 'failed'"
        )
    if status in _EMPTY_STATUSES and payload is not None:
        raise ContractError(
            f"{payload_name} cannot be carried under status "
            f"'{status.value}'; a failed/unavailable stage has no data"
        )


@dataclass(frozen=True)
class DepthResult:
    """Depth stage outcome for one evidence item (canonical DepthMap)."""

    evidence_id: str
    status: StageStatus
    depth_map: Optional[DepthMap] = None
    detail: str = ""

    def __post_init__(self):
        _validate_payload_or_absence(self.status, self.depth_map, "depth_map")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"evidence_id": self.evidence_id,
                             "status": self.status.value}
        if self.depth_map is not None:
            d["depth_map"] = {
                "evidence_id": self.depth_map.evidence_id,
                "width": self.depth_map.width,
                "height": self.depth_map.height,
                "unit": self.depth_map.unit,
                "rows": len(self.depth_map.values),
            }
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass(frozen=True)
class PoseResult:
    """Pose stage outcome for one evidence item (canonical camera pose)."""

    evidence_id: str
    status: StageStatus
    pose: Optional[ReconstructedCameraPose] = None
    detail: str = ""

    def __post_init__(self):
        _validate_payload_or_absence(self.status, self.pose, "pose")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"evidence_id": self.evidence_id,
                             "status": self.status.value}
        if self.pose is not None:
            d["pose"] = {
                "evidence_id": self.pose.evidence_id,
                "position": tuple(self.pose.position),
                "rotation": tuple(self.pose.rotation),
            }
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass(frozen=True)
class ObservationResult:
    """Detection/segmentation stage outcome for one evidence item.

    Empty detections/segmentations under COMPLETE are a legitimate
    outcome (the stage ran and found nothing) -- but the same lists under
    FAILED/UNAVAILABLE are forbidden (no zombie data).
    """

    evidence_id: str
    status: StageStatus
    detections: Optional[List[Any]] = None
    segmentations: Optional[List[Any]] = None
    detail: str = ""

    def __post_init__(self):
        if self.status in _EMPTY_STATUSES:
            if self.detections:
                raise ContractError(
                    "detections cannot be carried under status "
                    f"'{self.status.value}'; a failed/unavailable stage "
                    "has no data")
            if self.segmentations:
                raise ContractError(
                    "segmentations cannot be carried under status "
                    f"'{self.status.value}'; a failed/unavailable stage "
                    "has no data")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"evidence_id": self.evidence_id,
                             "status": self.status.value,
                             "detections": list(self.detections or []),
                             "segmentations": list(self.segmentations or [])}
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass(frozen=True)
class FusionResult:
    """Fusion stage outcome for one quantity (canonical FusedEstimate)."""

    quantity: str
    status: StageStatus
    fused: Optional[Any] = None  # reconstruction.fusion.fusion.FusedEstimate
    detail: str = ""

    def __post_init__(self):
        _validate_payload_or_absence(self.status, self.fused, "fused")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"quantity": self.quantity,
                             "status": self.status.value}
        if self.fused is not None:
            d["fused"] = {
                "value": self.fused.value,
                "unit": str(self.fused.unit),
                "precision": self.fused.precision,
                "confidence": self.fused.confidence,
                "provenance": str(self.fused.provenance.value)
                if hasattr(self.fused.provenance, "value")
                else str(self.fused.provenance),
                "method": self.fused.method,
                "conflict_count": len(self.fused.conflicts),
            }
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass(frozen=True)
class QualityResult:
    """Evidence-quality stage outcome (canonical EvidenceQualityReport)."""

    status: StageStatus
    report: Optional[EvidenceQualityReport] = None
    detail: str = ""

    def __post_init__(self):
        _validate_payload_or_absence(self.status, self.report, "report")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status.value}
        if self.report is not None:
            d["report"] = self.report.to_dict()
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass(frozen=True)
class ReconstructionContract:
    """The top-level V1 surface: one orchestrated reconstruction run,
    composed VERBATIM from the orchestrator's own records.

    status is the run's diagnostics.final_status ("success" | "partial" |
    "failed") -- never rewritten, never upgraded. detail preserves the
    run's error text verbatim. result is the canonical ReconstructionResult
    exactly as the backend produced it.
    """

    contract_version: str
    status: str  # orchestrator vocabulary: "success" | "partial" | "failed"
    result: Optional[CanonicalReconstructionResult]
    diagnostics: Dict[str, Any]
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "contract_version": self.contract_version,
            "status": self.status,
            "diagnostics": dict(self.diagnostics),
        }
        if self.result is not None:
            d["result"] = {
                "registration_status": self.result.registration_status,
                "point_count": len(self.result.points),
                "camera_pose_count": len(self.result.camera_poses),
            }
        if self.detail:
            d["detail"] = self.detail
        return d


def contract_reconstruction_from_run(run: ReconstructionRun) -> ReconstructionContract:
    """Compose the V1 surface from the orchestrator's ReconstructionRun.

    Everything is taken from the run itself: status = diagnostics'
    final_status, detail = diagnostics' error, result = the run's result
    when the run has one. A failed run carries its diagnostics (the
    structured failure story) but never a result payload.
    """
    final_status = run.diagnostics.final_status
    if final_status not in ("success", "partial", "failed"):
        raise ContractError(
            f"unknown orchestrator final_status '{final_status}'")
    return ReconstructionContract(
        contract_version=CONTRACT_VERSION,
        status=final_status,
        result=run.result if final_status != "failed" else None,
        diagnostics=run.diagnostics.to_dict(),
        detail=run.diagnostics.error,
    )
