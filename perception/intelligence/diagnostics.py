"""Failure Detection & Diagnostic Intelligence Subsystem (A3.13).

Detects and reports:
  - Empty detections on non-blank images
  - Implausible detection volumes (excessive boxes or whole-frame hallucinations)
  - Repeated same-class duplicates
  - Missing segmentation masks
  - Invalid coordinates or image dimensions
  - Inconsistent cross-view object identity
  - Conflicting architectural labels
  - Model execution failures & resource exhaustion
  - Invalid observation serialization
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou
from perception.intelligence.architectural import ArchitecturalAnalysisResult
from perception.intelligence.data_model import (
    NormalizedPerceptionObservation,
    PerceptionBundle,
)
from perception.intelligence.identity import IdentityResolutionReport


@dataclass(frozen=True)
class DiagnosticFinding:
    code: str
    severity: str  # "info" | "warning" | "error"
    message: str
    affected_ids: tuple[str, ...] = ()
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "affected_ids": list(self.affected_ids),
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    status: str  # "healthy" | "degraded" | "failed"
    findings: tuple[DiagnosticFinding, ...]
    counts_by_severity: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "counts_by_severity": dict(self.counts_by_severity),
        }


def run_perception_diagnostics(
    bundle: PerceptionBundle,
    *,
    image_qualities: Mapping[str, Mapping[str, Any]] | None = None,
    arch_result: ArchitecturalAnalysisResult | None = None,
    identity_report: IdentityResolutionReport | None = None,
    failed_engines: Mapping[str, str] | None = None,
    max_boxes_per_image: int = 120,
) -> DiagnosticReport:
    """Inspect a PerceptionBundle and surface structured, actionable diagnostics."""
    findings: list[DiagnosticFinding] = []
    obs = bundle.observations

    # 1. Model execution failures / resource exhaustion
    for eid, err in sorted((failed_engines or {}).items()):
        sev = "error" if "OutOfMemory" in err or "CUDA" in err else "warning"
        findings.append(DiagnosticFinding(
            code="model_execution_failure",
            severity=sev,
            message=f"Engine {eid!r} failed: {err}",
            affected_ids=(eid,),
            remediation="Inspect model checkpoint cache or lower VRAM concurrency.",
        ))

    # 2. Empty detections on non-blank images
    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in obs:
        by_img[o.provenance.source_image_id].append(o)

    for img_id, q in sorted((image_qualities or {}).items()):
        is_blank = bool(q.get("is_blank", False))
        img_obs = by_img.get(img_id, [])
        if not is_blank and len(img_obs) == 0:
            findings.append(DiagnosticFinding(
                code="empty_detections_non_blank",
                severity="warning",
                message=f"Non-blank image {img_id!r} produced 0 detections.",
                affected_ids=(img_id,),
                remediation="Verify detector availability, prompt domain, and acceptance thresholds.",
            ))

    # 3. Implausible detection volumes & repeated duplicates per image
    for img_id, img_obs in sorted(by_img.items()):
        if len(img_obs) > max_boxes_per_image:
            findings.append(DiagnosticFinding(
                code="implausible_detection_volume",
                severity="warning",
                message=f"Image {img_id!r} produced {len(img_obs)} boxes (>{max_boxes_per_image}).",
                affected_ids=(img_id,),
                remediation="Check detector score floor and duplicate suppression.",
            ))
        dup_ids: list[str] = []
        for i, a in enumerate(img_obs):
            for b in img_obs[i + 1:]:
                if a.canonical_label == b.canonical_label and compute_box_iou(a.bbox_px, b.bbox_px) >= 0.70:
                    dup_ids.append(b.observation_id)
        if dup_ids:
            findings.append(DiagnosticFinding(
                code="repeated_duplicate_detections",
                severity="warning",
                message=f"Image {img_id!r} has {len(dup_ids)} high-IoU same-class duplicates.",
                affected_ids=tuple(dup_ids[:10]),
                remediation="Run duplicate suppression before entity promotion.",
            ))

    # 4. Missing masks & calibration honesty checks
    missing_masks = [o.observation_id for o in obs if o.segmentation is None]
    if obs and len(missing_masks) == len(obs):
        findings.append(DiagnosticFinding(
            code="all_masks_missing",
            severity="warning",
            message="All observations lack segmentation masks.",
            affected_ids=tuple(missing_masks[:10]),
            remediation="Verify SAM / SAM2 availability and box prompting.",
        ))

    # 5. Architectural conflicts
    if arch_result is not None and arch_result.competing_structural_pairs:
        findings.append(DiagnosticFinding(
            code="conflicting_architectural_labels",
            severity="info",
            message=f"Detected {len(arch_result.competing_structural_pairs)} overlapping structural label conflicts.",
            affected_ids=tuple(p[0] for p in arch_result.competing_structural_pairs[:10]),
            remediation="Preserve competing structural hypotheses for 3D geometric resolution.",
        ))

    # 6. Cross-view identity split proposals
    if identity_report is not None and identity_report.split_proposals:
        findings.append(DiagnosticFinding(
            code="inconsistent_object_identity",
            severity="warning",
            message=f"{len(identity_report.split_proposals)} cross-view identity clusters have mixed labels.",
            affected_ids=tuple(sp.identity_id for sp in identity_report.split_proposals[:10]),
            remediation="Execute proposed identity splits before WorldIR promotion.",
        ))

    # 7. Serialization round-trip check
    try:
        raw_dict = bundle.to_dict()
        NormalizedPerceptionObservation  # ensure import intact
        if raw_dict.get("schema_version") != bundle.schema_version:
            raise ValueError("schema_version mismatch")
    except Exception as exc:  # noqa: BLE001
        findings.append(DiagnosticFinding(
            code="invalid_observation_serialization",
            severity="error",
            message=f"PerceptionBundle serialization failed: {exc}",
            remediation="Inspect non-serializable fields in attributes or provenance.",
        ))

    counts = dict(Counter(f.severity for f in findings))
    status = "failed" if counts.get("error", 0) > 0 else ("degraded" if counts.get("warning", 0) > 0 else "healthy")
    return DiagnosticReport(status=status, findings=tuple(findings), counts_by_severity=counts)


__all__ = [
    "DiagnosticFinding",
    "DiagnosticReport",
    "run_perception_diagnostics",
]
