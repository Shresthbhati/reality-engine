"""Production Diagnostics 2.0 (A4.13).

Comprehensive diagnostic engine checking:
  - False positives / false negatives (when ground truth exists)
  - Empty detections & duplicate instances
  - Broken masks & missing depth
  - Invalid coordinates & inconsistent labels
  - Cross-view identity failures
  - Model timeouts, GPU memory pressure, and dataset corruption
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox, evaluate_detections_against_ground_truth
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.datasets.indoor_registry import DatasetIntegrityReport
from perception.intelligence.architectural import ArchitecturalAnalysisResult
from perception.intelligence.data_model import PerceptionBundle
from perception.intelligence.depth_aware import DepthAwarePerceptionResult
from perception.intelligence.diagnostics import DiagnosticFinding, DiagnosticReport, run_perception_diagnostics
from perception.intelligence.identity import IdentityResolutionReport


def run_production_diagnostics_v2(
    bundle: PerceptionBundle,
    *,
    image_qualities: Mapping[str, Mapping[str, Any]] | None = None,
    arch_result: ArchitecturalAnalysisResult | None = None,
    identity_report: IdentityResolutionReport | None = None,
    depth_results: Sequence[DepthAwarePerceptionResult] = (),
    dataset_integrity: DatasetIntegrityReport | None = None,
    gt_by_image: Mapping[str, Sequence[GroundTruthBox]] | None = None,
    failed_engines: Mapping[str, str] | None = None,
    timed_out_engines: Sequence[str] = (),
    peak_vram_mb: float | None = None,
    vram_limit_mb: float = 5800.0,
) -> DiagnosticReport:
    """Run base perception diagnostics plus depth, timeout, VRAM, dataset, and GT FP/FN checks."""
    base = run_perception_diagnostics(
        bundle,
        image_qualities=image_qualities,
        arch_result=arch_result,
        identity_report=identity_report,
        failed_engines=failed_engines,
    )
    findings = list(base.findings)

    for eid in timed_out_engines:
        findings.append(DiagnosticFinding(
            code="model_timeout",
            severity="error",
            message=f"Engine {eid!r} exceeded its wall-clock timeout.",
            affected_ids=(eid,),
            remediation="Reduce tile density or run sequentially on smaller batch.",
        ))

    if peak_vram_mb is not None and peak_vram_mb >= vram_limit_mb:
        findings.append(DiagnosticFinding(
            code="gpu_memory_pressure",
            severity="warning",
            message=f"Peak VRAM {peak_vram_mb:.1f} MB approached limit {vram_limit_mb:.1f} MB.",
            remediation="Unload models between stages and reduce batch/tile resolution.",
        ))

    for dr in depth_results:
        missing_depth = [s.observation_id for s in dr.object_depths if s.median_depth is None]
        if missing_depth:
            findings.append(DiagnosticFinding(
                code="missing_depth_association",
                severity="warning",
                message=f"{len(missing_depth)} objects in {dr.image_id!r} lack valid depth.",
                affected_ids=tuple(missing_depth[:10]),
                remediation="Verify depth map validity mask and camera coverage.",
            ))

    if dataset_integrity is not None and not dataset_integrity.is_healthy:
        findings.append(DiagnosticFinding(
            code="dataset_corruption_or_leakage",
            severity="error",
            message=f"Dataset {dataset_integrity.dataset_id!r} failed integrity (corrupt={len(dataset_integrity.corrupt_samples)}, leakage={dataset_integrity.leakage_detected}).",
            remediation="Repair corrupt annotations or regenerate scene-disjoint splits.",
        ))

    if gt_by_image:
        preds: dict[str, list[DetectionBox]] = {}
        for o in bundle.observations:
            preds.setdefault(o.provenance.source_image_id, []).append(
                DetectionBox(o.observation_id, o.canonical_label, o.bbox_px, o.scores.raw_score or 0.5)
            )
        m = evaluate_detections_against_ground_truth(preds, dict(gt_by_image))
        if m["total_fp"] > 0 or m["total_fn"] > 0:
            findings.append(DiagnosticFinding(
                code="ground_truth_fp_fn_summary",
                severity="info",
                message=f"GT evaluation: TP={m['total_tp']}, FP={m['total_fp']}, FN={m['total_fn']}.",
                remediation="Inspect per-class breakdown for threshold or prompt tuning.",
            ))

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    status = "failed" if counts.get("error", 0) > 0 else ("degraded" if counts.get("warning", 0) > 0 else "healthy")
    return DiagnosticReport(status=status, findings=tuple(findings), counts_by_severity=counts)


__all__ = ["run_production_diagnostics_v2"]
