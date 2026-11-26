"""A5 Reliability Benchmark & A4-vs-A5 Comparison Engine.

Computes quantitative reliability metrics for A5 runs and compares them
against A4 baselines without conflating pipeline counts with ground-truth
physical accuracy:
  - Mask coverage rate & empty/rejected mask rates
  - Duplicate suppression rate & competing-hypothesis preservation
  - Cross-view identity outcome breakdown (confirmed / ambiguous / rejected)
  - Architectural & furniture false-positive rejection rates
  - GPU lifecycle & EvidenceMatrix persistence verification
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from perception.intelligence.production_pipeline_a5 import A5ProductionResult


@dataclass(frozen=True)
class A5ReliabilityMetrics:
    run_id: str
    raw_observations: int
    canonical_observations: int
    accepted_observations: int
    masks_produced: int
    empty_masks: int
    mask_coverage_fraction: float
    mean_mask_quality: float
    duplicate_suppressed_count: int
    duplicate_reduction_rate: float
    competing_hypotheses_count: int
    identity_confirmed_count: int
    identity_ambiguous_count: int
    identity_rejected_count: int
    arch_fp_rejected_count: int
    furniture_fp_rejected_count: int
    evidence_matrix_persisted: bool
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_a5_reliability_metrics(result: A5ProductionResult) -> A5ReliabilityMetrics:
    """Extract structured A5 reliability metrics from an ``A5ProductionResult``."""
    raw_n = int(result.dedup_result.summary.get("input_count", 0))
    canon_n = int(result.dedup_result.summary.get("canonical_count", 0))
    acc_n = len(result.bundle.observations)
    masks_n = result.segmentation_report.masks_produced
    mask_frac = round(masks_n / float(max(1, raw_n)), 4) if raw_n > 0 else 0.0
    persisted = bool(
        result.claude_verification.evidence_matrix_persisted
        if result.claude_verification
        else False
    )
    return A5ReliabilityMetrics(
        run_id=result.run_id,
        raw_observations=raw_n,
        canonical_observations=canon_n,
        accepted_observations=acc_n,
        masks_produced=masks_n,
        empty_masks=result.segmentation_report.empty_masks,
        mask_coverage_fraction=mask_frac,
        mean_mask_quality=result.segmentation_report.mean_mask_quality,
        duplicate_suppressed_count=int(result.dedup_result.summary.get("suppressed_count", 0)),
        duplicate_reduction_rate=float(result.dedup_result.summary.get("duplicate_reduction_rate", 0.0)),
        competing_hypotheses_count=int(result.dedup_result.summary.get("competing_hypotheses_count", 0)),
        identity_confirmed_count=result.identity_v2.confirmed_count,
        identity_ambiguous_count=result.identity_v2.ambiguous_count,
        identity_rejected_count=result.identity_v2.rejected_count,
        arch_fp_rejected_count=int(result.refinement_report.summary.get("rejected_architectural_count", 0)),
        furniture_fp_rejected_count=int(result.refinement_report.summary.get("rejected_furniture_count", 0)),
        evidence_matrix_persisted=persisted,
        wall_time_s=result.wall_time_s,
    )


def compare_a4_vs_a5_reliability(
    a4_baseline: dict[str, Any],
    a5_metrics: A5ReliabilityMetrics,
) -> dict[str, Any]:
    """Compare A5 reliability metrics against an A4 baseline summary dict."""
    a4_masks = int(a4_baseline.get("masks_produced", 0))
    a4_dups = int(a4_baseline.get("duplicate_warnings", 0))
    a4_confirmed = int(a4_baseline.get("confirmed_identities", 0))
    return {
        "a4_masks_produced": a4_masks,
        "a5_masks_produced": a5_metrics.masks_produced,
        "mask_gain": a5_metrics.masks_produced - a4_masks,
        "a4_duplicate_warnings": a4_dups,
        "a5_duplicates_suppressed": a5_metrics.duplicate_suppressed_count,
        "a4_confirmed_identities": a4_confirmed,
        "a5_confirmed_identities": a5_metrics.identity_confirmed_count,
        "a5_ambiguous_preserved": a5_metrics.identity_ambiguous_count,
        "a5_evidence_matrix_persisted": a5_metrics.evidence_matrix_persisted,
        "improved_mask_coverage": a5_metrics.masks_produced > a4_masks,
        "improved_duplicate_control": a5_metrics.duplicate_suppressed_count >= a4_dups,
    }


__all__ = [
    "A5ReliabilityMetrics",
    "compare_a4_vs_a5_reliability",
    "compute_a5_reliability_metrics",
]
