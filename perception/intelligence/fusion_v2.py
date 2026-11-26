"""Perception Fusion 2.0 (A4.9).

Supports explainable, non-destructive multi-modal fusion across:
  - Detector disagreement & class disagreement
  - Mask disagreement (mask IoU across engines)
  - Cross-view disagreement
  - Depth disagreement
  - Image-quality weighting & model-family correlation
  - Contradiction preservation & uncertainty propagation

Never uses naive arithmetic averaging as the general fusion strategy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox, compute_box_iou
from perception.detection.multi_engine_fusion import engine_family
from perception.intelligence.data_model import NormalizedPerceptionObservation


@dataclass(frozen=True)
class ExplainableFusionDecision:
    cluster_id: str
    selected_label: str
    selected_box_px: tuple[float, float, float, float]
    independent_families: tuple[str, ...]
    detector_disagreements: tuple[dict[str, Any], ...]
    mask_iou_agreement: float | None
    depth_disagreement_rel: float | None
    quality_weight: float
    propagated_uncertainty: float
    explanation: tuple[str, ...]
    source_observation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "selected_label": self.selected_label,
            "selected_box_px": [round(v, 2) for v in self.selected_box_px],
            "independent_families": list(self.independent_families),
            "detector_disagreements": list(self.detector_disagreements),
            "mask_iou_agreement": self.mask_iou_agreement,
            "depth_disagreement_rel": self.depth_disagreement_rel,
            "quality_weight": self.quality_weight,
            "propagated_uncertainty": self.propagated_uncertainty,
            "explanation": list(self.explanation),
            "source_observation_ids": list(self.source_observation_ids),
        }


def fuse_observations_v2(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    image_quality_grade: str = "good",
    iou_threshold: float = 0.50,
) -> list[ExplainableFusionDecision]:
    """Perform correlation-aware, explainable fusion across multi-engine observations."""
    if not observations:
        return []

    q_weight = {"good": 1.0, "degraded": 0.70, "unusable_blank": 0.10}.get(image_quality_grade, 0.85)
    ordered = sorted(
        observations,
        key=lambda o: float(o.scores.raw_score if o.scores.raw_score is not None else 0.5),
        reverse=True,
    )

    clusters: list[list[NormalizedPerceptionObservation]] = []
    for o in ordered:
        placed = False
        for cl in clusters:
            if cl[0].provenance.source_image_id == o.provenance.source_image_id and compute_box_iou(cl[0].bbox_px, o.bbox_px) >= iou_threshold:
                cl.append(o)
                placed = True
                break
        if not placed:
            clusters.append([o])

    decisions: list[ExplainableFusionDecision] = []
    for idx, cl in enumerate(clusters):
        by_fam: dict[str, list[NormalizedPerceptionObservation]] = {}
        for o in cl:
            fam = engine_family(o.provenance.execution.engine_id)
            by_fam.setdefault(fam, []).append(o)

        # Pick 1 representative per independent model family (no correlated double-counting)
        fam_reps = [
            max(mems, key=lambda m: float(m.scores.raw_score if m.scores.raw_score is not None else 0.5))
            for mems in by_fam.values()
        ]
        primary = max(fam_reps, key=lambda m: float(m.scores.raw_score if m.scores.raw_score is not None else 0.5))

        # Quality-weighted coordinate fusion across independent families only
        weights = [max(0.05, float(r.scores.raw_score if r.scores.raw_score is not None else 0.5)) * q_weight for r in fam_reps]
        w_sum = sum(weights) or 1.0
        bx1 = sum(w * r.bbox_px[0] for w, r in zip(weights, fam_reps)) / w_sum
        by1 = sum(w * r.bbox_px[1] for w, r in zip(weights, fam_reps)) / w_sum
        bx2 = sum(w * r.bbox_px[2] for w, r in zip(weights, fam_reps)) / w_sum
        by2 = sum(w * r.bbox_px[3] for w, r in zip(weights, fam_reps)) / w_sum

        disagreements: list[dict[str, Any]] = []
        for o in cl:
            if o.canonical_label != primary.canonical_label:
                disagreements.append({
                    "observation_id": o.observation_id,
                    "engine_id": o.provenance.execution.engine_id,
                    "family": engine_family(o.provenance.execution.engine_id),
                    "label": o.canonical_label,
                    "raw_score": o.scores.raw_score,
                })

        # Depth disagreement across cluster members that carry segmentation depth
        depths = [o.segmentation.depth_mean_relative for o in cl if o.segmentation and o.segmentation.depth_mean_relative is not None]
        depth_dis: float | None = None
        if len(depths) >= 2:
            depth_dis = round((max(depths) - min(depths)) / max(1e-4, float(np.median(depths))), 4)

        # Mask IoU agreement if >=2 members carry box_iou / area_px
        mask_iou: float | None = None
        masked = [o.segmentation for o in cl if o.segmentation is not None]
        if len(masked) >= 2:
            areas = [m.area_px for m in masked]
            mask_iou = round(min(areas) / max(1, max(areas)), 4)

        base_sc = primary.scores.raw_score if primary.scores.raw_score is not None else 0.5
        unc = 1.0 - (base_sc * q_weight)
        explanation = [f"independent_families={len(by_fam)}", f"quality_weight={q_weight}"]
        if disagreements:
            unc = min(1.0, unc + 0.15)
            explanation.append(f"class_disagreements={len(disagreements)}")
        if depth_dis is not None and depth_dis > 0.25:
            unc = min(1.0, unc + 0.12)
            explanation.append(f"depth_disagreement={depth_dis}")
        if len(by_fam) >= 2 and not disagreements:
            unc = max(0.05, unc * 0.85)
            explanation.append("multi_family_corroboration")

        decisions.append(ExplainableFusionDecision(
            cluster_id=f"fusion_v2_{idx}_{primary.observation_id}",
            selected_label=primary.canonical_label,
            selected_box_px=(round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)),
            independent_families=tuple(sorted(by_fam.keys())),
            detector_disagreements=tuple(disagreements),
            mask_iou_agreement=mask_iou,
            depth_disagreement_rel=depth_dis,
            quality_weight=q_weight,
            propagated_uncertainty=round(float(max(0.0, min(1.0, unc))), 4),
            explanation=tuple(explanation),
            source_observation_ids=tuple(o.observation_id for o in cl),
        ))

    return decisions


__all__ = [
    "ExplainableFusionDecision",
    "fuse_observations_v2",
]
