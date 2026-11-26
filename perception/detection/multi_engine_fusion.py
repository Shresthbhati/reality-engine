"""Multi-engine detection routing, correlation-aware fusion, and cross-view identity (A4, A5, A7, Priority 2).

Implements:
  1. Task-specific detector routing (avoiding blind execution of every model).
  2. Correlation-aware multi-engine evidence fusion:
     - Groups engines by model family so correlated models do not double-count.
     - Preserves competing class hypotheses when engines disagree.
     - Preserves raw detector scores and keeps calibrated_confidence=None unless
       backed by a fitted calibrator.
     - Incorporates image-quality telemetry into uncertainty without fabricating scores.
  3. Ambiguity-preserving cross-view object association:
     - Uses semantic compatibility, appearance/embedding similarity, geometric
       compatibility (when cameras are available), and occlusion/truncation state.
     - Never forces a merge when two visually similar objects (e.g. two chairs)
       are ambiguous; preserves candidate links for downstream reconstruction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    LearnedConfidenceCalibrator,
    compute_box_iou,
)
from engine.multiengine.spatial_intelligence.taxonomy import (
    are_semantically_compatible,
    resolve_taxonomy_class,
)

ENGINE_FAMILIES: dict[str, str] = {
    "grounding_dino": "grounding-dino",
    "grounding_dino_tiled": "grounding-dino",
    "grounding_dino_subpart": "grounding-dino",
    "owlvit": "owl-vit",
    "florence2": "florence2",
    "mask_rcnn": "maskrcnn",
    "maskrcnn": "maskrcnn",
    "yolo": "yolo",
    "yolo_world": "yolo",
    "rt_detr": "rtdetr",
    "detectron2": "detectron2",
}


def engine_family(engine_id: str) -> str:
    base = str(engine_id).split(":")[0].lower()
    for prefix, fam in ENGINE_FAMILIES.items():
        if base == prefix or base.startswith(prefix + "_"):
            return fam
    return base


def select_detection_engines(
    available_engines: Sequence[str],
    *,
    budget: str = "balanced",
    is_low_contrast: bool = False,
) -> list[str]:
    """Task-specific detector routing without blindly executing every model.

    - ``fast``: primary open-vocab detector (``grounding_dino``) or ``mask_rcnn``.
    - ``balanced``: ``grounding_dino`` + ``mask_rcnn`` (two independent model families).
    - ``forensic`` or ``is_low_contrast=True``: adds ``florence2`` as a generative VLM second opinion.
    """
    avail = list(dict.fromkeys(available_engines))
    avail_set = set(avail)
    selected: list[str] = []

    def _add(eid: str) -> None:
        if eid in avail_set and eid not in selected:
            selected.append(eid)

    if budget == "fast":
        _add("grounding_dino")
        if not selected:
            _add("mask_rcnn")
        if not selected:
            _add("owlvit")
        return selected

    _add("grounding_dino")
    _add("mask_rcnn")
    if budget == "forensic" or is_low_contrast:
        _add("florence2")
    if not selected and avail:
        selected.append(avail[0])
    return selected


@dataclass(frozen=True)
class FusedDetectionHypothesis:
    """Multi-engine fused detection preserving competing labels, provenance, and calibration state."""

    box_id: str
    primary_label: str
    canonical_label: str
    box_2d: tuple[float, float, float, float]
    raw_scores: dict[str, float | None]
    supporting_engines: tuple[str, ...]
    supporting_families: tuple[str, ...]
    independent_family_count: int
    competing_hypotheses: tuple[dict[str, Any], ...] = ()
    calibrated_confidence: float | None = None
    calibration_status: str = "uncalibrated"
    uncertainty: float | None = None
    source_image_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_id": self.box_id,
            "primary_label": self.primary_label,
            "canonical_label": self.canonical_label,
            "box_2d": [round(v, 2) for v in self.box_2d],
            "raw_scores": dict(self.raw_scores),
            "supporting_engines": list(self.supporting_engines),
            "supporting_families": list(self.supporting_families),
            "independent_family_count": self.independent_family_count,
            "competing_hypotheses": list(self.competing_hypotheses),
            "calibrated_confidence": self.calibrated_confidence,
            "calibration_status": self.calibration_status,
            "uncertainty": self.uncertainty,
            "source_image_id": self.source_image_id,
            "provenance": dict(self.provenance),
        }


def fuse_multi_engine_detections(
    engine_detections: Mapping[str, Sequence[DetectionBox]],
    *,
    iou_threshold: float = 0.50,
    is_low_contrast: bool = False,
    calibrator: LearnedConfidenceCalibrator | None = None,
) -> list[FusedDetectionHypothesis]:
    """Correlation-aware multi-engine fusion (no blind averaging).

    - Clusters overlapping boxes across engines.
    - Deduplicates votes within the same model family before counting independent corroboration.
    - Preserves semantically incompatible labels on overlapping boxes as ``competing_hypotheses``.
    - Keeps ``calibrated_confidence=None`` unless a fitted ``calibrator`` is supplied.
    """
    flat: list[tuple[str, DetectionBox]] = []
    for eid, dets in engine_detections.items():
        for d in dets:
            flat.append((eid, d))

    # Sort descending by raw score (None treated as 0.5 for ordering only, never reported as measured)
    flat.sort(
        key=lambda item: float(item[1].raw_score if item[1].raw_score is not None else item[1].confidence),
        reverse=True,
    )

    clusters: list[list[tuple[str, DetectionBox]]] = []
    for eid, det in flat:
        placed = False
        for cl in clusters:
            anchor_eid, anchor = cl[0]
            if compute_box_iou(det.box_2d, anchor.box_2d) >= iou_threshold:
                cl.append((eid, det))
                placed = True
                break
        if not placed:
            clusters.append([(eid, det)])

    fused: list[FusedDetectionHypothesis] = []
    for cidx, cl in enumerate(clusters):
        # Compute geometry weighted by per-family best score (prevents correlated same-family models
        # from dominating box coordinates)
        by_family: dict[str, list[tuple[str, DetectionBox]]] = {}
        for eid, det in cl:
            fam = engine_family(eid)
            by_family.setdefault(fam, []).append((eid, det))

        fam_weights: list[tuple[float, DetectionBox]] = []
        raw_scores: dict[str, float | None] = {}
        supporting_engines: list[str] = []
        for fam, members in by_family.items():
            best_eid, best_det = max(
                members,
                key=lambda m: float(m[1].raw_score if m[1].raw_score is not None else m[1].confidence),
            )
            emitted = best_det.attributes.get("confidence_emitted", True)
            w = float(best_det.raw_score if best_det.raw_score is not None else best_det.confidence) if emitted else 0.5
            fam_weights.append((max(0.05, w), best_det))
            for eid, det in members:
                em = det.attributes.get("confidence_emitted", True)
                sc = float(det.raw_score if det.raw_score is not None else det.confidence) if em else None
                raw_scores[eid] = round(sc, 4) if sc is not None else None
                if eid not in supporting_engines:
                    supporting_engines.append(eid)

        w_sum = sum(w for w, _ in fam_weights) or 1.0
        bx1 = sum(w * d.x1 for w, d in fam_weights) / w_sum
        by1 = sum(w * d.y1 for w, d in fam_weights) / w_sum
        bx2 = sum(w * d.x2 for w, d in fam_weights) / w_sum
        by2 = sum(w * d.y2 for w, d in fam_weights) / w_sum

        # Primary label: from highest-scoring family representative
        _, primary_det = max(fam_weights, key=lambda x: x[0])
        primary_label = primary_det.class_name
        canon_label = resolve_taxonomy_class(primary_label).name

        # Preserve competing hypotheses when engines disagree on canonical class
        primary_tax_id = resolve_taxonomy_class(primary_label).taxonomy_id
        competing: list[dict[str, Any]] = []
        seen_comp: set[tuple[str, str]] = set()
        for eid, det in cl:
            det_tax_id = resolve_taxonomy_class(det.class_name).taxonomy_id
            if det_tax_id != primary_tax_id or not are_semantically_compatible(primary_label, det.class_name, allow_parent_part=True):
                key = (eid, det.class_name)
                if key not in seen_comp:
                    seen_comp.add(key)
                    em = det.attributes.get("confidence_emitted", True)
                    competing.append({
                        "engine_id": eid,
                        "family": engine_family(eid),
                        "label": det.class_name,
                        "raw_score": round(float(det.confidence), 4) if em else None,
                    })

        # Uncertainty & calibration integrity (A5)
        measured_scores = [v for v in raw_scores.values() if v is not None]
        rep_score = max(measured_scores) if measured_scores else None
        cal_prob: float | None = None
        cal_status = "uncalibrated"
        if calibrator is not None and calibrator.is_fitted and rep_score is not None:
            cres = calibrator.calibrate(rep_score, class_name=canon_label)
            cal_prob = round(cres.calibrated_probability, 4)
            cal_status = cres.method

        base_unc = (1.0 - (cal_prob if cal_prob is not None else rep_score)) if rep_score is not None else 0.5
        if competing:
            base_unc = min(1.0, base_unc + 0.15)
        if is_low_contrast:
            base_unc = min(1.0, base_unc + 0.10)
        if len(by_family) >= 2 and not competing:
            base_unc = max(0.05, base_unc * 0.85)

        fused.append(FusedDetectionHypothesis(
            box_id=f"fused_{cidx}_{primary_det.box_id}",
            primary_label=primary_label,
            canonical_label=canon_label,
            box_2d=(bx1, by1, bx2, by2),
            raw_scores=raw_scores,
            supporting_engines=tuple(supporting_engines),
            supporting_families=tuple(sorted(by_family.keys())),
            independent_family_count=len(by_family),
            competing_hypotheses=tuple(competing),
            calibrated_confidence=cal_prob,
            calibration_status=cal_status,
            uncertainty=round(float(base_unc), 4),
            source_image_id=primary_det.source_image_id,
            provenance={
                "fusion_method": "family_grouped_wbf",
                "is_low_contrast": is_low_contrast,
                "has_competing_labels": bool(competing),
            },
        ))
    return fused


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(max(-1.0, min(1.0, dot / (na * nb))))


@dataclass(frozen=True)
class CrossViewMatchDecision:
    """Ambiguity-preserving cross-view identity decision (A7)."""

    source_box_id: str
    target_box_id: str | None
    status: str  # "matched" | "ambiguous" | "unmatched"
    similarity: float
    candidate_target_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_box_id": self.source_box_id,
            "target_box_id": self.target_box_id,
            "status": self.status,
            "similarity": round(self.similarity, 4),
            "candidate_target_ids": list(self.candidate_target_ids),
            "reasons": list(self.reasons),
        }


def associate_cross_view_detections(
    source_boxes: Sequence[DetectionBox],
    target_boxes: Sequence[DetectionBox],
    *,
    geometric_score_fn: Callable[[DetectionBox, DetectionBox], float | None] | None = None,
    min_similarity: float = 0.55,
    ambiguity_margin: float = 0.08,
) -> list[CrossViewMatchDecision]:
    """Associate detections across two views while preserving ambiguity.

    - Two visually similar objects (e.g. two identical chairs) within ``ambiguity_margin``
      stay ``ambiguous`` instead of forcing a false merge.
    - Partially occluded or truncated views of the same object are still eligible to match
      when semantic + appearance/embedding + geometric cues align.
    """
    decisions: list[CrossViewMatchDecision] = []
    for src in source_boxes:
        scored: list[tuple[float, DetectionBox, tuple[str, ...]]] = []
        for tgt in target_boxes:
            if src.source_image_id and tgt.source_image_id and src.source_image_id == tgt.source_image_id:
                continue
            if not are_semantically_compatible(src.class_name, tgt.class_name, allow_parent_part=False):
                continue

            reasons: list[str] = ["semantic_compatible"]
            cues: list[tuple[float, float]] = []  # (weight, score)

            # Visual embedding (DINOv2) if present in attributes, else color histogram
            emb_a = src.attributes.get("dinov2_embedding")
            emb_b = tgt.attributes.get("dinov2_embedding")
            if emb_a and emb_b:
                sim_emb = max(0.0, _cosine_sim(emb_a, emb_b))
                cues.append((0.50, sim_emb))
                reasons.append(f"embedding={sim_emb:.3f}")
            elif src.appearance_features and tgt.appearance_features:
                sim_app = max(0.0, _cosine_sim(src.appearance_features, tgt.appearance_features))
                cues.append((0.40, sim_app))
                reasons.append(f"appearance={sim_app:.3f}")

            # Geometric compatibility (epipolar / 3D distance) when camera geometry is supplied
            if geometric_score_fn is not None:
                g_score = geometric_score_fn(src, tgt)
                if g_score is not None:
                    cues.append((0.50, float(max(0.0, min(1.0, g_score)))))
                    reasons.append(f"geometry={g_score:.3f}")

            # Relative box size compatibility (relaxed when either view is truncated/occluded)
            partial = src.is_truncated or tgt.is_truncated or src.occlusion_level != "none" or tgt.occlusion_level != "none"
            if src.area > 0 and tgt.area > 0 and not partial:
                ratio = min(src.area, tgt.area) / max(src.area, tgt.area)
                cues.append((0.15, ratio))

            if not cues:
                continue
            total_w = sum(w for w, _ in cues)
            sim = sum(w * s for w, s in cues) / total_w
            if sim >= min_similarity:
                scored.append((sim, tgt, tuple(reasons)))

        if not scored:
            decisions.append(CrossViewMatchDecision(
                source_box_id=src.box_id,
                target_box_id=None,
                status="unmatched",
                similarity=0.0,
                reasons=("no_compatible_candidate",),
            ))
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        best_sim, best_tgt, best_reasons = scored[0]
        if len(scored) > 1 and (best_sim - scored[1][0]) < ambiguity_margin:
            ambig_ids = tuple(item[1].box_id for item in scored if (best_sim - item[0]) < ambiguity_margin)
            decisions.append(CrossViewMatchDecision(
                source_box_id=src.box_id,
                target_box_id=None,
                status="ambiguous",
                similarity=best_sim,
                candidate_target_ids=ambig_ids,
                reasons=best_reasons + ("multiple_similar_candidates_preserved",),
            ))
        else:
            decisions.append(CrossViewMatchDecision(
                source_box_id=src.box_id,
                target_box_id=best_tgt.box_id,
                status="matched",
                similarity=best_sim,
                candidate_target_ids=(best_tgt.box_id,),
                reasons=best_reasons,
            ))

    return decisions


__all__ = [
    "CrossViewMatchDecision",
    "ENGINE_FAMILIES",
    "FusedDetectionHypothesis",
    "associate_cross_view_detections",
    "engine_family",
    "fuse_multi_engine_detections",
    "select_detection_engines",
]
