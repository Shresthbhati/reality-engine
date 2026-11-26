"""A5.2 Duplicate Detection Elimination Subsystem.

Resolves duplicate detections across:
  - Multiple detection engines (Grounding DINO, OWLv2, YOLO-World, Florence-2)
  - Overlapping image crops / tiled inference
  - Multi-scale passes
  - Repeated architectural detections

Invariants:
  - Never merges two separate chairs merely because they look identical
    (requires spatial overlap above threshold).
  - Preserves competing class hypotheses when overlapping detections disagree
    without a clear semantic winner.
  - Preserves full provenance of all suppressed duplicates.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou
from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible
from perception.detection.multi_engine_fusion import engine_family
from perception.intelligence.data_model import NormalizedPerceptionObservation


@dataclass(frozen=True)
class CompetingHypothesisRecord:
    kept_observation_id: str
    competing_observation_id: str
    kept_label: str
    competing_label: str
    box_iou: float
    mask_iou: float | None
    reason: str


@dataclass(frozen=True)
class DuplicateClusterRecord:
    cluster_id: str
    kept_observation_id: str
    suppressed_observation_ids: tuple[str, ...]
    canonical_label: str
    source_engines: tuple[str, ...]
    max_box_iou: float
    max_mask_iou: float | None
    reason: str


@dataclass(frozen=True)
class DuplicateResolutionResult:
    canonical_observations: tuple[NormalizedPerceptionObservation, ...]
    suppressed_observations: tuple[NormalizedPerceptionObservation, ...]
    clusters: tuple[DuplicateClusterRecord, ...]
    competing_hypotheses: tuple[CompetingHypothesisRecord, ...]
    summary: dict[str, Any] = field(default_factory=dict)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(max(-1.0, min(1.0, dot / (na * nb))))


def _decode_rle(rle: dict[str, Any] | None) -> list[int] | None:
    if not rle or "counts" not in rle or "size" not in rle:
        return None
    h, w = int(rle["size"][0]), int(rle["size"][1])
    total = h * w
    if total <= 0:
        return None
    flat = [0] * total
    idx = 0
    val = 0
    for cnt in rle["counts"]:
        c = int(cnt)
        if c < 0:
            return None
        end = min(total, idx + c)
        if val == 1:
            for k in range(idx, end):
                flat[k] = 1
        idx = end
        val = 1 - val
    return flat


def compute_rle_mask_iou(
    rle_a: dict[str, Any] | None,
    rle_b: dict[str, Any] | None,
) -> float | None:
    """Compute exact mask IoU between two binary RLE masks of equal dimensions."""
    if not rle_a or not rle_b:
        return None
    if rle_a.get("size") != rle_b.get("size"):
        return None
    fa = _decode_rle(rle_a)
    fb = _decode_rle(rle_b)
    if fa is None or fb is None or len(fa) != len(fb):
        return None
    inter = sum(1 for a, b in zip(fa, fb) if a and b)
    union = sum(1 for a, b in zip(fa, fb) if a or b)
    if union == 0:
        return 0.0
    return round(inter / float(union), 4)


def _appearance_similarity(
    a: NormalizedPerceptionObservation,
    b: NormalizedPerceptionObservation,
) -> float | None:
    if a.appearance.embedding_vector and b.appearance.embedding_vector:
        return max(0.0, _cosine(a.appearance.embedding_vector, b.appearance.embedding_vector))
    if a.appearance.color_histogram_24 and b.appearance.color_histogram_24:
        return max(0.0, _cosine(a.appearance.color_histogram_24, b.appearance.color_histogram_24))
    return None


def _box_center_distance_norm(
    a: NormalizedPerceptionObservation,
    b: NormalizedPerceptionObservation,
) -> float:
    ax = 0.5 * (a.bbox_norm[0] + a.bbox_norm[2])
    ay = 0.5 * (a.bbox_norm[1] + a.bbox_norm[3])
    bx = 0.5 * (b.bbox_norm[0] + b.bbox_norm[2])
    by = 0.5 * (b.bbox_norm[1] + b.bbox_norm[3])
    return math.hypot(ax - bx, ay - by)


def _rank_score(obs: NormalizedPerceptionObservation) -> tuple[float, float, float]:
    has_mask = 1.0 if (obs.segmentation is not None and obs.segmentation.area_px > 0) else 0.0
    raw = float(obs.scores.raw_score if obs.scores.raw_score is not None else 0.5)
    area = max(0.0, (obs.bbox_norm[2] - obs.bbox_norm[0]) * (obs.bbox_norm[3] - obs.bbox_norm[1]))
    return (has_mask, raw, area)


def resolve_duplicate_detections_v2(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    same_class_iou_threshold: float = 0.55,
    cross_engine_iou_threshold: float = 0.62,
    mask_iou_threshold: float = 0.50,
    arch_iou_threshold: float = 0.60,
    center_dist_max_norm: float = 0.15,
) -> DuplicateResolutionResult:
    """Eliminate duplicate detections within each image while preserving separate objects and conflicts."""
    if not observations:
        return DuplicateResolutionResult(
            canonical_observations=(),
            suppressed_observations=(),
            clusters=(),
            competing_hypotheses=(),
            summary={
                "input_count": 0,
                "canonical_count": 0,
                "suppressed_count": 0,
                "duplicate_reduction_rate": 0.0,
                "competing_hypotheses_count": 0,
            },
        )

    by_image: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for obs in observations:
        by_image[obs.provenance.source_image_id].append(obs)

    kept_all: list[NormalizedPerceptionObservation] = []
    suppressed_all: list[NormalizedPerceptionObservation] = []
    clusters: list[DuplicateClusterRecord] = []
    competing: list[CompetingHypothesisRecord] = []

    for img_id in sorted(by_image.keys()):
        ordered = sorted(by_image[img_id], key=_rank_score, reverse=True)
        local_kept: list[NormalizedPerceptionObservation] = []
        cluster_members: dict[str, list[NormalizedPerceptionObservation]] = {}
        cluster_ious: dict[str, list[float]] = {}
        cluster_mask_ious: dict[str, list[float]] = {}
        cluster_reasons: dict[str, list[str]] = {}

        for cand in ordered:
            matched_kept: NormalizedPerceptionObservation | None = None
            match_reason = ""
            best_box_iou = 0.0
            best_mask_iou: float | None = None

            for rep in local_kept:
                biou = compute_box_iou(rep.bbox_px, cand.bbox_px)
                miou = (
                    compute_rle_mask_iou(rep.segmentation.rle_mask, cand.segmentation.rle_mask)
                    if (rep.segmentation is not None and cand.segmentation is not None)
                    else None
                )
                cdist = _box_center_distance_norm(rep, cand)
                compat = are_semantically_compatible(
                    rep.canonical_label,
                    cand.canonical_label,
                    allow_parent_part=False,
                )

                # Different classes overlapping strongly -> preserve competing hypothesis, do NOT merge blindly
                if not compat:
                    if biou >= 0.72 or (miou is not None and miou >= 0.70):
                        competing.append(CompetingHypothesisRecord(
                            kept_observation_id=rep.observation_id,
                            competing_observation_id=cand.observation_id,
                            kept_label=rep.canonical_label,
                            competing_label=cand.canonical_label,
                            box_iou=round(biou, 4),
                            mask_iou=miou,
                            reason="high_overlap_class_disagreement",
                        ))
                    continue

                # Same or compatible class: check spatial overlap so two separate chairs never merge
                same_eng = rep.provenance.execution.engine_id == cand.provenance.execution.engine_id
                same_fam = (
                    engine_family(rep.provenance.execution.engine_id)
                    == engine_family(cand.provenance.execution.engine_id)
                )
                is_crop_or_scale = (
                    rep.provenance.inference_mode != cand.provenance.inference_mode
                    or rep.provenance.tile_id != cand.provenance.tile_id
                )
                is_arch = rep.category == "architecture" or cand.category == "architecture"

                thresh = (
                    arch_iou_threshold
                    if is_arch
                    else (same_class_iou_threshold if (same_eng or same_fam or is_crop_or_scale) else cross_engine_iou_threshold)
                )

                mask_dup = miou is not None and miou >= mask_iou_threshold
                box_dup = biou >= thresh and cdist <= center_dist_max_norm

                if not (mask_dup or box_dup):
                    continue

                # If both have appearance embeddings and they strongly disagree, keep separate
                app_sim = _appearance_similarity(rep, cand)
                if app_sim is not None and app_sim < 0.40 and biou < 0.75:
                    continue

                if biou >= best_box_iou:
                    matched_kept = rep
                    best_box_iou = biou
                    best_mask_iou = miou
                    if mask_dup:
                        match_reason = "mask_iou_duplicate"
                    elif is_crop_or_scale:
                        match_reason = "crop_or_multiscale_duplicate"
                    elif is_arch:
                        match_reason = "repeated_architectural_duplicate"
                    elif same_eng:
                        match_reason = "same_engine_duplicate"
                    else:
                        match_reason = "cross_engine_duplicate"

            if matched_kept is None:
                local_kept.append(cand)
                cluster_members[cand.observation_id] = []
                cluster_ious[cand.observation_id] = []
                cluster_mask_ious[cand.observation_id] = []
                cluster_reasons[cand.observation_id] = []
            else:
                suppressed_all.append(cand)
                cluster_members[matched_kept.observation_id].append(cand)
                cluster_ious[matched_kept.observation_id].append(best_box_iou)
                if best_mask_iou is not None:
                    cluster_mask_ious[matched_kept.observation_id].append(best_mask_iou)
                cluster_reasons[matched_kept.observation_id].append(match_reason)

        kept_all.extend(local_kept)
        for rep in local_kept:
            sups = cluster_members[rep.observation_id]
            if not sups:
                continue
            engines = tuple(sorted({
                rep.provenance.execution.engine_id,
                *(s.provenance.execution.engine_id for s in sups),
            }))
            m_ious = cluster_mask_ious[rep.observation_id]
            clusters.append(DuplicateClusterRecord(
                cluster_id=f"dup_{img_id}_{rep.observation_id}",
                kept_observation_id=rep.observation_id,
                suppressed_observation_ids=tuple(s.observation_id for s in sups),
                canonical_label=rep.canonical_label,
                source_engines=engines,
                max_box_iou=round(max(cluster_ious[rep.observation_id]), 4),
                max_mask_iou=round(max(m_ious), 4) if m_ious else None,
                reason=cluster_reasons[rep.observation_id][0],
            ))

    n_in = len(observations)
    n_sup = len(suppressed_all)
    return DuplicateResolutionResult(
        canonical_observations=tuple(kept_all),
        suppressed_observations=tuple(suppressed_all),
        clusters=tuple(clusters),
        competing_hypotheses=tuple(competing),
        summary={
            "input_count": n_in,
            "canonical_count": len(kept_all),
            "suppressed_count": n_sup,
            "duplicate_reduction_rate": round(n_sup / float(max(1, n_in)), 4),
            "competing_hypotheses_count": len(competing),
        },
    )


__all__ = [
    "CompetingHypothesisRecord",
    "DuplicateClusterRecord",
    "DuplicateResolutionResult",
    "compute_rle_mask_iou",
    "resolve_duplicate_detections_v2",
]
