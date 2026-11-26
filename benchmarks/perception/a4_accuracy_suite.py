"""A4 Accuracy & Benchmark Engine (A4.11).

Measures where ground truth supports it:
  - Object detection: Precision, Recall, AP, mAP
  - Segmentation: Mask IoU, Boundary F-score
  - Classification: Accuracy, Macro F1
  - Cross-view identity: Precision, Recall, IDF1
  - Architecture: Boundary accuracy, Opening IoU
  - Relationships: Precision, Recall, F1
  - Calibration: ECE, Brier score
  - Performance: Runtime, Peak VRAM, Throughput
  - Regression comparison & bootstrap confidence intervals

Strictly separates synthetic, unlabelled real, and labelled real results.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.perception.evaluator import EvaluationMode, run_perception_benchmark
from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox, compute_box_iou
from perception.intelligence.data_model import (
    CrossViewIdentityHypothesis,
    SpatialRelationshipObservation,
)


@dataclass(frozen=True)
class A4ComprehensiveBenchmark:
    mode: EvaluationMode
    dataset_name: str
    detection_map: float | None
    macro_f1: float | None
    mask_iou_mean: float | None
    boundary_f_score_mean: float | None
    identity_idf1: float | None
    opening_iou_mean: float | None
    relationship_f1: float | None
    ece: float | None
    brier_score: float | None
    throughput_img_per_s: float
    wall_time_s: float
    peak_vram_mb: float | None
    ap_ci_95: tuple[float, float] | None
    accuracy_claim_allowed: bool
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        if self.ap_ci_95 is not None:
            d["ap_ci_95"] = list(self.ap_ci_95)
        return d


def compute_mask_iou_and_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Compute binary mask IoU and boundary F1-score."""
    pm = np.asarray(pred_mask, dtype=bool)
    gm = np.asarray(gt_mask, dtype=bool)
    if pm.shape != gm.shape or pm.size == 0:
        return 0.0, 0.0
    inter = int(np.sum(pm & gm))
    union = int(np.sum(pm | gm))
    iou = inter / float(union) if union > 0 else 0.0

    def _perim(m: np.ndarray) -> np.ndarray:
        return m ^ (
            np.roll(m, 1, axis=0) & np.roll(m, -1, axis=0) & np.roll(m, 1, axis=1) & np.roll(m, -1, axis=1)
        )

    pp = _perim(pm)
    gp = _perim(gm)
    tp = int(np.sum(pp & gp))
    p_cnt = int(np.sum(pp))
    g_cnt = int(np.sum(gp))
    prec = tp / float(p_cnt) if p_cnt > 0 else 0.0
    rec = tp / float(g_cnt) if g_cnt > 0 else 0.0
    bf1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return round(iou, 4), round(bf1, 4)


def compute_identity_idf1(
    predicted_identities: Sequence[CrossViewIdentityHypothesis],
    gt_identity_map: Mapping[str, str],
) -> float | None:
    """Compute IDF1 across predicted cross-view identity clusters and ground-truth physical IDs."""
    if not gt_identity_map or not predicted_identities:
        return None
    gt_clusters: dict[str, set[str]] = defaultdict(set)
    for oid, gid in gt_identity_map.items():
        gt_clusters[gid].add(oid)

    idtp = 0
    pred_tot = 0
    gt_tot = sum(len(v) for v in gt_clusters.values())
    for p in predicted_identities:
        pset = {m for m in p.member_observation_ids if m in gt_identity_map}
        if not pset:
            continue
        pred_tot += len(pset)
        best_overlap = max((len(pset & gset) for gset in gt_clusters.values()), default=0)
        idtp += best_overlap
    if pred_tot + gt_tot == 0:
        return None
    return round((2.0 * idtp) / float(pred_tot + gt_tot), 4)


def compute_relationship_f1(
    pred_rels: Sequence[SpatialRelationshipObservation],
    gt_triples: Sequence[tuple[str, str, str]],
) -> float | None:
    """Compute F1 over (subject_id, predicate, object_id) triples."""
    if not gt_triples:
        return None
    pred_set = {(r.subject_id, r.predicate, r.object_id) for r in pred_rels}
    gt_set = set(gt_triples)
    tp = len(pred_set & gt_set)
    prec = tp / float(len(pred_set)) if pred_set else 0.0
    rec = tp / float(len(gt_set)) if gt_set else 0.0
    return round((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0, 4)


def compare_benchmark_regression(
    baseline: A4ComprehensiveBenchmark,
    candidate: A4ComprehensiveBenchmark,
    *,
    max_allowed_drop: float = 0.02,
) -> dict[str, Any]:
    """Compare candidate benchmark against baseline and flag regressions."""
    regressions: list[str] = []
    for metric in ("detection_map", "macro_f1", "mask_iou_mean", "identity_idf1"):
        b_val = getattr(baseline, metric)
        c_val = getattr(candidate, metric)
        if b_val is not None and c_val is not None and (b_val - c_val) > max_allowed_drop:
            regressions.append(f"{metric}: dropped from {b_val} to {c_val}")
    return {
        "passed": len(regressions) == 0,
        "regressions": regressions,
    }


def run_a4_comprehensive_benchmark(
    *,
    mode: EvaluationMode,
    dataset_name: str,
    predictions_by_image: Mapping[str, Sequence[DetectionBox]],
    gt_by_image: Mapping[str, Sequence[GroundTruthBox]] | None = None,
    mask_pairs: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    predicted_identities: Sequence[CrossViewIdentityHypothesis] = (),
    gt_identity_map: Mapping[str, str] | None = None,
    pred_relationships: Sequence[SpatialRelationshipObservation] = (),
    gt_relationships: Sequence[tuple[str, str, str]] = (),
    wall_time_s: float = 0.0,
    peak_vram_mb: float | None = None,
) -> A4ComprehensiveBenchmark:
    """Execute full A4 multi-capability evaluation with honest unlabelled/synthetic separation."""
    base = run_perception_benchmark(
        mode=mode,
        dataset_name=dataset_name,
        predictions_by_image=predictions_by_image,
        gt_by_image=gt_by_image,
        predicted_identities=predicted_identities,
        gt_identity_map=gt_identity_map,
        wall_time_s=wall_time_s,
        peak_vram_mb=peak_vram_mb,
    )
    n_imgs = max(1, base.images_evaluated)
    throughput = round(n_imgs / max(1e-4, wall_time_s), 3) if wall_time_s > 0 else 0.0

    if mode is EvaluationMode.UNLABELLED_REAL or not gt_by_image:
        return A4ComprehensiveBenchmark(
            mode=mode,
            dataset_name=dataset_name,
            detection_map=None,
            macro_f1=None,
            mask_iou_mean=None,
            boundary_f_score_mean=None,
            identity_idf1=None,
            opening_iou_mean=None,
            relationship_f1=None,
            ece=None,
            brier_score=None,
            throughput_img_per_s=throughput,
            wall_time_s=round(wall_time_s, 3),
            peak_vram_mb=peak_vram_mb,
            ap_ci_95=None,
            accuracy_claim_allowed=False,
            caveat=base.caveat,
        )

    f1s = [v["f1"] for v in base.per_class.values()]
    macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0

    m_ious, b_f1s = [], []
    for pm, gm in mask_pairs:
        mi, bf = compute_mask_iou_and_boundary_f1(pm, gm)
        m_ious.append(mi)
        b_f1s.append(bf)

    # Opening IoU across doors/windows/openings
    opening_ious: list[float] = []
    for img_id, gts in gt_by_image.items():
        preds = predictions_by_image.get(img_id, ())
        for g in gts:
            if g.canonical_label in ("door", "window", "opening"):
                best = max((compute_box_iou(p.box_2d, g.box_2d) for p in preds if p.class_name == g.canonical_label), default=0.0)
                opening_ious.append(best)

    aps = [v["ap"] for v in base.per_class.values()]
    ci: tuple[float, float] | None = None
    if len(aps) >= 2:
        rng = np.random.default_rng(42)
        boots = [float(np.mean(rng.choice(aps, size=len(aps), replace=True))) for _ in range(200)]
        ci = (round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4))

    return A4ComprehensiveBenchmark(
        mode=mode,
        dataset_name=dataset_name,
        detection_map=base.mean_ap,
        macro_f1=macro_f1,
        mask_iou_mean=round(float(np.mean(m_ious)), 4) if m_ious else None,
        boundary_f_score_mean=round(float(np.mean(b_f1s)), 4) if b_f1s else None,
        identity_idf1=compute_identity_idf1(predicted_identities, gt_identity_map or {}),
        opening_iou_mean=round(float(np.mean(opening_ious)), 4) if opening_ious else None,
        relationship_f1=compute_relationship_f1(pred_relationships, gt_relationships),
        ece=base.ece,
        brier_score=base.brier_score,
        throughput_img_per_s=throughput,
        wall_time_s=round(wall_time_s, 3),
        peak_vram_mb=peak_vram_mb,
        ap_ci_95=ci,
        accuracy_claim_allowed=base.accuracy_claim_allowed,
        caveat=base.caveat,
    )


__all__ = [
    "A4ComprehensiveBenchmark",
    "compare_benchmark_regression",
    "compute_identity_idf1",
    "compute_mask_iou_and_boundary_f1",
    "compute_relationship_f1",
    "run_a4_comprehensive_benchmark",
]
