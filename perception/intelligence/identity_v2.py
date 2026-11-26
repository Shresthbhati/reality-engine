"""A5.3 Cross-View Identity Reliability Subsystem.

Improves cross-view object identity resolution with:
  - Automatic crop appearance extraction (24-bin RGB histogram + optional DINOv2)
  - Camera-pose uncertainty-aware geometric compatibility (using ``CameraPoseConstraint.position_sigma_m``)
  - Occlusion-aware scale & viewpoint tolerance
  - Repeated-object disambiguation (4 identical chairs stay 4 identities)
  - Explicit 3-way match outcomes: ``confirmed_match``, ``rejected_match``, ``ambiguous_match``
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible
from perception.intelligence.data_model import (
    AppearanceDescriptor,
    CrossViewIdentityHypothesis,
    NormalizedPerceptionObservation,
)
from perception.intelligence.identity import (
    CameraPoseConstraint,
    IdentityMergeProposal,
    IdentityResolutionReport,
    IdentitySplitProposal,
)


@dataclass(frozen=True)
class PairwiseIdentityDecision:
    obs_a_id: str
    obs_b_id: str
    image_a_id: str
    image_b_id: str
    canonical_label: str
    similarity_score: float
    outcome: str  # "confirmed_match" | "ambiguous_match" | "rejected_match"
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class IdentityResolutionV2Report:
    base_report: IdentityResolutionReport
    pairwise_decisions: tuple[PairwiseIdentityDecision, ...]
    confirmed_count: int
    ambiguous_count: int
    rejected_count: int
    summary: dict[str, Any] = field(default_factory=dict)


def extract_crop_appearance_descriptor(
    image_rgb: np.ndarray,
    bbox_px: tuple[float, float, float, float],
) -> AppearanceDescriptor:
    """Extract a normalized 24-bin RGB histogram and dominant color from an observation crop."""
    h, w = image_rgb.shape[:2]
    x1 = max(0, min(w - 1, int(round(bbox_px[0]))))
    y1 = max(0, min(h - 1, int(round(bbox_px[1]))))
    x2 = max(x1 + 1, min(w, int(round(bbox_px[2]))))
    y2 = max(y1 + 1, min(h, int(round(bbox_px[3]))))
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return AppearanceDescriptor()

    hist_parts: list[float] = []
    for ch in range(3):
        hist, _ = np.histogram(crop[..., ch], bins=8, range=(0, 256))
        hist_parts.extend(float(v) for v in hist)
    norm = math.sqrt(sum(v * v for v in hist_parts))
    if norm > 1e-9:
        hist_parts = [round(v / norm, 5) for v in hist_parts]

    mean_rgb = tuple(round(float(crop[..., ch].mean()), 2) for ch in range(3))
    r, g, b = mean_rgb
    if max(r, g, b) < 55:
        dom = "dark"
    elif min(r, g, b) > 200:
        dom = "white"
    elif r > g + 25 and r > b + 25:
        dom = "red_warm"
    elif b > r + 20 and b > g + 15:
        dom = "blue_cool"
    elif g > r + 15 and g > b + 15:
        dom = "green"
    else:
        dom = "neutral"

    return AppearanceDescriptor(
        color_histogram_24=tuple(hist_parts),
        dominant_color_name=dom,
        mean_rgb=(mean_rgb[0], mean_rgb[1], mean_rgb[2]),
    )


def enrich_observations_with_appearance(
    observations: Sequence[NormalizedPerceptionObservation],
    images_rgb: Mapping[str, np.ndarray] | None,
) -> list[NormalizedPerceptionObservation]:
    """Populate missing ``AppearanceDescriptor`` fields from ``images_rgb`` crops."""
    if not images_rgb:
        return list(observations)
    out: list[NormalizedPerceptionObservation] = []
    for o in observations:
        if o.appearance.color_histogram_24 or o.appearance.embedding_vector:
            out.append(o)
            continue
        img = images_rgb.get(o.provenance.source_image_id)
        if img is None:
            out.append(o)
            continue
        desc = extract_crop_appearance_descriptor(img, o.bbox_px)
        out.append(replace(o, appearance=desc))
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(max(-1.0, min(1.0, dot / (na * nb))))


def _pose_aware_compat(
    cam_a: CameraPoseConstraint | None,
    cam_b: CameraPoseConstraint | None,
    obs_a: NormalizedPerceptionObservation,
    obs_b: NormalizedPerceptionObservation,
) -> tuple[float | None, str]:
    """Compute geometric compatibility accounting for camera baseline and pose uncertainty."""
    ca_x = 0.5 * (obs_a.bbox_norm[0] + obs_a.bbox_norm[2])
    ca_y = 0.5 * (obs_a.bbox_norm[1] + obs_a.bbox_norm[3])
    cb_x = 0.5 * (obs_b.bbox_norm[0] + obs_b.bbox_norm[2])
    cb_y = 0.5 * (obs_b.bbox_norm[1] + obs_b.bbox_norm[3])
    shift = math.hypot(ca_x - cb_x, ca_y - cb_y)

    if cam_a is None or cam_b is None:
        # Without camera poses, only strong normalized proximity gives partial geometric support
        return round(max(0.0, min(1.0, 1.0 - shift / 0.35)), 4), "no_camera_pose_2d_proximity"

    dx = cam_a.position_m[0] - cam_b.position_m[0]
    dy = cam_a.position_m[1] - cam_b.position_m[1]
    dz = cam_a.position_m[2] - cam_b.position_m[2]
    baseline = math.sqrt(dx * dx + dy * dy + dz * dz)
    sigma = float((cam_a.position_sigma_m or 0.0) + (cam_b.position_sigma_m or 0.0))
    tol = 0.25 + 0.15 * min(3.0, baseline) + min(0.25, 0.5 * sigma)
    score = max(0.0, min(1.0, 1.0 - shift / max(0.10, tol)))
    return round(score, 4), f"pose_baseline={baseline:.2f}m_sigma={sigma:.2f}m"


def _score_pair_v2(
    a: NormalizedPerceptionObservation,
    b: NormalizedPerceptionObservation,
    cameras: Mapping[str, CameraPoseConstraint],
) -> tuple[float, tuple[str, ...]]:
    if a.provenance.source_image_id == b.provenance.source_image_id:
        return 0.0, ("same_image_cannot_merge",)
    if not are_semantically_compatible(a.canonical_label, b.canonical_label, allow_parent_part=False):
        return 0.0, ("semantic_incompatible",)

    cues: list[tuple[float, float]] = []
    reasons: list[str] = ["semantic_compatible"]

    if a.appearance.embedding_vector and b.appearance.embedding_vector:
        es = max(0.0, _cosine(a.appearance.embedding_vector, b.appearance.embedding_vector))
        cues.append((0.45, es))
        reasons.append(f"embedding={es:.3f}")
    elif a.appearance.color_histogram_24 and b.appearance.color_histogram_24:
        hs = max(0.0, _cosine(a.appearance.color_histogram_24, b.appearance.color_histogram_24))
        cues.append((0.35, hs))
        reasons.append(f"histogram={hs:.3f}")

    geom, g_reason = _pose_aware_compat(
        cameras.get(a.provenance.source_image_id),
        cameras.get(b.provenance.source_image_id),
        a,
        b,
    )
    if geom is not None:
        w_g = 0.45 if cameras else 0.30
        cues.append((w_g, geom))
        reasons.append(f"{g_reason}:{geom:.3f}")

    # Scale consistency (relaxed when occluded or partial)
    aw = max(1e-6, a.bbox_norm[2] - a.bbox_norm[0])
    ah = max(1e-6, a.bbox_norm[3] - a.bbox_norm[1])
    bw = max(1e-6, b.bbox_norm[2] - b.bbox_norm[0])
    bh = max(1e-6, b.bbox_norm[3] - b.bbox_norm[1])
    area_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh)
    if a.is_partial or b.is_partial or a.occlusion_level != "none" or b.occlusion_level != "none":
        cues.append((0.08, max(0.5, area_ratio)))
        reasons.append("occlusion_relaxed_scale")
    else:
        cues.append((0.18, area_ratio))

    if not cues:
        return 0.0, ("insufficient_cues",)
    w_sum = sum(w for w, _ in cues)
    return round(sum(w * v for w, v in cues) / w_sum, 4), tuple(reasons)


def resolve_cross_view_identities_v2(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    images_rgb: Mapping[str, np.ndarray] | None = None,
    cameras: Mapping[str, CameraPoseConstraint] | None = None,
    confirm_threshold: float = 0.68,
    ambiguous_threshold: float = 0.50,
    ambiguity_margin: float = 0.08,
) -> IdentityResolutionV2Report:
    """Resolve cross-view identities with 3-way outcomes (confirmed, ambiguous, rejected)."""
    enriched = enrich_observations_with_appearance(observations, images_rgb)
    cam_map = cameras or {}

    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in enriched:
        by_img[o.provenance.source_image_id].append(o)

    images = sorted(by_img.keys())
    assigned: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    meta: dict[str, dict[str, Any]] = {}
    ambiguous_links: dict[str, set[str]] = defaultdict(set)
    pairwise_decisions: list[PairwiseIdentityDecision] = []
    split_proposals: list[IdentitySplitProposal] = []
    obs_by_id = {o.observation_id: o for o in enriched}

    confirmed_pairs = 0
    ambiguous_pairs = 0
    rejected_pairs = 0

    for idx, img_id in enumerate(images):
        cur_list = by_img[img_id]
        if idx == 0:
            for o in cur_list:
                iid = f"id_{o.observation_id}"
                assigned[o.observation_id] = iid
                members[iid] = [o.observation_id]
                meta[iid] = {
                    "label": o.canonical_label,
                    "score": 1.0,
                    "reasons": ("seed_view",),
                    "status": "singleton",
                }
            continue

        # Score every observation in current image against every existing cluster's latest rep
        used_clusters: set[str] = set()
        for o in cur_list:
            scored: list[tuple[str, float, tuple[str, ...]]] = []
            for iid, mem_ids in members.items():
                if iid in used_clusters:
                    continue
                rep = obs_by_id[mem_ids[-1]]
                if rep.provenance.source_image_id == img_id:
                    continue
                sc, rsn = _score_pair_v2(rep, o, cam_map)
                if sc >= ambiguous_threshold:
                    scored.append((iid, sc, rsn))
                elif are_semantically_compatible(rep.canonical_label, o.canonical_label, allow_parent_part=False):
                    rejected_pairs += 1
                    pairwise_decisions.append(PairwiseIdentityDecision(
                        obs_a_id=rep.observation_id,
                        obs_b_id=o.observation_id,
                        image_a_id=rep.provenance.source_image_id,
                        image_b_id=img_id,
                        canonical_label=o.canonical_label,
                        similarity_score=sc,
                        outcome="rejected_match",
                        reasons=rsn,
                    ))

            scored.sort(key=lambda x: x[1], reverse=True)
            if not scored:
                iid = f"id_{o.observation_id}"
                assigned[o.observation_id] = iid
                members[iid] = [o.observation_id]
                meta[iid] = {
                    "label": o.canonical_label,
                    "score": 1.0,
                    "reasons": ("new_singleton",),
                    "status": "singleton",
                }
                continue

            best_iid, best_sc, best_rsn = scored[0]
            runner_sc = scored[1][1] if len(scored) > 1 else 0.0
            rep_best = obs_by_id[members[best_iid][-1]]

            if best_sc >= confirm_threshold and (best_sc - runner_sc) >= ambiguity_margin:
                used_clusters.add(best_iid)
                assigned[o.observation_id] = best_iid
                members[best_iid].append(o.observation_id)
                meta[best_iid]["score"] = round(min(meta[best_iid]["score"], best_sc), 4)
                meta[best_iid]["reasons"] = best_rsn
                meta[best_iid]["status"] = "confirmed_match"
                confirmed_pairs += 1
                pairwise_decisions.append(PairwiseIdentityDecision(
                    obs_a_id=rep_best.observation_id,
                    obs_b_id=o.observation_id,
                    image_a_id=rep_best.provenance.source_image_id,
                    image_b_id=img_id,
                    canonical_label=o.canonical_label,
                    similarity_score=best_sc,
                    outcome="confirmed_match",
                    reasons=best_rsn,
                ))
            else:
                # Ambiguous match: keep as separate singleton AND record competing links
                iid = f"id_{o.observation_id}"
                assigned[o.observation_id] = iid
                members[iid] = [o.observation_id]
                meta[iid] = {
                    "label": o.canonical_label,
                    "score": best_sc,
                    "reasons": (*best_rsn, "preserved_ambiguity"),
                    "status": "ambiguous",
                }
                ambiguous_pairs += 1
                for cand_iid, _, _ in scored[:3]:
                    ambiguous_links[iid].add(cand_iid)
                    ambiguous_links[cand_iid].add(iid)
                pairwise_decisions.append(PairwiseIdentityDecision(
                    obs_a_id=rep_best.observation_id,
                    obs_b_id=o.observation_id,
                    image_a_id=rep_best.provenance.source_image_id,
                    image_b_id=img_id,
                    canonical_label=o.canonical_label,
                    similarity_score=best_sc,
                    outcome="ambiguous_match",
                    reasons=(*best_rsn, "below_confirm_or_margin"),
                ))

    identities: list[CrossViewIdentityHypothesis] = []
    for iid in sorted(members.keys()):
        mids = tuple(members[iid])
        m = meta[iid]
        lbls = {obs_by_id[mid].canonical_label for mid in mids}
        status = m["status"]
        if len(lbls) > 1:
            status = "split_proposed"
            split_proposals.append(IdentitySplitProposal(
                proposal_id=f"split_{iid}",
                identity_id=iid,
                conflicting_observation_ids=(mids[0], mids[-1]),
                reason=f"mixed_labels:{sorted(lbls)}",
            ))
        identities.append(CrossViewIdentityHypothesis(
            identity_id=iid,
            member_observation_ids=mids,
            canonical_label=m["label"],
            status=status,
            identity_score=m["score"],
            competing_identity_ids=tuple(sorted(ambiguous_links.get(iid, set()))),
            evidence_reasons=m["reasons"],
        ))

    base = IdentityResolutionReport(
        identities=tuple(identities),
        merge_proposals=(),
        split_proposals=tuple(split_proposals),
        summary={
            "total_identities": len(identities),
            "confirmed_matches": sum(1 for i in identities if i.status == "confirmed_match"),
            "ambiguous": sum(1 for i in identities if i.status == "ambiguous"),
            "singletons": sum(1 for i in identities if i.status == "singleton"),
            "split_proposals": len(split_proposals),
        },
    )
    return IdentityResolutionV2Report(
        base_report=base,
        pairwise_decisions=tuple(pairwise_decisions),
        confirmed_count=confirmed_pairs,
        ambiguous_count=ambiguous_pairs,
        rejected_count=rejected_pairs,
        summary={
            **base.summary,
            "pairwise_confirmed": confirmed_pairs,
            "pairwise_ambiguous": ambiguous_pairs,
            "pairwise_rejected": rejected_pairs,
        },
    )


__all__ = [
    "IdentityResolutionV2Report",
    "PairwiseIdentityDecision",
    "enrich_observations_with_appearance",
    "extract_crop_appearance_descriptor",
    "resolve_cross_view_identities_v2",
]
