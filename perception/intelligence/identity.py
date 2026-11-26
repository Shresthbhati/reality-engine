"""Cross-view Object Identity and Association Engine (A3.4).

Implements ambiguity-preserving multi-view identity resolution using:
  - Visual appearance embeddings (DINOv2 / color histograms)
  - Semantic class compatibility
  - Bounding-box & mask geometry
  - Camera-relative geometric constraints (via ``CameraPoseConstraint``)
  - Occlusion / partial-view tolerance
  - Identity revision, merge proposals, and split proposals

Critical invariant:
  - Four visually similar chairs in a room MUST retain four separate physical
    identities unless camera-relative geometry + appearance uniquely establish
    they are the same physical chair.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.spatial_intelligence.taxonomy import are_semantically_compatible
from perception.intelligence.data_model import (
    CrossViewIdentityHypothesis,
    NormalizedPerceptionObservation,
)


@dataclass(frozen=True)
class CameraPoseConstraint:
    """Stable interface for camera poses supplied by Claude's reconstruction pipeline."""

    image_id: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    position_sigma_m: float | None = None
    is_metric: bool = False


@dataclass(frozen=True)
class IdentityMergeProposal:
    proposal_id: str
    identity_ids: tuple[str, str]
    similarity_score: float
    reasons: tuple[str, ...]
    requires_human_or_3d_check: bool = True


@dataclass(frozen=True)
class IdentitySplitProposal:
    proposal_id: str
    identity_id: str
    conflicting_observation_ids: tuple[str, str]
    reason: str


@dataclass(frozen=True)
class IdentityResolutionReport:
    identities: tuple[CrossViewIdentityHypothesis, ...]
    merge_proposals: tuple[IdentityMergeProposal, ...]
    split_proposals: tuple[IdentitySplitProposal, ...]
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


def _pairwise_similarity(
    a: NormalizedPerceptionObservation,
    b: NormalizedPerceptionObservation,
    *,
    epipolar_fn: Callable[[NormalizedPerceptionObservation, NormalizedPerceptionObservation], float | None] | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Compute multi-cue similarity [0, 1] between two observations from different images."""
    if a.provenance.source_image_id == b.provenance.source_image_id:
        return 0.0, ("same_image_cannot_merge",)
    if not are_semantically_compatible(a.canonical_label, b.canonical_label, allow_parent_part=False):
        return 0.0, ("semantic_incompatible",)

    cues: list[tuple[float, float]] = []
    reasons: list[str] = ["semantic_compatible"]

    if a.appearance.embedding_vector and b.appearance.embedding_vector:
        emb_sim = max(0.0, _cosine(a.appearance.embedding_vector, b.appearance.embedding_vector))
        cues.append((0.45, emb_sim))
        reasons.append(f"embedding={emb_sim:.3f}")
    elif a.appearance.color_histogram_24 and b.appearance.color_histogram_24:
        hist_sim = max(0.0, _cosine(a.appearance.color_histogram_24, b.appearance.color_histogram_24))
        cues.append((0.35, hist_sim))
        reasons.append(f"histogram={hist_sim:.3f}")

    if epipolar_fn is not None:
        g = epipolar_fn(a, b)
        if g is not None:
            g_clamped = float(max(0.0, min(1.0, g)))
            cues.append((0.45, g_clamped))
            reasons.append(f"geometry={g_clamped:.3f}")

    # Normalized box scale consistency (relaxed if either view is partial/occluded)
    if not (a.is_partial or b.is_partial or a.occlusion_level != "none" or b.occlusion_level != "none"):
        aw = max(1e-6, a.bbox_norm[2] - a.bbox_norm[0])
        ah = max(1e-6, a.bbox_norm[3] - a.bbox_norm[1])
        bw = max(1e-6, b.bbox_norm[2] - b.bbox_norm[0])
        bh = max(1e-6, b.bbox_norm[3] - b.bbox_norm[1])
        area_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh)
        cues.append((0.15, area_ratio))

    if not cues:
        return 0.0, ("insufficient_cues",)
    w_sum = sum(w for w, _ in cues)
    score = sum(w * v for w, v in cues) / w_sum
    return round(float(score), 4), tuple(reasons)


def resolve_cross_view_identities(
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    epipolar_fn: Callable[[NormalizedPerceptionObservation, NormalizedPerceptionObservation], float | None] | None = None,
    match_threshold: float = 0.62,
    ambiguity_margin: float = 0.08,
) -> IdentityResolutionReport:
    """Resolve cross-view object identities while preserving ambiguity and preventing false merges."""
    by_image: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in observations:
        by_image[o.provenance.source_image_id].append(o)

    images = sorted(by_image.keys())
    assigned_cluster: dict[str, str] = {}
    cluster_members: dict[str, list[str]] = {}
    cluster_meta: dict[str, dict[str, Any]] = {}
    ambiguous_links: dict[str, set[str]] = defaultdict(set)
    merge_proposals: list[IdentityMergeProposal] = []
    split_proposals: list[IdentitySplitProposal] = []

    obs_by_id = {o.observation_id: o for o in observations}

    # Initialize first image observations as seed identities
    for idx, img_id in enumerate(images):
        cur_list = by_image[img_id]
        if idx == 0:
            for o in cur_list:
                iid = f"id_{o.observation_id}"
                assigned_cluster[o.observation_id] = iid
                cluster_members[iid] = [o.observation_id]
                cluster_meta[iid] = {
                    "label": o.canonical_label,
                    "score": 1.0,
                    "reasons": ("seed_view",),
                    "status": "singleton",
                }
            continue

        # Score each observation in cur_list against existing identities
        used_identities: set[str] = set()
        for o in cur_list:
            candidates: list[tuple[float, str, tuple[str, ...]]] = []
            for iid, members in cluster_members.items():
                # Never merge two observations from the SAME image into one identity
                if any(obs_by_id[m].provenance.source_image_id == img_id for m in members):
                    continue
                # Score against most recent representative of the identity
                rep = obs_by_id[members[-1]]
                sim, reasons = _pairwise_similarity(rep, o, epipolar_fn=epipolar_fn)
                if sim >= match_threshold:
                    candidates.append((sim, iid, reasons))

            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                iid = f"id_{o.observation_id}"
                assigned_cluster[o.observation_id] = iid
                cluster_members[iid] = [o.observation_id]
                cluster_meta[iid] = {
                    "label": o.canonical_label,
                    "score": 1.0,
                    "reasons": ("unmatched_new_identity",),
                    "status": "singleton",
                }
                continue

            best_sim, best_iid, best_reasons = candidates[0]
            # Ambiguity check: if >=2 identities are close in score, OR best_iid was already claimed
            # in this image (e.g. 2 identical chairs in view B matching 1 chair in view A), DO NOT MERGE
            is_ambig = (
                (len(candidates) > 1 and (best_sim - candidates[1][0]) < ambiguity_margin)
                or (best_iid in used_identities)
            )
            if is_ambig:
                new_iid = f"id_{o.observation_id}"
                assigned_cluster[o.observation_id] = new_iid
                cluster_members[new_iid] = [o.observation_id]
                comp_ids = tuple(c[1] for c in candidates if (best_sim - c[0]) < ambiguity_margin or c[1] == best_iid)
                for cid in comp_ids:
                    ambiguous_links[new_iid].add(cid)
                    ambiguous_links[cid].add(new_iid)
                cluster_meta[new_iid] = {
                    "label": o.canonical_label,
                    "score": best_sim,
                    "reasons": best_reasons + ("ambiguity_preserved_no_merge",),
                    "status": "ambiguous",
                }
                merge_proposals.append(IdentityMergeProposal(
                    proposal_id=f"merge_{new_iid}_{best_iid}",
                    identity_ids=(new_iid, best_iid),
                    similarity_score=best_sim,
                    reasons=best_reasons,
                    requires_human_or_3d_check=True,
                ))
            else:
                used_identities.add(best_iid)
                assigned_cluster[o.observation_id] = best_iid
                cluster_members[best_iid].append(o.observation_id)
                cluster_meta[best_iid]["score"] = best_sim
                cluster_meta[best_iid]["reasons"] = best_reasons
                cluster_meta[best_iid]["status"] = "confirmed_match"

    # Check for semantic contradictions inside any cluster -> propose split
    identities: list[CrossViewIdentityHypothesis] = []
    for iid, members in cluster_members.items():
        labels = {obs_by_id[m].canonical_label for m in members}
        status = cluster_meta[iid]["status"]
        if len(labels) > 1 and len(members) >= 2:
            status = "split_proposed"
            split_proposals.append(IdentitySplitProposal(
                proposal_id=f"split_{iid}",
                identity_id=iid,
                conflicting_observation_ids=(members[0], members[-1]),
                reason=f"cluster_contains_mixed_labels:{sorted(labels)}",
            ))
        identities.append(CrossViewIdentityHypothesis(
            identity_id=iid,
            member_observation_ids=tuple(members),
            canonical_label=cluster_meta[iid]["label"],
            status=status,
            identity_score=float(cluster_meta[iid]["score"]),
            competing_identity_ids=tuple(sorted(ambiguous_links.get(iid, set()))),
            evidence_reasons=tuple(cluster_meta[iid]["reasons"]),
        ))

    summary = {
        "total_identities": len(identities),
        "confirmed_matches": sum(1 for i in identities if i.status == "confirmed_match"),
        "ambiguous": sum(1 for i in identities if i.status == "ambiguous"),
        "singletons": sum(1 for i in identities if i.status == "singleton"),
        "merge_proposals": len(merge_proposals),
        "split_proposals": len(split_proposals),
    }
    return IdentityResolutionReport(
        identities=tuple(identities),
        merge_proposals=tuple(merge_proposals),
        split_proposals=tuple(split_proposals),
        summary=summary,
    )


__all__ = [
    "CameraPoseConstraint",
    "IdentityMergeProposal",
    "IdentityResolutionReport",
    "IdentitySplitProposal",
    "resolve_cross_view_identities",
]
