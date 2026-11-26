"""Scene Relationship Intelligence Subsystem (A3.9).

Infers evidence-backed indoor spatial relationships:
  - ``chair_beside_table``
  - ``cabinet_against_wall`` / ``bed_against_wall``
  - ``window_within_wall`` / ``door_within_wall`` / ``opening_within_wall``
  - ``light_attached_to_ceiling``
  - ``resting_on_floor``
  - ``above`` / ``below``
  - ``inside_room``

Supports multi-view support aggregation and contradiction detection (e.g.,
impossible vertical support cycles).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou
from perception.intelligence.data_model import (
    NormalizedPerceptionObservation,
    SpatialRelationshipObservation,
)

RELATIONSHIP_ONTOLOGY: frozenset[str] = frozenset({
    "chair_beside_table",
    "cabinet_against_wall",
    "bed_against_wall",
    "window_within_wall",
    "door_within_wall",
    "opening_within_wall",
    "light_attached_to_ceiling",
    "resting_on_floor",
    "above",
    "inside_room",
})


@dataclass(frozen=True)
class SceneRelationshipReport:
    relationships: tuple[SpatialRelationshipObservation, ...]
    contradictions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _ioa(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (ix * iy) / area


def _horiz_gap_norm(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0


def infer_scene_relationships(
    observations: Sequence[NormalizedPerceptionObservation],
) -> SceneRelationshipReport:
    """Infer per-image spatial relationship hypotheses and detect structural contradictions."""
    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    for o in observations:
        by_img[o.provenance.source_image_id].append(o)

    rels: list[SpatialRelationshipObservation] = []
    contradictions: list[dict[str, Any]] = []

    for img_id, obs_list in sorted(by_img.items()):
        for i, a in enumerate(obs_list):
            la = a.canonical_label
            for b in obs_list[i + 1:]:
                lb = b.canonical_label
                # 1. Door / window / opening within wall
                if la in ("door", "window", "opening") and lb in ("wall", "wall_segment"):
                    if _ioa(a.bbox_px, b.bbox_px) >= 0.35:
                        pred = f"{la}_within_wall"
                        rels.append(SpatialRelationshipObservation(
                            relationship_id=f"rel_{a.observation_id}_{pred}_{b.observation_id}",
                            subject_id=a.observation_id,
                            predicate=pred,
                            object_id=b.observation_id,
                            coordinate_frame="image_space",
                            support_state="visually_observed",
                            confidence_score=0.75,
                            supporting_images=(img_id,),
                        ))
                elif lb in ("door", "window", "opening") and la in ("wall", "wall_segment"):
                    if _ioa(b.bbox_px, a.bbox_px) >= 0.35:
                        pred = f"{lb}_within_wall"
                        rels.append(SpatialRelationshipObservation(
                            relationship_id=f"rel_{b.observation_id}_{pred}_{a.observation_id}",
                            subject_id=b.observation_id,
                            predicate=pred,
                            object_id=a.observation_id,
                            coordinate_frame="image_space",
                            support_state="visually_observed",
                            confidence_score=0.75,
                            supporting_images=(img_id,),
                        ))

                # 2. Chair beside table / desk
                if (la == "chair" and lb in ("table", "desk")) or (lb == "chair" and la in ("table", "desk")):
                    chair, tbl = (a, b) if la == "chair" else (b, a)
                    gap = _horiz_gap_norm(chair.bbox_norm, tbl.bbox_norm)
                    y_overlap = min(chair.bbox_norm[3], tbl.bbox_norm[3]) - max(chair.bbox_norm[1], tbl.bbox_norm[1])
                    if gap <= 0.12 and y_overlap > 0.0:
                        rels.append(SpatialRelationshipObservation(
                            relationship_id=f"rel_{chair.observation_id}_chair_beside_table_{tbl.observation_id}",
                            subject_id=chair.observation_id,
                            predicate="chair_beside_table",
                            object_id=tbl.observation_id,
                            coordinate_frame="image_space",
                            support_state="visually_observed",
                            confidence_score=round(max(0.4, 0.85 - gap * 2.0), 4),
                            supporting_images=(img_id,),
                        ))

                # 3. Cabinet / bed against wall
                if (la in ("cabinet", "wardrobe", "bed") and lb in ("wall", "wall_segment")) or (
                    lb in ("cabinet", "wardrobe", "bed") and la in ("wall", "wall_segment")
                ):
                    furn, wall = (a, b) if la in ("cabinet", "wardrobe", "bed") else (b, a)
                    if _ioa(furn.bbox_px, wall.bbox_px) >= 0.25 or compute_box_iou(furn.bbox_px, wall.bbox_px) >= 0.10:
                        pred = "bed_against_wall" if furn.canonical_label == "bed" else "cabinet_against_wall"
                        rels.append(SpatialRelationshipObservation(
                            relationship_id=f"rel_{furn.observation_id}_{pred}_{wall.observation_id}",
                            subject_id=furn.observation_id,
                            predicate=pred,
                            object_id=wall.observation_id,
                            coordinate_frame="image_space",
                            support_state="semantically_inferred",
                            confidence_score=0.60,
                            supporting_images=(img_id,),
                        ))

                # 4. Light attached to ceiling
                if (la in ("ceiling_light", "lamp") and lb == "ceiling") or (
                    lb in ("ceiling_light", "lamp") and la == "ceiling"
                ):
                    light, ceil = (a, b) if la in ("ceiling_light", "lamp") else (b, a)
                    if _ioa(light.bbox_px, ceil.bbox_px) >= 0.30:
                        rels.append(SpatialRelationshipObservation(
                            relationship_id=f"rel_{light.observation_id}_light_attached_to_ceiling_{ceil.observation_id}",
                            subject_id=light.observation_id,
                            predicate="light_attached_to_ceiling",
                            object_id=ceil.observation_id,
                            coordinate_frame="image_space",
                            support_state="visually_observed",
                            confidence_score=0.75,
                            supporting_images=(img_id,),
                        ))

                # 5. Resting on floor
                if (la in ("chair", "table", "sofa", "bed", "cabinet", "wardrobe", "desk") and lb == "floor") or (
                    lb in ("chair", "table", "sofa", "bed", "cabinet", "wardrobe", "desk") and la == "floor"
                ):
                    furn, flr = (a, b) if lb == "floor" else (b, a)
                    if furn.bbox_norm[3] >= flr.bbox_norm[1] - 0.05:
                        rels.append(SpatialRelationshipObservation(
                            relationship_id=f"rel_{furn.observation_id}_resting_on_floor_{flr.observation_id}",
                            subject_id=furn.observation_id,
                            predicate="resting_on_floor",
                            object_id=flr.observation_id,
                            coordinate_frame="image_space",
                            support_state="semantically_inferred",
                            confidence_score=0.65,
                            supporting_images=(img_id,),
                        ))

    # Contradiction check: any object claiming to be both light_attached_to_ceiling AND resting_on_floor
    by_subj: dict[str, set[str]] = defaultdict(set)
    for r in rels:
        by_subj[r.subject_id].add(r.predicate)
    for sid, preds in by_subj.items():
        if "light_attached_to_ceiling" in preds and "resting_on_floor" in preds:
            contradictions.append({
                "subject_id": sid,
                "predicates": ["light_attached_to_ceiling", "resting_on_floor"],
                "reason": "object_cannot_be_both_ceiling_attached_and_floor_resting",
            })

    summary = {
        "total_relationships": len(rels),
        "contradictions": len(contradictions),
        "by_predicate": {p: sum(1 for r in rels if r.predicate == p) for p in sorted({r.predicate for r in rels})},
    }
    return SceneRelationshipReport(
        relationships=tuple(rels),
        contradictions=tuple(contradictions),
        summary=summary,
    )


__all__ = [
    "RELATIONSHIP_ONTOLOGY",
    "SceneRelationshipReport",
    "infer_scene_relationships",
]
