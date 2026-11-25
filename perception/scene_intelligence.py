"""Scene Intelligence Engine (perception/scene_intelligence.py).

Transforms isolated VisualObservations into a connected, evidence-backed
environmental understanding for Claude's spatial compiler:
  - Hierarchical open-vocabulary object classification
  - Image-space vs. world-space visual scene graph & candidate relationships
  - Multi-cue room & environment hypotheses (never from 1 isolated object)
  - Material & appearance hypotheses (distinct from physical material facts)
  - Perception-level observability & grounded active capture guidance
  - Video temporal tracking (appearance, disappearance, static vs. moving)
  - Sequential RTX 4050 GPU model scheduling with real VRAM/device telemetry
  - Non-destructive WorldIR adapter (world.metadata["scene_intelligence"])
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from perception.materials.material_hypotheses import hypothesize_materials
from perception.tracking.temporal import TimedObservation, link_temporally
from perception.world_bridge import (
    BridgeExecutionReport,
    VisualObservation,
    assess_image_quality,
    bridge_observations_to_world,
    build_visual_observations_from_detections,
)
from world_ir.observations import AssociationDecision, AssociationState, DepthSource
from world_ir.world_v1 import WorldIR


ARCHITECTURAL_QUERIES: Tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "window", "column", "beam",
    "stairs", "railing", "archway", "cabinet", "table", "chair",
    "sofa", "bed", "desk", "lamp", "sink", "toilet", "refrigerator",
)

# ---------------------------------------------------------------------------
# Phase 3: Hierarchical Open-Vocabulary Object Intelligence
# ---------------------------------------------------------------------------

OBJECT_HIERARCHY: Dict[str, Tuple[str, ...]] = {
    "chair": ("Furniture", "Seating", "Chair"),
    "dining chair": ("Furniture", "Seating", "Chair", "Dining chair"),
    "armchair": ("Furniture", "Seating", "Chair", "Armchair"),
    "couch": ("Furniture", "Seating", "Sofa"),
    "sofa": ("Furniture", "Seating", "Sofa"),
    "bed": ("Furniture", "Sleeping", "Bed"),
    "dining table": ("Furniture", "Table", "Dining table"),
    "desk": ("Furniture", "Table", "Desk"),
    "table": ("Furniture", "Table"),
    "cabinet": ("Furniture", "Storage", "Cabinet"),
    "bookshelf": ("Furniture", "Storage", "Bookshelf"),
    "tv": ("Electronics", "Display", "Television"),
    "monitor": ("Electronics", "Display", "Monitor"),
    "laptop": ("Electronics", "Computing", "Laptop"),
    "refrigerator": ("Appliances", "Kitchen", "Refrigerator"),
    "oven": ("Appliances", "Kitchen", "Oven"),
    "microwave": ("Appliances", "Kitchen", "Microwave"),
    "sink": ("Fixtures", "Plumbing", "Sink"),
    "toilet": ("Fixtures", "Plumbing", "Toilet"),
    "shower": ("Fixtures", "Plumbing", "Shower"),
    "lamp": ("Lighting", "Fixture", "Lamp"),
    "potted plant": ("Decor", "Plant", "Potted plant"),
    "wall": ("Architecture", "Surface", "Wall"),
    "floor": ("Architecture", "Surface", "Floor"),
    "ceiling": ("Architecture", "Surface", "Ceiling"),
    "door": ("Architecture", "Opening", "Door"),
    "window": ("Architecture", "Opening", "Window"),
    "stairs": ("Architecture", "Circulation", "Stairs"),
    "column": ("Architecture", "Structural", "Column"),
    "beam": ("Architecture", "Structural", "Beam"),
    "railing": ("Architecture", "Safety", "Railing"),
}


def resolve_hierarchical_label(
    raw_label: str,
    confidence: Optional[float],
    *,
    min_confidence: float = 0.25,
) -> Tuple[str, Tuple[str, ...]]:
    """Map raw label to hierarchical path; uncertain detections remain Unknown."""
    clean = (raw_label or "").strip().lower()
    if not clean or clean in ("unknown", "other", "object"):
        return "unknown", ("Unknown",)
    if confidence is not None and confidence < min_confidence:
        return "unknown", ("Unknown",)
    path = OBJECT_HIERARCHY.get(clean)
    if path is not None:
        return clean, path
    return clean, ("Object", clean.title())


# ---------------------------------------------------------------------------
# Phase 1 & 4: Visual Scene Graph & Spatial Relationship Evidence
# ---------------------------------------------------------------------------

class RelationshipType(str, Enum):
    INSIDE = "INSIDE"
    CONTAINS = "CONTAINS"
    ADJACENT_TO = "ADJACENT_TO"
    ATTACHED_TO = "ATTACHED_TO"
    SUPPORTED_BY = "SUPPORTED_BY"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    IN_FRONT_OF = "IN_FRONT_OF"
    BEHIND = "BEHIND"
    CONNECTED_TO = "CONNECTED_TO"
    FACING = "FACING"
    OVERLAPPING = "OVERLAPPING"
    PART_OF = "PART_OF"
    BLOCKING = "BLOCKING"
    VISIBLE_FROM = "VISIBLE_FROM"


class RelationshipSupport(str, Enum):
    VISUALLY_OBSERVED = "visually_observed"
    GEOMETRICALLY_SUPPORTED = "geometrically_supported"
    SEMANTICALLY_INFERRED = "semantically_inferred"
    WEAKLY_SUPPORTED = "weakly_supported"
    CONTRADICTORY = "contradictory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CandidateRelationship:
    source_observation_id: str
    target_observation_id: str
    rel_type: RelationshipType
    coordinate_frame: str  # "image_space" | "world_metric" | "world_relative"
    support_state: RelationshipSupport
    confidence: Optional[float]
    uncertainty_m: Optional[float]
    evidence_refs: Tuple[str, ...]
    basis: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rel_type"] = self.rel_type.value
        d["support_state"] = self.support_state.value
        d["evidence_refs"] = list(self.evidence_refs)
        return d


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def extract_candidate_relationships(
    observations: Sequence[VisualObservation],
    *,
    adjacency_radius_m: float = 1.5,
) -> List[CandidateRelationship]:
    """Extract candidate image-space and world-space relationships without conflating them."""
    rels: List[CandidateRelationship] = []
    for i, a in enumerate(observations):
        for b in observations[i + 1:]:
            ev_refs = tuple(sorted({a.source_evidence_id, b.source_evidence_id}))
            # 1. Image-space relationships (same image/frame only)
            if a.source_image_id == b.source_image_id and a.frame_index == b.frame_index:
                if a.bbox_px is not None and b.bbox_px is not None:
                    iou = _bbox_iou(a.bbox_px, b.bbox_px)
                    if iou > 0.15:
                        if a.model_id != b.model_id and a.semantic_label.lower() != b.semantic_label.lower() and iou >= 0.35:
                            rels.append(CandidateRelationship(
                                source_observation_id=a.observation_id,
                                target_observation_id=b.observation_id,
                                rel_type=RelationshipType.OVERLAPPING,
                                coordinate_frame="image_space",
                                support_state=RelationshipSupport.CONTRADICTORY,
                                confidence=round(min(1.0, iou), 3),
                                uncertainty_m=None,
                                evidence_refs=ev_refs,
                                basis=f"Cross-model disagreement ({a.model_id}:{a.semantic_label} vs {b.model_id}:{b.semantic_label}, IoU={iou:.2f})",
                            ))
                        else:
                            rels.append(CandidateRelationship(
                                source_observation_id=a.observation_id,
                                target_observation_id=b.observation_id,
                                rel_type=RelationshipType.OVERLAPPING,
                                coordinate_frame="image_space",
                                support_state=RelationshipSupport.VISUALLY_OBSERVED,
                                confidence=round(min(1.0, iou), 3),
                                uncertainty_m=None,
                                evidence_refs=ev_refs,
                                basis=f"2D bbox IoU={iou:.2f} in {a.source_image_id}",
                            ))
                        if a.detection_category == "opening" and b.semantic_label.lower() == "wall":
                            rels.append(CandidateRelationship(
                                source_observation_id=a.observation_id,
                                target_observation_id=b.observation_id,
                                rel_type=RelationshipType.PART_OF,
                                coordinate_frame="image_space",
                                support_state=RelationshipSupport.SEMANTICALLY_INFERRED,
                                confidence=round(min(0.8, iou + 0.2), 3),
                                uncertainty_m=None,
                                evidence_refs=ev_refs,
                                basis=f"Opening {a.semantic_label} overlaps wall in {a.source_image_id}",
                            ))
                        elif b.detection_category == "opening" and a.semantic_label.lower() == "wall":
                            rels.append(CandidateRelationship(
                                source_observation_id=b.observation_id,
                                target_observation_id=a.observation_id,
                                rel_type=RelationshipType.PART_OF,
                                coordinate_frame="image_space",
                                support_state=RelationshipSupport.SEMANTICALLY_INFERRED,
                                confidence=round(min(0.8, iou + 0.2), 3),
                                uncertainty_m=None,
                                evidence_refs=ev_refs,
                                basis=f"Opening {b.semantic_label} overlaps wall in {a.source_image_id}",
                            ))
                        da = a.visual_attributes.get("estimated_depth_raw")
                        db = b.visual_attributes.get("estimated_depth_raw")
                        if isinstance(da, (int, float)) and isinstance(db, (int, float)) and abs(da - db) > 0.05:
                            front, back = (a, b) if da < db else (b, a)
                            rels.append(CandidateRelationship(
                                source_observation_id=front.observation_id,
                                target_observation_id=back.observation_id,
                                rel_type=RelationshipType.BLOCKING,
                                coordinate_frame="image_space",
                                support_state=RelationshipSupport.VISUALLY_OBSERVED,
                                confidence=round(min(0.9, iou + 0.2), 3),
                                uncertainty_m=None,
                                evidence_refs=ev_refs,
                                basis=f"Foreground {front.semantic_label} occludes {back.semantic_label} (IoU={iou:.2f})",
                            ))
                    cx_a = (a.bbox_px[0] + a.bbox_px[2]) / 2.0
                    cx_b = (b.bbox_px[0] + b.bbox_px[2]) / 2.0
                    cy_a = (a.bbox_px[1] + a.bbox_px[3]) / 2.0
                    cy_b = (b.bbox_px[1] + b.bbox_px[3]) / 2.0
                    if abs(cx_a - cx_b) > abs(cy_a - cy_b):
                        rtype = RelationshipType.LEFT_OF if cx_a < cx_b else RelationshipType.RIGHT_OF
                    else:
                        rtype = RelationshipType.ABOVE if cy_a < cy_b else RelationshipType.BELOW
                    rels.append(CandidateRelationship(
                        source_observation_id=a.observation_id,
                        target_observation_id=b.observation_id,
                        rel_type=rtype,
                        coordinate_frame="image_space",
                        support_state=RelationshipSupport.VISUALLY_OBSERVED,
                        confidence=0.65,
                        uncertainty_m=None,
                        evidence_refs=ev_refs,
                        basis=f"2D pixel center offset in {a.source_image_id}",
                    ))

            # 2. World-space relationships (requires 3D world_pos on both observations)
            if a.world_pos is not None and b.world_pos is not None:
                is_metric = a.depth_source in (DepthSource.METRIC, DepthSource.MVS) and b.depth_source in (DepthSource.METRIC, DepthSource.MVS)
                frame = "world_metric" if is_metric else "world_relative"
                support = RelationshipSupport.GEOMETRICALLY_SUPPORTED if is_metric else RelationshipSupport.WEAKLY_SUPPORTED
                dist = math.dist(a.world_pos, b.world_pos)
                unc = round(math.hypot(a.sigma_m or 0.25, b.sigma_m or 0.25), 4)

                # Vertical support / floor resting (order-independent)
                dz_ab = a.world_pos[2] - b.world_pos[2]
                xy_dist = math.hypot(a.world_pos[0] - b.world_pos[0], a.world_pos[1] - b.world_pos[1])
                if b.semantic_label.lower() == "floor" and 0.0 <= dz_ab <= 1.2:
                    rels.append(CandidateRelationship(
                        source_observation_id=a.observation_id,
                        target_observation_id=b.observation_id,
                        rel_type=RelationshipType.SUPPORTED_BY,
                        coordinate_frame=frame,
                        support_state=support,
                        confidence=0.75 if is_metric else 0.45,
                        uncertainty_m=unc,
                        evidence_refs=ev_refs,
                        basis=f"3D vertical offset dz={dz_ab:.2f}m above floor",
                    ))
                elif a.semantic_label.lower() == "floor" and 0.0 <= -dz_ab <= 1.2:
                    rels.append(CandidateRelationship(
                        source_observation_id=b.observation_id,
                        target_observation_id=a.observation_id,
                        rel_type=RelationshipType.SUPPORTED_BY,
                        coordinate_frame=frame,
                        support_state=support,
                        confidence=0.75 if is_metric else 0.45,
                        uncertainty_m=unc,
                        evidence_refs=ev_refs,
                        basis=f"3D vertical offset dz={-dz_ab:.2f}m above floor",
                    ))
                # Opening within wall candidate
                elif a.detection_category == "opening" and b.semantic_label.lower() == "wall" and dist <= adjacency_radius_m:
                    rels.append(CandidateRelationship(
                        source_observation_id=a.observation_id,
                        target_observation_id=b.observation_id,
                        rel_type=RelationshipType.ATTACHED_TO,
                        coordinate_frame=frame,
                        support_state=support,
                        confidence=0.7 if is_metric else 0.4,
                        uncertainty_m=unc,
                        evidence_refs=ev_refs,
                        basis=f"Opening within {dist:.2f}m of wall observation",
                    ))
                elif dist <= adjacency_radius_m:
                    rels.append(CandidateRelationship(
                        source_observation_id=a.observation_id,
                        target_observation_id=b.observation_id,
                        rel_type=RelationshipType.ADJACENT_TO,
                        coordinate_frame=frame,
                        support_state=support,
                        confidence=round(max(0.3, 1.0 - dist / adjacency_radius_m), 3),
                        uncertainty_m=unc,
                        evidence_refs=ev_refs,
                        basis=f"3D separation {dist:.2f}m <= {adjacency_radius_m:.2f}m (xy={xy_dist:.2f}m)",
                    ))
    return rels


# ---------------------------------------------------------------------------
# Phase 2: Room & Environment Classification (Multi-Cue Only)
# ---------------------------------------------------------------------------

ROOM_CUE_TABLE: Dict[str, Tuple[str, ...]] = {
    "living_room": ("sofa", "couch", "tv", "coffee table", "armchair", "bookshelf"),
    "bedroom": ("bed", "nightstand", "dresser", "wardrobe", "lamp"),
    "kitchen": ("refrigerator", "oven", "microwave", "sink", "stove", "cabinet"),
    "bathroom": ("toilet", "sink", "shower", "bathtub", "mirror"),
    "dining_room": ("dining table", "dining chair", "chair"),
    "office": ("desk", "monitor", "laptop", "chair", "bookshelf"),
    "conference_room": ("dining table", "table", "chair", "tv", "monitor"),
    "corridor": ("corridor", "door", "railing"),
    "storage_room": ("cabinet", "bookshelf", "shelf"),
    "utility_room": ("sink", "cabinet", "refrigerator", "pipe"),
    "retail_space": ("shelf", "cabinet", "counter", "display"),
    "commercial_workspace": ("desk", "chair", "monitor", "column", "beam"),
}

ROOM_FUNCTION_MAP: Dict[str, str] = {
    "living_room": "residential_living",
    "bedroom": "sleeping",
    "kitchen": "food_prep",
    "bathroom": "hygiene",
    "dining_room": "dining",
    "office": "work_collaboration",
    "conference_room": "work_collaboration",
    "corridor": "circulation",
    "storage_room": "storage",
    "utility_room": "wet_utility",
    "retail_space": "retail",
    "commercial_workspace": "work_collaboration",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class RoomHypothesis:
    room_type: str
    score: float
    supporting_observation_ids: Tuple[str, ...]
    supporting_labels: Tuple[str, ...]
    status: str  # "plausible" | "ambiguous" | "unknown"
    room_function: str = "unknown"
    room_identity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supporting_observation_ids"] = list(self.supporting_observation_ids)
        d["supporting_labels"] = list(self.supporting_labels)
        return d


def classify_room_hypotheses(
    observations: Sequence[VisualObservation],
    *,
    ocr_tokens: Sequence[str] = (),
) -> List[RoomHypothesis]:
    """Infer ranked room hypotheses from >=2 distinct cues; 0 or 1 cue stays UNKNOWN."""
    label_to_obs: Dict[str, List[str]] = {}
    for o in observations:
        lbl = o.semantic_label.strip().lower()
        if lbl and lbl != "unknown":
            label_to_obs.setdefault(lbl, []).append(o.observation_id)

    ocr_low = {t.strip().lower() for t in ocr_tokens if t.strip()}
    hypotheses: List[RoomHypothesis] = []
    for room_type, cues in ROOM_CUE_TABLE.items():
        matched_labels = [c for c in cues if c in label_to_obs]
        if room_type.replace("_", " ") in ocr_low:
            matched_labels.append(f"ocr:{room_type}")
        # Never classify a room from one isolated object alone
        if len(matched_labels) < 2:
            continue
        obs_ids = tuple(sorted({oid for lbl in matched_labels for oid in label_to_obs.get(lbl, ())}))
        score = round(min(0.95, len(matched_labels) / max(3.0, len(cues) * 0.7)), 3)
        hypotheses.append(RoomHypothesis(
            room_type=room_type,
            score=score,
            supporting_observation_ids=obs_ids,
            supporting_labels=tuple(matched_labels),
            status="plausible",
            room_function=ROOM_FUNCTION_MAP.get(room_type, "unknown"),
        ))

    if not hypotheses:
        return [RoomHypothesis(
            room_type="unknown",
            score=0.0,
            supporting_observation_ids=tuple(o.observation_id for o in observations[:10]),
            supporting_labels=tuple(sorted(label_to_obs.keys())),
            status="unknown",
            room_function="unknown",
        )]

    hypotheses.sort(key=lambda h: h.score, reverse=True)
    if len(hypotheses) >= 2 and abs(hypotheses[0].score - hypotheses[1].score) < 0.15:
        return [
            RoomHypothesis(
                room_type=h.room_type,
                score=h.score,
                supporting_observation_ids=h.supporting_observation_ids,
                supporting_labels=h.supporting_labels,
                status="ambiguous",
                room_function=h.room_function,
            )
            for h in hypotheses
        ]
    return hypotheses


# ---------------------------------------------------------------------------
# Phase 5 & 6: Perception Observability & Active Capture Guidance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaptureSuggestion:
    kind: str  # "suggested_viewpoint" | "hypothetical_viewpoint" | "unreachable_viewpoint"
    action: str
    reason: str
    priority: str  # "high" | "medium" | "low"
    missing_evidence_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneObservabilityReport:
    observation_count: int
    distinct_views: int
    architectural_elements_seen: int
    objects_seen: int
    blurry_views: int
    low_light_views: int
    ambiguous_labels: int
    unresolved_relationships: int
    suggestions: List[CaptureSuggestion]
    missing_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["suggestions"] = [s.to_dict() for s in self.suggestions]
        return d


def assess_scene_observability(
    observations: Sequence[VisualObservation],
    relationships: Sequence[CandidateRelationship],
    *,
    processed_views: Sequence[str] = (),
) -> SceneObservabilityReport:
    """Measure perception-level observability and generate grounded capture suggestions."""
    views = {o.source_image_id for o in observations} | set(processed_views)
    arch_labels = {o.semantic_label.lower() for o in observations if o.detection_category in ("architecture", "opening")}
    obj_count = sum(1 for o in observations if o.detection_category == "object")
    blurry = sum(
        1 for o in observations
        if (o.processing_config.get("image_quality") or {}).get("is_blurry") is True
    )
    motion_blur = sum(
        1 for o in observations
        if (o.processing_config.get("image_quality") or {}).get("is_motion_blurred") is True
    )
    low_light = sum(
        1 for o in observations
        if (o.processing_config.get("image_quality") or {}).get("is_underexposed") is True
    )
    noisy = sum(
        1 for o in observations
        if (o.processing_config.get("image_quality") or {}).get("is_noisy") is True
    )
    ambiguous = sum(1 for o in observations if o.semantic_label.lower() == "unknown")
    unresolved_rels = sum(
        1 for r in relationships
        if r.support_state in (RelationshipSupport.WEAKLY_SUPPORTED, RelationshipSupport.UNKNOWN)
    )

    missing_evidence: List[str] = []
    suggestions: List[CaptureSuggestion] = []

    if len(views) < 2:
        missing_evidence.append("insufficient_camera_viewpoints")
        suggestions.append(CaptureSuggestion(
            kind="suggested_viewpoint",
            action="Capture additional viewpoints from opposite corners",
            reason=f"Only {len(views)} viewpoint(s) observed; multi-view overlap needed for 3D correspondence.",
            priority="high",
            missing_evidence_type="insufficient_camera_viewpoints",
        ))
    if "wall" not in arch_labels:
        missing_evidence.append("unseen_wall_surfaces")
        suggestions.append(CaptureSuggestion(
            kind="hypothetical_viewpoint",
            action="Capture enclosing walls and corners",
            reason="No wall observations detected yet.",
            priority="high",
            missing_evidence_type="unseen_wall_surfaces",
        ))
    if "floor" not in arch_labels:
        missing_evidence.append("incomplete_floor_visibility")
        suggestions.append(CaptureSuggestion(
            kind="hypothetical_viewpoint",
            action="Capture downward view showing floor plane and wall-floor junctions",
            reason="No floor observations recorded.",
            priority="medium",
            missing_evidence_type="incomplete_floor_visibility",
        ))
    if "ceiling" not in arch_labels:
        missing_evidence.append("incomplete_ceiling_visibility")
        suggestions.append(CaptureSuggestion(
            kind="hypothetical_viewpoint",
            action="Photograph the ceiling",
            reason="No ceiling observations recorded.",
            priority="low",
            missing_evidence_type="incomplete_ceiling_visibility",
        ))
    if "door" not in arch_labels and "doorway" not in arch_labels:
        missing_evidence.append("partially_visible_doorways")
    if "window" not in arch_labels:
        missing_evidence.append("unresolved_windows")
    if blurry > 0 or motion_blur > 0:
        me_key = "motion_blur" if motion_blur > 0 else "blurry_views"
        missing_evidence.append(me_key)
        suggestions.append(CaptureSuggestion(
            kind="suggested_viewpoint",
            action="Re-capture blurry frames while holding camera steady",
            reason=f"{blurry} observation(s) came from blurry frames (motion_blur={motion_blur}).",
            priority="medium",
            missing_evidence_type=me_key,
        ))
    if low_light > 0:
        missing_evidence.append("underexposed_regions")
        suggestions.append(CaptureSuggestion(
            kind="suggested_viewpoint",
            action="Improve lighting and re-capture dark areas",
            reason=f"{low_light} observation(s) were underexposed.",
            priority="medium",
            missing_evidence_type="underexposed_regions",
        ))
    if noisy > 0:
        missing_evidence.append("high_sensor_noise")
    if unresolved_rels > 0:
        missing_evidence.append("poorly_observed_structural_boundaries")
        suggestions.append(CaptureSuggestion(
            kind="suggested_viewpoint",
            action="Move closer or capture metric depth / scale reference",
            reason=f"{unresolved_rels} spatial relationship(s) lack metric 3D support.",
            priority="medium",
            missing_evidence_type="poorly_observed_structural_boundaries",
        ))

    return SceneObservabilityReport(
        observation_count=len(observations),
        distinct_views=len(views),
        architectural_elements_seen=len(arch_labels),
        objects_seen=obj_count,
        blurry_views=blurry,
        low_light_views=low_light,
        ambiguous_labels=ambiguous,
        unresolved_relationships=unresolved_rels,
        suggestions=suggestions,
        missing_evidence=missing_evidence,
    )


# ---------------------------------------------------------------------------
# Phase 7: Material & Appearance Intelligence
# ---------------------------------------------------------------------------

def extract_appearance_and_materials(
    image_rgb: np.ndarray,
    bbox_px: Optional[Tuple[float, float, float, float]],
    kind: str,
    *,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Extract colour/roughness appearance and ranked hypotheses (never physical material facts)."""
    import cv2

    h, w = image_rgb.shape[:2]
    used_mask = False
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == (h, w) and int(m.sum()) >= 16:
            pixels = image_rgb[m]
            gray_pixels = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)[m] if image_rgb.ndim == 3 else image_rgb[m]
            mean_rgb = [round(float(c), 1) for c in pixels.mean(axis=0)]
            roughness = round(float(np.std(gray_pixels)) / 128.0, 4)
            used_mask = True
        else:
            mask = None

    if not used_mask:
        if bbox_px is not None:
            x0, y0 = max(0, min(w - 1, int(bbox_px[0]))), max(0, min(h - 1, int(bbox_px[1])))
            x1, y1 = max(x0 + 1, min(w, int(bbox_px[2]))), max(y0 + 1, min(h, int(bbox_px[3])))
            crop = image_rgb[y0:y1, x0:x1]
        else:
            crop = image_rgb
        if crop.size == 0:
            return {"mean_color_rgb": None, "texture_roughness": None, "material_hypotheses": [], "material_verified": False, "mask_cropped": False}
        mean_rgb = [round(float(c), 1) for c in crop.mean(axis=(0, 1))]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        roughness = round(float(np.std(gray)) / 128.0, 4)

    hyps = hypothesize_materials(mean_rgb, kind, texture_roughness=roughness)
    return {
        "mean_color_rgb": mean_rgb,
        "texture_roughness": roughness,
        "material_hypotheses": hyps,
        "material_verified": False,
        "mask_cropped": used_mask,
    }


# ---------------------------------------------------------------------------
# Phase 8: Temporal Scene Tracking
# ---------------------------------------------------------------------------

def track_temporal_scene(observations: Sequence[VisualObservation]) -> Dict[str, Any]:
    """Associate timed 3D observations into persistent tracks; preserve untimed separately."""
    timed = [
        TimedObservation(
            observation_id=o.observation_id,
            label=o.semantic_label,
            position=o.world_pos,
            evidence_id=o.source_evidence_id,
            captured_at=o.timestamp_s,
            confidence=o.confidences.detection,
        )
        for o in observations
        if o.world_pos is not None
    ]
    tracks, untimed = link_temporally(timed)
    return {
        "tracks": [t.to_dict() for t in tracks],
        "untimed_observation_ids": [u.observation_id for u in untimed],
        "track_count": len(tracks),
    }


# ---------------------------------------------------------------------------
# Phase 9 & 11: Unified Scene Graph + Sequential RTX 4050 Pipeline + WorldIR Adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualSceneGraph:
    observations: Tuple[VisualObservation, ...]
    relationships: Tuple[CandidateRelationship, ...]
    room_hypotheses: Tuple[RoomHypothesis, ...]
    observability: SceneObservabilityReport
    temporal_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_count": len(self.observations),
            "hierarchical_labels": {
                o.observation_id: list(resolve_hierarchical_label(o.semantic_label, o.confidences.detection)[1])
                for o in self.observations
            },
            "relationships": [r.to_dict() for r in self.relationships],
            "room_hypotheses": [h.to_dict() for h in self.room_hypotheses],
            "observability": self.observability.to_dict(),
            "temporal_summary": self.temporal_summary,
        }


def build_visual_scene_graph(
    observations: Sequence[VisualObservation],
    *,
    ocr_tokens: Sequence[str] = (),
    processed_views: Sequence[str] = (),
) -> VisualSceneGraph:
    rels = extract_candidate_relationships(observations)
    rooms = classify_room_hypotheses(observations, ocr_tokens=ocr_tokens)
    obs_report = assess_scene_observability(observations, rels, processed_views=processed_views)
    temporal = track_temporal_scene(observations)
    return VisualSceneGraph(
        observations=tuple(observations),
        relationships=tuple(rels),
        room_hypotheses=tuple(rooms),
        observability=obs_report,
        temporal_summary=temporal,
    )


def extract_structural_regions(scene_graph: VisualSceneGraph) -> Dict[str, Any]:
    """Summarize structural surface regions (walls, floors, ceilings, openings, boundaries).

    A detected/segmented wall region is explicitly NOT a reconstructed 3D wall plane;
    observations without metric geometry stay marked ``plane_reconstructed=False``.
    """
    walls, floors, ceilings, openings = [], [], [], []
    for obs in scene_graph.observations:
        lbl = obs.semantic_label.lower()
        seg_q = (obs.processing_config or {}).get("segmentation_quality") or {}
        depth_ev = (obs.processing_config or {}).get("depth_evidence") or {}
        rec = {
            "observation_id": obs.observation_id,
            "image_id": obs.source_image_id,
            "label": lbl,
            "has_valid_mask": seg_q.get("validity") == "valid",
            "median_depth": depth_ev.get("median_depth"),
            "metric_scale_status": depth_ev.get("metric_scale_status", "relative_uncalibrated"),
            "plane_reconstructed": bool(obs.wall_slab is not None and obs.wall_slab.start_xyz_m is not None),
            "partial_or_unresolved": obs.world_pos is None,
        }
        if lbl == "wall":
            walls.append(rec)
        elif lbl == "floor":
            floors.append(rec)
        elif lbl == "ceiling":
            ceilings.append(rec)
        elif obs.detection_category == "opening" or lbl in ("door", "doorway", "window", "window frame", "archway", "structural opening"):
            openings.append(rec)

    transitions = [
        r.to_dict()
        for r in scene_graph.relationships
        if r.rel_type in (RelationshipType.PART_OF, RelationshipType.ADJACENT_TO, RelationshipType.ATTACHED_TO, RelationshipType.BLOCKING)
    ]
    return {
        "wall_regions": walls,
        "floor_regions": floors,
        "ceiling_regions": ceilings,
        "opening_regions": openings,
        "structural_transitions": transitions,
        "counts": {
            "walls": len(walls),
            "floors": len(floors),
            "ceilings": len(ceilings),
            "openings": len(openings),
            "transitions": len(transitions),
        },
    }


def bridge_scene_to_world(
    world: WorldIR,
    observations: Sequence[VisualObservation],
    *,
    ocr_tokens: Sequence[str] = (),
) -> Tuple[VisualSceneGraph, BridgeExecutionReport]:
    """Attach VisualSceneGraph + observations into WorldIR and run Claude's compiler checks."""
    sg = build_visual_scene_graph(observations, ocr_tokens=ocr_tokens)
    world.metadata["scene_intelligence"] = sg.to_dict()
    report = bridge_observations_to_world(world, observations)
    return sg, report


def run_sequential_perception_pipeline(
    image_paths: Sequence[str],
    *,
    score_threshold: float = 0.35,
    owlvit_threshold: float = 0.12,
    use_owlvit: bool = True,
    use_grounding_dino: bool = False,
    grounding_dino_threshold: float = 0.28,
    use_florence2: bool = False,
    use_sam2: bool = False,
    use_dinov2: bool = False,
    queries: Sequence[str] = ARCHITECTURAL_QUERIES,
    max_edge: Optional[int] = None,
    extract_materials: bool = False,
) -> Dict[str, Any]:
    """Run Depth Anything V2 -> Mask R-CNN -> OWL-ViT -> Grounding DINO -> Florence-2 -> SAM2 -> DINOv2 sequentially on RTX 4050."""
    import cv2
    import tempfile
    from pathlib import Path
    import torch
    from PIL import Image
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.availability import real_models_by_name
    from perception.depth.depth_anything_backend import DepthAnythingV2Backend
    from perception.detection.backends import OwlVitDetector
    from perception.detection.maskrcnn_backend import MaskRCNNDetector
    from perception.detection.normalized import (
        CoordinateConvention,
        NormalizedDetection,
        normalize_backend_output,
    )
    from perception.evidence_enrichment import (
        DINOv2LocalEmbedder,
        associate_cross_view_detections,
        compute_observation_depth_evidence,
    )
    from perception.segmentation.arch_segmenter import segment_normalized_detections

    avail = real_models_by_name()
    if avail["depth_anything"].status != "available" or avail["mask_rcnn"].status != "available":
        raise RuntimeError("Depth Anything V2 or Mask R-CNN backend unavailable")

    def _prepare_rgb(path: str) -> np.ndarray:
        bgr = cv2.imread(path)
        if bgr is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if max_edge is not None and max(rgb.shape[:2]) > max_edge:
            h, w = rgb.shape[:2]
            scale = max_edge / float(max(h, w))
            rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return rgb

    rgb_images = [_prepare_rgb(str(p)) for p in image_paths]
    tmp_dir_obj = tempfile.TemporaryDirectory() if max_edge is not None else None
    try:
        effective_paths: List[str] = []
        if tmp_dir_obj is not None:
            for i, rgb in enumerate(rgb_images):
                tp = str(Path(tmp_dir_obj.name) / f"resized_{i}.jpg")
                cv2.imwrite(tp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                effective_paths.append(tp)
        else:
            effective_paths = [str(p) for p in image_paths]

        items = [EvidenceItem(id=f"ev_{i}", kind=EvidenceKind.PHOTO, source_uri=ep) for i, ep in enumerate(effective_paths)]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            free_before, total_vram = torch.cuda.mem_get_info()
        else:
            free_before, total_vram = 0, 0

        t_start = time.perf_counter()

        # Stage 1: Depth Anything V2 (then unload)
        t0 = time.perf_counter()
        depth_backend = DepthAnythingV2Backend()
        depth_maps = depth_backend.estimate_depth(items)
        t_depth = time.perf_counter() - t0
        depth_backend.unload()

        # Stage 2: Mask R-CNN (then unload)
        t1 = time.perf_counter()
        detector = MaskRCNNDetector(score_threshold=score_threshold)
        det_results = detector.detect(items)
        t_det = time.perf_counter() - t1
        detector.unload()

        # Stage 3: Open-vocabulary OWL-ViT architectural detector (then unload)
        owl_results: List[List[Any]] = [[] for _ in items]
        t_owl = 0.0
        if use_owlvit and avail["owlvit"].status == "available":
            t2 = time.perf_counter()
            owl = OwlVitDetector(threshold=owlvit_threshold)
            for idx, rgb in enumerate(rgb_images):
                owl_results[idx] = owl.detect(rgb, queries)
            t_owl = time.perf_counter() - t2
            owl.unload()

        # Stage 4: Grounding DINO open-vocabulary architectural detector (then unload)
        gd_batches: List[Any] = [None for _ in items]
        t_gdino = 0.0
        if use_grounding_dino and avail.get("grounding_dino") and avail["grounding_dino"].status == "available":
            from perception.detection.grounded_arch import GroundingDinoDetector

            t3 = time.perf_counter()
            gd = GroundingDinoDetector(box_threshold=grounding_dino_threshold)
            for idx, (item, rgb) in enumerate(zip(items, rgb_images)):
                gd_batches[idx] = gd.detect_normalized(rgb, queries, image_id=item.id)
            t_gdino = time.perf_counter() - t3
            gd.unload()

        # Stage 5: Florence-2 second-opinion detector (then unload)
        fl_batches: List[Any] = [None for _ in items]
        t_florence = 0.0
        if use_florence2 and avail.get("florence2") and avail["florence2"].status == "available":
            from perception.detection.florence2_detector import Florence2Detector

            t4 = time.perf_counter()
            fl = Florence2Detector()
            for idx, (item, rgb) in enumerate(zip(items, rgb_images)):
                h_i, w_i = rgb.shape[:2]
                raw_fl = fl.detect(Image.fromarray(rgb))
                fl_batches[idx] = normalize_backend_output(
                    raw_fl,
                    image_width=w_i,
                    image_height=h_i,
                    convention=CoordinateConvention.PIXEL_XYXY,
                    model_name="florence2",
                    model_version="Florence-2-base",
                    image_id=item.id,
                    clip=True,
                )
            t_florence = time.perf_counter() - t4
            fl.unload()

        # Normalize Mask R-CNN and OWL-ViT via single normalization boundary
        all_norm_batches: List[List[Any]] = [[] for _ in items]
        for idx, (item, rgb, dres, o_dets) in enumerate(zip(items, rgb_images, det_results, owl_results)):
            h_i, w_i = rgb.shape[:2]
            all_norm_batches[idx].append(
                normalize_backend_output(
                    dres.detections,
                    image_width=w_i,
                    image_height=h_i,
                    model_name="maskrcnn_resnet50_fpn",
                    model_version="coco_v1",
                    image_id=item.id,
                    clip=True,
                )
            )
            if o_dets:
                all_norm_batches[idx].append(
                    normalize_backend_output(
                        o_dets,
                        image_width=w_i,
                        image_height=h_i,
                        model_name="owlvit-base-patch32",
                        model_version="1.0",
                        image_id=item.id,
                        clip=True,
                    )
                )
            if gd_batches[idx] is not None:
                all_norm_batches[idx].append(gd_batches[idx])
            if fl_batches[idx] is not None:
                all_norm_batches[idx].append(fl_batches[idx])

        # Stage 6: SAM 2.1 architectural segmentation (then unload)
        from perception.detection.normalized import suppress_duplicate_normalized_detections

        dedup_batches: List[Tuple[NormalizedDetection, ...]] = []
        for idx in range(len(items)):
            flat_raw = [d for b in all_norm_batches[idx] for d in b.detections]
            kept_dets, _ = suppress_duplicate_normalized_detections(flat_raw, iou_threshold=0.65)
            dedup_batches.append(kept_dets)

        seg_outputs: List[Any] = [None for _ in items]
        mask_by_det_id: Dict[str, Any] = {}
        t_sam2 = 0.0
        if use_sam2 and avail.get("sam2_hf") and avail["sam2_hf"].status == "available":
            from perception.segmentation.sam2_box_prompter import Sam2BoxPrompter

            t5 = time.perf_counter()
            sam = Sam2BoxPrompter()
            for idx, (item, rgb, dmap) in enumerate(zip(items, rgb_images, depth_maps)):
                flat_dets = list(dedup_batches[idx])
                if flat_dets:
                    depth_arr_i = np.asarray(dmap.values, dtype=np.float32)
                    seg_out = segment_normalized_detections(
                        rgb,
                        flat_dets,
                        image_id=item.id,
                        depth_map=depth_arr_i,
                        prompter=sam,
                        unload_after=False,
                    )
                    seg_outputs[idx] = seg_out
                    for em in seg_out.masks:
                        mask_by_det_id[em.detection_id] = em
            t_sam2 = time.perf_counter() - t5
            sam.unload()

        # Stage 7: DINOv2 cross-view embeddings (then unload)
        t_dinov2 = 0.0
        dinov2_embs: Dict[str, np.ndarray] = {}
        if use_dinov2 and avail.get("dinov2") and avail["dinov2"].status == "available":
            t6 = time.perf_counter()
            emb_model = DINOv2LocalEmbedder()
            crops: List[np.ndarray] = []
            crop_ids: List[str] = []
            for rgb, batches in zip(rgb_images, all_norm_batches):
                h_i, w_i = rgb.shape[:2]
                for b in batches:
                    for d in b.detections:
                        x0 = int(max(0, min(w_i - 1, d.bbox_px[0])))
                        y0 = int(max(0, min(h_i - 1, d.bbox_px[1])))
                        x1 = int(max(x0 + 1, min(w_i, d.bbox_px[2])))
                        y1 = int(max(y0 + 1, min(h_i, d.bbox_px[3])))
                        crops.append(rgb[y0:y1, x0:x1])
                        crop_ids.append(d.detection_id)
            if crops:
                vecs = emb_model.embed_crops(crops)
                for cid, vec in zip(crop_ids, vecs):
                    dinov2_embs[cid] = vec
            t_dinov2 = time.perf_counter() - t6
            emb_model.unload()

        if torch.cuda.is_available():
            free_after, _ = torch.cuda.mem_get_info()
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            total_vram_mb = total_vram / (1024 * 1024)
            device_used_delta_mb = max(0.0, (free_before - free_after) / (1024 * 1024))
        else:
            peak_alloc_mb, total_vram_mb, device_used_delta_mb = 0.0, 0.0, 0.0

        all_obs: List[VisualObservation] = []
        all_flat_norm: List[NormalizedDetection] = []
        rgb_by_view = {it.id: rgb for it, rgb in zip(items, rgb_images)}
        for item, rgb, dmap, batches in zip(items, rgb_images, depth_maps, all_norm_batches):
            qual = assess_image_quality(rgb)
            depth_arr = np.asarray(dmap.values, dtype=np.float32)

            for batch in batches:
                if not batch.detections:
                    continue
                det_dicts: List[Dict[str, Any]] = []
                for d in batch.detections:
                    all_flat_norm.append(d)
                    em = mask_by_det_id.get(d.detection_id)
                    m_arr = em.mask if (em is not None and em.validity == "valid") else None
                    depth_ev = compute_observation_depth_evidence(
                        d.detection_id,
                        depth_arr,
                        d.bbox_px,
                        mask=m_arr,
                        depth_source=DepthSource.RELATIVE,
                        depth_model_name="depth_anything_v2_small",
                        depth_model_version="hf_v1",
                    )
                    entry: Dict[str, Any] = {
                        "label": d.canonical_label,
                        "bbox_px": list(d.bbox_px),
                        "score": d.confidence if d.confidence is not None else 0.5,
                        "depth_evidence": depth_ev.to_dict(),
                    }
                    if em is not None:
                        entry["segmentation_quality"] = {
                            "validity": em.validity,
                            "mask_to_box_ratio": em.mask_to_box_ratio,
                            "boundary_sharpness": em.boundary_sharpness,
                            "uncertainty": em.segmentation_uncertainty,
                            "model_name": em.model_name,
                        }
                    if extract_materials:
                        entry["material_info"] = extract_appearance_and_materials(
                            rgb, d.bbox_px, d.canonical_label, mask=m_arr
                        )
                    det_dicts.append(entry)

                all_obs.extend(
                    build_visual_observations_from_detections(
                        source_evidence_id=item.id,
                        source_image_id=item.id,
                        model_id=batch.model_name,
                        model_version=batch.model_version,
                        depth_map=depth_arr,
                        depth_source=DepthSource.RELATIVE,
                        quality=qual,
                        detections=det_dicts,
                    )
                )

        cross_view_assocs = associate_cross_view_detections(
            all_flat_norm,
            embeddings_by_id=dinov2_embs or None,
            rgb_by_view=rgb_by_view,
        )
        sg = build_visual_scene_graph(all_obs, processed_views=[it.id for it in items])
        contradictory = sum(1 for r in sg.relationships if r.support_state is RelationshipSupport.CONTRADICTORY)
        valid_masks = sum(1 for em in mask_by_det_id.values() if em.validity == "valid")
        structural_boundaries = sum(len(so.structural_boundaries) for so in seg_outputs if so is not None)
        return {
            "scene_graph": sg,
            "cross_view_associations": [a.to_dict() for a in cross_view_assocs],
            "segmentation_summary": {
                "masks_evaluated": len(mask_by_det_id),
                "valid_masks": valid_masks,
                "structural_boundaries": structural_boundaries,
            },
            "telemetry": {
                "cuda": torch.cuda.is_available(),
                "max_edge": max_edge,
                "image_count": len(items),
                "total_time_s": round(time.perf_counter() - t_start, 3),
                "depth_time_s": round(t_depth, 3),
                "detection_time_s": round(t_det, 3),
                "owlvit_time_s": round(t_owl, 3),
                "grounding_dino_time_s": round(t_gdino, 3),
                "florence2_time_s": round(t_florence, 3),
                "sam2_time_s": round(t_sam2, 3),
                "dinov2_time_s": round(t_dinov2, 3),
                "peak_torch_alloc_mb": round(peak_alloc_mb, 1),
                "device_vram_delta_mb": round(device_used_delta_mb, 1),
                "total_device_vram_mb": round(total_vram_mb, 1),
                "observations_total": len(all_obs),
                "valid_masks": valid_masks,
                "cross_model_disagreements": contradictory,
                "cross_view_associations": len(cross_view_assocs),
            },
        }
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()
