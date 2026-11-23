"""Room Completeness and Missing-Object Reasoning Engine (Sprint 6).

Assesses scene coverage, physical presence states, missing structural components,
and evidence gaps across interior rooms and exterior scenes.
Does not invent objects where evidence is missing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    EXPECTED_KEY_COMPONENTS,
)
from engine.multiengine.spatial_intelligence.taxonomy import (
    Domain,
    TaxonomyClass,
    resolve_taxonomy_class,
)


class SceneCategory(str, Enum):
    """Canonical scene and room categories."""

    BEDROOM = "bedroom"
    LIVING_ROOM = "living_room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"
    CORRIDOR = "corridor"
    EXTERIOR_BUILDING = "exterior_building"
    STREET_ENVIRONMENT = "street_environment"
    UNKNOWN = "unknown"


class EvidencePresenceState(str, Enum):
    """Grounded evidence states for physical objects."""

    NOT_PRESENT = "not_present"
    PRESENT_NOT_VISIBLE = "present_not_visible"
    POSSIBLY_PRESENT = "possibly_present"
    DETECTED_UNRESOLVED = "detected_unresolved"
    VERIFIED_PRESENT = "verified_present"


@dataclass(frozen=True)
class SceneCompletenessProfile:
    """Expected structural elements and characteristic furnishings for a scene category."""

    scene_category: SceneCategory
    required_structural: tuple[str, ...]  # Essential boundaries (e.g. floor, wall, door)
    typical_fixtures: tuple[str, ...]     # Common fixtures (e.g. socket, ceiling_light)
    characteristic_furnishings: tuple[str, ...]  # Furniture typical for this room type


SCENE_COMPLETENESS_PROFILES: dict[SceneCategory, SceneCompletenessProfile] = {
    SceneCategory.BEDROOM: SceneCompletenessProfile(
        scene_category=SceneCategory.BEDROOM,
        required_structural=("wall", "floor", "door"),
        typical_fixtures=("socket", "switch", "ceiling_light"),
        characteristic_furnishings=("bed", "wardrobe", "lamp"),
    ),
    SceneCategory.LIVING_ROOM: SceneCompletenessProfile(
        scene_category=SceneCategory.LIVING_ROOM,
        required_structural=("wall", "floor", "door", "window"),
        typical_fixtures=("socket", "switch", "ceiling_light"),
        characteristic_furnishings=("sofa", "table", "chair", "television"),
    ),
    SceneCategory.KITCHEN: SceneCompletenessProfile(
        scene_category=SceneCategory.KITCHEN,
        required_structural=("wall", "floor"),
        typical_fixtures=("socket", "switch", "sink", "faucet", "ceiling_light"),
        characteristic_furnishings=("cabinet", "refrigerator"),
    ),
    SceneCategory.BATHROOM: SceneCompletenessProfile(
        scene_category=SceneCategory.BATHROOM,
        required_structural=("wall", "floor", "door"),
        typical_fixtures=("sink", "faucet", "toilet", "mirror", "ceiling_light"),
        characteristic_furnishings=(),
    ),
    SceneCategory.OFFICE: SceneCompletenessProfile(
        scene_category=SceneCategory.OFFICE,
        required_structural=("wall", "floor", "door"),
        typical_fixtures=("socket", "switch", "ceiling_light"),
        characteristic_furnishings=("desk", "chair", "table", "shelf"),
    ),
    SceneCategory.CORRIDOR: SceneCompletenessProfile(
        scene_category=SceneCategory.CORRIDOR,
        required_structural=("wall", "floor", "ceiling", "door"),
        typical_fixtures=("ceiling_light", "switch"),
        characteristic_furnishings=(),
    ),
    SceneCategory.EXTERIOR_BUILDING: SceneCompletenessProfile(
        scene_category=SceneCategory.EXTERIOR_BUILDING,
        required_structural=("building_facade", "ground"),
        typical_fixtures=("door", "window"),
        characteristic_furnishings=(),
    ),
    SceneCategory.STREET_ENVIRONMENT: SceneCompletenessProfile(
        scene_category=SceneCategory.STREET_ENVIRONMENT,
        required_structural=("road", "footpath"),
        typical_fixtures=("lamppost", "bollard"),
        characteristic_furnishings=("bench", "trash_bin"),
    ),
}


@dataclass
class ObjectEvidenceAssessment:
    """Detailed evidence assessment for a detected or expected object."""

    entity_id: str
    class_name: str
    presence_state: EvidencePresenceState
    confidence: float
    uncertainty: float
    view_count: int
    is_partially_visible: bool = False
    is_occluded: bool = False
    is_identity_unresolved: bool = False
    missing_components: tuple[str, ...] = ()
    view_consistency_score: float = 1.0
    evidence_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "presence_state": self.presence_state.value,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "view_count": self.view_count,
            "is_partially_visible": self.is_partially_visible,
            "is_occluded": self.is_occluded,
            "is_identity_unresolved": self.is_identity_unresolved,
            "missing_components": list(self.missing_components),
            "view_consistency_score": round(self.view_consistency_score, 3),
            "evidence_notes": self.evidence_notes,
        }


@dataclass
class RoomCompletenessReport:
    """Comprehensive scene evidence and coverage audit report."""

    scene_category: SceneCategory
    coverage_score: float  # 0.0 to 1.0 based on structural boundaries and observation coverage
    evidence_completeness_score: float  # 0.0 to 1.0 based on verified vs unresolved ratio
    verified_entities: tuple[ObjectEvidenceAssessment, ...]
    unresolved_entities: tuple[ObjectEvidenceAssessment, ...]
    possibly_present_entities: tuple[str, ...]
    missing_expected_components: dict[str, tuple[str, ...]]
    coverage_gaps: tuple[str, ...]
    inconsistent_views: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_category": self.scene_category.value,
            "coverage_score": round(self.coverage_score, 3),
            "evidence_completeness_score": round(self.evidence_completeness_score, 3),
            "verified_count": len(self.verified_entities),
            "unresolved_count": len(self.unresolved_entities),
            "possibly_present": list(self.possibly_present_entities),
            "missing_components": {k: list(v) for k, v in self.missing_expected_components.items()},
            "coverage_gaps": list(self.coverage_gaps),
            "inconsistent_views": list(self.inconsistent_views),
            "verified_entities": [e.to_dict() for e in self.verified_entities],
            "unresolved_entities": [e.to_dict() for e in self.unresolved_entities],
        }


class RoomCompletenessEngine:
    """Evaluates scene completeness without fabricating unobserved objects."""

    def __init__(self) -> None:
        pass

    def infer_scene_category(self, observed_classes: Sequence[str]) -> SceneCategory:
        """Infers the scene category from the distribution of observed classes."""
        c_set = {c.lower() for c in observed_classes}

        # Check bathroom signatures
        if "toilet" in c_set or "shower" in c_set or "bathtub" in c_set:
            return SceneCategory.BATHROOM
        # Kitchen signatures
        if "refrigerator" in c_set or "oven" in c_set or "microwave" in c_set or "dishwasher" in c_set:
            return SceneCategory.KITCHEN
        # Bedroom signatures
        if "bed" in c_set or "wardrobe" in c_set:
            return SceneCategory.BEDROOM
        # Office signatures
        if "desk" in c_set and "chair" in c_set and not ("bed" in c_set or "sofa" in c_set):
            return SceneCategory.OFFICE
        # Living room signatures
        if "sofa" in c_set or ("table" in c_set and "chair" in c_set and "television" in c_set):
            return SceneCategory.LIVING_ROOM
        # Street / exterior signatures
        if "road" in c_set or "footpath" in c_set or "lamppost" in c_set:
            return SceneCategory.STREET_ENVIRONMENT
        if "building_facade" in c_set or "fence" in c_set:
            return SceneCategory.EXTERIOR_BUILDING

        return SceneCategory.UNKNOWN

    def assess_completeness(
        self,
        entities: Sequence[Any],
        forced_category: SceneCategory | None = None,
        camera_view_count: int = 1,
    ) -> RoomCompletenessReport:
        """Assesses evidence completeness across detected entities and expected room structures."""
        observed_classes: list[str] = []
        for e in entities:
            cname = getattr(e, "class_name", getattr(e, "primary_class", "unknown")).lower()
            observed_classes.append(cname)

        category = forced_category or self.infer_scene_category(observed_classes)
        profile = SCENE_COMPLETENESS_PROFILES.get(category, SceneCompletenessProfile(
            scene_category=category,
            required_structural=("wall", "floor"),
            typical_fixtures=("ceiling_light",),
            characteristic_furnishings=(),
        ))

        verified: list[ObjectEvidenceAssessment] = []
        unresolved: list[ObjectEvidenceAssessment] = []
        missing_comp_dict: dict[str, tuple[str, ...]] = {}
        coverage_gaps: list[str] = []
        inconsistent_views: list[str] = []

        # Audit each detected entity
        for ent in entities:
            eid = getattr(ent, "entity_id", getattr(ent, "canonical_entity_id", "ent"))
            cname = getattr(ent, "class_name", getattr(ent, "primary_class", "unknown")).lower()
            conf = float(getattr(ent, "confidence", 0.5))
            views = getattr(ent, "member_observation_ids", ())
            view_cnt = len(views) if views else 1
            is_mv = bool(getattr(ent, "is_multiview", view_cnt > 1))
            is_trunc = bool(getattr(ent, "is_truncated", False))
            occl = str(getattr(ent, "occlusion_level", "none"))

            # Missing parts check on host entities
            missing_parts: tuple[str, ...] = ()
            attrs = getattr(ent, "attributes", getattr(ent, "details", {}))
            if isinstance(attrs, dict) and "missing_expected_parts" in attrs:
                missing_parts = tuple(attrs["missing_expected_parts"])
            elif cname in EXPECTED_KEY_COMPONENTS:
                child_ids = getattr(ent, "child_box_ids", getattr(ent, "child_ids", ()))
                if not child_ids:
                    missing_parts = EXPECTED_KEY_COMPONENTS[cname]

            if missing_parts:
                missing_comp_dict[eid] = missing_parts

            notes: list[str] = []
            if is_trunc:
                notes.append("Truncated by image border")
            if occl in ("partial", "heavy"):
                notes.append(f"Occlusion level: {occl}")
            if not is_mv and camera_view_count > 1:
                notes.append("Single-view observation in multi-view capture")

            # Presence state
            if is_mv and conf >= 0.60:
                pstate = EvidencePresenceState.VERIFIED_PRESENT
            elif conf < 0.35:
                pstate = EvidencePresenceState.POSSIBLY_PRESENT
            else:
                pstate = EvidencePresenceState.DETECTED_UNRESOLVED

            assessment = ObjectEvidenceAssessment(
                entity_id=eid,
                class_name=cname,
                presence_state=pstate,
                confidence=conf,
                uncertainty=round(1.0 - conf, 4),
                view_count=view_cnt,
                is_partially_visible=is_trunc,
                is_occluded=occl in ("partial", "heavy"),
                is_identity_unresolved=not is_mv,
                missing_components=missing_parts,
                evidence_notes=notes,
            )

            if pstate == EvidencePresenceState.VERIFIED_PRESENT:
                verified.append(assessment)
            else:
                unresolved.append(assessment)

        # Audit structural requirements
        obs_set = set(observed_classes)
        for req in profile.required_structural:
            if not any(req in c for c in obs_set):
                coverage_gaps.append(f"Missing evidence for essential structural boundary: '{req}'")

        # Possibly present characteristic entities
        possibly_present: list[str] = []
        for char_ent in profile.characteristic_furnishings:
            if not any(char_ent in c for c in obs_set):
                possibly_present.append(f"{char_ent} (typical for {category.value}, not observed)")

        # Coverage score: fraction of required structural boundaries observed
        req_total = len(profile.required_structural)
        req_observed = sum(1 for req in profile.required_structural if any(req in c for c in obs_set))
        coverage_score = float(req_observed) / max(1.0, float(req_total))

        # Evidence completeness: verified vs total entities
        total_ents = len(verified) + len(unresolved)
        completeness_score = float(len(verified)) / max(1.0, float(total_ents))

        return RoomCompletenessReport(
            scene_category=category,
            coverage_score=coverage_score,
            evidence_completeness_score=completeness_score,
            verified_entities=tuple(verified),
            unresolved_entities=tuple(unresolved),
            possibly_present_entities=tuple(possibly_present),
            missing_expected_components=missing_comp_dict,
            coverage_gaps=tuple(coverage_gaps),
            inconsistent_views=tuple(inconsistent_views),
        )
