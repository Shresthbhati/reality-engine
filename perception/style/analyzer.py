"""Evidence-Grounded Architectural & Interior Design Style Analyzer (Wave 1, Block 8).

Evaluates the design style of a room, building, or reconstructed scene by synthesizing:
1. Material distributions (dominant & secondary surface materials).
2. Architectural proportions (window-to-wall ratios, ceiling-to-span ratios).
3. Ornamental density (presence of mouldings, cornices, balusters, carved elements vs flush planes).
4. Structural geometry (pointed arches, rounded arches, clean planar walls, exposed ducting/columns).
5. Color palette tendencies.

Integrity Rule:
Style assessment is an interpretive layer stored in metadata; it never fabricates or
mutates underlying measured geometric or physical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from perception.style.taxonomy import (
    DesignStyle,
    OrnamentationLevel,
    STYLE_PROFILES,
    StyleProfile,
)
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR


@dataclass
class StyleAssessment:
    """Evidence-grounded design style classification result."""
    primary_style: DesignStyle
    confidence: float
    secondary_style: Optional[DesignStyle] = None
    secondary_confidence: float = 0.0
    detected_ornamentation: OrnamentationLevel = OrnamentationLevel.LOW_RESTRAINED
    measured_window_to_wall_ratio: Optional[float] = None
    dominant_materials: List[str] = field(default_factory=list)
    matched_features: List[str] = field(default_factory=list)
    evidence_notes: List[str] = field(default_factory=list)
    style_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_style": self.primary_style.value,
            "confidence": round(self.confidence, 3),
            "secondary_style": self.secondary_style.value if self.secondary_style else None,
            "secondary_confidence": round(self.secondary_confidence, 3),
            "detected_ornamentation": self.detected_ornamentation.value,
            "measured_window_to_wall_ratio": round(self.measured_window_to_wall_ratio, 3) if self.measured_window_to_wall_ratio is not None else None,
            "dominant_materials": list(self.dominant_materials),
            "matched_features": list(self.matched_features),
            "evidence_notes": list(self.evidence_notes),
            "style_scores": {k: round(v, 3) for k, v in self.style_scores.items()},
        }


def analyze_scene_style(
    world: WorldIR,
    room_id: Optional[str] = None,
    visual_tags: Sequence[str] = (),
) -> StyleAssessment:
    """Analyze the architectural design style of a scene or specific room."""
    # 1. Collect entities in scope
    entities = list(world.entities.values())
    if room_id:
        entities = [e for e in entities if e.id == room_id or e.parent_id == room_id]

    # 2. Extract material distribution from entities and world materials
    material_counts: Dict[str, int] = {}
    for mat in world.materials.values():
        cat = mat.category.lower() if hasattr(mat, "category") and mat.category else "unknown"
        if cat != "unknown":
            material_counts[cat] = material_counts.get(cat, 0) + 1

    for e in entities:
        mat_prop = e.custom_properties.get("material") or e.custom_properties.get("material_category")
        if mat_prop:
            m_str = str(mat_prop).lower()
            material_counts[m_str] = material_counts.get(m_str, 0) + 1

    sorted_materials = [m for m, _ in sorted(material_counts.items(), key=lambda x: x[1], reverse=True)]
    top_materials = sorted_materials[:3]

    # 3. Assess ornamentation from architectural entities
    ornament_score = 0
    moulding_found = False
    arch_found = False
    exposed_conduit_found = False

    for e in entities:
        c_class = str(e.custom_properties.get("entity_class", "")).lower()
        if "moulding" in c_class or "cornice" in c_class or "skirting" in c_class:
            ornament_score += 2
            moulding_found = True
        elif "arch" in c_class:
            arch_found = True
            ornament_score += 1
        elif "conduit" in c_class or "duct" in c_class:
            exposed_conduit_found = True

    # Check tags as well
    tags_lower = [t.lower() for t in visual_tags]
    if any("carved" in t or "ornate" in t for t in tags_lower):
        ornament_score += 3
    if any("minimal" in t or "flush" in t for t in tags_lower):
        ornament_score -= 2

    if ornament_score <= 0:
        ornament_level = OrnamentationLevel.NONE_MINIMAL
    elif ornament_score <= 2:
        ornament_level = OrnamentationLevel.LOW_RESTRAINED
    elif ornament_score <= 4:
        ornament_level = OrnamentationLevel.MODERATE
    elif ornament_score <= 7:
        ornament_level = OrnamentationLevel.HIGH_ELABORATE
    else:
        ornament_level = OrnamentationLevel.HIGHLY_DECORATIVE

    # 4. Measure window-to-wall ratio if geometry bounds are present
    total_wall_area = 0.0
    total_window_area = 0.0
    for e in entities:
        e_type = str(e.type).lower() if hasattr(e, "type") else ""
        c_class = str(e.custom_properties.get("entity_class", "")).lower()
        dims = e.custom_properties.get("dimensions_m")
        if dims and len(dims) >= 2:
            area = float(dims[0]) * float(dims[1])
            if "wall" in e_type or "wall" in c_class:
                total_wall_area += area
            elif "window" in e_type or "window" in c_class:
                total_window_area += area

    wwr: Optional[float] = None
    if total_wall_area > 0.0:
        wwr = min(1.0, total_window_area / total_wall_area)

    # 5. Score against all style profiles
    scores: Dict[DesignStyle, float] = {}
    matched_features_map: Dict[DesignStyle, List[str]] = {}

    for style, profile in STYLE_PROFILES.items():
        score = 0.0
        matches: List[str] = []

        # Material overlap
        for mat in top_materials:
            if mat in profile.dominant_materials:
                score += 3.0
                matches.append(f"Dominant material match: {mat}")
            elif mat in profile.secondary_materials:
                score += 1.5
                matches.append(f"Secondary material match: {mat}")

        # Ornamentation compatibility
        if ornament_level == profile.ornamentation:
            score += 2.5
            matches.append(f"Ornamentation level matches {profile.ornamentation.value}")
        elif abs(list(OrnamentationLevel).index(ornament_level) - list(OrnamentationLevel).index(profile.ornamentation)) == 1:
            score += 1.0

        # WWR compatibility
        if wwr is not None:
            min_r, max_r = profile.window_to_wall_ratio
            if min_r <= wwr <= max_r:
                score += 2.0
                matches.append(f"Window-to-wall ratio {wwr:.2f} within range [{min_r}, {max_r}]")

        # Specific feature triggers
        if exposed_conduit_found and style == DesignStyle.INDUSTRIAL:
            score += 4.0
            matches.append("Exposed conduit/infrastructure detected")
        if arch_found and style in (DesignStyle.MEDITERRANEAN, DesignStyle.GOTHIC, DesignStyle.TRADITIONAL_INDIAN):
            score += 3.0
            matches.append("Arched architectural opening detected")
        if moulding_found and style in (DesignStyle.CLASSICAL, DesignStyle.ART_DECO, DesignStyle.TRADITIONAL_INDIAN):
            score += 2.5
            matches.append("Architectural mouldings/cornices present")

        # Visual tag keyword boost
        for tag in tags_lower:
            for feat in profile.distinguishing_features:
                if feat.lower() in tag or tag in feat.lower():
                    score += 2.0
                    matches.append(f"Visual feature match: '{feat}'")

        scores[style] = max(0.0, score)
        matched_features_map[style] = matches

    # Normalize scores
    total_score = sum(scores.values())
    if total_score > 0.0:
        normalized_scores = {s.value: v / total_score for s, v in scores.items()}
    else:
        normalized_scores = {s.value: 1.0 / len(scores) for s in scores}

    # Rank styles
    sorted_styles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_style, best_raw = sorted_styles[0]
    second_style, second_raw = sorted_styles[1]

    confidence = min(0.95, normalized_scores[best_style.value] * 1.8) if best_raw > 0 else 0.20
    second_conf = min(0.80, normalized_scores[second_style.value] * 1.8) if second_raw > 0 else 0.0

    notes: List[str] = [
        f"Assessed across {len(entities)} scene entities and {len(top_materials)} detected surface materials.",
    ]
    if wwr is not None:
        notes.append(f"Measured window-to-wall ratio: {wwr * 100:.1f}%.")
    else:
        notes.append("Window-to-wall ratio unmeasured due to insufficient planar facade data.")

    return StyleAssessment(
        primary_style=best_style if best_raw > 0 else DesignStyle.ECLECTIC_UNKNOWN,
        confidence=confidence,
        secondary_style=second_style if second_raw > 1.0 else None,
        secondary_confidence=second_conf,
        detected_ornamentation=ornament_level,
        measured_window_to_wall_ratio=wwr,
        dominant_materials=top_materials,
        matched_features=matched_features_map.get(best_style, []),
        evidence_notes=notes,
        style_scores=normalized_scores,
    )
