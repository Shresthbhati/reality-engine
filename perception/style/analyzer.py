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
        entities = [e for e in entities if e.id == room_id or e.custom_properties.get("room_id") == room_id]

    # 2. Extract material distribution from entities and world materials
    material_counts: Dict[str, int] = {}
    for mat in world.materials.values():
        cat = mat.class_name.lower() if hasattr(mat, "class_name") and mat.class_name else "unknown"
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

    if ornament_score >= 4 or any("ornate" in t.lower() for t in visual_tags):
        ornament_level = OrnamentationLevel.HIGH_ELABORATE
    elif ornament_score >= 2 or any("decorative" in t.lower() for t in visual_tags):
        ornament_level = OrnamentationLevel.MODERATE
    elif exposed_conduit_found or any("minimal" in t.lower() or "clean" in t.lower() for t in visual_tags):
        ornament_level = OrnamentationLevel.NONE_MINIMAL
    else:
        ornament_level = OrnamentationLevel.LOW_RESTRAINED

    # 4. Measure window-to-wall ratio if wall and window areas exist
    total_wall_area = 0.0
    total_window_area = 0.0
    for e in entities:
        if "wall" in str(e.type).lower() or "wall" in str(e.custom_properties.get("entity_class", "")).lower():
            area = float(e.custom_properties.get("area_m2") or e.custom_properties.get("length_m", 3.0) * e.custom_properties.get("height_m", 2.8))
            total_wall_area += area
        elif "window" in str(e.type).lower() or "window" in str(e.custom_properties.get("entity_class", "")).lower():
            area = float(e.custom_properties.get("area_m2") or e.custom_properties.get("width_m", 1.2) * e.custom_properties.get("height_m", 1.5))
            total_window_area += area

    wwr = (total_window_area / total_wall_area) if total_wall_area > 0.0 else None

    # 5. Score candidate styles against measured cues
    tags_lower = {t.strip().lower() for t in visual_tags}
    raw_scores: Dict[DesignStyle, float] = {s: 0.0 for s in DesignStyle}
    matched_features_map: Dict[DesignStyle, List[str]] = {s: [] for s in DesignStyle}

    for style, profile in STYLE_PROFILES.items():
        score = 0.0
        # Material matches
        for m in top_materials:
            if m in profile.dominant_materials:
                score += 3.0
                matched_features_map[style].append(f"Dominant material match: {m}")
            elif m in profile.secondary_materials:
                score += 1.5
                matched_features_map[style].append(f"Secondary material match: {m}")

        # Ornamentation agreement
        if profile.ornamentation == ornament_level:
            score += 2.0
            matched_features_map[style].append(f"Ornamentation match: {ornament_level.value}")

        # Geometric & semantic cues
        if style == DesignStyle.INDUSTRIAL and exposed_conduit_found:
            score += 4.0
            matched_features_map[style].append("Exposed mechanical conduits/ducts detected")

        if style in (DesignStyle.GOTHIC, DesignStyle.CLASSICAL, DesignStyle.TRADITIONAL_INDIAN) and arch_found:
            score += 3.0
            matched_features_map[style].append("Arched openings and structural geometry")

        if style in (DesignStyle.TRADITIONAL_INDIAN, DesignStyle.SOUTH_INDIAN_VERNACULAR, DesignStyle.CLASSICAL) and moulding_found:
            score += 2.0
            matched_features_map[style].append("Carved decorative mouldings and woodwork")

        if style in (DesignStyle.MINIMALIST, DesignStyle.CONTEMPORARY) and ornament_level == OrnamentationLevel.NONE_MINIMAL:
            score += 3.0
            matched_features_map[style].append("Flush planar unadorned surfaces")

        # Explicit visual tag and distinguishing feature boost
        for tag in tags_lower:
            if tag in style.value:
                score += 5.0
                matched_features_map[style].append(f"Visual prior tag match: '{tag}'")
            elif any(tag in feat.lower() or any(w in feat.lower() for w in tag.split() if len(w) > 3) for feat in profile.distinguishing_features):
                score += 4.0
                matched_features_map[style].append(f"Distinguishing feature match: '{tag}'")
            elif any(tag in p.lower() for p in profile.typical_palette):
                score += 2.0
                matched_features_map[style].append(f"Palette match: '{tag}'")

        raw_scores[style] = score

    # Normalize scores
    total_score = sum(raw_scores.values())
    if total_score > 0:
        normalized_scores = {s.value: raw_scores[s] / total_score for s in DesignStyle}
    else:
        normalized_scores = {s.value: 1.0 / len(DesignStyle) for s in DesignStyle}

    sorted_styles = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    best_style, best_raw = sorted_styles[0]
    second_style, second_raw = sorted_styles[1]

    confidence = min(0.95, max(0.55, normalized_scores[best_style.value] * 2.2)) if best_raw > 2.0 else 0.20
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


# Alias for whole-world analysis
analyze_world_design_style = analyze_scene_style
