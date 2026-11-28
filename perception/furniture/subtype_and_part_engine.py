"""Team D — Furniture Subtype and Part Engine.

Fine-grained subtype classification and component hierarchy builder for
furniture and indoor objects.

Capabilities:
  - Fine-grained subtype identification (e.g., "office chair" vs "dining chair")
  - Component hierarchy with confidence-weighted assembly
  - Material hypothesis aggregation
  - Attribute extraction (color, texture, finish, style)
  - Subpart annotation from bounding-box geometry
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("reality.furniture.subtype_and_part_engine")


# ---------------------------------------------------------------------------
# Fine-grained subtype taxonomy
# ---------------------------------------------------------------------------
_SUBTYPES: dict[str, list[str]] = {
    "chair": [
        "dining chair", "office chair", "armchair", "accent chair",
        "rocking chair", "folding chair", "bar stool", "counter stool",
        "lounge chair", "task chair", "executive chair",
    ],
    "sofa": [
        "2-seater sofa", "3-seater sofa", "l-shaped sofa", "sectional sofa",
        "sleeper sofa", "chesterfield sofa", "mid-century sofa",
        "loveseat", "settee", "futon",
    ],
    "table": [
        "dining table", "coffee table", "side table", "end table",
        "console table", "kitchen table", "conference table",
        "folding table", "picnic table", "pedestal table",
    ],
    "desk": [
        "writing desk", "computer desk", "standing desk", "l-shaped desk",
        "secretary desk", "reception desk", "drafting table",
    ],
    "bed": [
        "twin bed", "full bed", "queen bed", "king bed",
        "bunk bed", "platform bed", "sleigh bed", "canopy bed",
        "adjustable bed", "cot", "daybed",
    ],
    "cabinet": [
        "kitchen cabinet", "bathroom cabinet", "medicine cabinet",
        "tv cabinet", "storage cabinet", "filing cabinet",
        "china cabinet", "curio cabinet", "liquor cabinet",
    ],
    "wardrobe": [
        "single wardrobe", "double wardrobe", "sliding door wardrobe",
        "walk-in wardrobe", "armoire", "chifforobe",
    ],
    "bookshelf": [
        "open bookshelf", "closed bookcase", "floating shelf",
        "ladder shelf", "barrister bookcase", "built-in bookshelf",
    ],
    "lamp": [
        "floor lamp", "table lamp", "desk lamp", "arc lamp",
        "torchiere", "clip-on lamp", "pendant light",
    ],
    "mirror": [
        "full-length mirror", "dresser mirror", "bathroom mirror",
        "vanity mirror", "frameless mirror", "framed mirror",
        "leaning mirror", "wall mirror",
    ],
}

# ---------------------------------------------------------------------------
# Material hypotheses
# ---------------------------------------------------------------------------
_MATERIAL_CUES: dict[str, list[str]] = {
    "wood": ["wooden", "oak", "walnut", "pine", "mahogany", "teak", "grain"],
    "metal": ["metallic", "steel", "aluminum", "iron", "chrome", "brass"],
    "fabric": ["upholstered", "fabric", "textile", "velvet", "linen", "cotton", "chenille"],
    "leather": ["leather", "faux leather", "pleather", "bonded leather"],
    "glass": ["glass", "tempered glass", "frosted glass", "transparent", "clear"],
    "plastic": ["plastic", "resin", "acrylic", "polycarbonate"],
    "marble": ["marble", "stone", "granite", "quartz", "travertine"],
    "rattan": ["rattan", "wicker", "cane", "bamboo"],
    "concrete": ["concrete", "cement", "industrial"],
}

_STYLE_CUES: dict[str, list[str]] = {
    "modern": ["modern", "contemporary", "minimalist", "sleek", "clean lines"],
    "traditional": ["traditional", "classic", "antique", "vintage", "ornate"],
    "mid-century": ["mid-century", "retro", "50s", "60s", "tapered leg"],
    "industrial": ["industrial", "raw", "exposed", "rustic metal"],
    "scandinavian": ["scandinavian", "nordic", "danish", "swedish", "simple"],
    "bohemian": ["bohemian", "boho", "eclectic", "colorful", "pattern"],
}


@dataclass(frozen=True)
class MaterialHypothesisRecord:
    """A material hypothesis for a detected furniture item."""
    material: str
    confidence: float    # 0–1
    source: str          # "label_text" | "visual_cue" | "geometric_heuristic"
    cue_tokens: tuple[str, ...]


@dataclass(frozen=True)
class StyleHypothesisRecord:
    """A design-style hypothesis for a detected furniture item."""
    style: str
    confidence: float
    source: str
    cue_tokens: tuple[str, ...]


@dataclass
class FurnitureSubtypeResult:
    """Result of subtype classification for one furniture detection."""
    observation_id: str
    canonical_label: str
    subtypes: list[tuple[str, float]]       # [(subtype_name, confidence)] desc
    material_hypotheses: list[MaterialHypothesisRecord]
    style_hypotheses: list[StyleHypothesisRecord]
    component_hypotheses: list["ComponentHypothesis"]
    notes: list[str]


@dataclass(frozen=True)
class ComponentHypothesis:
    """A predicted structural component of a furniture item."""
    component_id: str
    parent_observation_id: str
    component_label: str
    bbox_px: tuple[float, float, float, float]
    relative_position: str    # "top" | "center" | "bottom" | "left" | "right"
    confidence: float
    is_structural: bool       # True for legs, frame; False for cushion, handle


# ---------------------------------------------------------------------------
# Component geometries per object type
# ---------------------------------------------------------------------------
_COMPONENT_SPECS: dict[str, list[tuple[str, tuple[float, float, float, float], bool]]] = {
    # (label, (rx1, ry1, rx2, ry2) relative to bbox, is_structural)
    "chair": [
        ("backrest",  (0.1, 0.0, 0.9, 0.5),  True),
        ("seat",      (0.1, 0.5, 0.9, 0.75), True),
        ("armrest",   (0.0, 0.3, 0.2, 0.7),  True),
        ("leg",       (0.1, 0.75, 0.9, 1.0), True),
        ("cushion",   (0.1, 0.45, 0.9, 0.7), False),
    ],
    "sofa": [
        ("backrest",  (0.0, 0.0, 1.0, 0.5),  True),
        ("seat",      (0.0, 0.55, 1.0, 0.85), True),
        ("armrest",   (0.0, 0.1, 0.12, 0.9), True),
        ("cushion",   (0.1, 0.5, 0.9, 0.85), False),
        ("leg",       (0.1, 0.85, 0.9, 1.0), True),
    ],
    "table": [
        ("top",       (0.0, 0.0, 1.0, 0.2),  True),
        ("leg",       (0.05, 0.2, 0.3, 1.0), True),
        ("shelf",     (0.1, 0.5, 0.9, 0.75), False),
        ("drawer",    (0.1, 0.1, 0.9, 0.3),  False),
    ],
    "desk": [
        ("top",       (0.0, 0.0, 1.0, 0.2),  True),
        ("leg",       (0.05, 0.2, 0.25, 1.0), True),
        ("drawer",    (0.3, 0.2, 0.9, 0.5),  False),
        ("keyboard_tray", (0.1, 0.15, 0.9, 0.3), False),
    ],
    "bed": [
        ("headboard", (0.0, 0.0, 1.0, 0.3),  True),
        ("footboard", (0.0, 0.8, 1.0, 1.0),  True),
        ("mattress",  (0.0, 0.3, 1.0, 0.8),  True),
        ("frame",     (0.0, 0.7, 1.0, 1.0),  True),
        ("leg",       (0.05, 0.85, 0.2, 1.0), True),
    ],
    "cabinet": [
        ("door",      (0.0, 0.0, 1.0, 0.95), True),
        ("handle",    (0.75, 0.2, 0.95, 0.8), False),
        ("shelf",     (0.05, 0.3, 0.95, 0.7), False),
        ("body",      (0.0, 0.0, 1.0, 1.0),  True),
    ],
    "wardrobe": [
        ("door",      (0.0, 0.0, 1.0, 0.95), True),
        ("handle",    (0.45, 0.2, 0.55, 0.8), False),
        ("rod",       (0.1, 0.1, 0.9, 0.3),  False),
        ("body",      (0.0, 0.0, 1.0, 1.0),  True),
    ],
}


def _infer_subtypes(
    canonical_label: str,
    bbox_px: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> list[tuple[str, float]]:
    """Infer furniture subtypes from geometric cues."""
    x1, y1, x2, y2 = bbox_px
    w = x2 - x1
    h = y2 - y1
    aspect = w / max(h, 1.0)
    rel_h = h / max(image_height, 1.0)
    rel_w = w / max(image_width, 1.0)
    label = canonical_label.lower()

    subtypes = _SUBTYPES.get(label, [])
    if not subtypes:
        return []

    scored: list[tuple[str, float]] = []
    for subtype in subtypes:
        score = 0.4  # base score

        # Geometric cues per label
        if label == "chair":
            if "office" in subtype and rel_w < 0.15:
                score += 0.3
            elif "dining" in subtype and rel_w < 0.20:
                score += 0.25
            elif "bar stool" in subtype and rel_h > 0.35:
                score += 0.2
            elif "armchair" in subtype and aspect > 0.8:
                score += 0.2
        elif label == "sofa":
            if "l-shaped" in subtype and aspect > 1.8:
                score += 0.3
            elif "loveseat" in subtype and rel_w < 0.3:
                score += 0.25
            elif "3-seater" in subtype and rel_w > 0.4:
                score += 0.25
        elif label == "table":
            if "coffee table" in subtype and rel_h < 0.15:
                score += 0.3
            elif "dining table" in subtype and rel_h > 0.2:
                score += 0.25
            elif "side table" in subtype and rel_w < 0.15:
                score += 0.2
        elif label == "bed":
            if "king" in subtype and rel_w > 0.5:
                score += 0.3
            elif "queen" in subtype and 0.35 < rel_w < 0.55:
                score += 0.25
            elif "twin" in subtype and rel_w < 0.35:
                score += 0.25
            elif "bunk" in subtype and rel_h > 0.4:
                score += 0.2

        scored.append((subtype, min(0.95, score)))

    return sorted(scored, key=lambda x: -x[1])[:5]


def _extract_material_hypotheses(label: str) -> list[MaterialHypothesisRecord]:
    """Heuristic material hypotheses from label tokens."""
    label_lower = label.lower()
    results: list[MaterialHypothesisRecord] = []
    for material, cues in _MATERIAL_CUES.items():
        matched = [c for c in cues if c in label_lower]
        if matched:
            confidence = min(0.90, 0.5 + 0.15 * len(matched))
            results.append(MaterialHypothesisRecord(
                material=material,
                confidence=confidence,
                source="label_text",
                cue_tokens=tuple(matched),
            ))
    # Default material heuristics per object type
    base = label_lower.split()[-1] if label_lower else ""
    if not results:
        defaults = {
            "chair": "fabric", "sofa": "fabric", "table": "wood",
            "desk": "wood", "cabinet": "wood", "wardrobe": "wood",
            "bed": "wood", "shelf": "wood", "bookshelf": "wood",
        }
        default_mat = defaults.get(base, "unknown")
        if default_mat != "unknown":
            results.append(MaterialHypothesisRecord(
                material=default_mat,
                confidence=0.35,
                source="geometric_heuristic",
                cue_tokens=(),
            ))
    return results


def _build_component_hypotheses(
    canonical_label: str,
    bbox_px: tuple[float, float, float, float],
    observation_id: str,
    image_width: int,
    image_height: int,
) -> list[ComponentHypothesis]:
    """Build component hypotheses from bbox geometry."""
    specs = _COMPONENT_SPECS.get(canonical_label.lower(), [])
    if not specs:
        return []

    x1, y1, x2, y2 = bbox_px
    pw = x2 - x1
    ph = y2 - y1
    if pw <= 0 or ph <= 0:
        return []

    result: list[ComponentHypothesis] = []
    for comp_label, (rx1, ry1, rx2, ry2), is_struct in specs:
        cx1 = max(0.0, min(float(image_width), x1 + rx1 * pw))
        cy1 = max(0.0, min(float(image_height), y1 + ry1 * ph))
        cx2 = max(0.0, min(float(image_width), x1 + rx2 * pw))
        cy2 = max(0.0, min(float(image_height), y1 + ry2 * ph))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        cy_mid = (ry1 + ry2) / 2
        pos = "top" if cy_mid < 0.33 else "bottom" if cy_mid > 0.66 else "center"
        area_ratio = (rx2 - rx1) * (ry2 - ry1)
        confidence = min(0.85, 0.35 + 0.6 * area_ratio)

        cid = hashlib.sha1(
            f"{observation_id}:{comp_label}:{cx1:.0f}:{cy1:.0f}".encode()
        ).hexdigest()[:8]
        result.append(ComponentHypothesis(
            component_id=f"comp_{cid}",
            parent_observation_id=observation_id,
            component_label=comp_label,
            bbox_px=(cx1, cy1, cx2, cy2),
            relative_position=pos,
            confidence=confidence,
            is_structural=is_struct,
        ))
    return result


def classify_furniture_subtype(
    observation_id: str,
    canonical_label: str,
    bbox_px: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    *,
    min_subtype_confidence: float = 0.35,
) -> FurnitureSubtypeResult:
    """Classify fine-grained furniture subtype and components.

    Args:
        observation_id: Parent observation identifier.
        canonical_label: Canonical ontology label (e.g., "chair", "sofa").
        bbox_px: (x1, y1, x2, y2) bounding box in pixels.
        image_width: Image width for geometric analysis.
        image_height: Image height for geometric analysis.
        min_subtype_confidence: Minimum confidence to include a subtype.

    Returns:
        ``FurnitureSubtypeResult`` with subtypes, materials, components.
    """
    subtypes = [
        (name, conf) for name, conf in _infer_subtypes(
            canonical_label, bbox_px, image_width, image_height
        )
        if conf >= min_subtype_confidence
    ]

    materials = _extract_material_hypotheses(canonical_label)

    # Style hypotheses
    styles: list[StyleHypothesisRecord] = []
    label_lower = canonical_label.lower()
    for style, cues in _STYLE_CUES.items():
        matched = [c for c in cues if c in label_lower]
        if matched:
            styles.append(StyleHypothesisRecord(
                style=style,
                confidence=min(0.80, 0.4 + 0.2 * len(matched)),
                source="label_text",
                cue_tokens=tuple(matched),
            ))

    components = _build_component_hypotheses(
        canonical_label, bbox_px, observation_id, image_width, image_height
    )

    notes: list[str] = []
    if not subtypes:
        notes.append(f"No confident subtypes for '{canonical_label}' — returning generic.")
    if not materials:
        notes.append("No material hypotheses inferred — using 'unknown'.")

    return FurnitureSubtypeResult(
        observation_id=observation_id,
        canonical_label=canonical_label,
        subtypes=subtypes,
        material_hypotheses=materials,
        style_hypotheses=styles,
        component_hypotheses=components,
        notes=notes,
    )
