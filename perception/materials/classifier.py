"""Surface material and texture classification (Master Development Plan Task 6.10).

Classifies surface materials into canonical categories:
- wood
- glass
- metal
- concrete
- plaster (or paint)
- brick
- tile
- stone
- fabric

Evidence-grounded: combines CIELAB / LCH color science, measured texture roughness
(spatial frequency and gradient energy across texels), and specular sheen
(directional reflection highlight spread) with architectural semantic context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.materials.surface_appearance import linear_to_lab, srgb_to_linear

STATE = "ESTIMATED"
CONFIDENCE_CAP_COLOR_ONLY = 0.40
CONFIDENCE_CAP_WITH_TEXTURE = 0.65
UNIFORM_DELTA_E = 6.0


class MaterialCategory(str, Enum):
    """Canonical architectural and object material categories."""

    WOOD = "wood"
    GLASS = "glass"
    METAL = "metal"
    CONCRETE = "concrete"
    PLASTER = "plaster"
    BRICK = "brick"
    TILE = "tile"
    STONE = "stone"
    FABRIC = "fabric"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MaterialEvidence:
    """Measured optical and texture properties of a surface."""

    color_rgb: Sequence[float]
    spread_delta_e: Optional[float] = None
    texture_roughness: Optional[float] = None  # 0.0 (smooth) to 1.0 (rough/textured)
    specular_sheen: Optional[float] = None     # 0.0 (matte) to 1.0 (specular/metallic)
    surface_kind: str = "surface"              # wall, floor, ceiling, window, furniture, opening, etc.
    is_translucent: bool = False
    color_state: str = "measured_approximate"


@dataclass(frozen=True)
class MaterialClassification:
    """A ranked material hypothesis derived from optical and geometric evidence."""

    material: str
    category: MaterialCategory
    score: float
    confidence: float
    basis: str
    evidence: Dict[str, Any]
    state: str = STATE
    asserted: bool = False
    note: str = "hypothesis from measured color and texture statistics; not verified by physical contact"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material,
            "category": self.category.value,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "basis": self.basis,
            "evidence": self.evidence,
            "state": self.state,
            "asserted": self.asserted,
            "note": self.note,
        }


def _lch(rgb: Sequence[float]) -> Tuple[float, float, float]:
    """Convert sRGB [0..255] to CIELAB L*, C*, h (hue in degrees [0..360))."""
    lab = linear_to_lab(srgb_to_linear(np.asarray(rgb, dtype=np.float64)))
    L = float(lab[0])
    C = float(math.hypot(lab[1], lab[2]))
    h = float(math.degrees(math.atan2(lab[2], lab[1])) % 360.0)
    return L, C, h


def extract_texture_statistics(texels_rgb: np.ndarray) -> Dict[str, float]:
    """Extract texture roughness and specular sheen from a 2D RGB or RGBA texel patch.

    - texture_roughness: spatial gradient energy normalized to [0..1]
    - specular_sheen: highlight kurtosis/skewness normalized to [0..1]
    """
    arr = np.asarray(texels_rgb, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Convert to luminance Y
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    elif arr.ndim == 2:
        lum = arr
    else:
        return {"texture_roughness": 0.0, "specular_sheen": 0.0}

    h, w = lum.shape[:2]
    if h < 3 or w < 3:
        return {"texture_roughness": 0.0, "specular_sheen": 0.0}

    # Finite difference gradients
    dy = np.abs(lum[1:, :] - lum[:-1, :])
    dx = np.abs(lum[:, 1:] - lum[:, :-1])

    grad_mag = np.mean(dx) + np.mean(dy)
    # Normalize by dynamic range
    val_range = max(1.0, float(np.ptp(lum)))
    norm_roughness = float(np.clip((grad_mag / val_range) * 3.0, 0.0, 1.0))

    # Specular sheen from upper percentile brightness concentration
    p99 = float(np.percentile(lum, 99))
    p50 = float(np.percentile(lum, 50))
    sheen = float(np.clip((p99 - p50) / max(10.0, val_range), 0.0, 1.0))

    return {
        "texture_roughness": round(norm_roughness, 3),
        "specular_sheen": round(sheen, 3),
    }


def classify_material(evidence: MaterialEvidence) -> List[MaterialClassification]:
    """Classify material candidates ranked by evidentiary support."""
    if evidence.color_rgb is None or evidence.color_state == "not_measured":
        return []

    L, C, h = _lch(evidence.color_rgb)
    uniform = evidence.spread_delta_e is not None and evidence.spread_delta_e <= UNIFORM_DELTA_E
    roughness = evidence.texture_roughness if evidence.texture_roughness is not None else (0.15 if uniform else 0.40)
    sheen = evidence.specular_sheen if evidence.specular_sheen is not None else 0.15
    kind = str(evidence.surface_kind).lower()

    ev_dict = {
        "color_rgb": [int(round(v)) for v in evidence.color_rgb],
        "L": round(L, 1),
        "chroma": round(C, 1),
        "hue_deg": round(h, 0),
        "spread_delta_e": None if evidence.spread_delta_e is None else round(float(evidence.spread_delta_e), 2),
        "texture_roughness": round(roughness, 3),
        "specular_sheen": round(sheen, 3),
        "surface_kind": kind,
    }

    candidates: List[Tuple[str, MaterialCategory, float, str]] = []

    # 1. Glass: translucent, high sheen, or window opening
    is_window_or_glass = (
        evidence.is_translucent
        or "window" in kind
        or "glazing" in kind
        or (sheen >= 0.50 and C <= 15.0 and roughness <= 0.10)
    )
    if is_window_or_glass:
        glass_score = 0.50 + (0.35 if evidence.is_translucent else 0.0) + (0.20 if "window" in kind else 0.0) + (0.15 if sheen >= 0.4 else 0.0)
        candidates.append((
            "glass",
            MaterialCategory.GLASS,
            min(1.0, glass_score),
            "translucent or high specular sheen on low-chroma surface",
        ))

    # 2. Wood: warm amber/brown hue, moderate chroma, medium roughness
    is_warm_brown = 15.0 <= h <= 85.0 and 10.0 <= C <= 55.0 and 20.0 <= L <= 78.0
    wood_score = 0.20
    if is_warm_brown:
        wood_score += 0.45
    if 0.10 <= roughness <= 0.55:
        wood_score += 0.15
    if kind in ("floor", "door", "furniture", "table", "chair", "desk", "wardrobe"):
        wood_score += 0.15
    if wood_score >= 0.40:
        candidates.append((
            "wood_floor" if kind == "floor" else "wood",
            MaterialCategory.WOOD,
            min(1.0, wood_score),
            "warm brown hue range with moderate grain texture",
        ))

    # 3. Plaster / Paint: light, uniform, low roughness, low sheen
    is_light_diffuse = L >= 55.0 and C <= 32.0 and sheen <= 0.30
    plaster_score = 0.25
    if is_light_diffuse:
        plaster_score += 0.35
    if uniform:
        plaster_score += 0.20
    if roughness <= 0.20:
        plaster_score += 0.10
    if kind in ("wall", "ceiling"):
        plaster_score += 0.15
        candidates.append((
            "painted_wall" if kind == "wall" else "painted_ceiling",
            MaterialCategory.PLASTER,
            min(1.0, plaster_score),
            "flat uniform finish with diffuse reflectance",
        ))
    elif plaster_score >= 0.45:
        candidates.append((
            "plaster",
            MaterialCategory.PLASTER,
            min(1.0, plaster_score),
            "diffuse uniform matte finish",
        ))

    # 4. Brick: terracotta/reddish hue, high texture roughness
    is_brick_hue = (10.0 <= h <= 45.0 or 340.0 <= h <= 360.0) and 15.0 <= C <= 60.0 and 25.0 <= L <= 65.0
    brick_score = 0.15
    if is_brick_hue:
        brick_score += 0.45
    if roughness >= 0.30:
        brick_score += 0.25
    if kind in ("wall", "exterior"):
        brick_score += 0.10
    if brick_score >= 0.45:
        candidates.append((
            "brick",
            MaterialCategory.BRICK,
            min(1.0, brick_score),
            "terracotta/reddish hue with pronounced surface roughness",
        ))

    # 5. Concrete: neutral grey, matte, medium lightness
    is_neutral_grey = C <= 14.0 and 25.0 <= L <= 80.0
    concrete_score = 0.20
    if is_neutral_grey:
        concrete_score += 0.35
    if sheen <= 0.25:
        concrete_score += 0.15
    if 0.05 <= roughness <= 0.45:
        concrete_score += 0.10
    if kind in ("floor", "road", "wall", "column", "beam"):
        concrete_score += 0.15
    if concrete_score >= 0.40:
        mat_name = "concrete_or_tile_floor" if kind == "floor" else "concrete"
        candidates.append((
            mat_name,
            MaterialCategory.CONCRETE,
            min(1.0, concrete_score),
            "neutral grey chroma with matte mineral appearance",
        ))

    # 6. Tile: smooth glaze with moderate/high sheen or bathroom/kitchen floor
    tile_score = 0.20
    if sheen >= 0.30:
        tile_score += 0.30
    if uniform:
        tile_score += 0.15
    if kind in ("floor", "bathroom", "kitchen"):
        tile_score += 0.20
    if tile_score >= 0.45:
        candidates.append((
            "tile",
            MaterialCategory.TILE,
            min(1.0, tile_score),
            "specular glaze sheen with smooth surface finish",
        ))

    # 7. Metal: neutral low chroma, high sheen
    is_metallic = C <= 15.0 and sheen >= 0.45
    if is_metallic or "metallic" in kind or "conduit" in kind or "frame" in kind:
        metal_score = 0.30 + (0.40 if sheen >= 0.45 else 0.15) + (0.20 if C <= 12.0 else 0.0)
        candidates.append((
            "metal",
            MaterialCategory.METAL,
            min(1.0, metal_score),
            "high specular sheen with neutral metallic reflectance",
        ))

    # 8. Stone: neutral/earthy tones with notable roughness
    is_stone_color = (C <= 25.0 and 25.0 <= L <= 82.0)
    if is_stone_color and roughness >= 0.25:
        stone_score = 0.30 + 0.25 * (roughness) + (0.15 if sheen <= 0.20 else 0.0)
        candidates.append((
            "stone",
            MaterialCategory.STONE,
            min(1.0, stone_score),
            "mineral texture roughness with earthy neutral tone",
        ))

    # 9. Fabric: diffuse matte on furnishings
    is_furnishing = kind in ("sofa", "chair", "bed", "curtain", "rug", "cushion", "furniture")
    if is_furnishing and sheen <= 0.15:
        fabric_score = 0.40 + 0.30 + (0.15 if roughness <= 0.40 else 0.0)
        candidates.append((
            "fabric",
            MaterialCategory.FABRIC,
            min(1.0, fabric_score),
            "diffuse matte textile texture on furnishing",
        ))

    # Sort descending by score
    candidates.sort(key=lambda c: -c[2])

    has_texture_data = evidence.texture_roughness is not None or evidence.specular_sheen is not None
    cap = CONFIDENCE_CAP_WITH_TEXTURE if has_texture_data else CONFIDENCE_CAP_COLOR_ONLY

    out: List[MaterialClassification] = []
    seen = set()
    for name, cat, score, basis in candidates:
        if name in seen:
            continue
        seen.add(name)
        conf = min(cap, score * 0.55 if has_texture_data else score * 0.50)
        out.append(
            MaterialClassification(
                material=name,
                category=cat,
                score=score,
                confidence=conf,
                basis=basis,
                evidence=ev_dict,
            )
        )

    return out
