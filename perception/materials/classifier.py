"""Surface material and texture classification with PBR parameter estimation (P1, Epic E06-1).

Classifies surface materials across 12 canonical architectural categories:
1. WOOD
2. GLASS
3. METAL
4. CONCRETE
5. PLASTER (or paint)
6. BRICK
7. TILE
8. STONE
9. FABRIC
10. PLASTIC
11. LEATHER
12. ASPHALT

Evidence-grounded: combines CIELAB / LCH color science, measured texture roughness
(spatial frequency and gradient energy across texels), and specular sheen
(directional reflection highlight spread) with architectural semantic context.
Predicts physically-based rendering (PBR) parameters: albedo, roughness, metallic,
normal_scale, IOR, and transmission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.materials.surface_appearance import linear_to_lab, srgb_to_linear
from world_ir.schema_v1 import Entity, Material
from world_ir.world_v1 import WorldIR

STATE = "ESTIMATED"
CONFIDENCE_CAP_COLOR_ONLY = 0.50
CONFIDENCE_CAP_WITH_TEXTURE = 0.85
UNIFORM_DELTA_E = 6.0


class MaterialCategory(str, Enum):
    """12 Canonical architectural and object material categories."""

    WOOD = "wood"
    GLASS = "glass"
    METAL = "metal"
    CONCRETE = "concrete"
    PLASTER = "plaster"
    BRICK = "brick"
    TILE = "tile"
    STONE = "stone"
    FABRIC = "fabric"
    PLASTIC = "plastic"
    LEATHER = "leather"
    ASPHALT = "asphalt"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PBRParameters:
    """Physically-Based Rendering (PBR) surface parameters."""

    category: MaterialCategory
    material_name: str
    albedo_rgb: Tuple[float, float, float]
    roughness: float
    metallic: float
    normal_scale: float = 1.0
    ior: float = 1.50
    transmission: float = 0.0
    emissive_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "material_name": self.material_name,
            "albedo_rgb": [round(c, 4) for c in self.albedo_rgb],
            "roughness": round(self.roughness, 3),
            "metallic": round(self.metallic, 3),
            "normal_scale": round(self.normal_scale, 3),
            "ior": round(self.ior, 3),
            "transmission": round(self.transmission, 3),
            "emissive_rgb": [round(c, 4) for c in self.emissive_rgb],
        }


# Canonical baseline PBR physical properties across 12 architectural materials
_CANONICAL_PBR_BASELINES: Dict[MaterialCategory, Dict[str, float]] = {
    MaterialCategory.WOOD: {"roughness": 0.55, "metallic": 0.0, "normal_scale": 0.60, "ior": 1.50, "transmission": 0.0},
    MaterialCategory.GLASS: {"roughness": 0.05, "metallic": 0.0, "normal_scale": 0.05, "ior": 1.52, "transmission": 0.95},
    MaterialCategory.METAL: {"roughness": 0.20, "metallic": 0.95, "normal_scale": 0.20, "ior": 2.50, "transmission": 0.0},
    MaterialCategory.CONCRETE: {"roughness": 0.85, "metallic": 0.0, "normal_scale": 0.80, "ior": 1.54, "transmission": 0.0},
    MaterialCategory.PLASTER: {"roughness": 0.75, "metallic": 0.0, "normal_scale": 0.30, "ior": 1.45, "transmission": 0.0},
    MaterialCategory.BRICK: {"roughness": 0.90, "metallic": 0.0, "normal_scale": 1.00, "ior": 1.55, "transmission": 0.0},
    MaterialCategory.TILE: {"roughness": 0.20, "metallic": 0.0, "normal_scale": 0.25, "ior": 1.53, "transmission": 0.0},
    MaterialCategory.STONE: {"roughness": 0.80, "metallic": 0.0, "normal_scale": 0.90, "ior": 1.54, "transmission": 0.0},
    MaterialCategory.FABRIC: {"roughness": 0.92, "metallic": 0.0, "normal_scale": 0.70, "ior": 1.48, "transmission": 0.0},
    MaterialCategory.PLASTIC: {"roughness": 0.35, "metallic": 0.0, "normal_scale": 0.15, "ior": 1.46, "transmission": 0.0},
    MaterialCategory.LEATHER: {"roughness": 0.60, "metallic": 0.0, "normal_scale": 0.50, "ior": 1.48, "transmission": 0.0},
    MaterialCategory.ASPHALT: {"roughness": 0.95, "metallic": 0.0, "normal_scale": 0.95, "ior": 1.55, "transmission": 0.0},
    MaterialCategory.UNKNOWN: {"roughness": 0.70, "metallic": 0.0, "normal_scale": 0.50, "ior": 1.50, "transmission": 0.0},
}


@dataclass(frozen=True)
class MaterialEvidence:
    """Measured optical and texture properties of a surface."""

    color_rgb: Sequence[float]
    spread_delta_e: Optional[float] = None
    texture_roughness: Optional[float] = None  # 0.0 (mirror/smooth) to 1.0 (rough/textured)
    specular_sheen: Optional[float] = None     # 0.0 (matte) to 1.0 (specular/metallic)
    surface_kind: str = "surface"              # wall, floor, ceiling, window, furniture, road, etc.
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
    pbr_parameters: Optional[PBRParameters] = None
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
            "pbr_parameters": self.pbr_parameters.to_dict() if self.pbr_parameters else None,
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
    - specular_sheen: highlight concentration normalized to [0..1]
    """
    arr = np.asarray(texels_rgb, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    elif arr.ndim == 2:
        lum = arr
    else:
        return {"texture_roughness": 0.0, "specular_sheen": 0.0}

    h, w = lum.shape[:2]
    if h < 3 or w < 3:
        return {"texture_roughness": 0.0, "specular_sheen": 0.0}

    dy = np.abs(lum[1:, :] - lum[:-1, :])
    dx = np.abs(lum[:, 1:] - lum[:, :-1])

    grad_mag = np.mean(dx) + np.mean(dy)
    val_range = max(1.0, float(np.ptp(lum)))
    norm_roughness = float(np.clip((grad_mag / val_range) * 3.0, 0.0, 1.0))

    p99 = float(np.percentile(lum, 99))
    p50 = float(np.percentile(lum, 50))
    sheen = float(np.clip((p99 - p50) / max(10.0, val_range), 0.0, 1.0))

    return {
        "texture_roughness": round(norm_roughness, 3),
        "specular_sheen": round(sheen, 3),
    }


def compute_pbr_parameters(
    category: MaterialCategory,
    material_name: str,
    evidence: MaterialEvidence,
) -> PBRParameters:
    """Predict calibrated PBR parameters by fusing canonical baselines with measured evidence."""
    base = _CANONICAL_PBR_BASELINES.get(category, _CANONICAL_PBR_BASELINES[MaterialCategory.UNKNOWN])

    # Linear albedo in [0..1]
    raw_rgb = evidence.color_rgb[:3]
    if any(c > 1.0 for c in raw_rgb):
        albedo = tuple(float(c / 255.0) for c in raw_rgb)
    else:
        albedo = tuple(float(c) for c in raw_rgb)

    # Refine roughness based on measured texture roughness and sheen
    roughness = base["roughness"]
    if evidence.texture_roughness is not None:
        roughness = float(np.clip(0.4 * roughness + 0.6 * evidence.texture_roughness, 0.01, 0.98))
    if evidence.specular_sheen is not None and evidence.specular_sheen > 0.40 and category != MaterialCategory.ASPHALT:
        roughness = float(np.clip(roughness * (1.0 - 0.35 * evidence.specular_sheen), 0.01, 0.98))

    # Metallic: strictly dielectric (0.0) except for METAL
    metallic = base["metallic"]
    if category == MaterialCategory.METAL:
        if evidence.specular_sheen is not None:
            metallic = float(np.clip(0.80 + 0.20 * evidence.specular_sheen, 0.70, 1.0))
        else:
            metallic = 0.95
    else:
        metallic = 0.0

    # Normal scale
    normal_scale = base["normal_scale"]
    if evidence.texture_roughness is not None and category != MaterialCategory.GLASS:
        normal_scale = float(np.clip(normal_scale * (0.6 + 0.8 * evidence.texture_roughness), 0.05, 1.8))

    return PBRParameters(
        category=category,
        material_name=material_name,
        albedo_rgb=albedo,
        roughness=roughness,
        metallic=metallic,
        normal_scale=normal_scale,
        ior=base["ior"],
        transmission=base["transmission"],
    )


def classify_material(evidence: MaterialEvidence) -> List[MaterialClassification]:
    """Classify material candidates across 12 canonical architectural categories."""
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

    # 1. Glass
    is_window_or_glass = (
        evidence.is_translucent
        or "window" in kind
        or "glazing" in kind
        or (sheen >= 0.65 and C <= 15.0 and roughness <= 0.05 and kind not in ("metal", "conduit", "fixture"))
    )
    if is_window_or_glass:
        glass_score = 0.60 + (0.35 if evidence.is_translucent else 0.0) + (0.30 if "window" in kind or "glazing" in kind else 0.0) + (0.10 if sheen >= 0.7 else 0.0)
        candidates.append((
            "glass",
            MaterialCategory.GLASS,
            min(1.0, glass_score),
            "translucent or high specular sheen on low-chroma surface",
        ))

    # 2. Wood
    is_warm_brown = 15.0 <= h <= 85.0 and 10.0 <= C <= 55.0 and 20.0 <= L <= 78.0
    is_wood_kind = kind in ("floor", "door", "table", "desk", "wardrobe", "shelf", "furniture")
    if is_warm_brown and sheen <= 0.24 and is_wood_kind:
        wood_score = 0.50 + 0.30 + (0.15 if 0.15 <= roughness <= 0.55 else 0.0)
        candidates.append((
            "wood_floor" if kind == "floor" else "wood",
            MaterialCategory.WOOD,
            min(1.0, wood_score),
            "warm brown hue range with moderate grain texture",
        ))
    elif is_warm_brown and sheen <= 0.22 and not is_wood_kind:
        wood_score = 0.40 + 0.20
        candidates.append((
            "wood",
            MaterialCategory.WOOD,
            min(1.0, wood_score),
            "warm brown hue range",
        ))

    # 3. Plaster / Paint
    is_light_diffuse = L >= 55.0 and C <= 32.0 and sheen <= 0.25
    if is_light_diffuse and kind in ("wall", "ceiling"):
        plaster_score = 0.55 + (0.25 if uniform else 0.10) + (0.15 if roughness <= 0.20 else 0.0)
        candidates.append((
            "painted_wall" if kind == "wall" else "painted_ceiling",
            MaterialCategory.PLASTER,
            min(1.0, plaster_score),
            "flat uniform finish with diffuse reflectance",
        ))
    elif is_light_diffuse and roughness <= 0.25:
        plaster_score = 0.45 + (0.20 if uniform else 0.0)
        candidates.append((
            "plaster",
            MaterialCategory.PLASTER,
            min(1.0, plaster_score),
            "diffuse uniform matte finish",
        ))

    # 4. Brick
    is_brick_hue = (10.0 <= h <= 45.0 or 340.0 <= h <= 360.0) and 15.0 <= C <= 60.0 and 25.0 <= L <= 65.0
    if is_brick_hue and roughness >= 0.40:
        brick_score = 0.50 + (0.30 if kind in ("wall", "facade", "exterior") else 0.15) + (0.15 if roughness >= 0.55 else 0.0)
        candidates.append((
            "brick",
            MaterialCategory.BRICK,
            min(1.0, brick_score),
            "terracotta/reddish hue with pronounced surface roughness",
        ))

    # 5. Concrete
    is_neutral_grey = C <= 14.0 and 30.0 <= L <= 82.0
    is_conc_kind = kind in ("column", "beam", "foundation", "structure", "concrete")
    if is_neutral_grey and is_conc_kind and sheen <= 0.25:
        concrete_score = 0.60 + 0.30 + (0.10 if roughness >= 0.30 else 0.0)
        candidates.append((
            "concrete",
            MaterialCategory.CONCRETE,
            min(1.0, concrete_score),
            "structural architectural concrete element",
        ))
    elif is_neutral_grey and sheen <= 0.20 and 0.25 <= roughness <= 0.55 and kind not in ("road", "street", "curb", "sidewalk", "pavement"):
        concrete_score = 0.45 + 0.20
        candidates.append((
            "concrete",
            MaterialCategory.CONCRETE,
            min(1.0, concrete_score),
            "neutral grey chroma with matte mineral appearance",
        ))

    # 6. Tile
    is_tile_context = kind in ("floor", "bathroom", "kitchen")
    if is_tile_context and sheen >= 0.35 and roughness <= 0.25 and not evidence.is_translucent:
        tile_score = 0.60 + 0.30 + (0.10 if sheen >= 0.50 else 0.0)
        candidates.append((
            "tile",
            MaterialCategory.TILE,
            min(1.0, tile_score),
            "specular glaze sheen with smooth surface finish",
        ))

    # 7. Metal
    is_metal_kind = (
        kind in ("metal", "metallic")
        or (kind in ("conduit", "frame", "fixture", "pipe", "hardware") and sheen >= 0.50)
    )
    if is_metal_kind or (sheen >= 0.65 and C <= 15.0 and not evidence.is_translucent and kind not in ("window", "glazing", "bathroom", "kitchen", "floor", "wall", "ceiling", "sofa", "bed")):
        metal_score = 0.60 + (0.35 if is_metal_kind else 0.20) + (0.10 if sheen >= 0.75 else 0.0)
        candidates.append((
            "metal",
            MaterialCategory.METAL,
            min(1.0, metal_score),
            "high specular sheen with neutral metallic reflectance",
        ))

    # 8. Stone
    is_stone_color = (C <= 25.0 and 25.0 <= L <= 82.0)
    is_stone_kind = (
        kind in ("stone", "exterior", "surface", "facade", "wall")
        and kind not in ("column", "beam", "foundation", "road", "street", "curb")
    )
    if is_stone_color and roughness >= 0.35 and is_stone_kind:
        stone_score = 0.50 + 0.35 * roughness + (0.10 if roughness >= 0.55 else 0.0)
        candidates.append((
            "stone",
            MaterialCategory.STONE,
            min(1.0, stone_score),
            "mineral texture roughness with earthy neutral tone",
        ))

    # 9. Fabric
    is_furnishing = kind in ("sofa", "chair", "bed", "curtain", "rug", "cushion", "furniture", "fabric")
    if is_furnishing and sheen <= 0.15:
        fabric_score = 0.60 + 0.30 + (0.10 if roughness <= 0.50 else 0.0)
        candidates.append((
            "fabric",
            MaterialCategory.FABRIC,
            min(1.0, fabric_score),
            "diffuse matte textile texture on furnishing",
        ))

    # 10. Plastic / Polymer
    is_plastic_context = kind in ("switch", "socket", "outlet", "switchboard", "conduit", "appliance", "trim", "casing")
    if is_plastic_context and roughness <= 0.25 and sheen >= 0.25:
        plastic_score = 0.60 + 0.35 + (0.05 if uniform else 0.0)
        candidates.append((
            "plastic",
            MaterialCategory.PLASTIC,
            min(1.0, plastic_score),
            "uniform synthetic polymer sheen with low micro-roughness",
        ))

    # 11. Leather
    is_leather_context = kind in ("sofa", "chair", "couch", "furniture", "upholstery")
    is_leather_tone = (15.0 <= h <= 65.0 or L <= 40.0) and (C <= 45.0)
    if is_leather_context and is_leather_tone and sheen >= 0.25 and roughness >= 0.15:
        leather_score = 0.65 + 0.30 + (0.05 if 0.20 <= roughness <= 0.40 else 0.0)
        candidates.append((
            "leather",
            MaterialCategory.LEATHER,
            min(1.0, leather_score),
            "moderate specular sheen with fine grain organic texture",
        ))

    # 12. Asphalt / Bitumen
    is_asphalt_context = kind in ("road", "street", "curb", "sidewalk", "pavement", "exterior", "asphalt")
    is_dark_matte = L <= 40.0 and C <= 10.0 and sheen <= 0.15 and roughness >= 0.50
    if is_asphalt_context and is_dark_matte:
        asphalt_score = 0.65 + 0.30 + (0.05 if roughness >= 0.70 else 0.0)
        candidates.append((
            "asphalt",
            MaterialCategory.ASPHALT,
            min(1.0, asphalt_score),
            "dark matte granular aggregate with high surface roughness",
        ))

    # Sort candidates descending by score
    candidates.sort(key=lambda c: -c[2])

    has_texture_data = evidence.texture_roughness is not None or evidence.specular_sheen is not None
    cap = CONFIDENCE_CAP_WITH_TEXTURE if has_texture_data else CONFIDENCE_CAP_COLOR_ONLY

    out: List[MaterialClassification] = []
    seen = set()
    for name, cat, score, basis in candidates:
        if name in seen:
            continue
        seen.add(name)
        conf = min(cap, score * 0.90 if has_texture_data else score * 0.50)
        pbr = compute_pbr_parameters(cat, name, evidence)
        out.append(
            MaterialClassification(
                material=name,
                category=cat,
                score=score,
                confidence=conf,
                basis=basis,
                evidence=ev_dict,
                pbr_parameters=pbr,
            )
        )

    return out


def assign_pbr_material(
    world: WorldIR,
    entity: Entity,
    evidence: MaterialEvidence,
) -> Material:
    """Classify and assign a rich PBR material to an entity in WorldIR."""
    classifications = classify_material(evidence)
    if classifications:
        top_mat = classifications[0]
        pbr = top_mat.pbr_parameters or compute_pbr_parameters(top_mat.category, top_mat.material, evidence)
        cat_name = top_mat.category.value
    else:
        pbr = compute_pbr_parameters(MaterialCategory.UNKNOWN, "default_material", evidence)
        cat_name = "unknown"

    mat_id = f"mat-{entity.id}"
    mat_props = {
        "category": cat_name,
        "roughness": pbr.roughness,
        "metallic": pbr.metallic,
        "normal_scale": pbr.normal_scale,
        "ior": pbr.ior,
        "transmission": pbr.transmission,
        "albedo_rgb": list(pbr.albedo_rgb),
    }

    material = Material(
        id=mat_id,
        name=f"{cat_name}_{entity.name or entity.id}",
        class_name=cat_name,
        color_rgb=list(pbr.albedo_rgb),
        properties=mat_props,
    )

    world.materials[mat_id] = material
    entity.material_ids = [mat_id]

    if entity.custom_properties is None:
        entity.custom_properties = {}
    entity.custom_properties["material_id"] = mat_id
    entity.custom_properties["material"] = mat_id
    entity.custom_properties["pbr_material"] = pbr.to_dict()

    return material
