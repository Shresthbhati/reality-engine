"""Visual Material, Color, Finish, and Operational State Recognition (Phase 4).

Extracts evidence-grounded material categories, PBR finishes, CIELAB dominant colors,
and object operational states from 2D image crops and masks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MaterialType(str, Enum):
    WOOD = "wood"
    GLASS = "glass"
    METAL = "metal"
    CONCRETE = "concrete"
    STONE = "stone"
    TILE = "tile"
    FABRIC = "fabric"
    PLASTIC = "plastic"
    PLASTER = "plaster"
    BRICK = "brick"
    LEATHER = "leather"
    ASPHALT = "asphalt"
    UNKNOWN = "unknown"


class SurfaceFinish(str, Enum):
    MATTE = "matte"
    GLOSSY = "glossy"
    METALLIC = "metallic"
    ROUGH = "rough"
    TRANSPARENT = "transparent"
    UNKNOWN = "unknown"


class ObjectState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    ON = "on"
    OFF = "off"
    UNSPECIFIED = "unspecified"


# Reference colors in CIELAB space: (L, a, b, name, srgb_hex)
_NAMED_COLOR_REFERENCES = (
    (95.0, 0.0, 0.0, "pure_white", "#ffffff"),
    (80.0, 0.0, 0.0, "light_gray", "#cccccc"),
    (50.0, 0.0, 0.0, "medium_gray", "#808080"),
    (20.0, 0.0, 0.0, "dark_charcoal", "#333333"),
    (5.0, 0.0, 0.0, "black", "#000000"),
    (55.0, 15.0, 35.0, "wood_brown", "#8b5a2b"),
    (35.0, 10.0, 20.0, "dark_wood", "#4a2e18"),
    (85.0, 5.0, 20.0, "beige", "#f5f5dc"),
    (45.0, 60.0, 40.0, "red", "#cc2222"),
    (50.0, -35.0, 30.0, "green", "#228822"),
    (40.0, 10.0, -50.0, "blue", "#1e40af"),
    (85.0, -10.0, 75.0, "yellow", "#eab308"),
    (75.0, 0.0, 0.0, "metallic_silver", "#b8b8b8"),
    (70.0, 10.0, 45.0, "metallic_gold", "#d4af37"),
)


def _srgb_to_cielab(rgb_norm: np.ndarray) -> np.ndarray:
    """Converts normalized sRGB array (shape (..., 3) in [0, 1]) to CIELAB (D65 illuminant)."""
    # Linearize sRGB
    mask = rgb_norm > 0.04045
    linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # sRGB to XYZ (D65)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Reference white D65: (95.047, 100.0, 108.883)
    xn, yn, zn = x / 0.95047, y / 1.00000, z / 1.08883

    epsilon = 0.008856
    kappa = 903.3

    fx = np.where(xn > epsilon, xn ** (1.0 / 3.0), (kappa * xn + 16.0) / 116.0)
    fy = np.where(yn > epsilon, yn ** (1.0 / 3.0), (kappa * yn + 16.0) / 116.0)
    fz = np.where(zn > epsilon, zn ** (1.0 / 3.0), (kappa * zn + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return np.stack([L, a, b_val], axis=-1)


@dataclass(frozen=True)
class EntityVisualAttributes:
    """Detailed visual, material, and operational attributes of an entity."""

    material: MaterialType = MaterialType.UNKNOWN
    material_confidence: float = 0.0
    surface_finish: SurfaceFinish = SurfaceFinish.UNKNOWN
    dominant_color_name: str = "unknown"
    dominant_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    operational_state: ObjectState = ObjectState.UNSPECIFIED
    state_confidence: float = 0.0
    is_reflective: bool = False
    is_transparent: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material.value,
            "material_confidence": round(self.material_confidence, 3),
            "surface_finish": self.surface_finish.value,
            "dominant_color_name": self.dominant_color_name,
            "dominant_rgb": [round(c, 4) for c in self.dominant_rgb],
            "operational_state": self.operational_state.value,
            "state_confidence": round(self.state_confidence, 3),
            "is_reflective": self.is_reflective,
            "is_transparent": self.is_transparent,
            "details": self.details,
        }


def extract_visual_attributes(
    image_rgb: np.ndarray,
    box_2d: Sequence[float],
    mask_2d: np.ndarray | None = None,
    class_hint: str = "",
) -> EntityVisualAttributes:
    """Extracts dominant color, texture, surface finish, and material from an image crop."""
    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, min(w_img - 1, int(round(box_2d[0]))))
    y1 = max(0, min(h_img - 1, int(round(box_2d[1]))))
    x2 = max(0, min(w_img, int(round(box_2d[2]))))
    y2 = max(0, min(h_img, int(round(box_2d[3]))))

    if x2 <= x1 or y2 <= y1:
        return EntityVisualAttributes()

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return EntityVisualAttributes()

    if mask_2d is not None and mask_2d.shape[:2] == (h_img, w_img):
        crop_mask = mask_2d[y1:y2, x1:x2].astype(bool)
        if np.any(crop_mask):
            pixels = crop[crop_mask]
        else:
            pixels = crop.reshape(-1, 3)
    else:
        # Central 60% crop to avoid background bleed
        ch, cw = crop.shape[:2]
        my1, my2 = int(0.2 * ch), int(0.8 * ch)
        mx1, mx2 = int(0.2 * cw), int(0.8 * cw)
        pixels = crop[my1:my2, mx1:mx2].reshape(-1, 3) if (my2 > my1 and mx2 > mx1) else crop.reshape(-1, 3)

    if len(pixels) == 0:
        return EntityVisualAttributes()

    # Normalize to [0, 1]
    pix_norm = pixels.astype(np.float64) / 255.0
    mean_rgb = np.median(pix_norm, axis=0)

    # 1. CIELAB Dominant Color
    lab = _srgb_to_cielab(mean_rgb[None, :])[0]
    best_color_name = "unknown"
    best_delta_e = float("inf")

    for ref_L, ref_a, ref_b, name, _ in _NAMED_COLOR_REFERENCES:
        # Standard Delta E (CIE 1976)
        delta_e = math.sqrt((lab[0] - ref_L) ** 2 + (lab[1] - ref_a) ** 2 + (lab[2] - ref_b) ** 2)
        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_color_name = name

    # 2. Texture Roughness & High Frequency Energy
    # Convert crop to grayscale
    gray_crop = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    if gray_crop.shape[0] >= 3 and gray_crop.shape[1] >= 3:
        gy, gx = np.gradient(gray_crop.astype(np.float64))
        grad_mag = np.hypot(gx, gy)
        roughness_energy = float(np.mean(grad_mag))
        specular_ratio = float(np.mean(gray_crop > 240))
    else:
        roughness_energy = 0.0
        specular_ratio = 0.0

    # 3. Material & Finish Inference grounded in visual evidence and class context
    c_lower = class_hint.lower()
    mat = MaterialType.UNKNOWN
    finish = SurfaceFinish.MATTE
    mat_conf = 0.35
    is_refl = False
    is_trans = False

    if "glass" in c_lower or "window" in c_lower:
        mat = MaterialType.GLASS
        finish = SurfaceFinish.TRANSPARENT
        is_trans = True
        mat_conf = 0.85
    elif any(k in c_lower for k in ("table", "chair", "bed", "wardrobe", "door")) and ("wood" in best_color_name or "brown" in best_color_name):
        mat = MaterialType.WOOD
        finish = SurfaceFinish.MATTE if roughness_energy > 8.0 else SurfaceFinish.GLOSSY
        mat_conf = 0.75
    elif any(k in c_lower for k in ("socket", "switch", "switchboard")):
        mat = MaterialType.PLASTIC
        finish = SurfaceFinish.GLOSSY if specular_ratio > 0.02 else SurfaceFinish.MATTE
        mat_conf = 0.80
    elif any(k in c_lower for k in ("faucet", "pipe", "radiator", "lamppost", "pole", "handle")):
        mat = MaterialType.METAL
        finish = SurfaceFinish.METALLIC
        is_refl = True
        mat_conf = 0.80
    elif any(k in c_lower for k in ("wall", "ceiling")):
        mat = MaterialType.PLASTER
        finish = SurfaceFinish.MATTE
        mat_conf = 0.70
    elif "floor" in c_lower:
        if "tile" in best_color_name or specular_ratio > 0.05:
            mat = MaterialType.TILE
            finish = SurfaceFinish.GLOSSY
        elif "wood" in best_color_name:
            mat = MaterialType.WOOD
            finish = SurfaceFinish.GLOSSY
        else:
            mat = MaterialType.CONCRETE
            finish = SurfaceFinish.MATTE
        mat_conf = 0.65
    elif any(k in c_lower for k in ("sofa", "couch", "cushion", "curtain", "carpet", "rug")):
        mat = MaterialType.FABRIC
        finish = SurfaceFinish.MATTE
        mat_conf = 0.80
    elif "road" in c_lower or "asphalt" in c_lower:
        mat = MaterialType.ASPHALT
        finish = SurfaceFinish.ROUGH
        mat_conf = 0.85
    elif "kerb" in c_lower or "footpath" in c_lower:
        mat = MaterialType.CONCRETE
        finish = SurfaceFinish.ROUGH
        mat_conf = 0.75

    # 4. Operational state (e.g. door open vs closed, light on vs off)
    op_state = ObjectState.UNSPECIFIED
    state_conf = 0.0
    if "door" in c_lower:
        # Aspect ratio of 2D bounding box can hint open vs closed
        w_px = x2 - x1
        h_px = y2 - y1
        aspect = w_px / max(1, h_px)
        if aspect < 0.25:
            op_state = ObjectState.OPEN
            state_conf = 0.60
        else:
            op_state = ObjectState.CLOSED
            state_conf = 0.70
    elif "light" in c_lower or "lamp" in c_lower:
        if specular_ratio > 0.15 or np.mean(gray_crop) > 220:
            op_state = ObjectState.ON
            state_conf = 0.80
        else:
            op_state = ObjectState.OFF
            state_conf = 0.70

    return EntityVisualAttributes(
        material=mat,
        material_confidence=mat_conf,
        surface_finish=finish,
        dominant_color_name=best_color_name,
        dominant_rgb=(float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])),
        operational_state=op_state,
        state_confidence=state_conf,
        is_reflective=is_refl,
        is_transparent=is_trans,
        details={
            "delta_e_closest_color": round(best_delta_e, 2),
            "texture_gradient_energy": round(roughness_energy, 2),
            "specular_pixel_fraction": round(specular_ratio, 4),
        },
    )
