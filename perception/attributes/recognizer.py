"""Evidence-backed attribute recognition for physical entities.

Strictly separates directly observed visual/geometric evidence
(``provenance="observed"``) from ontology/contextual priors
(``provenance="inferred"``). Never invents material, color, or depth
attributes when the supporting pixel or depth evidence is absent.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from perception.attributes.taxonomy import AttributeCategory
from perception.engine_evidence import ObservationRecord

ProvenanceKind = Literal["observed", "inferred"]

_LEXICAL_MATERIAL_CUES: dict[str, str] = {
    "wood": "wood",
    "wooden": "wood",
    "timber": "wood",
    "glass": "glass",
    "glazed": "glass",
    "metal": "metal",
    "metallic": "metal",
    "steel": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "concrete": "concrete",
    "brick": "masonry",
    "stone": "stone",
    "marble": "stone",
    "ceramic": "ceramic",
    "tile": "ceramic",
    "plastic": "polymer",
    "pvc": "polymer",
    "fabric": "fabric",
    "leather": "leather",
    "gypsum": "plaster",
    "drywall": "plaster",
}

_LEXICAL_STATE_CUES: dict[str, str] = {
    "open": "open",
    "closed": "closed",
    "ajar": "partially_open",
    "broken": "damaged",
    "cracked": "damaged",
    "rusted": "corroded",
}


@dataclass(frozen=True)
class AttributeEvidence:
    """Single evidence-grounded attribute observation or prior."""

    attribute_name: str
    value: str
    category: AttributeCategory
    provenance: ProvenanceKind
    evidence_source: str
    confidence: float
    uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityAttributeBundle:
    """All recognized attributes for one ObservationRecord."""

    evidence_id: str
    label: str
    ontology_path: str
    attributes: tuple[AttributeEvidence, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def observed(self) -> tuple[AttributeEvidence, ...]:
        return tuple(a for a in self.attributes if a.provenance == "observed")

    @property
    def inferred(self) -> tuple[AttributeEvidence, ...]:
        return tuple(a for a in self.attributes if a.provenance == "inferred")


def _classify_rgb_color(mean_rgb: np.ndarray) -> tuple[str, str]:
    """Return (color_name, brightness_band) from mean [R, G, B] in 0..255."""
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    brightness = "dark" if lum < 75.0 else ("bright" if lum > 185.0 else "medium")

    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = 0.0 if mx < 1.0 else (mx - mn) / mx
    if sat < 0.16:
        if lum < 55.0:
            return "black", brightness
        if lum > 200.0:
            return "white", brightness
        return "gray", brightness

    if r >= g and r >= b:
        if g > 140 and b < 90:
            return "yellow" if r - g < 45 else "orange", brightness
        if r - g < 40 and r - b > 40:
            return "brown", brightness
        return "red", brightness
    if g >= r and g >= b:
        return "green", brightness
    return "blue", brightness


def _extract_crop_pixels(
    obs: ObservationRecord,
    image_rgb: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """Extract RGB pixels (N, 3) from mask or bbox_px, plus source tag."""
    h, w = image_rgb.shape[:2]
    if obs.mask is not None:
        m = np.asarray(obs.mask, dtype=bool)
        if m.shape[:2] == (h, w) and int(m.sum()) > 0:
            return image_rgb[m], "mask_pixels"

    if obs.bbox_px is not None and len(obs.bbox_px) == 4:
        x1, y1, x2, y2 = obs.bbox_px
        ix1 = max(0, min(w - 1, int(round(x1))))
        iy1 = max(0, min(h - 1, int(round(y1))))
        ix2 = max(ix1 + 1, min(w, int(round(x2))))
        iy2 = max(iy1 + 1, min(h, int(round(y2))))
        crop = image_rgb[iy1:iy2, ix1:ix2]
        if crop.size > 0:
            return crop.reshape(-1, 3), "bbox_crop_pixels"

    return None, "none"


def extract_observed_attributes(
    obs: ObservationRecord,
    image_rgb: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
) -> EntityAttributeBundle:
    """Extract strictly evidence-backed (``observed``) attributes from an observation.

    - Color, brightness, and texture variance are emitted ONLY if ``image_rgb``
      is supplied and covers the observation mask or bbox.
    - Depth band is emitted ONLY if ``depth_map`` is supplied and overlaps the
      observation mask or bbox.
    - Geometric aspect/orientation is emitted ONLY if ``obs.bbox_px`` is valid.
    - Lexical material/state cues are emitted ONLY if explicitly present in the
      observed label text.
    """
    attrs: list[AttributeEvidence] = []
    ontology_path = str(obs.metadata.get("ontology_path") or obs.label or "")
    raw_label = str(obs.metadata.get("raw_label") or obs.label or "").lower()

    # 1. Visual pixel attributes (color, brightness, surface texture variance)
    if image_rgb is not None and image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
        pixels, src = _extract_crop_pixels(obs, image_rgb[:, :, :3])
        if pixels is not None and len(pixels) > 0:
            mean_rgb = pixels.astype(np.float32).mean(axis=0)
            std_lum = float(pixels.astype(np.float32).std())
            color_name, brightness = _classify_rgb_color(mean_rgb)
            attrs.append(AttributeEvidence(
                attribute_name="dominant_color",
                value=color_name,
                category=AttributeCategory.MATERIALS_FINISHES,
                provenance="observed",
                evidence_source=src,
                confidence=0.80 if src == "mask_pixels" else 0.65,
                uncertainty=0.20 if src == "mask_pixels" else 0.35,
                metadata={"mean_rgb": [round(float(v), 1) for v in mean_rgb]},
            ))
            attrs.append(AttributeEvidence(
                attribute_name="surface_brightness",
                value=brightness,
                category=AttributeCategory.MATERIALS_FINISHES,
                provenance="observed",
                evidence_source=src,
                confidence=0.85 if src == "mask_pixels" else 0.70,
                uncertainty=0.15 if src == "mask_pixels" else 0.30,
            ))
            texture = "uniform_smooth" if std_lum < 18.0 else ("textured" if std_lum > 42.0 else "moderate_texture")
            attrs.append(AttributeEvidence(
                attribute_name="surface_texture",
                value=texture,
                category=AttributeCategory.MATERIALS_FINISHES,
                provenance="observed",
                evidence_source=src,
                confidence=0.75 if src == "mask_pixels" else 0.60,
                uncertainty=0.25 if src == "mask_pixels" else 0.40,
                metadata={"luminance_std": round(std_lum, 2)},
            ))

    # 2. Geometric bbox attributes (aspect ratio / orientation / small-object flag)
    if obs.bbox_px is not None and len(obs.bbox_px) == 4:
        bw = max(0.0, float(obs.bbox_px[2]) - float(obs.bbox_px[0]))
        bh = max(0.0, float(obs.bbox_px[3]) - float(obs.bbox_px[1]))
        if bw > 0.0 and bh > 0.0:
            ratio = bw / bh
            orient = "horizontal" if ratio >= 1.35 else ("vertical" if ratio <= 0.74 else "compact")
            attrs.append(AttributeEvidence(
                attribute_name="2d_orientation",
                value=orient,
                category=AttributeCategory.STRUCTURAL_ELEMENTS,
                provenance="observed",
                evidence_source="bbox_geometry",
                confidence=0.90,
                uncertainty=0.10,
                metadata={"width_px": round(bw, 1), "height_px": round(bh, 1), "aspect_ratio": round(ratio, 3)},
            ))
            area = bw * bh
            scale_band = "tiny" if area < 1024 else ("small" if area < 4096 else ("medium" if area < 40000 else "large"))
            attrs.append(AttributeEvidence(
                attribute_name="apparent_pixel_scale",
                value=scale_band,
                category=AttributeCategory.ARCHITECTURAL_DETAILS,
                provenance="observed",
                evidence_source="bbox_geometry",
                confidence=0.95,
                uncertainty=0.05,
                metadata={"area_px2": round(area, 1)},
            ))

    # 3. Depth-backed distance band (only when depth_map is provided)
    if depth_map is not None and depth_map.ndim == 2:
        dh, dw = depth_map.shape
        d_vals: np.ndarray | None = None
        d_src = "none"
        if obs.mask is not None:
            m = np.asarray(obs.mask, dtype=bool)
            if m.shape == (dh, dw) and int(m.sum()) > 0:
                d_vals = depth_map[m]
                d_src = "depth_mask_crop"
        if d_vals is None and obs.bbox_px is not None and len(obs.bbox_px) == 4:
            x1, y1, x2, y2 = obs.bbox_px
            ix1 = max(0, min(dw - 1, int(round(x1))))
            iy1 = max(0, min(dh - 1, int(round(y1))))
            ix2 = max(ix1 + 1, min(dw, int(round(x2))))
            iy2 = max(iy1 + 1, min(dh, int(round(y2))))
            crop_d = depth_map[iy1:iy2, ix1:ix2]
            if crop_d.size > 0:
                d_vals = crop_d.reshape(-1)
                d_src = "depth_bbox_crop"
        if d_vals is not None and d_vals.size > 0:
            finite = d_vals[np.isfinite(d_vals)]
            if finite.size > 0:
                med = float(np.median(finite))
                all_med = float(np.nanmedian(depth_map)) if np.isfinite(depth_map).any() else med
                rel = "foreground" if med < 0.85 * all_med else ("background" if med > 1.15 * all_med else "midground")
                attrs.append(AttributeEvidence(
                    attribute_name="relative_depth_band",
                    value=rel,
                    category=AttributeCategory.STRUCTURAL_ELEMENTS,
                    provenance="observed",
                    evidence_source=d_src,
                    confidence=0.78 if d_src == "depth_mask_crop" else 0.65,
                    uncertainty=0.22 if d_src == "depth_mask_crop" else 0.35,
                    metadata={"median_depth": round(med, 4), "scene_median_depth": round(all_med, 4)},
                ))

    # 4. Explicit lexical cues from model label (e.g. "wooden door", "glass railing")
    tokens = set(raw_label.replace("_", " ").replace("-", " ").split())
    for tok, mat in _LEXICAL_MATERIAL_CUES.items():
        if tok in tokens:
            attrs.append(AttributeEvidence(
                attribute_name="material_cue",
                value=mat,
                category=AttributeCategory.MATERIALS_FINISHES,
                provenance="observed",
                evidence_source="label_lexical",
                confidence=float(obs.confidence) if obs.confidence is not None else 0.6,
                uncertainty=round(1.0 - float(obs.confidence), 3) if obs.confidence is not None else 0.4,
                metadata={"matched_token": tok},
            ))
            break
    for tok, st in _LEXICAL_STATE_CUES.items():
        if tok in tokens:
            attrs.append(AttributeEvidence(
                attribute_name="operational_state",
                value=st,
                category=AttributeCategory.DOOR_HARDWARE,
                provenance="observed",
                evidence_source="label_lexical",
                confidence=float(obs.confidence) if obs.confidence is not None else 0.55,
                uncertainty=round(1.0 - float(obs.confidence), 3) if obs.confidence is not None else 0.45,
                metadata={"matched_token": tok},
            ))
            break

    return EntityAttributeBundle(
        evidence_id=obs.evidence_id,
        label=obs.label,
        ontology_path=ontology_path,
        attributes=tuple(attrs),
        summary={
            "n_observed": len(attrs),
            "n_inferred": 0,
            "sources": sorted({a.evidence_source for a in attrs}),
        },
    )


def infer_prior_attributes(
    obs: ObservationRecord,
    ontology_attributes: dict[str, Sequence[str]] | None = None,
) -> tuple[AttributeEvidence, ...]:
    """Emit explicitly tagged ``provenance="inferred"`` ontology priors (never marked observed)."""
    if not ontology_attributes:
        return ()
    priors: list[AttributeEvidence] = []
    for key, values in ontology_attributes.items():
        if not values:
            continue
        priors.append(AttributeEvidence(
            attribute_name=str(key),
            value=str(values[0]),
            category=AttributeCategory.MATERIALS_FINISHES,
            provenance="inferred",
            evidence_source="ontology_prior",
            confidence=0.35,
            uncertainty=0.65,
            metadata={"candidate_values": [str(v) for v in values]},
        ))
    return tuple(priors)
