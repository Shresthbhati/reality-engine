"""Material & Appearance Intelligence Subsystem (A3.8).

Covers: wood, metal, glass, concrete, brick, stone, ceramic, fabric, plastic, paint.
Combines color descriptors, texture roughness, specular highlight ratio, and
cross-view material consistency while strictly flagging every output as an
unverified hypothesis (``is_physically_verified=False``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.intelligence.data_model import (
    AppearanceDescriptor,
    MaterialHypothesisRecord,
)
from perception.materials.material_hypotheses import hypothesize_materials

SUPPORTED_MATERIALS: tuple[str, ...] = (
    "wood", "metal", "glass", "concrete", "brick",
    "stone", "ceramic", "fabric", "plastic", "paint",
)

_COLOR_NAMES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (20, 20, 20)),
    ("white", (240, 240, 240)),
    ("gray", (128, 128, 128)),
    ("brown", (125, 80, 45)),
    ("beige", (215, 195, 165)),
    ("red", (180, 50, 50)),
    ("blue", (55, 95, 175)),
    ("green", (60, 140, 75)),
    ("yellow", (215, 190, 60)),
)


def _nearest_color_name(rgb: tuple[float, float, float]) -> str:
    best_name = "unknown"
    best_d2 = float("inf")
    for name, ref in _COLOR_NAMES:
        d2 = (rgb[0] - ref[0]) ** 2 + (rgb[1] - ref[1]) ** 2 + (rgb[2] - ref[2]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_name = name
    return best_name


def extract_appearance_and_materials(
    image_rgb: np.ndarray,
    box_px: tuple[float, float, float, float],
    *,
    canonical_label: str,
    mask_2d: np.ndarray | None = None,
    embedding: Sequence[float] = (),
    embedding_model: str | None = None,
) -> tuple[AppearanceDescriptor, tuple[MaterialHypothesisRecord, ...]]:
    """Compute color/texture appearance descriptors and ranked material hypotheses."""
    h, w = image_rgb.shape[:2]
    x1 = max(0, min(w - 1, int(round(box_px[0]))))
    y1 = max(0, min(h - 1, int(round(box_px[1]))))
    x2 = max(0, min(w, int(round(box_px[2]))))
    y2 = max(0, min(h, int(round(box_px[3]))))
    if x2 <= x1 or y2 <= y1:
        return AppearanceDescriptor(), ()

    crop = image_rgb[y1:y2, x1:x2]
    if mask_2d is not None and mask_2d.shape[:2] == (h, w):
        sub_mask = mask_2d[y1:y2, x1:x2]
        pixels = crop[sub_mask] if np.any(sub_mask) else crop.reshape(-1, 3)
    else:
        pixels = crop.reshape(-1, 3)

    if pixels.size == 0:
        return AppearanceDescriptor(), ()

    mean_rgb = (
        round(float(np.mean(pixels[:, 0])), 2),
        round(float(np.mean(pixels[:, 1])), 2),
        round(float(np.mean(pixels[:, 2])), 2),
    )
    color_name = _nearest_color_name(mean_rgb)

    r_h, _ = np.histogram(pixels[:, 0], bins=8, range=(0, 256), density=True)
    g_h, _ = np.histogram(pixels[:, 1], bins=8, range=(0, 256), density=True)
    b_h, _ = np.histogram(pixels[:, 2], bins=8, range=(0, 256), density=True)
    hist = np.concatenate([r_h, g_h, b_h])
    norm = float(np.linalg.norm(hist))
    if norm > 1e-6:
        hist = hist / norm

    luma = (crop[..., 0] * 0.299 + crop[..., 1] * 0.587 + crop[..., 2] * 0.114).astype(np.float32)
    roughness = 0.0
    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        gx = np.abs(np.diff(luma, axis=1))
        gy = np.abs(np.diff(luma, axis=0))
        roughness = round(float((np.mean(gx) + np.mean(gy)) * 0.5 / 255.0), 4)

    specular = float(np.mean(np.max(pixels, axis=1) > 240))
    is_glass_hint = canonical_label.lower() in ("window", "window_glass", "door_glazing", "mirror")

    raw_hyps = hypothesize_materials(
        mean_rgb,
        canonical_label,
        texture_roughness=roughness,
        specular_sheen=specular,
        is_translucent=is_glass_hint,
    )
    records = tuple(
        MaterialHypothesisRecord(
            material_category=str(h_item.get("category", "unknown")),
            material_name=str(h_item.get("material_name", h_item.get("category", "unknown"))),
            hypothesis_score=round(float(h_item.get("confidence", 0.3)), 4),
            evidence_basis=f"color={color_name}, roughness={roughness}, specular={round(specular, 3)}",
            is_physically_verified=False,
        )
        for h_item in raw_hyps[:3]
    )

    desc = AppearanceDescriptor(
        color_histogram_24=tuple(round(float(v), 5) for v in hist),
        embedding_vector=tuple(float(v) for v in embedding),
        embedding_model=embedding_model,
        dominant_color_name=color_name,
        mean_rgb=mean_rgb,
        texture_roughness=roughness,
    )
    return desc, records


@dataclass(frozen=True)
class CrossViewMaterialConsensus:
    identity_id: str
    consensus_material: str
    support_fraction: float
    view_votes: dict[str, int]
    is_consistent: bool
    is_physically_verified: bool = False


def aggregate_cross_view_materials(
    identity_id: str,
    per_view_materials: Sequence[Sequence[MaterialHypothesisRecord]],
) -> CrossViewMaterialConsensus:
    """Aggregate material hypotheses across views of the same object identity."""
    votes: Counter[str] = Counter()
    total = 0
    for hyps in per_view_materials:
        if not hyps:
            continue
        votes[hyps[0].material_category] += 1
        total += 1
    if total == 0:
        return CrossViewMaterialConsensus(identity_id, "unknown", 0.0, {}, False, False)
    top_mat, top_cnt = votes.most_common(1)[0]
    frac = round(top_cnt / float(total), 4)
    return CrossViewMaterialConsensus(
        identity_id=identity_id,
        consensus_material=top_mat,
        support_fraction=frac,
        view_votes=dict(votes),
        is_consistent=(frac >= 0.60),
        is_physically_verified=False,
    )


__all__ = [
    "CrossViewMaterialConsensus",
    "SUPPORTED_MATERIALS",
    "aggregate_cross_view_materials",
    "extract_appearance_and_materials",
]
