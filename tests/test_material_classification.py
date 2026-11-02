"""Unit tests for surface material and texture classification (Master Development Plan Task 6.10)."""

import numpy as np
import pytest

from perception.materials.classifier import (
    CONFIDENCE_CAP_COLOR_ONLY,
    CONFIDENCE_CAP_WITH_TEXTURE,
    MaterialCategory,
    MaterialEvidence,
    classify_material,
    extract_texture_statistics,
)


def test_classify_glass_from_translucency_and_sheen():
    # Glass in a window opening with high specular sheen
    ev = MaterialEvidence(
        color_rgb=(210, 220, 225),
        specular_sheen=0.65,
        texture_roughness=0.05,
        surface_kind="window",
        is_translucent=True,
    )
    results = classify_material(ev)
    assert len(results) >= 1
    top = results[0]
    assert top.category == MaterialCategory.GLASS
    assert top.material == "glass"
    assert top.score >= 0.8
    assert top.confidence > CONFIDENCE_CAP_COLOR_ONLY  # Texture evidence raises confidence cap


def test_classify_metal_from_neutral_chroma_and_sheen():
    # Polished steel / aluminium: neutral grey (180, 180, 185) with high sheen
    ev = MaterialEvidence(
        color_rgb=(180, 180, 185),
        specular_sheen=0.55,
        texture_roughness=0.08,
        surface_kind="frame",
    )
    results = classify_material(ev)
    top = results[0]
    assert top.category == MaterialCategory.METAL
    assert top.material == "metal"
    assert top.score >= 0.7


def test_classify_brick_from_terracotta_hue_and_roughness():
    # Red-orange brick: RGB (170, 75, 45) with rough mortar texture
    ev = MaterialEvidence(
        color_rgb=(170, 75, 45),
        spread_delta_e=12.0,
        texture_roughness=0.45,
        surface_kind="wall",
    )
    results = classify_material(ev)
    top = results[0]
    assert top.category == MaterialCategory.BRICK
    assert top.material == "brick"
    assert "terracotta" in top.basis or "roughness" in top.basis


def test_classify_stone_from_earthy_roughness():
    # Rough limestone / granite: RGB (160, 155, 145), roughness 0.50, low sheen
    ev = MaterialEvidence(
        color_rgb=(160, 155, 145),
        texture_roughness=0.50,
        specular_sheen=0.10,
        surface_kind="wall",
    )
    results = classify_material(ev)
    categories = [r.category for r in results]
    assert MaterialCategory.STONE in categories
    stone_res = next(r for r in results if r.category == MaterialCategory.STONE)
    assert stone_res.score >= 0.5


def test_classify_tile_from_smooth_glaze_sheen():
    # Glazed ceramic tile: white/blue tile with high sheen and low roughness
    ev = MaterialEvidence(
        color_rgb=(240, 240, 245),
        spread_delta_e=4.0,
        specular_sheen=0.45,
        texture_roughness=0.10,
        surface_kind="bathroom",
    )
    results = classify_material(ev)
    categories = [r.category for r in results]
    assert MaterialCategory.TILE in categories


def test_classify_fabric_on_furnishing():
    # Fabric on a sofa: RGB (80, 95, 110), completely matte
    ev = MaterialEvidence(
        color_rgb=(80, 95, 110),
        specular_sheen=0.05,
        texture_roughness=0.25,
        surface_kind="sofa",
    )
    results = classify_material(ev)
    top = results[0]
    assert top.category == MaterialCategory.FABRIC
    assert top.material == "fabric"


def test_extract_texture_statistics_smooth_vs_rough():
    # 1. Flat smooth patch: identical pixels
    smooth_tex = np.ones((32, 32, 3), dtype=np.uint8) * 180
    smooth_stats = extract_texture_statistics(smooth_tex)
    assert smooth_stats["texture_roughness"] == 0.0

    # 2. Checkerboard / rough patch: high spatial frequency gradients
    rough_tex = np.zeros((32, 32, 3), dtype=np.uint8)
    rough_tex[::2, ::2] = 255
    rough_stats = extract_texture_statistics(rough_tex)
    assert rough_stats["texture_roughness"] > 0.4

    # 3. Specular patch: mostly dark with intense specular highlight dots
    specular_tex = np.zeros((32, 32, 3), dtype=np.uint8) + 30
    specular_tex[10:14, 10:14] = 255  # localized bright specular spot
    spec_stats = extract_texture_statistics(specular_tex)
    assert spec_stats["specular_sheen"] > 0.3


def test_texture_raises_confidence_cap():
    # Color-only evidence
    ev_color_only = MaterialEvidence(
        color_rgb=(150, 110, 70),
        surface_kind="floor",
    )
    res_color = classify_material(ev_color_only)
    assert res_color[0].confidence <= CONFIDENCE_CAP_COLOR_ONLY

    # Color + measured texture roughness
    ev_with_tex = MaterialEvidence(
        color_rgb=(150, 110, 70),
        texture_roughness=0.30,
        surface_kind="floor",
    )
    res_tex = classify_material(ev_with_tex)
    assert res_tex[0].confidence > CONFIDENCE_CAP_COLOR_ONLY
    assert res_tex[0].confidence <= CONFIDENCE_CAP_WITH_TEXTURE
