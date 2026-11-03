"""Texture synthesis, UV projection, and seam blending."""

from .projection import (
    PinholeCamera,
    PlanarSurface,
    TextureAtlas,
    multi_band_seam_blend,
    project_and_blend_planar_texture,
)

__all__ = [
    "PinholeCamera",
    "PlanarSurface",
    "TextureAtlas",
    "multi_band_seam_blend",
    "project_and_blend_planar_texture",
]
