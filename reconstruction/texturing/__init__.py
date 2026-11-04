"""Texturing subsystem for Reality Engine (Blueprint 30 Task 10.1)."""

from .projection import (
    MeshTextureProjector,
    TextureResult,
    apply_texture_projection,
)

__all__ = [
    "MeshTextureProjector",
    "TextureResult",
    "apply_texture_projection",
]
