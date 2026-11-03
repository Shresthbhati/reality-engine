"""Architectural & Interior Design Style Intelligence (Wave 1, Block 8)."""

from perception.style.analyzer import StyleAssessment, analyze_scene_style
from perception.style.taxonomy import (
    STYLE_PROFILES,
    DesignStyle,
    OrnamentationLevel,
    StyleProfile,
)

__all__ = [
    "DesignStyle",
    "OrnamentationLevel",
    "StyleProfile",
    "STYLE_PROFILES",
    "StyleAssessment",
    "analyze_scene_style",
]
