"""Architectural and Interior Design Style Taxonomy (Wave 1, Block 8).

Defines canonical global architectural and interior design styles, their material palettes,
geometric proportion signatures, ornamental density, and defining spatial patterns.

All style evaluations must be evidence-grounded: style labels must NEVER substitute
for direct geometric and material evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple


class OrnamentationLevel(str, Enum):
    """Density and prominence of decorative mouldings, carvings, and trims."""
    NONE_MINIMAL = "none_minimal"      # Flush planes, shadow gaps, no moulding
    LOW_RESTRAINED = "low_restrained"  # Clean simple trims, utilitarian
    MODERATE = "moderate"              # Defined architraves, baseboards, simple cornices
    HIGH_ELABORATE = "high_elaborate"  # Carved woodwork, rosettes, dentils, pilasters
    HIGHLY_DECORATIVE = "highly_decorative"  # Complex sculptural reliefs, intricate fretwork


class DesignStyle(str, Enum):
    """14 Canonical architectural and interior design styles."""
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    MINIMALIST = "minimalist"
    INDUSTRIAL = "industrial"
    SCANDINAVIAN = "scandinavian"
    TRADITIONAL_INDIAN = "traditional_indian"
    SOUTH_INDIAN_VERNACULAR = "south_indian_vernacular"
    MEDITERRANEAN = "mediterranean"
    GOTHIC = "gothic"
    CLASSICAL = "classical"
    JAPANESE = "japanese"
    BRUTALIST = "brutalist"
    ART_DECO = "art_deco"
    REGIONAL_VERNACULAR = "regional_vernacular"
    ECLECTIC_UNKNOWN = "eclectic_unknown"


@dataclass(frozen=True)
class StyleProfile:
    """Canonical geometric, material, and spatial characteristics of a design style."""
    style: DesignStyle
    display_name: str
    description: str
    dominant_materials: Tuple[str, ...]
    secondary_materials: Tuple[str, ...]
    ornamentation: OrnamentationLevel
    window_to_wall_ratio: Tuple[float, float]  # (min, max) expected ratio
    typical_palette: Tuple[str, ...]          # Descriptive color families
    distinguishing_features: Tuple[str, ...]


STYLE_PROFILES: Dict[DesignStyle, StyleProfile] = {
    DesignStyle.MINIMALIST: StyleProfile(
        style=DesignStyle.MINIMALIST,
        display_name="Minimalist",
        description="Radical simplicity, monochromatic palettes, flush concealed fixtures, unadorned surfaces.",
        dominant_materials=("plaster", "concrete", "glass"),
        secondary_materials=("wood", "metal"),
        ornamentation=OrnamentationLevel.NONE_MINIMAL,
        window_to_wall_ratio=(0.20, 0.60),
        typical_palette=("white", "light_grey", "black_accents"),
        distinguishing_features=("concealed storage", "flush baseboards", "shadow gaps", "open floorplans"),
    ),
    DesignStyle.SCANDINAVIAN: StyleProfile(
        style=DesignStyle.SCANDINAVIAN,
        display_name="Scandinavian",
        description="Warm minimalism, light timber finishes, ample natural illumination, soft textiles.",
        dominant_materials=("wood", "plaster"),
        secondary_materials=("glass", "fabric", "metal"),
        ornamentation=OrnamentationLevel.LOW_RESTRAINED,
        window_to_wall_ratio=(0.25, 0.50),
        typical_palette=("warm_white", "pale_oak", "soft_grey", "muted_blue"),
        distinguishing_features=("light wood flooring", "generous window daylighting", "clean organic curves"),
    ),
    DesignStyle.INDUSTRIAL: StyleProfile(
        style=DesignStyle.INDUSTRIAL,
        display_name="Industrial",
        description="Exposed infrastructure, raw utilitarian finishes, metal framing, distressed brick and concrete.",
        dominant_materials=("brick", "concrete", "metal"),
        secondary_materials=("wood", "glass"),
        ornamentation=OrnamentationLevel.NONE_MINIMAL,
        window_to_wall_ratio=(0.30, 0.70),
        typical_palette=("charcoal", "raw_brick_red", "grey_concrete", "black_iron"),
        distinguishing_features=("exposed conduit and ducts", "black steel window mullions", "distressed surfaces"),
    ),
    DesignStyle.BRUTALIST: StyleProfile(
        style=DesignStyle.BRUTALIST,
        display_name="Brutalist",
        description="Monolithic raw board-formed concrete, geometric massing, architectural honesty.",
        dominant_materials=("concrete",),
        secondary_materials=("glass", "metal", "stone"),
        ornamentation=OrnamentationLevel.NONE_MINIMAL,
        window_to_wall_ratio=(0.10, 0.35),
        typical_palette=("raw_grey", "deep_shadow", "aggregate_stone"),
        distinguishing_features=("board-formed concrete texture", "deep recessed window reveals", "monolithic columns"),
    ),
    DesignStyle.TRADITIONAL_INDIAN: StyleProfile(
        style=DesignStyle.TRADITIONAL_INDIAN,
        display_name="Traditional Indian",
        description="Intricate carved woodwork, stone masonry, brass/copper fixtures, central courtyards, jaali screens.",
        dominant_materials=("wood", "stone", "brick"),
        secondary_materials=("metal", "tile", "plaster"),
        ornamentation=OrnamentationLevel.HIGH_ELABORATE,
        window_to_wall_ratio=(0.10, 0.30),
        typical_palette=("teak_brown", "terracotta", "sandstone_buff", "brass_gold", "deep_ochre"),
        distinguishing_features=("carved wooden pillars", "jaali lattice screens", "courtyard orientation", "ornate thresholds"),
    ),
    DesignStyle.SOUTH_INDIAN_VERNACULAR: StyleProfile(
        style=DesignStyle.SOUTH_INDIAN_VERNACULAR,
        display_name="South Indian Vernacular",
        description="Sloped terracotta tile roofs, heavy teak/rosewood pillars, Athangudi patterned floor tiles, open verandahs.",
        dominant_materials=("wood", "tile", "stone"),
        secondary_materials=("plaster", "brick"),
        ornamentation=OrnamentationLevel.HIGH_ELABORATE,
        window_to_wall_ratio=(0.12, 0.28),
        typical_palette=("terracotta_red", "dark_teak", "vibrant_tile_patterns", "lime_white"),
        distinguishing_features=("verandah (thinnai)", "ornate wooden bracket columns", "geometric Athangudi tiles", "pitched roofs"),
    ),
    DesignStyle.MEDITERRANEAN: StyleProfile(
        style=DesignStyle.MEDITERRANEAN,
        display_name="Mediterranean",
        description="Stucco / lime plaster walls, terracotta tiles, arched openings, wrought iron accents.",
        dominant_materials=("plaster", "tile", "stone"),
        secondary_materials=("metal", "wood"),
        ornamentation=OrnamentationLevel.MODERATE,
        window_to_wall_ratio=(0.15, 0.35),
        typical_palette=("lime_white", "terracotta_warm", "azure_blue", "olive_green"),
        distinguishing_features=("rounded plaster corners", "curved arched openings", "terracotta roof tiles", "wrought iron railings"),
    ),
    DesignStyle.JAPANESE: StyleProfile(
        style=DesignStyle.JAPANESE,
        display_name="Japanese Traditional / Japandi",
        description="Harmonious natural wood, tatami proportions, sliding shoji screens, low horizontal lines.",
        dominant_materials=("wood", "plaster"),
        secondary_materials=("glass", "stone", "fabric"),
        ornamentation=OrnamentationLevel.LOW_RESTRAINED,
        window_to_wall_ratio=(0.20, 0.50),
        typical_palette=("natural_cedar", "washi_offwhite", "bamboo_tan", "slate_dark"),
        distinguishing_features=("sliding lattice doors", "low furniture heights", "horizontal framing lines", "tatami modules"),
    ),
    DesignStyle.ART_DECO: StyleProfile(
        style=DesignStyle.ART_DECO,
        display_name="Art Deco",
        description="Geometric glamour, zigzags, stepped forms, polished marble, brass/chrome metallic accents.",
        dominant_materials=("stone", "metal", "glass"),
        secondary_materials=("wood", "plaster"),
        ornamentation=OrnamentationLevel.HIGH_ELABORATE,
        window_to_wall_ratio=(0.20, 0.45),
        typical_palette=("black", "polished_gold", "emerald", "marble_white", "chrome"),
        distinguishing_features=("stepped mouldings", "chevron / sunburst motifs", "contrasting geometric inlays"),
    ),
    DesignStyle.CLASSICAL: StyleProfile(
        style=DesignStyle.CLASSICAL,
        display_name="Classical / Neoclassical",
        description="Formal symmetry, classical orders (columns/capitals), coffered ceilings, layered mouldings.",
        dominant_materials=("plaster", "stone", "wood"),
        secondary_materials=("glass", "metal"),
        ornamentation=OrnamentationLevel.HIGH_ELABORATE,
        window_to_wall_ratio=(0.18, 0.38),
        typical_palette=("ivory_white", "stone_limestone", "gold_leaf", "deep_walnut"),
        distinguishing_features=("cornice and frieze mouldings", "fluted columns", "symmetrical opening arrangements"),
    ),
    DesignStyle.GOTHIC: StyleProfile(
        style=DesignStyle.GOTHIC,
        display_name="Gothic / Neo-Gothic",
        description="Pointed arches, ribbed vaulting, vertical emphasis, dark stained timber and stonework.",
        dominant_materials=("stone", "wood"),
        secondary_materials=("glass", "metal"),
        ornamentation=OrnamentationLevel.HIGHLY_DECORATIVE,
        window_to_wall_ratio=(0.15, 0.35),
        typical_palette=("weathered_stone", "dark_oak", "stained_glass_hues"),
        distinguishing_features=("pointed lancet arches", "tracery patterns", "vertical ribbed columns"),
    ),
    DesignStyle.MODERN: StyleProfile(
        style=DesignStyle.MODERN,
        display_name="Modern (Mid-Century / International)",
        description="Form follows function, horizontal ribbon windows, planar surfaces, structural clarity.",
        dominant_materials=("glass", "concrete", "metal"),
        secondary_materials=("wood", "plaster"),
        ornamentation=OrnamentationLevel.NONE_MINIMAL,
        window_to_wall_ratio=(0.30, 0.65),
        typical_palette=("white", "teak_medium", "primary_accents", "steel_grey"),
        distinguishing_features=("ribbon glazing", "cantilevers", "open spatial flow", "clean rectangular planes"),
    ),
    DesignStyle.CONTEMPORARY: StyleProfile(
        style=DesignStyle.CONTEMPORARY,
        display_name="Contemporary",
        description="Fluid curves, sustainable materials, high-tech integration, neutral backgrounds with bold textures.",
        dominant_materials=("plaster", "glass", "wood"),
        secondary_materials=("metal", "stone", "concrete"),
        ornamentation=OrnamentationLevel.LOW_RESTRAINED,
        window_to_wall_ratio=(0.25, 0.55),
        typical_palette=("taupe", "warm_grey", "natural_wood", "black_metal"),
        distinguishing_features=("curvilinear elements", "indirect LED cove illumination", "large-format tile floors"),
    ),
}
