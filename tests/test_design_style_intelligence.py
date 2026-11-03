"""Tests for Architectural and Interior Design Style Intelligence (Wave 1, Block 8)."""

from __future__ import annotations

import pytest

from perception.style import (
    DesignStyle,
    OrnamentationLevel,
    STYLE_PROFILES,
    StyleAssessment,
    analyze_scene_style,
)
from world_ir.schema_v1 import Entity, EntityType, Material, Provenance
from world_ir.world_v1 import WorldIR


def test_style_profiles_complete():
    """Verify all 13 canonical design styles are registered with profiles."""
    expected_styles = [
        DesignStyle.MINIMALIST,
        DesignStyle.SCANDINAVIAN,
        DesignStyle.INDUSTRIAL,
        DesignStyle.BRUTALIST,
        DesignStyle.TRADITIONAL_INDIAN,
        DesignStyle.SOUTH_INDIAN_VERNACULAR,
        DesignStyle.MEDITERRANEAN,
        DesignStyle.JAPANESE,
        DesignStyle.ART_DECO,
        DesignStyle.CLASSICAL,
        DesignStyle.GOTHIC,
        DesignStyle.MODERN,
        DesignStyle.CONTEMPORARY,
    ]
    for s in expected_styles:
        assert s in STYLE_PROFILES
        profile = STYLE_PROFILES[s]
        assert len(profile.dominant_materials) > 0
        assert len(profile.typical_palette) > 0
        assert profile.ornamentation in OrnamentationLevel


def test_industrial_style_classification():
    world = WorldIR()
    # Add materials typical of industrial style
    mat_brick = Material(id="m-brick", name="Exposed Red Brick", class_name="brick", provenance=Provenance.OBSERVED)
    mat_metal = Material(id="m-metal", name="Black Steel Framing", class_name="metal", provenance=Provenance.OBSERVED)
    mat_concrete = Material(id="m-concrete", name="Polished Concrete Floor", class_name="concrete", provenance=Provenance.OBSERVED)
    world.materials[mat_brick.id] = mat_brick
    world.materials[mat_metal.id] = mat_metal
    world.materials[mat_concrete.id] = mat_concrete

    # Add entities
    e_conduit = Entity(
        id="ent-conduit-1",
        name="Exposed Electrical Conduit",
        type=EntityType.STRUCTURE,
        custom_properties={"entity_class": "electrical.conduit"},
    )
    e_wall = Entity(
        id="ent-wall-1",
        name="Brick Wall",
        type=EntityType.WALL,
        custom_properties={"dimensions_m": [5.0, 3.0, 0.2], "material": "brick"},
    )
    world.entities[e_conduit.id] = e_conduit
    world.entities[e_wall.id] = e_wall

    assessment = analyze_scene_style(world, visual_tags=["exposed brick", "duct", "charcoal metal"])

    assert isinstance(assessment, StyleAssessment)
    assert assessment.primary_style == DesignStyle.INDUSTRIAL
    assert assessment.confidence >= 0.50
    assert any("brick" in m for m in assessment.dominant_materials)
    assert assessment.detected_ornamentation == OrnamentationLevel.NONE_MINIMAL


def test_traditional_indian_style_classification():
    world = WorldIR()
    mat_wood = Material(id="m-wood", name="Carved Teak Wood", class_name="wood", provenance=Provenance.OBSERVED)
    mat_stone = Material(id="m-stone", name="Sandstone", class_name="stone", provenance=Provenance.OBSERVED)
    world.materials[mat_wood.id] = mat_wood
    world.materials[mat_stone.id] = mat_stone

    e_arch = Entity(
        id="ent-arch-1",
        name="Carved Stone Arch",
        type=EntityType.ARCH,
        custom_properties={"entity_class": "structural.arch"},
    )
    e_column = Entity(
        id="ent-col-1",
        name="Carved Wooden Column",
        type=EntityType.COLUMN,
        custom_properties={"entity_class": "structural.column"},
    )
    e_moulding = Entity(
        id="ent-mould-1",
        name="Ceiling Moulding",
        type=EntityType.STRUCTURE,
        custom_properties={"entity_class": "finish.moulding"},
    )
    world.entities[e_arch.id] = e_arch
    world.entities[e_column.id] = e_column
    world.entities[e_moulding.id] = e_moulding

    assessment = analyze_scene_style(world, visual_tags=["carved wooden pillars", "ornate courtyard", "teak"])

    assert assessment.primary_style == DesignStyle.TRADITIONAL_INDIAN
    assert assessment.detected_ornamentation in (OrnamentationLevel.HIGH_ELABORATE, OrnamentationLevel.HIGHLY_DECORATIVE)
    assert assessment.confidence >= 0.50
    assert any("Carved" in feat or "Arched" in feat or "mouldings" in feat for feat in assessment.matched_features)


def test_style_assessment_evidence_integrity():
    """Ensure style analyzer does not mutate underlying world entities or geometry."""
    world = WorldIR()
    e = Entity(id="ent-test-1", name="Test Wall", type=EntityType.WALL)
    world.entities[e.id] = e

    assessment = analyze_scene_style(world)
    assert assessment is not None
    # Underlying entity must be completely untouched
    assert e.name == "Test Wall"
    assert e.type == EntityType.WALL
    assert len(world.geometries) == 0
