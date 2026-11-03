"""Tests for 12-Class PBR Material Classification and glTF/USD Assignment (Epic E06-1)."""

from __future__ import annotations

import math
from typing import List, Tuple
import pytest
import numpy as np

from perception.materials.classifier import (
    MaterialCategory,
    MaterialEvidence,
    classify_material,
    compute_pbr_parameters,
    assign_pbr_material,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from exporters.gltf.exporter import export_to_gltf_with_report
from exporters.usd.exporter import export_to_usda_with_report


def test_12_class_pbr_material_macro_f1_benchmark():
    """Benchmark evaluation across 12 canonical architectural materials asserting Macro F1 >= 90%."""

    benchmark_cases: List[Tuple[MaterialEvidence, MaterialCategory]] = [
        # 1. WOOD
        (MaterialEvidence(color_rgb=[160, 105, 55], texture_roughness=0.35, specular_sheen=0.15, surface_kind="floor"), MaterialCategory.WOOD),
        (MaterialEvidence(color_rgb=[130, 80, 40], texture_roughness=0.40, specular_sheen=0.12, surface_kind="door"), MaterialCategory.WOOD),
        (MaterialEvidence(color_rgb=[185, 135, 80], texture_roughness=0.30, specular_sheen=0.20, surface_kind="table"), MaterialCategory.WOOD),
        (MaterialEvidence(color_rgb=[110, 65, 30], texture_roughness=0.45, specular_sheen=0.10, surface_kind="desk"), MaterialCategory.WOOD),
        (MaterialEvidence(color_rgb=[175, 120, 65], texture_roughness=0.32, specular_sheen=0.18, surface_kind="furniture"), MaterialCategory.WOOD),

        # 2. GLASS
        (MaterialEvidence(color_rgb=[230, 240, 245], texture_roughness=0.02, specular_sheen=0.85, surface_kind="window", is_translucent=True), MaterialCategory.GLASS),
        (MaterialEvidence(color_rgb=[220, 235, 240], texture_roughness=0.03, specular_sheen=0.75, surface_kind="glazing", is_translucent=True), MaterialCategory.GLASS),
        (MaterialEvidence(color_rgb=[210, 225, 230], texture_roughness=0.01, specular_sheen=0.90, surface_kind="window"), MaterialCategory.GLASS),
        (MaterialEvidence(color_rgb=[240, 245, 250], texture_roughness=0.02, specular_sheen=0.70, surface_kind="opening", is_translucent=True), MaterialCategory.GLASS),
        (MaterialEvidence(color_rgb=[205, 220, 225], texture_roughness=0.04, specular_sheen=0.80, surface_kind="window"), MaterialCategory.GLASS),

        # 3. METAL
        (MaterialEvidence(color_rgb=[180, 185, 190], texture_roughness=0.15, specular_sheen=0.80, surface_kind="conduit"), MaterialCategory.METAL),
        (MaterialEvidence(color_rgb=[150, 150, 155], texture_roughness=0.10, specular_sheen=0.85, surface_kind="fixture"), MaterialCategory.METAL),
        (MaterialEvidence(color_rgb=[200, 200, 205], texture_roughness=0.08, specular_sheen=0.90, surface_kind="frame"), MaterialCategory.METAL),
        (MaterialEvidence(color_rgb=[120, 122, 125], texture_roughness=0.18, specular_sheen=0.75, surface_kind="metallic"), MaterialCategory.METAL),
        (MaterialEvidence(color_rgb=[215, 215, 220], texture_roughness=0.12, specular_sheen=0.88, surface_kind="metal"), MaterialCategory.METAL),

        # 4. CONCRETE
        (MaterialEvidence(color_rgb=[145, 148, 150], texture_roughness=0.45, specular_sheen=0.10, surface_kind="column"), MaterialCategory.CONCRETE),
        (MaterialEvidence(color_rgb=[160, 162, 165], texture_roughness=0.38, specular_sheen=0.12, surface_kind="beam"), MaterialCategory.CONCRETE),
        (MaterialEvidence(color_rgb=[135, 137, 140], texture_roughness=0.42, specular_sheen=0.08, surface_kind="foundation"), MaterialCategory.CONCRETE),
        (MaterialEvidence(color_rgb=[155, 155, 158], texture_roughness=0.35, specular_sheen=0.15, surface_kind="structure"), MaterialCategory.CONCRETE),
        (MaterialEvidence(color_rgb=[130, 132, 135], texture_roughness=0.48, specular_sheen=0.05, surface_kind="concrete"), MaterialCategory.CONCRETE),

        # 5. PLASTER
        (MaterialEvidence(color_rgb=[235, 235, 230], texture_roughness=0.10, specular_sheen=0.08, surface_kind="wall", spread_delta_e=3.0), MaterialCategory.PLASTER),
        (MaterialEvidence(color_rgb=[242, 240, 238], texture_roughness=0.08, specular_sheen=0.05, surface_kind="ceiling", spread_delta_e=2.5), MaterialCategory.PLASTER),
        (MaterialEvidence(color_rgb=[225, 228, 225], texture_roughness=0.12, specular_sheen=0.10, surface_kind="wall", spread_delta_e=4.0), MaterialCategory.PLASTER),
        (MaterialEvidence(color_rgb=[245, 245, 242], texture_roughness=0.06, specular_sheen=0.04, surface_kind="ceiling", spread_delta_e=1.8), MaterialCategory.PLASTER),
        (MaterialEvidence(color_rgb=[220, 222, 218], texture_roughness=0.14, specular_sheen=0.09, surface_kind="wall", spread_delta_e=3.5), MaterialCategory.PLASTER),

        # 6. BRICK
        (MaterialEvidence(color_rgb=[168, 62, 45], texture_roughness=0.65, specular_sheen=0.08, surface_kind="wall"), MaterialCategory.BRICK),
        (MaterialEvidence(color_rgb=[185, 75, 52], texture_roughness=0.58, specular_sheen=0.06, surface_kind="facade"), MaterialCategory.BRICK),
        (MaterialEvidence(color_rgb=[152, 55, 38], texture_roughness=0.70, specular_sheen=0.05, surface_kind="exterior"), MaterialCategory.BRICK),
        (MaterialEvidence(color_rgb=[175, 68, 48], texture_roughness=0.62, specular_sheen=0.10, surface_kind="wall"), MaterialCategory.BRICK),
        (MaterialEvidence(color_rgb=[160, 58, 42], texture_roughness=0.66, specular_sheen=0.07, surface_kind="facade"), MaterialCategory.BRICK),

        # 7. TILE
        (MaterialEvidence(color_rgb=[230, 232, 235], texture_roughness=0.10, specular_sheen=0.55, surface_kind="bathroom"), MaterialCategory.TILE),
        (MaterialEvidence(color_rgb=[215, 220, 225], texture_roughness=0.12, specular_sheen=0.48, surface_kind="kitchen"), MaterialCategory.TILE),
        (MaterialEvidence(color_rgb=[240, 240, 242], texture_roughness=0.08, specular_sheen=0.60, surface_kind="floor"), MaterialCategory.TILE),
        (MaterialEvidence(color_rgb=[205, 210, 215], texture_roughness=0.14, specular_sheen=0.52, surface_kind="bathroom"), MaterialCategory.TILE),
        (MaterialEvidence(color_rgb=[225, 228, 230], texture_roughness=0.09, specular_sheen=0.58, surface_kind="kitchen"), MaterialCategory.TILE),

        # 8. STONE
        (MaterialEvidence(color_rgb=[155, 145, 135], texture_roughness=0.55, specular_sheen=0.12, surface_kind="surface"), MaterialCategory.STONE),
        (MaterialEvidence(color_rgb=[140, 132, 125], texture_roughness=0.62, specular_sheen=0.10, surface_kind="exterior"), MaterialCategory.STONE),
        (MaterialEvidence(color_rgb=[168, 158, 148], texture_roughness=0.50, specular_sheen=0.14, surface_kind="surface"), MaterialCategory.STONE),
        (MaterialEvidence(color_rgb=[132, 125, 118], texture_roughness=0.68, specular_sheen=0.08, surface_kind="exterior"), MaterialCategory.STONE),
        (MaterialEvidence(color_rgb=[160, 150, 140], texture_roughness=0.52, specular_sheen=0.11, surface_kind="stone"), MaterialCategory.STONE),

        # 9. FABRIC
        (MaterialEvidence(color_rgb=[120, 85, 145], texture_roughness=0.35, specular_sheen=0.06, surface_kind="sofa"), MaterialCategory.FABRIC),
        (MaterialEvidence(color_rgb=[70, 110, 150], texture_roughness=0.32, specular_sheen=0.08, surface_kind="chair"), MaterialCategory.FABRIC),
        (MaterialEvidence(color_rgb=[190, 180, 170], texture_roughness=0.38, specular_sheen=0.05, surface_kind="curtain"), MaterialCategory.FABRIC),
        (MaterialEvidence(color_rgb=[140, 95, 80], texture_roughness=0.40, specular_sheen=0.07, surface_kind="cushion"), MaterialCategory.FABRIC),
        (MaterialEvidence(color_rgb=[85, 130, 95], texture_roughness=0.34, specular_sheen=0.05, surface_kind="rug"), MaterialCategory.FABRIC),

        # 10. PLASTIC
        (MaterialEvidence(color_rgb=[245, 245, 248], texture_roughness=0.08, specular_sheen=0.42, surface_kind="switch"), MaterialCategory.PLASTIC),
        (MaterialEvidence(color_rgb=[240, 242, 245], texture_roughness=0.06, specular_sheen=0.38, surface_kind="socket"), MaterialCategory.PLASTIC),
        (MaterialEvidence(color_rgb=[235, 238, 240], texture_roughness=0.09, specular_sheen=0.45, surface_kind="outlet"), MaterialCategory.PLASTIC),
        (MaterialEvidence(color_rgb=[230, 232, 235], texture_roughness=0.10, specular_sheen=0.35, surface_kind="conduit"), MaterialCategory.PLASTIC),
        (MaterialEvidence(color_rgb=[248, 248, 250], texture_roughness=0.05, specular_sheen=0.40, surface_kind="switchboard"), MaterialCategory.PLASTIC),

        # 11. LEATHER
        (MaterialEvidence(color_rgb=[65, 42, 28], texture_roughness=0.30, specular_sheen=0.32, surface_kind="sofa"), MaterialCategory.LEATHER),
        (MaterialEvidence(color_rgb=[85, 55, 35], texture_roughness=0.28, specular_sheen=0.35, surface_kind="chair"), MaterialCategory.LEATHER),
        (MaterialEvidence(color_rgb=[50, 32, 22], texture_roughness=0.34, specular_sheen=0.28, surface_kind="couch"), MaterialCategory.LEATHER),
        (MaterialEvidence(color_rgb=[95, 62, 40], texture_roughness=0.25, specular_sheen=0.36, surface_kind="furniture"), MaterialCategory.LEATHER),
        (MaterialEvidence(color_rgb=[58, 38, 26], texture_roughness=0.32, specular_sheen=0.30, surface_kind="upholstery"), MaterialCategory.LEATHER),

        # 12. ASPHALT
        (MaterialEvidence(color_rgb=[45, 45, 48], texture_roughness=0.75, specular_sheen=0.05, surface_kind="road"), MaterialCategory.ASPHALT),
        (MaterialEvidence(color_rgb=[50, 52, 55], texture_roughness=0.70, specular_sheen=0.08, surface_kind="street"), MaterialCategory.ASPHALT),
        (MaterialEvidence(color_rgb=[40, 42, 44], texture_roughness=0.80, specular_sheen=0.04, surface_kind="curb"), MaterialCategory.ASPHALT),
        (MaterialEvidence(color_rgb=[48, 48, 50], texture_roughness=0.72, specular_sheen=0.06, surface_kind="pavement"), MaterialCategory.ASPHALT),
        (MaterialEvidence(color_rgb=[52, 54, 56], texture_roughness=0.68, specular_sheen=0.07, surface_kind="sidewalk"), MaterialCategory.ASPHALT),
    ]

    all_categories = [
        MaterialCategory.WOOD, MaterialCategory.GLASS, MaterialCategory.METAL,
        MaterialCategory.CONCRETE, MaterialCategory.PLASTER, MaterialCategory.BRICK,
        MaterialCategory.TILE, MaterialCategory.STONE, MaterialCategory.FABRIC,
        MaterialCategory.PLASTIC, MaterialCategory.LEATHER, MaterialCategory.ASPHALT,
    ]

    true_positives = {cat: 0 for cat in all_categories}
    false_positives = {cat: 0 for cat in all_categories}
    false_negatives = {cat: 0 for cat in all_categories}

    for ev, expected_cat in benchmark_cases:
        results = classify_material(ev)
        assert len(results) > 0, f"No classification for {expected_cat}"
        predicted_cat = results[0].category

        if predicted_cat == expected_cat:
            true_positives[expected_cat] += 1
        else:
            false_negatives[expected_cat] += 1
            if predicted_cat in false_positives:
                false_positives[predicted_cat] += 1

    f1_scores = []
    for cat in all_categories:
        tp = true_positives[cat]
        fp = false_positives[cat]
        fn = false_negatives[cat]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n12-Class PBR Material Classifier Macro F1: {macro_f1 * 100:.2f}%")

    # Assert Macro F1 >= 90% per backlog specification
    assert macro_f1 >= 0.90, f"Macro F1 {macro_f1} is below 90% threshold"


def test_pbr_parameter_bounds_and_physical_calibration():
    # Glass: high transmission, minimal roughness, 0 metallic
    ev_glass = MaterialEvidence(color_rgb=[230, 240, 245], texture_roughness=0.02, specular_sheen=0.85, surface_kind="window", is_translucent=True)
    res_glass = classify_material(ev_glass)[0]
    pbr_glass = res_glass.pbr_parameters
    assert pbr_glass.category == MaterialCategory.GLASS
    assert pbr_glass.transmission >= 0.90
    assert pbr_glass.roughness <= 0.10
    assert pbr_glass.metallic == 0.0
    assert 1.50 <= pbr_glass.ior <= 1.55

    # Metal: high metallic, low-to-medium roughness
    ev_metal = MaterialEvidence(color_rgb=[180, 185, 190], texture_roughness=0.15, specular_sheen=0.80, surface_kind="conduit")
    res_metal = classify_material(ev_metal)[0]
    pbr_metal = res_metal.pbr_parameters
    assert pbr_metal.category == MaterialCategory.METAL
    assert pbr_metal.metallic >= 0.85
    assert pbr_metal.roughness <= 0.35

    # Concrete: high roughness, 0 metallic, high normal scale
    ev_conc = MaterialEvidence(color_rgb=[145, 148, 150], texture_roughness=0.45, specular_sheen=0.10, surface_kind="column")
    res_conc = classify_material(ev_conc)[0]
    pbr_conc = res_conc.pbr_parameters
    assert pbr_conc.category == MaterialCategory.CONCRETE
    assert pbr_conc.roughness >= 0.60
    assert pbr_conc.metallic == 0.0
    assert pbr_conc.normal_scale >= 0.60

    # Asphalt: very dark albedo, high roughness
    ev_asphalt = MaterialEvidence(color_rgb=[45, 45, 48], texture_roughness=0.75, specular_sheen=0.05, surface_kind="road")
    res_asphalt = classify_material(ev_asphalt)[0]
    pbr_asphalt = res_asphalt.pbr_parameters
    assert pbr_asphalt.category == MaterialCategory.ASPHALT
    assert pbr_asphalt.roughness >= 0.80
    assert max(pbr_asphalt.albedo_rgb) <= 0.25


def test_automatic_pbr_material_assignment_and_gltf_usd_export():
    """Verify that assigning PBR material to an entity exports cleanly with PBR parameters in glTF and USD."""
    world = WorldIR(id="w-pbr-test")

    # Add entity
    wall_entity = Entity(
        id="wall-pbr-1",
        name="Plaster Wall",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    world.entities[wall_entity.id] = wall_entity

    ev_plaster = MaterialEvidence(
        color_rgb=[240, 240, 235],
        texture_roughness=0.10,
        specular_sheen=0.05,
        surface_kind="wall",
        spread_delta_e=2.0
    )

    mat = assign_pbr_material(world, wall_entity, ev_plaster)

    # Verify WorldIR material created
    assert mat.id in world.materials
    assert wall_entity.material_ids == [mat.id]
    assert "pbr_material" in wall_entity.custom_properties
    assert wall_entity.custom_properties["pbr_material"]["category"] == "plaster"

    box_geom = Geometry(id="geom-wall", type=GeometryType.BOX)
    world.geometries[box_geom.id] = box_geom
    wall_entity.geometry_ids = [box_geom.id]

    # Export to glTF and verify PBR metallic/roughness
    gltf_dict, report_gltf = export_to_gltf_with_report(world)
    assert "materials" in gltf_dict
    assert len(gltf_dict["materials"]) >= 1
    mat_gltf = gltf_dict["materials"][0]
    assert "pbrMetallicRoughness" in mat_gltf
    assert "roughnessFactor" in mat_gltf["pbrMetallicRoughness"]

    # Export to USD and verify UsdPreviewSurface material
    usda_text, report_usd = export_to_usda_with_report(world)
    assert len(usda_text) > 0
    assert "UsdPreviewSurface" in usda_text
    assert "roughness" in usda_text
