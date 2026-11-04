"""Unit tests for Task 10.2: Material classification as attributes on entities (Blueprint 11 & Task 10.2)."""

import pytest

from perception.entities.attributes import (
    EntityAttributes,
    classify_and_attach_entity_material,
    infer_entity_attributes,
)
from world_ir.schema_v1 import Entity, EntityType


def test_tag_based_material_classification():
    # Door with explicit wood tag
    attr = infer_entity_attributes(
        class_id="opening.door",
        visual_tags=["oak", "wood_grain"],
    )
    assert attr.material_label == "wood"
    assert attr.material_confidence is not None
    assert attr.material_confidence >= 0.80

    # Chair with leather tag
    attr_leather = infer_entity_attributes(
        class_id="furniture.chair",
        visual_tags=["black", "leather", "office_swivel"],
    )
    assert attr_leather.material_label == "leather"
    assert attr_leather.material_confidence is not None
    assert attr_leather.material_confidence >= 0.80

    # Faucet with metal tag
    attr_metal = infer_entity_attributes(
        class_id="fixtures/plumbing/faucet",
        visual_tags=["chrome", "metal", "shiny"],
    )
    assert attr_metal.material_label == "metal"


def test_optical_color_evidence_material_classification():
    # Warm brown color [140, 90, 45] -> optical classification produces wood
    attr_opt = infer_entity_attributes(
        class_id="furniture.table",
        color_rgb=(140.0, 90.0, 45.0),
    )
    assert attr_opt.material_label == "wood"
    assert attr_opt.material_confidence is not None
    assert attr_opt.material_confidence >= 0.40

    # Translucent / sky blue color [200, 220, 240] on window -> glass
    attr_glass = infer_entity_attributes(
        class_id="opening.window",
        color_rgb=(200.0, 220.0, 240.0),
    )
    assert attr_glass.material_label in ("glass", "plastic")


def test_architectural_class_prior_fallback():
    # Sofa with no tags or colors -> fabric prior
    attr_sofa = infer_entity_attributes(class_id="furniture.sofa")
    assert attr_sofa.material_label == "fabric"
    assert attr_sofa.material_confidence == pytest.approx(0.70)

    # Electrical socket with no tags -> plastic prior
    attr_socket = infer_entity_attributes(class_id="fixtures/electrical/socket")
    assert attr_socket.material_label == "plastic"
    assert attr_socket.material_confidence == pytest.approx(0.80)


def test_classify_and_attach_entity_material_and_entity_properties():
    entity = Entity(
        id="ent-chair-01",
        type=EntityType.STRUCTURE,
        name="DiningChair",
        semantic_labels=["wooden_chair", "dining"],
    )

    # Attach material attributes
    attrs = classify_and_attach_entity_material(entity)
    assert attrs.material_label == "wood"

    # Verify custom_properties and Entity properties
    assert entity.custom_properties.get("material") == "wood"
    assert entity.custom_properties.get("material_confidence") is not None

    # Properties on Entity instance
    assert entity.material == "wood"
    assert entity.material_confidence == attrs.material_confidence


def test_entity_attributes_serialization_roundtrip_with_material():
    attr = EntityAttributes(
        class_id="furniture.desk",
        material_label="wood",
        material_confidence=0.85,
        color_rgb=(120.0, 80.0, 40.0),
        confidence=0.90,
    )

    d = attr.to_dict()
    assert d["material"] == "wood"
    assert d["material_label"] == "wood"
    assert d["material_confidence"] == 0.85

    restored = EntityAttributes.from_dict(d)
    assert restored.material_label == "wood"
    assert restored.material_confidence == 0.85
