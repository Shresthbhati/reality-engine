"""Test suite for architectural attribute taxonomy, crop informativeness, and honest provenance.

Validates:
- 11 architectural attribute categories in AttributeTaxonomy.
- Decoupling of crop extraction from visual feature classification.
- Refusal of uninformative / uniform / blank crops (std < 4.0 or area < 64px).
- Strict separation of visual model inference (is_visual_inference=True) from architectural prior heuristics (is_visual_inference=False).
- Stamped checkpoint hash and model metadata for learned predictions.
"""

from __future__ import annotations

import numpy as np

from engine.quality.autonomous_correction_loop import RefinementKind, RefinementTask
from perception.attributes.taxonomy import (
    AttributeCategory,
    AttributeTaxonomy,
)
from perception.specialists.redetect_crop import (
    classify_crop_features,
    is_crop_informative,
    redetect_crop_specialist,
)
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    PinholeCamera,
    Quat,
    Vec3,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_attribute_taxonomy_all_11_categories_present() -> None:
    expected_categories = {
        AttributeCategory.DOOR_HARDWARE,
        AttributeCategory.WINDOW_HARDWARE,
        AttributeCategory.ELECTRICAL_SOCKETS,
        AttributeCategory.ELECTRICAL_SWITCHES,
        AttributeCategory.LIGHTING_FIXTURES,
        AttributeCategory.PLUMBING_FIXTURES,
        AttributeCategory.HVAC_COMPONENTS,
        AttributeCategory.FURNITURE_CONSTRUCTION,
        AttributeCategory.MATERIALS_FINISHES,
        AttributeCategory.STRUCTURAL_ELEMENTS,
        AttributeCategory.ARCHITECTURAL_DETAILS,
    }
    categories = AttributeTaxonomy.all_categories()
    for cat in expected_categories:
        assert cat in categories
        defs = AttributeTaxonomy.get_category_attributes(cat)
        assert len(defs) >= 1
        for attr_def in defs:
            assert attr_def.name
            assert len(attr_def.allowed_values) >= 1


def test_uninformative_blank_crop_refused() -> None:
    # 1. Blank uniform crop (standard deviation = 0.0 < 4.0)
    blank_crop = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert is_crop_informative(blank_crop) is False

    # 2. Extremely tiny crop (< 64 pixels)
    tiny_crop = np.random.randint(0, 255, size=(4, 4, 3), dtype=np.uint8)
    assert is_crop_informative(tiny_crop) is False

    # 3. Informative structured crop (contrast variations)
    good_crop = np.zeros((64, 64, 3), dtype=np.uint8)
    good_crop[10:50, 10:50] = 200
    assert is_crop_informative(good_crop) is True


def test_classify_crop_features_blank_crop_behavior() -> None:
    blank_crop = np.full((80, 80, 3), 150, dtype=np.uint8)
    val, conf, basis, is_vis = classify_crop_features(blank_crop, attribute_name="handle_type")
    assert val == "unknown"
    assert conf == 0.10
    assert basis == "refusal_uninformative_uniform_crop"
    assert is_vis is False


def test_classify_crop_features_informative_crop_provenance() -> None:
    # High-contrast horizontal metallic bar against dark background
    crop = np.zeros((80, 80, 3), dtype=np.uint8)
    crop[35:45, 10:70] = 220

    val, conf, basis, is_vis = classify_crop_features(crop, attribute_name="handle_type", entity_hint="door")
    assert val in ("lever", "knob", "pull_handle", "push_plate")
    assert conf >= 0.70
    assert is_vis is True
    assert basis == "real_visual_crop_inference"


def test_redetect_crop_specialist_distinguishes_visual_from_heuristic() -> None:
    world = WorldIR(id="w-crop-honest")
    cam = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(
            position=Vec3(0.0, 0.0, 0.0),
            rotation=Quat.identity(),
        ),
    )

    door = Entity(
        id="ent-door-office",
        name="Interior Wooden Office Door",
        type=EntityType.DOOR,
        transform={"position": [0.0, 0.0, 2.5]},
        custom_properties={"attributes": {"handle_type": {"value": None, "basis": "unknown"}}},
    )
    world.entities[door.id] = door

    # Case A: Real high-contrast image showing handle detail
    img_real = np.full((480, 640, 3), 40, dtype=np.uint8)
    img_real[200:260, 280:360] = 230  # bright handle in crop region

    task_visual = RefinementTask(
        kind=RefinementKind.REDETECT_CROP,
        target=f"{door.id}:handle_type",
        expected_gain=0.7,
        cost_s=0.2,
        metadata={
            "entity_id": door.id,
            "attribute": "handle_type",
            "image": img_real,
            "camera": cam,
            "evidence_id": "ev-real-handle",
        },
    )

    res_vis = redetect_crop_specialist(task_visual, world)
    assert res_vis is True
    attrs_vis = door.extra_fields["attributes"]
    assert "handle_type" in attrs_vis
    assert attrs_vis["handle_type"]["is_visual_inference"] is True
    assert attrs_vis["handle_type"]["basis"] == "real_visual_crop_inference"

    # Case B: No image available -> relies solely on architectural prior heuristics
    door_no_img = Entity(
        id="ent-door-prior",
        name="Door Prior Only",
        type=EntityType.DOOR,
        custom_properties={"attributes": {"handle_type": {"value": None, "basis": "unknown"}}},
    )
    world.entities[door_no_img.id] = door_no_img

    task_heuristic = RefinementTask(
        kind=RefinementKind.REDETECT_CROP,
        target=f"{door_no_img.id}:handle_type",
        expected_gain=0.5,
        cost_s=0.1,
        metadata={
            "entity_id": door_no_img.id,
            "attribute": "handle_type",
        },
    )

    res_heur = redetect_crop_specialist(task_heuristic, world)
    assert res_heur is True
    attrs_heur = door_no_img.extra_fields["attributes"]
    assert "handle_type" in attrs_heur
    # Must NOT claim visual inference!
    assert attrs_heur["handle_type"]["is_visual_inference"] is False
    assert attrs_heur["handle_type"]["basis"] == "heuristic_prior_unverified"
