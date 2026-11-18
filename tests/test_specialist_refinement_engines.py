"""Tests for Specialist Refinement Engines (Universal Spatial Intelligence Directive Phase 7)."""

from __future__ import annotations

import pytest

from engine.quality.autonomous_correction_loop import (
    AutonomousCorrectionLoop,
    RefinementKind,
    RefinementTask,
)
from engine.quality.refinement_engines import registered_executor
from perception.specialists import (
    more_views_specialist,
    reassociate_specialist,
    redetect_crop_specialist,
    register_all_specialist_engines,
    segment_specialist,
    unregister_all_specialist_engines,
)
from world_ir import Entity, EntityType, Transform, Vector3, WorldIR


def test_redetect_crop_specialist_resolves_unknown_attribute():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-door-1",
        name="Front Door",
        type=EntityType.DOOR,
        custom_properties={
            "attributes": {"lock_type": {"value": None, "basis": "unknown"}},
            "candidate_crops": {"lock_type": 128},
        },
    )
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.REDETECT_CROP,
        target="ent-door-1:lock_type",
        expected_gain=0.6,
        cost_s=0.5,
        metadata={"entity_id": "ent-door-1", "attribute": "lock_type", "crop_px": 128},
    )

    result = redetect_crop_specialist(task, world)
    assert result is True
    assert ent.extra_fields["attributes"]["lock_type"]["value"] == "mortise_lock"
    assert ent.extra_fields["attributes"]["lock_type"]["p"] >= 0.8
    assert ent.extra_fields["redetected_crop"] is True


def test_redetect_crop_specialist_with_custom_classifier():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(id="ent-box", name="Safe", type=EntityType.FURNITURE)
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.REDETECT_CROP,
        target="ent-box:lock_type",
        expected_gain=0.6,
        cost_s=0.5,
        metadata={"entity_id": "ent-box", "attribute": "lock_type", "crop_px": 160},
    )

    def custom_classifier(req):
        return {"value": "biometric_scanner", "p": 0.99, "basis": "custom_neural"}

    result = redetect_crop_specialist(task, world, classifier=custom_classifier)
    assert result is True
    assert ent.extra_fields["attributes"]["lock_type"]["value"] == "biometric_scanner"
    assert ent.extra_fields["attributes"]["lock_type"]["p"] == 0.99


def test_segment_specialist_generates_mask_and_contour():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-chair",
        name="Office Chair",
        type=EntityType.FURNITURE,
        custom_properties={"eligible_for_mask": True},
    )
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.SEGMENT,
        target="ent-chair",
        expected_gain=0.75,
        cost_s=0.3,
        metadata={"entity_id": "ent-chair"},
    )

    result = segment_specialist(task, world)
    assert result is True
    assert ent.extra_fields["has_mask"] is True
    assert ent.extra_fields["mask_ref"] == 1
    assert "segmentation" in ent.extra_fields
    assert ent.extra_fields["segmentation"]["num_vertices"] == 8

    # Second call should return False (already segmented)
    assert segment_specialist(task, world) is False


def test_reassociate_specialist_majority_and_tie_breaking():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(id="ent-couch", name="Living Room Sofa", type=EntityType.UNKNOWN)
    world.entities[ent.id] = ent

    # Majority vote test
    task_majority = RefinementTask(
        kind=RefinementKind.REASSOCIATE,
        target="ent-couch",
        expected_gain=0.5,
        cost_s=0.2,
        metadata={"entity_id": "ent-couch", "votes": {"sofa": 4, "chair": 1}},
    )
    assert reassociate_specialist(task_majority, world) is True
    assert ent.extra_fields["resolved_class"] == "sofa"

    # Tie breaking with context hints
    ent_bed = Entity(id="ent-bed", name="Master Bedroom Piece", type=EntityType.UNKNOWN)
    world.entities[ent_bed.id] = ent_bed

    task_tie = RefinementTask(
        kind=RefinementKind.REASSOCIATE,
        target="ent-bed",
        expected_gain=0.5,
        cost_s=0.2,
        metadata={
            "entity_id": "ent-bed",
            "votes": {"bed": 3, "sofa": 3},
            "context_hints": {"room_type": "bedroom"},
        },
    )
    assert reassociate_specialist(task_tie, world) is True
    assert ent_bed.extra_fields["resolved_class"] == "bed"


def test_more_views_specialist_generates_guidance():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-monument",
        name="Statue",
        type=EntityType.STRUCTURE,
        transform={"position": [2.0, 3.0, 0.0]},
        custom_properties={"needs_more_views": True},
    )
    world.entities[ent.id] = ent

    task = RefinementTask(
        kind=RefinementKind.MORE_VIEWS,
        target="ent-monument",
        expected_gain=0.2,
        cost_s=0.001,
        metadata={"entity_id": "ent-monument"},
    )

    result = more_views_specialist(task, world)
    assert result is True
    assert ent.extra_fields["needs_more_views"] is False
    assert ent.extra_fields["capture_guidance_provided"] is True
    assert "capture_requests" in world.metadata
    req = world.metadata["capture_requests"][0]
    assert req["entity_id"] == "ent-monument"
    assert len(req["recommended_poses"]) == 4
    assert req["recommended_poses"][0]["look_at"] == [2.0, 3.0, 0.0]


def test_end_to_end_loop_with_registered_specialists():
    register_all_specialist_engines()
    try:
        world = WorldIR(id="w-full", name="Full Refinement World")
        ent1 = Entity(
            id="e1",
            name="Almirah",
            type=EntityType.FURNITURE,
            custom_properties={
                "attributes": {"lock_type": {"value": None}},
                "candidate_crops": {"lock_type": 128},
            },
        )
        ent2 = Entity(
            id="e2",
            name="Table",
            type=EntityType.FURNITURE,
            custom_properties={"eligible_for_mask": True},
        )
        ent3 = Entity(
            id="e3",
            name="Bed Candidate",
            type=EntityType.FURNITURE,
            custom_properties={"class_votes": {"bed": 5, "couch": 2}},
        )
        ent4 = Entity(
            id="e4",
            name="Sculpture",
            type=EntityType.FURNITURE,
            custom_properties={"needs_more_views": True},
        )

        world.entities[ent1.id] = ent1
        world.entities[ent2.id] = ent2
        world.entities[ent3.id] = ent3
        world.entities[ent4.id] = ent4

        report = AutonomousCorrectionLoop.run_refinement_loop(
            world,
            max_iterations=10,
            executor=registered_executor,
        )

        assert report.tasks_succeeded >= 4
        assert report.tasks_unavailable == 0
        assert ent1.extra_fields["attributes"]["lock_type"]["value"] is not None
        assert ent2.extra_fields["has_mask"] is True
        assert ent3.extra_fields["resolved_class"] == "bed"
        assert ent4.extra_fields["needs_more_views"] is False
        assert len(world.metadata.get("capture_requests", [])) >= 1
    finally:
        unregister_all_specialist_engines()
