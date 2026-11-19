"""Test suite for segmentation lifecycle states, polygon topology with holes, and honest fallback tracking.

Validates:
- Explicit lifecycle states: VISUAL_MASK, GEOMETRIC_APPROXIMATION, NO_MASK.
- Topology extraction with interior holes (e.g. donut-shaped object / window frame).
- Disconnected components (islands).
- Refusal of empty / degenerate binary masks (resolves to NO_MASK).
- Geometric approximation fallback strictly flagged as is_visual_mask=False.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.quality.autonomous_correction_loop import RefinementKind, RefinementTask
from perception.specialists.segment import (
    MaskState,
    extract_mask_topology,
    segment_specialist,
)
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def test_extract_mask_topology_with_interior_hole() -> None:
    # Construct a 100x100 mask with a solid square and a square hole in the middle (donut / picture frame)
    # Outer ring: [20:80, 20:80]
    # Hole: [35:65, 35:65]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask[35:65, 35:65] = 0

    contour, area_px, topo = extract_mask_topology(mask)

    assert len(contour) >= 4
    assert topo.is_valid is True
    # Area = (60 * 60) - (30 * 30) = 3600 - 900 ~ 2640 px (polygon contour area)
    assert area_px == pytest.approx(2640.0, abs=200.0)
    assert topo.num_components == 1
    # Verify that the interior hole was detected and captured
    assert topo.num_holes >= 1
    assert len(topo.components[0].holes) >= 1


def test_extract_mask_topology_disconnected_components() -> None:
    # Construct two separate islands (e.g. occluded entity visible in two parts)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255  # Island 1: 20x20
    mask[60:85, 60:85] = 255  # Island 2: 25x25

    contour, area_px, topo = extract_mask_topology(mask)

    assert topo.is_valid is True
    assert topo.num_components == 2
    assert area_px == pytest.approx(937.0, abs=50.0)
    assert len(topo.components) == 2


def test_segment_specialist_visual_mask_state() -> None:
    world = WorldIR(id="w-seg")
    ent = Entity(id="ent-chair", name="Office Chair", type=EntityType.FURNITURE)
    world.entities[ent.id] = ent

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:70, 30:80] = 255  # 50x50 square

    task = RefinementTask(
        kind=RefinementKind.SEGMENT,
        target=ent.id,
        expected_gain=0.8,
        cost_s=0.2,
        metadata={
            "entity_id": ent.id,
            "mask": mask,
            "evidence_id": "ev-cam-001",
            "confidence": 0.94,
        },
    )

    result = segment_specialist(task, world)
    assert result is True
    ef = ent.extra_fields
    assert ef["mask_state"] == MaskState.VISUAL_MASK.value
    assert ef["is_visual_mask"] is True
    assert ef["segmentation"]["source_evidence_id"] == "ev-cam-001"
    assert ef["segmentation"]["confidence"] == 0.94
    assert ef["segmentation"]["topology"]["num_components"] == 1


def test_segment_specialist_empty_mask_resolves_to_no_mask() -> None:
    world = WorldIR(id="w-seg-empty")
    ent = Entity(id="ent-table", name="Coffee Table", type=EntityType.FURNITURE)
    world.entities[ent.id] = ent

    # Completely blank mask
    empty_mask = np.zeros((80, 80), dtype=np.uint8)

    task = RefinementTask(
        kind=RefinementKind.SEGMENT,
        target=ent.id,
        expected_gain=0.5,
        cost_s=0.1,
        metadata={
            "entity_id": ent.id,
            "mask": empty_mask,
        },
    )

    result = segment_specialist(task, world)
    assert result is True
    ef = ent.extra_fields
    assert ef["mask_state"] == MaskState.NO_MASK.value
    assert ef["is_visual_mask"] is False
    assert ef["failure_reason"] == "empty_or_degenerate_mask"


def test_segment_specialist_geometric_approximation_fallback() -> None:
    world = WorldIR(id="w-seg-geom")
    ent = Entity(
        id="ent-cabinet",
        name="Wall Cabinet",
        type=EntityType.FURNITURE,
        transform={"position": [1.0, 2.0, 1.5]},
        custom_properties={"bounding_box_2d": [100, 120, 250, 320]},
    )
    world.entities[ent.id] = ent

    # Task provides NO visual mask, only 2D bounding box fallback in metadata
    task = RefinementTask(
        kind=RefinementKind.SEGMENT,
        target=ent.id,
        expected_gain=0.4,
        cost_s=0.1,
        metadata={
            "entity_id": ent.id,
            "bbox_2d": [100, 120, 250, 320],
        },
    )

    result = segment_specialist(task, world)
    assert result is True
    ef = ent.extra_fields
    assert ef["mask_state"] == MaskState.GEOMETRIC_APPROXIMATION.value
    assert ef["is_visual_mask"] is False
    assert ef["segmentation"]["is_visual_mask"] is False
    assert ef["segmentation"]["backend"] == "bounding_box_approximation_non_visual"
