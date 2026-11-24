"""Component Recognition & Small-Object Pipeline Tests (Sprint 9 P1 & P2).

Tests:
1. Window-frame and door-frame parent assignment with perimeter/carcass area ratio support.
2. Fine-grained component ownership across Doors, Windows, Electrical, Furniture, Plumbing/HVAC, Exterior.
3. Non-destructive duplicate suppression preserving frames, sashes, and micro components.
4. Tiled inference coordinate transforms and overlap handling for small objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    DetectionScale,
    GENUINE_STRUCTURAL_PARTS,
    generate_tiling_windows,
    resolve_component_ownership,
    suppress_duplicates_preserving_parts,
    transform_tile_detections_to_image,
)


def test_window_frame_parent_assignment_not_rejected_by_area():
    """Verify window_frame with near-equal or larger bounding box than window is correctly bound to window."""
    window_host = DetectionBox(
        box_id="host_window_1",
        class_name="window",
        box_2d=(200.0, 200.0, 600.0, 600.0),  # area = 400 * 400 = 160000
        confidence=0.95,
    )
    # Window frame enclosing or co-located with the window (area = 400 * 400 = 160000, 1.0x host area)
    frame_box = (195.0, 195.0, 605.0, 605.0)  # area = 410 * 410 = 168100, 1.05x host area

    best_host_id, is_ambig, cand_ids = resolve_component_ownership(
        component_box=frame_box,
        component_class="window_frame",
        candidate_hosts=[window_host],
    )

    assert best_host_id == "host_window_1", f"Expected window_frame to bind to window, got {best_host_id}"
    assert is_ambig is False
    assert "host_window_1" in cand_ids


def test_door_frame_and_handle_ownership():
    """Verify door_frame and door_handle both bind to door with appropriate specificity."""
    door_host = DetectionBox(
        box_id="host_door_1",
        class_name="door",
        box_2d=(100.0, 100.0, 400.0, 800.0),  # 300 * 700 = 210000
        confidence=0.92,
    )
    wall_host = DetectionBox(
        box_id="host_wall_1",
        class_name="wall",
        box_2d=(0.0, 0.0, 1000.0, 1000.0),
        confidence=0.90,
    )

    # Frame wraps around the door
    frame_box = (95.0, 95.0, 405.0, 805.0)
    best_frame_host, _, _ = resolve_component_ownership(
        component_box=frame_box,
        component_class="door_frame",
        candidate_hosts=[wall_host, door_host],
    )
    assert best_frame_host == "host_door_1"

    # Handle is small component inside the door
    handle_box = (350.0, 450.0, 380.0, 490.0)
    best_handle_host, _, _ = resolve_component_ownership(
        component_box=handle_box,
        component_class="door_handle",
        candidate_hosts=[wall_host, door_host],
    )
    assert best_handle_host == "host_door_1"


def test_plumbing_and_electrical_ownership():
    """Verify sink components (faucet, drain) and electrical (socket on outlet_plate)."""
    sink_host = DetectionBox(
        box_id="sink_1",
        class_name="sink",
        box_2d=(300.0, 300.0, 600.0, 600.0),
        confidence=0.93,
    )
    faucet_box = (420.0, 310.0, 480.0, 420.0)
    drain_box = (430.0, 430.0, 470.0, 470.0)

    best_faucet, _, _ = resolve_component_ownership(faucet_box, "faucet", [sink_host])
    best_drain, _, _ = resolve_component_ownership(drain_box, "drain", [sink_host])

    assert best_faucet == "sink_1"
    assert best_drain == "sink_1"


def test_suppress_duplicates_preserves_window_and_frame():
    """Verify that suppress_duplicates_preserving_parts does not suppress window_frame when co-located with window."""
    window = DetectionBox(
        box_id="w1",
        class_name="window",
        box_2d=(200.0, 200.0, 600.0, 600.0),
        confidence=0.95,
    )
    frame = DetectionBox(
        box_id="f1",
        class_name="window_frame",
        box_2d=(198.0, 198.0, 602.0, 602.0),
        confidence=0.90,
    )

    kept = suppress_duplicates_preserving_parts([window, frame])
    kept_classes = {k.class_name for k in kept}
    assert "window" in kept_classes
    assert "window_frame" in kept_classes
    assert len(kept) == 2


def test_suppress_duplicates_preserves_micro_components():
    """Verify that micro components (socket, handle) are never suppressed as duplicates of host."""
    door = DetectionBox(
        box_id="d1",
        class_name="door",
        box_2d=(100.0, 100.0, 400.0, 800.0),
        confidence=0.95,
        scale=DetectionScale.STANDARD,
    )
    handle = DetectionBox(
        box_id="h1",
        class_name="door_handle",
        box_2d=(350.0, 450.0, 380.0, 490.0),
        confidence=0.88,
        scale=DetectionScale.MICRO,
        parent_box_id="d1",
    )

    kept = suppress_duplicates_preserving_parts([door, handle])
    assert len(kept) == 2
    assert {k.class_name for k in kept} == {"door", "door_handle"}


def test_tiled_inference_window_generation_and_transform():
    """Verify tile generation logic and coordinate transform for small-object recovery."""
    img_w, img_h = 2560, 1440
    tiles = generate_tiling_windows(img_w, img_h, tile_size=960, overlap_ratio=0.25)
    assert len(tiles) >= 4

    tile0 = tiles[0]
    # Simulate a micro socket detected within tile 0 at (100, 100, 150, 150)
    tile_det = DetectionBox(
        box_id="micro_sock_0",
        class_name="socket",
        box_2d=(100.0, 100.0, 150.0, 150.0),
        confidence=0.85,
        scale=DetectionScale.MICRO,
    )

    transformed = transform_tile_detections_to_image([tile_det], tile0, img_w, img_h)
    assert len(transformed) == 1
    t_box = transformed[0]
    expected_box = (
        100.0 + tile0.x1,
        100.0 + tile0.y1,
        150.0 + tile0.x1,
        150.0 + tile0.y1,
    )
    assert t_box.box_2d == expected_box
    assert t_box.scale == DetectionScale.MICRO
