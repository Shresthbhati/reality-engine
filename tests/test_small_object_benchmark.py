"""Tests for Small-Object and Architectural Detail Evaluation Harness.

Validates:
- High-resolution tile grid generation with overlap margins
- Coordinate translation from tile space back to original image space
- Non-Maximum Suppression (NMS) merging across tile boundaries
- Comparative benchmark measuring recall improvement from tiled inference
"""

from __future__ import annotations

import numpy as np

from perception.evaluation import (
    SmallObjectBenchmarkHarness,
    SmallObjectBenchmarkReport,
    SmallObjectDetection,
    TiledInferenceEngine,
    TileSpecification,
)


def test_tile_generation_coverage():
    tiler = TiledInferenceEngine(tile_size=640, overlap_ratio=0.20)
    w, h = 1920, 1080
    tiles = tiler.generate_tiles(w, h)

    assert len(tiles) > 0
    # Verify all tiles are within image bounds
    for t in tiles:
        assert isinstance(t, TileSpecification)
        assert 0 <= t.origin_x < w
        assert 0 <= t.origin_y < h
        assert t.width <= 640
        assert t.height <= 640
        assert t.origin_x + t.width <= w
        assert t.origin_y + t.height <= h

    # Verify complete coverage of corners
    origins = [(t.origin_x, t.origin_y) for t in tiles]
    assert (0, 0) in origins
    # Check that bottom-right pixel is reached by at least one tile
    last_tile = tiles[-1]
    assert last_tile.origin_x + last_tile.width == w
    assert last_tile.origin_y + last_tile.height == h


def test_coordinate_mapping_tile_to_global():
    tiler = TiledInferenceEngine(tile_size=640)
    tile = TileSpecification(
        tile_id="test_tile",
        origin_x=500,
        origin_y=300,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
    )
    tile_detections = [
        {"label": "electrical socket", "box_xyxy": [50.0, 40.0, 90.0, 80.0], "score": 0.92},
        {"label": "door handle", "box_xyxy": [200.0, 150.0, 240.0, 180.0], "score": 0.88},
    ]

    mapped = tiler.map_tile_boxes_to_global(tile_detections, tile)
    assert len(mapped) == 2

    d0 = mapped[0]
    assert isinstance(d0, SmallObjectDetection)
    assert d0.box_xyxy_global == (550.0, 340.0, 590.0, 380.0)
    assert d0.box_xywh_global == (550.0, 340.0, 40.0, 40.0)
    assert d0.is_tiled_detection is True


def test_nms_suppresses_boundary_duplicates():
    tiler = TiledInferenceEngine(tile_size=640, iou_nms_threshold=0.40)

    # Simulate two overlapping tiles detecting the same socket
    # Tile 1 detected socket at [498, 298, 542, 342] with score 0.85
    # Tile 2 detected socket at [500, 300, 540, 340] with score 0.94
    # Independent third detection at [800, 400, 850, 450]
    det1 = SmallObjectDetection(
        label="socket",
        canonical_path="electrical.socket",
        score=0.85,
        box_xyxy_global=(498.0, 298.0, 542.0, 342.0),
        box_xywh_global=(498.0, 298.0, 44.0, 44.0),
        area_px=44.0 * 44.0,
        source_tile_id="tile_1",
        is_tiled_detection=True,
    )
    det2 = SmallObjectDetection(
        label="socket",
        canonical_path="electrical.socket",
        score=0.94,
        box_xyxy_global=(500.0, 300.0, 540.0, 340.0),
        box_xywh_global=(500.0, 300.0, 40.0, 40.0),
        area_px=40.0 * 40.0,
        source_tile_id="tile_2",
        is_tiled_detection=True,
    )
    det3 = SmallObjectDetection(
        label="socket",
        canonical_path="electrical.socket",
        score=0.89,
        box_xyxy_global=(800.0, 400.0, 850.0, 450.0),
        box_xywh_global=(800.0, 400.0, 50.0, 50.0),
        area_px=50.0 * 50.0,
        source_tile_id="tile_3",
        is_tiled_detection=True,
    )

    merged = tiler.apply_nms([det1, det2, det3])
    # The duplicate (det1) should be suppressed, keeping det2 (higher score) and det3
    assert len(merged) == 2
    assert merged[0].score == 0.94
    assert merged[0].source_tile_id == "tile_2"
    assert merged[1].score == 0.89


def test_small_object_benchmark_harness_execution():
    harness = SmallObjectBenchmarkHarness()

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Ground truth: three micro items (two sockets, one door handle)
    ground_truth = [
        {"label": "socket", "canonical_path": "electrical.socket", "box_xyxy": [150, 150, 190, 190], "category": "electrical"},
        {"label": "socket", "canonical_path": "electrical.socket", "box_xyxy": [1200, 600, 1240, 640], "category": "electrical"},
        {"label": "door handle", "canonical_path": "hardware.door_handle", "box_xyxy": [850, 450, 910, 480], "category": "hardware"},
    ]

    # Detector function: only detects items when image is a high-resolution crop (tile width <= 640)
    def mock_detector(image_pil, prompts):
        w, h = image_pil.size
        if w <= 640 and h <= 640:
            # Crop inference succeeds on micro details
            # If crop contains [150, 150], detect socket
            return [
                {"label": "socket", "canonical_path": "electrical.socket", "box_xyxy": [150, 150, 190, 190], "score": 0.88}
            ]
        # Whole-image naive downsampling misses micro details
        return []

    report = harness.run_benchmark(img, mock_detector, ground_truth)
    assert isinstance(report, SmallObjectBenchmarkReport)
    assert report.total_ground_truth_items == 3
    assert report.total_detections_naive == 0
    assert report.total_detections_tiled > 0
    assert report.recall_naive == 0.0
    assert report.recall_tiled > 0.0
    assert report.small_object_recovery_gain > 0.0
