"""Phase 3 regression tests for small-object & fine-detail recognition."""
from __future__ import annotations

import numpy as np
from PIL import Image

from perception.detection.ontology_grounding import nms, tile_windows
from perception.evaluation.small_object_benchmark import SmallObjectBenchmarkHarness


def test_class_aware_nms_preserves_distinct_neighboring_classes() -> None:
    # A switch and a socket sitting at nearly identical overlapping coordinates
    # must BOTH survive class-aware NMS, while a duplicate socket is suppressed.
    boxes = [
        (100.0, 100.0, 140.0, 140.0),  # socket (high score)
        (102.0, 101.0, 142.0, 141.0),  # switch (different class, high IoU)
        (103.0, 102.0, 143.0, 142.0),  # duplicate socket from overlapping tile
    ]
    scores = [0.91, 0.85, 0.72]
    labels = ["socket", "switch", "socket"]

    kept = nms(boxes, scores, iou_threshold=0.5, labels=labels)
    assert kept == [0, 1]


def test_tile_windows_cover_full_image_and_preserve_coordinates() -> None:
    windows = tile_windows(4000, 3000, scale=3, overlap=0.2)
    assert len(windows) == 9
    assert windows[0][0] == 0 and windows[0][1] == 0
    assert windows[-1][2] == 4000 and windows[-1][3] == 3000


def test_small_object_benchmark_requires_class_match_and_tracks_category_detected() -> None:
    arr = np.full((512, 512, 3), 200, dtype=np.uint8)
    pil_img = Image.fromarray(arr)

    gt = [
        {"box_xyxy": [100.0, 100.0, 130.0, 130.0], "category": "socket", "label": "socket"},
        {"box_xyxy": [300.0, 300.0, 330.0, 330.0], "category": "switch", "label": "switch"},
    ]

    def fake_detector(img: Image.Image, prompts):
        return [
            {"box_xyxy": [100.0, 100.0, 130.0, 130.0], "label": "wall", "score": 0.9},
            {"box_xyxy": [300.0, 300.0, 330.0, 330.0], "label": "switch", "score": 0.88},
        ]

    bench = SmallObjectBenchmarkHarness()
    report = bench.run_benchmark(pil_img, fake_detector, gt, micro_prompts=["socket", "switch"])
    assert report.recall_tiled == 0.5
    assert report.category_breakdown["socket"]["count"] == 1
    assert report.category_breakdown["socket"]["detected"] == 0
    assert report.category_breakdown["socket"]["recall"] == 0.0
    assert report.category_breakdown["switch"]["count"] == 1
    assert report.category_breakdown["switch"]["detected"] == 1
    assert report.category_breakdown["switch"]["recall"] == 1.0


def test_detection_adapter_partial_failure_grades_partial(tmp_path, monkeypatch) -> None:
    from perception import detection_adapter

    good = tmp_path / "good.png"
    arr = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(good)
    missing = tmp_path / "missing.png"

    monkeypatch.setattr(detection_adapter, "_load_detector", lambda _: (object(), object(), "cpu"))
    monkeypatch.setattr(
        detection_adapter,
        "_detect_in_image",
        lambda *a, **kw: [((10.0, 10.0, 50.0, 50.0), 0.9, "socket")],
    )

    out = detection_adapter._grounding_dino_run([str(good), str(missing)])
    assert out.status == "partial"
    assert len(out.observations) == 1
    assert out.observations[0].model_id.hardware_class == "cpu"
    assert len(out.summary["failed_images"]) == 1
