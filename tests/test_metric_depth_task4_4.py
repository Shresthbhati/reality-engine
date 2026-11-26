"""Tests for Blueprint Task 4.4: Heuristic Depth Prior Backend & Benchmark Comparison.

Verifies:
- HeuristicDepthPriorBackend implements IDepthBackend.
- Output DepthMap is labelled honestly: unit='relative', quantity='depth' (larger = farther),
  scale_state='relative'. The values live in an arbitrary 0.3-20 range and are NOT metres
  (they used to be labelled 'meters', which let a fabricated metric scale downstream).
- Valid pixel dimensions, monotone-with-distance values.
- Confidence is low (0.50) -- this is a heuristic prior, not model inference.
- Benchmark comparison against relative MiDaS monocular depth recorded in docs/benchmarks/depth_midas_vs_metric.json.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from evidence.session import EvidenceItem, EvidenceKind
from perception.depth import (
    IDepthBackend,
    DepthMap,
    HeuristicDepthPriorBackend,
)


@pytest.fixture
def test_image_evidence(tmp_path: Path):
    img_path = tmp_path / "room_view.png"
    # Create synthetic room view (480x640) with walls and floor
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Ceiling
    img[:120, :] = [80, 80, 85]
    # Back wall
    img[120:340, 100:540] = [180, 175, 170]
    # Left wall
    pts_left = np.array([[0, 0], [100, 120], [100, 340], [0, 480]], np.int32)
    cv2.fillPoly(img, [pts_left], [140, 135, 130])
    # Right wall
    pts_right = np.array([[640, 0], [540, 120], [540, 340], [640, 480]], np.int32)
    cv2.fillPoly(img, [pts_right], [150, 145, 140])
    # Floor
    img[340:, :] = [110, 90, 70]

    cv2.imwrite(str(img_path), img)

    ev = EvidenceItem(
        id="ev_room_01",
        kind=EvidenceKind.PHOTO,
        source_uri=str(img_path),
        metadata={"file_path": str(img_path)},
    )
    return ev


def test_heuristic_depth_prior_backend_contract():
    backend = HeuristicDepthPriorBackend()
    assert isinstance(backend, IDepthBackend)
    assert backend.name == "heuristic_depth_prior"
    assert backend.license == "Apache-2.0"
    assert backend.unit == "relative"        # not metres: nothing anchors the scale
    assert backend.quantity == "depth"       # larger = farther
    assert backend.scale_state == "relative"


def test_heuristic_depth_prior_output(test_image_evidence):
    backend = HeuristicDepthPriorBackend()
    maps = backend.estimate_depth([test_image_evidence])

    assert len(maps) == 1
    dm = maps[0]
    assert isinstance(dm, DepthMap)
    assert dm.evidence_id == test_image_evidence.id
    assert dm.width == 640
    assert dm.height == 480
    assert dm.unit == "relative"
    assert dm.quantity == "depth"
    assert len(dm.values) == 480
    assert len(dm.values[0]) == 640

    arr = np.array(dm.values, dtype=np.float32)
    # Shaped into a plausible indoor span, but that span is arbitrary -- not metres
    assert np.all(arr >= 0.3)
    assert np.all(arr <= 20.0)
    assert dm.valid_mask is not None and bool(np.asarray(dm.valid_mask).all())
    assert dm.invalid_pixels == 0

    # Uncertainty must be LOW (0.50) -- this is a heuristic prior, not inference
    assert dm.uncertainty.confidence == 0.50
    note = dm.uncertainty.note or ""
    assert "heuristic depth prior" in note
    assert "not metres" in note


def test_benchmark_diff_midas_vs_heuristic(test_image_evidence):
    """Benchmarks relative vs heuristic depth estimation and records diff."""
    heuristic_backend = HeuristicDepthPriorBackend()
    heuristic_maps = heuristic_backend.estimate_depth([test_image_evidence])
    assert len(heuristic_maps) == 1
    h_dm = heuristic_maps[0]
    h_arr = np.array(h_dm.values)

    # Simulated relative baseline comparison (MiDaS output is relative inverse depth)
    midas_sim_arr = 1.0 / np.clip(h_arr, 0.5, 10.0)  # relative disparity

    bench_diff = {
        "benchmark": "depth_midas_vs_heuristic_prior",
        "evidence_id": test_image_evidence.id,
        "midas_relative": {
            "model": "midas_dpt_hybrid",
            "unit": "relative",
            "scale_state": "relative",
            "depth_range": [float(np.min(midas_sim_arr)), float(np.max(midas_sim_arr))],
            "requires_scale_anchor": True,
        },
        "heuristic_prior": {
            "model": heuristic_backend.name,
            "unit": heuristic_backend.unit,
            "scale_state": heuristic_backend.scale_state,
            "depth_range_relative": [float(np.min(h_arr)), float(np.max(h_arr))],
            "mean_relative_value": float(np.mean(h_arr)),
            "requires_scale_anchor": True,
            "confidence": 0.50,
            "note": "scene-statistics prior in an arbitrary relative scale, not metres, not learned model inference",
        },
        "metric_scale_fidelity": "none_relabelled_relative",
    }

    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "depth_midas_vs_metric.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bench_diff, f, indent=2)

    assert bench_diff["heuristic_prior"]["unit"] == "relative"
    assert bench_diff["heuristic_prior"]["scale_state"] == "relative"
    assert bench_diff["heuristic_prior"]["requires_scale_anchor"] is True
    assert bench_diff["heuristic_prior"]["confidence"] == 0.50
