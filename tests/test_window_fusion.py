"""T-3.1: a shut window the segmenter labelled as wall is recovered from a detector box, and a window it already found is left alone.

Masks are built by hand (240 x 320): a wall everywhere, a floor strip, and a window box. Fusing must never touch floor/ceiling
pixels, never override a window the segmenter already found, and must refuse a box that mostly covers non-wall surfaces.
"""
import numpy as np

from perception.layout.window_fusion import fuse_window_boxes

H, W = 240, 320


def _masks():
    m = {k: np.zeros((H, W), bool) for k in ("floor", "ceiling", "wall", "window", "door")}
    m["wall"][:200] = True
    m["floor"][200:] = True
    return m


def test_a_shut_window_labelled_as_wall_becomes_a_window():
    m = _masks()
    out, fused = fuse_window_boxes(m, [("window", 0.7, (100, 40, 200, 140))])
    assert len(fused) == 1
    assert out["window"][90, 150] and not out["wall"][90, 150]
    assert out["wall"][90, 10]  # wall elsewhere is untouched
    assert out["window"].sum() > 0.6 * 100 * 100


def test_a_window_the_segmenter_already_found_is_left_alone():
    m = _masks()
    m["window"][50:130, 110:190], m["wall"][50:130, 110:190] = True, False
    out, fused = fuse_window_boxes(m, [("window", 0.9, (100, 40, 200, 140))])
    assert fused == [] and (out["window"] == m["window"]).all()


def test_floor_pixels_are_never_converted_and_low_scores_are_ignored():
    m = _masks()
    out, fused = fuse_window_boxes(m, [("window", 0.2, (100, 40, 200, 140)), ("window", 0.8, (100, 190, 200, 239))])
    assert fused == [] and not out["window"].any()
    assert out["floor"][220, 150]


def test_the_input_masks_are_not_modified():
    m = _masks()
    fuse_window_boxes(m, [("window", 0.7, (100, 40, 200, 140))])
    assert not m["window"].any()


def test_the_analysis_wrapper_fuses_and_records_an_unavailable_detector():
    from types import SimpleNamespace

    from engine.pipeline.single_image_layout import fuse_window_detections

    an = SimpleNamespace(masks=_masks(), rgb=np.zeros((H, W, 3), np.uint8))
    ok = fuse_window_detections(an, lambda rgb: [("window", 0.7, (100, 40, 200, 140))])
    assert ok.window_fusion["fused"] and ok.masks["window"].any() and not an.masks["window"].any()

    def broken(rgb):
        raise RuntimeError("no weights")

    bad = fuse_window_detections(an, broken)
    assert bad.window_fusion["fused"] == [] and "no weights" in bad.window_fusion["reason"]
    assert (bad.masks["wall"] == an.masks["wall"]).all()
