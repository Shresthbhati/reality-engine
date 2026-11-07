"""Visual evidence for 'is this the same room?': feature matches between the photos of two rooms (static walls, windows, floor texture).

Ground truth: the textured synthetic room (tests/layout_synth.py). The same room from another standpoint shares many features; a room with
different surface textures shares none.
"""
import numpy as np
import pytest
from PIL import Image

import tests.layout_synth as ls
from engine.pipeline.room_visual import visual_agreement


def _save(view, path):
    Image.fromarray(view.rgb).save(path)
    return path


def _other_textures(fn):
    saved = dict(ls._TEXTURES)
    ls._TEXTURES.update({name: ls._texture(900 + i) for i, name in enumerate(ls.PLANES)})
    try:
        return fn()
    finally:
        ls._TEXTURES.clear()
        ls._TEXTURES.update(saved)


def test_the_same_room_from_other_standpoints_scores_high(tmp_path):
    a = [_save(ls.render(0.0, 0.0, 0.0, 0.0, hfov=80.0), tmp_path / "a0.png"), _save(ls.render(0.9, -0.2, 28.0, 0.0, hfov=80.0), tmp_path / "a1.png")]
    b = [_save(ls.render(0.5, -0.1, 12.0, 0.0, hfov=80.0), tmp_path / "b0.png")]
    res = visual_agreement(a, b)
    assert res["best_inliers"] >= 40 and res["score"] >= 0.8 and res["pairs_tested"] == 2


def test_a_room_with_different_surfaces_scores_low(tmp_path):
    a = [_save(ls.render(0.0, 0.0, 0.0, 0.0, hfov=80.0), tmp_path / "a0.png")]
    b = [_save(_other_textures(lambda: ls.render(0.0, 0.0, 0.0, 0.0, hfov=80.0)), tmp_path / "b0.png")]
    res = visual_agreement(a, b)
    assert res["best_inliers"] < 15 and res["score"] < 0.3


def test_no_photos_gives_no_evidence_not_a_zero(tmp_path):
    assert visual_agreement([], [tmp_path / "x.png"]) is None
    assert visual_agreement([tmp_path / "missing.png"], [tmp_path / "also-missing.png"]) is None


def test_the_number_of_pairs_tested_is_bounded(tmp_path):
    a = [_save(ls.render(0.0, 0.0, float(k), 0.0, hfov=80.0), tmp_path / f"a{k}.png") for k in range(5)]
    b = [_save(ls.render(0.3, 0.0, 5.0, 0.0, hfov=80.0), tmp_path / "b.png")]
    assert visual_agreement(a, b, max_pairs=3)["pairs_tested"] == 3
