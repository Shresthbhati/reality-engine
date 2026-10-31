import numpy as np
import pytest

from perception.detection.interface import ImageDetection
from perception.detection.masks import apply_mask, mask_coverage, transient_mask

SHAPE = (40, 60, 3)


def D(label, box=(10, 10, 20, 20), mask=None):
    return ImageDetection(label, 0.9, box, mask)


def test_box_with_margin():
    m = transient_mask(SHAPE, [D("person")], margin=0)
    assert m.shape == (40, 60) and m.dtype == bool and m.sum() == 100
    m2 = transient_mask(SHAPE, [D("person")], margin=2)
    assert m2.sum() == 14 * 14 and m2[8, 8] and not m2[7, 7]


def test_margin_clipped_to_image():
    m = transient_mask(SHAPE, [D("car", (0, 0, 5, 5))], margin=10)
    assert m[:15, :15].all() and m.sum() == 15 * 15


def test_polygon_mask_used_when_present():
    pm = np.zeros((40, 60), bool)
    pm[30:35, 40:45] = True
    m = transient_mask(SHAPE, [D("person", (0, 0, 5, 5), pm)], margin=3)
    assert np.array_equal(m, pm)


def test_bad_mask_shape():
    with pytest.raises(ValueError):
        transient_mask(SHAPE, [D("person", mask=np.zeros((3, 3), bool))])


def test_keep_over_drop_and_unknown_ignored():
    dets = [D("person", (10, 10, 30, 30)), D("barricade", (20, 10, 30, 30)), D("spaceship", (0, 0, 5, 5))]
    m = transient_mask(SHAPE, dets, margin=0)
    assert m[15, 12] and not m[15, 25] and not m[2, 2]


def test_apply_mask_fills():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, SHAPE, dtype=np.uint8)
    mask = transient_mask(SHAPE, [D("person")], margin=0)
    out, valid = apply_mask(img, mask, "inpaint")
    assert out.shape == SHAPE and out.dtype == np.uint8 and np.array_equal(out[~mask], img[~mask])
    assert np.array_equal(valid, ~mask)
    out, _ = apply_mask(img, mask, "black")
    assert (out[mask] == 0).all()
    out, _ = apply_mask(img, mask, "transparent")
    assert out.shape == (40, 60, 4) and (out[..., 3][mask] == 0).all() and (out[..., 3][~mask] == 255).all()
    out, valid = apply_mask(img, np.zeros((40, 60), bool))
    assert np.array_equal(out, img) and valid.all()
    with pytest.raises(ValueError):
        apply_mask(img, mask, "blur")


def test_coverage():
    assert mask_coverage(np.zeros((4, 4), bool)) == 0.0
    assert mask_coverage(np.ones((4, 4), bool)) == 1.0
