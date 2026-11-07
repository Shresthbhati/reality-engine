"""Furniture footprint, height and class from the segmentation mask + metric depth (not from a box width).

Ground truth: a pinhole camera at the origin (y DOWN, z forward) 1.5 m above a floor; a box 2.0 m wide (x -1..1), 1.0 m deep
(z 3..4) and 0.8 m tall, ray-cast into a depth map. Its mask is every pixel the box covers.
"""
import math

import numpy as np
import pytest

from perception.layout.furniture_masks import class_and_mask, footprint_from_mask

W, H, HFOV = 320, 240, 80.0
F = (W / 2) / math.tan(math.radians(HFOV) / 2)
FLOOR_Y, TOP_Y = 1.5, 1.5 - 0.8  # y is down: the floor is +1.5, the box top 0.7


def _scene(x0=-1.0, x1=1.0, z0=3.0, z1=4.0):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rx, ry = (u - W / 2) / F, (v - H / 2) / F
    depth = np.full((H, W), 30.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zf = np.where(ry > 1e-6, FLOOR_Y / ry, np.inf)
        depth = np.where(np.isfinite(zf), np.minimum(depth, zf), depth)
        zt = np.where(ry > 1e-6, TOP_Y / ry, np.inf)              # top face (y = 0.7)
        xt = rx * zt
        top = (zt >= z0) & (zt <= z1) & (xt >= x0) & (xt <= x1) & (ry > 1e-6)
        xf = rx * z0                                              # front face (z = z0)
        yf = ry * z0
        front = (xf >= x0) & (xf <= x1) & (yf >= TOP_Y) & (yf <= FLOOR_Y)
        mask = top | front
        depth = np.where(top, zt, np.where(front & ~top, z0, depth))
    return depth, mask


def test_footprint_height_and_centre_come_from_the_mask():
    depth, mask = _scene()
    fp = footprint_from_mask(mask, depth, F, W / 2, H / 2, up=(0.0, -1.0, 0.0), camera_height_m=1.5)
    assert fp["length_m"] == pytest.approx(2.0, abs=0.15) and fp["width_m"] == pytest.approx(1.0, abs=0.15)
    assert fp["height_m"] == pytest.approx(0.8, abs=0.1)
    assert fp["centre_xz"][0] == pytest.approx(0.0, abs=0.15) and fp["centre_xz"][1] == pytest.approx(3.5, abs=0.15)
    assert abs(math.cos(fp["yaw_rad"])) > 0.97  # the long axis runs along x


def test_a_mask_with_too_few_pixels_gives_no_footprint():
    depth, _ = _scene()
    tiny = np.zeros((H, W), bool)
    tiny[100:103, 100:103] = True
    assert footprint_from_mask(tiny, depth, F, W / 2, H / 2, up=(0.0, -1.0, 0.0), camera_height_m=1.5) is None


def test_the_segmenter_labels_give_the_class_and_the_mask_inside_a_box():
    label_map = np.zeros((H, W), np.int32)            # 0 = wall
    label_map[100:160, 60:200] = 1                    # coffee table
    label_map[100:130, 60:100] = 2                    # a bit of cabinet in the same box
    id2label = {0: "wall", 1: "coffee table", 2: "cabinet"}
    got = class_and_mask(label_map, id2label, (50, 90, 210, 170), hint="desk")
    assert got is not None
    cls, mask = got
    assert cls == "coffee_table" and mask[120, 150] and not mask[110, 70]
    assert class_and_mask(label_map, id2label, (0, 0, 40, 40), hint="sofa") is None  # nothing furniture-like in that box
