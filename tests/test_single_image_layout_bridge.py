"""Level 0 with a solved layout: one floor, one ceiling, the visible walls and a window with metric size.

The regression this guards: sparse, noisy points promoted five "floors" (one above the ceiling) and mislocated the window.
With the layout, the same room must come out as exactly one of each surface. Ground truth is the ray-cast box room from
test_surface_layout (camera 1.5 m up, ceiling 2.6 m, window 1.4 m wide, sill 0.9 m).
"""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import tests.test_surface_layout as room
from engine.pipeline.single_image import bootstrap_single_image
from evidence.image_check import inspect_image
from evidence.session import EvidenceItem, EvidenceKind


@pytest.fixture()
def bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(room, "HFOV", 60.0)  # the bootstrap assumes a 60 degree field of view without EXIF
    depth, masks = room._render(pitch_deg=5.0)
    p = tmp_path / "room.png"
    Image.fromarray(np.random.default_rng(1).integers(90, 120, (room.H, room.W, 3), dtype=np.uint8)).save(p)
    item = EvidenceItem(id="ev-room", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    analysis = SimpleNamespace(depth_m=depth, masks=masks)
    return bootstrap_single_image(item, inspect_image(p), layout_analysis=analysis)


def _of(res, kind):
    return [e for e in res.world.entities.values() if e.type.value == kind]


def test_one_floor_one_ceiling_and_only_real_walls(bootstrap):
    assert len(_of(bootstrap, "floor")) == 1
    assert len(_of(bootstrap, "ceiling")) == 1
    # at 60 degrees the right wall is a few-centimetre sliver (correctly dropped as weak); never more than the 3 real walls
    assert 2 <= len(_of(bootstrap, "wall")) <= 3
    lf = bootstrap.facts["layout"]
    assert lf["used"] and lf["camera_height_m"] == pytest.approx(1.5, abs=0.08)
    assert lf["room_height_m"] == pytest.approx(2.6, abs=0.1)


def test_window_has_measured_metric_size_and_sits_on_a_promoted_wall(bootstrap):
    (win,) = _of(bootstrap, "window")
    dims = win.custom_properties["bootstrap"]["dimensions"]
    assert dims["width_m"] == pytest.approx(1.4, abs=0.15)
    assert dims["sill_m"] == pytest.approx(0.9, abs=0.12)
    host = win.custom_properties["bootstrap"]["host_entity_id"]
    assert host in bootstrap.world.entities and bootstrap.world.entities[host].type.value == "wall"
    assert win.confidence <= 0.3  # a monocular estimate never claims more than that


def test_the_scale_is_labelled_an_estimate_not_a_measurement(bootstrap):
    scale = bootstrap.world.metadata["scale"]
    assert scale["state"] == "relative" and "ESTIMATE" in scale["note"]


def test_without_a_floor_the_layout_is_refused_and_ransac_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(room, "HFOV", 60.0)
    depth, masks = room._render()
    masks = dict(masks, floor=np.zeros_like(masks["floor"]))
    p = tmp_path / "room.png"
    Image.fromarray(np.random.default_rng(1).integers(90, 120, (room.H, room.W, 3), dtype=np.uint8)).save(p)
    item = EvidenceItem(id="ev-room", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    res = bootstrap_single_image(item, inspect_image(p), layout_analysis=SimpleNamespace(depth_m=depth, masks=masks))
    assert res.facts["layout"]["used"] is False and "floor" in res.facts["layout"]["reason"]
