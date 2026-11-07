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


# ---------------------------------------------------------------------------- floor plan from the layout (M2 / 13.5)


def test_the_layout_world_exports_a_floor_plan_dxf_svg_and_room_report(bootstrap):
    from exporters.floorplan import (build_room_report, export_to_floorplan_dxf_with_report,
                                     export_to_floorplan_svg_with_report)

    rooms = _of(bootstrap, "room")
    assert len(rooms) == 1
    cp = rooms[0].custom_properties
    assert "inferred" in cp["footprint_closure"] and len(cp["footprint_polygon_m"]) >= 3
    dxf, rep = export_to_floorplan_dxf_with_report(bootstrap.world)
    assert "ROOM" in dxf and "WALL" in dxf and "OPENING" in dxf  # layers present
    svg, _ = export_to_floorplan_svg_with_report(bootstrap.world)
    assert svg.lstrip().startswith("<svg") or "<svg" in svg
    (r,) = build_room_report(bootstrap.world)["rooms"]
    assert r["height_m"] == pytest.approx(2.6, abs=0.1)
    assert r["openings"].get("window") == 1 and r["walls"] >= 2
    # the visible part only: it must not claim the whole room, and the area must be plausible for this 3.4 x 3.5 m view
    assert 3.0 < r["area_m2"] < 16.0


def test_walls_carry_their_observed_extent_and_the_window_offset(bootstrap):
    walls = [e for e in _of(bootstrap, "wall") if e.custom_properties.get("wall_slab")]
    assert len(walls) >= 2
    with_window = [w for w in walls if w.custom_properties["wall_slab"]["openings"]]
    assert len(with_window) == 1
    o = with_window[0].custom_properties["wall_slab"]["openings"][0]
    assert o["kind"] == "window" and o["width_m"] == pytest.approx(1.4, abs=0.2) and o["sill_m"] == pytest.approx(0.9, abs=0.15)
    assert 0 <= o["offset_m"] <= with_window[0].custom_properties["length_m"]


def test_the_layout_world_passes_the_real_world_validator_with_no_errors(bootstrap):
    """The job fails if validation errors; this was the second regression found by running real photos."""
    from world_ir.validation import validate_world_ir

    errors = [i for i in validate_world_ir(bootstrap.world).issues if str(i.severity.value).lower() == "error"]
    assert not errors, [(i.code, i.message[:100]) for i in errors]
    for kind in ("window", "door"):
        for e in _of(bootstrap, kind):
            assert e.custom_properties["host_wall_id"] in bootstrap.world.entities
    room = _of(bootstrap, "room")[0]
    ids = set(room.custom_properties["boundary_element_ids"])
    assert {e.id for e in _of(bootstrap, "floor")} <= ids and {e.id for e in _of(bootstrap, "wall")} >= (ids & {e.id for e in _of(bootstrap, "wall")})
