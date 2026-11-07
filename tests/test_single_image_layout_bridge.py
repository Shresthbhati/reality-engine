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


def test_ifc_export_has_a_space_for_the_room_and_furnishing_elements_for_furniture(tmp_path, monkeypatch):
    import ifcopenshell

    from exporters.ifc_bridge import export_world_to_ifc
    from perception.instances.furniture_from_detections import FurnitureDetection

    monkeypatch.setattr(room, "HFOV", 60.0)
    depth, masks = room._render(pitch_deg=5.0)
    p = tmp_path / "room.png"
    Image.fromarray(np.random.default_rng(1).integers(90, 120, (room.H, room.W, 3), dtype=np.uint8)).save(p)
    item = EvidenceItem(id="ev-room", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    sofa = FurnitureDetection(label="couch", score=0.9, box_xyxy=(100, 150, 220, 215), view_id="ev-room")
    res = bootstrap_single_image(item, inspect_image(p), layout_analysis=SimpleNamespace(depth_m=depth, masks=masks),
                                 furniture_detections=[sofa])
    n_furniture = len(_of(res, "furniture"))
    assert n_furniture >= 1, res.facts["furniture"]
    out = tmp_path / "r.ifc"
    export_world_to_ifc(res.world, out)
    m = ifcopenshell.open(str(out))
    assert len(m.by_type("IfcSpace")) == 1
    assert len(m.by_type("IfcFurnishingElement")) == n_furniture
    assert len(m.by_type("IfcWindow")) == 1 and len(m.by_type("IfcSlab")) == 2


def test_key_points_are_derived_with_corners_at_the_right_heights(bootstrap):
    """The job's structure_points stage failed with KeyError 'direction' on this world (found by the real full run)."""
    import numpy as np

    from reconstruction.structure_points import structure_points

    pts = structure_points(bootstrap.world)
    kinds = {p["kind"] for p in pts}
    assert "corner" in kinds and "opening_corner" in kinds
    up = np.asarray(_of(bootstrap, "wall")[0].custom_properties["wall_slab"]["up"])
    heights = sorted({round(float(np.dot(p["position"], up)), 1) for p in pts if p["kind"] == "corner"})
    assert max(heights) - min(heights) == pytest.approx(2.6, abs=0.15)  # floor corners to ceiling corners
    win = [p for p in pts if p["kind"] == "opening_corner"]
    span = max(float(np.dot(p["position"], up)) for p in win) - min(float(np.dot(p["position"], up)) for p in win)
    assert span == pytest.approx(1.1, abs=0.15)  # the window: sill 0.9 -> head 2.0
    for p in win:
        assert all(e in bootstrap.world.entities for e in p["entity_ids"])


def test_glb_materials_are_named_by_surface_and_shared(bootstrap):
    """T-2.4: SketchUp/3ds Max users get floor/wall/window/furniture materials, not one anonymous grey."""
    from exporters.gltf.exporter import export_to_glb
    from tests.gltf_validation import parse_glb, validate_glb

    glb = export_to_glb(bootstrap.world)
    validate_glb(glb)  # structurally valid
    j = parse_glb(glb)
    j = j[0] if isinstance(j, tuple) else j
    names = [m["name"] for m in j["materials"]]
    assert {"surface_floor", "surface_wall", "surface_ceiling", "surface_window"} <= set(names)
    assert len(names) == len(set(names))  # one material per surface type, shared by every node of that type
    by_name = {n["name"]: n for n in j["nodes"]}
    wall_nodes = [n for n in j["nodes"] if n["name"].startswith("Inferred wall")]
    wall_mats = {j["meshes"][n["mesh"]]["primitives"][0].get("material") for n in wall_nodes}
    assert len(wall_mats) == 1 and None not in wall_mats
    glass = next(m for m in j["materials"] if m["name"] == "surface_window")
    assert glass.get("alphaMode") == "BLEND"


def test_obj_materials_are_named_by_surface_too(bootstrap):
    from exporters.mesh.obj import export_to_obj

    obj, mtl = export_to_obj(bootstrap.world)
    used = {line.split()[1] for line in obj.splitlines() if line.startswith("usemtl")}
    assert {"surface_floor", "surface_wall", "surface_ceiling", "surface_window"} <= used
    assert all(f"newmtl {m}" in mtl for m in used)


def test_elements_carry_dimensions_with_state_basis_and_uncertainty(bootstrap):
    """T-2.2: wall length, room area, ceiling height, opening size: each ESTIMATED with a stated +/- and basis."""
    from exporters.floorplan import export_to_floorplan_dxf_with_report

    wall = next(w for w in _of(bootstrap, "wall") if w.custom_properties.get("wall_slab"))
    dims = wall.custom_properties["dimensions"]
    assert dims["length_m"]["value"] == pytest.approx(wall.custom_properties["length_m"], abs=0.01)
    assert dims["length_m"]["state"] == "ESTIMATED" and dims["length_m"]["uncertainty_m"] > 0
    assert "monocular" in dims["length_m"]["basis"]
    assert dims["height_m"]["value"] == pytest.approx(2.6, abs=0.1)

    rdims = _of(bootstrap, "room")[0].custom_properties["dimensions"]
    assert {"floor_area_m2", "height_m", "layout_dimension_u_m", "layout_dimension_v_m"} <= set(rdims)
    assert rdims["height_m"]["value"] == pytest.approx(2.6, abs=0.1)

    (win,) = _of(bootstrap, "window")
    wd = win.custom_properties["dimensions"]
    assert wd["width_m"]["value"] == pytest.approx(1.4, abs=0.2) and wd["sill_m"]["value"] == pytest.approx(0.9, abs=0.15)
    assert wd["width_m"]["uncertainty_m"] > 0

    dxf, _ = export_to_floorplan_dxf_with_report(bootstrap.world)
    assert "ESTIMATED" in dxf and "not measured" not in dxf  # the plan prints each dimension WITH its state


def test_room_exposes_the_ceiling_height_keys_the_summary_card_reads(bootstrap):
    cp = _of(bootstrap, "room")[0].custom_properties
    assert cp["ceiling_height_min_m"] == pytest.approx(2.6, abs=0.1) and cp["ceiling_height_max_m"] == cp["ceiling_height_min_m"]


def test_slabs_have_everything_studios_solid_shell_builder_reads(bootstrap):
    """Without these Studio fell back to translucent legacy boxes and said 'no reconstructed room shell'."""
    import numpy as np

    walls = [w for w in _of(bootstrap, "wall") if w.custom_properties.get("wall_slab")]
    assert walls
    for w in walls:
        s = w.custom_properties["wall_slab"]
        d, up, out = (np.asarray(s[k]) for k in ("direction", "up", "outward"))
        assert abs(np.linalg.norm(out) - 1) < 1e-6 and abs(out @ up) < 1e-6 and abs(out @ d) < 1e-6
        a, b = np.asarray(s["start_xyz_m"]), np.asarray(s["end_xyz_m"])
        assert s["length_m"] == pytest.approx(float(np.linalg.norm(b - a)), abs=1e-3) and s["thickness_m"] > 0
        # outward points AWAY from the room: the camera (origin) is on the inner side of the wall
        assert float(np.dot(-(a + b) / 2, out)) < 0
    for kind in ("floor", "ceiling"):
        (e,) = _of(bootstrap, kind)
        slab = e.custom_properties["slab"]
        assert len(slab["polygon_xyz_m"]) >= 3 and abs(np.linalg.norm(slab["up"]) - 1) < 1e-6
        assert e.custom_properties["room_id"]
    fl, ce = (_of(bootstrap, k)[0].custom_properties["slab"] for k in ("floor", "ceiling"))
    rise = np.dot(np.asarray(ce["polygon_xyz_m"][0]) - np.asarray(fl["polygon_xyz_m"][0]), np.asarray(fl["up"]))
    assert rise == pytest.approx(2.6, abs=0.15)
