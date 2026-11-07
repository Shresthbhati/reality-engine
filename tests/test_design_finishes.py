"""T-4b.2: wall paint and floor material are a design layer. The survey (measured colour, texture, geometry) is never edited, the
as-built exports are byte-identical with or without finishes, and a finish is undoable and survives a rebuild."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiosqlite")

from provenance import Provenance  # noqa: E402
from world_ir import Entity, EntityType  # noqa: E402
from world_ir.design_edits import DesignEditError, apply  # noqa: E402
from world_ir.design_finishes import PAINTS, clear_finish, finish_options, set_finish  # noqa: E402
from world_ir.edit_carry import carry_edits  # noqa: E402
from world_ir.schema_v1 import Geometry, GeometryType, Vector3  # noqa: E402
from world_ir.validation import validate_world_ir  # noqa: E402
from world_ir.world_v1 import WorldIR  # noqa: E402
from tests.test_design_edits_api import _room, studio  # noqa: E402,F401  (fixture reuse)


def _with_floor(scale: bool = True) -> WorldIR:
    w = _room(scale)
    w.geometries["g-floor"] = Geometry(id="g-floor", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(5, 0.1, 4),
                                       provenance=Provenance.RECONSTRUCTED)
    w.entities["floor"] = Entity(id="floor", type=EntityType.FLOOR, name="floor", geometry_ids=["g-floor"],
                                 provenance=Provenance.RECONSTRUCTED, confidence=0.8,
                                 custom_properties={"mean_color_rgb": [120, 100, 80]})
    w.entities["wall-n"].custom_properties = {"mean_color_rgb": [200, 190, 180]}
    return w


def test_paint_colour_and_floor_material_are_stored_apart_from_the_measured_colour():
    w = _with_floor()
    set_finish(w, "wall-n", paint="sage")
    set_finish(w, "floor", material="oak")
    wall, floor = w.entities["wall-n"].custom_properties, w.entities["floor"].custom_properties
    assert wall["design_finish"]["colour_rgb"] == list(PAINTS["sage"].rgb) and wall["design_finish"]["provenance"] == "GENERATED"
    assert floor["design_finish"]["kind"] == "material" and floor["design_finish"]["material"] == "oak"
    assert wall["mean_color_rgb"] == [200, 190, 180] and floor["mean_color_rgb"] == [120, 100, 80], "the survey colour is untouched"
    set_finish(w, "wall-n", colour_rgb=[10, 20, 30])
    assert w.entities["wall-n"].custom_properties["design_finish"]["colour_rgb"] == [10, 20, 30]
    assert validate_world_ir(w).is_valid()


def test_one_edit_can_finish_several_surfaces_and_clear_restores_the_survey_look():
    w = _with_floor()
    w.entities["wall-s"] = Entity(id="wall-s", type=EntityType.WALL, name="wall", provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    cmd = set_finish(w, "wall-n", paint="navy", also_entity_ids=["wall-s"])
    assert cmd["args"]["entity_ids"] == ["wall-n", "wall-s"]
    assert all("design_finish" in w.entities[i].custom_properties for i in ("wall-n", "wall-s"))
    clear_finish(w, "wall-n", also_entity_ids=["wall-s"])
    assert all("design_finish" not in (w.entities[i].custom_properties or {}) for i in ("wall-n", "wall-s"))
    assert w.entities["wall-n"].custom_properties["mean_color_rgb"] == [200, 190, 180]


def test_bad_requests_are_refused_and_change_nothing():
    w = _with_floor()
    before = w.to_dict()
    for kwargs in (dict(entity_id="sofa", paint="sage"), dict(entity_id="ghost", paint="sage"), dict(entity_id="wall-n"),
                   dict(entity_id="wall-n", paint="sage", material="oak"), dict(entity_id="wall-n", material="oak"),
                   dict(entity_id="floor", material="gold"), dict(entity_id="wall-n", paint="nope"),
                   dict(entity_id="wall-n", colour_rgb=[256, 0, 0]), dict(entity_id="wall-n", colour_rgb=[1, 2]),
                   dict(entity_id="wall-n", colour_rgb=[1.5, 2, 3]), dict(entity_id="wall-n", colour_rgb=[True, 2, 3]),
                   dict(entity_id="wall-n", paint="sage", also_entity_ids=["sofa"])):
        with pytest.raises(DesignEditError):
            set_finish(w, **kwargs)
    assert w.to_dict() == before
    with pytest.raises(DesignEditError):
        apply(w, "wall-n", {"op": "explode"})


def test_a_finish_survives_a_rebuild_from_the_photos():
    w = _with_floor()
    set_finish(w, "wall-n", paint="terracotta")
    set_finish(w, "floor", material="walnut")
    fresh = _with_floor()
    report = carry_edits(fresh, w.metadata["user_edits"])
    assert not report["skipped"], report
    assert fresh.entities["wall-n"].custom_properties["design_finish"]["paint"] == "terracotta"
    assert fresh.entities["floor"].custom_properties["design_finish"]["material"] == "walnut"
    assert fresh.entities["wall-n"].custom_properties["mean_color_rgb"] == [200, 190, 180]


def test_clearing_is_remembered_so_a_rebuild_does_not_bring_the_finish_back():
    w = _with_floor()
    set_finish(w, "wall-n", paint="navy")
    clear_finish(w, "wall-n")
    fresh = _with_floor()
    fresh.entities["wall-n"].custom_properties["design_finish"] = {"colour_rgb": [1, 2, 3]}     # a stale finish from before
    carry_edits(fresh, w.metadata["user_edits"])
    assert not fresh.entities["wall-n"].custom_properties.get("design_finish")


# ---- through the real API ---------------------------------------------------------------------------------------------

@pytest.fixture()
def room_with_floor(studio, monkeypatch):
    import tests.test_design_edits_api as m

    monkeypatch.setattr(m, "_room", _with_floor)
    c, make = studio
    return c, make()


def _exports(c, wid, formats):
    out = {}
    for fmt in formats:
        r = c.post(f"/api/worlds/{wid}/export", json={"format": fmt})
        if r.status_code == 200:
            # the response names the exported file by the sha256 of its bytes (content-addressed); version_id changes per edit
            out[fmt] = r.json()["artifact_uri"]
    return out


def test_as_built_exports_are_identical_before_and_after_finishing(room_with_floor):
    c, wid = room_with_floor
    formats = ("glb", "dxf", "svg", "csv", "gltf")
    before = _exports(c, wid, formats)
    # glb/gltf export this room; the plan formats (dxf, svg, csv) need a room entity this small test world does not have
    assert {"glb", "gltf"} <= set(before), f"only these exported: {sorted(before)}"
    assert _exports(c, wid, formats) == before, "exports must be deterministic for the comparison to mean anything"

    assert c.post(f"/api/worlds/{wid}/surfaces/wall-n/finish", json={"paint": "navy"}).status_code == 200
    assert c.post(f"/api/worlds/{wid}/surfaces/floor/finish", json={"material": "oak"}).status_code == 200
    ents = c.get(f"/api/worlds/{wid}/worldir").json()["entities"]
    assert ents["wall-n"]["custom_properties"]["design_finish"]["paint"] == "navy"
    assert ents["floor"]["custom_properties"]["design_finish"]["material"] == "oak"

    assert _exports(c, wid, formats) == before, "the as-built export must not change when only finishes changed"


def test_finish_is_one_undoable_version_and_the_api_refuses_misuse(room_with_floor):
    c, wid = room_with_floor
    n = len(c.get(f"/api/worlds/{wid}/versions").json()["items"])
    r = c.post(f"/api/worlds/{wid}/surfaces/wall-n/finish", json={"colour_rgb": [30, 40, 50]})
    assert r.status_code == 200 and r.json()["operation"] == "design_finish"
    assert len(c.get(f"/api/worlds/{wid}/versions").json()["items"]) == n + 1
    assert c.post(f"/api/worlds/{wid}/undo").status_code == 200
    assert "design_finish" not in (c.get(f"/api/worlds/{wid}/worldir").json()["entities"]["wall-n"]["custom_properties"] or {})
    assert c.post(f"/api/worlds/{wid}/redo").status_code == 200

    for path, body in (("surfaces/sofa/finish", {"paint": "navy"}), ("surfaces/wall-n/finish", {"material": "oak"}),
                       ("surfaces/wall-n/finish", {}), ("surfaces/wall-n/finish", {"colour_rgb": [999, 0, 0]})):
        assert c.post(f"/api/worlds/{wid}/{path}", json=body).status_code == 422, (path, body)
    assert c.post(f"/api/worlds/{wid}/surfaces/wall-n/clear-finish", json={}).status_code == 200


def test_the_finishes_endpoint_lists_swatches(studio):
    c, _ = studio
    body = c.get("/api/finishes").json()
    assert {m["id"] for m in body["floor_materials"]} >= {"oak", "concrete"} and {p["id"] for p in body["paints"]} >= {"sage", "navy"}
    assert "not matched" in body["note"]
    assert "finish" in finish_options()["note"].lower() or "colours" in finish_options()["note"].lower()
