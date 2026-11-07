"""T-4b.1: move / rotate / delete / replace furniture as logged, GENERATED design commands that never touch the survey."""
from __future__ import annotations

import math

import pytest

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.design_edits import DesignEditError, apply, delete, move, replace, rotate
from world_ir.edit_carry import carry_edits
from world_ir.furniture_catalog import CATALOG, catalog_items
from world_ir.schema_v1 import Geometry, GeometryType, Vector3
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR


def _dim(v):
    return {"value": v, "unit": "m", "state": "ESTIMATED", "basis": "test"}


def _world() -> WorldIR:
    w = WorldIR(id="w")
    w.metadata["scale"] = {"state": "metric"}
    w.geometries["g-wall"] = Geometry(id="g-wall", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(5, 2.6, 0.2),
                                      provenance=Provenance.RECONSTRUCTED)
    w.entities["wall-n"] = Entity(id="wall-n", type=EntityType.WALL, name="wall", geometry_ids=["g-wall"],
                                  provenance=Provenance.RECONSTRUCTED, confidence=0.8)
    w.geometries["g-sofa"] = Geometry(id="g-sofa", type=GeometryType.BOX, bounds_min=Vector3(1, 0, 1), bounds_max=Vector3(3, 0.85, 2),
                                      provenance=Provenance.ESTIMATED)
    w.entities["sofa"] = Entity(
        id="sofa", type=EntityType.FURNITURE, name="Sofa", transform={"position": {"x": 2.0, "y": 0.0, "z": 1.5}},
        geometry_ids=["g-sofa"], semantic_labels=["furniture_candidate"], provenance=Provenance.INFERRED, confidence=0.6,
        custom_properties={"category": "sofa", "entity_class": "furniture.sofa", "yaw_rad": 0.0, "mean_color_rgb": [10, 20, 30],
                           "dimensions": {"length_m": _dim(2.0), "width_m": _dim(1.0), "height_m": _dim(0.85)}})
    return w


def test_move_changes_position_and_keeps_the_box_in_step():
    w = _world()
    cmd = move(w, "sofa", 3.0, 2.0)
    assert w.entities["sofa"].transform["position"] == {"x": 3.0, "y": 0.0, "z": 2.0}
    g = w.geometries["g-sofa"]
    assert (g.bounds_min.x, g.bounds_max.x, g.bounds_min.z, g.bounds_max.z) == (2.0, 4.0, 1.5, 2.5)
    assert cmd["provenance"] == "GENERATED" and w.metadata["design_commands"][-1]["op"] == "move"
    assert validate_world_ir(w).is_valid()


def test_rotate_turns_the_footprint_and_stores_a_unit_quaternion():
    w = _world()
    rotate(w, "sofa", math.pi / 2)
    e = w.entities["sofa"]
    assert e.custom_properties["yaw_rad"] == pytest.approx(math.pi / 2) and e.custom_properties["yaw_basis"] == "user"
    q = e.transform["rotation"]
    assert math.sqrt(sum(q[a] ** 2 for a in "wxyz")) == pytest.approx(1.0)
    g = w.geometries["g-sofa"]                      # a 2.0 x 1.0 sofa turned a quarter: now 1.0 wide in x, 2.0 deep in z
    assert (g.bounds_max.x - g.bounds_min.x) == pytest.approx(1.0) and (g.bounds_max.z - g.bounds_min.z) == pytest.approx(2.0)


def test_replace_swaps_class_and_size_and_marks_the_new_piece_generated():
    w = _world()
    replace(w, "sofa", "armchair")
    e = w.entities["sofa"]
    cp = e.custom_properties
    assert cp["category"] == "armchair" and cp["entity_class"] == "furniture.armchair" and e.name == "Armchair"
    assert cp["dimensions"]["length_m"]["value"] == CATALOG["armchair"].length_m
    assert "catalog default" in cp["dimensions"]["length_m"]["basis"] and cp["dimensions"]["length_m"]["state"] != "MEASURED"
    assert "mean_color_rgb" not in cp, "the colour was measured from the OLD object"
    assert e.provenance == Provenance.GENERATED
    assert w.metadata["design_commands"][-1]["args"] == {"catalog_id": "armchair", "was": "furniture.sofa"}


def test_a_catalog_size_is_refused_in_a_world_with_no_real_scale():
    w = _world()
    w.metadata.pop("scale")
    with pytest.raises(DesignEditError, match="Set scale"):
        replace(w, "sofa", "armchair")
    move(w, "sofa", 1.0, 1.0)                       # moving and turning need no size, so they still work
    rotate(w, "sofa", 0.5)


def test_delete_removes_the_piece_its_geometry_and_references_but_not_the_survey():
    w = _world()
    w.entities["wall-n"].relationships = []
    delete(w, "sofa")
    assert "sofa" not in w.entities and "g-sofa" not in w.geometries
    assert "wall-n" in w.entities and "g-wall" in w.geometries
    assert validate_world_ir(w).is_valid()


def test_only_furniture_can_be_edited_and_bad_input_changes_nothing():
    w = _world()
    before = w.to_dict()
    for fn, args in ((move, ("wall-n", 1, 1)), (rotate, ("wall-n", 1)), (delete, ("wall-n",)), (replace, ("wall-n", "sofa-3-seat")),
                     (move, ("sofa", float("nan"), 1)), (move, ("sofa", 1e9, 1)), (move, ("sofa", True, 1)), (rotate, ("sofa", "a")),
                     (replace, ("sofa", "no-such-item")), (move, ("ghost", 1, 1))):
        with pytest.raises(DesignEditError):
            fn(w, *args)
    assert w.to_dict() == before
    with pytest.raises(DesignEditError):
        apply(w, "sofa", {"op": "explode"})


def test_move_rotate_and_replace_survive_a_rebuild_but_the_survey_does_not_change():
    w = _world()
    move(w, "sofa", 3.0, 2.0)
    rotate(w, "sofa", 1.0)
    replace(w, "sofa", "armchair")
    fresh = _world()                                  # the same room rebuilt from the photos: the sofa is back where it was detected
    report = carry_edits(fresh, w.metadata["user_edits"])
    assert not report["skipped"], report
    e = fresh.entities["sofa"]
    assert e.transform["position"]["x"] == 3.0 and e.custom_properties["yaw_rad"] == pytest.approx(1.0)
    assert e.custom_properties["category"] == "armchair" and e.name == "Armchair"
    assert fresh.entities["wall-n"].to_dict() == _world().entities["wall-n"].to_dict()


def test_the_catalog_is_complete_for_every_class_the_viewer_draws():
    drawn = {"sofa", "armchair", "chair", "dining_table", "coffee_table", "desk", "bed", "wardrobe", "bookshelf", "dresser",
             "nightstand", "tv_stand", "tv", "lamp", "stool", "cabinet"}
    assert drawn <= {i["category"] for i in catalog_items()}
    assert all(i["basis"].startswith("nominal") and i["length_m"] > 0 and i["width_m"] > 0 and i["height_m"] > 0 for i in catalog_items())
