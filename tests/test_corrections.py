"""F6: user corrections are VERIFIED evidence and survive a rebuild.

SYNTHETIC: a compiled fidelity room with an explicit metric scale. Asserts the replay contract, not real-room
accuracy. A relative world must refuse a VERIFIED dimension (there, a typed metre is a scale reference, not a fact).
"""

from __future__ import annotations

import copy

import pytest

from tests.test_room_fidelity_fixture import _compiled
from world_ir import EntityType
from world_ir.corrections import (
    CARRY_MATCH_TOL_M, CORRECTABLE_PROPERTIES, CorrectionError, carry_corrections, make_correction,
    record_correction,
)
from world_ir.measure import stamp_world_dimensions


def _metric_world():
    _obs, world, _diag = copy.deepcopy(_compiled("albedo", False))
    return world


def _slab(world):
    return next(e for e in world.entities.values()
                if e.type == EntityType.WALL and (e.custom_properties or {}).get("wall_slab"))


def test_correction_becomes_verified_and_is_registered():
    world = _metric_world()
    world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0, "dimension_state": "MEASURED"}
    stamp_world_dimensions(world)
    wall = _slab(world)
    rec = make_correction(world, wall.id, "length_m", 5.25, "tape measure, north wall")
    assert rec["dimension"]["state"] == "VERIFIED" and rec["dimension"]["unit"] == "m"
    record_correction(world, rec)
    assert wall.custom_properties["length_m"] == pytest.approx(5.25)
    assert wall.custom_properties["dimensions"]["length_m"]["state"] == "VERIFIED"
    assert "user-entered" in wall.custom_properties["dimensions"]["length_m"]["basis"]
    assert world.metadata["user_corrections"] == [rec]


def test_same_entity_same_property_replaces_the_previous_record():
    world = _metric_world()
    world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    wall = _slab(world)
    record_correction(world, make_correction(world, wall.id, "length_m", 5.0))
    record_correction(world, make_correction(world, wall.id, "length_m", 5.1))
    assert len(world.metadata["user_corrections"]) == 1
    assert world.metadata["user_corrections"][0]["value"] == pytest.approx(5.1)


def test_relative_world_refuses_a_verified_dimension():
    world = _metric_world()
    world.metadata["scale"] = {"state": "relative"}
    with pytest.raises(CorrectionError, match="not in metres"):
        make_correction(world, _slab(world).id, "length_m", 5.0)


@pytest.mark.parametrize("prop,value", [("colour", 3.0), ("length_m", -1.0), ("length_m", 0.0), ("length_m", float("nan"))])
def test_bad_property_or_value_is_refused(prop, value):
    world = _metric_world()
    world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    with pytest.raises(CorrectionError):
        make_correction(world, _slab(world).id, prop, value)
    assert "length_m" in CORRECTABLE_PROPERTIES


def test_correction_survives_a_rebuild_by_id_and_by_anchor():
    world = _metric_world()
    world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    wall = _slab(world)
    rec = record_correction(world, make_correction(world, wall.id, "height_m", 2.71, "tape"))
    # same world rebuilt: the deterministic slab id matches
    _obs, rebuilt, _ = copy.deepcopy(_compiled("albedo", False))
    rebuilt.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    out = carry_corrections(rebuilt, [rec])
    assert out["applied"] and not out["skipped"]
    assert rebuilt.entities[wall.id].custom_properties["height_m"] == pytest.approx(2.71)
    assert rebuilt.metadata["user_corrections"] == [rec]

    # id changed (e.g. a room id changed): the stored anchor finds the same slab. Use a FRESH compile so no earlier
    # carry can claim the record by its old id.
    _o, fresh, _d = copy.deepcopy(_compiled("albedo", False))
    fresh.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    target = next(e for e in fresh.entities.values()
                  if e.type == EntityType.WALL and (e.custom_properties or {}).get("wall_slab")
                  and abs(e.custom_properties["wall_slab"]["start_xyz_m"][2]
                          - wall.custom_properties["wall_slab"]["start_xyz_m"][2]) < 1e-6
                  and abs(e.custom_properties["wall_slab"]["start_xyz_m"][0]
                          - wall.custom_properties["wall_slab"]["start_xyz_m"][0]) < 1e-6)
    fresh.entities.pop(target.id)
    target.id = "renamed-wall"
    target.custom_properties.pop("dimensions", None)
    target.custom_properties.pop("height_m", None)
    fresh.entities["renamed-wall"] = target
    out2 = carry_corrections(fresh, [{**rec, "entity_id": "gone"}])
    assert out2["applied"][0]["matched_by"].startswith("nearest")
    assert out2["applied"][0]["entity_id"] == "renamed-wall"


def test_carry_never_applies_to_a_distant_or_different_type_surface():
    world = _metric_world()
    world.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    wall = _slab(world)
    rec = make_correction(world, wall.id, "length_m", 5.0)
    rec["entity_id"] = "gone"
    rec["anchor"] = [999.0, 999.0, 999.0]
    _obs, rebuilt, _ = copy.deepcopy(_compiled("albedo", False))
    rebuilt.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    out = carry_corrections(rebuilt, [rec])
    assert not out["applied"] and out["skipped"]
    assert f"{CARRY_MATCH_TOL_M} m" in out["skipped"][0]["reason"]
