"""E203: evaluate every room, not the first (``evaluate_world_accuracy``)."""

from __future__ import annotations

import copy
import json

from perception.architecture.accuracy_report import (
    evaluate_measured_room_accuracy,
    evaluate_world_accuracy,
)
from tests.test_measured_room_accuracy_report import _perfect_world_and_gt
from world_ir import EntityType, WorldIR

DX = 10.0  # room B sits 10 m east of room A


def _shift_xy(value, dx):
    return [value[0] + dx, *value[1:]]


def _room_b(world: WorldIR, gt: dict) -> tuple[dict, dict]:
    """Copy of room A moved +DX in x, with its own ids -> (entities by id, ground truth)."""
    ents = {}
    for ent in world.entities.values():
        new = copy.deepcopy(ent)
        new.id = "room-2" if ent.type == EntityType.ROOM else f"{ent.id}-b"
        cp = new.custom_properties
        if "room_id" in cp:
            cp["room_id"] = "room-2"
        if "footprint_polygon_m" in cp:
            cp["footprint_polygon_m"] = [_shift_xy(p, DX) for p in cp["footprint_polygon_m"]]
        slab = cp.get("wall_slab")
        if slab:
            slab["start_xyz_m"] = _shift_xy(slab["start_xyz_m"], DX)
            slab["end_xyz_m"] = _shift_xy(slab["end_xyz_m"], DX)
        for key in ("position", "position_m"):
            if key in cp:
                cp[key] = _shift_xy(cp[key], DX)
        ents[new.id] = new
    gt_b = copy.deepcopy(gt)
    gt_b["room"]["id"] = "room-2"
    for w in gt_b["walls"]:
        w["id"] += "-b"
        w["start"] = _shift_xy(w["start"], DX)
        w["end"] = _shift_xy(w["end"], DX)
    for o in gt_b["openings"]:
        o["center"] = _shift_xy(o["center"], DX)
    for f in gt_b["fixtures"]:
        f["position_m"] = _shift_xy(f["position_m"], DX)
    return ents, gt_b


def _two_rooms() -> tuple[WorldIR, dict]:
    world, gt_a = _perfect_world_and_gt()
    ents_b, gt_b = _room_b(world, gt_a)
    world.entities.update(ents_b)
    return world, {"rooms": [gt_a, gt_b]}


def _by_gt_id(rep, gt_id):
    return next(r for r in rep.rooms if r.room_id == gt_id)


def test_two_perfect_rooms_both_evaluated_and_gate_passes():
    world, gt = _two_rooms()
    rep = evaluate_world_accuracy(world, gt)
    assert [r.room_id for r in rep.rooms] == ["room-1", "room-2"]
    assert rep.missing_rooms == [] and rep.extra_rooms == []
    for r in rep.rooms:
        assert r.total_checks == 13 and r.meets_tolerances and r.missing == [] and r.extra == []
    assert rep.meets_tolerances is True


def test_missing_second_room_is_a_missing_room_and_fails_gate():
    world, gt = _two_rooms()
    for eid in [e for e in world.entities if e == "room-2" or e.endswith("-b")]:
        del world.entities[eid]
    rep = evaluate_world_accuracy(world, gt)
    assert rep.missing_rooms == [{"gt_id": "room-2"}]
    assert [r.room_id for r in rep.rooms] == ["room-1"] and rep.rooms[0].meets_tolerances
    assert rep.meets_tolerances is False


def test_defect_is_attributed_to_the_right_room():
    world, gt = _two_rooms()
    del world.entities["w-4-b"]
    rep = evaluate_world_accuracy(world, gt)
    assert _by_gt_id(rep, "room-1").meets_tolerances is True
    b = _by_gt_id(rep, "room-2")
    assert b.meets_tolerances is False
    assert [m["element_type"] for m in b.missing] == ["wall"]
    assert rep.meets_tolerances is False


def test_rooms_without_shared_ids_match_by_centroid():
    world, gt = _two_rooms()
    # Reconstruction knows nothing about the surveyor's ids.
    for old, new in (("room-1", "recon-A"), ("room-2", "recon-B")):
        ent = world.entities.pop(old)
        ent.id = new
        world.entities[new] = ent
        for w in world.entities.values():
            if w.custom_properties.get("room_id") == old:
                w.custom_properties["room_id"] = new
    # Present the ground-truth rooms in the opposite order to prove it is not positional.
    gt = {"rooms": list(reversed(gt["rooms"]))}
    rep = evaluate_world_accuracy(world, gt)
    assert sorted(r.room_id for r in rep.rooms) == ["room-1", "room-2"]
    assert rep.missing_rooms == [] and rep.extra_rooms == []
    assert all(r.meets_tolerances for r in rep.rooms) and rep.meets_tolerances is True


def test_unsurveyed_reconstructed_room_is_listed_but_not_gated():
    world, gt = _two_rooms()
    ghost = copy.deepcopy(world.entities["room-2"])
    ghost.id = "ghost"
    ghost.custom_properties["footprint_polygon_m"] = [[50, 50], [55, 50], [55, 54], [50, 54]]
    world.entities[ghost.id] = ghost
    rep = evaluate_world_accuracy(world, gt)
    assert rep.extra_rooms == [{"element_id": "ghost"}]
    assert rep.meets_tolerances is True


def test_single_room_ground_truth_matches_the_single_room_entry_point():
    world, gt = _perfect_world_and_gt()
    rep = evaluate_world_accuracy(world, gt)  # no "rooms" key: one room
    single = evaluate_measured_room_accuracy(world, gt)
    assert len(rep.rooms) == 1
    assert rep.rooms[0].to_dict() == single.to_dict()
    assert rep.meets_tolerances is single.meets_tolerances is True


def test_no_ground_truth_rooms_or_empty_world_fails_gate():
    world, gt = _two_rooms()
    assert evaluate_world_accuracy(world, {}).meets_tolerances is False
    empty = evaluate_world_accuracy(WorldIR(id="w", name="Empty"), gt)
    assert [m["gt_id"] for m in empty.missing_rooms] == ["room-1", "room-2"]
    assert empty.meets_tolerances is False


def test_report_serialises_and_prints():
    world, gt = _two_rooms()
    rep = evaluate_world_accuracy(world, gt)
    d = rep.to_dict()
    assert d["meets_tolerances"] is True and len(d["rooms"]) == 2
    json.dumps(d)
    assert "World Gate: PASS (2 room(s) evaluated)" in rep.format_table()
