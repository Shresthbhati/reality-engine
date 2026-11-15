"""E202: adversarial fixtures for the measured-room accuracy evaluator.

Each test breaks one thing in an otherwise perfect world and asserts the evaluator notices it
through ``class_scores``, ``missing``, ``extra`` and ``meets_tolerances`` -- not just that it
does not crash. Worlds are built with the same helper as ``test_measured_room_accuracy_report``.
"""

from __future__ import annotations

import copy

from perception.architecture.accuracy_report import evaluate_measured_room_accuracy
from tests.test_measured_room_accuracy_report import _perfect_world_and_gt
from world_ir import EntityType, WorldIR


def _cs(rep, etype):
    return next(c for c in rep.class_scores if c.element_type == etype)


def _kinds(entries):
    return [e["element_type"] for e in entries]


def test_baseline_is_clean():
    """Control: the unmodified fixture passes, so every failure below is caused by its mutation."""
    world, gt = _perfect_world_and_gt()
    rep = evaluate_measured_room_accuracy(world, gt)
    assert rep.meets_tolerances is True and rep.missing == [] and rep.extra == []


def test_missing_opening():
    world, gt = _perfect_world_and_gt()
    del world.entities["win-1"]
    rep = evaluate_measured_room_accuracy(world, gt)
    cs = _cs(rep, "window")
    assert cs.n_ground_truth == 1 and cs.n_detected == 0 and cs.n_matched == 0
    assert cs.recall == 0.0
    assert rep.missing == [{"element_type": "window", "gt_id": "window-0"}]
    assert rep.extra == []
    assert _cs(rep, "door").recall == 1.0  # the other opening is untouched
    assert rep.meets_tolerances is False


def test_opening_with_wrong_width_fails_check_but_keeps_recall():
    world, gt = _perfect_world_and_gt()
    world.entities["d-1"].custom_properties["width_m"] = 0.908 + 0.10  # +10 cm
    rep = evaluate_measured_room_accuracy(world, gt)
    cs = _cs(rep, "door")
    assert cs.recall == 1.0 and cs.precision == 1.0
    width = next(c for c in rep.checks if c.element_id == "d-1" and c.dimension_name == "width")
    assert width.passed is False and width.error_abs > 0.02
    assert rep.failed_checks == 1
    assert rep.missing == [] and rep.extra == []
    assert rep.meets_tolerances is False


def test_duplicate_wall_entity_drops_precision():
    world, gt = _perfect_world_and_gt()
    dup = copy.deepcopy(world.entities["w-1"])
    dup.id = "w-1-dup"
    world.entities[dup.id] = dup
    rep = evaluate_measured_room_accuracy(world, gt)
    cs = _cs(rep, "wall")
    assert cs.n_detected == 5 and cs.n_matched == 4
    assert cs.precision == 0.8 and cs.recall == 1.0
    assert len(rep.extra) == 1
    assert rep.extra[0]["element_type"] == "wall"
    assert rep.extra[0]["element_id"] in {"w-1", "w-1-dup"}  # either copy may lose the tie
    assert rep.missing == []


def test_door_labelled_as_window():
    world, gt = _perfect_world_and_gt()
    world.entities["d-1"].type = EntityType.WINDOW
    rep = evaluate_measured_room_accuracy(world, gt)
    door, window = _cs(rep, "door"), _cs(rep, "window")
    assert door.n_detected == 0 and door.recall == 0.0
    assert window.n_detected == 2 and window.n_matched == 1
    assert window.precision < 1.0 and window.recall == 1.0
    assert {"element_type": "door", "gt_id": "door-0"} in rep.missing
    assert {"element_type": "window", "element_id": "d-1"} in rep.extra
    assert rep.meets_tolerances is False


def test_whole_room_shifted_two_metres_is_all_missing_and_all_extra():
    world, gt = _perfect_world_and_gt()
    for ent in world.entities.values():
        cp = ent.custom_properties
        slab = cp.get("wall_slab")
        if slab:
            slab["start_xyz_m"][0] += 2.0
            slab["end_xyz_m"][0] += 2.0
        for key in ("position", "position_m"):
            if key in cp:
                cp[key] = [cp[key][0] + 2.0, *cp[key][1:]]
    rep = evaluate_measured_room_accuracy(world, gt)
    for etype, n in (("wall", 4), ("door", 1), ("window", 1), ("fixture", 1)):
        cs = _cs(rep, etype)
        assert cs.n_matched == 0 and cs.recall == 0.0 and cs.precision == 0.0, etype
        assert _kinds(rep.missing).count(etype) == n, etype
        assert _kinds(rep.extra).count(etype) == n, etype
    assert rep.meets_tolerances is False


def test_empty_world():
    gt = _perfect_world_and_gt()[1]
    rep = evaluate_measured_room_accuracy(WorldIR(id="w-empty", name="Empty"), gt)
    assert rep.room_found is False
    for etype in ("wall", "door", "window", "fixture"):
        cs = _cs(rep, etype)
        assert cs.recall == 0.0 and cs.precision == 0.0 and cs.n_detected == 0, etype
    assert len(rep.missing) == 4 + 1 + 1 + 1 and rep.extra == []
    assert rep.meets_tolerances is False


def test_zero_openings_in_ground_truth_and_none_detected():
    world, gt = _perfect_world_and_gt()
    gt["openings"] = []
    del world.entities["d-1"], world.entities["win-1"]
    rep = evaluate_measured_room_accuracy(world, gt)
    for kind in ("door", "window"):
        cs = _cs(rep, kind)
        assert cs.n_ground_truth == 0 and cs.n_detected == 0
        assert cs.recall == 0.0 and cs.precision == 0.0 and cs.f1 == 0.0  # 0/0 is defined as 0.0
    assert not [m for m in rep.missing if m["element_type"] in {"door", "window", "opening"}]
    assert rep.extra == []
    assert _cs(rep, "wall").recall == 1.0
    assert rep.meets_tolerances is True  # decided by walls (and the room) only
