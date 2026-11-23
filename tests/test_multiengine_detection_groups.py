"""Image-level detection grouping (what different engines saw of the same object in one image). SYNTHETIC boxes."""

from __future__ import annotations

import itertools
import random

import pytest

from engine.multiengine.detection_groups import box_iou, group_detections
from engine.multiengine.observation import ObservationDraft, make_observation

_COUNTER = itertools.count(1)  # keeps timestamps (hence observation ids) distinct


def _o(engine, source, box, label="chair"):
    n = next(_COUNTER)
    return make_observation(
        ObservationDraft(task="detection", quantity="bbox", source_id=source, coordinate_system=f"image:{source}",
                         image_region=box, predicted={"label": label}, confidence=0.9),
        engine_id=engine, model_version="v1", execution_config={},
        processed_at=f"2026-01-01T{n // 3600:02d}:{n // 60 % 60:02d}:{n % 60:02d}+00:00",
    )


def test_the_same_object_seen_by_two_engines_in_one_image_is_one_group():
    a, b = _o("e1", "img", (10, 10, 110, 110)), _o("e2", "img", (12, 8, 112, 108))
    (g,) = group_detections([a, b], iou_min=0.5)
    assert set(g.members) == {a.observation_id, b.observation_id} and g.engines == ("e1", "e2")
    assert g.representative in g.members and g.group_id.startswith("det_")


def test_different_objects_and_different_images_stay_apart():
    a, b = _o("e1", "img", (0, 0, 50, 50)), _o("e2", "img", (200, 200, 250, 250))  # no overlap
    c = _o("e2", "img2", (0, 0, 50, 50))  # same box, another image
    assert len(group_detections([a, b, c], iou_min=0.5)) == 3


def test_an_engine_never_merges_two_of_its_own_detections():
    a1, a2 = _o("e1", "img", (0, 0, 100, 100)), _o("e1", "img", (5, 5, 105, 105))  # two heavily overlapping boxes, one engine
    b = _o("e2", "img", (2, 2, 102, 102))
    groups = group_detections([a1, a2, b], iou_min=0.5)
    assert all(len(g.engines) == len(g.members) for g in groups)  # at most one detection per engine per group
    assert sorted(len(g.members) for g in groups) == [1, 2]  # b joins one of them; the other stays its own object


def test_a_class_disagreement_stays_inside_one_group_it_is_not_split_off():
    a, b = _o("e1", "img", (0, 0, 100, 100), "chair"), _o("e2", "img", (1, 1, 101, 101), "sofa")
    (g,) = group_detections([a, b], iou_min=0.5)
    assert len(g.members) == 2  # canonical fusion reports the label conflict; grouping must not hide it


def test_result_is_independent_of_input_order_and_validates_its_parameter():
    obs = [_o(f"e{i % 3}", f"img{i % 2}", (i * 7 % 40, i * 5 % 40, 100 + i * 7 % 40, 100 + i * 5 % 40)) for i in range(12)]
    ref = group_detections(obs, iou_min=0.4)
    rng = random.Random(3)
    for _ in range(10):
        shuffled = obs[:]
        rng.shuffle(shuffled)
        assert group_detections(shuffled, iou_min=0.4) == ref
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            group_detections(obs, iou_min=bad)
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and box_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def _o2(engine, source, box, label, **predicted):
    o = _o(engine, source, box, label)
    return o if not predicted else _with_predicted(o, predicted)


def _with_predicted(o, extra):
    from dataclasses import replace

    return replace(o, predicted={**o.predicted, **extra})


def test_one_engine_boxing_the_same_object_twice_with_different_labels_is_one_group_with_a_class_conflict():
    """Measured on real photographs: a wall plate emitted by ONE engine as 'socket' and 'switchboard plate' with an identical box."""
    a, b = _o("pipe", "img", (0, 79, 42, 114), "socket"), _o("pipe", "img", (0, 79, 42, 114), "switchboard plate")
    (g,) = group_detections([a, b], iou_min=0.5)
    assert set(g.members) == {a.observation_id, b.observation_id}


def test_the_same_engine_exception_is_narrow_same_label_partial_overlap_and_parts_stay_separate():
    same_label = [_o("pipe", "img", (0, 0, 100, 100), "chair"), _o("pipe", "img", (0, 0, 100, 100), "chair")]
    assert len(group_detections(same_label, iou_min=0.5)) == 2                       # two reports with one label are not merged here
    partial = [_o("pipe", "img", (0, 0, 100, 100), "chair"), _o("pipe", "img", (10, 10, 110, 110), "table")]
    assert len(group_detections(partial, iou_min=0.5)) == 2                          # IoU ~0.68 < 0.95: two objects
    parent = _o2("pipe", "img", (0, 0, 100, 100), "door", pipeline_box_id="p1")
    child = _o2("pipe", "img", (0, 0, 100, 100), "handle", pipeline_box_id="c1", parent_pipeline_box_id="p1")
    assert len(group_detections([parent, child], iou_min=0.5)) == 2                  # a recorded part is a different object even at identical extents
    far = [_o("pipe", "img", (0, 0, 100, 100), "chair"), _o("pipe", "img", (0, 0, 94, 100), "table")]
    assert len(group_detections(far, iou_min=0.5)) == 2                              # IoU 0.94 is below the duplicate threshold


def test_the_duplicate_rule_is_order_independent():
    boxes = [_o("pipe", "img", (0, 79, 42, 114), "socket"), _o("pipe", "img", (0, 79, 42, 114), "switchboard plate"), _o("e2", "img", (1, 80, 41, 113), "socket")]
    ref = [g.members for g in group_detections(boxes, iou_min=0.5)]
    for _ in range(5):
        random.shuffle(boxes)
        assert [g.members for g in group_detections(boxes, iou_min=0.5)] == ref
