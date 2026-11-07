"""Filtering/merging rules for open-vocabulary furniture detections (model-free)."""
import pytest

from engine.pipeline.open_vocab_furniture import clean_detections
from perception.instances.furniture_from_detections import FurnitureDetection

SRC, GRID = (1000, 500), (500, 250)


def _clean(raw):
    return clean_detections(raw, view_id="v", src_size=SRC, grid_size=GRID)


def test_boxes_are_rescaled_to_the_placer_grid():
    (d,) = _clean([("sofa", 0.8, (100, 200, 500, 400))])
    assert d.box_xyxy == (50, 100, 250, 200) and d.label == "sofa" and d.view_id == "v"


def test_duplicate_boxes_of_one_class_keep_the_higher_score():
    out = _clean([("sofa", 0.5, (100, 200, 500, 400)), ("sofa", 0.9, (105, 205, 505, 405))])
    assert [d.score for d in out] == [0.9]


def test_the_same_box_under_two_different_classes_is_kept_for_both():
    # NMS is class-aware: a box that is both a "bookshelf" and a "tv stand" candidate is a real ambiguity, not a duplicate
    out = _clean([("bookshelf", 0.6, (100, 100, 400, 300)), ("tv stand", 0.5, (100, 100, 400, 300))])
    assert {d.label for d in out} == {"bookshelf", "tv stand"}


def test_a_box_covering_nearly_the_whole_frame_is_a_scene_false_positive():
    assert _clean([("bed", 0.9, (0, 0, 990, 495))]) == []


def test_a_speck_is_not_furniture():
    assert _clean([("chair", 0.9, (10, 10, 20, 20))]) == []


def test_labels_outside_the_placer_vocabulary_are_dropped_with_no_guessing():
    assert _clean([("potted plant", 0.9, (100, 100, 300, 300))]) == []


def test_aliases_resolve_but_the_prompt_label_is_preserved():
    (d,) = _clean([("television", 0.7, (100, 100, 300, 250))])
    assert d.label == "television"  # the placer canonicalises it (-> tv); we never rewrite what the model said


def test_merge_keeps_one_label_per_object_across_classes():
    from engine.pipeline.open_vocab_furniture import merge_detections

    gd = [FurnitureDetection("coffee table", 0.55, (100, 100, 300, 200), "v")]
    rcnn = [FurnitureDetection("dining table", 0.80, (105, 102, 305, 202), "v"), FurnitureDetection("couch", 0.9, (400, 100, 700, 300), "v")]
    out = merge_detections(gd, rcnn)
    assert sorted(d.label for d in out) == ["couch", "dining table"]  # same object, higher score wins, whatever the class
