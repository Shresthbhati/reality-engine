"""B05: attributes are asserted only with a clear margin; ties stay unknown; correlated views are capped."""
import numpy as np

from perception.detection.attributes import classify_attribute, classify_attribute_from_crop, combine_views

VALUES = ["key", "numpad", "padlock"]


def test_clear_winner_is_chosen():
    r = classify_attribute(VALUES, [0.2, 0.9, 0.3], box_threshold=0.35, obs_id="o1")
    assert r["value"] == "numpad" and r["basis"] == "visual" and r["obs"] == ["o1"]


def test_near_tie_is_unknown_but_candidates_are_kept():
    r = classify_attribute(VALUES, [0.60, 0.61, 0.2], box_threshold=0.35)
    assert r["value"] is None and set(r["candidates"]) == set(VALUES)


def test_a_confident_margin_below_the_box_threshold_is_still_unknown():
    assert classify_attribute(VALUES, [0.05, 0.30, 0.04], box_threshold=0.35)["value"] is None


def test_three_agreeing_weak_views_decide_but_two_conflicting_do_not():
    weak = classify_attribute(VALUES, [0.45, 0.42, 0.40], box_threshold=0.35)
    assert weak["value"] is None
    assert combine_views(VALUES, [weak, weak, weak], 0.35)["value"] == "key"
    a = classify_attribute(VALUES, [0.9, 0.2, 0.2], 0.35); b = classify_attribute(VALUES, [0.2, 0.9, 0.2], 0.35)
    c = combine_views(VALUES, [a, b], 0.35)
    assert c["value"] is None and c["candidates"]["key"] > 0.3 and c["candidates"]["numpad"] > 0.3


def test_evidence_is_capped_at_three_views():
    one = classify_attribute(VALUES, [0.5, 0.45, 0.2], 0.0)
    three = combine_views(VALUES, [one] * 3, 0.0)["candidates"]["key"]
    ten = combine_views(VALUES, [one] * 10, 0.0)["candidates"]["key"]
    assert abs(three - ten) < 1e-9


def test_crop_is_padded_and_every_value_gets_a_phrase_call():
    img = np.zeros((100, 100, 3), np.uint8); seen = []
    def det(crop, phrase): seen.append((crop.shape, phrase)); return 0.9 if phrase == "p2" else 0.1
    r = classify_attribute_from_crop(img, (40, 40, 60, 60), VALUES, {"key": "p1", "numpad": "p2", "padlock": "p3"}, det, 0.35)
    assert r["value"] == "numpad" and len(seen) == 3 and seen[0][0][:2] == (26, 26)
