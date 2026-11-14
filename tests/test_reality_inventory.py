"""The inventory contract never lets a field vanish silently, and the evaluator scores detection, small objects and
cross-view identity without letting a good room-level number hide a missed socket or a split chair."""

from __future__ import annotations

import pytest

from benchmarks.reality_benchmark.inventory import (
    evaluate_inventory, inventory_metric_results, iou, validate_inventory,
)
from benchmarks.reality_benchmark.scorer import score_scenario, validate_results

NA = {k: "not measured in this test" for k in ("instance_count", "mask", "position_3d", "dimensions", "orientation",
                                               "material", "visibility", "evidence")}


def rec(image, entity, sub, box, *, inst=None, conf=None, cat="furniture"):
    r = {"image_id": image, "entity_id": entity, "category": cat, "subtype": sub, "bbox": list(box), "unavailable": dict(NA)}
    if inst is not None:
        r["instance_id"] = inst
    if conf is not None:
        r["confidence"] = conf
    for k in NA:
        r.setdefault(k, None)
    return r


def truth_set():
    """Two photographs of one room: a sofa and a lamp seen in both, a socket seen once (small object)."""
    return [rec("a", "t1", "sofa", (10, 10, 110, 60), inst="sofa-1"), rec("b", "t2", "sofa", (20, 20, 120, 70), inst="sofa-1"),
            rec("a", "t3", "lamp", (200, 10, 230, 90), inst="lamp-1"), rec("b", "t4", "lamp", (210, 12, 240, 92), inst="lamp-1"),
            rec("a", "t5", "socket", (300, 300, 312, 312), inst="socket-1", cat="electrical")]


def perfect_pred():
    return [rec("a", "E-sofa", "sofa", (10, 10, 110, 60), conf=0.9), rec("b", "E-sofa", "sofa", (20, 20, 120, 70), conf=0.9),
            rec("a", "E-lamp", "lamp", (200, 10, 230, 90), conf=0.8), rec("b", "E-lamp", "lamp", (210, 12, 240, 92), conf=0.8),
            rec("a", "E-sock", "socket", (300, 300, 312, 312), conf=0.7, cat="electrical")]


def test_a_perfect_inventory_scores_perfectly():
    m = evaluate_inventory(perfect_pred(), truth_set())
    assert (m["precision"], m["recall"], m["f1"], m["map50"], m["map50_95"]) == (100.0,) * 5
    assert m["duplicate_object_rate"] == 0.0 and m["cross_view_entity_matching"] == 100.0 and m["merged_entity_ids"] == []
    assert m["groups"]["electrical"]["recall"] == 100.0


def test_a_missed_socket_is_not_hidden_by_a_good_room_level_recall():
    pred = [p for p in perfect_pred() if p["subtype"] != "socket"]
    m = evaluate_inventory(pred, truth_set())
    assert m["recall"] == 80.0 and m["precision"] == 100.0                 # 4 of 5 found: looks fine overall...
    assert m["groups"]["electrical"]["recall"] == 0.0                      # ...but the small-object group is reported on its own
    ids = {r["metric_id"]: r for r in inventory_metric_results(pred, truth_set())}
    assert ids["E.electrical.recall"]["value"] == 0.0 and ids["A.recall"]["value"] == 80.0


def test_the_same_chair_split_into_two_ids_is_a_duplicate_and_a_cross_view_failure():
    pred = perfect_pred()
    pred[1] = rec("b", "E-sofa-2", "sofa", (20, 20, 120, 70), conf=0.9)    # view b got a different canonical id
    m = evaluate_inventory(pred, truth_set())
    assert m["duplicate_object_rate"] == pytest.approx(100 * 1 / 4)        # 4 canonical ids, one of them redundant
    assert m["cross_view_entity_matching"] == 50.0                         # sofa inconsistent, lamp consistent


def test_two_different_objects_sharing_one_id_are_reported_as_a_merge_error():
    t = truth_set()
    t.append(rec("a", "t6", "lamp", (400, 10, 430, 90), inst="lamp-2"))
    pred = perfect_pred() + [rec("a", "E-lamp", "lamp", (400, 10, 430, 90), conf=0.8)]   # lamp-2 wrongly folded into E-lamp
    assert evaluate_inventory(pred, t)["merged_entity_ids"] == ["E-lamp"]


def test_right_box_wrong_class_counts_against_precision_and_classification_accuracy():
    pred = perfect_pred()
    pred[0] = rec("a", "E-sofa", "armchair", (10, 10, 110, 60), conf=0.9)
    m = evaluate_inventory(pred, truth_set())
    assert m["precision"] == 80.0 and m["classification_accuracy"] == pytest.approx(100 * 4 / 5)


def test_false_positives_lower_precision_not_recall():
    pred = perfect_pred() + [rec("a", "E-ghost", "sofa", (500, 500, 600, 560), conf=0.95)]
    m = evaluate_inventory(pred, truth_set())
    assert m["recall"] == 100.0 and m["precision"] == pytest.approx(100 * 5 / 6) and m["map50"] < 100.0


def test_no_cross_view_ground_truth_means_no_cross_view_number():
    t = [r for r in truth_set() if r["image_id"] == "a"]
    p = [r for r in perfect_pred() if r["image_id"] == "a"]
    assert evaluate_inventory(p, t)["cross_view_entity_matching"] is None
    assert "C.cross_view_entity_matching" not in {r["metric_id"] for r in inventory_metric_results(p, t)}


def test_fields_cannot_vanish_silently():
    ok = perfect_pred()[0]
    assert validate_inventory([ok]) == []
    assert any("without an 'unavailable' reason" in e for e in validate_inventory([{**ok, "unavailable": {}}]))
    assert any("bbox" in e for e in validate_inventory([{**ok, "bbox": [5, 5, 1, 1]}]))
    assert any("confidence" in e for e in validate_inventory([{**ok, "confidence": None}]))
    assert any("instance_id" in e for e in validate_inventory([{**truth_set()[0], "instance_id": None}], truth=True))
    assert any("visibility" in e for e in validate_inventory([{**ok, "visibility": "gone"}]))
    with pytest.raises(ValueError):
        evaluate_inventory([{**ok, "category": ""}], truth_set())


def test_metric_results_flow_into_the_benchmark_scorer_and_small_samples_stay_provisional():
    results = inventory_metric_results(perfect_pred(), truth_set())
    assert validate_results(results) == []
    by = {r["metric_id"]: r for r in results}
    assert by["A.precision"]["status"] == "PROVISIONAL" and "below the minimum" in by["A.precision"]["reason"]   # n = 5
    card = score_scenario("001", results)
    assert card["verdict"] == "INCOMPLETE" and card["categories"]["A"]["measured"] == 0   # provisional is not a measurement
    assert by["A.precision"]["confidence_interval_95"]["lower"] is not None


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
