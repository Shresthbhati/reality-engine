import json

from benchmarks.scene_complete.score import iou, score


def _w(d, name, objs):
    (d / name).write_text(json.dumps({"objects": objs}))


def test_iou_identity_and_disjoint():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_score_counts_tp_fn_fp_and_missing_prediction(tmp_path):
    lab, pred = tmp_path / "l", tmp_path / "p"
    lab.mkdir(); pred.mkdir()
    sofa = {"class": "sofa", "tier": "furniture", "box": [0, 0, 10, 10]}
    _w(lab, "a.json", [sofa, {"class": "tv", "tier": "appliance", "box": [20, 0, 30, 10]}])
    _w(lab, "b.json", [sofa])  # no prediction file -> miss
    _w(pred, "a.json", [sofa, {"class": "lamp", "tier": "appliance", "box": [0, 0, 5, 5]}])
    r = score(lab, pred)
    assert r["tiers"]["furniture"]["tp"] == 1 and r["tiers"]["furniture"]["fn"] == 1
    assert r["tiers"]["appliance"]["fn"] == 1 and r["tiers"]["appliance"]["fp"] == 1
    assert abs(r["overall"]["recall"] - 1 / 3) < 1e-9
