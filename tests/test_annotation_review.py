"""The review loop: predictions out to COCO / Label Studio, corrected labels back, honest per-label scores."""
from __future__ import annotations

import json

import pytest

from benchmarks import annotation_review as ar

PRED = {
    "e1": [{"region_id": "r1", "label": "sofa", "confidence": 0.9, "bbox_px": [100.0, 100.0, 300.0, 200.0]},
           {"region_id": "r2", "label": "plant", "confidence": 0.6, "bbox_px": [400.0, 50.0, 450.0, 150.0]}],
    "e2": [{"region_id": "r3", "label": "sofa", "confidence": 0.8, "bbox_px": [10.0, 10.0, 110.0, 60.0]}],
}
SIZES = {"e1": (640, 480), "e2": (640, 480)}
IMAGES = {"e1": "a/e1.jpg", "e2": "a/e2.jpg"}


def test_coco_export_is_well_formed_and_keeps_the_pipeline_ids():
    coco = ar.to_coco(PRED, IMAGES, SIZES)
    assert {c["name"] for c in coco["categories"]} == {"sofa", "plant"}
    assert [i["evidence_id"] for i in coco["images"]] == ["e1", "e2"] and len(coco["annotations"]) == 3
    a = coco["annotations"][0]
    assert a["bbox"] == [100.0, 100.0, 200.0, 100.0] and a["area"] == 20000.0 and a["attributes"]["region_id"] == "r1"
    json.dumps(coco)


def test_label_studio_export_uses_percent_units_and_declares_every_label():
    tasks, config = ar.to_label_studio(PRED, {"e1": "http://x/e1.jpg", "e2": "http://x/e2.jpg"}, SIZES)
    v = tasks[0]["predictions"][0]["result"][0]["value"]
    assert (v["x"], v["y"], v["width"], v["height"]) == (100 * 100 / 640, 100 * 100 / 480, 100 * 200 / 640, 100 * 100 / 480)
    assert v["rectanglelabels"] == ["sofa"] and '<Label value="plant"/>' in config and '<Label value="sofa"/>' in config


def test_round_trip_through_coco_reads_back_the_boxes_and_labels():
    back = ar.read_coco(ar.to_coco(PRED, IMAGES, SIZES))
    assert back["e1"][0] == {"label": "sofa", "bbox_px": [100.0, 100.0, 300.0, 200.0]} and len(back["e2"]) == 1


def test_score_counts_matches_misses_and_false_alarms_per_label():
    truth = {"e1": [{"label": "sofa", "bbox_px": [105.0, 100.0, 300.0, 205.0]},     # matches the predicted sofa
                    {"label": "rug", "bbox_px": [0.0, 300.0, 600.0, 470.0]}],        # the pipeline missed the rug
             "e2": []}                                                                # reviewed: nothing there -> sofa is a false alarm
    r = ar.score(PRED, truth)
    assert r["status"] == "scored" and r["images_reviewed"] == 2
    assert r["per_label"]["sofa"] == {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0}
    assert r["per_label"]["plant"]["fp"] == 1 and r["per_label"]["plant"]["precision"] == 0.0
    assert r["per_label"]["rug"]["fn"] == 1 and r["per_label"]["rug"]["recall"] == 0.0


def test_an_image_nobody_reviewed_is_not_scored():
    r = ar.score(PRED, {"e1": [{"label": "sofa", "bbox_px": [100.0, 100.0, 300.0, 200.0]}]})
    assert r["images_reviewed"] == 1 and "e2" not in str(r) and r["per_label"]["sofa"]["fp"] == 0


def test_no_review_means_no_numbers():
    r = ar.score(PRED, {})
    assert r["status"] == "unreviewed" and r["per_label"] == {}


def test_fiftyone_missing_gives_the_install_hint():
    try:
        import fiftyone  # noqa: F401
        pytest.skip("fiftyone is installed")
    except ImportError:
        with pytest.raises(ImportError, match="pip install fiftyone"):
            ar.to_fiftyone(ar.to_coco(PRED, IMAGES, SIZES), ".")


def test_cli_export_then_score_round_trip(tmp_path):
    from PIL import Image

    from scripts.export_annotation_review import main

    for eid in ("e1", "e2"):
        Image.new("RGB", (640, 480)).save(tmp_path / f"{eid}.jpg")
    (tmp_path / "report.json").write_text(json.dumps({"stages": {"perception": {"detections_2d": PRED}}}), encoding="utf-8")
    (tmp_path / "images.json").write_text(json.dumps({e: str(tmp_path / f"{e}.jpg") for e in ("e1", "e2")}), encoding="utf-8")
    assert main(["export", "--report", str(tmp_path / "report.json"), "--images", str(tmp_path / "images.json"),
                 "--out", str(tmp_path / "review")]) == 0
    assert (tmp_path / "review" / "coco.json").exists() and (tmp_path / "review" / "label_studio_config.xml").exists()
    # "review" = accept every prediction as-is; the score must then be perfect
    assert main(["score", "--report", str(tmp_path / "report.json"), "--reviewed", str(tmp_path / "review" / "coco.json"),
                 "--out", str(tmp_path / "score.json")]) == 0
    per = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["per_label"]
    assert all(v["precision"] == 1.0 and v["recall"] == 1.0 for v in per.values())


def test_extra_boxes_from_a_second_detector_are_marked_and_unscored():
    extra = {"e1": [{"label": "lamp", "bbox_px": [500.0, 10.0, 540.0, 90.0]}]}
    coco = ar.to_coco(PRED, IMAGES, SIZES, extra_boxes=extra, extra_info={"reality_engine": {"version_id": "v9"}})
    assert coco["info"]["reality_engine"] == {"version_id": "v9"} and "description" in coco["info"]
    assert {c["name"] for c in coco["categories"]} == {"sofa", "plant", "lamp"}
    second = [a for a in coco["annotations"] if a["attributes"]["source"] == "second_opinion"]
    assert len(second) == 1 and second[0]["score"] is None and second[0]["bbox"] == [500.0, 10.0, 40.0, 80.0]
    # a reviewer who keeps the lamp makes it a miss for the pipeline; one who deletes it changes nothing
    kept = ar.read_coco(coco)
    r = ar.score(PRED, kept)
    assert r["per_label"]["lamp"]["fn"] == 1 and r["per_label"]["lamp"]["recall"] == 0.0
