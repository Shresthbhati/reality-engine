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


def test_small_fitting_detections_join_the_review_package_marked_as_such():
    perception = {
        "detections_2d": PRED,
        "small_fittings": {"status": "ran", "detections_by_class": {"electrical.switchboard.module.socket": 1},
                           "detections": [{"view": "e1", "class": "electrical.switchboard.module.socket", "box_px": [300.0, 400.0, 330.0, 430.0],
                                           "score": 0.61, "stage": "fine"},
                                          {"view": "not-an-image-we-have", "class": "x", "box_px": [0, 0, 1, 1], "score": 0.9, "stage": "fine"}]},
    }
    coco = ar.coco_from_perception(perception, IMAGES, SIZES, version_id="v1")
    marked = [a for a in coco["annotations"] if a["attributes"]["source"] == "small_fitting"]
    assert len(marked) == 1 and marked[0]["score"] == 0.61 and marked[0]["bbox"] == [300.0, 400.0, 30.0, 30.0]
    assert coco["info"]["reality_engine"]["small_fittings"]["status"] == "ran"
    assert "electrical.switchboard.module.socket" in {c["name"] for c in coco["categories"]}
    # nothing recorded -> no section, no extra boxes
    plain = ar.coco_from_perception({"detections_2d": PRED}, IMAGES, SIZES)
    assert plain["info"]["reality_engine"]["small_fittings"] is None
    assert not [a for a in plain["annotations"] if a["attributes"]["source"] != "prediction"]


def test_score_sources_measures_how_much_of_each_extra_source_a_human_kept():
    extra = {"e1": [{"label": "lamp", "bbox_px": [500.0, 10.0, 540.0, 90.0], "source": "second_opinion"},
                    {"label": "rug", "bbox_px": [0.0, 300.0, 100.0, 400.0], "source": "second_opinion"}],
             "e2": [{"label": "lamp", "bbox_px": [20.0, 20.0, 60.0, 100.0], "source": "second_opinion"}]}
    exported = ar.to_coco(PRED, IMAGES, SIZES, extra_boxes=extra)
    reviewed = json.loads(json.dumps(exported))
    # the reviewer keeps e1's lamp, deletes e1's rug, and never opens e2 (drops that image from the review)
    reviewed["annotations"] = [a for a in reviewed["annotations"]
                               if not (a["attributes"]["source"] == "second_opinion" and a["category_id"] == next(
                                   c["id"] for c in reviewed["categories"] if c["name"] == "rug"))]
    reviewed["images"] = [im for im in reviewed["images"] if im["evidence_id"] == "e1"]
    reviewed["annotations"] = [a for a in reviewed["annotations"] if a["image_id"] == reviewed["images"][0]["id"]]
    r = ar.score_sources(exported, reviewed)
    s = r["sources"]["second_opinion"]
    assert r["status"] == "scored" and (s["proposed"], s["kept"], s["precision"]) == (2, 1, 0.5)      # e2 was not reviewed: not counted
    assert s["per_label"]["lamp"]["precision"] == 1.0 and s["per_label"]["rug"]["precision"] == 0.0


def test_score_sources_with_no_extra_boxes_or_no_review_produces_no_number():
    exported = ar.to_coco(PRED, IMAGES, SIZES)
    assert ar.score_sources(exported, exported)["status"] == "unreviewed"


def test_validate_training_dataset_rejects_unreviewed_prelabels():
    """RE-DATA-000005: Model pre-labels must not become truth; reviewed flag required."""
    draft_pkg = {
        "status": "DRAFT - not ground truth; every record is unreviewed",
        "records": [
            {"entity_id": "draft-001", "review_status": "unreviewed", "category": "furniture"},
        ],
    }
    with pytest.raises(ar.UnreviewedTrainingDataError) as exc:
        ar.validate_training_dataset(draft_pkg)
    assert "unreviewed" in str(exc.value)

    # Missing reviewed flag rejected
    unflagged = [{"id": "rec-1", "label": "chair"}]
    with pytest.raises(ar.UnreviewedTrainingDataError) as exc:
        ar.validate_training_dataset(unflagged)
    assert "review_status='reviewed'" in str(exc.value)

    # Reviewed records accepted
    reviewed = [{"entity_id": "gt-001", "review_status": "reviewed", "category": "furniture"}]
    assert ar.validate_training_dataset(reviewed) == reviewed


def test_small_object_review_queue_quarantines_drafts_and_prevents_unreviewed_use():
    """RE-PERC-000003: Review queue isolates unreviewed drafts from production and training."""
    raw_drafts = [
        {"entity_id": f"draft-{i:03d}", "category": "electrical", "subtype": "wall socket",
         "bbox": [10.0 * i, 10.0 * i, 10.0 * i + 30.0, 10.0 * i + 30.0],
         "review_status": "unreviewed", "image_id": "e1"}
        for i in range(10)
    ]
    queue = ar.SmallObjectReviewQueue(raw_drafts)
    assert queue.pending_count == 10
    assert queue.reviewed_count == 0
    assert queue.rejected_count == 0

    # Ground truth export is empty when nothing reviewed
    assert queue.get_ground_truth() == []

    # Attempting to validate raw unreviewed drafts raises UnreviewedTrainingDataError
    with pytest.raises(ar.UnreviewedTrainingDataError):
        ar.validate_training_dataset(queue.get_pending())

    # Approve one
    app = queue.approve("draft-000", instance_id="socket-inst-01")
    assert app["review_status"] == "reviewed"
    assert app["instance_id"] == "socket-inst-01"
    assert queue.pending_count == 9
    assert queue.reviewed_count == 1

    # Edit another
    ed = queue.edit("draft-001", corrected_bbox=[15.0, 15.0, 45.0, 45.0], corrected_label="switch")
    assert ed["review_status"] == "reviewed"
    assert ed["subtype"] == "switch"
    assert ed["bbox"] == [15.0, 15.0, 45.0, 45.0]
    assert queue.pending_count == 8
    assert queue.reviewed_count == 2

    # Reject a false positive (e.g. railing as wiring)
    rej = queue.reject("draft-002", reason="railing false positive")
    assert rej["review_status"] == "rejected"
    assert queue.pending_count == 7
    assert queue.rejected_count == 1

    # Verified ground truth now contains exactly the 2 approved/edited items
    gt = queue.get_ground_truth()
    assert len(gt) == 2
    assert ar.validate_training_dataset(gt) == gt

    # Export remaining pending to COCO for human review
    coco = queue.export_coco(IMAGES, SIZES)
    assert len(coco["annotations"]) == 7
    assert coco["info"]["review_queue"] == "small_objects"


